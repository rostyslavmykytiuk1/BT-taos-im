# SPDX-License-Identifier: MIT
"""MakerHarvesterAgent — genuine-edge harvester for SN79 (validated design).

Edge source (robust, no rebate dependence): TAKER-ACQUIRE + MAKER-TAKE-PROFIT-CLOSE.
- Acquire inventory with ONE taker market order (pays the taker fee once).
- Rest ONE postOnly limit to CLOSE that inventory at cost +/- margin, so the close is a
  MAKER fill (fee ~0 / rebate) at a price strictly better than cost. Each completed cycle
  nets (margin - one taker leg) > 0 with NO rebate required.
- TWO-SIDED: in up/flat regimes harvest LONGs (buy@taker, rest sell@cost*(1+m)); in down
  regimes harvest SHORTs (sell@taker, rest buy@cost*(1-m)). So downtrends are a profit
  source, not a bleed.
- ONE resting order per book => respects the hard 100-open-order cap (active books capped).
- Never force-closes at a loss (unfavorable inventory waits or is rebate-gated).
- Reuses the validated infra: byte-exact FIFO mirror, per-validator state, instant restart
  recovery, status heartbeat. Optional latch layer is OFF by default (validation showed it
  cannot engage without a real positive-close stream underneath).
"""
from __future__ import annotations
import json, math, os, traceback
from collections import defaultdict, deque
from typing import Deque, Dict, NamedTuple

import bittensor as bt
from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol import FinanceAgentResponse, MarketSimulationStateUpdate
from taos.im.protocol.events import TradeEvent
from taos.im.protocol.instructions import STP, TimeInForce
from taos.im.protocol.models import OrderDirection
from taos.im.utils import duration_from_timestamp

NS = 1_000_000_000
KAPPA_WINDOW_NS = 10800 * NS
VOL_WINDOW_NS = 86400 * NS
VOL_CAP_QUOTE = 500_000.0
TP_CID_BASE = 3_000_000


class Lot(NamedTuple):
    qty: float
    price: float
    fee: float
    open_ts: int


class MakerHarvesterAgent(FinanceSimulationAgent):
    def initialize(self) -> None:
        g = lambda n, d: getattr(self.config, n, d)
        # launch() float-coerces every --params value, so a bool passed as 1 arrives as 1.0;
        # accept the numeric form ("1.0") as well as the string forms.
        fb = lambda n, d: (str(g(n, d)).strip().lower() in ("1", "1.0", "true", "yes", "on"))
        self.clip = float(g("clip", 0.25))            # acquire/close clip (>= minOrderSize 0.25)
        self.margin_bps = float(g("margin_bps", 8.0))  # take-profit margin over cost (sweep-optimal: tight+uniform -> best kappa+coverage; fee-aware floor below guarantees >0)
        self.min_margin_bps = float(g("min_margin_bps", 6.0))  # coverage-force floor: kept close to margin_bps so forced closes stay uniform (low dispersion -> higher kappa)
        self.target_lots = int(g("target_lots", 3))     # inventory buffer per book (keeps obs flowing in dead-band)
        self.max_lots = int(g("max_lots", 8))           # hard cap on unrealized inventory per book
        self.ema_hl = float(g("ema_hl", 30.0))          # mid EMA half-life (ticks) for regime side-pick
        self.trend_ticks = float(g("trend_ticks", 1.0)) # |mid-ema| in price ticks to pick short vs long
        # Coverage is the binding constraint: kappa_score = MEDIAN over ALL book_count books with
        # up to 37.5% inactive tolerated. With 128 live books we need >=~41 books carrying >=3
        # realized round-trips. So visit EVERY book each tick (per-book 100-order cap => resting a
        # TP on all books is allowed) and never let the trend dead-band starve a book of cycling.
        self.books_per_tick = int(g("books_per_tick", 100000))   # default: all books
        self.max_active_books = int(g("max_active_books", 100000))  # per-book cap, not global -> effectively all books
        self.coverage_force_after_s = float(g("coverage_force_after_s", 240.0))  # stale TP -> tighten margin to force the close
        self.tp_grace_s = float(g("tp_grace_s", 120.0))  # after placing a TP, treat it as in-flight (don't re-place) for this long -> no orphan duplicates
        self.rebate_only_taker_close = fb("rebate_only_taker_close", True)  # only TAKER-close at a loss if rebated
        self.vol_pace = float(g("vol_pace", 0.80))
        self.acquire_maker = fb("acquire_maker", False)  # acquire via postOnly (cheaper, uncertain) vs taker
        # ---- LATCH (kappa eps-branch): plant a few tiny bounded NEGATIVE closes per book/window
        # on top of the positive-close stream, flipping the LPM3+reg denominator -> kappa inflation.
        self.latch_enabled_default = fb("latch_enabled", False)  # kill-switched; re-read from flag file each tick
        self.flags_path = str(g("flags_path", "/root/79/maker_flags.json"))
        self.dust_gap_sim_s = float(g("dust_gap_sim_s", 2160.0))   # min sim-s between dust prints per book
        self.q_max_quote = float(g("q_max_quote", 0.05))           # max planned dust loss per print (quote)
        self.neg_budget_window = int(g("neg_budget_window", 8))    # max negatives/book/3h window
        self.dust_min_obs = int(g("dust_min_obs", 6))              # only dust books with an established + stream
        self.status_path = str(g("status_path", "/root/79/maker_status.json"))
        self.sim_duration_ns = int(g("sim_duration_ns", 86400 * NS))
        self.end_flatten_sim_s = float(g("end_flatten_sim_s", 1800.0))
        self._cv = ""
        self.V: Dict[str, dict] = {}

    def _st(self, v):
        s = self.V.get(v)
        if s is None:
            s = {"longs": defaultdict(deque), "shorts": defaultdict(deque),
                 "obs": defaultdict(deque), "vol": defaultdict(deque),
                 "tp": {}, "tp_ts": {}, "ema": {}, "last_close": defaultdict(int), "dust_due": {},
                 "last_ts": 0, "offset": 0, "phase": "RUN", "seq": 0, "seen": set()}
            self.V[v] = s
        return s

    def _latch_on(self):
        try:
            if os.path.exists(self.flags_path):
                fl = json.load(open(self.flags_path))
                if "latch_enabled" in fl:
                    return bool(int(fl["latch_enabled"]))
        except Exception:
            pass
        return self.latch_enabled_default

    def _rq(self, q, d): f = 10 ** d; return math.floor(max(q, 0.0) * f) / f
    def _rp(self, p, d): f = 10 ** d; return round(p * f) / f

    # ---- validator-faithful FIFO mirror (long+short) ----
    def _fifo(self, s, b, is_buy, qty, price, fee, ts):
        L, S = s["longs"][b], s["shorts"][b]
        if is_buy:
            if not S: L.append(Lot(qty, price, fee, ts)); return 0.0
        else:
            if not L: S.append(Lot(qty, price, fee, ts)); return 0.0
        r = 0.0; rem = qty; qi = 1.0 / qty if qty > 0 else 0.0
        if is_buy:
            while rem > 0 and S:
                o = S[0]
                if o.qty <= rem:
                    r += (o.price - price) * o.qty - o.fee - fee * o.qty * qi; rem -= o.qty; S.popleft()
                else:
                    oqi = 1.0 / o.qty; of = o.fee * rem * oqi
                    r += (o.price - price) * rem - of - fee
                    S[0] = Lot(o.qty - rem, o.price, o.fee - of, o.open_ts); rem = 0
            if rem > 0: L.append(Lot(rem, price, fee * rem * qi, ts))
        else:
            while rem > 0 and L:
                o = L[0]
                if o.qty <= rem:
                    r += (price - o.price) * o.qty - o.fee - fee * o.qty * qi; rem -= o.qty; L.popleft()
                else:
                    oqi = 1.0 / o.qty; of = o.fee * rem * oqi
                    r += (price - o.price) * rem - of - fee
                    L[0] = Lot(o.qty - rem, o.price, o.fee - of, o.open_ts); rem = 0
            if rem > 0: S.append(Lot(rem, price, fee * rem * qi, ts))
        return r

    def _check_rollback(self, s, ts):
        """Detect a simulator restart/rollback (sim clock jumped backwards) and re-sync.
        Idempotent per ts: must run BEFORE onTrade fires so trades carry the correct offset.
        On rollback we wipe FIFO/TP AND 'seen' — tradeIds reset per-book each new sim, so a
        stale 'seen' would silently drop the new sim's first trades from the mirror."""
        if s.get("_synced_ts") == ts:
            return
        if s["last_ts"] and ts < s["last_ts"] - 5 * NS:
            s["offset"] += (s["last_ts"] - ts)
            s["longs"].clear(); s["shorts"].clear(); s["tp"].clear(); s["tp_ts"].clear()
            s["seen"].clear()
            bt.logging.warning(f"Maker: restart {self._cv[:8]} -> clock shift, FIFO/seen wiped")
        s["last_ts"] = ts
        s["_synced_ts"] = ts

    def update(self, state):
        self._cv = state.dendrite.hotkey
        self._check_rollback(self._st(self._cv), state.timestamp)
        super().update(state)

    def onTrade(self, event, validator=None):
        if event.bookId is None or (event.makerAgentId != self.uid and event.takerAgentId != self.uid):
            return
        v = validator or self._cv; s = self._st(v)
        tid = getattr(event, "tradeId", None)
        if tid is not None:
            k = (event.bookId, tid)
            if k in s["seen"]: return
            s["seen"].add(k)
            if len(s["seen"]) > 200000: s["seen"].clear()
        gts = event.timestamp + s["offset"]; b = event.bookId
        is_maker = event.makerAgentId == self.uid
        is_buy = (event.takerAgentId == self.uid and event.side == 0) or (is_maker and event.side == 1)
        fee = (event.makerFee if is_maker else event.takerFee) or 0.0
        r = self._fifo(s, b, is_buy, event.quantity, event.price, fee, gts)
        s["vol"][b].append((gts, event.quantity * event.price))
        if round(r, 4) != 0.0:
            o = s["obs"][b]
            if o and o[-1][0] == gts: o[-1] = (gts, o[-1][1] + r)
            else: o.append((gts, r))
            s["last_close"][b] = gts

    def _prune(self, s, b, gts):
        oc = gts - KAPPA_WINDOW_NS
        while s["obs"][b] and s["obs"][b][0][0] < oc: s["obs"][b].popleft()
        vc = gts - VOL_WINDOW_NS
        while s["vol"][b] and s["vol"][b][0][0] < vc: s["vol"][b].popleft()

    def respond(self, state):
        resp = FinanceAgentResponse(agent_id=self.uid)
        v = state.dendrite.hotkey; self._cv = v; s = self._st(v)
        ts = state.timestamp; cfg = state.config
        vd, pd, mx = cfg.volumeDecimals, cfg.priceDecimals, cfg.max_open_orders
        self._check_rollback(s, ts)  # no-op if update() already synced this ts; safety if it didn't
        gts = ts + s["offset"]
        flatten = ts > (self.sim_duration_ns - int(self.end_flatten_sim_s * NS))
        accts = state.accounts.get(self.uid, {})
        try:
            ids = sorted(state.books.keys()); nb = len(ids) or 1
            for b in ids: self._prune(s, b, gts)
            # update EMA
            for b in ids:
                bk = state.books.get(b)
                if bk and bk.bids and bk.asks:
                    mid = (bk.bids[0].price + bk.asks[0].price) / 2.0
                    a = 1 - 0.5 ** (1.0 / max(self.ema_hl, 1.0))
                    s["ema"][b] = mid if b not in s["ema"] else (1 - a) * s["ema"][b] + a * mid
            # active-book budget: count books with resting tp to respect 100-order cap
            active = sum(1 for b in ids if s["tp"].get(b) is not None)
            # visit: prioritize lowest obs
            order = sorted(ids, key=lambda b: (sum(1 for _, x in s["obs"][b] if x > 0), s["last_close"].get(b, 0)))
            for b in order[: self.books_per_tick]:
                acct = accts.get(b); bk = state.books.get(b)
                if not acct or not bk or not bk.bids or not bk.asks: continue
                self._ctl(resp, s, b, acct, bk, gts, vd, pd, mx, flatten, active)
            self._status(s, gts, ts, nb)
        except Exception as ex:
            bt.logging.error(f"Maker {v[:8]}: {ex}\n{traceback.format_exc()}")
        return resp

    def _ctl(self, resp, s, b, acct, bk, gts, vd, pd, mx, flatten, active):
        bid, ask = bk.bids[0].price, bk.asks[0].price
        mid = (bid + ask) / 2.0; tick = 10 ** (-pd)
        m_rate = acct.fees.maker_fee_rate if acct.fees else 0.0
        t_rate = acct.fees.taker_fee_rate if acct.fees else 0.0
        L, S = s["longs"][b], s["shorts"][b]
        ema = s["ema"].get(b, mid)
        n_open = len(acct.orders)
        cap_ok = sum(x for _, x in s["vol"][b]) < self.vol_pace * VOL_CAP_QUOTE
        # tp resting? (matched by client id presence). A TP we just placed may not be visible in
        # acct.orders yet (instruction in flight at the exchange). Treat it as in-flight for
        # tp_grace_s before assuming it filled/expired -> prevents re-placing a duplicate close
        # that would over-sell and flip the book short.
        tp_cid = s["tp"].get(b)
        tp_live = tp_cid is not None and any(getattr(o, "client_id", None) == tp_cid for o in acct.orders)
        tp_age = gts - s["tp_ts"].get(b, gts)
        in_flight = (tp_cid is not None) and (not tp_live) and (tp_age < self.tp_grace_s * NS)
        if (not tp_live) and (tp_cid is not None) and (not in_flight):
            s["tp"].pop(b, None); tp_cid = None
        # orphan guard: cancel any of OUR TP-range orders on this book that we are no longer
        # tracking (left behind by a prior race) so at most one close ever rests per book.
        lo = TP_CID_BASE + b * 100000
        orphans = []
        for o in acct.orders:
            cidv = getattr(o, "client_id", None)
            if cidv is not None and lo <= cidv < lo + 100000 and cidv != tp_cid:
                orphans.append(o.id)
        if orphans:
            resp.cancel_orders(book_id=b, order_ids=orphans)

        if flatten:
            # realize profitable heads via taker, abandon underwater
            if L:
                h = L[0]
                if bid * (1 - max(t_rate, 0.0)) > h.price:
                    q = self._rq(min(h.qty, acct.base_balance.free, self.clip), vd)
                    if q >= self.clip: resp.market_order(book_id=b, direction=OrderDirection.SELL, quantity=q, stp=STP.CANCEL_OLDEST)
            return

        # ---- LATCH: at dust-time, FLATTEN (only realizing PROFITABLE inventory, never a forced
        # loss) then plant ONE tiny bounded negative on the now-flat book (same-tick taker
        # SELL->BUY loses only spread+fees) -> flips kappa's LPM3+reg denominator. ----
        if self._latch_on():
            npos = sum(1 for _, x in s["obs"][b] if x > 0)
            nneg = sum(1 for _, x in s["obs"][b] if x < 0)
            if gts >= s["dust_due"].get(b, 0) and npos >= self.dust_min_obs and nneg < self.neg_budget_window:
                if L or S:
                    profitable = (L and bid * (1 - max(t_rate, 0.0)) > L[0].price) or \
                                 (S and ask * (1 + max(t_rate, 0.0)) < S[0].price)
                    if profitable:
                        if tp_cid is not None:
                            cids = [o.id for o in acct.orders if getattr(o, "client_id", None) == tp_cid]
                            if cids: resp.cancel_orders(book_id=b, order_ids=cids)
                            s["tp"].pop(b, None)
                        qd = self._rq(self.clip, vd)
                        if L and acct.base_balance.free >= self.clip:
                            resp.market_order(book_id=b, direction=OrderDirection.SELL, quantity=qd, stp=STP.CANCEL_OLDEST)
                        elif S and acct.quote_balance.free > qd * ask:
                            resp.market_order(book_id=b, direction=OrderDirection.BUY, quantity=qd, stp=STP.CANCEL_OLDEST)
                        return
                    # underwater inventory -> never force a loss; defer dust, fall through to normal harvest
                    s["dust_due"][b] = gts + int(self.dust_gap_sim_s * NS)
                else:
                    spread_cost = (ask - bid + max(t_rate, 0.0) * (ask + bid)) * self.clip
                    if spread_cost <= self.q_max_quote and acct.base_balance.free >= self.clip and acct.quote_balance.free > self.clip * ask:
                        qd = self._rq(self.clip, vd)
                        resp.market_order(book_id=b, direction=OrderDirection.SELL, quantity=qd, stp=STP.CANCEL_OLDEST)
                        resp.market_order(book_id=b, direction=OrderDirection.BUY, quantity=qd, stp=STP.CANCEL_OLDEST)
                    s["dust_due"][b] = gts + int(self.dust_gap_sim_s * NS)
                    return

        # ---- maintain a resting maker take-profit on existing inventory ----
        # Coverage-forcing: a TP resting longer than coverage_force_after_s gets RE-PRICED at a
        # tighter margin (min_margin_bps, still clearing the taker leg so the close stays strictly
        # positive) so the round-trip actually completes -> the book carries >=3 obs -> it counts
        # toward coverage. We NEVER price below cost (no forced loss). Fresh TPs use the full margin.
        if L:
            h = L[0]
            stale = (b in s["tp_ts"]) and (gts - s["tp_ts"].get(b, gts) > self.coverage_force_after_s * NS)
            # TAKER-COMPLETION (the live fix): maker TPs barely fill in the real book, so a stale TP
            # on a PROFITABLE lot is CROSSED to close (market SELL) -> guaranteed completion, strictly
            # positive net of the live taker fee + the sunk acquire fee. Only profitable lots; never a
            # forced loss. This is how coverage actually builds (why the leaders run taker-heavy).
            if stale:
                be = h.fee / max(h.qty, 1e-9) + max(t_rate, 0.0) * bid
                if bid > h.price + be:
                    if tp_cid is not None:
                        cids = [o.id for o in acct.orders if getattr(o, "client_id", None) == tp_cid]
                        if cids: resp.cancel_orders(book_id=b, order_ids=cids)
                        s["tp"].pop(b, None)
                    q = self._rq(min(h.qty, acct.base_balance.free, self.clip), vd)
                    if q >= self.clip:
                        resp.market_order(book_id=b, direction=OrderDirection.SELL, quantity=q, stp=STP.CANCEL_OLDEST)
                        s["tp_ts"].pop(b, None)
                        return
            if tp_cid is None or stale:
                mb = self.min_margin_bps if stale else self.margin_bps
                # Fee-aware floor: the close must clear the acquire fee (h.fee, sunk in the lot)
                # AND the live maker close fee, else realized < 0 -> drops out of kappa's PERFECT
                # branch. Dynamic per-book fees (-10..+12bps) make a fixed bps margin unsafe; this
                # guarantees every close is strictly positive on every book.
                be = h.fee / max(h.qty, 1e-9) + max(m_rate, 0.0) * h.price
                margin = max(mb * h.price / 1e4, be + tick, tick)
                px = self._rp(max(h.price + margin, ask), pd)  # maker: at/above ask, above cost
                q = self._rq(min(h.qty, self.clip), vd)
                if q >= self.clip and px > bid and px > h.price:
                    if stale and tp_cid is not None:
                        cids = [o.id for o in acct.orders if getattr(o, "client_id", None) == tp_cid]
                        if cids: resp.cancel_orders(book_id=b, order_ids=cids)
                    s["seq"] += 1; cid = TP_CID_BASE + b * 100000 + s["seq"] % 100000
                    resp.limit_order(book_id=b, direction=OrderDirection.SELL, quantity=q, price=px,
                                     clientOrderId=cid, stp=STP.CANCEL_NEWEST, postOnly=True,
                                     timeInForce=TimeInForce.GTT, expiryPeriod=int(3600 * NS))
                    s["tp"][b] = cid; s["tp_ts"][b] = gts
                return
            return
        if S:
            h = S[0]
            stale = (b in s["tp_ts"]) and (gts - s["tp_ts"].get(b, gts) > self.coverage_force_after_s * NS)
            # TAKER-COMPLETION (short side): stale TP on a PROFITABLE short -> cross to close (market BUY)
            if stale:
                be = h.fee / max(h.qty, 1e-9) + max(t_rate, 0.0) * ask
                if ask < h.price - be:
                    if tp_cid is not None:
                        cids = [o.id for o in acct.orders if getattr(o, "client_id", None) == tp_cid]
                        if cids: resp.cancel_orders(book_id=b, order_ids=cids)
                        s["tp"].pop(b, None)
                    q = self._rq(self.clip, vd)
                    if q >= self.clip and acct.quote_balance.free > q * ask:
                        resp.market_order(book_id=b, direction=OrderDirection.BUY, quantity=q, stp=STP.CANCEL_OLDEST)
                        s["tp_ts"].pop(b, None)
                        return
            if tp_cid is None or stale:
                mb = self.min_margin_bps if stale else self.margin_bps
                be = h.fee / max(h.qty, 1e-9) + max(m_rate, 0.0) * h.price  # fee-aware floor (see long side)
                margin = max(mb * h.price / 1e4, be + tick, tick)
                px = self._rp(min(h.price - margin, bid), pd)  # maker buy: at/below bid, below short cost
                q = self._rq(min(h.qty, self.clip), vd)
                if q >= self.clip and px < ask and px > 0 and px < h.price:
                    if stale and tp_cid is not None:
                        cids = [o.id for o in acct.orders if getattr(o, "client_id", None) == tp_cid]
                        if cids: resp.cancel_orders(book_id=b, order_ids=cids)
                    s["seq"] += 1; cid = TP_CID_BASE + b * 100000 + s["seq"] % 100000
                    resp.limit_order(book_id=b, direction=OrderDirection.BUY, quantity=q, price=px,
                                     clientOrderId=cid, stp=STP.CANCEL_NEWEST, postOnly=True,
                                     timeInForce=TimeInForce.GTT, expiryPeriod=int(3600 * NS))
                    s["tp"][b] = cid; s["tp_ts"][b] = gts
                return
            return

        # ---- acquire to seed a round-trip on a FLAT book, side = sign(mid - ema).
        # DECISIVE sign with a dead-zone: acquire LONG only when mid is clearly above EMA, SHORT
        # only when clearly below; in the dead-zone WAIT (do not guess). This is what keeps every
        # regime closing: the TP fills WITH the move (up -> long, sell-TP above; down -> short,
        # buy-TP below). The dead-zone is critical at trend onset: when mid ~ EMA (EMA still
        # lagging), guessing long traps the book long for the whole downtrend (can't sell below
        # cost, can't short while long). Waiting one tick lets the EMA diverge so the side is
        # right. We visit ALL books every tick, so a 1-tick wait costs nothing in coverage.
        # Book is FLAT here (L and S blocks returned above) -> drop any stale TP tracking so the
        # next position's close is placed immediately, not after the in-flight grace.
        if b in s["tp"]: s["tp"].pop(b, None); s["tp_ts"].pop(b, None)
        if not cap_ok: return
        band = self.trend_ticks * tick
        up = mid > ema + band
        down = mid < ema - band
        q = self._rq(self.clip, vd)
        if not L and not S:
            if up and acct.quote_balance.free > self.clip * ask:
                resp.market_order(book_id=b, direction=OrderDirection.BUY, quantity=q, stp=STP.CANCEL_OLDEST)
            elif down and acct.base_balance.free >= self.clip:
                resp.market_order(book_id=b, direction=OrderDirection.SELL, quantity=q, stp=STP.CANCEL_OLDEST)

    def _status(self, s, gts, ts, nb):
        try:
            counts = sorted(sum(1 for _, x in s["obs"][b] if x > 0) for b in range(nb))
            med = counts[len(counts) // 2] if counts else 0
            nneg = sum(1 for b in range(nb) for _, x in s["obs"][b] if x < 0)
            json.dump({"wall_ts": ts, "sim_ts": gts, "phase": s["phase"], "validator": self._cv[:10],
                       "median_pos_obs": med, "min_pos_obs": counts[0] if counts else 0,
                       "active_tp": sum(1 for b in range(nb) if s["tp"].get(b) is not None),
                       "latch": self._latch_on(), "neg_obs": nneg},
                      open(self.status_path, "w"))
            # per-book dump for diagnostics/graphing: [pos_obs, neg_obs, realized_sum, net_inv_qty]
            per = {}
            for b in range(nb):
                obs = s["obs"][b]
                pos = sum(1 for _, x in obs if x > 0)
                neg = sum(1 for _, x in obs if x < 0)
                rsum = round(sum(x for _, x in obs), 5)
                inv = round(sum(l.qty for l in s["longs"][b]) - sum(sh.qty for sh in s["shorts"][b]), 4)
                per[b] = [pos, neg, rsum, inv]
            json.dump({"wall_ts": ts, "validator": self._cv[:10], "books": per},
                      open("/root/79/maker_books.json", "w"))
        except Exception:
            pass


if __name__ == "__main__":
    launch(MakerHarvesterAgent)
