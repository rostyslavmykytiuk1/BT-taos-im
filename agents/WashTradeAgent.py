"""
WashTradeAgent — two-UID coordinated round-trip engine for an SN-79 scoring-vuln PoC.

RESPONSIBLE-DISCLOSURE RESEARCH ONLY. See docs/WASH_TRADE_POC_AND_DISCLOSURE.md.
Demonstrates that two operator-controlled UIDs can manufacture consistent positive realized
PnL on ONE UID (the WINNER), undetected by the validator's same-UID self-trade filter
(`Ma == Ta` only, so cross-UID trades are never flagged; self_vol stays 0).

MECHANISM
  Two miners on one box coordinate through LOCAL side-channel files (per winner-uid, per
  validator). Each round trip the WINNER alternates a LONG RT and a SHORT RT:
    * ENTRY is a MARKETABLE limit-IOC that crosses the spread → fills instantly (long = BUY at
      the ask, short = SELL at the bid). It is a TAKER fill. NEVER leveraged: a short simply
      SELLS base the agent already owns (we start with both base and quote), guarded so we
      never sell more than we hold — so NO loans are ever taken.
    * EXIT is a RESTING limit at the ACTUAL entry fill price ± gap, where gap covers BOTH legs'
      fees: gap = (taker_fee + maker_fee)*mid + margin. While the winner is STILL HOLDING it
      PUBLISHES the exit (side, price, remaining-held qty) to the channel every step; it stops
      publishing only when the position is genuinely closed.
    * SINK reads the channel and fires a marketable IOC at the exit price, DYNAMICALLY sized to
      sweep the WHOLE book depth up through that price (so it is guaranteed to reach the winner's
      resting order, sweeping any strangers in between) — bounded ONLY by its owned balance, never
      a cap and never a loan. It does not "fire once and forget": while the winner keeps
      publishing (still holding), the sink keeps sweeping (guarded against duplicate in-flight
      orders) until the position is actually filled. So "sink acted but didn't reach the winner"
      cannot persist — it retries until the winner is closed (or a stranger fills it first, which
      is fine — the winner still wins).
    * Alternation keeps the sink balanced; it also flattens residual swept inventory at market.
  The winner wins every RT (gap, funded by the sink). self_vol = 0 (winner != sink).

PARTNER-AGNOSTIC WINNER
  Winner writes to wash_<own_uid>_<val>.json; it needs NOTHING about the sink. The SINK is
  configured with wash_partner_uid = <winner uid> to find that file. Swap the sink (dereg →
  register/repurpose another with the same wash_partner_uid) without touching the winner.

ACTIVITY (10-min window)
  Each active book completes >=1 RT within the validator's ~600 sim-s window: cadence wash_gap_*
  is < that, an activity backstop forces a cycle if a book idles wash_activity_s, and a stuck
  position is force-flattened at wash_giveup_s (< the window) — which itself completes a RT.

Params (QUOTE the whole AGENT_PARAMS string in .env so spaces survive sourcing):
  wash_role          winner | sink            (REQUIRED)
  wash_partner_uid   int   (SINK only)        (the winner's uid; winner ignores it)
  wash_channel_dir   str   (default /dev/shm)
  wash_books         auto | "12,37,88"        (auto = quietest)
  wash_max_books     int   (default 80)
  wash_lot_min/max   float (0.27 / 0.40)
  wash_gap_min_s/max_s float (240 / 480)      (per-book sim-seconds between RTs; < 600 window)
  wash_margin_bps    float (default 3.0)      (winner profit margin above the round-trip fee)
  wash_activity_s    float (default 480)
  wash_giveup_s      float (default 420)      (force-flatten a stuck position; < 600 window)
  wash_entry_slip_bps float (default 12)      (slippage cap on the marketable entry)
  wash_sink_flatten_lots float (default 2.0)
  wash_summary_s     float (default 30)
  wash_debug         0 | 1
"""

import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol import FinanceAgentResponse, MarketSimulationStateUpdate
from taos.im.protocol.events import TradeEvent
from taos.im.protocol.models import OrderDirection, STP, TimeInForce

_NS = 1_000_000_000
EXCHANGE_MIN_ORDER_SIZE = 0.25

DEF_LOT_MIN = 0.27
DEF_LOT_MAX = 0.40
DEF_GAP_MIN_S = 240.0
DEF_GAP_MAX_S = 480.0
DEF_MARGIN_BPS = 3.0
DEF_ACTIVITY_S = 480.0
DEF_GIVEUP_S = 420.0
DEF_ENTRY_SLIP_BPS = 12.0
DEF_SINK_FLATTEN_LOTS = 2.0
DEF_MAX_BOOKS = 80
DEF_SUMMARY_S = 30.0
PENDING_TIMEOUT_S = 5.0
QUOTE_EXPIRY_S = 30.0

KAPPA_TAU = 0.0
KAPPA_MIN_OBS = 3
KAPPA_RT_HISTORY_S = 10_800.0
MAX_INACTIVE_RATIO = 0.375

MAIN_VALIDATOR = "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF"

IDLE, ENTRY, EXIT = 0, 1, 2


@dataclass
class _Inv:
    longs: deque = field(default_factory=deque)
    shorts: deque = field(default_factory=deque)


@dataclass
class _BookState:
    seen_ns: int = 0
    pending_ns: int = 0
    last_rt_ns: int = 0
    next_open_ns: int = 0
    open_ns: int = 0
    phase: int = IDLE
    direction: int = 1          # winner: +1 long-RT, -1 short-RT (toggles each completed RT)
    cur_lot: float = 0.0
    gap: float = 0.0            # price gap locked when the entry is fired
    exit_px: float = 0.0        # set from the ACTUAL entry fill price ± gap
    rts: int = 0
    net_pnl: float = 0.0
    partner_fills: int = 0
    stranger_fills: int = 0
    rt_events: list[tuple[int, float]] = field(default_factory=list)


class WashTradeAgent(FinanceSimulationAgent):

    # ------------------------------------------------------------------ setup
    def initialize(self) -> None:
        bt.logging.set_info()
        cfg = self.config
        raw_role = str(getattr(cfg, "wash_role", "winner")).strip().lower()
        self.role = {"maker": "winner", "taker": "sink"}.get(raw_role, raw_role)
        if self.role not in ("winner", "sink"):
            bt.logging.warning(f"[Wash uid={self.uid}] bad wash_role={raw_role!r}; defaulting to winner")
            self.role = "winner"
        self.partner_uid = int(float(getattr(cfg, "wash_partner_uid", -1)))
        self.channel_dir = str(getattr(cfg, "wash_channel_dir", "/dev/shm"))

        lot_single = getattr(cfg, "wash_lot", None)
        if lot_single is not None:
            self.lot_min = self.lot_max = float(lot_single)
        else:
            self.lot_min = float(getattr(cfg, "wash_lot_min", DEF_LOT_MIN))
            self.lot_max = float(getattr(cfg, "wash_lot_max", DEF_LOT_MAX))
        if self.lot_max < self.lot_min:
            self.lot_min, self.lot_max = self.lot_max, self.lot_min

        self.max_books = int(float(getattr(cfg, "wash_max_books", DEF_MAX_BOOKS)))
        self.gap_min_ns = int(float(getattr(cfg, "wash_gap_min_s", DEF_GAP_MIN_S)) * _NS)
        self.gap_max_ns = int(float(getattr(cfg, "wash_gap_max_s", DEF_GAP_MAX_S)) * _NS)
        if self.gap_max_ns < self.gap_min_ns:
            self.gap_min_ns, self.gap_max_ns = self.gap_max_ns, self.gap_min_ns
        self.margin_bps = float(getattr(cfg, "wash_margin_bps", DEF_MARGIN_BPS))
        self.activity_ns = int(float(getattr(cfg, "wash_activity_s", DEF_ACTIVITY_S)) * _NS)
        self.giveup_ns = int(float(getattr(cfg, "wash_giveup_s", DEF_GIVEUP_S)) * _NS)
        self.entry_slip = float(getattr(cfg, "wash_entry_slip_bps", DEF_ENTRY_SLIP_BPS)) / 1e4
        self.sink_flatten_lots = float(getattr(cfg, "wash_sink_flatten_lots", DEF_SINK_FLATTEN_LOTS))
        self.summary_ns = int(float(getattr(cfg, "wash_summary_s", DEF_SUMMARY_S)) * _NS)
        self.debug = bool(int(float(getattr(cfg, "wash_debug", 0))))
        self.pending_timeout_ns = int(PENDING_TIMEOUT_S * _NS)
        self.quote_expiry_ns = int(QUOTE_EXPIRY_S * _NS)
        self.kappa_rt_history_ns = int(KAPPA_RT_HISTORY_S * _NS)

        books_raw = str(getattr(cfg, "wash_books", "auto")).strip().lower()
        self.fixed_books = None if books_raw in ("", "auto", "none", "all") else \
            [int(b) for b in books_raw.replace(" ", "").split(",") if b != ""]

        self.exch_min = EXCHANGE_MIN_ORDER_SIZE
        self._flat_eps = 0.5 * 10 ** (-4)
        self._tick = 0.01
        self._pd: int | None = None
        self._vd: int | None = None
        self._book_count = 128
        self._rng = random.Random()

        self.inv: dict[str, dict[int, _Inv]] = {}
        self.books_state: dict[str, dict[int, _BookState]] = {}
        self._sim_id: dict[str, str] = {}
        self._step_ts_ns: dict[str, int] = {}
        self._last_summary_ns: dict[str, int] = {}
        self._active_validator: str | None = None

        try:
            os.makedirs(self.channel_dir, exist_ok=True)
        except OSError:
            pass

        bt.logging.info(
            f"[Wash uid={self.uid}] ROLE={self.role} partner={self.partner_uid} chan={self.channel_dir} "
            f"lot=[{self.lot_min},{self.lot_max}] max_books={self.max_books} "
            f"gap=[{self.gap_min_ns/_NS:.0f},{self.gap_max_ns/_NS:.0f}]s margin={self.margin_bps}bps "
            f"activity={self.activity_ns/_NS:.0f}s giveup={self.giveup_ns/_NS:.0f}s debug={self.debug}")

    # ------------------------------------------------------------------ lifecycle
    def update(self, state: MarketSimulationStateUpdate) -> None:
        self._active_validator = state.dendrite.hotkey
        self._step_ts_ns[self._active_validator] = int(state.timestamp)
        self._ensure_simulation(self._active_validator, state.config.simulation_id)
        super().update(state)

    def _ensure_simulation(self, validator: str, simulation_id: str | None) -> None:
        if self._sim_id.get(validator) == simulation_id:
            return
        self.inv.pop(validator, None)
        self.books_state.pop(validator, None)
        self._last_summary_ns.pop(validator, None)
        if self.role == "winner":
            self._write_channel(validator, simulation_id, {})   # clear stale exits on a new sim
        if simulation_id is not None:
            self._sim_id[validator] = simulation_id
        else:
            self._sim_id.pop(validator, None)
        bt.logging.info(f"[Wash uid={self.uid}] new simulation: {validator[:8]} sim_id={simulation_id}")

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        validator = state.dendrite.hotkey
        cfg = state.config
        sim_id = cfg.simulation_id
        self._sync_precision(cfg.priceDecimals, cfg.volumeDecimals)
        self._book_count = getattr(cfg, "book_count", self._book_count)
        now = state.timestamp
        accounts = state.accounts.get(self.uid, {})   # per-request → no shared-field race
        targets = self._target_books(state, accounts)
        chan = self._read_channel(validator, sim_id) if self.role == "sink" else {}
        for book_id in targets:
            book = state.books.get(book_id)
            account = accounts.get(book_id) if book else None
            if book is None or account is None or not book.bids or not book.asks:
                continue
            try:
                if self.role == "winner":
                    self._step_winner(response, validator, book_id, book, account, now)
                else:
                    self._step_sink(response, validator, book_id, book, account, now, chan.get(str(book_id)))
            except Exception as ex:
                bt.logging.warning(f"[Wash uid={self.uid}] step {book_id}: {ex}")
        if self.role == "winner":
            self._publish_exits(validator, sim_id)
        self._maybe_summary(validator, now)
        return response

    def _target_books(self, state, accounts) -> list[int]:
        avail = [b for b in accounts.keys() if b in state.books]
        if self.fixed_books is not None:
            return [b for b in self.fixed_books if b in avail]
        scored = []
        for b in avail:
            bk = state.books.get(b)
            if not bk or not bk.bids or not bk.asks:
                continue
            mid = 0.5 * (bk.bids[0].price + bk.asks[0].price)
            if mid <= 0:
                continue
            spread_bps = (bk.asks[0].price - bk.bids[0].price) / mid * 1e4
            touch = (bk.bids[0].quantity or 0.0) + (bk.asks[0].quantity or 0.0)
            scored.append((spread_bps / (1.0 + touch), b))
        scored.sort(reverse=True)
        cap = self.max_books if self.max_books > 0 else len(scored)
        return [b for _, b in scored[:cap]]

    # ------------------------------------------------------------------ WINNER
    def _step_winner(self, response, validator, book_id, book, account, now) -> None:
        inv = self._inv(validator, book_id)
        st = self._bstate(validator, book_id)
        if st.seen_ns == 0:
            st.seen_ns = now
            st.next_open_ns = now + self._rng.randint(0, self.gap_max_ns)
        if self._waiting(st, now):
            return
        net = self._net_qty(inv)
        best_bid, best_ask = book.bids[0].price, book.asks[0].price
        mid = 0.5 * (best_bid + best_ask)

        if abs(net) >= self.exch_min:                       # holding → rest exit, or give up
            st.phase = EXIT
            if st.open_ns and (now - st.open_ns) >= self.giveup_ns:
                self._flatten(response, validator, account, book_id, net, best_bid, best_ask, "giveup")
                return
            self._post_exit(response, validator, account, book_id, net)
            return

        st.phase = IDLE                                     # flat → open
        idle = (st.last_rt_ns and (now - st.last_rt_ns) >= self.activity_ns) or \
               (st.last_rt_ns == 0 and (now - st.seen_ns) >= self.activity_ns)
        if now < st.next_open_ns and not idle:
            return
        gap = self._gap_px(account, mid)
        lot = self._rand_lot()
        free_q = self._free(account.quote_balance)
        free_b = self._free(account.base_balance)
        if st.direction > 0:                                # LONG RT: marketable BUY (uses quote)
            px = round(best_ask * (1.0 + self.entry_slip), self._pd)
            if free_q < lot * px:
                self._dbg(validator, f"SKIP book={book_id} low_quote")
                return
            st.gap, st.cur_lot, st.phase = gap, lot, ENTRY
            self._submit(response, validator, book_id, OrderDirection.BUY, lot, px, ioc=True)
            self._dbg(validator, f"ENTRY book={book_id} long BUY {lot}@{px}")
        else:                                               # SHORT RT: SELL OWNED base (no loan)
            if free_b < lot:                                # never sell more than we hold → no borrow
                self._dbg(validator, f"SKIP book={book_id} low_base (no short w/o owned base)")
                return
            px = round(best_bid * (1.0 - self.entry_slip), self._pd)
            st.gap, st.cur_lot, st.phase = gap, lot, ENTRY
            self._submit(response, validator, book_id, OrderDirection.SELL, lot, px, ioc=True)
            self._dbg(validator, f"ENTRY book={book_id} short SELL {lot}@{px}")

    def _post_exit(self, response, validator, account, book_id, net) -> None:
        st = self._bstate(validator, book_id)
        qty = round(abs(net), self._vd_or(4))
        px = st.exit_px
        if px <= 0:
            return
        if net > 0:                                         # long → SELL exit (sell owned base)
            qty = round(min(qty, self._free(account.base_balance)), self._vd_or(4))
            self._rest(response, validator, account, book_id, OrderDirection.SELL, qty, px)
        else:                                               # short → BUY exit (cover with quote)
            self._rest(response, validator, account, book_id, OrderDirection.BUY, qty, px)

    # ------------------------------------------------------------------ SINK
    def _step_sink(self, response, validator, book_id, book, account, now, exit_entry) -> None:
        inv = self._inv(validator, book_id)
        st = self._bstate(validator, book_id)
        if st.seen_ns == 0:
            st.seen_ns = now
        if self._waiting(st, now):
            return
        net = self._net_qty(inv)
        best_bid, best_ask = book.bids[0], book.asks[0]

        cap = self.sink_flatten_lots * max(self.lot_max, self.exch_min)   # flatten residual
        if abs(net) > cap:
            self._flatten(response, validator, account, book_id, net, best_bid.price, best_ask.price, "rebalance")
            return

        if not exit_entry:                                  # winner not holding here → nothing to fill
            return
        side = exit_entry.get("s")
        price = exit_entry.get("p")
        wqty = exit_entry.get("q")
        if price is None or wqty is None or price <= 0:
            return
        # DYNAMIC, NO CAP: sweep the ENTIRE book depth up to (and incl) the winner's exit price, so the
        # marketable IOC is guaranteed to reach the winner's resting order. Bounded ONLY by owned balance
        # (never a loan). Over-sizing is safe — the IOC limit at `price` never fills past the winner's
        # exit, so excess cancels. The `_waiting` guard above stops duplicate in-flight orders; while the
        # winner is still holding it keeps re-publishing its remaining qty, so we retry until it's filled.
        if side == int(OrderDirection.SELL):               # winner SELL exit → sink BUYS up through it
            depth = sum((a.quantity or 0.0) for a in book.asks if a.price <= price + self._tick / 2)
            q = min(depth + wqty, self._free(account.quote_balance) / max(price, 1e-9))   # +wqty buffer
            q = round(q, self._vd_or(4))
            if q >= self.exch_min:
                self._submit(response, validator, book_id, OrderDirection.BUY, q, round(price, self._pd), ioc=True)
        else:                                              # winner BUY exit → sink SELLS owned base down through it
            depth = sum((bd.quantity or 0.0) for bd in book.bids if bd.price >= price - self._tick / 2)
            q = min(depth + wqty, self._free(account.base_balance))   # owned base only, no loan
            q = round(q, self._vd_or(4))
            if q >= self.exch_min:
                self._submit(response, validator, book_id, OrderDirection.SELL, q, round(price, self._pd), ioc=True)

    def _flatten(self, response, validator, account, book_id, net, best_bid, best_ask, reason) -> None:
        slip = self.entry_slip
        self._cancel_all(response, account, book_id)
        if net > 0:                                         # sell owned base
            qty = round(min(abs(net), self._free(account.base_balance)), self._vd_or(4))
            px = round(best_bid * (1.0 - slip), self._pd)
            self._submit(response, validator, book_id, OrderDirection.SELL, qty, px, ioc=True)
        else:                                               # buy back (cover) with quote
            px = round(best_ask * (1.0 + slip), self._pd)
            qty = round(min(abs(net), self._free(account.quote_balance) / max(px, 1e-9)), self._vd_or(4))
            self._submit(response, validator, book_id, OrderDirection.BUY, qty, px, ioc=True)
        if self._rt_log_enabled(validator):
            bt.logging.info(f"[Wash uid={self.uid}] FLATTEN book={book_id} {reason} net={net:+.4f} @~{px}")

    # ------------------------------------------------------------------ channel (two-way)
    def _chan_path(self, validator: str, winner_uid: int) -> str:
        return os.path.join(self.channel_dir, f"wash_{winner_uid}_{validator[:8]}.json")

    def _publish_exits(self, validator: str, sim_id) -> None:
        # Publish CURRENT held qty for every book the winner is still holding. The winner keeps
        # publishing (with the shrinking remaining qty) until its position is genuinely closed — so a
        # sink that missed (or only partially filled) keeps seeing the exit and retries. No seq/ack:
        # "published" == "still waiting to be filled". Stops automatically when the RT completes.
        books = {}
        for b, st in self.books_state.get(validator, {}).items():
            if st.phase != EXIT or st.exit_px <= 0:
                continue
            net = self._net_qty(self._inv(validator, b))
            held = abs(net)
            if held < self.exch_min:
                continue
            side = OrderDirection.SELL if net > 0 else OrderDirection.BUY
            books[str(b)] = {"s": int(side), "p": st.exit_px, "q": round(held, self._vd_or(4))}
        self._write_channel(validator, sim_id, books)

    def _write_channel(self, validator: str, sim_id, books: dict) -> None:
        self._atomic_write(self._chan_path(validator, self.uid), {"sim": sim_id, "ts": self._step_ts_ns.get(validator, 0), "books": books}, validator)

    def _read_channel(self, validator: str, sim_id) -> dict:
        d = self._safe_read(self._chan_path(validator, self.partner_uid))
        if d.get("sim") != sim_id:                          # reject other-sim data
            return {}
        return d.get("books", {})

    def _atomic_write(self, path: str, payload: dict, validator: str) -> None:
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        except OSError as ex:
            self._dbg(validator, f"channel write failed: {ex}")

    @staticmethod
    def _safe_read(path: str) -> dict:
        try:
            with open(path) as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    # ------------------------------------------------------------------ order submit (NO leverage, NO loans)
    def _waiting(self, st: _BookState, now: int) -> bool:
        if st.pending_ns and (now - st.pending_ns) < self.pending_timeout_ns:
            return True
        st.pending_ns = 0
        return False

    def _rest(self, response, validator, account, book_id, direction, qty, price) -> None:
        """Maker: post a passive (post_only) GTT limit; keep an identical resting order else repost.
        Never leveraged — a SELL only rests size we own (guarded by the caller)."""
        if qty < self.exch_min or price <= 0:
            return
        if direction == OrderDirection.SELL and self._free(account.base_balance) < qty - self._flat_eps:
            return                                          # never rest a sell beyond owned base
        vtick = 0.5 * 10 ** (-self._vd_or(4))
        for o in (account.orders or []):
            side = OrderDirection.BUY if o.side == 0 else OrderDirection.SELL
            if side == direction and o.price is not None and abs(o.price - price) < self._tick / 2 \
                    and abs((o.quantity or 0.0) - qty) < vtick:
                return
        self._cancel_all(response, account, book_id)
        response.limit_order(book_id=book_id, direction=direction, quantity=qty, price=price,
                             postOnly=True, timeInForce=TimeInForce.GTT, expiryPeriod=self.quote_expiry_ns,
                             stp=STP.CANCEL_OLDEST)
        self._bstate(validator, book_id).pending_ns = self._step_ts_ns.get(validator, 0)

    def _submit(self, response, validator, book_id, direction, qty, price, *, ioc: bool = False) -> None:
        if qty < self.exch_min or price <= 0:
            return
        response.limit_order(book_id=book_id, direction=direction, quantity=qty, price=price,
                             postOnly=False, timeInForce=TimeInForce.IOC if ioc else TimeInForce.GTC,
                             stp=STP.CANCEL_OLDEST)
        self._bstate(validator, book_id).pending_ns = self._step_ts_ns.get(validator, 0)

    def _cancel_all(self, response, account, book_id) -> None:
        if account.orders:
            response.cancel_orders(book_id, [o.id for o in account.orders])

    # ------------------------------------------------------------------ fills
    def onTrade(self, event: TradeEvent, validator: str | None = None) -> None:
        if event.bookId is None:
            return
        validator = validator or self._active_validator
        if validator is None:
            return
        if self.uid == event.takerAgentId:
            is_buy = event.side == OrderDirection.BUY
            fee, cp = event.takerFee, event.makerAgentId
        elif self.uid == event.makerAgentId:
            is_buy = event.side == OrderDirection.SELL
            fee, cp = event.makerFee, event.takerAgentId
        else:
            return
        ts = int(event.timestamp) if event.timestamp else self._step_ts_ns.get(validator, 0)
        st = self._bstate(validator, event.bookId)
        if cp == self.partner_uid:
            st.partner_fills += 1
        else:
            st.stranger_fills += 1
        if self.debug and self._rt_log_enabled(validator):
            bt.logging.info(f"[Wash uid={self.uid} FILL] book={event.bookId} role={self.role} "
                            f"{'BUY' if is_buy else 'SELL'} q={event.quantity}@{event.price} fee={fee:+.5f} cp={cp}")
        self._apply_fill(validator, event.bookId, is_buy, event.quantity, event.price, fee, ts)

    def _apply_fill(self, validator, book_id, is_buy, qty, price, fee, ts) -> None:
        inv = self._inv(validator, book_id)
        st = self._bstate(validator, book_id)
        st.pending_ns = 0
        was_flat = abs(self._net_qty(inv)) < self._flat_eps
        realized, rtv, matched_ts, gross = self._match_fifo(inv, is_buy, qty, price, fee, ts)
        now_net = self._net_qty(inv)
        if was_flat and abs(now_net) >= self._flat_eps and self.role == "winner":
            # ENTRY just filled → lock the exit price off the ACTUAL fill price
            st.open_ns = ts
            entry_avg = inv.longs[0][2] if now_net > 0 else inv.shorts[0][2]
            gap = st.gap if st.gap > 0 else self._tick
            st.exit_px = round(entry_avg + gap, self._pd) if now_net > 0 else round(entry_avg - gap, self._pd)
            st.phase = EXIT
        elif was_flat and abs(now_net) >= self._flat_eps:
            st.open_ns = ts
        if rtv > 0 and abs(now_net) < self.exch_min:
            st.last_rt_ns = ts
            st.open_ns = 0
            st.rts += 1
            st.net_pnl += realized
            st.rt_events.append((ts, realized))
            self._prune_rt(st, ts)
            if self.role == "winner":
                st.direction = -st.direction              # alternate long/short RT
                st.phase = IDLE
                st.exit_px = 0.0
                st.next_open_ns = ts + self._rng.randint(self.gap_min_ns, self.gap_max_ns)
            if self._rt_log_enabled(validator):
                bt.logging.info(f"[Wash uid={self.uid} RT] book={book_id} role={self.role} net={realized:+.5f} "
                                f"book_rts={st.rts} book_net={st.net_pnl:+.4f} partner={st.partner_fills} stranger={st.stranger_fills}")

    def _match_fifo(self, inv: _Inv, is_buy, qty, price, fee, ts):
        close_book = inv.shorts if is_buy else inv.longs
        open_book = inv.longs if is_buy else inv.shorts
        realized = gross = rtv = 0.0
        remaining = qty
        matched_ts = None
        qinv = 1.0 / qty if qty > 0 else 0.0
        while remaining > self._flat_eps and close_book:
            o_ts, o_qty, o_px, o_fee = close_book[0]
            if matched_ts is None:
                matched_ts = o_ts
            take = min(o_qty, remaining)
            price_pnl = (o_px - price) * take if is_buy else (price - o_px) * take
            if o_qty <= remaining + self._flat_eps:
                close_fee, open_fee = fee * o_qty * qinv, o_fee
                close_book.popleft()
            else:
                close_fee, open_fee = fee * take * qinv, o_fee * (take / o_qty)
                close_book[0] = (o_ts, o_qty - take, o_px, o_fee - open_fee)
            realized += price_pnl - open_fee - close_fee
            gross += price_pnl
            rtv += take
            remaining -= take
        if remaining > self._flat_eps:
            open_book.append((ts, remaining, price, fee * remaining * qinv))
        return realized, rtv, matched_ts, gross

    # ------------------------------------------------------------------ summary / kappa proxy
    def _maybe_summary(self, validator, now) -> None:
        if not self._rt_log_enabled(validator):
            return
        last = self._last_summary_ns.get(validator, 0)
        if last and (now - last) < self.summary_ns:
            return
        self._last_summary_ns[validator] = now
        states = self.books_state.get(validator, {})
        if not states:
            return
        rts = sum(s.rts for s in states.values())
        partner = sum(s.partner_fills for s in states.values())
        stranger = sum(s.stranger_fills for s in states.values())
        net = sum(s.net_pnl for s in states.values())
        active = sum(1 for s in states.values() if s.rts > 0)
        held = sum(1 for v in self.inv.get(validator, {}).values() if abs(self._net_qty(v)) >= self.exch_min)
        k = self._kappa_proxy(validator, now, self._book_count)
        bt.logging.info(f"[Wash uid={self.uid} SUMMARY] role={self.role} active={active}/{self._book_count} held={held} "
                        f"rts={rts} net={net:+.4f} partner={partner} stranger={stranger} "
                        f"match={(100*partner/(partner+stranger)) if (partner+stranger) else 0:.0f}% "
                        f"kappa={f'{k:.4f}' if k is not None else 'n/a'}")

    def _kappa_proxy(self, validator, now, book_count) -> float | None:
        states = self.books_state.get(validator, {})
        cutoff = now - self.kappa_rt_history_ns
        grid = sorted({t for s in states.values() for t, _ in s.rt_events if t >= cutoff})
        if len(grid) < KAPPA_MIN_OBS:
            return None
        scored = []
        for s in states.values():
            by = {t: p for t, p in s.rt_events if t >= cutoff}
            if sum(1 for v in by.values() if v != 0.0) < KAPPA_MIN_OBS:
                continue
            k = self._kappa3([by.get(t, 0.0) for t in grid])
            if k is not None:
                scored.append(max(0.0, min(1.0, (k + 2.5) / 5.0)))
        if not scored:
            return None
        max_inactive = int(MAX_INACTIVE_RATIO * book_count)
        inactive = book_count - len(scored)
        data = scored + ([0.0] * (inactive - max_inactive) if inactive > max_inactive else [])
        return self._median(data)

    @staticmethod
    def _median(v):
        if not v:
            return 0.0
        s = sorted(v); m = len(s) // 2
        return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])

    @classmethod
    def _kappa3(cls, series, tau=KAPPA_TAU):
        if not series or sum(1 for x in series if x != 0.0) < KAPPA_MIN_OBS:
            return None
        med = cls._median(series)
        mad = max(cls._median([abs(x - med) for x in series]), 1e-6)
        r = [x / mad for x in series]
        n = len(r); mean_r = sum(r) / n
        lpm3 = sum(max(tau - x, 0.0) ** 3 for x in r) / n
        upm3 = sum(max(x - tau, 0.0) ** 3 for x in r) / n
        std = math.sqrt(sum((x - mean_r) ** 2 for x in r) / n)
        reg = ((abs(mean_r) + std) * 0.1) ** 3
        eps = 1e-2 if mean_r > tau else 1e-6
        if lpm3 > eps:
            return (mean_r - tau) / ((lpm3 + reg) ** (1 / 3))
        if mean_r > tau:
            return (mean_r - tau) / ((upm3 + reg) ** (1 / 3))
        return 0.0

    def _prune_rt(self, st, now):
        cutoff = now - self.kappa_rt_history_ns
        if st.rt_events and st.rt_events[0][0] < cutoff:
            st.rt_events = [(t, p) for t, p in st.rt_events if t >= cutoff]

    # ------------------------------------------------------------------ helpers
    def _gap_px(self, account, mid):
        """Gap must beat BOTH legs: ENTRY is a taker, EXIT is a maker. Either rate can be a rebate."""
        mf = self._maker_fee_rate(account); mf = mf if mf is not None else 0.0
        tf = self._taker_fee_rate(account); tf = tf if tf is not None else 0.0
        return max(self._tick, (tf + mf) * mid + (self.margin_bps / 1e4) * mid)

    def _rand_lot(self):
        vd = self._vd_or(4)
        floor = round(self.exch_min + 10 ** (-vd), vd)
        lo = max(self.lot_min, floor)
        return round(self._rng.uniform(lo, max(lo, self.lot_max)), vd)

    def _vd_or(self, default):
        return self._vd if self._vd is not None else default

    def _dbg(self, validator, msg):
        if self.debug and self._rt_log_enabled(validator):
            bt.logging.info(f"[Wash uid={self.uid} DBG] {msg}")

    def _sync_precision(self, pd, vd):
        if pd == self._pd and vd == self._vd:
            return
        self._pd, self._vd = pd, vd
        self._tick = 10 ** (-pd)
        self.exch_min = max(EXCHANGE_MIN_ORDER_SIZE, 10 ** (-vd))
        self._flat_eps = 0.5 * 10 ** (-vd)
        floor = round(self.exch_min + 10 ** (-vd), vd)
        self.lot_min = round(max(self.lot_min, floor), vd)
        self.lot_max = round(max(self.lot_max, self.lot_min), vd)
        bt.logging.info(f"[Wash uid={self.uid}] pd={pd} tick={self._tick} vd={vd} lot=[{self.lot_min},{self.lot_max}]")

    def _inv(self, validator, book_id):
        return self.inv.setdefault(validator, {}).setdefault(book_id, _Inv())

    def _bstate(self, validator, book_id):
        return self.books_state.setdefault(validator, {}).setdefault(book_id, _BookState())

    @staticmethod
    def _long_qty(inv):
        return sum(q for _, q, _, _ in inv.longs)

    @staticmethod
    def _short_qty(inv):
        return sum(q for _, q, _, _ in inv.shorts)

    def _net_qty(self, inv):
        return self._long_qty(inv) - self._short_qty(inv)

    @staticmethod
    def _free(balance):
        return (balance.free or 0.0) if balance is not None else 0.0

    def _maker_fee_rate(self, account):
        fees = getattr(account, "fees", None)
        rate = getattr(fees, "maker_fee_rate", None) if fees is not None else None
        try:
            return float(rate) if rate is not None else None
        except (TypeError, ValueError):
            return None

    def _taker_fee_rate(self, account):
        fees = getattr(account, "fees", None)
        rate = getattr(fees, "taker_fee_rate", None) if fees is not None else None
        try:
            return float(rate) if rate is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _rt_log_enabled(validator):
        return validator == MAIN_VALIDATOR


if __name__ == "__main__":
    launch(WashTradeAgent)
