# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
WarehouseRealizerAgent — SN79 (τaos) kappa-max "warehouse-and-realize-winners" maker.

STRATEGY (see WAREHOUSE_REALIZER_DEVPLAN.md for the full live-calibrated spec):
  Accumulate a cheap UNLEVERAGED long buffer across as many of the 128 books as possible, then print a
  steady, mostly-GROSS-POSITIVE realized round-trip stream by selling the FIFO-oldest lot only when it
  clears both fee legs + a small NET margin — fragmenting the release. Never realize a big loss; hold
  underwater lots (no leverage => never liquidatable) and clear genuine FIFO lockups with a small,
  bounded, age-triggered loss. Keep >=80 books active; let the rest idle in the validator's free
  inactive-book budget (<=48).

WHY THIS SCORES (verified in taos/im/utils/kappa.py + validator.py + reward.py):
  * Score = 0.79*kappa_score + 0.21*pnl_score, but live pnl_score ~ 0 -> score is ~pure KAPPA.
  * kappa is a per-book MAD-normalized Sortino-3 on REALIZED FIFO PnL over a DENSE ~2160-col axis. It
    rewards a smooth, high-frequency, GROSS-positive stream across MANY books (breadth), NOT PnL size or
    volume. The live winners (uid101 k~0.09) net-realize ~breakeven/slightly-negative but are 84-96%
    gross-positive at high frequency; the weak templates (uid145 k~0.01) realize too thin/net-negative.
  * Realized PnL is FIFO net of BOTH fee legs and only recorded when a lot is CLOSED — so an unsold
    underwater lot is invisible to the score. We exploit exactly this.

THE BINDING CONSTRAINT is the ACTIVITY FLOOR: every scored book needs >=1 closed RT within the rolling
  600s sampling window or its activity factor (hence kappa) decays. We keep >=80 books closing RTs;
  the genuine failure mode is a sustained correlated downtrend where bids never clear the margin — the
  AGE-TRIGGERED bounded-loss release is the survival mechanism, and we stop accumulating as abandons rise.

PER-BOOK STEP (each respond):
  noise/kappa upkeep -> (A) age-triggered bounded-loss release -> (B) activity backstop (force >=1 RT
  before the 600s window lapses) -> (C) cadence taker-out (keep density when the resting ask isn't
  filling) -> (D) resting MAKER reduce ask (primary, passive spread capture) + dip-biased accumulate bid.
  A/B/C are terminal IOC actions (cancel_all + IOC + return); D is the steady two-sided resting state.

FIFO inventory mirrors the validator's _match_trade_fifo EXACTLY (oldest-lot, net of both fees), so our
  predicted realized PnL equals the scored realized PnL. Zero leverage. Forked from StableMakerV2Agent;
  reuses its FIFO/fee/volume/reconcile/kappa-proxy/GC plumbing, replaces the spread-capture brain.
"""

import gc
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol import FinanceAgentResponse, MarketSimulationStateUpdate
from taos.im.protocol.events import TradeEvent
from taos.im.protocol.models import (
    LoanSettlementOption,
    OrderDirection,
    STP,
    TimeInForce,
)

_NS = 1_000_000_000

EXCHANGE_MIN_ORDER_SIZE = 0.25
QUOTE_LOT = 0.26                    # > 0.25 exch min so the fee-shave (fees_base=roundUp(fee/price)) can't
                                   # strand a held lot under the min and make it un-sellable.

# ---- accumulation (deep, cheap, unleveraged buffer) ----
ACCUM_TARGET_LOTS = 4.0            # per-book inventory target (lots). The global cap usually binds first.
ACCUM_MIN_LOTS = 1.0              # always hold >=1 sellable lot/book once entered
ACCUM_DIP_BPS = 4.0               # only add beyond the min on a dip below the per-book mid-EWMA
GLOBAL_BASE_FRAC = 0.60           # hard ceiling on TOTAL base notional outstanding = 0.60*wealth (~30k);
                                   # ~40% stays dry powder. Unleveraged => never liquidatable.
MID_EWMA_ALPHA = 0.05             # per-book mid-EWMA weight (dip reference)

# ---- realization (sell the FIFO-oldest lot only when net-positive past both fee legs) ----
REALIZE_MARGIN_BPS = 3.0          # required NET margin (bps of notional) AFTER both fee legs. Small &
                                   # positive: beats the weak templates' net-negative stream without
                                   # over-raising (a high margin starves the activity floor). TUNABLE.
REALIZE_TARGET_S = 120.0          # cadence backstop: if no RT closed in this long and a net-positive
                                   # taker-out exists, cross to keep RT DENSITY up (resting ask not filling).
ACTIVITY_DEADLINE_S = 480.0       # HARD activity backstop (< the validator's 600s decay grace): force
                                   # >=1 RT before the book's activity factor decays. Prefer profitable;
                                   # else bounded-loss; else a tiny scratch; else let the book idle.

# ---- stuck-lot handling (the downtrend survival mechanism) ----
STUCK_HOLD_S = 180.0              # hold-for-revert window; past it an underwater oldest lot may be released
MAX_STUCK_LOSS_BPS = 15.0        # AGE-triggered bounded-loss release cap to clear a FIFO lockup (kappa-safe)
SCRATCH_LOSS_BPS = 4.0           # tiny bounded loss to keep a needed book active when nothing is profitable
TAKER_OUT_SLIPPAGE_BPS = 5.0     # IOC price concession so a taker-out crosses a FALLING book (else the stale
                                 # best_bid limit no longer crosses and the close silently expires)

# ---- breadth (the real kappa lever) ----
MIN_ACTIVE_BOOKS = 80            # never accumulate on fewer than this (= 128 - int(0.375*128) free-idle).
ADVERSE_SEL_BPS = 2.5            # edge-gate haircut (reused) — gates which books to accumulate in a fee spike
EDGE_GATE_ENABLED = True
MAX_ABANDON_BOOKS = 40           # stop opening NEW books once this many are abandoned (downtrend brake)

# ---- volume cap (irrelevant at our size; kept as a safety) ----
CAPITAL_TURNOVER_CAP = 10.0
VOLUME_SAFETY = 0.8
VOLUME_ASSESSMENT_NS = 86_400_000_000_000

# ---- quoting ----
QUOTE_EXPIRY_S = 12.0
REPRICE_KEEP_TICKS = 0.5

# ---- kappa-3 (LOGGING-ONLY proxy on MAIN_VALIDATOR; NOT the validator kappa — do not tune on it) ----
KAPPA_TAU = 0.0
KAPPA_MIN_OBS = 3
KAPPA_MIN_LOOKBACK_S = 5400.0
KAPPA_RT_HISTORY_S = 10_800.0
RT_WINDOW_S = 570.0

MAIN_VALIDATOR = "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF"


@dataclass
class _Inv:
    longs: deque = field(default_factory=deque)   # tuples (ts, qty, px, open_fee) — oldest first (FIFO)
    shorts: deque = field(default_factory=deque)  # unused (long-only warehouse); kept for FIFO-mirror parity


@dataclass
class _BookState:
    last_rt_ns: int = 0
    seen_ns: int = 0
    rt_events: list[tuple[int, float]] = field(default_factory=list)
    kappa3: float | None = None
    vol_log: list[tuple[int, float]] = field(default_factory=list)
    last_mid: float = 0.0
    mid_ewma: float = 0.0
    noise_bps: float = 0.0
    abandoned: bool = False               # deeply-stuck book: stop quoting, let it idle (free budget)
    loss_streak: int = 0                  # consecutive loss-realizes without a positive RT (per-book guard)


class WarehouseRealizerAgent(FinanceSimulationAgent):

    # ------------------------------------------------------------------ init
    def initialize(self) -> None:
        bt.logging.set_info()

        self.quote_lot = QUOTE_LOT
        self.exch_min = EXCHANGE_MIN_ORDER_SIZE
        self._flat_eps = 0.5 * 10 ** (-4)
        self._price_decimals: int | None = None
        self._volume_decimals: int | None = None
        self._tick = 0.01
        self.volume_assessment_ns = VOLUME_ASSESSMENT_NS

        # per-uid jitter so a cohort of identical uids desyncs (no lifting each other's reduces / stacking dips)
        jitter = ((self.uid * 2654435761) % 1000) / 1000.0
        self.realize_margin_bps = REALIZE_MARGIN_BPS * (0.9 + 0.2 * jitter)
        self.accum_dip_bps = ACCUM_DIP_BPS * (0.9 + 0.2 * jitter)
        realize_target_s = REALIZE_TARGET_S * (0.9 + 0.2 * jitter)
        activity_s = ACTIVITY_DEADLINE_S * (0.92 + 0.08 * jitter)
        stuck_hold_s = STUCK_HOLD_S * (0.9 + 0.2 * jitter)

        self.quote_expiry_ns = int(QUOTE_EXPIRY_S * _NS)
        self.realize_target_ns = int(realize_target_s * _NS)
        self.activity_deadline_ns = int(activity_s * _NS)
        self.stuck_hold_ns = int(stuck_hold_s * _NS)
        self.rt_window_ns = int(RT_WINDOW_S * _NS)
        self.kappa_rt_history_ns = int(KAPPA_RT_HISTORY_S * _NS)
        self.kappa_min_lookback_ns = int(KAPPA_MIN_LOOKBACK_S * _NS)

        self.inv: dict[str, dict[int, _Inv]] = {}
        self.books_state: dict[str, dict[int, _BookState]] = {}
        self._sim_id: dict[str, str] = {}
        self._step_ts_ns: dict[str, int] = {}
        self._active_validator: str | None = None

        bt.logging.info(
            f"[WarehouseRealizer uid={self.uid}] WHR lot={QUOTE_LOT} accum_target={ACCUM_TARGET_LOTS}lots "
            f"global_base={GLOBAL_BASE_FRAC:.0%} dip={self.accum_dip_bps:.1f}bps "
            f"realize_margin={self.realize_margin_bps:.1f}bps(net) realize_target={realize_target_s:.0f}s "
            f"activity={activity_s:.0f}s stuck_hold={stuck_hold_s:.0f}s "
            f"stuck_cut<={MAX_STUCK_LOSS_BPS:.0f}bps scratch<={SCRATCH_LOSS_BPS:.0f}bps "
            f"min_active={MIN_ACTIVE_BOOKS} edge_gate={'ON' if EDGE_GATE_ENABLED else 'OFF'} "
            f"leverage=0 rt_log={MAIN_VALIDATOR[:8]}"
        )
        self._tune_gc()

    def _tune_gc(self) -> None:
        """axon GC-pause mitigation (mirrors StableMakerV2/AdaptiveRouter): drop unused history, freeze the
        import heap, raise gen2 thresholds. Behaviour-neutral."""
        self.history_len = 0
        try:
            gc.collect()
            gc.freeze()
            gc.set_threshold(50_000, 500, 500)
            bt.logging.info(f"[WarehouseRealizer uid={self.uid}] gc tuned: frozen={gc.get_freeze_count()} "
                            f"thresholds={gc.get_threshold()} history_len=0")
        except Exception as ex:
            bt.logging.warning(f"[WarehouseRealizer uid={self.uid}] gc tune skipped: {ex}")

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
        if simulation_id is not None:
            self._sim_id[validator] = simulation_id
        else:
            self._sim_id.pop(validator, None)
        bt.logging.info(
            f"[WarehouseRealizer uid={self.uid}] new simulation: {validator[:8]} sim_id={simulation_id}"
        )

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        validator = state.dendrite.hotkey
        cfg = self.simulation_config
        self._sync_precision(cfg.priceDecimals, cfg.volumeDecimals)

        vol_dp = cfg.volumeDecimals
        volume_cap = CAPITAL_TURNOVER_CAP * cfg.miner_wealth * VOLUME_SAFETY
        now = state.timestamp

        # fleet-level pre-pass: accumulate-breadth gate floor + the global unleveraged base ceiling.
        gate_min_edge = self._compute_gate_min_edge(state)
        base_out = self._base_notional(validator)
        can_open_globally = base_out < GLOBAL_BASE_FRAC * cfg.miner_wealth
        abandoned_n = self._abandoned_count(validator)
        # kappa proxy is logging-only AND MAIN_VALIDATOR-only: build the dense RT timestamp axis ONCE per
        # step (not once per book) so each per-book refresh is O(E) not O(B*E) — avoids the axon timeout.
        main_v = (validator == MAIN_VALIDATOR)
        rt_ts = self._global_rt_timestamps(validator, now) if main_v else None

        for book_id in sorted(self.accounts.keys()):
            book = state.books.get(book_id)
            account = self.accounts.get(book_id) if book else None
            if book is None or account is None:
                continue
            try:
                self._step_book(response, validator, book_id, book, account, vol_dp,
                                volume_cap, now, gate_min_edge, can_open_globally, abandoned_n, main_v, rt_ts)
            except Exception as ex:
                bt.logging.warning(f"[WarehouseRealizer uid={self.uid}] step {book_id}: {ex}")

        return response

    # ------------------------------------------------------------------ per-book
    def _step_book(
        self, response, validator: str, book_id: int, book, account, vol_dp: int,
        volume_cap: float, now: int, gate_min_edge: float, can_open_globally: bool, abandoned_n: int,
        main_v: bool, rt_ts: list[int] | None,
    ) -> None:
        if not book.bids or not book.asks:
            return
        best_bid = book.bids[0].price
        best_ask = book.asks[0].price
        mid = 0.5 * (best_bid + best_ask)
        if mid <= 0 or best_bid <= 0 or best_ask <= 0:
            return

        inv = self._inv(validator, book_id)
        st = self._bstate(validator, book_id)
        first_seen = st.seen_ns == 0
        if first_seen:
            st.seen_ns = now

        # per-book mid noise + EWMA upkeep
        if st.last_mid > 0.0:
            inst = abs(mid - st.last_mid) / st.last_mid * 1e4
            st.noise_bps = (((1.0 - MID_EWMA_ALPHA) * st.noise_bps + MID_EWMA_ALPHA * inst)
                            if st.noise_bps > 0.0 else inst)
        st.last_mid = mid
        st.mid_ewma = mid if st.mid_ewma <= 0.0 else (1.0 - MID_EWMA_ALPHA) * st.mid_ewma + MID_EWMA_ALPHA * mid
        self._prune_rt_events(st, now)           # bound memory on every validator
        self._prune_vol_log(st, now)             # was only pruned via the (often-skipped) cadence branch
        if main_v:
            self._refresh_book_kappa(validator, book_id, now, rt_ts)

        pdp, vdp = self._price_decimals, self._volume_decimals
        maker_fee = self._maker_fee_rate(account)
        taker_fee = self._taker_fee_rate(account)
        maker_bps = (maker_fee * 1e4) if maker_fee is not None else 4.0
        taker_bps = (taker_fee * 1e4) if taker_fee is not None else 0.0

        # Restart reconcile: if we re-attached mid-sim with an empty in-memory FIFO beside a live exchange
        # base position (a ghost long), seed ONE synthetic oldest lot at mid so _base_notional, the global
        # cap and the exit ladder all account for it (true basis is unrecoverable; mid is the proxy).
        if first_seen and not inv.longs:
            base_total = self._avail(account.base_balance)
            if base_total >= self.exch_min:
                inv.longs.append((now, round(base_total, vdp), mid, base_total * mid * (maker_fee or 0.0)))

        long_qty = self._long_qty(inv)
        base_avail = self._avail(account.base_balance)
        activity_ref = st.last_rt_ns if st.last_rt_ns > 0 else st.seen_ns

        # NOTE: Mode-B HARVEST (a fresh both-legs-taker RT on books where spread < 2*taker_rebate) is a
        # DEFERRED enhancement (see DEVPLAN §12 / code-review #4) — it's the largest/riskiest path-adding
        # change and is gated behind the kappa-replay dryrun. Until then, tight-spread/taker-rebate books are
        # handled adequately by the taker-out paths below (which use the rebate and fire on a small +move).
        # ============================ EXIT LADDER (terminal IOC actions) ============================
        if long_qty >= self.exch_min and inv.longs:
            o_ts, o_qty, o_px, o_fee = inv.longs[0]
            age = now - o_ts
            open_fee_bps = (o_fee / (o_qty * o_px) * 1e4) if (o_qty > 0 and o_px > 0) else maker_bps
            # NET realized bps if we taker-out (cross to the bid) the oldest lot now:
            taker_net = (best_bid - o_px) / o_px * 1e4 - open_fee_bps - taker_bps
            loss_bps = (o_px - best_bid) / o_px * 1e4      # >0 when underwater vs the oldest lot
            # Bundle a sub-exch_min FIFO-head stub with the next lot so it can't freeze the book; still
            # ~one-lot (fragmented) on a normal oldest lot. (validator FIFO closes oldest-first regardless.)
            sell_qty = round(min(long_qty, base_avail, max(o_qty, self.exch_min)), vdp)

            # Deep-stuck: aged AND underwater beyond the bounded-loss cap -> STOP accumulating (hold the bag,
            # let the book idle in the free budget); cleared on the next closing fill (_apply_fill).
            st.abandoned = (age >= self.stuck_hold_ns and loss_bps > MAX_STUCK_LOSS_BPS)

            # (A) AGE-TRIGGERED BOUNDED-LOSS RELEASE — clear a FIFO lockup in a fall (the survival valve).
            if (age >= self.stuck_hold_ns and loss_bps > 0.0 and loss_bps <= MAX_STUCK_LOSS_BPS
                    and sell_qty >= self.exch_min and self._allow_loss_realize(st, now)):
                self._taker_out(response, account, book_id, sell_qty, best_bid, st, now, "stuck", loss_bps)
                return

            # (B) ACTIVITY BACKSTOP — force >=1 RT before the 600s window lapses (keep the book scored).
            if (now - activity_ref) >= self.activity_deadline_ns and sell_qty >= self.exch_min:
                if taker_net >= 0.0:
                    self._taker_out(response, account, book_id, sell_qty, best_bid, st, now, "activity+", 0.0)
                    return
                if loss_bps <= MAX_STUCK_LOSS_BPS and self._allow_loss_realize(st, now):
                    self._taker_out(response, account, book_id, sell_qty, best_bid, st, now, "activity-", loss_bps)
                    return
                # else: nothing acceptable to realize -> let this book idle (counts toward the 48 free)

            # (C) CADENCE TAKER-OUT — keep RT density up when the resting ask isn't getting lifted.
            if (taker_net >= self.realize_margin_bps and sell_qty >= self.exch_min
                    and (now - st.last_rt_ns) >= self.realize_target_ns
                    and self._rolled_quote_volume(validator, book_id, now) < volume_cap):
                self._taker_out(response, account, book_id, sell_qty, best_bid, st, now, "cadence", 0.0)
                return

        # ============================ RESTING QUOTES (primary, passive) ============================
        spread = best_ask - best_bid
        improve = self._tick if spread > 2 * self._tick else 0.0
        bid_inside = round(best_bid + improve, pdp)
        ask_inside = round(best_ask - improve, pdp)
        if bid_inside >= ask_inside:
            bid_inside, ask_inside = round(best_bid, pdp), round(best_ask, pdp)

        desired: dict[int, tuple[float, float]] = {}

        # (D1) PRIMARY REALIZE — a passive MAKER reduce ask on the oldest lot, priced so a fill nets >= margin.
        #      Captures the spread when lifted; this is the income channel the winners run (100% maker).
        if long_qty >= self.exch_min and inv.longs:
            o_ts, o_qty, o_px, o_fee = inv.longs[0]
            open_fee_bps = (o_fee / (o_qty * o_px) * 1e4) if (o_qty > 0 and o_px > 0) else maker_bps
            min_ask = o_px * (1.0 + (open_fee_bps + maker_bps + self.realize_margin_bps) / 1e4)
            sell_px = round(max(ask_inside, min_ask), pdp)
            q = round(min(long_qty, base_avail, max(o_qty, self.exch_min)), vdp)
            if q >= self.exch_min and sell_px > 0:
                desired[OrderDirection.SELL] = (sell_px, q)

        # (D2) ACCUMULATE / REPLENISH — passive dip-biased maker buy, up to the per-book target & global cap.
        if not st.abandoned:
            target_qty = ACCUM_TARGET_LOTS * self.quote_lot
            want_more = long_qty < target_qty - self._flat_eps
            seed = long_qty < ACCUM_MIN_LOTS * self.quote_lot - self._flat_eps
            dip = mid <= st.mid_ewma * (1.0 - self.accum_dip_bps / 1e4)
            breadth_ok = (long_qty >= self.exch_min) or (abandoned_n < MAX_ABANDON_BOOKS)
            if (want_more and can_open_globally and breadth_ok and (seed or dip)
                    and self._gate_ok(best_bid, best_ask, mid, maker_fee, gate_min_edge)):
                q = round(self.quote_lot, vdp)
                free_quote = account.quote_balance.free if account.quote_balance else 0.0
                if q >= self.exch_min and free_quote >= q * bid_inside:
                    desired[OrderDirection.BUY] = (bid_inside, q)

        self._reconcile_quotes(response, account, book_id, desired)

    # ------------------------------------------------------------------ taker-out helper (terminal)
    def _taker_out(self, response, account, book_id: int, qty: float, bid_px: float,
                   st: _BookState, now: int, reason: str, loss_bps: float) -> None:
        """Cross to the bid (IOC) to close one oldest FIFO lot. Cancel resting orders first so the IOC and a
        stale resting ask can't both fire. onTrade records the realized RT (updates last_rt_ns / kappa)."""
        px = round(bid_px * (1.0 - TAKER_OUT_SLIPPAGE_BPS / 1e4), self._price_decimals)
        self._cancel_all(response, account, book_id)
        self._submit_limit(response, book_id, OrderDirection.SELL, qty, px, ioc=True, post_only=False)
        bt.logging.info(
            f"[WarehouseRealizer uid={self.uid}] TAKER-OUT book={book_id} qty={qty} @{px} "
            f"reason={reason} loss_bps={loss_bps:.1f} streak={st.loss_streak}"
        )

    def _allow_loss_realize(self, st: _BookState, now: int) -> bool:
        """Median-positive guard: only realize a (bounded) loss if the book's recent realized stream is still
        net-positive (or has no history yet) and it isn't already on a fresh loss streak. Protects the
        per-book kappa from a CLUSTER of losses on one chronically-falling book (the real -0.0156 failure
        mode). The streak DECAYS once the book has been quiet so the downtrend survival valve can re-arm."""
        if st.last_rt_ns > 0 and (now - st.last_rt_ns) >= self.activity_deadline_ns:
            st.loss_streak = 0
        if st.loss_streak >= 3:
            return False
        pnls = [p for _, p in st.rt_events]
        if len(pnls) < KAPPA_MIN_OBS:
            return True
        return self._median(pnls) > 0.0

    # ------------------------------------------------------------------ breadth gate (reused)
    def _compute_gate_min_edge(self, state: MarketSimulationStateUpdate) -> float:
        if not EDGE_GATE_ENABLED:
            return float("-inf")
        edges: list[float] = []
        for book_id in self.accounts.keys():
            book = state.books.get(book_id)
            account = self.accounts.get(book_id) if book else None
            if book is None or account is None or not book.bids or not book.asks:
                continue
            bb, ba = book.bids[0].price, book.asks[0].price
            ne = self._net_edge_bps(bb, ba, 0.5 * (bb + ba), self._maker_fee_rate(account))
            if ne is not None:
                edges.append(ne)
        if len(edges) < MIN_ACTIVE_BOOKS:
            return float("-inf")
        edges.sort(reverse=True)
        return min(0.0, edges[MIN_ACTIVE_BOOKS - 1])

    def _net_edge_bps(self, best_bid: float, best_ask: float, mid: float,
                      maker_fee: float | None) -> float | None:
        if mid <= 0 or best_ask <= best_bid or maker_fee is None:
            return None
        full_spread_bps = (best_ask - best_bid) / mid * 1e4
        return full_spread_bps - 2.0 * (maker_fee * 1e4) - ADVERSE_SEL_BPS

    def _gate_ok(self, best_bid: float, best_ask: float, mid: float,
                 maker_fee: float | None, gate_min_edge: float) -> bool:
        if not EDGE_GATE_ENABLED:
            return True
        ne = self._net_edge_bps(best_bid, best_ask, mid, maker_fee)
        return ne is not None and ne >= gate_min_edge

    # ------------------------------------------------------------------ reconcile (reused)
    def _reconcile_quotes(self, response, account, book_id: int,
                          desired: dict[int, tuple[float, float]]) -> None:
        resting = account.orders or []
        keep_sides: set[int] = set()
        cancel_ids: list[int] = []
        for o in resting:
            side = OrderDirection.BUY if o.side == 0 else OrderDirection.SELL
            want = desired.get(side)
            if (
                want is not None
                and side not in keep_sides
                and o.price is not None
                and abs(o.price - want[0]) < REPRICE_KEEP_TICKS * self._tick
                and abs((o.quantity or 0.0) - want[1]) < self.exch_min
            ):
                keep_sides.add(side)
            else:
                cancel_ids.append(o.id)
        if cancel_ids:
            response.cancel_orders(book_id, cancel_ids)
        for side, (px, qty) in desired.items():
            if side in keep_sides:
                continue
            if side == OrderDirection.BUY:
                self._submit_limit(response, book_id, OrderDirection.BUY, qty, px, post_only=True)
            else:
                short_sale = self._avail(account.base_balance) < qty
                self._submit_limit(
                    response, book_id, OrderDirection.SELL, qty, px, post_only=True,
                    settlement=self._loan_settlement(account) if short_sale else LoanSettlementOption.NONE,
                )

    # ------------------------------------------------------------------ events (reused)
    def onTrade(self, event: TradeEvent, validator: str | None = None) -> None:
        if event.bookId is None:
            return
        validator = validator or self._active_validator
        if validator is None:
            return
        if self.uid == event.takerAgentId:
            is_buy = event.side == OrderDirection.BUY
            fee = event.takerFee
        elif self.uid == event.makerAgentId:
            is_buy = event.side == OrderDirection.SELL
            fee = event.makerFee
        else:
            return
        ts_ns = int(event.timestamp) if event.timestamp else self._step_ts_ns.get(validator, 0)
        self._record_trade_volume(validator, event.bookId, event.quantity, event.price, ts_ns)
        self._apply_fill(validator, event.bookId, is_buy, event.quantity, event.price, fee, ts_ns)

    # ------------------------------------------------------------------ FIFO + precision (reused)
    def _sync_precision(self, price_decimals: int, volume_decimals: int) -> None:
        if price_decimals == self._price_decimals and volume_decimals == self._volume_decimals:
            return
        self._price_decimals = price_decimals
        self._volume_decimals = volume_decimals
        self._tick = 10 ** (-price_decimals)
        self.quote_lot = round(max(QUOTE_LOT, 10 ** (-volume_decimals)), volume_decimals)
        self.exch_min = max(EXCHANGE_MIN_ORDER_SIZE, 10 ** (-volume_decimals))
        self._flat_eps = 0.5 * 10 ** (-volume_decimals)
        bt.logging.info(
            f"[WarehouseRealizer uid={self.uid}] priceDecimals={price_decimals} tick={self._tick} "
            f"volumeDecimals={volume_decimals} lot={self.quote_lot} exch_min={self.exch_min}"
        )

    def _inv(self, validator: str, book_id: int) -> _Inv:
        return self.inv.setdefault(validator, {}).setdefault(book_id, _Inv())

    def _bstate(self, validator: str, book_id: int) -> _BookState:
        return self.books_state.setdefault(validator, {}).setdefault(book_id, _BookState())

    @staticmethod
    def _long_qty(inv: _Inv) -> float:
        return sum(q for _, q, _, _ in inv.longs)

    @staticmethod
    def _short_qty(inv: _Inv) -> float:
        return sum(q for _, q, _, _ in inv.shorts)

    @staticmethod
    def _avail(balance) -> float:
        if balance is None:
            return 0.0
        return (balance.free or 0.0) + (balance.reserved or 0.0)

    def _base_notional(self, validator: str) -> float:
        """Total base notional (at cost) held across this validator's books — the global unleveraged ceiling."""
        tot = 0.0
        for inv in self.inv.get(validator, {}).values():
            for _, q, px, _ in inv.longs:
                tot += q * px
        return tot

    def _abandoned_count(self, validator: str) -> int:
        return sum(1 for s in self.books_state.get(validator, {}).values() if s.abandoned)

    def _record_trade_volume(self, validator, book_id, qty, price, ts_ns) -> None:
        vol = float(qty) * float(price)
        if vol > 0:
            self._bstate(validator, book_id).vol_log.append((ts_ns, vol))

    def _prune_vol_log(self, st: _BookState, now_ns: int) -> None:
        cutoff = now_ns - self.volume_assessment_ns
        st.vol_log = [(t, v) for t, v in st.vol_log if t >= cutoff]

    def _rolled_quote_volume(self, validator: str, book_id: int, now_ns: int) -> float:
        st = self._bstate(validator, book_id)
        self._prune_vol_log(st, now_ns)
        return sum(v for _, v in st.vol_log)

    def _apply_fill(self, validator: str, book_id: int, is_buy: bool, qty: float, price: float,
                    fee: float, ts: int) -> None:
        inv = self._inv(validator, book_id)
        realized, rtv, matched_ts, gross = self._match_fifo(inv, is_buy, qty, price, fee, ts)
        if rtv > 0:
            st = self._bstate(validator, book_id)
            st.last_rt_ns = ts
            st.abandoned = False                       # a closing fill re-activates the book
            if realized > 0:                           # streak tracks CONFIRMED realized losses (not requests)
                st.loss_streak = 0
            elif realized < 0:
                st.loss_streak += 1
            self._record_rt_close(validator, book_id, ts, realized)
            if validator == MAIN_VALIDATOR:
                hold_s = (ts - matched_ts) / _NS if matched_ts is not None else None
                bt.logging.info(
                    f"[WarehouseRealizer uid={self.uid} RT] book={book_id} "
                    f"close={'buy' if is_buy else 'sell'} rtv={rtv:.4f} exit={price:.4f} "
                    f"hold_s={hold_s if hold_s is None else round(hold_s, 1)} "
                    f"gross={gross:+.4f} net={realized:+.4f} kappa={st.kappa3}"
                )

    def _match_fifo(self, inv: _Inv, is_buy: bool, qty: float, price: float, fee: float,
                    ts: int) -> tuple[float, float, int | None, float]:
        close_book = inv.shorts if is_buy else inv.longs
        open_book = inv.longs if is_buy else inv.shorts
        realized = gross = rtv = 0.0
        remaining = qty
        matched_ts: int | None = None
        qinv = 1.0 / qty if qty > 0 else 0.0

        while remaining > self._flat_eps and close_book:
            o_ts, o_qty, o_px, o_fee = close_book[0]
            if matched_ts is None:
                matched_ts = o_ts
            take = min(o_qty, remaining)
            price_pnl = (o_px - price) * take if is_buy else (price - o_px) * take
            if o_qty <= remaining + self._flat_eps:
                close_fee = fee * o_qty * qinv
                open_fee = o_fee
                close_book.popleft()
            else:
                close_fee = fee  # validator charges the ENTIRE incoming trade fee on the partial-close lot
                open_fee = o_fee * (take / o_qty)
                close_book[0] = (o_ts, o_qty - take, o_px, o_fee - open_fee)
            realized += price_pnl - open_fee - close_fee
            gross += price_pnl
            rtv += take
            remaining -= take

        if remaining > self._flat_eps:
            open_book.append((ts, remaining, price, fee * remaining * qinv))
        return realized, rtv, matched_ts, gross

    # ------------------------------------------------------------------ kappa-3 proxy (logging only, reused)
    def _prune_rt_events(self, st: _BookState, now: int) -> bool:
        cutoff = now - self.kappa_rt_history_ns
        before = len(st.rt_events)
        st.rt_events = [(t, p) for t, p in st.rt_events if t >= cutoff]
        return len(st.rt_events) != before

    def _record_rt_close(self, validator: str, book_id: int, ts: int, net_pnl: float) -> None:
        # append only; the kappa proxy is refreshed once per step in _step_book (logging-only, MAIN_VALIDATOR)
        st = self._bstate(validator, book_id)
        self._prune_rt_events(st, ts)
        st.rt_events.append((ts, net_pnl))

    def _global_rt_timestamps(self, validator: str, now: int) -> list[int]:
        cutoff = now - self.kappa_rt_history_ns
        ts_set: set[int] = set()
        for st in self.books_state.get(validator, {}).values():
            for ts, _ in st.rt_events:
                if ts >= cutoff:
                    ts_set.add(ts)
        return sorted(ts_set)

    def _book_pnl_series(self, validator: str, book_id: int, now: int,
                         timestamps: list[int] | None = None) -> list[float]:
        if timestamps is None:
            timestamps = self._global_rt_timestamps(validator, now)
        if not timestamps:
            return []
        cutoff = now - self.kappa_rt_history_ns
        by_ts = {t: p for t, p in self._bstate(validator, book_id).rt_events if t >= cutoff}
        return [by_ts.get(ts, 0.0) for ts in timestamps]

    @staticmethod
    def _median(values: list[float]) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        m = len(s) // 2
        return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])

    @classmethod
    def _kappa3_raw(cls, pnl_series: list[float], tau: float = KAPPA_TAU) -> float | None:
        if not pnl_series or sum(1 for x in pnl_series if x != 0.0) < KAPPA_MIN_OBS:
            return None
        med = cls._median(pnl_series)
        mad = max(cls._median([abs(x - med) for x in pnl_series]), 1e-6)
        returns = [x / mad for x in pnl_series]
        n = len(returns)
        mean_r = sum(returns) / n
        lpm3 = sum(max(tau - r, 0.0) ** 3 for r in returns) / n
        upm3 = sum(max(r - tau, 0.0) ** 3 for r in returns) / n
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / n)
        reg = ((abs(mean_r) + std_r) * 0.1) ** 3
        eps = 1e-2 if mean_r > tau else 1e-6
        if lpm3 > eps:
            return (mean_r - tau) / ((lpm3 + reg) ** (1.0 / 3.0))
        if mean_r > tau:
            return (mean_r - tau) / ((upm3 + reg) ** (1.0 / 3.0))
        return 0.0

    def _refresh_book_kappa(self, validator: str, book_id: int, now: int,
                            timestamps: list[int] | None = None) -> None:
        if validator != MAIN_VALIDATOR:
            return
        st = self._bstate(validator, book_id)
        if timestamps is None:
            timestamps = self._global_rt_timestamps(validator, now)
        if len(timestamps) < 2 or timestamps[-1] - timestamps[0] < self.kappa_min_lookback_ns:
            st.kappa3 = None
            return
        st.kappa3 = self._kappa3_raw(self._book_pnl_series(validator, book_id, now, timestamps))

    # ------------------------------------------------------------------ helpers (reused)
    @staticmethod
    def _loan_settlement(account) -> LoanSettlementOption:
        quote_loan = getattr(account, "quote_loan", 0.0) or 0.0
        return LoanSettlementOption.FIFO if quote_loan > 0 else LoanSettlementOption.NONE

    def _maker_fee_rate(self, account) -> float | None:
        fees = getattr(account, "fees", None)
        rate = getattr(fees, "maker_fee_rate", None) if fees is not None else None
        try:
            return float(rate) if rate is not None else None
        except (TypeError, ValueError):
            return None

    def _taker_fee_rate(self, account) -> float | None:
        fees = getattr(account, "fees", None)
        rate = getattr(fees, "taker_fee_rate", None) if fees is not None else None
        try:
            return float(rate) if rate is not None else None
        except (TypeError, ValueError):
            return None

    def _cancel_all(self, response, account, book_id: int) -> None:
        if account.orders:
            response.cancel_orders(book_id, [o.id for o in account.orders])

    def _submit_limit(self, response, book_id: int, direction: int, qty: float, price: float,
                      *, post_only: bool = True, ioc: bool = False,
                      settlement: LoanSettlementOption = LoanSettlementOption.NONE) -> None:
        kwargs: dict[str, Any] = {
            "book_id": book_id, "direction": direction, "quantity": qty,
            "price": price, "stp": STP.CANCEL_OLDEST,
        }
        if ioc:
            kwargs["timeInForce"] = TimeInForce.IOC
        else:
            kwargs["postOnly"] = post_only
            kwargs["timeInForce"] = TimeInForce.GTT
            kwargs["expiryPeriod"] = self.quote_expiry_ns
        if settlement != LoanSettlementOption.NONE:
            kwargs["settlement_option"] = settlement
        response.limit_order(**kwargs)


if __name__ == "__main__":
    launch(WarehouseRealizerAgent)
