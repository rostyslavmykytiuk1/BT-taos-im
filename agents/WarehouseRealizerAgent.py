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

THE BINDING CONSTRAINT is BREADTH: keep >=~80 books each closing >=3 realized RTs per 3h (the kappa gate).
  The activity factor is ~NEGLIGIBLE (validator decay ~0.01%/tick + a unit bug), so quiet losers are HELD
  indefinitely, NOT force-traded for cadence. The genuine failure mode is a sustained correlated downtrend
  where bids never clear the margin; the loss ladder below bounds it (never hold forever) and we stop
  accumulating as abandons rise.

PER-BOOK STEP (each respond):
  mid-EWMA/kappa upkeep -> (B0) 60bps catastrophe stop (past the revert window) -> (B) breadth-protection
  backstop (bounded <=15bps, only when active breadth is at risk) -> (B2) 3300s hard max-hold (force-realize
  the oldest lot; never-hold-forever) -> (C) cadence taker-out (density) -> (D1/D2) resting two-sided MAKER
  reduce-ask + dip-biased accumulate bid. B0/B/B2/C are terminal IOC actions (cancel_all + IOC + return); D
  is the steady two-sided resting state.

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
REALIZE_MARGIN_BPS = 1.5          # required NET margin (bps of notional) AFTER both fee legs, measured at the
                                   # WORST-CASE taker-out fill (best_bid less the IOC slippage). So this is the
                                   # GUARANTEED floor on every cadence RT even in a thin book -> every realize
                                   # is strictly positive (counts toward the >=3 nonzero gate, zero downside).
                                   # Kept low to MAXIMIZE clean-positive DENSITY (the lever to beat uid56);
                                   # expected fill ~best_bid nets ~MARGIN+slip. TUNABLE.
REALIZE_TARGET_S = 120.0          # cadence backstop: if no RT closed in this long and a net-positive
                                   # taker-out exists, cross to keep RT DENSITY up (resting ask not filling).
REALIZE_MAX_LOTS = 2.0            # cap a single realize at this many lots -> fragments a big lot (e.g. a
                                   # restart ghost-seed lot) into small realized entries (low kappa variance).
TAKER_SIDE_MARGIN_BPS = 1.0      # hysteresis cushion for the PER-LEG taker-vs-maker execution choice
                                 # (taker cheaper than maker on a leg iff spread < maker_fee - taker_fee).
BACKSTOP_DEADLINE_S = 2700.0      # The activity factor is ~NEGLIGIBLE (validator decay ~0.01%/tick + a unit
                                   # bug kills the acceleration), so we do NOT force losses for cadence. This
                                   # is the latest a quiet book may ride before we *consider* a forced RT —
                                   # set to the ~3-obs/3h kappa-gate cadence (KAPPA_RT_HISTORY_S/4=10800/4) so
                                   # even a never-winning book keeps >=3 RTs/3h IF we must trade it. Losers are
                                   # otherwise HELD indefinitely (pure warehouse, like uid56).
BREADTH_RISK_MARGIN = 8           # only force a bounded loss-RT (to defend the >=80 active-book floor) when
                                   # the live count of gate-clearing books < MIN_ACTIVE_BOOKS + this margin.

# ---- stuck-lot handling (the downtrend survival mechanism) ----
STUCK_HOLD_S = 180.0              # hold-for-revert window; past it an underwater oldest lot may be released
MAX_STUCK_LOSS_BPS = 15.0        # AGE-triggered bounded-loss release cap to clear a FIFO lockup (kappa-safe)
MK_STOP_BPS = 60.0               # BIG catastrophe stop-loss (the OTHER half of never-hold-forever): once a lot
                                  # has cleared the sharp-dump-revert window (STUCK_HOLD_S) AND is still this far
                                  # underwater at the MARKET, it's a genuine sustained adverse move (normal dip
                                  # noise is <=~30bps), so force-realize -> the tail loss is bounded ~60bps, not
                                  # the ~600bps a fast trender would reach by the time the max-hold cuts it.
MK_MAX_HOLD_S = 3300.0           # HARD per-book max-hold (owner's PERMANENT never-hold-forever rule): past
                                  # this the FIFO-oldest lot is force-realized REGARDLESS of breadth or loss
                                  # magnitude. Set < lookback/3 (10800/3=3600) so a one-way-drift book still
                                  # turns >=3 RTs/3h (breadth can't silently collapse), and bounds every hold
                                  # (no position forever). Breaks the sustained-drift deadlock where a
                                  # >15bps-underwater book had NO realize path. One/book/~MAX_HOLD, jitter-
                                  # spread -> no cluster (a lone larger loss is cbrt+MAD damped).
CUT_COOLDOWN_S = 600.0           # after a LOSS realize on a book, rate-limit: no further loss-cut (B0 stop /
                                  # B- backstop) AND no re-accumulation on that book for this long. Stops the
                                  # per-STEP flood where a deeply-underwater lot (e.g. an inherited ghost, or a
                                  # downtrend re-accum churn) gets chunk-stoplossed every tick -> kappa-tanking
                                  # loss cluster + over-trade. Holds the loser instead (warehouse thesis), which
                                  # ALSO gives the market time to revert (-> sold as a WINNER). The hard max-hold
                                  # (B2) is NOT cooldown-gated, so never-hold-forever still holds.
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
    mid_ewma: float = 0.0
    abandoned: bool = False               # deeply-stuck book: stop quoting, let it idle (free budget)
    loss_streak: int = 0                  # consecutive loss-realizes without a positive RT (per-book guard)
    last_cut_ns: int = 0                  # last LOSS realize on this book; rate-limits loss cuts + re-accum


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
        backstop_s = BACKSTOP_DEADLINE_S * (0.92 + 0.08 * jitter)
        stuck_hold_s = STUCK_HOLD_S * (0.9 + 0.2 * jitter)
        max_hold_s = MK_MAX_HOLD_S * (0.92 + 0.08 * jitter)
        self.cut_cooldown_ns = int(CUT_COOLDOWN_S * (0.9 + 0.2 * jitter) * _NS)

        self.quote_expiry_ns = int(QUOTE_EXPIRY_S * _NS)
        self.realize_target_ns = int(realize_target_s * _NS)
        self.backstop_deadline_ns = int(backstop_s * _NS)
        self.stuck_hold_ns = int(stuck_hold_s * _NS)
        self.max_hold_ns = int(max_hold_s * _NS)
        self.kappa_rt_history_ns = int(KAPPA_RT_HISTORY_S * _NS)
        self.kappa_min_lookback_ns = int(KAPPA_MIN_LOOKBACK_S * _NS)

        self.inv: dict[str, dict[int, _Inv]] = {}
        self.books_state: dict[str, dict[int, _BookState]] = {}
        self._sim_id: dict[str, str] = {}
        self._step_ts_ns: dict[str, int] = {}
        self._active_validator: str | None = None
        self._cap_remaining = float("inf")  # per-step base-cap budget, recomputed every respond() before use

        bt.logging.info(
            f"[WarehouseRealizer uid={self.uid}] WHR lot={QUOTE_LOT} accum_target={ACCUM_TARGET_LOTS}lots "
            f"global_base={GLOBAL_BASE_FRAC:.0%} dip={self.accum_dip_bps:.1f}bps "
            f"realize_margin={self.realize_margin_bps:.1f}bps(net) realize_target={realize_target_s:.0f}s "
            f"backstop={backstop_s:.0f}s(breadth-gated) stuck_hold={stuck_hold_s:.0f}s "
            f"stuck_cut<={MAX_STUCK_LOSS_BPS:.0f}bps maxhold={max_hold_s:.0f}s stop={MK_STOP_BPS:.0f}bps "
            f"cut_cooldown={self.cut_cooldown_ns/_NS:.0f}s "
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

        volume_cap = CAPITAL_TURNOVER_CAP * cfg.miner_wealth * VOLUME_SAFETY
        now = state.timestamp

        # fleet-level pre-pass: accumulate-breadth gate floor + the global unleveraged base ceiling.
        gate_min_edge = self._compute_gate_min_edge(state)
        base_out = self._base_notional(validator)
        can_open_globally = base_out < GLOBAL_BASE_FRAC * cfg.miner_wealth
        # running per-step base-cap budget: the DETERMINISTIC taker-buys could otherwise overshoot the global
        # base cap within one correlated-dip step (the once-per-step can_open snapshot would be breached).
        self._cap_remaining = GLOBAL_BASE_FRAC * cfg.miner_wealth - base_out
        abandoned_n = self._abandoned_count(validator)
        # Breadth guard: count books currently clearing the >=3-obs kappa gate. We HOLD losers indefinitely
        # (pure warehouse) UNLESS this active count is near the 80-floor — only then do we allow bounded
        # loss-RTs to defend the cliff. (The activity factor is ~negligible, so cadence never forces a loss.)
        active_count = sum(1 for s in self.books_state.get(validator, {}).values()
                           if sum(1 for _, p in s.rt_events if p != 0.0) >= KAPPA_MIN_OBS)
        breadth_at_risk = active_count < (MIN_ACTIVE_BOOKS + BREADTH_RISK_MARGIN)
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
                self._step_book(response, validator, book_id, book, account, volume_cap, now,
                                gate_min_edge, can_open_globally, abandoned_n, main_v, rt_ts, breadth_at_risk)
            except Exception as ex:
                bt.logging.warning(f"[WarehouseRealizer uid={self.uid}] step {book_id}: {ex}")

        return response

    # ------------------------------------------------------------------ per-book
    def _step_book(
        self, response, validator: str, book_id: int, book, account,
        volume_cap: float, now: int, gate_min_edge: float, can_open_globally: bool, abandoned_n: int,
        main_v: bool, rt_ts: list[int] | None, breadth_at_risk: bool,
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

        # per-book mid-EWMA upkeep (the dip reference for accumulation)
        st.mid_ewma = mid if st.mid_ewma <= 0.0 else (1.0 - MID_EWMA_ALPHA) * st.mid_ewma + MID_EWMA_ALPHA * mid
        self._prune_rt_events(st, now)           # bound rt_events memory on every validator
        self._prune_vol_log(st, now)             # bound vol_log memory (the volume-cap input)
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

        # Execution side is chosen PER LEG below (see taker_side_good): taker in tight spreads (cheaper basis +
        # certain fill), maker in wide spreads (spread capture). No separate "harvest mode".
        # ============================ EXIT LADDER (terminal IOC actions) ============================
        if long_qty >= self.exch_min and inv.longs:
            o_ts, o_qty, o_px, o_fee = inv.longs[0]
            age = now - o_ts
            # Bundle a sub-exch_min FIFO-head stub with the next lot so it can't freeze the book; cap at
            # REALIZE_MAX_LOTS so a big (e.g. ghost) lot fragments. (validator FIFO closes oldest-first.)
            sell_qty = round(min(long_qty, base_avail,
                                 max(o_qty, self.exch_min), REALIZE_MAX_LOTS * self.quote_lot), vdp)
            # CRITICAL: gate every realize on the AGGREGATE net across the WHOLE sell_qty span (NOT the oldest
            # lot alone) — else a profitable head stub spilling into a higher-basis underwater lot would leak an
            # UNGUARDED negative realize via the cadence/activity-positive paths (the negative-skew kappa-tank
            # this agent exists to prevent). span_net mirrors _match_fifo across every lot the sell would touch.
            # Gate every taker-out realize on the price the IOC ACTUALLY crosses at (best_bid less the IOC
            # slippage concession), NOT the raw best_bid snapshot — else a thin top-of-book lets the fill slip
            # below the snapshot so a "profit" (B+) or a <=15bps bounded-loss (B-/cadence) gate leaks an
            # UNGUARDED or over-cap negative realize. close_rate biases to a + cost when the taker fee is
            # unknown (None -> 0 would overstate net and leak a marginal loss through the winners-only gates).
            close_rate = taker_fee if taker_fee is not None else 0.0004
            # Round to the SAME tick _taker_out submits at, so the GATE price == the executed IOC fill floor:
            # a sub-tick down-round then can't push a span_net>=0 "backstop+" realize negative.
            sell_px_taker = self._taker_out_px(best_bid)
            span_net = self._span_net_bps(inv, sell_qty, sell_px_taker, close_rate)   # REALIZABLE net (B/C)
            loss_bps = max(0.0, -span_net)
            span_net_mkt = self._span_net_bps(inv, sell_qty, best_bid, close_rate)    # true MARKET depth

            # Deep-stuck: aged AND deeply underwater at the MARKET -> STOP accumulating (don't average down
            # into a falling bag). The lot is HELD (unleveraged -> never liquidatable); idles in the free budget.
            st.abandoned = (age >= self.stuck_hold_ns and max(0.0, -span_net_mkt) > MAX_STUCK_LOSS_BPS)

            # (B0) BIG CATASTROPHE STOP-LOSS — the magnitude half of never-hold-forever. Past the sharp-dump
            #      revert window (stuck_hold), a lot still > MK_STOP_BPS underwater is a genuine sustained
            #      adverse move (not normal dip noise) -> force-realize REGARDLESS of breadth so the tail loss
            #      is bounded ~60bps instead of running to the max-hold's ~600bps. (B0/B2 together = the rule.)
            if (age >= self.stuck_hold_ns and -span_net_mkt >= MK_STOP_BPS and sell_qty >= self.exch_min
                    and (now - st.last_cut_ns) >= self.cut_cooldown_ns):
                self._taker_out(response, account, book_id, sell_qty, best_bid, st, now, "stoploss", loss_bps)
                return

            # WAREHOUSE THESIS: NO early loss-cut on NORMAL (<60bps) dips — a small underwater lot is HELD to
            # wait for the bounce (-> sold as a WINNER by C); kappa needs only ~3 RTs/3h so we have huge slack.
            # Loss realizers are the breadth backstop (B), the hard max-hold (B2), and the catastrophe stop (B0
            # above). This patience is the whole point of beating the warehouse competitor (uid56).

            # (B) BREADTH-PROTECTION BACKSTOP — the activity factor is ~negligible, so we do NOT force losses
            #     for cadence; losers are HELD indefinitely (pure warehouse, wait for the bounce). The ONLY
            #     time we cut a loser is to defend the >=80 active-book floor: if breadth is AT RISK and this
            #     book has gone quiet past the 3-obs-gate cadence, realize ONE RT to keep it scored — a profit
            #     if available, else a small bounded (<=15bps, median-guarded) loss. Else the loser just rides.
            if (breadth_at_risk and (now - activity_ref) >= self.backstop_deadline_ns
                    and sell_qty >= self.exch_min):
                if span_net >= 0.0:
                    self._taker_out(response, account, book_id, sell_qty, best_bid, st, now, "backstop+", 0.0)
                    return
                if (0.0 < loss_bps <= MAX_STUCK_LOSS_BPS and self._allow_loss_realize(st, now)
                        and (now - st.last_cut_ns) >= self.cut_cooldown_ns):
                    self._taker_out(response, account, book_id, sell_qty, best_bid, st, now, "backstop-", loss_bps)
                    return

            # (B2) HARD MAX-HOLD — the owner's PERMANENT never-hold-forever rule [feedback_never-hold-forever]:
            #      once the FIFO-oldest lot has been held past MK_MAX_HOLD_S, force-realize it REGARDLESS of
            #      breadth or loss magnitude. This is the ONLY realizer that fires for a >15bps-underwater book
            #      (B caps at 15bps, C/D1 need a profit), so it breaks the sustained-drift deadlock, bounds
            #      every hold, and (MAX_HOLD < lookback/3) guarantees >=3 RTs/3h so breadth can't silently
            #      collapse in a one-way market. A lone larger loss here is cbrt+MAD damped; jitter de-syncs it.
            if age >= self.max_hold_ns and sell_qty >= self.exch_min:
                self._taker_out(response, account, book_id, sell_qty, best_bid, st, now, "maxhold", loss_bps)
                return

            # (C) CADENCE TAKER-OUT — keep RT density up when the resting ask isn't getting lifted.
            if (span_net >= self.realize_margin_bps and sell_qty >= self.exch_min
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

        # PER-LEG execution choice (the unification that replaces "harvest mode"): taker is cheaper than maker
        # on a leg exactly when the spread is tighter than the fee gap, because
        #   taker_basis - maker_basis = spread - (maker_fee - taker_fee).
        # So taker is the better execution iff spread < (maker_fee - taker_fee). Applied to BUY (D2) and
        # SELL (D1 vs the taker-out) identically. The market can move while we hold; this is re-evaluated
        # every step from the live per-book fees, regardless of how a lot was originally bought.
        spread_bps = spread / mid * 1e4
        taker_side_good = ((maker_bps - taker_bps) - spread_bps) > TAKER_SIDE_MARGIN_BPS

        desired: dict[int, tuple[float, float]] = {}

        # (D1) SELL via the MAKER leg — in WIDE spreads, rest a passive reduce ask priced so a fill nets >=
        #      margin (captures the spread; the winners' 100%-maker income channel). In TIGHT spreads
        #      (taker_side_good) we SKIP the resting ask and sell via the exit-ladder TAKER-OUT instead
        #      (cheaper + certain), so the sell leg picks its side symmetrically with the buy leg.
        if (not taker_side_good) and long_qty >= self.exch_min and inv.longs:
            o_ts, o_qty, o_px, o_fee = inv.longs[0]
            q = round(min(long_qty, base_avail,
                          max(o_qty, self.exch_min), REALIZE_MAX_LOTS * self.quote_lot), vdp)
            # Price the ask so EVERY lot in the q-span nets >= margin when lifted (NOT just the oldest), so a
            # multi-lot fill can't realize a loss on a higher-basis lot behind a stub (same span-leak guard).
            min_ask = self._span_min_sell_px(inv, q, maker_bps, self.realize_margin_bps)
            sell_px = round(max(ask_inside, min_ask), pdp)
            if q >= self.exch_min and sell_px > 0:
                desired[OrderDirection.SELL] = (sell_px, q)

        # (D2) ACCUMULATE / REPLENISH — passive dip-biased maker buy, up to the per-book target & global cap.
        # Suppressed during the post-cut cooldown: don't re-buy a book we just loss-cut (breaks the downtrend
        # buy->stoploss->rebuy churn; also lets a just-cut inherited ghost settle instead of re-accumulating).
        if not st.abandoned and (now - st.last_cut_ns) >= self.cut_cooldown_ns:
            target_qty = ACCUM_TARGET_LOTS * self.quote_lot
            want_more = long_qty < target_qty - self._flat_eps
            seed = long_qty < ACCUM_MIN_LOTS * self.quote_lot - self._flat_eps
            dip = mid <= st.mid_ewma * (1.0 - self.accum_dip_bps / 1e4)
            breadth_ok = (long_qty >= self.exch_min) or (abandoned_n < MAX_ABANDON_BOOKS)
            if (want_more and can_open_globally and breadth_ok and (seed or dip)
                    and self._gate_ok(best_bid, best_ask, mid, maker_fee, gate_min_edge)):
                q = round(self.quote_lot, vdp)
                free_quote = account.quote_balance.free if account.quote_balance else 0.0
                # Don't use the CERTAIN taker-fill to average DOWN into a falling book — there, fall back to
                # the patient maker bid whose uncertain fill naturally THROTTLES accumulation into a fall (the
                # brake the old probabilistic-bid design relied on). taker-buy only when flat or in-profit.
                under_water = bool(inv.longs) and best_bid < inv.longs[0][2]
                if (taker_side_good and not under_water and q >= self.exch_min
                        and free_quote >= q * best_ask
                        and q * best_ask <= self._cap_remaining
                        and self._rolled_quote_volume(validator, book_id, now) < volume_cap):
                    # TIGHT spread, not averaging into a loss -> BUY via taker (cheaper basis + certain fill).
                    self._submit_limit(response, book_id, OrderDirection.BUY, q, round(best_ask, pdp),
                                       ioc=True, post_only=False)
                    self._cap_remaining -= q * best_ask       # reserve within this step's base budget
                elif (q >= self.exch_min and free_quote >= q * bid_inside
                        and self._rolled_quote_volume(validator, book_id, now) < volume_cap):
                    # WIDE spread OR tight-but-underwater -> patient MAKER bid (uncertain fill = throttle).
                    # NOT reserved against _cap_remaining: the maker bid's fill is UNCERTAIN, and reserving it
                    # against the shared per-step budget starved high-book-id books near the cap (the ascending
                    # book loop let low-ids eat the budget first -> breadth skew + per-step cancel churn). The
                    # 0.60 ceiling is held by the can_open_globally snapshot + the next-step base_out recompute;
                    # only the DETERMINISTIC same-step taker-buy leg reserves. Volume-gated like the taker leg.
                    desired[OrderDirection.BUY] = (bid_inside, q)

        self._reconcile_quotes(response, account, book_id, desired)

    # ------------------------------------------------------------------ taker-out helper (terminal)
    def _taker_out_px(self, best_bid: float) -> float:
        """The exact price a taker-out IOC crosses at (best_bid less the slippage concession, rounded to the
        tick). SINGLE source of truth: the realize GATE (span_net at this px) and the submitted order MUST
        use the identical value for FIFO-parity — the gated net then equals the worst-case executed net."""
        return round(best_bid * (1.0 - TAKER_OUT_SLIPPAGE_BPS / 1e4), self._price_decimals)

    def _taker_out(self, response, account, book_id: int, qty: float, bid_px: float,
                   st: _BookState, now: int, reason: str, loss_bps: float) -> None:
        """Cross to the bid (IOC) to close one oldest FIFO lot. Cancel resting orders first so the IOC and a
        stale resting ask can't both fire. onTrade records the realized RT (updates last_rt_ns / kappa)."""
        px = self._taker_out_px(bid_px)
        if loss_bps > 0.0:
            st.last_cut_ns = now       # start the per-book cooldown (rate-limits loss cuts + re-accumulation)
        self._cancel_all(response, account, book_id)
        self._submit_limit(response, book_id, OrderDirection.SELL, qty, px, ioc=True, post_only=False)
        bt.logging.info(
            f"[WarehouseRealizer uid={self.uid}] TAKER-OUT book={book_id} qty={qty} @{px} "
            f"reason={reason} loss_bps={loss_bps:.1f} streak={st.loss_streak}"
        )

    def _span_net_bps(self, inv: _Inv, qty: float, sell_px: float, close_rate: float | None) -> float:
        """Aggregate NET realized (bps of filled notional) for selling `qty` at `sell_px` across the FIFO lots
        it would span, net of each lot's open fee and the close fee at `close_rate` (signed: + cost, - rebate).
        Mirrors _match_fifo's accounting EXACTLY (same full/partial close-fee branches) WITHOUT mutating, so
        the gate equals the SCORED realized. Gating realizes on this (not the oldest lot alone) stops a
        profitable head stub from leaking a loss on a higher-basis lot behind it."""
        if qty <= self._flat_eps or sell_px <= 0:
            return 0.0
        fee = qty * sell_px * (close_rate or 0.0)      # the trade's total close fee (mirrors the event fee)
        qinv = 1.0 / qty
        remaining = qty
        gross = open_fees = close_fees = filled = 0.0
        for o_ts, o_qty, o_px, o_fee in inv.longs:
            if remaining <= self._flat_eps:
                break
            take = min(o_qty, remaining)
            gross += (sell_px - o_px) * take
            if o_qty <= remaining + self._flat_eps:    # FULL close — mirror _match_fifo exactly
                close_fees += fee * o_qty * qinv
                open_fees += o_fee
            else:                                       # PARTIAL close
                close_fees += fee
                open_fees += o_fee * (take / o_qty)
            filled += take
            remaining -= take
        if filled <= self._flat_eps:
            return 0.0
        return (gross - open_fees - close_fees) / (filled * sell_px) * 1e4

    def _span_min_sell_px(self, inv: _Inv, qty: float, close_bps: float, margin_bps: float) -> float:
        """Lowest ask at which selling `qty` nets >= margin on EVERY spanned FIFO lot (so a multi-lot fill can't
        realize a loss on a higher-basis lot) = max over spanned lots of lot_px*(1+(open_fee+close+margin))."""
        remaining = qty
        px = 0.0
        for o_ts, o_qty, o_px, o_fee in inv.longs:
            if remaining <= self._flat_eps:
                break
            ofb = (o_fee / (o_qty * o_px) * 1e4) if (o_qty > 0 and o_px > 0) else close_bps
            px = max(px, o_px * (1.0 + (ofb + close_bps + margin_bps) / 1e4))
            remaining -= min(o_qty, remaining)
        return px

    def _allow_loss_realize(self, st: _BookState, now: int) -> bool:
        """Cluster guard for the bounded-loss backstop — the real defense against the -0.0156 "many losses on
        one falling book" failure. Realize a forced loss ONLY if (a) the book hasn't already taken 3
        consecutive losses with no intervening winner, AND (b) its recent realized stream is still
        median-positive. loss_streak is a TRUE consecutive-loss counter (reset on a winner / incremented on a
        loss in _apply_fill) — do NOT decay it on a timer here: the old decay keyed on backstop_deadline_ns,
        the SAME predicate that arms the backstop, so the >=3 guard could never trip (dead code). A
        thin-history book (<3 obs) is NOT yet gate-clearing, so a forced loss only SEEDS a negative without
        defending breadth -> hold it (return False); it earns its first obs from WINNERS, not manufactured losses."""
        if st.loss_streak >= 3:
            return False
        pnls = [p for _, p in st.rt_events]
        if len(pnls) < KAPPA_MIN_OBS:
            return False
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
                # LONG-ONLY / ZERO-LEVERAGE: never rest a sell for more base than we own -> a reduce-ask can
                # NEVER open a short (the whole never-liquidatable warehouse thesis). Clamp to held base and
                # drop the inherited loan-settled short_sale path entirely.
                owned = self._avail(account.base_balance)
                # FLOOR (not round) to the vol tick so the clamp can't creep a half-ULP ABOVE owned and rest a
                # marginally-short reduce-ask: int() truncates toward zero -> sell_qty <= min(qty, owned) always.
                scale = 10 ** self._volume_decimals
                sell_qty = int(min(qty, owned) * scale) / scale
                if sell_qty >= self.exch_min:
                    self._submit_limit(response, book_id, OrderDirection.SELL, sell_qty, px, post_only=True)

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
        if ts_ns <= 0:
            # No usable timestamp (event before the first step, both event.timestamp and _step_ts_ns unset):
            # skip rather than stamp a lot with ts=0, which would make age = now-0 huge and instantly trip the
            # B0/B2 age cuts. The ghost-seed reconciles any resulting FIFO/exchange gap on the book's first step.
            return
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
        if qty <= 0:                                 # degenerate fill: nothing to match (silent fee=0 would
            return 0.0, 0.0, None, 0.0               # under-charge and break parity) -> no-op, loudly nothing.
        close_book = inv.shorts if is_buy else inv.longs
        open_book = inv.longs if is_buy else inv.shorts
        realized = gross = rtv = 0.0
        remaining = qty
        matched_ts: int | None = None
        qinv = 1.0 / qty

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
    def _prune_rt_events(self, st: _BookState, now: int) -> None:
        cutoff = now - self.kappa_rt_history_ns
        st.rt_events = [(t, p) for t, p in st.rt_events if t >= cutoff]

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
                         timestamps: list[int]) -> list[float]:
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
        st = self._bstate(validator, book_id)
        if timestamps is None:                       # direct callers (e.g. the dryrun) may not pre-build it
            timestamps = self._global_rt_timestamps(validator, now)
        if len(timestamps) < 2 or timestamps[-1] - timestamps[0] < self.kappa_min_lookback_ns:
            st.kappa3 = None
            return
        st.kappa3 = self._kappa3_raw(self._book_pnl_series(validator, book_id, now, timestamps))

    # ------------------------------------------------------------------ helpers (reused)
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
                      *, post_only: bool = True, ioc: bool = False) -> None:
        # long-only warehouse: no loan/short settlement -> orders are always default (NONE) settlement.
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
        response.limit_order(**kwargs)


if __name__ == "__main__":
    launch(WarehouseRealizerAgent)
