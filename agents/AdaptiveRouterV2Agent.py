"""
AdaptiveRouterV2Agent — per-book maker/taker/idle router for subnet 79 (optimized fork of
AdaptiveRouterAgent). V2 CHANGES (2026-06-26, evidence/A-B-derived; taker leg unchanged):
  (1) MAKER reduce walks only to BREAKEVEN (never the touch), holds losers for reversion, and realizes a
      loss ONLY on a trending-loser stop = MK_STOP_LOSS_BPS (V2.1: 15bps; no 90s time-cut). The breakeven
      floor is the win (fills 68% vs v1 57% live + banks deep reversions); the stop was retuned 35→15 after
      the A/B showed 35bps cuts are cube-bombs (38³ craters kappa) while 15 caps the downside ~13×.
  (2) maker fee CEILING re-enabled at 8bps (was off) with a cur==MODE_MAKER in-mode bypass — auto-gates
      maker OFF in maker-pays regimes, ON when fees are cheap; never ejects an in-position maker.
  (3) ROUTING classifies on a 60s EMA of the half-spread (de-noise chop); the RAW spread stays in the
      _open execution gate.
  (4) idle CLIFF-FIRST override — near the 48-book free-drop budget, relax the maker fallback to rest
      FREE passive quotes rather than force-cross taker (a free unfilled quote beats a guaranteed loss).
  (5) NO maker→taker demote on a recent-PnL blip (it contaminated the per-book Sortino).
  (6) dwell 180→300s, activity 480→1200s, fallback edge 0.5→1.0.

Goal: stay near the top in ANY market by routing each book to the playbook that is +EV given the
book's LIVE fee regime, mirroring the proven top miners:
  * TAKER mode  (mirrors UID 126): on deep-rebate books, cross the spread for tiny clips and
    recycle in seconds. The rebate (both legs) cushions a fast stop into a net win. Small bounded
    same-side inventory building is allowed while the rebate stays deep.
  * MAKER mode  (mirrors UID 109/149): on spread-rich books, post two-sided passive quotes and
    capture spread WIDER than the maker fee + an adverse-selection margin. When holding, work only
    the reducing side priced off the FIFO worst lot (no bagging), and cap every forced exit.
  * IDLE mode: for books where active trading would hurt more than idling. Two triggers: (1) fee
    regime — spread < maker_fee + adverse-selection cushion and no taker rebate; (2) PnL backoff —
    a book's net realized PnL over the last 10 min is negative (≥3 RTs in window), indicating live
    adverse selection or trending. Stay flat → the book gets Kappa=None and is DROPPED from the
    median (up to ~37.5% are free), which beats bleeding small losses into the median.

Calibrated from the top agents' own trade histories (other agents data/*.csv):
  * They ALL trade 128/128 books — none idle. We raise the maker entry bar above breakeven
    (MAKER_EDGE_ENTER = 1.5bps) to absorb ~0.8bps adverse-selection drag and idle the rest.
  * They route by FEE REGIME: UID 126 (#1, Kappa 0.263) is 100% taker on rebate books; UID 66
    (Kappa 0.104) is mixed and takes only where the taker side is rebated; UID 109/165 are pure
    makers — 165 makes profitably even at +11.5bps fee. We do the same: take where rebated, make
    everywhere else, no maker fee ceiling.
  * Kappa rises with FREQUENCY (149 at 1.7 RT/book/10min → Kappa 0.040; 66/109 at ~26 → 0.10),
    because per-book Kappa ≈ (share of step-timestamps closed positive)^(2/3). So we quote
    continuously with short cooldowns, NOT sparse high-target round-trips.

Why this wins (from taos/im/validator/reward.py + utils/kappa.py + _match_trade_fifo):
  * Score = 0.79·kappa + 0.21·pnl, BOTH per-book then median across books. The leaderboard tracks
    KAPPA (UID 126 is #1 with negative mark-to-market PnL). We optimize the per-book Kappa-3 median
    and keep activity = 1.0 on as many books as possible.
  * Kappa-3 CUBES the downside: one big loss dwarfs many tiny gains. So the dominant risk lever is
    POSITION SIZE — small clips + a tight inventory cap keep every forced cut small. We never bag.
  * Round-trip volume (the activity signal) is produced ONLY by a CLOSING fill, sampled every
    ~600s. Each book MUST close ≥1 round-trip per window or its activity factor decays.

Design notes addressing the known failure modes:
  * Routing has DWELL + HYSTERESIS and only switches when FLAT, so books do not flip-flop modes
    (the old DualEdge made ~2000 switches in an hour) and no position ever straddles two modes.
  * PnL SCALE MATCHING: both modes target a shared clip notional and cap a single round-trip loss
    to the same bound, so a book's realized-RT series keeps a consistent scale across mode changes
    (mixing scales would distort the per-book MAD and the cubic downside).
  * Single shared FIFO inventory per book (validator-faithful), so kappa/activity accounting is
    identical regardless of which mode opened a position.

Scalability: one class per mode (_TakerMode/_MakerMode/_IdleMode) over shared agent infrastructure
and a Router. ALLOWED_MODES gates which modes may run — set it to {"taker"} and the agent behaves
exactly like a pure taker scalper; {"maker"} for a pure maker; the full set for the router.

Per book each step:
  reconcile FIFO -> prune/refresh kappa -> if flat: PnL-backoff check (force idle if net-negative
    over 10min) -> (dwell elapsed?) re-route w/ hysteresis -> dispatch to the book's mode.step():
         risk/managed-exit (bounded) -> activity backstop (bounded) -> mode's profit engine.
"""

import gc
import math
import time
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
    OrderCurrency,
    OrderDirection,
    STP,
    TimeInForce,
)

_NS = 1_000_000_000

# ======================================================================== config
# Which modes the router may use. {"taker"} => pure taker (== TakerScalper); {"maker"} => pure
# maker; full set => adaptive per-book routing (the default).
ALLOWED_MODES = {"taker", "maker", "idle"}

# ---- shared sizing / precision ----
EXCHANGE_MIN_ORDER_SIZE = 0.25     # sim minOrderSize floor for any BASE order
TARGET_CLIP = 0.26                 # shared per-clip BASE lot. Just ABOVE the 0.25 exchange minimum
                                   # on purpose: a fee-paying BUY is settled by shaving the fee out
                                   # of the base received (ClearingManager: fees_base=roundUp(fee/px)),
                                   # so a 0.25 buy leaves ~0.2498 held — under the min, un-sellable,
                                   # so you MISS the exit until you re-accumulate. 0.26 keeps the held
                                   # lot >= 0.25 after the shave => always closeable in one order.
                                   # Still matches the top makers' clips (0.25-0.29) and stays small.

# ---- routing thresholds (bps), with hysteresis so a book does not flip-flop ----
# EVIDENCE (top-agent trade histories + live regime analysis): route by fee regime.
# * TAKER: books with taker rebate ≥ 1.5bps — the _open() gate (est_bps = 2×rebate − 2×half_spread
#   > 0) independently guards execution, so a 1.5bps-rebate book that has a wide spread won't trade
#   in taker mode even if routed there. The 1.5bps bar captures the bulk of the rebate range that
#   appeared during high-rebate regimes (formerly 3.0bps missed most books).
# * MAKER: books where passive capture net of BOTH fee legs is positive WITH an adverse-selection
#   cushion. Empirical data shows avg fee drag of ~0.8bps per RT from adverse selection; a 1.5bps
#   edge requirement (half_spread − maker_fee ≥ 1.5bps) absorbs that and leaves margin. Books at
#   the breakeven boundary (edge ~0bps) bleed in practice — idle is strictly better (kappa=None is
#   DROPPED from the median, whereas a book with negative kappa actively drags it down).
# * IDLE: any book that is neither rebate-eligible for taker nor adequately edgy for maker. The
#   validator allows up to 37.5% (48/128) idle books before kappa=None starts contributing 0.0
#   to the median; staying below that limit makes idling strictly dominant over losing trades.
TAKER_REBATE_ENTER_BPS = 1.5       # route to taker when rebate ≥ 1.5bps; execution gate in
                                   # _TakerMode._open() independently checks 2×rebate > 2×half_spread
TAKER_REBATE_EXIT_BPS = 0.75       # leave taker when rebate falls below 0.75bps (0.75bps hysteresis)
MAKER_EDGE_ENTER_BPS = 1.5         # require half_spread − maker_fee ≥ 1.5bps to enter maker;
                                   # empirically the minimum to survive adverse selection
MAKER_EDGE_EXIT_BPS = -0.5         # exit maker when edge falls below −0.5bps (2bps hysteresis band)
MAKER_MAX_FEE_BPS = 8.0            # V2: REAL ceiling (was 1000=off). A high maker fee = adverse-selection
                                   # signal → don't ENTER maker above it (self-gates maker OFF in maker-pays
                                   # regimes, ON when fees are cheap). In-mode BYPASS in _route so a book
                                   # already in maker isn't ejected by a transient fee tick (mirrors the
                                   # cur==taker spread bypass). ~8 keeps the 6-8bps maker-favorable band.
MAKER_FALLBACK_EDGE_BPS = 1.0      # V2: idle-overflow promote books with ≥ 1.0bps edge (was 0.5) — above
                                   # the −EV 0.5-edge books, below the 1.5 normal enter.
MAX_IDLE_BOOKS = 40                # allow up to 40 idle books before promoting borderline ones to
                                   # maker via fallback; 40 is comfortably below the 48-book
                                   # validator budget (37.5% × 128) where kappa=None is free
CLIFF_IDLE_BOOKS = 46              # V2 cliff-first override: above this, RELAX the maker fallback (rest
                                   # free passive quotes, fee-ceiling off) to pull idle back under 48 —
                                   # NEVER force-cross taker to rescue the cliff (a free quote beats a loss).
ROUTE_MIN_DWELL_S = 300.0          # V2: 300 (was 180) — slug maker↔taker churn that contaminates the ~20h
                                   # per-book kappa window; NOT 600 (would pin a book across a real sub-turn)
ROUTE_SPREAD_EMA_HALFLIFE_S = 60.0 # V2: half-life of the per-book ROUTING spread EMA (de-noise chop);
                                   # the RAW spread is still used for the _open execution gate
EMERGENCY_TAKER_EXIT_BPS = -1.0    # bypass dwell guard when taker rebate goes clearly negative
EMERGENCY_MAKER_EXIT_BPS = -3.0    # bypass dwell guard when maker edge goes deeply negative

# ---- per-book reactive PnL backoff ----
# If a book's net realized PnL over the last PNL_BACKOFF_WINDOW_S is negative (and at least
# PNL_BACKOFF_MIN_RTS round-trips completed in that window), route it to IDLE for a cooldown.
# Fires BEFORE _route() so a bleeding book bypasses the fallback_maker promotion entirely.
# Threshold is sum < 0 on raw quote PnL — scale-invariant across books at different price levels.
PNL_BACKOFF_WINDOW_S = 600.0       # rolling window for the check (~10 min, validator activity window)
PNL_BACKOFF_COOLDOWN_S = 660.0     # idle for 11 min before re-engaging — MUST exceed WINDOW so that
                                   # the triggering trades have fully aged out of the window before
                                   # re-evaluation (cooldown < window causes immediate re-trigger)
PNL_BACKOFF_MIN_RTS = 5            # require ≥5 RTs in the window to fire; guards noise / near-neutral books

# ---- shared round-trip economics / risk (PnL-scale matched across modes) ----
RT_LOSS_CAP_BPS = 4.0              # hard cap on a single forced-exit adverse move (both modes).
                                   # Tighter than before: kappa-3 cubes the downside, so a small,
                                   # consistent loss tail beats an occasional large cut.
RT_WINDOW_S = 570.0                # validator activity sampling window (~10 min)
RT_MAX = 30                        # max profit RTs per book per window
RT_EVENTS_RETENTION_S = 900.0      # RAM: rt_events (sim-time) feeds ONLY the <=600s pnl-backoff + <=570s
                                   # rt_count windows, so retain just above that (1.5x = 900s). It is NOT the
                                   # kappa history (that is kappa_events, wall-time, KAPPA_RT_HISTORY_S). Was
                                   # pruned to KAPPA_RT_HISTORY_S (10800s) = ~18x over-retention at sim pace.
CAPITAL_TURNOVER_CAP = 10.0        # volume cap = this * miner_wealth (24h) — avoid the ceiling
VOLUME_SAFETY = 0.8
VOLUME_ASSESSMENT_NS = 86_400_000_000_000

# ---- activity backstop: guarantee >=1 RT per book per window, bounded to one lot ----
ACTIVITY_DEADLINE_S = 1200.0       # V2: 1200 (was 480) — activity.impact=0 so only a MAJORITY of books
                                   # active in the 3h window is needed; ~9 forced RTs/window not ~22 (cuts
                                   # forced-loss drag). 1200 (not 1800) keeps margin for slow restarted books.

# ---- TAKER mode (mirrors UID 126) ----
TK_MIN_HOLD_S = 1.5
TK_MAX_HOLD_S = 4.0
TK_TP_BPS = 2.5
TK_SL_BPS = 4.0                    # rebate-cushioned; <= RT_LOSS_CAP_BPS in net terms
TK_REOPEN_GAP_S = 1.5              # throttle between a close and the next open
TK_MAX_INVENTORY_LOTS = 3          # bounded same-side build (126 averages in ~2-3 clips)
TK_PYRAMID_GAP_S = 1.0
TK_PYRAMID_MIN_REBATE_BPS = 3.0    # only stack while rebate is comfortably above entry threshold
                                   # (entry bar is now 1.5bps; 3bps = books with meaningful cushion)

# ---- MAKER mode (mirrors UID 109/149) ----
MK_TP_BPS = 10.0                   # target spread capture over the oldest lot
MK_TP_FEE_MULT = 2.0               # require target >= this * maker_fee + a tick
MK_QUOTE_EXPIRY_S = 12.0
MK_EXIT_WALK_START_S = 30.0        # rest reduce at full target below this lot age ...
MK_EXIT_GIVEUP_S = 150.0            # WALK-completion: reduce reaches BREAKEVEN by here, then rests for revert
                                   # (tape: ~3-4% of wins close 90-180s; 150s holds through the grind-up)
MK_MAX_HOLD_S = 180.0             # PROJECT RULE: never hold forever. The never-cut hold is ONLY for the
                                   # sharp-dump→revert window (MeanReversionAgent: ~123/128 books dump then grind
                                   # up over minutes; its tape-tuned close-anyway = 180s). If a lot is STILL
                                   # underwater after this, it did NOT revert → force-cut it (frees the book to
                                   # re-route, and bounds the bag/tail loss). Pairs with the 15bps big stop below:
                                   # a held loser is realized on EITHER underwater>=stop OR age>=this — never held
                                   # indefinitely. (Fixes the mode-stuck + bag/critical-loss failure mode.)
MK_STOP_LOSS_BPS = 20.0            # catastrophe stop above 20bps dump band (tape); below 35-60 cube-bomb zone
MK_IOC_SLIPPAGE_BPS = 4.0          # CEILING: max price concession on the forced IOC cut (distinct
                                   # from the trigger above; bounds realized slippage on the exit)
MK_IOC_ESCALATE_BPS = 8.0          # escalated slippage after 2+ consecutive IOC misses
MK_IOC_CROSS_BPS = 18.0            # wide-limit cross after 4+ misses (not a market order —
                                   # uncapped market orders risk gap fills; 18bps crosses almost
                                   # any normal spread while still bounding catastrophic fills)
MK_REENTRY_COOLDOWN_S = 120.0      # after a forced cut, pause before re-quoting. Matches PureMaker:
                                   # 20s was too short — a trending book gets re-entered and cut again.
MK_LOSS_STREAK_LIMIT = 5           # consecutive losing cuts on a book before a pause (toxic book)
MK_STREAK_COOLDOWN_S = 240.0       # length of that pause; shorter so a book rejoins coverage sooner
MK_MAX_INVENTORY_LOTS = 2.0        # hard per-book lot cap (was 3). Smaller max position => smaller
MK_MAX_INVENTORY_EQUITY_FRAC = 0.08  # worst-case adverse move => smaller loss tail. Main risk lever.

# ---- Kappa-3 (validator-faithful; 3h history) ----
KAPPA_TAU = 0.0
KAPPA_MIN_OBS = 3
KAPPA_MIN_LOOKBACK_S = 5400.0      # 90 min
KAPPA_RT_HISTORY_S = 10_800.0      # 3h

# RT logs only for the scoring validator.
MAIN_VALIDATOR = "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF"

# Mode name constants.
MODE_TAKER = "taker"
MODE_MAKER = "maker"
MODE_IDLE = "idle"

@dataclass
class _Inv:
    """FIFO inventory mirroring the validator's open_positions: oldest-first deques of lots
    (ts, qty, price, fee_per_unit_signed). Net = sum(longs) - sum(shorts). One shared inventory
    per book is used by whichever mode currently owns the book."""
    longs: deque = field(default_factory=deque)
    shorts: deque = field(default_factory=deque)


@dataclass
class _BookState:
    # routing
    mode: str = MODE_IDLE
    mode_since_ns: int = 0              # when the current mode was committed (dwell clock)
    # activity / kappa
    last_rt_ns: int = 0                # last close that generated round-trip volume
    last_cut_ns: int = 0               # last forced (managed-exit) cut; gates re-entry cooldown
    mk_loss_streak: int = 0            # consecutive losing maker cuts (reset on a positive close)
    mk_streak_cooldown_until_ns: int = 0  # pause maker entries on a persistently toxic book
    seen_ns: int = 0                   # first-seen ts; activity clock before the first RT
    rt_events: list[tuple[int, float]] = field(default_factory=list)   # (sim_ts, realized_pnl); ALL validators,
                                                                       # short-retained (RT_EVENTS_RETENTION_S):
                                                                       # feeds pnl-backoff + rt_count only
    kappa_events: list[tuple[int, float]] = field(default_factory=list) # (wall_ts, realized_pnl); MAIN validator
                                                                        # ONLY (kappa3 is logging-only)
    kappa3: float | None = None        # logging-only (RT log); never read by a routing/mode decision
    vol_log: list[tuple[int, float]] = field(default_factory=list)    # (ts, traded quote vol)
    # taker bookkeeping
    last_close_ns: int = 0             # last taker close (reopen throttle)
    last_add_ns: int = 0               # last same-side taker add (pyramid throttle)
    # per-book PnL backoff
    pnl_backoff_until_ns: int = 0      # if > now: book is in PnL-backoff, held at IDLE
    # managed-exit IOC escalation
    mk_ioc_miss_count: int = 0         # consecutive exit IOCs with no position reduction
    mk_ioc_prev_net: float = 0.0       # abs(net) when the last exit IOC was submitted
    # V2 routing-spread EMA (de-noise the instantaneous spread for ROUTING only; raw used in _open)
    spread_ema_bps: float = 0.0        # EMA of half_spread_bps; 0 = uninitialised
    spread_ema_ns: int = 0             # last EMA update ts (for the dt-based decay)


@dataclass
class _RtLogCtx:
    """Snapshot stashed when a position opens; finalized and logged at the closing RT fill."""
    mode: str = "?"
    open_reason: str = "?"
    side: str = "?"
    kappa_at_open: float | None = None
    close_reason: str = "fill"

class AdaptiveRouterV2Agent(FinanceSimulationAgent):
    # ------------------------------------------------------------------ setup
    def initialize(self) -> None:
        bt.logging.set_info()

        self.clip = TARGET_CLIP
        self.exch_min = EXCHANGE_MIN_ORDER_SIZE
        self._flat_eps = 0.5 * 10 ** (-4)
        self._price_decimals: int | None = None
        self._volume_decimals: int | None = None
        self._tick = 0.01
        self.volume_assessment_ns = VOLUME_ASSESSMENT_NS

        # Per-UID jitter so a fleet does not act in lockstep.
        jitter = ((self.uid * 2654435761) % 1000) / 1000.0  # Knuth multiplicative hash
        self.route_min_dwell_ns = int(ROUTE_MIN_DWELL_S * (0.9 + 0.2 * jitter) * _NS)
        self.spread_ema_halflife_ns = int(ROUTE_SPREAD_EMA_HALFLIFE_S * _NS)   # V2 routing-spread EMA
        self.activity_deadline_ns = int(ACTIVITY_DEADLINE_S * (0.92 + 0.08 * jitter) * _NS)
        self.rt_window_ns = int(RT_WINDOW_S * _NS)
        self.rt_events_retention_ns = int(RT_EVENTS_RETENTION_S * _NS)   # RAM: short rt_events prune window
        self.tk_max_hold_ns = int(TK_MAX_HOLD_S * (0.92 + 0.16 * jitter) * _NS)
        self.tk_min_hold_ns = int(TK_MIN_HOLD_S * _NS)
        self.tk_reopen_gap_ns = int(TK_REOPEN_GAP_S * (0.9 + 0.2 * jitter) * _NS)
        self.tk_pyramid_gap_ns = int(TK_PYRAMID_GAP_S * (0.9 + 0.2 * jitter) * _NS)
        self.mk_quote_expiry_ns = int(MK_QUOTE_EXPIRY_S * _NS)
        self.mk_walk_start_ns = int(MK_EXIT_WALK_START_S * _NS)
        self.mk_giveup_ns = int(MK_EXIT_GIVEUP_S * (0.9 + 0.2 * jitter) * _NS)
        self.mk_max_hold_ns = int(MK_MAX_HOLD_S * (0.9 + 0.2 * jitter) * _NS)   # never-hold-forever time cap
        self.mk_reentry_cooldown_ns = int(MK_REENTRY_COOLDOWN_S * _NS)
        self.mk_streak_cooldown_ns = int(MK_STREAK_COOLDOWN_S * _NS)
        self.kappa_rt_history_ns = int(KAPPA_RT_HISTORY_S * _NS)
        self.kappa_min_lookback_ns = int(KAPPA_MIN_LOOKBACK_S * _NS)
        self.pnl_backoff_window_ns = int(PNL_BACKOFF_WINDOW_S * _NS)
        self.pnl_backoff_cooldown_ns = int(PNL_BACKOFF_COOLDOWN_S * _NS)

        # Default mode = the most permissive allowed playbook (so a {"taker"}-only config always
        # runs taker and behaves like a pure scalper; idle is only a fallback when allowed).
        self.default_mode = (
            MODE_TAKER if MODE_TAKER in ALLOWED_MODES
            else MODE_MAKER if MODE_MAKER in ALLOWED_MODES else MODE_IDLE
        )
        self._modes = {
            MODE_TAKER: _TakerMode(),
            MODE_MAKER: _MakerMode(),
            MODE_IDLE: _IdleMode(),
        }

        self.inv: dict[str, dict[int, _Inv]] = {}
        self.books_state: dict[str, dict[int, _BookState]] = {}
        self._rt_log: dict[tuple[str, int], _RtLogCtx] = {}
        self._sim_id: dict[str, str] = {}
        self._step_ts_ns: dict[str, int] = {}
        self._active_validator: str | None = None

        bt.logging.info(
            f"[AdaptiveRouterV2 uid={self.uid}] modes={sorted(ALLOWED_MODES)} default={self.default_mode} "
            f"clip={TARGET_CLIP} dwell={ROUTE_MIN_DWELL_S:.0f}s "
            f"route(taker_rebate>={TAKER_REBATE_ENTER_BPS}/{TAKER_REBATE_EXIT_BPS}bps, "
            f"maker_edge>={MAKER_EDGE_ENTER_BPS}/{MAKER_EDGE_EXIT_BPS}bps, maker_fee<{MAKER_MAX_FEE_BPS}bps) "
            f"mk=reduce->breakeven,stop={MK_STOP_LOSS_BPS:.0f}bps,max_hold={MK_MAX_HOLD_S:.0f}s "
            f"route_ema={ROUTE_SPREAD_EMA_HALFLIFE_S:.0f}s idle_cap={MAX_IDLE_BOOKS}/cliff={CLIFF_IDLE_BOOKS} "
            f"rt_loss_cap={RT_LOSS_CAP_BPS}bps activity_deadline={ACTIVITY_DEADLINE_S:.0f}s "
            f"rt_max={RT_MAX} rt_log={MAIN_VALIDATOR[:8]} "
            f"pnl_backoff(window={PNL_BACKOFF_WINDOW_S:.0f}s cooldown={PNL_BACKOFF_COOLDOWN_S:.0f}s min_rts={PNL_BACKOFF_MIN_RTS})"
        )
        self._tune_gc()

    def _tune_gc(self) -> None:
        """RESPONSE-TIME (axon GC-pause mitigation): the asyncio/axon layer retains completed Task objects
        holding ~128-orderbook state, so the long-lived heap is large and every gen2 GC sweep rescans it —
        pauses spike to tens of ms and stretch handle() past the validator timeout. We control this process's
        GC. Measured: a gen2 gc.collect() drops ~34ms->0 after freeze. (1) history_len=0: the framework
        deep-copies the FULL 128-book state every step and keeps 10 (self.history) which we never read — skip it;
        (2) gc.freeze(): exclude the ~120k permanent import heap from every sweep; (3) raise thresholds: gen2
        sweeps far less often. All behaviour-neutral."""
        self.history_len = 0
        try:
            gc.collect()
            gc.freeze()
            gc.set_threshold(50_000, 500, 500)
            bt.logging.info(f"[AdaptiveRouterV2 uid={self.uid}] gc tuned: frozen={gc.get_freeze_count()} "
                            f"thresholds={gc.get_threshold()} history_len=0")
        except Exception as ex:
            bt.logging.warning(f"[AdaptiveRouterV2 uid={self.uid}] gc tune skipped: {ex}")

    # --------------------------------------------------------------- lifecycle
    def update(self, state: MarketSimulationStateUpdate) -> None:
        self._active_validator = state.dendrite.hotkey
        self._step_ts_ns[self._active_validator] = int(state.timestamp)
        self._ensure_simulation(self._active_validator, state.config.simulation_id)
        super().update(state)

    def _ensure_simulation(self, validator: str, simulation_id: str | None) -> None:
        """Drop per-validator state when the validator starts a new simulation."""
        if self._sim_id.get(validator) == simulation_id:
            return
        self.inv.pop(validator, None)
        self.books_state.pop(validator, None)
        self._rt_log = {k: v for k, v in self._rt_log.items() if k[0] != validator}
        if simulation_id is not None:
            self._sim_id[validator] = simulation_id
        else:
            self._sim_id.pop(validator, None)
        bt.logging.info(
            f"[AdaptiveRouterV2 uid={self.uid}] new simulation: {validator[:8]} sim_id={simulation_id}"
        )

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        validator = state.dendrite.hotkey
        cfg = self.simulation_config
        self._sync_precision(cfg.priceDecimals, cfg.volumeDecimals)

        vol_dp = cfg.volumeDecimals
        volume_cap = CAPITAL_TURNOVER_CAP * cfg.miner_wealth * VOLUME_SAFETY
        now = state.timestamp

        # Idle-book guard: if last step left >MAX_IDLE_BOOKS idle, promote borderline books to
        # maker with a relaxed edge threshold so we stay inside the 48-free-slot budget.
        idle_count = sum(
            1 for bst in (self.books_state.get(validator) or {}).values()
            if bst.mode == MODE_IDLE
        )
        fallback_maker = idle_count > MAX_IDLE_BOOKS
        # V2 cliff-first: above CLIFF_IDLE_BOOKS, relax the maker fallback HARD (edge→0, fee-ceiling off)
        # so borderline books rest FREE passive maker quotes and pull idle back under the 48 cliff —
        # never force-cross taker to rescue the cliff (a free unfilled quote beats a guaranteed loss).
        cliff = idle_count > CLIFF_IDLE_BOOKS

        for book_id in sorted(self.accounts.keys()):
            book = state.books.get(book_id)
            account = self.accounts.get(book_id) if book else None
            if book is None or account is None:
                continue
            try:
                self._step_book(response, validator, book_id, book, account,
                                vol_dp, volume_cap, now, fallback_maker, cliff)
            except Exception as ex:
                bt.logging.warning(f"[AdaptiveRouterV2 uid={self.uid}] step {book_id}: {ex}")

        return response

    # ------------------------------------------------------------------ per-book dispatch
    def _step_book(
        self, response, validator: str, book_id: int, book, account,
        vol_dp: int, volume_cap: float, now: int, fallback_maker: bool = False, cliff: bool = False,
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
        if st.seen_ns == 0:
            st.seen_ns = now
        # V2: update the routing-spread EMA (de-noise chop for ROUTING; _open keeps the raw spread).
        inst_half = (best_ask - best_bid) / mid * 0.5 * 1e4
        if st.spread_ema_bps <= 0.0 or st.spread_ema_ns == 0:
            st.spread_ema_bps = inst_half
        elif self.spread_ema_halflife_ns > 0:
            dt = now - st.spread_ema_ns
            alpha = 1.0 - 0.5 ** (dt / self.spread_ema_halflife_ns) if dt > 0 else 0.0
            st.spread_ema_bps += alpha * (inst_half - st.spread_ema_bps)
        st.spread_ema_ns = now
        if st.mode_since_ns == 0:
            # Backdate the dwell clock so the FIRST routing decision can fire immediately. Without
            # this every book is pinned to default_mode (taker) for ROUTE_MIN_DWELL_S, spending 3
            # minutes crossing the spread on books that should be maker/idle.
            st.mode = self.default_mode
            st.mode_since_ns = now - self.route_min_dwell_ns - 1
        pruned = self._prune_rt_events(st, now)
        if pruned and self._rt_log_enabled(validator):
            # kappa3 is LOGGING-ONLY and the RT log is MAIN-validator-only; skip the kappa refresh (and its
            # cross-book scan) for non-scoring validators. rt_events is still pruned above for all validators.
            self._refresh_book_kappa(validator, book_id, time.time_ns())

        net = self._net_qty(inv)
        flat = abs(net) < self.exch_min

        # Route only when FLAT and the dwell has elapsed, so no position straddles two modes and
        # books cannot flip-flop. A pending switch first cancels resting orders, then commits.
        if flat:
            # PnL backoff: checked BEFORE _route() so a bleeding book bypasses the fallback_maker
            # promotion. No dwell guard here — backoff is a safety trigger, not a fee-regime flip.
            if self._pnl_backoff_check(st, now):
                if st.mode != MODE_IDLE:
                    if account.orders:
                        self._cancel_all(response, account, book_id)
                        return
                    bt.logging.info(
                        f"[AdaptiveRouterV2 uid={self.uid}] PNL-BACKOFF {st.mode}->idle book={book_id}"
                    )
                    st.mode, st.mode_since_ns = MODE_IDLE, now
            else:
                # V2: NO maker→taker demote on a recent-PnL blip — flipping a mean-reverting maker book
                # contaminates its per-book Sortino for the ~20h window. The never-cut maker (hold for
                # revert) + the realized-PnL backoff below handle genuine losers instead.
                want = self._route(st, account, best_bid, best_ask, mid,
                                   fallback_maker=fallback_maker, cliff=cliff)
                if want != st.mode:
                    # Emergency flip: bypass the 180s dwell guard when the current mode has
                    # turned clearly net-negative (not just borderline — only on obviously-wrong-
                    # side regimes). Only reachable when flat, so no active position is disrupted.
                    taker_fee_em = self._taker_fee_rate(account)
                    rebate_em = (-taker_fee_em * 1e4) if taker_fee_em is not None else -1e9
                    maker_fee_em = self._maker_fee_rate(account)
                    mk_edge_em = (
                        (best_ask - best_bid) / mid * 0.5 * 1e4 if mid > 0 else 0.0
                    ) - ((maker_fee_em * 1e4) if maker_fee_em is not None else 1e9)
                    emergency = (
                        (st.mode == MODE_TAKER and rebate_em < EMERGENCY_TAKER_EXIT_BPS) or
                        (st.mode == MODE_MAKER and mk_edge_em < EMERGENCY_MAKER_EXIT_BPS)
                    )
                    if emergency or (now - st.mode_since_ns) >= self.route_min_dwell_ns:
                        if account.orders:
                            self._cancel_all(response, account, book_id)
                            return
                        bt.logging.info(
                            f"[AdaptiveRouterV2 uid={self.uid}] "
                            f"{'EMERGENCY-FLIP' if emergency else 'ROUTE'} "
                            f"{st.mode}->{want} book={book_id} "
                            f"taker_fee={self._taker_fee_rate(account)} maker_fee={self._maker_fee_rate(account)} "
                            f"spread_bps={(best_ask - best_bid) / mid * 1e4:.1f}"
                            + (" [cliff]" if cliff else (" [fallback-maker]" if fallback_maker else ""))
                        )
                        st.mode, st.mode_since_ns = want, now

        mode = self._modes.get(st.mode) or self._modes[self.default_mode]
        mode.step(self, response, validator, book_id, book, account, st, inv, net,
                  best_bid, best_ask, mid, vol_dp, volume_cap, now)

    # ------------------------------------------------------------------ routing
    def _route(self, st: _BookState, account, best_bid: float, best_ask: float, mid: float,
               *, fallback_maker: bool = False, cliff: bool = False) -> str:
        """Pick the +EV playbook for this book from the LIVE fee regime, with hysteresis: the
        current mode is held to a looser EXIT threshold than the ENTER threshold for switching in,
        so a book near a boundary stays put instead of flip-flopping. V2: classification uses the
        SMOOTHED (EMA) half-spread to de-noise chop; the raw spread stays in the _open exec gate."""
        cur = st.mode
        taker_fee = self._taker_fee_rate(account)
        maker_fee = self._maker_fee_rate(account)
        rebate_bps = (-taker_fee * 1e4) if taker_fee is not None else -1e9
        # V2: use the EMA half-spread for ROUTING (de-noise transient ticks); fall back to instantaneous
        # only before the EMA has warmed up.
        half_spread_bps = (st.spread_ema_bps if st.spread_ema_bps > 0.0
                           else ((best_ask - best_bid) / mid * 0.5 * 1e4 if mid > 0 else 0.0))
        maker_fee_bps = (maker_fee * 1e4) if maker_fee is not None else 1e9
        maker_edge_bps = half_spread_bps - maker_fee_bps   # capture half-spread, pay the maker fee

        taker_min_rebate = TAKER_REBATE_EXIT_BPS if cur == MODE_TAKER else TAKER_REBATE_ENTER_BPS
        # Maker enter bar: relaxed under the idle-overflow fallback, and HARD-relaxed (→0) at the cliff so
        # borderline books rest free maker quotes rather than tipping idle past the 48 budget.
        maker_enter_edge = (0.0 if cliff
                            else MAKER_FALLBACK_EDGE_BPS if fallback_maker
                            else MAKER_EDGE_ENTER_BPS)
        maker_min_edge = MAKER_EDGE_EXIT_BPS if cur == MODE_MAKER else maker_enter_edge
        # Taker viable only when the rebate covers the crossing cost (rebate > half_spread); books already
        # in taker skip the spread check (transient spikes shouldn't eject; _open gates execution).
        spread_viable = (cur == MODE_TAKER) or (rebate_bps > half_spread_bps)
        taker_ok = (MODE_TAKER in ALLOWED_MODES) and rebate_bps >= taker_min_rebate and spread_viable
        # Maker requires a net spread edge AND maker_fee below the ceiling (a high fee = adverse
        # selection). V2: the ceiling has a cur==MODE_MAKER BYPASS (don't eject an in-position maker on a
        # transient fee tick — mirrors the taker spread bypass) and a cliff bypass (free-quote rescue).
        maker_fee_ok = (cur == MODE_MAKER) or cliff or (maker_fee_bps < MAKER_MAX_FEE_BPS)
        maker_ok = ((MODE_MAKER in ALLOWED_MODES)
                    and maker_fee_ok
                    and maker_edge_bps >= maker_min_edge)

        if taker_ok and maker_ok:
            # Both edges available -> prefer the one with more margin above its (dynamic) enter bar.
            # Using maker_enter_edge (not the hardcoded constant) so that fallback-mode promotion of
            # borderline maker books is correctly reflected in the tiebreaker.
            return (MODE_TAKER if (rebate_bps - TAKER_REBATE_ENTER_BPS)
                    >= (maker_edge_bps - maker_enter_edge) else MODE_MAKER)
        if taker_ok:
            return MODE_TAKER
        if maker_ok:
            return MODE_MAKER
        if MODE_IDLE in ALLOWED_MODES:
            return MODE_IDLE
        return cur if cur in ALLOWED_MODES else self.default_mode

    # ------------------------------------------------------------------ events
    def onTrade(self, event: TradeEvent, validator: str | None = None) -> None:
        """Route maker AND taker fills into the shared FIFO. is_buy mirrors the validator:
        taker+BUY or maker+SELL-aggressor both mean WE bought."""
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

    # ------------------------------------------------------------------ FIFO state
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

    def _net_qty(self, inv: _Inv) -> float:
        return self._long_qty(inv) - self._short_qty(inv)

    @staticmethod
    def _side_avg(lots: deque) -> float:
        tot = sum(q for _, q, _, _ in lots)
        return sum(q * p for _, q, p, _ in lots) / tot if tot > 0 else 0.0

    def _apply_fill(
        self, validator: str, book_id: int, is_buy: bool, qty: float, price: float,
        fee: float, ts: int,
    ) -> None:
        inv = self._inv(validator, book_id)
        realized, rtv, matched_ts, gross = self._match_fifo(inv, is_buy, qty, price, fee, ts)
        if rtv > 0:
            st = self._bstate(validator, book_id)
            kappa_before = st.kappa3
            st.last_rt_ns = ts
            if st.mode == MODE_MAKER and realized > 0:
                st.mk_loss_streak = 0          # a winning maker close clears the toxic-book streak
            self._record_rt_close(validator, book_id, ts, realized)
            self._log_rt(validator, book_id, ts,
                         hold_s=(ts - matched_ts) / _NS if matched_ts else None,
                         side=("buy" if is_buy else "sell"), exit_px=price, rtv=rtv,
                         gross=gross, net=realized, kappa_before=kappa_before, kappa_after=st.kappa3)
        elif rtv == 0 and self._rt_log_enabled(validator) and (validator, book_id) not in self._rt_log:
            # Pure opening fill with no prior stash -> a passive maker quote got hit. Record context
            # now so the eventual closing RT logs a real open_reason/side instead of "?".
            st = self._bstate(validator, book_id)
            self._stash_open(validator, book_id, st, st.mode, "passive",
                             "long" if is_buy else "short")

    def _match_fifo(
        self, inv: _Inv, is_buy: bool, qty: float, price: float, fee: float, ts: int,
    ) -> tuple[float, float, int | None, float]:
        """FIFO-match a fill against opposing lots (validator-faithful). Returns
        (realized_net_of_fees, roundtrip_volume, oldest_matched_ts, gross_pnl)."""
        close_book = inv.shorts if is_buy else inv.longs   # buying closes shorts; selling longs
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
                close_fee = fee * take * qinv
                open_fee = o_fee * (take / o_qty)
                close_book[0] = (o_ts, o_qty - take, o_px, o_fee - open_fee)
            realized += price_pnl - open_fee - close_fee
            gross += price_pnl
            rtv += take
            remaining -= take
        if remaining > self._flat_eps:
            open_book.append((ts, remaining, price, fee * remaining * qinv))
        return realized, rtv, matched_ts, gross

    # ------------------------------------------------------------------ kappa-3
    def _prune_rt_events(self, st: _BookState, now: int) -> bool:
        cutoff = now - self.rt_events_retention_ns   # only the <=600s pnl-backoff/rt_count windows read this
        before = len(st.rt_events)
        st.rt_events = [(t, p) for t, p in st.rt_events if t >= cutoff]
        return len(st.rt_events) != before

    def _pnl_backoff_check(self, st: _BookState, now: int) -> bool:
        """True if net realized PnL over the rolling window is negative; holds the book at IDLE."""
        if st.pnl_backoff_until_ns > now:
            return True
        if st.pnl_backoff_until_ns > 0:
            st.pnl_backoff_until_ns = 0   # expired; reset so next trigger sets a clean deadline
        cutoff = now - self.pnl_backoff_window_ns
        recent = [p for ts, p in st.rt_events if ts >= cutoff]
        if len(recent) >= PNL_BACKOFF_MIN_RTS and sum(recent) < 0:
            st.pnl_backoff_until_ns = now + self.pnl_backoff_cooldown_ns
            return True
        return False

    def _record_rt_close(self, validator: str, book_id: int, ts: int, net_pnl: float) -> None:
        st = self._bstate(validator, book_id)
        self._prune_rt_events(st, ts)
        st.rt_events.append((ts, net_pnl))
        # kappa_events / kappa3 are LOGGING-ONLY (kappa3 feeds no routing/mode decision; the RT log is
        # MAIN-validator-only). Maintain them only for the scoring validator — saves the kappa_events list
        # and the per-RT cross-book kappa scan on every other validator, with zero change to orders/logs.
        if self._rt_log_enabled(validator):
            wall_ns = time.time_ns()
            cutoff = wall_ns - self.kappa_rt_history_ns
            st.kappa_events = [(t, p) for t, p in st.kappa_events if t >= cutoff]
            st.kappa_events.append((wall_ns, net_pnl))
            self._refresh_book_kappa(validator, book_id, wall_ns)

    def _global_rt_timestamps(self, validator: str, now: int) -> list[int]:
        cutoff = now - self.kappa_rt_history_ns
        ts_set: set[int] = set()
        for st in self.books_state.get(validator, {}).values():
            for ts, _ in st.kappa_events:
                if ts >= cutoff:
                    ts_set.add(ts)
        return sorted(ts_set)

    def _book_pnl_series(self, validator: str, book_id: int, now: int) -> list[float]:
        timestamps = self._global_rt_timestamps(validator, now)
        if not timestamps:
            return []
        cutoff = now - self.kappa_rt_history_ns
        by_ts = {t: p for t, p in self._bstate(validator, book_id).kappa_events if t >= cutoff}
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

    def _kappa_history_ready(self, validator: str, now: int) -> bool:
        ts = self._global_rt_timestamps(validator, now)
        return len(ts) >= 2 and ts[-1] - ts[0] >= self.kappa_min_lookback_ns

    def _refresh_book_kappa(self, validator: str, book_id: int, now: int) -> None:
        st = self._bstate(validator, book_id)
        if not self._kappa_history_ready(validator, now):
            st.kappa3 = None
            ts_list = self._global_rt_timestamps(validator, now)
            if ts_list:
                bt.logging.info(
                    f"[AdaptiveRouterV2 uid={self.uid}] kappa_not_ready book={book_id} "
                    f"ts_count={len(ts_list)} span_s={(ts_list[-1]-ts_list[0])/1e9:.0f} "
                    f"need_span_s={self.kappa_min_lookback_ns/1e9:.0f}"
                )
            else:
                bt.logging.info(
                    f"[AdaptiveRouterV2 uid={self.uid}] kappa_no_events book={book_id}"
                )
            return
        pnl = self._book_pnl_series(validator, book_id, now)
        result = self._kappa3_raw(pnl)
        if result is None:
            bt.logging.info(
                f"[AdaptiveRouterV2 uid={self.uid}] kappa_raw_none book={book_id} "
                f"series_len={len(pnl)} nonzero={sum(1 for x in pnl if x != 0.0)}"
            )
        st.kappa3 = result

    def _rt_count(self, st: _BookState, now: int, window_ns: int | None = None) -> int:
        cutoff = now - (self.rt_window_ns if window_ns is None else window_ns)
        return sum(1 for ts, _ in st.rt_events if ts >= cutoff)

    # ------------------------------------------------------------------ activity / volume
    def _activity_due(self, st: _BookState, now: int) -> bool:
        ref = st.last_rt_ns if st.last_rt_ns > 0 else st.seen_ns
        return (now - ref) >= self.activity_deadline_ns

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

    def _budget_ok(self, validator: str, book_id: int, st: _BookState, now: int, volume_cap: float) -> bool:
        return (self._rt_count(st, now) < RT_MAX
                and self._rolled_quote_volume(validator, book_id, now) < volume_cap)

    # ------------------------------------------------------------------ precision / market helpers
    def _sync_precision(self, price_decimals: int, volume_decimals: int) -> None:
        if price_decimals == self._price_decimals and volume_decimals == self._volume_decimals:
            return
        self._price_decimals = price_decimals
        self._volume_decimals = volume_decimals
        self._tick = 10 ** (-price_decimals)
        self.clip = round(max(TARGET_CLIP, 10 ** (-volume_decimals)), volume_decimals)
        self.exch_min = max(EXCHANGE_MIN_ORDER_SIZE, 10 ** (-volume_decimals))
        self._flat_eps = 0.5 * 10 ** (-volume_decimals)
        bt.logging.info(
            f"[AdaptiveRouterV2 uid={self.uid}] priceDecimals={price_decimals} tick={self._tick} "
            f"volumeDecimals={volume_decimals} clip={self.clip} exch_min={self.exch_min}"
        )

    @staticmethod
    def _avail(balance) -> float:
        if balance is None:
            return 0.0
        return (balance.free or 0.0) + (balance.reserved or 0.0)

    def _book_equity(self, account, mid: float) -> float:
        q, b = account.quote_balance, account.base_balance
        quote = ((q.free or 0.0) + (q.reserved or 0.0)) if q else 0.0
        base = ((b.free or 0.0) + (b.reserved or 0.0)) if b else 0.0
        return quote + base * mid

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

    @staticmethod
    def _microprice(book, mid: float) -> float:
        bid, ask = book.bids[0], book.asks[0]
        denom = bid.quantity + ask.quantity
        if denom <= 0:
            return mid
        return (ask.price * bid.quantity + bid.price * ask.quantity) / denom

    def _bias(self, book, mid: float) -> int:
        """microprice vs mid -> directional lean; tie -> long."""
        return OrderDirection.SELL if self._microprice(book, mid) < mid else OrderDirection.BUY

    def _cancel_all(self, response, account, book_id: int) -> None:
        if account.orders:
            response.cancel_orders(book_id, [o.id for o in account.orders])

    def _submit_market(
        self, response, book_id: int, direction: int, qty: float,
        *, leverage: float = 0.0, settlement: LoanSettlementOption = LoanSettlementOption.NONE,
    ) -> None:
        kwargs: dict[str, Any] = {
            "book_id": book_id, "direction": direction, "quantity": qty,
            "currency": OrderCurrency.BASE, "stp": STP.CANCEL_OLDEST,
        }
        if leverage > 0:
            kwargs["leverage"] = leverage
        if settlement != LoanSettlementOption.NONE:
            kwargs["settlement_option"] = settlement
        response.market_order(**kwargs)

    def _submit_limit(
        self, response, book_id: int, direction: int, qty: float, price: float,
        *, post_only: bool = True, ioc: bool = False,
        settlement: LoanSettlementOption = LoanSettlementOption.NONE,
    ) -> None:
        kwargs: dict[str, Any] = {
            "book_id": book_id, "direction": direction, "quantity": qty, "price": price,
            "stp": STP.CANCEL_OLDEST,
        }
        if ioc:
            kwargs["timeInForce"] = TimeInForce.IOC
        else:
            kwargs["postOnly"] = post_only
            kwargs["timeInForce"] = TimeInForce.GTT
            kwargs["expiryPeriod"] = self.mk_quote_expiry_ns
        if settlement != LoanSettlementOption.NONE:
            kwargs["settlement_option"] = settlement
        response.limit_order(**kwargs)

    # ------------------------------------------------------------------ RT logging
    @staticmethod
    def _rt_log_enabled(validator: str) -> bool:
        return validator == MAIN_VALIDATOR

    def _stash_open(self, validator: str, book_id: int, st: _BookState, mode: str,
                    reason: str, side: str) -> None:
        if not self._rt_log_enabled(validator):
            return
        self._rt_log[(validator, book_id)] = _RtLogCtx(
            mode=mode, open_reason=reason, side=side, kappa_at_open=st.kappa3)

    @staticmethod
    def _fmt_kappa_pair(before: float | None, after: float | None) -> str:
        if before is None and after is None:
            return "n/a"
        if before is None:
            return f"n/a->{after:.4f}"
        if after is None:
            return f"{before:.4f}->n/a"
        return f"{before:.4f}->{after:.4f}"

    def _log_rt(self, validator, book_id, ts, *, hold_s, side, exit_px, rtv, gross, net,
                kappa_before, kappa_after) -> None:
        if not self._rt_log_enabled(validator):
            return
        ctx = self._rt_log.pop((validator, book_id), _RtLogCtx())
        if ctx.mode == "?":
            ctx.mode = self._bstate(validator, book_id).mode   # maker fills open passively (no stash)
        hold_str = f"{hold_s:.2f}" if hold_s is not None else "n/a"
        bt.logging.info(
            f"[AdaptiveRouterV2 uid={self.uid} RT] book={book_id} mode={ctx.mode} "
            f"open={ctx.open_reason}/{ctx.side} close={ctx.close_reason} "
            f"rtv={rtv:.4f} exit={exit_px:.4f} hold_s={hold_str} "
            f"gross={gross:+.4f} net={net:+.4f} "
            f"kappa={self._fmt_kappa_pair(kappa_before, kappa_after)}"
        )


class _Mode:
    """Base class for a per-book playbook. Modes are stateless singletons; all per-book state
    lives on the shared agent (_BookState, _Inv), so a mode switch never loses inventory or
    accounting. Every mode shares the same bounded activity backstop so activity stays 1.0."""

    name = "?"

    def step(self, agent, response, validator, book_id, book, account, st, inv, net,
             best_bid, best_ask, mid, vol_dp, volume_cap, now) -> None:
        raise NotImplementedError

    # -- shared bounded activity backstop (one lot; keeps activity factor at 1.0) --
    def _activity_close(self, agent, response, book_id, account, inv, net,
                        best_bid, best_ask, vol_dp, *,
                        direction: int = OrderDirection.BUY) -> bool:
        """Force ONE round-trip-producing close (or seed one lot if flat) with capped slippage,
        so the book generates round-trip volume within the activity window. Bounded to a single
        lot, so the rare forced loss is tiny."""
        slip = RT_LOSS_CAP_BPS / 1e4
        pdp = agent._price_decimals
        long_q, short_q = agent._long_qty(inv), agent._short_qty(inv)
        lot = max(agent.clip, agent.exch_min)
        base_avail = agent._avail(account.base_balance)
        quote_avail = agent._avail(account.quote_balance)
        if long_q >= agent.exch_min:
            q = round(min(long_q, base_avail, lot), vol_dp)
            if q < agent.exch_min:
                return False
            agent._cancel_all(response, account, book_id)
            agent._submit_limit(response, book_id, OrderDirection.SELL, q,
                                round(best_bid * (1.0 - slip), pdp), ioc=True, post_only=False)
        elif short_q >= agent.exch_min:
            buy_px = best_ask * (1.0 + slip)
            q_max = quote_avail / buy_px if buy_px > 0 else short_q
            q = round(min(short_q, lot, q_max), vol_dp)
            if q < agent.exch_min:
                return False
            agent._cancel_all(response, account, book_id)
            agent._submit_limit(response, book_id, OrderDirection.BUY, q,
                                round(best_ask * (1.0 + slip), pdp), ioc=True, post_only=False,
                                settlement=agent._loan_settlement(account))
        else:
            q = lot
            if direction == OrderDirection.SELL and base_avail >= q:
                agent._cancel_all(response, account, book_id)
                agent._submit_limit(response, book_id, OrderDirection.SELL, q,
                                    round(best_bid * (1.0 - slip), pdp), ioc=True, post_only=False)
            else:
                if quote_avail < q * best_ask * (1.0 + slip):
                    return False
                agent._cancel_all(response, account, book_id)
                agent._submit_limit(response, book_id, OrderDirection.BUY, q,
                                    round(best_ask * (1.0 + slip), pdp), ioc=True, post_only=False)
        return True


class _IdleMode(_Mode):
    """Books idled by fee regime or PnL backoff. Do NOTHING (no forced activity RT).
    Per the validator's reward.py: a book with 0 round-trips gets kappa=None, which is DROPPED
    from the kappa median entirely (up to ~37.5% of books are free) -- the activity factor is
    never applied to a None book. Forcing an activity RT does NOT help (None stays None) and
    actively HURTS: the lossy RT gives the book a real negative kappa that drags the median.
    The right lever for too many idle books is the fallback_maker promotion, not trading them.
    Idle only drains residual inventory left over from a mode switch."""

    name = MODE_IDLE

    def step(self, agent, response, validator, book_id, book, account, st, inv, net,
             best_bid, best_ask, mid, vol_dp, volume_cap, now) -> None:
        # Only act to flatten a residual position carried in from a previous mode; otherwise stay
        # completely flat so the book remains kappa=None (free-dropped, not penalized).
        if abs(net) >= agent.exch_min:
            agent._stash_open(validator, book_id, st, self.name, "drain",
                              "long" if net >= 0 else "short")
            self._activity_close(agent, response, book_id, account, inv, net,
                                 best_bid, best_ask, vol_dp)


class _TakerMode(_Mode):
    """Deep-rebate scalper (mirrors UID 126). One small clip in the microprice-bias direction;
    exit on TP / SL / max-hold within seconds — the rebate on both legs cushions a stop into a
    net win. Bounded same-side inventory building is allowed while the rebate stays deep."""

    name = MODE_TAKER

    def step(self, agent, response, validator, book_id, book, account, st, inv, net,
             best_bid, best_ask, mid, vol_dp, volume_cap, now) -> None:
        if abs(net) >= agent.exch_min:
            if not self._exit(agent, response, validator, book_id, account, inv, net,
                              best_bid, best_ask, vol_dp, now, st):
                self._maybe_add(agent, response, validator, book_id, book, account, st, inv, net,
                                best_bid, best_ask, mid, volume_cap, now, vol_dp)
            return
        throttled = st.last_close_ns and (now - st.last_close_ns) < agent.tk_reopen_gap_ns
        if not throttled and self._open(agent, response, validator, book_id, book, account, st,
                                        best_bid, best_ask, mid, volume_cap, now, vol_dp):
            return
        if agent._activity_due(st, now):
            act_dir = agent._bias(book, mid)
            act_side = "long" if act_dir == OrderDirection.BUY else "short"
            agent._stash_open(validator, book_id, st, self.name, "activity", act_side)
            self._activity_close(agent, response, book_id, account, inv, net,
                                 best_bid, best_ask, vol_dp, direction=act_dir)

    def _exit(self, agent, response, validator, book_id, account, inv, net,
              best_bid, best_ask, vol_dp, now, st) -> bool:
        if net > 0:
            avg = agent._side_avg(inv.longs)
            ts0 = inv.longs[0][0]
            gross_bps = (best_bid - avg) / avg * 1e4 if avg > 0 else 0.0
        else:
            avg = agent._side_avg(inv.shorts)
            ts0 = inv.shorts[0][0]
            gross_bps = (avg - best_ask) / avg * 1e4 if avg > 0 else 0.0
        held = now - ts0
        if held < agent.tk_min_hold_ns:
            return False
        if gross_bps >= TK_TP_BPS:
            reason = "tp"
        elif gross_bps <= -TK_SL_BPS:
            reason = "sl"
        elif held >= agent.tk_max_hold_ns:
            reason = "time"
        else:
            return False
        if agent._rt_log_enabled(validator):
            ctx = agent._rt_log.get((validator, book_id))
            if ctx is not None:
                ctx.close_reason = reason
        agent._cancel_all(response, account, book_id)
        if net > 0:
            q = round(min(abs(net), agent._avail(account.base_balance)), vol_dp)
            if q < agent.exch_min:
                return True   # orders cancelled; lot too small to exit this step — return True
                              # so the caller does NOT call _maybe_add() on a SL/TP position
            agent._submit_market(response, book_id, OrderDirection.SELL, q)
        else:
            q = round(min(abs(net), agent._avail(account.quote_balance) / best_ask if best_ask > 0 else abs(net)), vol_dp)
            if q < agent.exch_min:
                return True   # same: prevent _maybe_add() from accumulating past the SL/TP
            agent._submit_market(response, book_id, OrderDirection.BUY, q,
                                 settlement=agent._loan_settlement(account))
        st.last_close_ns = now
        return True

    def _open(self, agent, response, validator, book_id, book, account, st,
              best_bid, best_ask, mid, volume_cap, now, vol_dp) -> bool:
        taker_fee = agent._taker_fee_rate(account)
        if taker_fee is None or not agent._budget_ok(validator, book_id, st, now, volume_cap):
            return False
        rebate_bps = -taker_fee * 1e4
        half_spread_bps = (best_ask - best_bid) / mid * 0.5 * 1e4 if mid > 0 else 1e9
        est_bps = 2.0 * rebate_bps - 2.0 * half_spread_bps   # rebate both legs minus 2 crossings
        if est_bps <= 0.0:
            return False
        direction = agent._bias(book, mid)
        q = round(agent.clip, vol_dp)
        if q < agent.exch_min:
            return False
        if direction == OrderDirection.BUY:
            if agent._avail(account.quote_balance) < q * best_ask:
                return False
            agent._submit_market(response, book_id, OrderDirection.BUY, q,
                                 settlement=agent._loan_settlement(account))
        else:
            if agent._avail(account.base_balance) < q:    # never naked-short from flat
                return False
            agent._submit_market(response, book_id, OrderDirection.SELL, q)
        agent._stash_open(validator, book_id, st, self.name, "scalp",
                          "long" if direction == OrderDirection.BUY else "short")
        return True

    def _maybe_add(self, agent, response, validator, book_id, book, account, st, inv, net,
                   best_bid, best_ask, mid, volume_cap, now, vol_dp) -> None:
        """Bounded same-side build while the rebate is DEEP (so a stacked stop stays net-positive),
        mirroring 126 averaging into a paid edge. Hard inventory cap keeps the unwind small."""
        if TK_MAX_INVENTORY_LOTS <= 1:
            return
        taker_fee = agent._taker_fee_rate(account)
        if taker_fee is None:
            return
        if -taker_fee * 1e4 < TK_PYRAMID_MIN_REBATE_BPS:
            return
        if abs(net) + agent.clip > TK_MAX_INVENTORY_LOTS * agent.clip + agent._flat_eps:
            return
        if st.last_add_ns and (now - st.last_add_ns) < agent.tk_pyramid_gap_ns:
            return
        if not agent._budget_ok(validator, book_id, st, now, volume_cap):
            return
        direction = agent._bias(book, mid)
        pos_dir = OrderDirection.BUY if net > 0 else OrderDirection.SELL
        if direction != pos_dir:                          # only press our own side
            return
        q = round(agent.clip, vol_dp)
        if q < agent.exch_min:
            return
        if direction == OrderDirection.BUY:
            if agent._avail(account.quote_balance) < q * best_ask:
                return
            agent._submit_market(response, book_id, OrderDirection.BUY, q,
                                 settlement=agent._loan_settlement(account))
        else:
            if agent._avail(account.base_balance) < q:
                return
            agent._submit_market(response, book_id, OrderDirection.SELL, q)
        st.last_add_ns = now

class _MakerMode(_Mode):
    """Two-sided spread capture (mirrors UID 109/149). Flat -> quote both sides inside the touch.
    Holding -> work ONLY the reducing side, priced off the FIFO worst lot and walked toward the
    touch with age, so every consumed lot is round-trip-positive and we never bag. A managed exit
    IOC-cuts an aged/underwater lot with capped slippage; an activity backstop guarantees a close
    each window. The router only sends spread-rich books here, so passive capture is +EV."""

    name = MODE_MAKER

    def step(self, agent, response, validator, book_id, book, account, st, inv, net,
             best_bid, best_ask, mid, vol_dp, volume_cap, now) -> None:
        if (self._risk_trim(agent, response, book_id, account, inv, net, mid,
                            vol_dp, best_bid, best_ask)
                or self._managed_exit(agent, response, validator, book_id, account, inv, net,
                                      best_bid, best_ask, vol_dp, now)):
            # Count forced IOC cuts (risk-trim or managed-exit) as losses until proven otherwise;
            # a positive close in _apply_fill resets the streak. After STREAK_LIMIT consecutive
            # cuts, pause maker entries for mk_streak_cooldown_ns to avoid re-bleeding on a
            # persistently trending book.
            st.last_cut_ns = now
            st.mk_loss_streak += 1
            if st.mk_loss_streak >= MK_LOSS_STREAK_LIMIT:
                st.mk_streak_cooldown_until_ns = now + agent.mk_streak_cooldown_ns
            return
        # V2 NEVER-CUT activity: force-close a HELD lot for activity ONLY if it exits at BREAKEVEN-or-
        # better (bid≥entry long / ask≤entry short). An underwater lot is left resting on its breakeven
        # reduce quote — never realize a loss just to register activity (maker = rest/walk, not cross).
        if agent._activity_due(st, now) and abs(net) >= agent.exch_min:
            px0 = agent._side_avg(inv.longs if net > 0 else inv.shorts)
            be_ok = (best_bid >= px0 > 0) if net > 0 else (0.0 < best_ask <= px0)
            if be_ok:
                agent._stash_open(validator, book_id, st, self.name, "activity",
                                  "long" if net >= 0 else "short")
                if self._activity_close(agent, response, book_id, account, inv, net,
                                        best_bid, best_ask, vol_dp):
                    return
        desired = self._desired_quotes(agent, validator, book_id, account, inv, net,
                                       best_bid, best_ask, mid, volume_cap, now, vol_dp, st)
        self._reconcile(agent, response, account, book_id, desired)

    def _risk_trim(self, agent, response, book_id, account, inv, net, mid,
                   vol_dp, best_bid: float = 0.0, best_ask: float = 0.0) -> bool:
        qty = abs(net)
        if qty < agent._flat_eps:
            return False
        lot_cap = MK_MAX_INVENTORY_LOTS * agent.clip
        equity = agent._book_equity(account, mid)
        notional_cap = MK_MAX_INVENTORY_EQUITY_FRAC * equity if equity > 0 else float("inf")
        excess = max(qty - lot_cap, (qty * mid - notional_cap) / mid if mid > 0 else 0.0)
        if excess <= agent._flat_eps:
            return False
        trim = round(min(qty, max(excess, agent.exch_min)), vol_dp)
        if trim < agent.exch_min:
            return False
        slip = MK_IOC_SLIPPAGE_BPS / 1e4
        pdp = agent._price_decimals
        ref_bid = best_bid if best_bid > 0 else mid
        ref_ask = best_ask if best_ask > 0 else mid
        if net > 0:
            trim = round(min(trim, agent._avail(account.base_balance)), vol_dp)
        else:
            buy_px = ref_ask * (1.0 + slip)
            q_max = agent._avail(account.quote_balance) / buy_px if buy_px > 0 else trim
            trim = round(min(trim, q_max), vol_dp)
        if trim < agent.exch_min:
            return False
        agent._cancel_all(response, account, book_id)
        if net > 0:
            agent._submit_limit(response, book_id, OrderDirection.SELL, trim,
                                round(ref_bid * (1.0 - slip), pdp), ioc=True, post_only=False)
        else:
            agent._submit_limit(response, book_id, OrderDirection.BUY, trim,
                                round(ref_ask * (1.0 + slip), pdp), ioc=True, post_only=False,
                                settlement=agent._loan_settlement(account))
        return True

    def _managed_exit(self, agent, response, validator, book_id, account, inv, net,
                      best_bid, best_ask, vol_dp, now) -> bool:
        if abs(net) < agent.exch_min:
            # Clear escalation state so a new position doesn't inherit a stale miss count.
            st = agent._bstate(validator, book_id)
            st.mk_ioc_miss_count = 0
            st.mk_ioc_prev_net = 0.0
            return False
        st = agent._bstate(validator, book_id)
        # Escalate slippage on consecutive IOC misses: a fixed-price IOC that doesn't cross on a
        # fast/wide book re-fires every step while the position bleeds. Escalating 4→8→18bps caps
        # the loss window. Final stage is a wide limit (not market) to bound catastrophic gap fills.
        if st.mk_ioc_prev_net > 0:
            if abs(net) >= st.mk_ioc_prev_net - agent._flat_eps:
                st.mk_ioc_miss_count += 1
            else:
                st.mk_ioc_miss_count = 0
                st.mk_ioc_prev_net = 0.0
        if st.mk_ioc_miss_count >= 4:
            slip = MK_IOC_CROSS_BPS / 1e4
        elif st.mk_ioc_miss_count >= 2:
            slip = MK_IOC_ESCALATE_BPS / 1e4
        else:
            slip = MK_IOC_SLIPPAGE_BPS / 1e4
        pdp = agent._price_decimals
        if net > 0:
            ts0, _, px0, _ = inv.longs[0]
            underwater = (px0 - best_bid) / px0 * 1e4 if px0 > 0 else 0.0
            timed_out = (now - ts0) >= agent.mk_max_hold_ns
            if underwater < MK_STOP_LOSS_BPS and not timed_out:   # hold for revert ONLY within BOTH the 15bps stop
                st.mk_ioc_miss_count = 0                          # AND the max-hold; else force-cut (never hold forever)
                st.mk_ioc_prev_net = 0.0
                return False
            reason = "time" if (timed_out and underwater < MK_STOP_LOSS_BPS) else "cut"
            q = round(min(agent._long_qty(inv), agent._avail(account.base_balance)), vol_dp)
            if q < agent.exch_min:
                st.mk_ioc_miss_count = 0
                st.mk_ioc_prev_net = 0.0
                return False
            st.mk_ioc_prev_net = abs(net)
            if st.mk_ioc_miss_count >= 2:
                label = "IOC-CROSS" if st.mk_ioc_miss_count >= 4 else "IOC-ESCALATE"
                bt.logging.info(
                    f"[AdaptiveRouterV2 uid={agent.uid}] {label} book={book_id} "
                    f"miss={st.mk_ioc_miss_count} slip={slip*1e4:.0f}bps"
                )
            self._tag_close(agent, validator, book_id, reason)
            agent._cancel_all(response, account, book_id)
            agent._submit_limit(response, book_id, OrderDirection.SELL, q,
                                round(best_bid * (1.0 - slip), pdp), ioc=True, post_only=False)
        else:
            ts0, _, px0, _ = inv.shorts[0]
            underwater = (best_ask - px0) / px0 * 1e4 if px0 > 0 else 0.0
            timed_out = (now - ts0) >= agent.mk_max_hold_ns
            if underwater < MK_STOP_LOSS_BPS and not timed_out:   # hold for revert ONLY within BOTH the 15bps stop
                st.mk_ioc_miss_count = 0                          # AND the max-hold; else force-cut (never hold forever)
                st.mk_ioc_prev_net = 0.0
                return False
            reason = "time" if (timed_out and underwater < MK_STOP_LOSS_BPS) else "cut"
            buy_px = best_ask * (1.0 + slip)
            q_max = agent._avail(account.quote_balance) / buy_px if buy_px > 0 else agent._short_qty(inv)
            q = round(min(agent._short_qty(inv), q_max), vol_dp)
            if q < agent.exch_min:
                st.mk_ioc_miss_count = 0
                st.mk_ioc_prev_net = 0.0
                return False
            st.mk_ioc_prev_net = abs(net)
            if st.mk_ioc_miss_count >= 2:
                label = "IOC-CROSS" if st.mk_ioc_miss_count >= 4 else "IOC-ESCALATE"
                bt.logging.info(
                    f"[AdaptiveRouterV2 uid={agent.uid}] {label} book={book_id} "
                    f"miss={st.mk_ioc_miss_count} slip={slip*1e4:.0f}bps"
                )
            self._tag_close(agent, validator, book_id, reason)
            agent._cancel_all(response, account, book_id)
            agent._submit_limit(response, book_id, OrderDirection.BUY, q,
                                round(best_ask * (1.0 + slip), pdp), ioc=True, post_only=False,
                                settlement=agent._loan_settlement(account))
        return True

    @staticmethod
    def _tag_close(agent, validator, book_id, reason) -> None:
        if agent._rt_log_enabled(validator):
            c = agent._rt_log.get((validator, book_id))
            if c is not None:
                c.close_reason = reason

    def _desired_quotes(self, agent, validator, book_id, account, inv, net,
                        best_bid, best_ask, mid, volume_cap, now, vol_dp, st) -> dict:
        maker_fee = agent._maker_fee_rate(account)
        pdp = agent._price_decimals
        desired: dict[int, tuple[float, float]] = {}
        fee_bps = (maker_fee * 1e4) if maker_fee is not None else 0.0
        floor_bps = MK_TP_FEE_MULT * fee_bps + (agent._tick / mid) * 1e4
        base_target = max(MK_TP_BPS, floor_bps) / 1e4

        free_base = account.base_balance.free if account.base_balance else 0.0
        free_quote = account.quote_balance.free if account.quote_balance else 0.0
        base_avail = agent._avail(account.base_balance)
        quote_avail = agent._avail(account.quote_balance)

        spread = best_ask - best_bid
        improve = agent._tick if spread > 2 * agent._tick else 0.0
        bid_inside = round(best_bid + improve, pdp)
        ask_inside = round(best_ask - improve, pdp)
        if bid_inside >= ask_inside:
            bid_inside, ask_inside = round(best_bid, pdp), round(best_ask, pdp)

        if net >= agent.exch_min:
            age = now - inv.longs[0][0]
            fifo_px = inv.longs[0][2]
            px = self._reduce_price(True, fifo_px, age, ask_inside, base_target, pdp, agent)
            q = round(min(agent._long_qty(inv), base_avail), vol_dp)
            if q >= agent.exch_min and px > 0:
                desired[OrderDirection.SELL] = (px, q)
        elif net <= -agent.exch_min:
            age = now - inv.shorts[0][0]
            fifo_px = inv.shorts[0][2]
            px = self._reduce_price(False, fifo_px, age, bid_inside, base_target, pdp, agent)
            q_max = quote_avail / px if px > 0 else agent._short_qty(inv)
            q = round(min(agent._short_qty(inv), q_max), vol_dp)
            if q >= agent.exch_min and px > 0:
                desired[OrderDirection.BUY] = (px, q)
        elif st.mk_streak_cooldown_until_ns and now < st.mk_streak_cooldown_until_ns:
            pass   # persistently toxic book (streak of losing cuts) -> long pause, stop quoting
        elif st.last_cut_ns > 0 and now - st.last_cut_ns < agent.mk_reentry_cooldown_ns:
            pass   # just cut a loser -> pause fresh entries so we don't re-bag a trend
        elif agent._budget_ok(validator, book_id, st, now, volume_cap):
            if st.mk_streak_cooldown_until_ns:    # streak cooldown elapsed -> clear and fresh start
                st.mk_loss_streak = 0
                st.mk_streak_cooldown_until_ns = 0
            q = round(agent.clip, vol_dp)
            if q >= agent.exch_min and free_base >= q:
                desired[OrderDirection.SELL] = (ask_inside, q)
            if q >= agent.exch_min and free_quote >= q * bid_inside:
                desired[OrderDirection.BUY] = (bid_inside, q)
        return desired

    def _reduce_price(self, is_long, px0, age_ns, touch_inside, base_target, pdp, agent) -> float:
        # V2 NEVER-CUT: walk the reduce limit from the profit target toward BREAKEVEN (px0) with lot age
        # — never to the touch, never below entry. A late passive fill nets ~0 (covers fees); a lot that
        # can't fill at breakeven is HELD for reversion and exits only via the stop IOC in
        # _managed_exit (uw >= MK_STOP_LOSS_BPS = 15bps). This is the core of the maker: the resting
        # reduce can never realize a loss; only a genuine ~15bps trend does.
        w = self._exit_walk(age_ns, agent)
        if is_long:
            target_px = max(touch_inside, px0 * (1.0 + base_target))   # sell at/above the target
            px = target_px + (px0 - target_px) * w                     # walk target -> breakeven
            return round(max(px, px0), pdp)                            # never sell below entry
        target_px = min(touch_inside, px0 * (1.0 - base_target))       # buy at/below the target
        px = target_px + (px0 - target_px) * w                         # walk target -> breakeven
        return round(min(px, px0), pdp)                                # never buy above entry

    @staticmethod
    def _exit_walk(age_ns, agent) -> float:
        if age_ns <= agent.mk_walk_start_ns:
            return 0.0
        if age_ns >= agent.mk_giveup_ns:
            return 1.0
        span = agent.mk_giveup_ns - agent.mk_walk_start_ns
        return (age_ns - agent.mk_walk_start_ns) / span if span > 0 else 1.0

    def _reconcile(self, agent, response, account, book_id, desired) -> None:
        resting = account.orders or []
        keep_sides: set[int] = set()
        cancel_ids: list[int] = []
        for o in resting:
            side = OrderDirection.BUY if o.side == 0 else OrderDirection.SELL
            want = desired.get(side)
            if (want is not None and side not in keep_sides and o.price is not None
                    and abs(o.price - want[0]) < agent._tick / 2
                    and abs((o.quantity or 0.0) - want[1]) < agent.exch_min):
                keep_sides.add(side)
            else:
                cancel_ids.append(o.id)
        if cancel_ids:
            response.cancel_orders(book_id, cancel_ids)
        for side, (px, qty) in desired.items():
            if side in keep_sides:
                continue
            if side == OrderDirection.BUY:
                agent._submit_limit(response, book_id, OrderDirection.BUY, qty, px, post_only=True)
            else:
                short_sale = agent._avail(account.base_balance) < qty
                agent._submit_limit(
                    response, book_id, OrderDirection.SELL, qty, px, post_only=True,
                    settlement=agent._loan_settlement(account) if short_sale else LoanSettlementOption.NONE)


if __name__ == "__main__":
    launch(AdaptiveRouterV2Agent)

