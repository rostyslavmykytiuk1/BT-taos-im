# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
AdaptiveRouterV2Agent — STABLE per-book maker/taker/idle router for subnet 79.

This is V2 of AdaptiveRouter. The live `AdaptiveRouterAgent.py` (V1) stays untouched as the A/B
control. V2 keeps V1's proven skeleton (shared FIFO `_match_fifo`, validator-faithful kappa-3,
per-sim reset, onTrade maker+taker routing) and upgrades three things, per ADAPTIVE_ROUTER_DEVPLAN.md:

  1) THE ROUTER IS SLUGGISH AND STICKY (the whole point of V2). V1 flipped modes on spread NOISE
     (~161 emergency-flips/slice, 40% at <6bps). Kappa scores each book over a ~3-SIM-HOUR window
     (≈20-24 real h) and pools ALL of a book's fills into one per-book Sortino series with zero
     mode-awareness — so a flip contaminates the scored kappa for ~a full window. V2 therefore:
       * routes on a TIME-DECAYED EMA of the spread (~5 sim-min half-life), not the raw tick;
       * uses SIM-TIME dwells matched to the kappa window (base ~30 sim-min, MAKER sticky ~50);
       * makes the dwell-bypassing EMERGENCY the FEE axis ONLY (a taker-rebate sign inversion,
         ~3 sim-min confirm) — the spread/maker-edge emergency is REMOVED;
       * adds a reverse-flip cooldown (~45 sim-min, fee-bypassable) and a per-book flip budget
         (~4 / window) to kill ping-pong on BOTH axes (maker<->taker AND the now-dominant maker<->idle).
     ALL routing time-constants are SIM-TIME (against state.timestamp), because the thing being
     protected — the kappa window — is sim-time. EXECUTION gates (per-trade open/exit) read the RAW tick.

  2) THE TAKER LEG IS TakerScalperV4 (into `_TakerMode`). Single-open (no pyramiding), market open/
     close, gate = rebate>=2bps AND est_pnl>0 (rebate beats the raw spread), TP 2.5 / hold 1.5-3s,
     SL = a TOGGLE (`tk_sl_bps`, default the tight ~2bps calm-regime cut; set 12 for fee-adverse).
     No internal sleep / no force-activity on flat books (the router owns idling).

  3) THE MAKER LEG IS PureMakerV4 (into `_MakerMode`). Reduce-walk to BREAKEVEN (not the touch),
     vol-scaled stop band 10-14bps (per-book mid-noise EWMA), 150s giveup, 1.5-lot cap, 1.5-tick
     reprice cushion. Held-only activity backstop; flat books idle at kappa=None (free-dropped).

IDLE = a thin both-axes-negative floor + the reactive PnL-backoff, capped at MAX_IDLE_BOOKS (<48 so
kappa=None stays free). No force-activity on flat books; idle only drains residual inventory.

LOT is UNIFIED to the shared clip (0.26) across legs (V4 standalone uses 0.30) so a book's realized-RT
series keeps a consistent scale across mode switches — kappa is per-book MAD-normalized, but mixing
two lot scales in one series adds avoidable noise to the MAD. The trading LOGIC matches V4/PMV4; only
the lot is the shared clip.

Score = 0.79·kappa + 0.21·pnl, per-book then median. kappa-3 CUBES the downside (LPM3), so the rules
everywhere are: bounded-small loss, consistency over magnitude, and — above all — DO NOT CHURN MODES.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol import FinanceAgentResponse, MarketSimulationStateUpdate
from taos.im.protocol.events import OrderPlacementEvent, TradeEvent
from taos.im.protocol.models import (
    LoanSettlementOption,
    OrderCurrency,
    OrderDirection,
    STP,
    TimeInForce,
)

_NS = 1_000_000_000

# ======================================================================== config
# Which modes the router may use. {"taker"} => pure taker; {"maker"} => pure maker; full set => the
# adaptive per-book router (the default).
ALLOWED_MODES = {"taker", "maker", "idle"}

# ---- shared sizing / precision ----
EXCHANGE_MIN_ORDER_SIZE = 0.25     # sim minOrderSize floor for any BASE order
TARGET_CLIP = 0.26                 # shared per-clip BASE lot, both legs. Just ABOVE 0.25 on purpose:
                                   # a fee-paying BUY is settled by shaving the fee out of the base
                                   # received, so a 0.25 buy leaves ~0.2498 (< min, un-sellable). 0.26
                                   # keeps the held lot >= 0.25 after the shave => always closeable.

# ---- routing thresholds (bps) on the SMOOTHED spread, with hysteresis ----
TAKER_REBATE_ENTER_BPS = 2.0       # route to taker when rebate >= 2bps (aligns with the V4 _open
                                   # floor KAPPA_MIN_REBATE_BPS=2; the _open gate independently checks
                                   # est_pnl>0 on the RAW spread)
TAKER_REBATE_EXIT_BPS = 0.75       # leave taker when rebate falls below 0.75bps (hysteresis)
MAKER_EDGE_ENTER_BPS = 1.5         # enter maker when smoothed half_spread - maker_fee >= 1.5bps
MAKER_EDGE_EXIT_BPS = -0.5         # leave maker only when edge < -0.5bps (2bps sticky band)
MAKER_MAX_FEE_BPS = 1_000.0        # effectively no ceiling — a wide spread redeems a high fee
MAKER_FALLBACK_EDGE_BPS = 0.5      # when the idle guard fires, promote books with >= 0.5bps edge
MAX_IDLE_BOOKS = 40                # promote borderline books to maker past this (< the 48-free budget)

# ---- routing cadence (SIM-TIME; fractions of the kappa window) ----
# The kappa window and the regime cadence are the SAME timescale (~3 sim-h ≈ 20-24 real-h ≈ ~1 regime/
# day). So a flip is expensive for ~a full window, not 180s -> route SLOWLY. Sim advances ~8x slower
# than wall-clock, so these are ~8x larger in real time (a 30-sim-min dwell ≈ ~4 real-h). Execution
# stays on the raw tick; only ROUTING is slowed.
ROUTE_SPREAD_EMA_HALFLIFE_S = 300.0    # ~5 sim-min: filter sub-minute noise, track the hours regime
ROUTE_MIN_DWELL_S = 1800.0             # ~30 sim-min: min time in a mode before any non-fee flip
MAKER_STICKY_DWELL_S = 3000.0          # ~50 sim-min: maker is home, hold through intra-hour wiggle
SPREAD_FLIP_PERSIST_S = 900.0          # ~15 sim-min: a spread-driven candidate must hold this long
FEE_CONFIRM_S = 180.0                  # ~3 sim-min: a fee-axis inversion must hold this long (then it
                                       # bypasses dwell/persistence/cooldown)
REVERSE_FLIP_COOLDOWN_S = 2700.0       # ~45 sim-min: block an immediate A->B->A reverse (fee-bypassed)
FLIP_BUDGET = 4                        # max committed flips per book per kappa window (anti-thrash)
EMERGENCY_TAKER_EXIT_BPS = 0.0         # fee-axis emergency: a taker book whose rebate falls below this
                                       # (taker now PAYS) leaves fast (after FEE_CONFIRM)

# ---- per-book reactive PnL backoff (kept from V1) ----
PNL_BACKOFF_WINDOW_S = 600.0       # rolling window (~10 min)
PNL_BACKOFF_COOLDOWN_S = 660.0     # idle this long after a trigger (must exceed WINDOW)
PNL_BACKOFF_MIN_RTS = 5            # require >= 5 RTs in the window to fire

# ---- shared round-trip economics / risk ----
RT_WINDOW_S = 570.0                # validator activity sampling window (~10 min)
CAPITAL_TURNOVER_CAP = 10.0
VOLUME_SAFETY = 0.8
VOLUME_ASSESSMENT_NS = 86_400_000_000_000
ACTIVITY_DEADLINE_S = 500.0        # HELD-only deep safety: force-close a position that outlived the
                                   # window (flat books are NEVER force-traded; they idle at None)
PENDING_TIMEOUT_S = 5.0            # taker: after a market open/close, wait this long for the fill before
                                   # assuming it was lost and re-deriving from net (rule 3, = V4)

# forced-exit SLIP ceiling for the maker held-safety / idle-drain concession.
MK_RT_LOSS_CAP_BPS = 5.0

# ---- TAKER mode (= TakerScalperV4) ----
# NB: the taker opens the shared clip (0.26), scale-matched across modes (V4 standalone uses 0.30).
TK_MIN_HOLD_S = 1.5
TK_MAX_HOLD_S = 3.0
TK_TP_BPS = 2.5                    # MIN_GROSS_TP_BPS (>= SL so per-RT gross skew is non-negative)
TK_SL_BPS = 2.0                    # DEFAULT tight cut (calm-regime base case, = proven V3/uid192).
                                   # A config TOGGLE `tk_sl_bps` overrides; set 12 for fee-adverse
                                   # (no-cut). NOT hardcoded — cut-vs-no-cut is regime-dependent.
TK_REOPEN_GAP_S = 2.0             # min gap between a close and the next profit open (MIN_REOPEN_GAP)
TK_REBATE_FLOOR_BPS = 2.0         # open ONLY when rebate >= this (= V4 KAPPA_MIN_REBATE_BPS)
TK_RT_MAX = 40                    # max profit RTs per book per window (V4)

# ---- MAKER mode (= PureMakerV4) ----
MK_TP_BPS = 8.0                   # base target (V4 TP_BPS_BASE; moot under the fee floor)
MK_TP_FEE_MULT = 2.0             # floor = 2x maker_fee + a tick (covers both legs)
MK_QUOTE_EXPIRY_S = 12.0
MK_EXIT_WALK_START_S = 30.0       # start walking reduce target->breakeven after 30s
MK_EXIT_GIVEUP_S = 150.0         # tight time-cut at ~2.5min (V4; V1/AR used 90s)
MK_STOP_FLOOR_BPS = 10.0         # vol-scaled stop band FLOOR (calm-book default)
MK_STOP_CAP_BPS = 14.0           # vol-scaled stop band CAP (volatile-book bound)
MK_STOP_NOISE_MULT = 6.0         # stop ≈ MULT x per-book mid-noise(bps), clamped to the band
MK_NOISE_EWMA_ALPHA = 0.05       # per-step EWMA weight for the mid-noise estimate (~20-step memory)
MK_IOC_SLIPPAGE_BPS = 4.0        # managed-exit IOC concession
MK_IOC_ESCALATE_BPS = 8.0        # after 2+ consecutive IOC misses
MK_IOC_CROSS_BPS = 18.0          # wide-limit cross after 4+ misses (bounds gap fills; not a market)
MK_RISK_TRIM_SLIPPAGE_BPS = 6.0  # inventory-cap risk-trim concession (V4)
MK_REENTRY_COOLDOWN_S = 120.0    # pause fresh entries after a managed cut
MK_MAX_INVENTORY_LOTS = 1.5      # small cap (V4; V1/AR used 2.0) — bounds each forced cut
MK_MAX_INVENTORY_EQUITY_FRAC = 0.10
MK_RT_MAX = 15                   # max RTs per book per window (V4; anti-overtrade)
REPRICE_KEEP_TICKS = 1.5         # keep a resting quote unless the desired price moved > 1.5 ticks

# ---- Kappa-3 (validator-faithful; 3h history) ----
KAPPA_TAU = 0.0
KAPPA_MIN_OBS = 3
KAPPA_MIN_LOOKBACK_S = 5400.0     # 90 min
KAPPA_RT_HISTORY_S = 10_800.0     # 3h (the routing-cadence anchor)

# ---- census (observability) ----
CENSUS_THROTTLE_S = 60.0          # wall-clock throttle for the per-validator mode census

# RT logs / census only for the scoring validator.
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
    # routing FSM
    mode: str = ""                     # "" = never-classified (set on first sight). NOT MODE_IDLE, so
                                       # the idle hard-cap's "don't touch an already-idle book" guard
                                       # (cur != MODE_IDLE) does not wrongly fire on a fresh book.
    mode_since_ns: int = 0              # when the current mode was committed (dwell clock)
    route_candidate: str = ""          # the mode we are currently considering switching TO
    candidate_since_ns: int = 0        # when route_candidate was first proposed (persistence clock)
    last_flip_ns: int = 0              # last committed flip (reverse-cooldown clock)
    last_flip_from: str = ""           # the mode we flipped OUT OF last (reverse detection)
    flip_count: int = 0                # committed flips in the current budget window
    flip_window_start_ns: int = 0      # start of the current flip-budget window
    # spread EMA (routing) + mid-noise EWMA (maker vol-stop)
    ema_val: float = 0.0               # smoothed half_spread (bps)
    ema_last_ns: int = 0
    ema_start_ns: int = 0              # first EMA sample (warm-up clock)
    last_mid: float = 0.0
    noise_bps: float = 0.0
    # activity / kappa
    last_rt_ns: int = 0
    last_cut_ns: int = 0               # last forced (managed-exit) cut; gates re-entry cooldown
    seen_ns: int = 0
    rt_events: list[tuple[int, float]] = field(default_factory=list)    # (sim_ts, realized_pnl)
    kappa_events: list[tuple[int, float]] = field(default_factory=list)  # (wall_ts, realized_pnl)
    kappa3: float | None = None
    vol_log: list[tuple[int, float]] = field(default_factory=list)
    # taker bookkeeping
    last_close_ns: int = 0             # last taker close (reopen throttle)
    pending_ns: int = 0                # taker: a market open/close is in flight (rule 3, single-open guard)
    # maker managed-exit IOC escalation
    exit_miss_count: int = 0
    exit_prev_net: float = 0.0
    # per-book PnL backoff
    pnl_backoff_until_ns: int = 0


@dataclass
class _RtLogCtx:
    mode: str = "?"
    open_reason: str = "?"
    side: str = "?"
    kappa_at_open: float | None = None
    close_reason: str = "fill"


@dataclass
class _Census:
    """Per-validator routing counters; reset per sim and after each emit."""
    route_flips: int = 0
    emergency_flips: int = 0
    pnl_backoffs: int = 0
    last_emit_wall_ns: int = 0


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

        # --- config toggles (baked into AGENT_PARAMS at launch; default = the V2/final behavior) ---
        self.tk_sl_bps = self._cfg_float("tk_sl_bps", TK_SL_BPS)        # taker cut width (toggle)
        # Taker FORCE-ACTIVITY (= V4's activity backstop): keep a ROUTED taker book dense/scored with a
        # rebate-funded RT when natural profit-opens go quiet. DISTINCT from idle (which is "no edge ->
        # don't trade"). ON by default (breadth wins for rebate-funded takers); A/B-able. Maker has NO
        # flat force-activity (maker RTs pay the fee) — the rebate-vs-fee asymmetry, plan §5.4.
        self.tk_force_activity = self._cfg_float("tk_force_activity", 1.0) > 0
        # Hard idle cap: never let >MAX_IDLE_BOOKS books idle at once (a NEW idle is redirected to the
        # least-bad trade axis; an already-idle book is never evicted — slots free only on edge
        # recovery). Protects the kappa median from the >48-idle 0.0-crater. Toggle (default ON).
        self.idle_hard_cap = self._cfg_float("idle_hard_cap", 1.0) > 0
        cad = self._cfg_float("route_cadence_mult", 1.0)               # scale ALL sim-time route timers
        self.route_cadence_mult = cad if cad > 0 else 1.0

        # Per-UID jitter so a fleet does not act in lockstep.
        jitter = ((self.uid * 2654435761) % 1000) / 1000.0
        cj = self.route_cadence_mult
        self.ema_tau_ns = int(ROUTE_SPREAD_EMA_HALFLIFE_S / math.log(2.0) * _NS)
        self.ema_halflife_ns = int(ROUTE_SPREAD_EMA_HALFLIFE_S * _NS)
        self.route_min_dwell_ns = int(ROUTE_MIN_DWELL_S * cj * (0.9 + 0.2 * jitter) * _NS)
        self.maker_sticky_dwell_ns = int(MAKER_STICKY_DWELL_S * cj * (0.9 + 0.2 * jitter) * _NS)
        self.spread_persist_ns = int(SPREAD_FLIP_PERSIST_S * cj * (0.9 + 0.2 * jitter) * _NS)
        self.fee_confirm_ns = int(FEE_CONFIRM_S * cj * (0.9 + 0.2 * jitter) * _NS)
        self.reverse_cooldown_ns = int(REVERSE_FLIP_COOLDOWN_S * cj * (0.9 + 0.2 * jitter) * _NS)
        # NOT scaled by route_cadence_mult: the flip budget is per KAPPA WINDOW (a fixed external
        # reference = the validator's scoring lookback), unlike the tunable dwell/persist timers.
        self.flip_window_ns = int(KAPPA_RT_HISTORY_S * _NS)

        self.activity_deadline_ns = int(ACTIVITY_DEADLINE_S * (0.92 + 0.08 * jitter) * _NS)
        self.pending_timeout_ns = int(PENDING_TIMEOUT_S * _NS)
        self.rt_window_ns = int(RT_WINDOW_S * _NS)
        self.tk_max_hold_ns = int(TK_MAX_HOLD_S * (0.92 + 0.16 * jitter) * _NS)
        self.tk_min_hold_ns = int(TK_MIN_HOLD_S * _NS)
        self.tk_reopen_gap_ns = int(TK_REOPEN_GAP_S * (0.9 + 0.2 * jitter) * _NS)
        self.mk_quote_expiry_ns = int(MK_QUOTE_EXPIRY_S * _NS)
        self.mk_walk_start_ns = int(MK_EXIT_WALK_START_S * _NS)
        self.mk_giveup_ns = int(MK_EXIT_GIVEUP_S * (0.9 + 0.2 * jitter) * _NS)
        self.mk_reentry_cooldown_ns = int(MK_REENTRY_COOLDOWN_S * _NS)
        self.kappa_rt_history_ns = int(KAPPA_RT_HISTORY_S * _NS)
        self.kappa_min_lookback_ns = int(KAPPA_MIN_LOOKBACK_S * _NS)
        self.pnl_backoff_window_ns = int(PNL_BACKOFF_WINDOW_S * _NS)
        self.pnl_backoff_cooldown_ns = int(PNL_BACKOFF_COOLDOWN_S * _NS)
        self.census_throttle_ns = int(CENSUS_THROTTLE_S * _NS)

        self.default_mode = (
            MODE_MAKER if MODE_MAKER in ALLOWED_MODES
            else MODE_TAKER if MODE_TAKER in ALLOWED_MODES else MODE_IDLE
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
        self._census: dict[str, _Census] = {}
        self._idle_live: dict[str, int] = {}   # live idle-occupancy counter (per validator)
        self._active_validator: str | None = None

        bt.logging.info(
            f"[AdaptiveRouterV2 uid={self.uid}] modes={sorted(ALLOWED_MODES)} default={self.default_mode} "
            f"clip={TARGET_CLIP} ROUTE(SIM-time x{cad:.2f}): ema_hl={ROUTE_SPREAD_EMA_HALFLIFE_S:.0f}s "
            f"dwell={ROUTE_MIN_DWELL_S:.0f}s maker_sticky={MAKER_STICKY_DWELL_S:.0f}s "
            f"persist={SPREAD_FLIP_PERSIST_S:.0f}s fee_confirm={FEE_CONFIRM_S:.0f}s "
            f"reverse_cd={REVERSE_FLIP_COOLDOWN_S:.0f}s flip_budget={FLIP_BUDGET}/window "
            f"route(taker_rebate>={TAKER_REBATE_ENTER_BPS}/{TAKER_REBATE_EXIT_BPS}bps, "
            f"maker_edge>={MAKER_EDGE_ENTER_BPS}/{MAKER_EDGE_EXIT_BPS}bps) "
            f"taker(tp={TK_TP_BPS} sl={self.tk_sl_bps}bps[toggle] hold={TK_MIN_HOLD_S}-{TK_MAX_HOLD_S}s "
            f"rebate_floor={TK_REBATE_FLOOR_BPS} single-open force_activity={'on' if self.tk_force_activity else 'off'}) "
            f"maker(giveup={MK_EXIT_GIVEUP_S:.0f}s stop=[{MK_STOP_FLOOR_BPS:.0f},{MK_STOP_CAP_BPS:.0f}]bps "
            f"walk->breakeven cap={MK_MAX_INVENTORY_LOTS}lot reprice={REPRICE_KEEP_TICKS}t) "
            f"idle(hard_cap={MAX_IDLE_BOOKS}{'' if self.idle_hard_cap else '(off)'} recovery-only no-force-activity) "
            f"rt_log={MAIN_VALIDATOR[:8]}"
        )

    def _cfg_float(self, name: str, default: float) -> float:
        try:
            return float(getattr(self.config, name, default))
        except (TypeError, ValueError):
            return default

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
        self._census.pop(validator, None)
        self._idle_live.pop(validator, None)
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

        idle_count = sum(
            1 for bst in (self.books_state.get(validator) or {}).values()
            if bst.mode == MODE_IDLE
        )
        # Seed the live idle counter from a TRUE count once per step (self-heals any drift); it is then
        # kept live at the _record_flip commit choke-point so the hard cap holds WITHIN a step too
        # (the old start-of-step snapshot let many books idle in one pass and overshoot the cap).
        self._idle_live[validator] = idle_count
        # Soft relief fires just BELOW the hard cap so borderline books prefer maker before any redirect.
        fallback_maker = idle_count > MAX_IDLE_BOOKS - 2

        for book_id in sorted(self.accounts.keys()):
            book = state.books.get(book_id)
            account = self.accounts.get(book_id) if book else None
            if book is None or account is None:
                continue
            try:
                self._step_book(response, validator, book_id, book, account,
                                vol_dp, volume_cap, now, fallback_maker)
            except Exception as ex:
                bt.logging.warning(f"[AdaptiveRouterV2 uid={self.uid}] step {book_id}: {ex}")

        self._maybe_census(validator)
        return response

    # ------------------------------------------------------------------ per-book dispatch
    def _step_book(
        self, response, validator: str, book_id: int, book, account,
        vol_dp: int, volume_cap: float, now: int, fallback_maker: bool = False,
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
        half_spread_bps = (best_ask - best_bid) / mid * 0.5 * 1e4

        # --- update the routing EMA + the maker mid-noise EWMA EVERY step (NOT only when flat), so
        #     timers/signals are fresh the instant a long-held book flattens (devplan §5.6c). ---
        self._update_spread_ema(st, half_spread_bps, now)
        if st.last_mid > 0.0:
            inst = abs(mid - st.last_mid) / st.last_mid * 1e4
            st.noise_bps = (((1.0 - MK_NOISE_EWMA_ALPHA) * st.noise_bps + MK_NOISE_EWMA_ALPHA * inst)
                            if st.noise_bps > 0.0 else inst)
        st.last_mid = mid

        # --- cold-start seeding: classify the book ONCE on first sight (instantaneous), seed every
        #     ns timer to `now` so no "now - 0 >= X" fires spuriously after a restart (devplan §5.6c). ---
        if st.seen_ns == 0:
            st.seen_ns = now
            idle_full = self.idle_hard_cap and self._idle_live.get(validator, 0) >= MAX_IDLE_BOOKS
            st.mode = self._route_decision(st, account, half_spread_bps, mid, fallback_maker, idle_full)
            st.mode_since_ns = now
            st.route_candidate = st.mode
            st.candidate_since_ns = now
            st.flip_window_start_ns = now
            if st.mode == MODE_IDLE:      # count the first commit toward the live idle occupancy
                self._idle_live[validator] = self._idle_live.get(validator, 0) + 1

        if self._prune_rt_events(st, now):
            self._refresh_book_kappa(validator, book_id, time.time_ns())

        net = self._net_qty(inv)
        flat = abs(net) < self.exch_min

        # Route only when FLAT (no position straddles two modes) — but the timers above already
        # advanced every step. A committed switch first cancels resting orders, then commits next step.
        if flat:
            self._maybe_reroute(response, validator, book_id, account, st,
                                best_bid, best_ask, mid, now, fallback_maker)

        mode = self._modes.get(st.mode) or self._modes[self.default_mode]
        mode.step(self, response, validator, book_id, book, account, st, inv, net,
                  best_bid, best_ask, mid, vol_dp, volume_cap, now)

    # ------------------------------------------------------------------ routing FSM
    def _update_spread_ema(self, st: _BookState, half_spread_bps: float, now: int) -> float:
        """Time-decayed EMA so a variable sim-Δt yields a stable half-life (devplan §5.6d)."""
        if st.ema_last_ns == 0:
            st.ema_val = half_spread_bps
            st.ema_last_ns = now
            st.ema_start_ns = now
            return st.ema_val
        dt = now - st.ema_last_ns
        st.ema_last_ns = now
        if dt <= 0:
            return st.ema_val
        alpha = 1.0 - math.exp(-dt / self.ema_tau_ns)
        st.ema_val += alpha * (half_spread_bps - st.ema_val)
        return st.ema_val

    def _maybe_reroute(self, response, validator, book_id, account, st,
                       best_bid, best_ask, mid, now, fallback_maker) -> None:
        """Decide and (gated) commit a mode switch on a FLAT book. Order: PnL-backoff safety ->
        candidate tracking -> fee-axis emergency (bypass) / dwell+persistence+cooldown+budget."""
        # (1) PnL backoff (safety): a bleeding book is held at idle for a cooldown. Bypasses dwell.
        if self._pnl_backoff_check(st, now):
            if st.mode != MODE_IDLE:
                if account.orders:
                    self._cancel_all(response, account, book_id)
                    return
                bt.logging.info(f"[AdaptiveRouterV2 uid={self.uid}] PNL-BACKOFF {st.mode}->idle book={book_id}")
                self._record_flip(validator, st, MODE_IDLE, now, emergency=False, backoff=True)
            return

        idle_full = self.idle_hard_cap and self._idle_live.get(validator, 0) >= MAX_IDLE_BOOKS
        want = self._route_decision(st, account, st.ema_val, mid, fallback_maker, idle_full)

        # (2) candidate tracking — the persistence clock arms while a non-current candidate holds.
        if want == st.mode:
            st.route_candidate = st.mode
            st.candidate_since_ns = now
            return
        if st.route_candidate != want:
            st.route_candidate = want
            st.candidate_since_ns = now

        # (3) fee-axis emergency: a taker-rebate SIGN inversion (the genuine regime signal), confirmed
        #     for ~FEE_CONFIRM, bypasses dwell/persistence/cooldown/warmup. The spread/maker-edge
        #     emergency of V1 is REMOVED.
        taker_fee = self._taker_fee_rate(account)
        rebate_bps = (-taker_fee * 1e4) if taker_fee is not None else -1e9
        fee_emergency = False
        if st.mode == MODE_TAKER and want != MODE_TAKER and rebate_bps < EMERGENCY_TAKER_EXIT_BPS:
            fee_emergency = True   # taker now PAYS -> leave fast
        elif st.mode != MODE_TAKER and want == MODE_TAKER and rebate_bps >= TAKER_REBATE_ENTER_BPS:
            fee_emergency = True   # a real rebate appeared -> enter fast
        if fee_emergency and (now - st.candidate_since_ns) < self.fee_confirm_ns:
            fee_emergency = False  # not yet confirmed

        if fee_emergency:
            allow = True
        else:
            warmup_ok = (now - st.ema_start_ns) >= self.ema_halflife_ns
            dwell_need = self.maker_sticky_dwell_ns if st.mode == MODE_MAKER else self.route_min_dwell_ns
            dwell_ok = (now - st.mode_since_ns) >= dwell_need
            persist_ok = (now - st.candidate_since_ns) >= self.spread_persist_ns
            reverse_blocked = (
                st.last_flip_ns > 0
                and want == st.last_flip_from
                and (now - st.last_flip_ns) < self.reverse_cooldown_ns
            )
            budget_ok = self._flip_budget_ok(st, now)
            allow = warmup_ok and dwell_ok and persist_ok and (not reverse_blocked) and budget_ok

        if not allow:
            return
        if account.orders:
            self._cancel_all(response, account, book_id)
            return   # cancel first; commit next step (no position straddles two modes)
        bt.logging.info(
            f"[AdaptiveRouterV2 uid={self.uid}] {'FEE-EMERGENCY' if fee_emergency else 'ROUTE'} "
            f"{st.mode}->{want} book={book_id} rebate={rebate_bps:.1f}bps "
            f"ema_half_spread={st.ema_val:.1f}bps maker_fee={self._maker_fee_rate(account)}"
            + (" [fallback-maker]" if fallback_maker else "")
        )
        self._record_flip(validator, st, want, now, emergency=fee_emergency, backoff=False)

    def _route_decision(self, st: _BookState, account, half_spread_bps: float, mid: float,
                        fallback_maker: bool, idle_full: bool = False) -> str:
        """Pick the +EV playbook from the (smoothed) fee regime, with hysteresis. Identical shape to
        V1's _route but fed the SMOOTHED half-spread and with no execution side-effects."""
        cur = st.mode
        taker_fee = self._taker_fee_rate(account)
        maker_fee = self._maker_fee_rate(account)
        rebate_bps = (-taker_fee * 1e4) if taker_fee is not None else -1e9
        maker_fee_bps = (maker_fee * 1e4) if maker_fee is not None else 1e9
        maker_edge_bps = half_spread_bps - maker_fee_bps

        taker_min_rebate = TAKER_REBATE_EXIT_BPS if cur == MODE_TAKER else TAKER_REBATE_ENTER_BPS
        maker_enter_edge = MAKER_FALLBACK_EDGE_BPS if fallback_maker else MAKER_EDGE_ENTER_BPS
        maker_min_edge = MAKER_EDGE_EXIT_BPS if cur == MODE_MAKER else maker_enter_edge
        # taker viable only when the rebate covers the (smoothed) crossing cost; books already in
        # taker skip the spread check (the raw-tick _open gate suppresses trading when too wide).
        spread_viable = (cur == MODE_TAKER) or (rebate_bps > half_spread_bps)
        taker_ok = (MODE_TAKER in ALLOWED_MODES) and rebate_bps >= taker_min_rebate and spread_viable
        maker_ok = ((MODE_MAKER in ALLOWED_MODES)
                    and maker_fee_bps < MAKER_MAX_FEE_BPS
                    and maker_edge_bps >= maker_min_edge)

        # MAKER is the sticky home: when both edges exist, prefer maker unless taker's margin above
        # its bar clearly exceeds maker's (a 1bps cushion biases toward staying maker).
        if taker_ok and maker_ok:
            taker_margin = rebate_bps - TAKER_REBATE_ENTER_BPS
            maker_margin = maker_edge_bps - maker_enter_edge
            return MODE_TAKER if taker_margin >= maker_margin + 1.0 else MODE_MAKER
        if maker_ok:
            return MODE_MAKER
        if taker_ok:
            return MODE_TAKER
        if MODE_IDLE in ALLOWED_MODES:
            # HARD CAP: a book ALREADY idle is never touched (recovery-only — a slot frees ONLY when an
            # idle book's edge genuinely recovers above, never by eviction). But do NOT add a NEW idle
            # book once the idle set is full — trade the LEAST-BAD axis instead (tie -> taker, which is
            # rebate-funded + SL-bounded; avoids manufacturing a fee-paying maker RT on a dead book).
            # Keeps idle <= MAX_IDLE_BOOKS so kappa=None stays free (>48 idle injects 0.0 into the median).
            if idle_full and cur != MODE_IDLE:
                # Prefer MAKER: a passive quote on a dead book RESTS and may never fill (~free), whereas
                # a redirected taker would FORCE-cross every ~500s and PAY the fee on a no-rebate book —
                # the exact cost the cap exists to avoid. Use taker only when its crossing is at least
                # rebate-free (rebate>=0) AND beats the maker on margin (breadth without paying to cross).
                taker_margin = rebate_bps - TAKER_REBATE_ENTER_BPS
                maker_margin = maker_edge_bps - maker_enter_edge
                taker_redirect_ok = (rebate_bps >= 0.0 and taker_margin >= maker_margin
                                     and MODE_TAKER in ALLOWED_MODES)
                if taker_redirect_ok:
                    return MODE_TAKER
                if MODE_MAKER in ALLOWED_MODES:
                    return MODE_MAKER
                return MODE_TAKER if MODE_TAKER in ALLOWED_MODES else MODE_IDLE
            return MODE_IDLE
        return cur if cur in ALLOWED_MODES else self.default_mode

    def _flip_budget_ok(self, st: _BookState, now: int) -> bool:
        # READ-ONLY check: an expired/unstarted window means the budget is fresh. The actual window
        # roll + count increment happen ONLY at the commit choke-point (_record_flip), so this check
        # has no side effects (was a redundant double-roll before).
        if st.flip_window_start_ns == 0 or (now - st.flip_window_start_ns) >= self.flip_window_ns:
            return True
        return st.flip_count < FLIP_BUDGET

    def _record_flip(self, validator: str, st: _BookState, want: str, now: int,
                     *, emergency: bool, backoff: bool) -> None:
        # Maintain the live idle-occupancy counter at this single commit choke-point (uses the OLD
        # st.mode, before the reassignment below). +1 entering idle, -1 leaving it.
        if st.mode != MODE_IDLE and want == MODE_IDLE:
            self._idle_live[validator] = self._idle_live.get(validator, 0) + 1
        elif st.mode == MODE_IDLE and want != MODE_IDLE:
            self._idle_live[validator] = max(0, self._idle_live.get(validator, 0) - 1)
        st.last_flip_from = st.mode
        st.last_flip_ns = now
        if st.flip_window_start_ns == 0 or (now - st.flip_window_start_ns) >= self.flip_window_ns:
            st.flip_window_start_ns = now
            st.flip_count = 0
        st.flip_count += 1
        st.mode = want
        st.mode_since_ns = now
        st.route_candidate = want
        st.candidate_since_ns = now
        c = self._census.setdefault(validator, _Census())
        if backoff:
            c.pnl_backoffs += 1
        elif emergency:
            c.emergency_flips += 1
        else:
            c.route_flips += 1

    # ------------------------------------------------------------------ census
    def _maybe_census(self, validator: str) -> None:
        if not self._rt_log_enabled(validator):
            return
        c = self._census.setdefault(validator, _Census())
        wall = time.time_ns()
        if c.last_emit_wall_ns and (wall - c.last_emit_wall_ns) < self.census_throttle_ns:
            return
        c.last_emit_wall_ns = wall
        states = self.books_state.get(validator, {})
        taker = sum(1 for s in states.values() if s.mode == MODE_TAKER)
        maker = sum(1 for s in states.values() if s.mode == MODE_MAKER)
        idle = sum(1 for s in states.values() if s.mode == MODE_IDLE)
        total = len(self.accounts) if self.accounts else (taker + maker + idle)
        unseen = max(0, total - (taker + maker + idle))
        kappas = [s.kappa3 for s in states.values() if s.kappa3 is not None]
        med = self._median(kappas) if kappas else None
        bt.logging.info(
            f"[AdaptiveRouterV2 uid={self.uid} CENSUS] taker={taker} maker={maker} idle={idle} "
            f"unseen={unseen} sum={taker+maker+idle+unseen}/{total} "
            f"idle_cap={MAX_IDLE_BOOKS}{'' if self.idle_hard_cap else '(off)'} idle_live={self._idle_live.get(validator, 0)} "
            f"flips={c.route_flips} fee_emerg={c.emergency_flips} backoffs={c.pnl_backoffs} "
            f"scored={len(kappas)} median_kappa={'n/a' if med is None else f'{med:.4f}'}"
        )
        c.route_flips = c.emergency_flips = c.pnl_backoffs = 0

    # ------------------------------------------------------------------ events
    def onOrderRejected(self, event: OrderPlacementEvent) -> None:
        """A rejected taker order never filled — clear the in-flight lock immediately so the book
        re-derives from net next step (instead of waiting out pending_timeout), and drop its stale
        open stash. REQUIRED companion to the pending_ns guard, else a reject wedges the book ~5s."""
        if event.bookId is None or not self._active_validator:
            return
        st = self._bstate(self._active_validator, event.bookId)
        st.pending_ns = 0
        self._rt_log.pop((self._active_validator, event.bookId), None)

    def onTrade(self, event: TradeEvent, validator: str | None = None) -> None:
        """Route maker AND taker fills into the shared FIFO."""
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
        self._bstate(validator, book_id).pending_ns = 0   # a fill resolved any in-flight taker order
        realized, rtv, matched_ts, gross = self._match_fifo(inv, is_buy, qty, price, fee, ts)
        if rtv > 0:
            st = self._bstate(validator, book_id)
            kappa_before = st.kappa3
            st.last_rt_ns = ts
            self._record_rt_close(validator, book_id, ts, realized)
            self._log_rt(validator, book_id, ts,
                         hold_s=(ts - matched_ts) / _NS if matched_ts else None,
                         side=("buy" if is_buy else "sell"), exit_px=price, rtv=rtv,
                         gross=gross, net=realized, kappa_before=kappa_before, kappa_after=st.kappa3)
        elif rtv == 0 and self._rt_log_enabled(validator) and (validator, book_id) not in self._rt_log:
            st = self._bstate(validator, book_id)
            self._stash_open(validator, book_id, st, st.mode, "passive",
                             "long" if is_buy else "short")

    def _match_fifo(
        self, inv: _Inv, is_buy: bool, qty: float, price: float, fee: float, ts: int,
    ) -> tuple[float, float, int | None, float]:
        """FIFO-match a fill against opposing lots (validator-faithful). Returns
        (realized_net_of_fees, roundtrip_volume, oldest_matched_ts, gross_pnl)."""
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
        cutoff = now - self.kappa_rt_history_ns
        before = len(st.rt_events)
        st.rt_events = [(t, p) for t, p in st.rt_events if t >= cutoff]
        return len(st.rt_events) != before

    def _pnl_backoff_check(self, st: _BookState, now: int) -> bool:
        if st.pnl_backoff_until_ns > now:
            return True
        if st.pnl_backoff_until_ns > 0:
            st.pnl_backoff_until_ns = 0
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
            return
        st.kappa3 = self._kappa3_raw(self._book_pnl_series(validator, book_id, now))

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

    def _budget_ok(self, validator: str, book_id: int, st: _BookState, now: int,
                   volume_cap: float, rt_max: int) -> bool:
        return (self._rt_count(st, now) < rt_max
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
        return self._avail(account.quote_balance) + self._avail(account.base_balance) * mid

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
    def _estimate_rt_pnl(taker_rate: float, book, qty: float) -> float:
        """Conservative taker RT on the RAW book: buy ask, sell bid, fees both legs (= V4)."""
        if not book.bids or not book.asks:
            return 0.0
        bid, ask = book.bids[0].price, book.asks[0].price
        if bid <= 0 or ask <= 0:
            return 0.0
        return (bid - ask) * qty - taker_rate * (ask + bid) * qty

    @staticmethod
    def _microprice(book, mid: float) -> float:
        bid, ask = book.bids[0], book.asks[0]
        denom = bid.quantity + ask.quantity
        if denom <= 0:
            return mid
        return (ask.price * bid.quantity + bid.price * ask.quantity) / denom

    def _bias(self, book, mid: float) -> int:
        """microprice vs mid -> directional lean; tie -> long (= V4 _book_bias)."""
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
            ctx.mode = self._bstate(validator, book_id).mode
        hold_str = f"{hold_s:.2f}" if hold_s is not None else "n/a"
        bt.logging.info(
            f"[AdaptiveRouterV2 uid={self.uid} RT] book={book_id} mode={ctx.mode} "
            f"open={ctx.open_reason}/{ctx.side} close={ctx.close_reason} "
            f"rtv={rtv:.4f} exit={exit_px:.4f} hold_s={hold_str} "
            f"gross={gross:+.4f} net={net:+.4f} "
            f"kappa={self._fmt_kappa_pair(kappa_before, kappa_after)}"
        )


class _Mode:
    """Base class for a per-book playbook. Modes are stateless singletons; all per-book state lives on
    the shared agent (_BookState, _Inv), so a mode switch never loses inventory or accounting."""

    name = "?"

    def step(self, agent, response, validator, book_id, book, account, st, inv, net,
             best_bid, best_ask, mid, vol_dp, volume_cap, now) -> None:
        raise NotImplementedError

    # -- shared bounded force-close of HELD inventory (mode-threaded slip; NEVER seeds a flat book) --
    def _force_close_held(self, agent, response, book_id, account, inv, net,
                          best_bid, best_ask, vol_dp, *, cap_bps: float) -> bool:
        """Force ONE round-trip-producing close of held inventory with capped slippage. Returns False
        on a flat book (V2 NEVER force-trades a flat book — it idles at kappa=None, free-dropped)."""
        long_q, short_q = agent._long_qty(inv), agent._short_qty(inv)
        if long_q < agent.exch_min and short_q < agent.exch_min:
            return False
        slip = cap_bps / 1e4
        pdp = agent._price_decimals
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
        else:
            buy_px = best_ask * (1.0 + slip)
            q_max = quote_avail / buy_px if buy_px > 0 else short_q
            q = round(min(short_q, lot, q_max), vol_dp)
            if q < agent.exch_min:
                return False
            agent._cancel_all(response, account, book_id)
            agent._submit_limit(response, book_id, OrderDirection.BUY, q,
                                round(best_ask * (1.0 + slip), pdp), ioc=True, post_only=False,
                                settlement=agent._loan_settlement(account))
        return True


class _IdleMode(_Mode):
    """Books idled by fee regime (both-axes-negative) or PnL backoff. Do NOTHING on a flat book so it
    stays kappa=None (free-dropped). Only drain a residual position carried in from a previous mode."""

    name = MODE_IDLE

    def step(self, agent, response, validator, book_id, book, account, st, inv, net,
             best_bid, best_ask, mid, vol_dp, volume_cap, now) -> None:
        if abs(net) >= agent.exch_min:
            agent._stash_open(validator, book_id, st, self.name, "drain",
                              "long" if net >= 0 else "short")
            self._force_close_held(agent, response, book_id, account, inv, net,
                                   best_bid, best_ask, vol_dp, cap_bps=MK_RT_LOSS_CAP_BPS)


class _TakerMode(_Mode):
    """Deep-rebate scalper = TakerScalperV4. SINGLE small clip in the microprice-bias direction; exit
    on TP / SL / max-hold within seconds. No pyramiding, no internal sleep. KEEPS V4's flat
    force-activity backstop (toggle tk_force_activity, default ON): a routed taker book has a real
    rebate, so a forced RT is rebate-funded (~breakeven) and keeps the book dense/scored — DISTINCT
    from idle (the router's decision to NOT trade a no-edge book). Open gate = rebate>=2bps AND
    est_pnl>0 on the raw spread."""

    name = MODE_TAKER

    def step(self, agent, response, validator, book_id, book, account, st, inv, net,
             best_bid, best_ask, mid, vol_dp, volume_cap, now) -> None:
        # Rule 3 (= V4): ONE taker market order in flight at a time. Market open/close orders do NOT
        # rest in account.orders, so without this guard a delayed or sub-exch-min partial fill lets the
        # next step re-submit before the position is recognized -> a transient 2nd lot (breaks the
        # single-open invariant). Wait pending_timeout for the fill, then re-derive from net.
        if st.pending_ns and (now - st.pending_ns) < agent.pending_timeout_ns:
            return
        st.pending_ns = 0
        if abs(net) >= agent.exch_min:
            self._exit(agent, response, validator, book_id, account, inv, net,
                       best_bid, best_ask, vol_dp, now, st)
            return
        throttled = st.last_close_ns and (now - st.last_close_ns) < agent.tk_reopen_gap_ns
        if not throttled and self._open(agent, response, validator, book_id, book, account, st,
                                        best_bid, best_ask, mid, volume_cap, now, vol_dp):
            return
        # FORCE-ACTIVITY (breadth) — DISTINCT from idle. A book routed here has a real rebate, so a
        # forced RT is rebate-funded (~breakeven) and keeps the book dense/scored (clears the >=3-RT
        # kappa gate). This is TakerScalperV4's activity backstop; idle is the router's decision to NOT
        # trade a no-edge book. (Maker has NO flat force-activity — maker RTs PAY the fee.)
        if agent.tk_force_activity and agent._activity_due(st, now):
            self._force_open(agent, response, validator, book_id, book, account, st,
                             best_bid, best_ask, mid, now, vol_dp)

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
        elif gross_bps <= -agent.tk_sl_bps:
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
                return True
            agent._submit_market(response, book_id, OrderDirection.SELL, q)
        else:
            q = round(min(abs(net),
                          agent._avail(account.quote_balance) / best_ask if best_ask > 0 else abs(net)),
                      vol_dp)
            if q < agent.exch_min:
                return True
            agent._submit_market(response, book_id, OrderDirection.BUY, q,
                                 settlement=agent._loan_settlement(account))
        st.last_close_ns = now
        st.pending_ns = now            # close in flight — block re-action until the fill (rule 3)
        return True

    def _open(self, agent, response, validator, book_id, book, account, st,
              best_bid, best_ask, mid, volume_cap, now, vol_dp) -> bool:
        taker_fee = agent._taker_fee_rate(account)
        if taker_fee is None or not agent._budget_ok(validator, book_id, st, now, volume_cap, TK_RT_MAX):
            return False
        rebate_bps = -taker_fee * 1e4
        if rebate_bps < TK_REBATE_FLOOR_BPS:          # = V4: open ONLY on a real rebate (>=2bps)
            return False
        q = round(agent.clip, vol_dp)
        if q < agent.exch_min:
            return False
        if agent._estimate_rt_pnl(taker_fee, book, q) <= 0.0:   # = V4: rebate must beat the raw spread
            return False
        direction = agent._bias(book, mid)
        if not self._submit_lot(agent, response, book_id, account, direction, q, best_bid, best_ask):
            return False
        st.pending_ns = now            # open in flight — block re-open until the fill (rule 3)
        agent._stash_open(validator, book_id, st, self.name, "scalp",
                          "long" if direction == OrderDirection.BUY else "short")
        return True

    def _force_open(self, agent, response, validator, book_id, book, account, st,
                    best_bid, best_ask, mid, now, vol_dp) -> bool:
        """Force ONE rebate-funded RT (bypassing the est_pnl / rebate-floor profit gate) so a routed
        taker book stays dense/scored across the activity window (= V4's backstop). Direction = bias;
        the position then exits normally via _exit, bounded by the tight SL."""
        q = round(agent.clip, vol_dp)
        if q < agent.exch_min:
            return False
        direction = agent._bias(book, mid)
        if not self._submit_lot(agent, response, book_id, account, direction, q, best_bid, best_ask):
            return False
        st.pending_ns = now            # forced open in flight — block re-open until the fill (rule 3)
        agent._stash_open(validator, book_id, st, self.name, "activity",
                          "long" if direction == OrderDirection.BUY else "short")
        return True

    @staticmethod
    def _submit_lot(agent, response, book_id, account, direction, q, best_bid, best_ask) -> bool:
        """Cross for one lot = V4 _taker_open: BUY needs quote; SELL uses base, else a margin short
        (keeps breadth on books with no base inventory). Returns whether an order was submitted."""
        if direction == OrderDirection.BUY:
            if best_ask <= 0 or agent._avail(account.quote_balance) < q * best_ask:
                return False
            agent._submit_market(response, book_id, OrderDirection.BUY, q,
                                 settlement=agent._loan_settlement(account))
        else:
            if best_bid <= 0:
                return False
            free_base = account.base_balance.free if account.base_balance else 0.0
            if free_base >= q:
                agent._submit_market(response, book_id, OrderDirection.SELL, q)
            else:
                quote_loan = getattr(account, "quote_loan", 0.0) or 0.0
                agent._submit_market(response, book_id, OrderDirection.SELL, q,
                                     leverage=0.0 if quote_loan > 0 else 1.0,
                                     settlement=agent._loan_settlement(account))
        return True


class _MakerMode(_Mode):
    """Two-sided spread capture = PureMakerV4. Flat -> quote both sides inside-on-wide. Holding ->
    work ONLY the reducing side, walked target->BREAKEVEN with age (never the touch), floored at the
    vol-scaled stop. Managed-exit IOC-cuts an aged/underwater lot with a bounded, escalating slip.
    Held-only activity backstop; flat books idle (the router, not this leg, decides idling)."""

    name = MODE_MAKER

    def step(self, agent, response, validator, book_id, book, account, st, inv, net,
             best_bid, best_ask, mid, vol_dp, volume_cap, now) -> None:
        # 1) MANAGED EXIT first (stop-loss / giveup) — before the risk guard so a stop is never blocked.
        if self._managed_exit(agent, response, validator, book_id, account, inv, net,
                              best_bid, best_ask, vol_dp, now, st):
            st.last_cut_ns = now
            return
        # 2) RISK GUARD — drain breached inventory after the stop has had its turn.
        if self._risk_trim(agent, response, book_id, account, net, mid, vol_dp):
            return
        # 3) ACTIVITY BACKSTOP (held-only) — force-close a held position that outlived the window
        #    (deep safety; the 150s giveup normally closes it first). FLAT books are NOT force-traded.
        if agent._activity_due(st, now) and abs(net) >= agent.exch_min:
            agent._stash_open(validator, book_id, st, self.name, "activity",
                              "long" if net >= 0 else "short")
            if self._force_close_held(agent, response, book_id, account, inv, net,
                                      best_bid, best_ask, vol_dp, cap_bps=MK_RT_LOSS_CAP_BPS):
                return
        # 4) DESIRED QUOTES — reduce-only when holding; two-sided entry when the budget gates clear.
        desired = self._desired_quotes(agent, validator, book_id, account, inv, net,
                                       best_bid, best_ask, mid, volume_cap, now, vol_dp, st)
        self._reconcile(agent, response, account, book_id, desired)

    def _stop_bps(self, st: _BookState) -> float:
        """Vol-scaled TIGHT stop within [FLOOR, CAP] = [10, 14]bps (= PMV4)."""
        scaled = MK_STOP_NOISE_MULT * st.noise_bps
        return min(MK_STOP_CAP_BPS, max(MK_STOP_FLOOR_BPS, scaled))

    def _risk_trim(self, agent, response, book_id, account, net, mid, vol_dp) -> bool:
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
        slip = MK_RISK_TRIM_SLIPPAGE_BPS / 1e4
        pdp = agent._price_decimals
        if net > 0:
            trim = round(min(trim, agent._avail(account.base_balance)), vol_dp)
        else:
            buy_px = mid * (1.0 + slip)
            q_max = agent._avail(account.quote_balance) / buy_px if buy_px > 0 else trim
            trim = round(min(trim, q_max), vol_dp)
        if trim < agent.exch_min:
            return False
        agent._cancel_all(response, account, book_id)
        if net > 0:
            agent._submit_limit(response, book_id, OrderDirection.SELL, trim,
                                round(mid * (1.0 - slip), pdp), ioc=True, post_only=False)
        else:
            agent._submit_limit(response, book_id, OrderDirection.BUY, trim,
                                round(mid * (1.0 + slip), pdp), ioc=True, post_only=False,
                                settlement=agent._loan_settlement(account))
        return True

    def _managed_exit(self, agent, response, validator, book_id, account, inv, net,
                      best_bid, best_ask, vol_dp, now, st) -> bool:
        if abs(net) < agent.exch_min:
            st.exit_miss_count = 0
            st.exit_prev_net = 0.0
            return False
        stop_bps = self._stop_bps(st)
        if st.exit_prev_net > 0:
            if abs(net) >= st.exit_prev_net - agent._flat_eps:
                st.exit_miss_count += 1
            else:
                st.exit_miss_count = 0
                st.exit_prev_net = 0.0
        if st.exit_miss_count >= 4:
            slip = MK_IOC_CROSS_BPS / 1e4
        elif st.exit_miss_count >= 2:
            slip = MK_IOC_ESCALATE_BPS / 1e4
        else:
            slip = MK_IOC_SLIPPAGE_BPS / 1e4
        pdp = agent._price_decimals
        if net > 0:
            ts, _, px0, _ = inv.longs[0]
            uw = (px0 - best_bid) / px0 * 1e4 if px0 > 0 else 0.0
            if not (now - ts >= agent.mk_giveup_ns or uw >= stop_bps):
                st.exit_miss_count = 0
                st.exit_prev_net = 0.0
                return False
            q = round(min(agent._long_qty(inv), agent._avail(account.base_balance)), vol_dp)
            if q < agent.exch_min:
                st.exit_miss_count = 0
                st.exit_prev_net = 0.0
                return False
            st.exit_prev_net = abs(net)
            self._tag_close(agent, validator, book_id, "cut")
            agent._cancel_all(response, account, book_id)
            agent._submit_limit(response, book_id, OrderDirection.SELL, q,
                                round(best_bid * (1.0 - slip), pdp), ioc=True, post_only=False)
        else:
            ts, _, px0, _ = inv.shorts[0]
            uw = (best_ask - px0) / px0 * 1e4 if px0 > 0 else 0.0
            if not (now - ts >= agent.mk_giveup_ns or uw >= stop_bps):
                st.exit_miss_count = 0
                st.exit_prev_net = 0.0
                return False
            buy_px = best_ask * (1.0 + slip)
            q_max = agent._avail(account.quote_balance) / buy_px if buy_px > 0 else agent._short_qty(inv)
            q = round(min(agent._short_qty(inv), q_max), vol_dp)
            if q < agent.exch_min:
                st.exit_miss_count = 0
                st.exit_prev_net = 0.0
                return False
            st.exit_prev_net = abs(net)
            self._tag_close(agent, validator, book_id, "cut")
            agent._cancel_all(response, account, book_id)
            agent._submit_limit(response, book_id, OrderDirection.BUY, q,
                                round(best_ask * (1.0 + slip), pdp), ioc=True, post_only=False,
                                settlement=agent._loan_settlement(account))
        if st.exit_miss_count >= 2:
            label = "IOC-CROSS" if st.exit_miss_count >= 4 else "IOC-ESCALATE"
            bt.logging.info(
                f"[AdaptiveRouterV2 uid={agent.uid}] {label} book={book_id} "
                f"miss={st.exit_miss_count} slip={slip*1e4:.0f}bps"
            )
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
        tp_floor_bps = MK_TP_FEE_MULT * fee_bps + (agent._tick / mid) * 1e4
        base_target = max(MK_TP_BPS, tp_floor_bps) / 1e4
        stop_bps = self._stop_bps(st)

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
            px = self._reduce_price(True, fifo_px, age, ask_inside, base_target,
                                    tp_floor_bps, stop_bps, pdp, agent)
            q = round(min(agent._long_qty(inv), base_avail), vol_dp)
            if q >= agent.exch_min and px > 0:
                desired[OrderDirection.SELL] = (px, q)
        elif net <= -agent.exch_min:
            age = now - inv.shorts[0][0]
            fifo_px = inv.shorts[0][2]
            px = self._reduce_price(False, fifo_px, age, bid_inside, base_target,
                                    tp_floor_bps, stop_bps, pdp, agent)
            q_max = quote_avail / px if px > 0 else agent._short_qty(inv)
            q = round(min(agent._short_qty(inv), q_max), vol_dp)
            if q >= agent.exch_min and px > 0:
                desired[OrderDirection.BUY] = (px, q)
        elif st.last_cut_ns > 0 and now - st.last_cut_ns < agent.mk_reentry_cooldown_ns:
            pass   # post-cut cooldown: don't re-enter immediately after a managed exit
        elif agent._budget_ok(validator, book_id, st, now, volume_cap, MK_RT_MAX):
            q = round(agent.clip, vol_dp)
            if q >= agent.exch_min and free_base >= q:
                desired[OrderDirection.SELL] = (ask_inside, q)
            if q >= agent.exch_min and free_quote >= q * bid_inside:
                desired[OrderDirection.BUY] = (bid_inside, q)
        return desired

    def _reduce_price(self, is_long, px0, age_ns, touch_inside, base_target,
                      walk_floor_bps, stop_bps, pdp, agent) -> float:
        """Walk the passive-reduce limit target -> BREAKEVEN with age (never the touch = PMV4), floored
        at the vol-scaled stop so a resting reduce never fills worse than the IOC cut."""
        w = self._exit_walk(age_ns, agent)
        stop = stop_bps / 1e4
        floor_t = walk_floor_bps / 1e4
        if is_long:
            target_px = max(touch_inside, px0 * (1.0 + base_target))
            walk_to = min(target_px, px0 * (1.0 + floor_t))   # breakeven sell, never below it
            px = target_px + (walk_to - target_px) * w
            return round(max(px, px0 * (1.0 - stop)), pdp)
        target_px = min(touch_inside, px0 * (1.0 - base_target))
        walk_to = max(target_px, px0 * (1.0 - floor_t))       # breakeven buy, never above it
        px = target_px + (walk_to - target_px) * w
        return round(min(px, px0 * (1.0 + stop)), pdp)

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
                    and abs(o.price - want[0]) < REPRICE_KEEP_TICKS * agent._tick
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
