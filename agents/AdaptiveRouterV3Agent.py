"""
AdaptiveRouterV3Agent — per-book REGIME-ADAPTIVE router for subnet 79 (fork of AdaptiveRouterV2Agent).

V3 CORE IDEA (2026-06-27, data + adversarial-review derived): the entire measured V2 loss is a ONE-SIDED held
maker lot caught by a directional move and realized at the 15bps stop (uid64: cuts −258 net / −9.11 kappa,
wiping +8.48 from good fills; corr(price-range, cut)=+0.44). So V3 classifies each book's CHARACTER live and
routes by it, using three PROVEN, O(1)-cheap algorithms:
  * Kaufman EFFICIENCY RATIO (ER = |net move| / path length over a time-windowed sub-sampled mid series) — canonical
    trend-vs-noise discriminator. ER→1 = one-sided trend; ER→0 = mean-reverting range. THE key signal
    (raw amplitude does NOT separate winners/losers; one-sidedness does).
  * EWMA volatility (RiskMetrics σ²ₙ=λσ²ₙ₋₁+(1−λ)rₙ²) — amplitude estimate for the magnitude gate + A-S spread.
  * Avellaneda-Stoikov inventory skew (reservation r = mid − q·γ·σ²) — the two-sided maker leans its quotes
    AGAINST inventory so it stops accumulating into a move (kills the adverse-selection that V2 suffered).
Per book → char ∈ {SMOOTH, CHOP, DIRECTIONAL}, latched with a 2-step ENTRY confirm (DIR_CONFIRM_STEPS) +
hysteresis + min-dwell so a single mid spike can't flip a held lot's regime:
  * SMOOTH / CHOP (low one-sidedness; CHOP = high-amplitude mean-reverting) → continuous TWO-SIDED never-cut
    maker (A-S-skewed) — the home engine. A held loser is realized ONLY by the 15bps never-cut stop (rare),
    WHOLE-SIDE in one shot.
  * DIRECTIONAL (one-sided trend) → do NOT open a passive maker lot; route to the PATIENT REBATE TAKER
    (uid62 archetype: ~59s holds, bleed WITH the move in rebate-funded clips, catastrophe-only stop) if the
    fee regime makes it +EV, else IDLE (free-drop; at the idle CLIFF, fall through to a resting maker instead).
    A maker lot ALREADY held when a book turns DIRECTIONAL just rides the pure never-cut stop (it can't re-route
    mid-position) — see below.
Held-lot cuts = PURE NEVER-CUT 15bps WHOLE-SIDE, regardless of character. The detector is used for ROUTING ONLY.
This was SETTLED empirically by the §10.1 offline kappa replay (tests/arv3_kappa_replay.py — real validator
kappa_3 on real MAKER and TREND price paths): two fancier ideas were tested and BOTH lost to pure never-cut-15:
(1) scaling the cut out one clip at a time (Δ≈-0.012 on the mean-reverting home regime — fragments one cut into
clustered negative RTs and forgoes the clean re-enter-lower on the revert); (2) a tighter 6bps stop on
DIRECTIONAL books (WORSE in EVERY regime incl. trends, uid184 Δ-0.020 — real trends are noisy so a tight stop is
whipsawed into many small realized losses that the cube-downside kappa punishes harder than one rare 15bps cut,
and trends often retrace <15bps). So the maker's only loss-realiser is the 15bps never-cut, for all regimes —
simplest, fastest, and the kappa-max choice. The PatientTaker still scales out (proven uid62 REBATE-taker
mechanism, a different context). The SMOOTH path is byte-equivalent to V2, so V3 cannot regress the calm regime.

Inherited V2 changes (taker leg unchanged):
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
    everywhere else, gated by an 8bps maker-fee ceiling (with an in-mode / idle-cliff bypass).
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
# Which modes the router may use. full set => adaptive per-book routing (the default). "ptaker" (V3) is the
# PatientTaker directional response (uid62 archetype). The fast _TakerMode stays available for A/B.
ALLOWED_MODES = {"taker", "maker", "idle", "ptaker"}

# ======================================================================= V3 regime-adaptive config
# Per-book CHARACTER detector — three PROVEN O(1) algorithms (Kaufman Efficiency Ratio + RiskMetrics EWMA vol),
# computed from sub-sampled mids in _step_book (onTrade only fires on OUR fills — verified self.events =
# state.notices[self.uid] — so it can't see market mids; the per-publish state IS the only continuous mid feed).
# TIME-BASED window (NOT a fixed publish count): the live publish_interval is 1 sim-second (simulation_0.xml
# step="1000000000"), so a fixed 10-publish window = ~10s — ~12x too short for a REGIME property and it badly
# mis-classifies (validated: tests/arv3_detector_realcsv.py). Instead sub-sample one mid every CHAR_SAMPLE_GAP_S
# and keep a CHAR_WINDOW_S span, so the detector timescale is decoupled from publish_interval for ANY interval
# up to ~CHAR_WINDOW_S/(CHAR_MIN_SAMPLES-1) ≈ 30s (covers the live 1s with huge margin). ABOVE that the window
# can't gather CHAR_MIN_SAMPLES mids and the detector stays SMOOTH (degrades to the V2 never-cut maker — a SAFE
# no-op, not a crash); `initialize` logs a WARNING if it ever sees such an interval so a config change can't
# silently disable the detector.
CHAR_WINDOW_S = 150.0              # Efficiency-Ratio window SPAN in sim-seconds (~2.5min — a real regime timescale)
CHAR_SAMPLE_GAP_S = 12.0          # min sim-seconds between recorded mids (sub-sample: filters 1s micro-noise so ER
                                   # measures the regime, not microstructure; also caps the window to ~13 samples)
CHAR_MIN_SAMPLES = 6              # need this many sub-sampled mids in the window before classifying (else SMOOTH)
CHAR_HIST_CAP = 24               # deque maxlen backstop (CHAR_WINDOW_S/CHAR_SAMPLE_GAP_S + slack)
VOL_EWMA_LAMBDA = 0.94             # RiskMetrics decay: vol_var = λ·vol_var + (1−λ)·r²  (per-SAMPLE return var)
ER_DIRT_ENTER = 0.65               # Efficiency Ratio ≥ this (AND amplitude gates) -> DIRECTIONAL (one-sided)
ER_DIRT_EXIT = 0.45                # ER ≤ this -> leave DIRECTIONAL (0.65/0.45 anti-flip band)
RVS_ENTER = 2.5                    # window range / full-spread ≥ this = amplitude big enough to matter (review:
                                   # 3.0 under-detected trends on the live wide-spread books; safe to loosen now
                                   # that a false DIRECTIONAL only diverts ROUTING — it never opens a fresh maker lot
                                   # and suppresses the maker ADD leg, but never cuts an existing held lot)
RVS_EXIT = 1.6                     # range/full-spread ≤ this -> leave DIRECTIONAL (2.5/1.6 anti-flip band)
EXC_ENTER_BPS = 6.0                # |window net move| ≥ this bps to call DIRECTIONAL (loosened 8→6 with RVS so the
                                   # protection actually engages; a false positive only costs a routing divert, not a cut)
DIR_CONFIRM_STEPS = 2              # is_dir must hold this many CONSECUTIVE samples (~24s) before latching
                                   # DIRECTIONAL — a single transient mid spike must not divert routing away from a
                                   # passive maker lot / suppress the maker ADD leg (review #8)
SPREAD_FLOOR_BPS = 4.0             # floor the range/spread denominator so tight books don't false-trip DIRECTIONAL
CHAR_MIN_DWELL_S = 90.0            # DIRECTIONAL latch min lifetime (ride the aftershock; mirrors ROUTE dwell)
KAPPA_REFRESH_MIN_GAP_S = 45.0     # min WALL seconds between per-book kappa3 recomputes (kappa3 is LOGGING-ONLY;
                                   # throttle the O(B²·E) cross-book scan that otherwise fires most steps — review #1)

# Avellaneda-Stoikov inventory skew for the two-sided maker: reservation r = mid − q·γ·σ². We lean BOTH quotes
# against inventory so the maker stops accumulating into a move (kills the V2 adverse-selection accumulation).
AS_GAMMA = 0.5                     # risk aversion; skew_bps = (net/clip)·AS_GAMMA·vol_bps, capped below
AS_MAX_SKEW_BPS = 6.0              # cap inventory skew so quotes stay sane
MK_SOFT_INVENTORY_LOTS = 1.5      # maker stops re-posting the ADD/entry side at/above this (nests under risk_trim 2.0)

# PatientTaker (uid62 archetype) — the DIRECTIONAL response: patient, rebate-funded, bleed-with-the-move in clips.
PT_TICK_S = 8.0                    # min seconds between PatientTaker actions per book (uid62 median inter-fill 8s)
PT_MIN_HOLD_S = 45.0               # reduce a lot only after this hold (jitter -> ~59s median, uid62)
PT_SOFT_INV_LOTS = 3.0             # start reducing / stop accumulating at this inventory
PT_MAX_INV_LOTS = 3.0              # hard inventory cap (uid62 typical peak, NOT its p90 — cube-tail safety)
PT_MAX_HOLD_S = 180.0             # PROJECT RULE: never hold forever. Force-close the whole patient position past
                                   # this even if the per-tick scale-out stalls (missed IOCs / pressing) — bounds the bag.
PT_CATASTROPHE_BPS = 20.0         # catastrophe stop, scaled DOWN by inventory (PT_CATASTROPHE/inv_lots) -> bounded cube
                                   # (review #5: 30→20 so the one un-fragmented worst-case RT is a smaller cube;
                                   # 1 lot=20bps, 3 lots≈6.7bps. The per-tick scale-out should exit well before this)
PT_REDUCE_SLIP_BPS = 2.0          # near-touch IOC slip on the patient reduce
PT_MAX_RT_COST_BPS = 3.0          # taker-pays gate: open only if (2·half_spread − 2·rebate) ≤ this, else IDLE

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
                                   # sharp-dump→revert window (MeanReversionAgent: dump then grind up over
                                   # minutes; tape-tuned close-anyway = 180s). Still underwater after this -> it
                                   # did NOT revert -> force-cut (frees the book to re-route + bounds the bag/tail).
                                   # Pairs with the 20bps catastrophe stop: realize a held loser on EITHER uw>=stop OR
                                   # age>=this — never indefinitely. (Fixes the mode-stuck + bag/critical-loss gap.)
MK_STOP_LOSS_BPS = 20.0            # catastrophe stop above 20bps dump band (tape); below 35-60 cube-bomb zone
                                   # (reduce still walks to BREAKEVEN; this is the ONLY loss-realiser)
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
MODE_PTAKER = "ptaker"   # V3 PatientTaker (uid62 archetype): directional response

# Book CHARACTER (V3 detector output)
CHAR_SMOOTH = "smooth"
CHAR_CHOP = "chop"
CHAR_DIRECTIONAL = "directional"

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
    rt_events: deque = field(default_factory=deque)   # (sim_ts, realized_pnl) FIFO; ALL validators,
                                                                       # short-retained (RT_EVENTS_RETENTION_S):
                                                                       # feeds pnl-backoff + rt_count only
    kappa_events: list[tuple[int, float]] = field(default_factory=list) # (wall_ts, realized_pnl); MAIN validator
                                                                        # ONLY (kappa3 is logging-only)
    kappa3: float | None = None        # logging-only (RT log); never read by a routing/mode decision
    kappa_refresh_ns: int = 0          # wall ts of the last kappa3 recompute (throttle the cross-book scan)
    vol_log: deque = field(default_factory=deque)    # (ts, traded quote vol), FIFO; pruned from the left
    vol_sum: float = 0.0               # running sum of vol_log volumes (O(1) rolled-volume; review #13)
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
    # V3 per-book CHARACTER detector (Kaufman Efficiency Ratio + EWMA vol over a time-windowed sub-sampled mid series)
    mid_hist: deque = field(default_factory=lambda: deque(maxlen=CHAR_HIST_CAP))  # recent (ts, mid) sub-samples
    last_mid_sample_ns: int = 0        # ts of the last recorded mid sub-sample (CHAR_SAMPLE_GAP_S throttle)
    vol_var: float = 0.0               # EWMA variance of per-sub-sample returns (RiskMetrics); 0 = uninitialised
    char: str = CHAR_SMOOTH            # latched book character; SMOOTH/CHOP keep maker, DIRECTIONAL diverts
    char_since_ns: int = 0             # when the current char latched (dwell clock)
    dir_streak: int = 0               # consecutive is_dir steps (entry confirmation before latching DIRECTIONAL)
    # V3 PatientTaker bookkeeping
    pt_last_act_ns: int = 0            # last PatientTaker action (per-book tick throttle)


@dataclass
class _RtLogCtx:
    """Snapshot stashed when a position opens; finalized and logged at the closing RT fill."""
    mode: str = "?"
    open_reason: str = "?"
    side: str = "?"
    close_reason: str = "fill"


class AdaptiveRouterV3Agent(FinanceSimulationAgent):
    # ------------------------------------------------------------------ setup
    def initialize(self) -> None:
        bt.logging.set_info()
        # RESPONSE-TIME: the framework's update() builds a per-book DEBUG string (balances/orders/levels/loans
        # for all 128 books) EVERY step and passes it to bt.logging.debug() — discarded at INFO, so it is pure
        # waste (~3.4ms/step, ~96% of update(); benchmarked). lazy_load=True skips that block with ZERO visible
        # change at INFO. Cheap per-miner, but it matters fleet-wide (≈30 miners on shared CPU → less contention
        # → fewer transport timeouts). Agent compute is ~0.5% of the ~1.5s response time; the rest is network +
        # state (de)serialization, which is a bittensor/host concern, not the agent's.
        if getattr(self, "config", None) is not None:
            self.config.lazy_load = True

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
        self.kappa_refresh_min_gap_ns = int(KAPPA_REFRESH_MIN_GAP_S * _NS)   # throttle the logging-only kappa scan
        self.pnl_backoff_window_ns = int(PNL_BACKOFF_WINDOW_S * _NS)
        self.pnl_backoff_cooldown_ns = int(PNL_BACKOFF_COOLDOWN_S * _NS)
        self.char_min_dwell_ns = int(CHAR_MIN_DWELL_S * _NS)        # V3 detector latch dwell
        self.char_window_ns = int(CHAR_WINDOW_S * _NS)             # V3 detector time-based window span
        self.char_sample_gap_ns = int(CHAR_SAMPLE_GAP_S * _NS)     # V3 detector sub-sample cadence
        self.pt_tick_ns = int(PT_TICK_S * (0.9 + 0.2 * jitter) * _NS)   # V3 PatientTaker per-book tick
        self.pt_min_hold_ns = int(PT_MIN_HOLD_S * (0.9 + 0.4 * jitter) * _NS)   # jitter -> ~45-63s
        self.pt_max_hold_ns = int(PT_MAX_HOLD_S * (0.9 + 0.2 * jitter) * _NS)   # never-hold-forever cap

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
            MODE_PTAKER: _PatientTakerMode(),
        }

        self.inv: dict[str, dict[int, _Inv]] = {}
        self.books_state: dict[str, dict[int, _BookState]] = {}
        self._rt_log: dict[tuple[str, int], _RtLogCtx] = {}
        self._sim_id: dict[str, str] = {}
        self._step_ts_ns: dict[str, int] = {}
        self._active_validator: str | None = None
        self._pub_checked = False           # one-time publish_interval sanity check (detector window feasibility)

        bt.logging.info(
            f"[AdaptiveRouterV3 uid={self.uid}] modes={sorted(ALLOWED_MODES)} default={self.default_mode} "
            f"clip={TARGET_CLIP} dwell={ROUTE_MIN_DWELL_S:.0f}s "
            f"route(taker_rebate>={TAKER_REBATE_ENTER_BPS}/{TAKER_REBATE_EXIT_BPS}bps, "
            f"maker_edge>={MAKER_EDGE_ENTER_BPS}/{MAKER_EDGE_EXIT_BPS}bps, maker_fee<{MAKER_MAX_FEE_BPS}bps) "
            f"mk=reduce->breakeven,stop={MK_STOP_LOSS_BPS:.0f}bps,max_hold={MK_MAX_HOLD_S:.0f}s "
            f"char(win={CHAR_WINDOW_S:.0f}s/samp{CHAR_SAMPLE_GAP_S:.0f}s ER>={ER_DIRT_ENTER}/{ER_DIRT_EXIT} "
            f"rvs>={RVS_ENTER}/{RVS_EXIT} net>={EXC_ENTER_BPS}bps confirm={DIR_CONFIRM_STEPS}) "
            f"as_skew(g={AS_GAMMA},cap={AS_MAX_SKEW_BPS}bps,soft_inv={MK_SOFT_INVENTORY_LOTS}) "
            f"ptaker(tick={PT_TICK_S:.0f}s hold={PT_MIN_HOLD_S:.0f}s inv<={PT_MAX_INV_LOTS} "
            f"cata={PT_CATASTROPHE_BPS:.0f}bps max_hold={PT_MAX_HOLD_S:.0f}s gate_rt<={PT_MAX_RT_COST_BPS}bps) "
            f"route_ema={ROUTE_SPREAD_EMA_HALFLIFE_S:.0f}s idle_cap={MAX_IDLE_BOOKS}/cliff={CLIFF_IDLE_BOOKS} "
            f"rt_loss_cap={RT_LOSS_CAP_BPS}bps activity_deadline={ACTIVITY_DEADLINE_S:.0f}s "
            f"rt_max={RT_MAX} rt_log={MAIN_VALIDATOR[:8]} "
            f"pnl_backoff(window={PNL_BACKOFF_WINDOW_S:.0f}s cooldown={PNL_BACKOFF_COOLDOWN_S:.0f}s min_rts={PNL_BACKOFF_MIN_RTS})"
        )
        self._tune_gc()

    def _tune_gc(self) -> None:
        """RESPONSE-TIME: the asyncio/axon layer retains completed Task objects that hold ~128-orderbook state,
        so the long-lived heap is large and EVERY gen2 GC sweep rescans it — pauses spike to tens of ms, and
        when one lands mid-handle() it stretches the response past the validator timeout. We can't fix axon's
        task lifecycle, but we control this process's GC. Measured (tests): a gen2 gc.collect() drops ~34ms -> 0
        after freeze. Three knobs, all behaviour-neutral:
          1) history_len=0  — the framework deep-copies the FULL 128-book state every step and keeps 10 copies
             (self.history); we never read it. Kills that per-step model_copy + a big chunk of the scanned heap.
          2) gc.freeze()    — move the ~120k permanent import/setup objects into the frozen gen so GC NEVER
             rescans them; gen2 sweeps then only touch the small per-request churn.
          3) raise thresholds — gen2 sweeps far less often, so a sweep rarely coincides with handle()."""
        self.history_len = 0
        try:
            gc.collect()
            gc.freeze()                       # exclude the permanent import/setup heap from all future sweeps
            gc.set_threshold(50_000, 500, 500)  # gen2 sweeps far less frequently
            bt.logging.info(f"[AdaptiveRouterV3 uid={self.uid}] gc tuned: frozen={gc.get_freeze_count()} "
                            f"thresholds={gc.get_threshold()} history_len=0")
        except Exception as ex:
            bt.logging.warning(f"[AdaptiveRouterV3 uid={self.uid}] gc tune skipped: {ex}")

    # --------------------------------------------------------------- lifecycle
    def update(self, state: MarketSimulationStateUpdate) -> None:
        self._active_validator = state.dendrite.hotkey
        self._step_ts_ns[self._active_validator] = int(state.timestamp)
        if not self._pub_checked:
            # One-time guard: the time-based CHARACTER window needs CHAR_MIN_SAMPLES mids within CHAR_WINDOW_S,
            # which holds only while publish_interval <= CHAR_WINDOW_S/(CHAR_MIN_SAMPLES-1). Past that the
            # detector silently stays SMOOTH (safe V2 fallback) — warn so a config change can't hide it.
            self._pub_checked = True
            pub_ns = getattr(state.config, "publish_interval", 0) or 0
            max_pub_ns = self.char_window_ns / max(CHAR_MIN_SAMPLES - 1, 1)
            if pub_ns > max_pub_ns:
                bt.logging.warning(
                    f"[AdaptiveRouterV3 uid={self.uid}] publish_interval={pub_ns/_NS:.1f}s > "
                    f"{max_pub_ns/_NS:.1f}s — CHARACTER detector will stay SMOOTH (V2 fallback); "
                    f"lower CHAR_SAMPLE_GAP_S / raise CHAR_WINDOW_S to re-enable regime routing."
                )
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
            f"[AdaptiveRouterV3 uid={self.uid}] new simulation: {validator[:8]} sim_id={simulation_id}"
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
                bt.logging.warning(f"[AdaptiveRouterV3 uid={self.uid}] step {book_id}: {ex}")

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
        # V3: update the per-book CHARACTER (Kaufman ER + EWMA vol; time-windowed sub-sampled mids). Bookkeeping.
        self._update_char(st, mid, now)
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
            # THROTTLE (review #1): _prune_rt_events returns True most steps (events age out of the 900s window
            # continuously), so this O(B²·E) cross-book scan would fire nearly every step on every active book.
            # kappa3 is a pure log field, so recompute at most once per KAPPA_REFRESH_MIN_GAP_S wall-seconds per
            # book (the on-CLOSE refresh in _record_rt_close stays un-throttled so the RT log is always fresh).
            wall = time.time_ns()
            if wall - st.kappa_refresh_ns >= self.kappa_refresh_min_gap_ns:
                st.kappa_refresh_ns = wall
                self._refresh_book_kappa(validator, book_id, wall)

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
                        f"[AdaptiveRouterV3 uid={self.uid}] PNL-BACKOFF {st.mode}->idle book={book_id}"
                    )
                    st.mode, st.mode_since_ns = MODE_IDLE, now
            else:
                # V2: NO maker→taker demote on a recent-PnL blip — flipping a mean-reverting maker book
                # contaminates its per-book Sortino for the ~20h window. The never-cut maker (hold for
                # revert) + the realized-PnL backoff below handle genuine losers instead.
                want = self._route(st, account, best_bid, best_ask, mid,
                                   fallback_maker=fallback_maker, cliff=cliff)
                if want != st.mode:
                    # Emergency flip: bypass the route dwell guard (ROUTE_MIN_DWELL_S) when the current mode has
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
                        (st.mode == MODE_MAKER and mk_edge_em < EMERGENCY_MAKER_EXIT_BPS) or
                        # V3: a maker book whose character just turned DIRECTIONAL must leave promptly
                        # (don't wait the 300s dwell to stop posting passive quotes into a trend).
                        (st.mode == MODE_MAKER and st.char == CHAR_DIRECTIONAL and want != MODE_MAKER)
                    )
                    if emergency or (now - st.mode_since_ns) >= self.route_min_dwell_ns:
                        if account.orders:
                            self._cancel_all(response, account, book_id)
                            return
                        bt.logging.info(
                            f"[AdaptiveRouterV3 uid={self.uid}] "
                            f"{'EMERGENCY-FLIP' if emergency else 'ROUTE'} "
                            f"{st.mode}->{want} book={book_id} "
                            f"taker_fee={taker_fee_em} maker_fee={maker_fee_em} "   # reuse (no re-lookup)
                            f"spread_bps={(best_ask - best_bid) / mid * 1e4:.1f}"
                            + (" [cliff]" if cliff else (" [fallback-maker]" if fallback_maker else ""))
                        )
                        st.mode, st.mode_since_ns = want, now

        mode = self._modes.get(st.mode) or self._modes[self.default_mode]
        mode.step(self, response, validator, book_id, book, account, st, inv, net,
                  best_bid, best_ask, mid, vol_dp, volume_cap, now)

    # ------------------------------------------------------------------ V3 character detector
    def _update_char(self, st: _BookState, mid: float, now: int) -> None:
        """Per-book CHARACTER via two proven O(window) signals over a TIME-BASED sub-sampled mid window:
          * Kaufman EFFICIENCY RATIO  er = |net move| / path length  — ~1 one-sided trend, ~0 mean-reverting.
          * RiskMetrics EWMA vol      vol_var = λ·vol_var + (1−λ)·r²  — amplitude (feeds the A-S maker skew).
        The window is a CHAR_WINDOW_S span of mids sub-sampled every CHAR_SAMPLE_GAP_S (NOT a fixed publish
        count) so the regime timescale is correct for any publish_interval up to ~30s (the live value is 1s),
        and 1s microstructure noise doesn't inflate the path length. Latches char in {SMOOTH, CHOP, DIRECTIONAL} with hysteresis +
        a min dwell. SMOOTH/CHOP keep the maker; DIRECTIONAL diverts. Warmup / thin window = SMOOTH (home)."""
        hist = st.mid_hist
        # SUB-SAMPLE: only record a mid once per CHAR_SAMPLE_GAP_S; other publishes leave char latched.
        if hist and (now - st.last_mid_sample_ns) < self.char_sample_gap_ns:
            return
        if hist:                                    # EWMA return variance from the last sub-sampled return
            base = hist[-1][1]
            if base > 0:
                r = (mid - base) / base
                st.vol_var = (VOL_EWMA_LAMBDA * st.vol_var + (1.0 - VOL_EWMA_LAMBDA) * r * r
                              if st.vol_var > 0.0 else r * r)
        hist.append((now, mid))
        st.last_mid_sample_ns = now
        cutoff = now - self.char_window_ns          # prune the front beyond the time window
        while len(hist) > 1 and hist[0][0] < cutoff:
            hist.popleft()
        if len(hist) < CHAR_MIN_SAMPLES:            # not enough span/samples yet -> stay SMOOTH
            return
        # ONE pass over the window: endpoints, Kaufman path length, range (no list materialisation)
        first = last = prev = 0.0
        path = mn = mx = 0.0
        for i, (_, px) in enumerate(hist):
            if i == 0:
                first = mn = mx = px
            else:
                path += abs(px - prev)
                if px < mn:
                    mn = px
                elif px > mx:
                    mx = px
            prev = px
            last = px
        net = abs(last - first)
        er = (net / path) if path > 1e-12 else 0.0  # Kaufman Efficiency Ratio
        full_spread_bps = max(2.0 * st.spread_ema_bps, SPREAD_FLOOR_BPS)
        rvs = ((mx - mn) / mid * 1e4 / full_spread_bps) if mid > 0 else 0.0   # range vs spread
        net_bps = (net / mid * 1e4) if mid > 0 else 0.0
        is_dir = (er >= ER_DIRT_ENTER and rvs >= RVS_ENTER and net_bps >= EXC_ENTER_BPS)
        st.dir_streak = min(st.dir_streak + 1, DIR_CONFIRM_STEPS) if is_dir else 0   # clamp (only the >= test matters)
        if st.char == CHAR_DIRECTIONAL:
            faded = (er <= ER_DIRT_EXIT or rvs <= RVS_EXIT)     # anti-flip exit band
            if faded and (now - st.char_since_ns) >= self.char_min_dwell_ns:
                st.char = CHAR_CHOP if rvs >= RVS_ENTER else CHAR_SMOOTH
                st.char_since_ns = now
        elif st.dir_streak >= DIR_CONFIRM_STEPS:    # entry confirmation: a 1-step spike can't latch DIRECTIONAL
            st.char = CHAR_DIRECTIONAL
            st.char_since_ns = now
        else:
            st.char = CHAR_CHOP if rvs >= RVS_ENTER else CHAR_SMOOTH

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

        # === V3 CHARACTER GATE (above the fee routing) ===
        # A DIRECTIONAL (one-sided trend) book must NOT open a passive maker lot — that is the V2 loss
        # (adverse selection -> the 15bps cut). Route it to the PatientTaker if a cross is ~+EV given the fee
        # regime, else IDLE (free-drop). SMOOTH/CHOP books fall through to the existing fee-based routing
        # (the two-sided never-cut maker cycles them — incl. high-amplitude CHOP). Applies uniformly to ALL
        # modes: routing is flat-gated upstream, so a flat PTAKER book on a still-directional book must STAY
        # ptaker/idle (not fee-route back into a maker lot); returning cur==ptaker is a harmless no-op flip.
        if st.char == CHAR_DIRECTIONAL:
            rt_cost_bps = 2.0 * half_spread_bps - 2.0 * max(rebate_bps, 0.0)   # net cost to cross both legs
            if MODE_PTAKER in ALLOWED_MODES and rt_cost_bps <= PT_MAX_RT_COST_BPS:
                return MODE_PTAKER                # cheap cross -> rebate-funded patient scale-out taker
            if not cliff:
                return MODE_IDLE if MODE_IDLE in ALLOWED_MODES else cur
            # At the CLIFF (idle overflow): a market-wide trend sends MANY books here at once and idling them
            # all blows the 48-book budget with no backstop (review #2). Forcing an expensive taker cross would
            # just guarantee a negative-kappa book, so instead FALL THROUGH to the fee router — which at the
            # cliff relaxes the maker enter-edge to 0 and rests a maker quote WHENEVER there is any spread edge
            # (half_spread >= maker_fee; true on essentially every real book). That maker earns the spread on
            # fills and holds adverse fills never-cut, realizing a loss only at the 15bps whole-side stop
            # (_managed_exit), which beats a 0.0 idle-median drag. If there is no spread edge it still idles.

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
        if event.bookId is None or event.quantity is None or event.price is None:
            return                              # malformed fill -> skip (onTrade isn't try/except-wrapped upstream)
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
        books = self.inv.get(validator)
        if books is None:
            books = self.inv[validator] = {}
        obj = books.get(book_id)
        if obj is None:                       # construct only on a miss (setdefault would build one every call)
            obj = books[book_id] = _Inv()
        return obj

    def _bstate(self, validator: str, book_id: int) -> _BookState:
        books = self.books_state.get(validator)
        if books is None:
            books = self.books_state[validator] = {}
        st = books.get(book_id)
        if st is None:                        # construct only on a miss (hot path: called many times per step)
            st = books[book_id] = _BookState()
        return st

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
            self._log_rt(validator, book_id,
                         hold_s=(ts - matched_ts) / _NS if matched_ts else None,
                         exit_px=price, rtv=rtv,
                         gross=gross, net=realized, kappa_before=kappa_before, kappa_after=st.kappa3)
        elif rtv == 0 and self._rt_log_enabled(validator) and (validator, book_id) not in self._rt_log:
            # Pure opening fill with no prior stash -> a passive maker quote got hit. Record context
            # now so the eventual closing RT logs a real open_reason/side instead of "?".
            st = self._bstate(validator, book_id)
            self._stash_open(validator, book_id, st.mode, "passive",
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
        # FIFO deque (appends are time-ordered) -> popleft the aged-out front. O(dropped), not an O(N) rebuild
        # every step. Only the <=600s pnl-backoff / <=570s rt_count windows read rt_events.
        cutoff = now - self.rt_events_retention_ns
        ev = st.rt_events
        dropped = False
        while ev and ev[0][0] < cutoff:
            ev.popleft()
            dropped = True
        return dropped

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

    def _book_pnl_series(self, validator: str, book_id: int, now: int,
                         timestamps: list[int] | None = None) -> list[float]:
        if timestamps is None:
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

    def _refresh_book_kappa(self, validator: str, book_id: int, now: int) -> None:
        st = self._bstate(validator, book_id)
        # Build the cross-book timestamp axis ONCE and reuse it for the readiness test AND the series builder
        # (was scanned 2-3x per call). kappa3 is logging-only, so this whole path is best-effort/cheap.
        timestamps = self._global_rt_timestamps(validator, now)
        if len(timestamps) < 2 or timestamps[-1] - timestamps[0] < self.kappa_min_lookback_ns:
            st.kappa3 = None
            if timestamps:
                bt.logging.info(
                    f"[AdaptiveRouterV3 uid={self.uid}] kappa_not_ready book={book_id} "
                    f"ts_count={len(timestamps)} span_s={(timestamps[-1]-timestamps[0])/1e9:.0f} "
                    f"need_span_s={self.kappa_min_lookback_ns/1e9:.0f}"
                )
            else:
                bt.logging.info(
                    f"[AdaptiveRouterV3 uid={self.uid}] kappa_no_events book={book_id}"
                )
            return
        pnl = self._book_pnl_series(validator, book_id, now, timestamps)
        result = self._kappa3_raw(pnl)
        if result is None:
            bt.logging.info(
                f"[AdaptiveRouterV3 uid={self.uid}] kappa_raw_none book={book_id} "
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
            st = self._bstate(validator, book_id)
            st.vol_log.append((ts_ns, vol))
            st.vol_sum += vol

    def _prune_vol_log(self, st: _BookState, now_ns: int) -> None:
        # FIFO deque (appends are time-ordered) -> popleft the aged-out front and decrement the running sum.
        # O(dropped) instead of rebuilding the whole list each call (review #13).
        cutoff = now_ns - self.volume_assessment_ns
        vl = st.vol_log
        while vl and vl[0][0] < cutoff:
            st.vol_sum -= vl.popleft()[1]
        if not vl:
            st.vol_sum = 0.0            # reset any float drift once the window empties

    def _rolled_quote_volume(self, validator: str, book_id: int, now_ns: int) -> float:
        st = self._bstate(validator, book_id)
        self._prune_vol_log(st, now_ns)
        return st.vol_sum

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
            f"[AdaptiveRouterV3 uid={self.uid}] priceDecimals={price_decimals} tick={self._tick} "
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

    def _stash_open(self, validator: str, book_id: int, mode: str,
                    reason: str, side: str) -> None:
        if not self._rt_log_enabled(validator):
            return
        self._rt_log[(validator, book_id)] = _RtLogCtx(mode=mode, open_reason=reason, side=side)

    @staticmethod
    def _fmt_kappa_pair(before: float | None, after: float | None) -> str:
        if before is None and after is None:
            return "n/a"
        if before is None:
            return f"n/a->{after:.4f}"
        if after is None:
            return f"{before:.4f}->n/a"
        return f"{before:.4f}->{after:.4f}"

    def _log_rt(self, validator, book_id, *, hold_s, exit_px, rtv, gross, net,
                kappa_before, kappa_after) -> None:
        if not self._rt_log_enabled(validator):
            return
        ctx = self._rt_log.pop((validator, book_id), _RtLogCtx())
        if ctx.mode == "?":
            ctx.mode = self._bstate(validator, book_id).mode   # maker fills open passively (no stash)
        hold_str = f"{hold_s:.2f}" if hold_s is not None else "n/a"
        bt.logging.info(
            f"[AdaptiveRouterV3 uid={self.uid} RT] book={book_id} mode={ctx.mode} "
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
    def _activity_close(self, agent, response, book_id, account, inv,
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
            agent._stash_open(validator, book_id, self.name, "drain",
                              "long" if net >= 0 else "short")
            self._activity_close(agent, response, book_id, account, inv,
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
                self._maybe_add(agent, response, validator, book_id, book, account, st, net,
                                best_ask, mid, volume_cap, now, vol_dp)
            return
        throttled = st.last_close_ns and (now - st.last_close_ns) < agent.tk_reopen_gap_ns
        if not throttled and self._open(agent, response, validator, book_id, book, account, st,
                                        best_bid, best_ask, mid, volume_cap, now, vol_dp):
            return
        if agent._activity_due(st, now):
            act_dir = agent._bias(book, mid)
            act_side = "long" if act_dir == OrderDirection.BUY else "short"
            agent._stash_open(validator, book_id, self.name, "activity", act_side)
            self._activity_close(agent, response, book_id, account, inv,
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
        agent._stash_open(validator, book_id, self.name, "scalp",
                          "long" if direction == OrderDirection.BUY else "short")
        return True

    def _maybe_add(self, agent, response, validator, book_id, book, account, st, net,
                   best_ask, mid, volume_cap, now, vol_dp) -> None:
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


class _PatientTakerMode(_Mode):
    """uid62 archetype — the V3 DIRECTIONAL response. Patient, rebate-funded, SCALE-OUT taker: enter one clip on
    the microprice lean, HOLD (no fast TP/SL), and unwind ONE clip per ~8s tick — reducing once the lot is old,
    inventory is high, OR the move turns against it (bleed WITH the move). Press the lean one clip only while it
    is young AND the move favors it (bounded). The ONLY hard stop is a catastrophe, scaled DOWN by inventory so
    a stacked reversal can't become a big cube. The rebate (both legs) funds the thin edge; the per-tick scale-out
    smooths the realised stream. NOT momentum-prediction — it harvests rebate and exits gradually."""

    name = MODE_PTAKER

    def step(self, agent, response, validator, book_id, book, account, st, inv, net,
             best_bid, best_ask, mid, vol_dp, volume_cap, now) -> None:
        if st.pt_last_act_ns and (now - st.pt_last_act_ns) < agent.pt_tick_ns:
            return   # per-book tick throttle (uid62 ~8s inter-action)
        if abs(net) >= agent.exch_min:
            self._manage(agent, response, validator, book_id, book, account, st, inv, net,
                         best_bid, best_ask, mid, vol_dp, volume_cap, now)
            return
        if self._open(agent, response, validator, book_id, book, account, st,
                      best_bid, best_ask, mid, volume_cap, now, vol_dp):
            st.pt_last_act_ns = now
            return
        if agent._activity_due(st, now):
            act_dir = agent._bias(book, mid)
            agent._stash_open(validator, book_id, self.name, "activity",
                              "long" if act_dir == OrderDirection.BUY else "short")
            if self._activity_close(agent, response, book_id, account, inv,
                                    best_bid, best_ask, vol_dp, direction=act_dir):
                st.pt_last_act_ns = now

    def _open(self, agent, response, validator, book_id, book, account, st,
              best_bid, best_ask, mid, volume_cap, now, vol_dp) -> bool:
        if not agent._budget_ok(validator, book_id, st, now, volume_cap):
            return False
        # re-check the regime-agnostic entry gate (route gated at flip; fees can drift): cross only if cheap
        taker_fee = agent._taker_fee_rate(account)
        rebate_bps = (-taker_fee * 1e4) if taker_fee is not None else 0.0
        half_spread_bps = (best_ask - best_bid) / mid * 0.5 * 1e4 if mid > 0 else 1e9
        if (2.0 * half_spread_bps - 2.0 * max(rebate_bps, 0.0)) > PT_MAX_RT_COST_BPS:
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
        agent._stash_open(validator, book_id, self.name, "patient",
                          "long" if direction == OrderDirection.BUY else "short")
        return True

    def _manage(self, agent, response, validator, book_id, book, account, st, inv, net,
                best_bid, best_ask, mid, vol_dp, volume_cap, now) -> None:
        clip = agent.clip
        inv_lots = abs(net) / clip if clip > 0 else 0.0
        if net > 0:
            avg = agent._side_avg(inv.longs); ts0 = inv.longs[0][0]
            underwater_bps = (avg - best_bid) / avg * 1e4 if avg > 0 else 0.0
        else:
            avg = agent._side_avg(inv.shorts); ts0 = inv.shorts[0][0]
            underwater_bps = (best_ask - avg) / avg * 1e4 if avg > 0 else 0.0
        hold_age = now - ts0
        # catastrophe-only stop, scaled DOWN by inventory -> bounded single-RT cube. Whole side, one IOC.
        if underwater_bps >= PT_CATASTROPHE_BPS / max(1.0, inv_lots):
            self._tag(agent, validator, book_id, "cut")
            agent._cancel_all(response, account, book_id)
            self._exit_side(agent, response, book_id, account, net, best_bid, best_ask,
                            vol_dp, qty=abs(net), slip_bps=MK_IOC_SLIPPAGE_BPS)
            st.pt_last_act_ns = now
            return
        # PROJECT RULE: never hold forever. Past the max-hold the patient bet hasn't paid -> force-close the
        # WHOLE side (the per-tick scale-out can stall on missed IOCs / pressing); bounds the bag, frees the book.
        if hold_age >= agent.pt_max_hold_ns:
            self._tag(agent, validator, book_id, "time")
            agent._cancel_all(response, account, book_id)
            self._exit_side(agent, response, book_id, account, net, best_bid, best_ask,
                            vol_dp, qty=abs(net), slip_bps=MK_IOC_SLIPPAGE_BPS)
            st.pt_last_act_ns = now
            return
        bias_with = ((agent._bias(book, mid) == OrderDirection.BUY) == (net > 0))   # microprice leans our way
        reduce_now = (hold_age >= agent.pt_min_hold_ns or inv_lots >= PT_SOFT_INV_LOTS or not bias_with)
        if reduce_now:
            self._tag(agent, validator, book_id, "revert")
            agent._cancel_all(response, account, book_id)
            self._exit_side(agent, response, book_id, account, net, best_bid, best_ask,
                            vol_dp, qty=min(abs(net), clip), slip_bps=PT_REDUCE_SLIP_BPS)   # scale out ONE clip
            st.pt_last_act_ns = now
        elif (underwater_bps <= 0.0 and inv_lots + 1.0 <= PT_MAX_INV_LOTS
              and agent._budget_ok(validator, book_id, st, now, volume_cap)):
            # press the lean ONE clip — only PYRAMID A WINNER (underwater<=0): never average down into a lot that
            # is already red, so a momentum-press can't stack into a reversal catastrophe (review #7)
            q = round(clip, vol_dp)                       # press the lean ONE clip (young + move favors)
            if q >= agent.exch_min:
                if net > 0 and agent._avail(account.quote_balance) >= q * best_ask:
                    agent._submit_market(response, book_id, OrderDirection.BUY, q,
                                         settlement=agent._loan_settlement(account))
                    st.pt_last_act_ns = now
                elif net < 0 and agent._avail(account.base_balance) >= q:
                    agent._submit_market(response, book_id, OrderDirection.SELL, q)
                    st.pt_last_act_ns = now

    @staticmethod
    def _exit_side(agent, response, book_id, account, net, best_bid, best_ask, vol_dp, *, qty, slip_bps) -> None:
        slip = slip_bps / 1e4
        pdp = agent._price_decimals
        if net > 0:                                       # long -> SELL near-touch IOC
            q = round(min(qty, agent._avail(account.base_balance)), vol_dp)
            if q < agent.exch_min:
                return
            agent._submit_limit(response, book_id, OrderDirection.SELL, q,
                                round(best_bid * (1.0 - slip), pdp), ioc=True, post_only=False)
        else:                                             # short -> BUY near-touch IOC
            buy_px = best_ask * (1.0 + slip)
            q_max = agent._avail(account.quote_balance) / buy_px if buy_px > 0 else qty
            q = round(min(qty, q_max), vol_dp)
            if q < agent.exch_min:
                return
            agent._submit_limit(response, book_id, OrderDirection.BUY, q,
                                round(best_ask * (1.0 + slip), pdp), ioc=True, post_only=False,
                                settlement=agent._loan_settlement(account))

    @staticmethod
    def _tag(agent, validator, book_id, reason) -> None:
        if agent._rt_log_enabled(validator):
            c = agent._rt_log.get((validator, book_id))
            if c is not None:
                c.close_reason = reason


class _MakerMode(_Mode):
    """Two-sided spread capture (mirrors UID 109/149). Flat -> quote both sides inside the touch.
    Holding -> work ONLY the reducing side, priced off the FIFO worst lot and walked toward the
    touch with age, so every consumed lot is round-trip-positive and we never bag. A managed exit
    IOC-cuts an aged/underwater lot with capped slippage; an activity backstop guarantees a close
    each window. The router only sends spread-rich books here, so passive capture is +EV."""

    name = MODE_MAKER

    def step(self, agent, response, validator, book_id, book, account, st, inv, net,
             best_bid, best_ask, mid, vol_dp, volume_cap, now) -> None:
        if (self._risk_trim(agent, response, book_id, account, net, mid,
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
                agent._stash_open(validator, book_id, self.name, "activity",
                                  "long" if net >= 0 else "short")
                if self._activity_close(agent, response, book_id, account, inv,
                                        best_bid, best_ask, vol_dp):
                    return
        desired = self._desired_quotes(agent, validator, book_id, account, inv, net,
                                       best_bid, best_ask, mid, volume_cap, now, vol_dp, st)
        self._reconcile(agent, response, account, book_id, desired)

    def _risk_trim(self, agent, response, book_id, account, net, mid,
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
        # PURE NEVER-CUT, WHOLE-SIDE, regardless of book character. A held loser is realized ONLY at the 15bps
        # catastrophe stop. The §10.1 offline kappa replay (tests/arv3_kappa_replay.py — real validator kappa_3
        # on real MAKER and TREND price paths) settled two earlier ideas, both DOMINATED by pure never-cut-15:
        #   (1) scaling the cut out one clip at a time — WORSE (Δ≈-0.012) on the mean-reverting home regime
        #       (fragments a cut into clustered negative RTs + forgoes the clean re-enter-lower on the revert);
        #   (2) a tighter 6bps stop on DIRECTIONAL books — WORSE in EVERY regime incl. trends (uid184 Δ-0.020),
        #       because real trends are noisy: a tight stop gets whipsawed into many small realized losses that
        #       the cube-downside kappa punishes harder than one rare 15bps cut, and trends often retrace <15bps.
        # The detector's value is ROUTING (don't OPEN a maker lot on a directional book — validated), NOT a
        # tighter held-lot cut. So the bps stop is a single 15bps catastrophe (replay-optimal). BUT — PROJECT
        # RULE: never hold forever. A held lot is ALSO force-cut once it has been held past MK_MAX_HOLD_S (the
        # sharp-dump→revert window); a lot still underwater after ~3min did NOT revert, so cutting it frees the
        # book to re-route and bounds the bag/tail (fixes the mode-stuck + critical-loss failure mode).
        stop_bps = MK_STOP_LOSS_BPS
        if net > 0:
            ts0, _, px0, _ = inv.longs[0]
            underwater = (px0 - best_bid) / px0 * 1e4 if px0 > 0 else 0.0
            timed_out = (now - ts0) >= agent.mk_max_hold_ns
            if underwater < stop_bps and not timed_out:   # hold for revert ONLY within BOTH the stop AND max-hold
                st.mk_ioc_miss_count = 0
                st.mk_ioc_prev_net = 0.0
                return False
            reason = "time" if (timed_out and underwater < stop_bps) else "cut"
            q = round(min(agent._long_qty(inv), agent._avail(account.base_balance)), vol_dp)
            if q < agent.exch_min:
                st.mk_ioc_miss_count = 0
                st.mk_ioc_prev_net = 0.0
                return False
            st.mk_ioc_prev_net = abs(net)
            if st.mk_ioc_miss_count >= 2:
                label = "IOC-CROSS" if st.mk_ioc_miss_count >= 4 else "IOC-ESCALATE"
                bt.logging.info(
                    f"[AdaptiveRouterV3 uid={agent.uid}] {label} book={book_id} "
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
            if underwater < stop_bps and not timed_out:   # hold for revert ONLY within BOTH the stop AND max-hold
                st.mk_ioc_miss_count = 0
                st.mk_ioc_prev_net = 0.0
                return False
            reason = "time" if (timed_out and underwater < stop_bps) else "cut"
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
                    f"[AdaptiveRouterV3 uid={agent.uid}] {label} book={book_id} "
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
                        best_bid, best_ask, mid, volume_cap, now, vol_dp, st) -> dict[int, tuple[float, float]]:
        maker_fee = agent._maker_fee_rate(account)
        pdp = agent._price_decimals
        desired: dict[int, tuple[float, float]] = {}
        fee_bps = (maker_fee * 1e4) if maker_fee is not None else 0.0
        floor_bps = MK_TP_FEE_MULT * fee_bps + (agent._tick / mid) * 1e4
        base_target = max(MK_TP_BPS, floor_bps) / 1e4

        base_bal, quote_bal = account.base_balance, account.quote_balance
        free_base = base_bal.free if base_bal else 0.0
        free_quote = quote_bal.free if quote_bal else 0.0
        base_avail = agent._avail(base_bal)
        quote_avail = agent._avail(quote_bal)

        spread = best_ask - best_bid
        improve = agent._tick if spread > 2 * agent._tick else 0.0
        bid_inside = round(best_bid + improve, pdp)
        ask_inside = round(best_ask - improve, pdp)
        if bid_inside >= ask_inside:
            bid_inside, ask_inside = round(best_bid, pdp), round(best_ask, pdp)

        # === Avellaneda-Stoikov inventory skew ===
        # Lean BOTH quotes against the open lot, sized by inventory(lots) × recent EWMA vol(bps), capped.
        # r = mid − q·γ·σ² in the bps-bounded form: LONG -> shift the pair DOWN (eager to sell, shy to add);
        # SHORT -> UP. Flat -> no shift. This is what the top two-sided makers do to mean-revert inventory.
        inv_lots = abs(net) / agent.clip if agent.clip > 0 else 0.0
        vol_bps = math.sqrt(st.vol_var) * 1e4 if st.vol_var > 0.0 else 0.0
        skew_bps = min(AS_MAX_SKEW_BPS, AS_GAMMA * inv_lots * vol_bps)
        sign = 1.0 if net > 0 else (-1.0 if net < 0 else 0.0)
        skew_frac = sign * skew_bps / 1e4
        bid_skew = round(bid_inside * (1.0 - skew_frac), pdp)
        ask_skew = round(ask_inside * (1.0 - skew_frac), pdp)
        if bid_skew <= 0:
            bid_skew = bid_inside
        if ask_skew <= 0:
            ask_skew = ask_inside
        toxic = bool(st.mk_streak_cooldown_until_ns and now < st.mk_streak_cooldown_until_ns)
        # CONTINUOUS two-sided: keep re-posting the spread-capture (ADD) leg while holding — UNLESS the book
        # is directional, inventory hit the soft cap, or the book is in a loss-streak pause (don't pile in).
        add_ok = (st.char != CHAR_DIRECTIONAL and inv_lots < MK_SOFT_INVENTORY_LOTS and not toxic
                  and agent._budget_ok(validator, book_id, st, now, volume_cap))

        if net >= agent.exch_min:
            # REDUCE side (SELL) — never-cut walk floored at BREAKEVEN; the A-S skew lets it rest closer to
            # market (faster exit) when inventory/vol is high, but _reduce_price never sells below entry.
            age = now - inv.longs[0][0]
            fifo_px = inv.longs[0][2]
            px = self._reduce_price(True, fifo_px, age, ask_skew, base_target, pdp, agent)
            q = round(min(agent._long_qty(inv), base_avail), vol_dp)
            if q >= agent.exch_min and px > 0:
                desired[OrderDirection.SELL] = (px, q)
            qa = round(agent.clip, vol_dp)   # ADD side (BUY) — skewed cheaper; off on a trend / over soft-cap
            if add_ok and qa >= agent.exch_min and bid_skew > 0 and free_quote >= qa * bid_skew:
                desired[OrderDirection.BUY] = (bid_skew, qa)
        elif net <= -agent.exch_min:
            age = now - inv.shorts[0][0]
            fifo_px = inv.shorts[0][2]
            px = self._reduce_price(False, fifo_px, age, bid_skew, base_target, pdp, agent)
            q_max = quote_avail / px if px > 0 else agent._short_qty(inv)
            q = round(min(agent._short_qty(inv), q_max), vol_dp)
            if q >= agent.exch_min and px > 0:
                desired[OrderDirection.BUY] = (px, q)
            qa = round(agent.clip, vol_dp)   # ADD side (SELL) — skewed dearer; off on a trend / over soft-cap
            if add_ok and qa >= agent.exch_min and free_base >= qa:
                desired[OrderDirection.SELL] = (ask_skew, qa)
        elif toxic:
            pass   # persistently toxic book (streak of losing cuts) -> long pause, stop quoting
        elif st.last_cut_ns > 0 and now - st.last_cut_ns < agent.mk_reentry_cooldown_ns:
            pass   # just cut a loser -> pause fresh entries so we don't re-bag a trend
        elif st.char == CHAR_DIRECTIONAL:
            pass   # book turned directional while flat -> don't open passive liquidity into a trend
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
    launch(AdaptiveRouterV3Agent)

