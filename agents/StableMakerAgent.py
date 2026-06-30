# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
StableMakerAgent — SELF-CONTAINED "stable-rank" two-sided maker for Subnet 79 (τaos).

GOAL: hold a reliable mid-pack rank (≈top-50) with LOW variance and LOW ops, NOT to win #1. The #1 spot
needs high-throughput rebate-taking in the episodic rebate windows (higher variance, constant retuning);
this agent deliberately trades the other side of that bargain — a single, simple, robust liquidity
provider that earns the spread when the spread pays and quietly idles when it doesn't.

It is PureMaker's proven maker engine (tiny clips, patient reduce-to-breakeven, bounded vol-stop, FIFO,
never-bag) made STANDALONE by folding in the ONE thing PureMaker delegated to AdaptiveRouter: a simple
per-book EDGE GATE. No regime detector, no router, no mode-switching — the gate *is* the regime filter.

WHY A MAKER AT ALL (kappa = per-book Sortino-3, which CUBES the downside → consistency ≫ magnitude):
  the validator's score is a market-making scorecard. It pays for a smooth, positive-tilted realized
  round-trip stream across many books — exactly what spread-capture produces. Volume/alpha/direction
  do not score; consistency does. And empirically ~98% of books are deeply mean-reverting (ranging),
  which is the ideal weather for a shopkeeper-style maker.

THE WHOLE STRATEGY IS 3 RULES:
  1) ONLY OPEN SHOP WHERE THE GAP PAYS — the BEST-K-by-edge GATE (new vs PureMaker). Rank books by their
     TRUE round-trip economics net_edge = full_spread − 2*maker_fee − ADVERSE_SEL_BPS (a winning RT banks
     ~the FULL spread, you pay the fee on BOTH legs, and adverse selection eats ~2.5bps — measured: gross
     +10.99 → net +0.02 over 5,350 RTs at fee ~4.25 ⇒ full_spread breakeven ~11bps; the naive half_spread−fee
     gate over-idled +EV wide books). Quote every +EV book — BUT never idle below MIN_ACTIVE_BOOKS (=80=128−48):
     free-drops idle books only up to int(0.375*128)=48; PAST that, idle books enter the median as 0.0 and
     the score COLLAPSES (verified in reward.py). So in a maker-PAYS fee spike we trade the 80 LEAST-BAD
     books rather than idle into a self-inflicted zero. Graceful degrade, never self-collapse. No detector.

  2) KEEP EVERY POSITION TINY — clip 0.26, inventory cap 1.5 lots, never average into a bag. Small
     size is the dominant lever: it bounds the single worst realized loss the cube punishes hardest.

  3) HOLD A LOSER PATIENTLY FOR THE BOUNCE, BUT NEVER FOREVER — reduce walks from the profit target
     toward BREAKEVEN with lot age (a late fill nets ~0, never gives away the spread). The 180s timer
     force-closes ONLY at breakeven-or-better (the revert has happened) and NEVER cuts at a loss; an
     underwater lot at the timer is HELD, not cut. A loss is realized only by the catastrophe vol-stop
     [18,25]bps or, rarely, the 510s activity backstop (the hard max-hold that bounds a lot the timer left
     holding) — together a stop AND a max-hold, per the project rule. This kills the manufactured small-loss
     tail (lots that revert just after 180s) without re-entering the wide-never-cut cube-bomb. Reentry cooldown.

QUOTING: inside-on-wide (improve = tick if spread > 2·tick) — best price → fills first → more RT
  density (needed to clear the kappa lookback gate). Keep-band tick/2 → repeg fast (dodge book-sweeps).

MECHANICS (per book, each step):
  prune/kappa/noise → managed exit → risk guard → activity backstop (held-only) → quote → reconcile.
  Managed exit runs FIRST so a stop is never blocked by the risk guard's early return.

  * Managed exit → IOC-cut the held side if underwater >= _stop_bps (vol-band) OR lot age >=
            EXIT_GIVEUP_S. Slippage-capped, with 4→8→18bps escalation if an IOC keeps missing.
  * Flat  → quote both sides inside-on-wide IFF the EDGE GATE passes AND the RT/volume budget clears;
            else IDLE (rests at kappa=None, free-dropped). Never force-seeds a flat book.
  * Hold  → only the reducing side; reduce walks target→breakeven, stop-floored. The gate does NOT
            force-close a held book — once in, the exit logic owns it (no thrash on a tightening spread).
  * Activity → a HELD position that outlives the window is force-closed (deep safety; the 180s giveup
            normally closes it first). FLAT books are never force-traded.
  * FIFO inventory mirrors the validator's _match_trade_fifo exactly (oldest-lot matching).

Provenance: PureMaker's proven tight-band maker engine + the standalone EDGE GATE. The never-cut,
  force-activity, at-touch, and reprice-cushion variants were each tested and dropped upstream.
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
QUOTE_LOT = 0.26                    # just ABOVE the 0.25 exchange min on purpose: a fee-paying BUY is
                                   # settled by shaving the fee out of the base received (exchange
                                   # ClearingManager: fees_base=roundUp(fee/price)), so a 0.25 buy
                                   # leaves ~0.2498 held — under the min and un-sellable, so the agent
                                   # MISSES the exit until it re-accumulates. 0.26 keeps the held lot
                                   # >= 0.25 after the shave => always closeable in one order.

# ---- inventory bounds ----
MAX_INVENTORY_LOTS = 1.5           # small cap: with the TIGHT stop each forced cut is already small;
                                   # size is the extra lever keeping any single realized loss bounded
                                   # (per the uid-60 study). We never average into the bag.
MAX_INVENTORY_EQUITY_FRAC = 0.10
RISK_TRIM_SLIPPAGE_BPS = 6.0

# ---- book gate: EDGE GATE, collapse-proof (the one addition vs PureMaker; makes the agent standalone) ----
# Quote a FLAT book only if it is BEST-K-by-edge. The gate has TWO parts, computed once per respond():
#   (a) net_edge_bps = full_spread − 2*maker_fee − ADVERSE_SEL_BPS  — the TRUE round-trip economics. A winning
#       RT banks ~the FULL spread (enter one touch, reduce toward the opposite touch), you pay the fee on BOTH
#       legs, and adverse selection / partial capture eats ~2.5bps. So ne=0 at full_spread = 2*fee + 2.5,
#       matching the measured breakeven (5,350 RTs grossed +10.99bps → netted +0.02bps at fee ~4.25bps ⇒
#       full_spread ~11bps). NOTE: gross is the FULL spread — using half_spread here would be ~2x too strict
#       (the OLD half_spread−fee>=1.5 gate over-idled +EV wide books). >=0 means genuinely +EV.
#   (b) AN N_ACTIVE FLOOR. The validator free-drops idle (kappa=None) books only up to
#       max_inactive = int(0.375*128) = 48; PAST that, each idle book enters the median as a literal 0.0 and
#       the score COLLAPSES to ~0 once active < ~40 (verified in reward.py). So idling is NOT "free past 48".
#       We therefore NEVER idle below MIN_ACTIVE_BOOKS (=128-48=80): we trade every +EV book, but if fewer
#       than 80 are +EV (a maker-PAYS fee spike) we trade the 80 LEAST-BAD by net_edge anyway — graceful
#       degrade, never self-collapse. (An active thin-positive book weakly dominates an idle 0.0.)
# Gates ENTRY only — a held book is owned by the exit logic regardless (no thrash on a tightening spread).
# Cheap-fee regime: all 128 are +EV -> trade all 128 (the floor never binds). Disable entirely with
# EDGE_GATE_ENABLED=False (then it trades all 128 like PureMaker).
EDGE_GATE_ENABLED = True
ADVERSE_SEL_BPS = 2.5              # adverse-selection + partial-capture haircut on a round-trip (measured)
MIN_ACTIVE_BOOKS = 80             # never quote fewer than this many books (= 128 - int(0.375*128) inactive
                                   # budget) so the kappa/pnl median can never collapse to 0 via injected zeros.
                                   # NOTE: this floors QUOTED books; some won't clear the validator's >=3-RT /
                                   # 90min kappa gate (attrition), so SCORED books can be < 80. If live data
                                   # shows >48 books resting at kappa=None in a spike, raise this to ~88-96.

# ---- profit target ----
# TIGHT-CUT base (= V1): take profit quickly. NOTE: in the maker-PAYS regime the fee floor dominates
# this base — effective TP = max(TP_BPS_BASE, TP_FEE_MULT × maker_fee_bps + a tick) ≈ 2×9bps ≈ 18bps,
# so a round-trip always covers both maker-fee legs. TP_BPS_BASE only binds if the regime flips to a
# maker-rebate (then we take the small 8bps capture rather than wait for more).
TP_BPS_BASE = 8.0
TP_FEE_MULT = 2.0                  # floor = 2× maker_fee (covers both legs + small buffer)
QUOTE_EXPIRY_S = 12.0

# ---- managed exit: BREAKEVEN-OR-BETTER timer + bounded catastrophe stop ----
# kappa-3 CUBES the downside (LPM3). Two failure modes bracket the design: (1) cutting losers TOO eagerly
# drags every RT to ~breakeven AND injects a small-loss tail (the cube punishes it); (2) a WIDE never-cut
# stop is a cube-bomb (a 35bps never-cut A/B scored kappa -0.0156 — one 35³ crater dwarfs many tiny gains).
# The win is in the MIDDLE: realize a loss ONLY at the bounded catastrophe stop; do NOT realize one at the
# time-cut. The 180s timer force-closes a lot ONLY when it can do so at BREAKEVEN-OR-BETTER (the revert has
# happened) — an underwater lot at the timer is HELD, not cut, and exits only via the stop or the 510s
# activity backstop (the hard max-hold). This removes the manufactured small-loss tail (the ~13% of lots
# that revert just after 180s) while preserving cadence (a reverted lot still closes, so the RT registers).
#   * STOP = vol-scaled band 18-25bps: the PRIMARY loss-realiser; cuts a genuine trend before it craters kappa.
#   * GIVEUP 180s: force-close ONLY at breakeven-or-better (else keep holding for the revert). NOT a cut.
#   * 510s activity backstop (_activity_close): the rare deep-safety max-hold; can realize a small sub-stop loss.
#   * the reduce walks only to BREAKEVEN, never the touch (see _reduce_price) — a late passive fill nets ~0.
EXIT_WALK_START_S = 30.0           # start walking reduce from target toward breakeven after 30s
EXIT_GIVEUP_S = 180.0              # breakeven-or-better force-close at 3min = the sharp-dump→revert window;
                                   # an underwater lot is NOT cut here (held to the stop / 510s backstop)
EXIT_STOP_LOSS_BPS = 18.0          # FLOOR of the vol-scaled stop band — above 20bps dump noise (tape)
EXIT_STOP_CAP_BPS = 25.0           # CAP of the band — catastrophe bound below cube-bomb zone
EXIT_STOP_NOISE_MULT = 6.0         # stop ≈ MULT × per-book per-step mid-noise(bps), clamped to band
NOISE_EWMA_ALPHA = 0.05            # EWMA weight for the per-book mid-noise estimate (~20-step memory)
EXIT_CUT_SLIPPAGE_BPS = 4.0        # initial IOC-cut price concession
EXIT_CUT_ESCALATE_BPS = 8.0        # escalated concession after 2+ consecutive IOC-cut misses: a
                                   # fixed-price IOC that doesn't cross on a fast/wide book re-fires
                                   # at the same price every step while the position bleeds.
EXIT_CUT_CROSS_BPS = 18.0          # wide-limit cross after 4+ misses — NOT a market order (uncapped
                                   # market fills risk catastrophic gaps); 18bps crosses almost any
                                   # normal spread while still bounding the realized slippage.
REENTRY_COOLDOWN_S = 120.0        # after a managed cut, wait before re-quoting: prevents
                                   # re-entering a trending book and taking another stop immediately.

# ---- idle / throttle backoff: NONE ----
# No loss-streak cooldown and no regime-backoff — a book is never idled or entry-paused; risk is
# bounded only on the exit side.

# ---- activity backstop (HELD positions only; flat books idle at kappa=None) ----
RT_WINDOW_S = 570.0
ACTIVITY_DEADLINE_S = 510.0        # HARD MAX-HOLD + deep safety: force-CLOSE a HELD position that has
                                   # outlived the window (kept < the validator's 600s grace). This is the
                                   # backstop that bounds an underwater lot the 180s breakeven-timer left
                                   # holding — together with the 18-25bps stop it satisfies the project rule
                                   # (every hold has BOTH a stop AND a max-hold). FLAT books are never
                                   # force-traded (idle at None, free up to the 48-book budget).
RT_MAX = 15                        # cap on round-trips per RT_WINDOW per book (anti-overtrade)
FORCE_TRIM_SLIPPAGE_BPS = 5.0

# ---- quoting: tight tick/2 repeg ----
REPRICE_KEEP_TICKS = 0.5           # keep-band = tick/2 (0.5×tick): repeg fast. A wider keep-band would
                                   # hold quotes resting in place as price drifts -> run over by
                                   # book-sweeps -> a catastrophic-walk loss tail that craters kappa.

# ---- volume cap ----
CAPITAL_TURNOVER_CAP = 10.0
VOLUME_SAFETY = 0.8
VOLUME_ASSESSMENT_NS = 86_400_000_000_000

# ---- kappa-3 (LOGGING-ONLY proxy — NOT the validator kappa; do not tune on it) ----
# This computes kappa over a SPARSE RT-only timestamp axis; the validator (taos/im/utils/kappa.py) uses a
# DENSE per-step zero-filled axis with a different MAD normalizer, so this OVERSTATES the real score (often
# 3-17x). It is never read by any trade/routing decision — only logged on MAIN_VALIDATOR for eyeballing.
# Judge live performance from the validator's /metrics kappa+placement, never from this number.
KAPPA_TAU = 0.0
KAPPA_MIN_OBS = 3
KAPPA_MIN_LOOKBACK_S = 5400.0
KAPPA_RT_HISTORY_S = 10_800.0

MAIN_VALIDATOR = "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF"


@dataclass
class _Inv:
    longs: deque = field(default_factory=deque)
    shorts: deque = field(default_factory=deque)


@dataclass
class _BookState:
    last_rt_ns: int = 0
    last_cut_ns: int = 0
    seen_ns: int = 0
    rt_events: list[tuple[int, float]] = field(default_factory=list)
    kappa3: float | None = None
    vol_log: list[tuple[int, float]] = field(default_factory=list)
    # per-book mid-noise estimate (EWMA of |Δmid| in bps) — scales the stop band
    last_mid: float = 0.0
    noise_bps: float = 0.0
    # managed-exit IOC escalation (anti-bleed)
    exit_miss_count: int = 0               # consecutive IOC-cut misses on the current position
    exit_prev_net: float = 0.0             # |net| at the last IOC-cut submit; detects a non-fill


class StableMakerAgent(FinanceSimulationAgent):

    def initialize(self) -> None:
        bt.logging.set_info()

        self.quote_lot = QUOTE_LOT
        self.exch_min = EXCHANGE_MIN_ORDER_SIZE
        self._flat_eps = 0.5 * 10 ** (-4)
        self._price_decimals: int | None = None
        self._volume_decimals: int | None = None
        self._tick = 0.01
        self.volume_assessment_ns = VOLUME_ASSESSMENT_NS

        jitter = ((self.uid * 2654435761) % 1000) / 1000.0
        self.tp_bps_base = TP_BPS_BASE * (0.92 + 0.16 * jitter)
        activity_s = ACTIVITY_DEADLINE_S * (0.92 + 0.08 * jitter)
        giveup_s = EXIT_GIVEUP_S * (0.9 + 0.2 * jitter)

        self.quote_expiry_ns = int(QUOTE_EXPIRY_S * _NS)
        self.rt_window_ns = int(RT_WINDOW_S * _NS)
        self.activity_deadline_ns = int(activity_s * _NS)
        self.exit_walk_start_ns = int(EXIT_WALK_START_S * _NS)
        self.exit_giveup_ns = int(giveup_s * _NS)
        self.reentry_cooldown_ns = int(REENTRY_COOLDOWN_S * _NS)
        self.kappa_rt_history_ns = int(KAPPA_RT_HISTORY_S * _NS)
        self.kappa_min_lookback_ns = int(KAPPA_MIN_LOOKBACK_S * _NS)

        self.inv: dict[str, dict[int, _Inv]] = {}
        self.books_state: dict[str, dict[int, _BookState]] = {}
        self._sim_id: dict[str, str] = {}
        self._step_ts_ns: dict[str, int] = {}
        self._active_validator: str | None = None

        bt.logging.info(
            f"[StableMaker uid={self.uid}] STABLE-MAKER lot={QUOTE_LOT} exch_min={self.exch_min} "
            f"edge_gate={'ON' if EDGE_GATE_ENABLED else 'OFF'}(full_spread-2*fee-{ADVERSE_SEL_BPS}bps, "
            f"floor>={MIN_ACTIVE_BOOKS}books) backoff=NONE "
            f"tp_base={self.tp_bps_base:.1f}bps tp_floor={TP_FEE_MULT}×fee "
            f"exit_walk={EXIT_WALK_START_S:.0f}-{giveup_s:.1f}s(->breakeven, NO loss-cut at timer) "
            f"stop_band=[{EXIT_STOP_LOSS_BPS:.0f},{EXIT_STOP_CAP_BPS:.0f}]bps×{EXIT_STOP_NOISE_MULT:.0f}noise "
            f"reentry={REENTRY_COOLDOWN_S}s "
            f"inv_cap={MAX_INVENTORY_LOTS}lot/{MAX_INVENTORY_EQUITY_FRAC:.0%}eq "
            f"activity={activity_s:.0f}s(max-hold) rt_max={RT_MAX} rt_log={MAIN_VALIDATOR[:8]}"
        )
        self._tune_gc()

    def _tune_gc(self) -> None:
        """RESPONSE-TIME (axon GC-pause mitigation, mirrors AdaptiveRouterV2): the asyncio/axon layer retains
        completed Task objects holding ~128-orderbook state, so every gen2 GC sweep rescans a large heap —
        pauses spike to tens of ms and can stretch handle() past the validator timeout. We own this process's
        GC. (1) history_len=0: the framework deep-copies the FULL 128-book state every step and keeps 10
        (self.history) which we never read — skip it; (2) gc.freeze(): exclude the permanent import heap from
        every sweep; (3) raise thresholds: gen2 sweeps far less often. All behaviour-neutral."""
        self.history_len = 0
        try:
            gc.collect()
            gc.freeze()
            gc.set_threshold(50_000, 500, 500)
            bt.logging.info(f"[StableMaker uid={self.uid}] gc tuned: frozen={gc.get_freeze_count()} "
                            f"thresholds={gc.get_threshold()} history_len=0")
        except Exception as ex:
            bt.logging.warning(f"[StableMaker uid={self.uid}] gc tune skipped: {ex}")

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
            f"[StableMaker uid={self.uid}] new simulation: {validator[:8]} sim_id={simulation_id}"
        )

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        validator = state.dendrite.hotkey
        cfg = self.simulation_config
        self._sync_precision(cfg.priceDecimals, cfg.volumeDecimals)

        vol_dp = cfg.volumeDecimals
        volume_cap = CAPITAL_TURNOVER_CAP * cfg.miner_wealth * VOLUME_SAFETY
        now = state.timestamp

        # Best-K-by-edge gate floor: one O(B) pre-pass to find the net_edge a FLAT book must clear to be
        # quoted this step (= min(0, edge of the MIN_ACTIVE_BOOKS-th best book)). Guarantees we never quote
        # fewer than MIN_ACTIVE_BOOKS, so the kappa/pnl median can never collapse via injected zeros.
        gate_min_edge = self._compute_gate_min_edge(state)

        for book_id in sorted(self.accounts.keys()):
            book = state.books.get(book_id)
            account = self.accounts.get(book_id) if book else None
            if book is None or account is None:
                continue
            try:
                self._step_book(response, validator, book_id, book, account,
                                vol_dp, volume_cap, now, gate_min_edge)
            except Exception as ex:
                bt.logging.warning(f"[StableMaker uid={self.uid}] step {book_id}: {ex}")

        return response

    def _compute_gate_min_edge(self, state: MarketSimulationStateUpdate) -> float:
        """Return the net_edge_bps a FLAT book must clear to be quoted = min(0, edge of the
        MIN_ACTIVE_BOOKS-th best book). Every +EV book (edge>=0) clears; if fewer than MIN_ACTIVE_BOOKS are
        +EV we still admit the least-bad ones down to the floor (never idle below it -> never self-collapse).
        Returns -inf if the gate is disabled or fewer than MIN_ACTIVE_BOOKS books are even rankable."""
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
            return float("-inf")        # can't reach the floor -> admit every rankable book
        edges.sort(reverse=True)
        return min(0.0, edges[MIN_ACTIVE_BOOKS - 1])

    # ------------------------------------------------------------------ per-book
    def _step_book(
        self, response, validator: str, book_id: int, book, account,
        vol_dp: int, volume_cap: float, now: int, gate_min_edge: float,
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
        # Per-book mid-noise EWMA (bps/step) — the scale for the vol-adaptive stop band. Regime-robust:
        # small on calm books (stop stays at the floor), large on volatile ones (stop widens so we
        # don't cut positions that are merely oscillating, not trending).
        if st.last_mid > 0.0:
            inst = abs(mid - st.last_mid) / st.last_mid * 1e4
            st.noise_bps = (((1.0 - NOISE_EWMA_ALPHA) * st.noise_bps + NOISE_EWMA_ALPHA * inst)
                            if st.noise_bps > 0.0 else inst)
        st.last_mid = mid
        if self._prune_rt_events(st, now):
            self._refresh_book_kappa(validator, book_id, now)

        net = self._net_qty(inv)
        maker_fee = self._maker_fee_rate(account)

        # 1) MANAGED EXIT — stop-loss / giveup takes priority over everything, including the
        #    inventory risk guard. Without this ordering, _risk_trim's early-return blocks
        #    _managed_exit on any over-cap book, leaving underwater positions with no stop-loss
        #    protection until the trim IOC finally fills (which can take hundreds of steps on
        #    illiquid books, letting losses grow to hundreds of bps).
        if self._managed_exit(response, book_id, account, inv, net, best_bid, best_ask, vol_dp, now, st):
            st.last_cut_ns = now
            return

        # 2) RISK GUARD — drains breached inventory after stop-loss has had its turn.
        if self._risk_trim(response, book_id, account, net, mid, vol_dp):
            return

        # 3) ACTIVITY BACKSTOP / HARD MAX-HOLD (held-only) — if a HELD position has outlived the window,
        #    IOC-close it to register the round-trip and bound the hold (this is the max-hold that backstops
        #    the breakeven-timer, which leaves an underwater lot holding). FLAT books return False here (idle
        #    at kappa=None) — never force-seeds. The 510s deadline stays < the validator's 600s decay grace.
        activity_ref = st.last_rt_ns if st.last_rt_ns > 0 else st.seen_ns
        if (now - activity_ref) >= self.activity_deadline_ns:
            if self._activity_close(response, book_id, account,
                                    inv, net, best_bid, best_ask, vol_dp):
                return

        # 4) DESIRED QUOTES — reduce-only when holding; two-sided entry when the budget gates (_entry_ok:
        #    RT count + volume cap) AND the best-K-by-edge gate (gate_min_edge) both clear.
        desired = self._desired_quotes(
            validator, book_id, account, inv, net,
            best_bid, best_ask, mid, maker_fee, volume_cap, now, gate_min_edge,
        )
        self._reconcile_quotes(response, account, book_id, desired)

    # ------------------------------------------------------------------ risk guard
    def _risk_trim(
        self, response, book_id: int, account, net: float,
        mid: float, vol_dp: int,
    ) -> bool:
        qty = abs(net)
        if qty < self._flat_eps:
            return False
        lot_cap = MAX_INVENTORY_LOTS * self.quote_lot
        equity = self._book_equity(account, mid)
        notional_cap = MAX_INVENTORY_EQUITY_FRAC * equity if equity > 0 else float("inf")
        over_lots = qty - lot_cap
        over_notional = (qty * mid - notional_cap) / mid if mid > 0 else 0.0
        excess = max(over_lots, over_notional)
        if excess <= self._flat_eps:
            return False
        trim = round(min(qty, max(excess, self.exch_min)), vol_dp)
        if trim < self.exch_min:
            return False
        slip = RISK_TRIM_SLIPPAGE_BPS / 1e4
        if net > 0:
            trim = round(min(trim, self._avail(account.base_balance)), vol_dp)
        else:
            buy_px = mid * (1.0 + slip)
            q_max = self._avail(account.quote_balance) / buy_px if buy_px > 0 else trim
            trim = round(min(trim, q_max), vol_dp)
        if trim < self.exch_min:
            return False
        self._cancel_all(response, account, book_id)
        if net > 0:
            px = round(mid * (1.0 - slip), self._price_decimals)
            self._submit_limit(response, book_id, OrderDirection.SELL, trim, px,
                               ioc=True, post_only=False)
        else:
            px = round(mid * (1.0 + slip), self._price_decimals)
            self._submit_limit(response, book_id, OrderDirection.BUY, trim, px,
                               ioc=True, post_only=False, settlement=self._loan_settlement(account))
        bt.logging.info(
            f"[StableMaker uid={self.uid}] RISK-TRIM book={book_id} net={net:+.4f} trim={trim}"
        )
        return True

    # ------------------------------------------------------------------ vol-scaled stop band
    def _stop_bps(self, st: _BookState) -> float:
        """Vol-scaled catastrophe stop within [FLOOR, CAP] = [18, 25]bps: cut a genuine trend before it
        craters kappa, but let a more volatile book breathe up to the CAP so we don't cut on pure noise.
        FLOOR is the calm-book default; CAP bounds a single realized loss below the cube-bomb zone. This is
        the PRIMARY loss-realiser (the 180s timer never cuts at a loss; the 510s backstop is a rare
        deep-safety that can also realize a small sub-stop loss). Adapts to each book's oscillation."""
        scaled = EXIT_STOP_NOISE_MULT * st.noise_bps
        return min(EXIT_STOP_CAP_BPS, max(EXIT_STOP_LOSS_BPS, scaled))

    # ------------------------------------------------------------------ managed exit
    def _managed_exit(
        self, response, book_id: int, account, inv: _Inv, net: float,
        best_bid: float, best_ask: float, vol_dp: int, now: int, st: _BookState,
    ) -> bool:
        """Realize the held side on ONE of two triggers — and ONLY one of them ever realizes a loss:
          * STOPPED  (uw >= the 18-25bps vol-stop): a genuine trend; IOC-cut at an escalating concession to
                     guarantee the exit (the catastrophe loss-realiser, bounded below the cube-bomb zone).
          * AGED_BE  (oldest lot past EXIT_GIVEUP_S AND already at BREAKEVEN-OR-BETTER, uw <= 0): the revert
                     has happened; close at the touch (slip 0) to bank the round-trip — NOT a loss.
        An aged lot that is still UNDERWATER (timer hit but uw in (0, stop)) is NEITHER — it is HELD for the
        revert, bounded by the stop above and the 510s activity backstop (the hard max-hold). This removes
        the manufactured small-loss tail (lots that revert just after 180s) while preserving cadence (a
        reverted lot still closes). STOPPED and AGED_BE are mutually exclusive (can't be >=18bps under and
        <=0 under at once)."""
        if abs(net) < self.exch_min:
            # Flat — clear escalation state so the NEXT position starts a fresh miss streak.
            st.exit_miss_count = 0
            st.exit_prev_net = 0.0
            return False
        stop_bps = self._stop_bps(st)            # vol-scaled: hold through noise, cut on trend
        # Escalate the STOP concession on consecutive IOC-cut misses: a fixed-price IOC that doesn't cross
        # on a fast/wide book re-fires every step while the position bleeds. Escalating 4→8→18bps caps the
        # loss window; the final stage is a wide LIMIT (not a market order) to bound gap fills. (Applies to
        # the STOPPED path only — the AGED_BE breakeven close always uses slip 0, never concedes.)
        if st.exit_prev_net > 0:
            if abs(net) >= st.exit_prev_net - self._flat_eps:
                st.exit_miss_count += 1          # |net| didn't shrink => the last IOC missed
            else:
                st.exit_miss_count = 0           # partial/full fill => streak broken
                st.exit_prev_net = 0.0
        if st.exit_miss_count >= 4:
            slip_stop = EXIT_CUT_CROSS_BPS / 1e4
        elif st.exit_miss_count >= 2:
            slip_stop = EXIT_CUT_ESCALATE_BPS / 1e4
        else:
            slip_stop = EXIT_CUT_SLIPPAGE_BPS / 1e4
        if net > 0:
            ts, _, px0, _ = inv.longs[0]
            uw = (px0 - best_bid) / px0 * 1e4 if px0 > 0 else 0.0
            stopped = uw >= stop_bps
            aged_be = (now - ts >= self.exit_giveup_ns) and uw <= 0.0   # timer + breakeven-or-better
            if not (stopped or aged_be):
                # Inside the band (held for revert) OR timer hit while still underwater (held, NOT cut) —
                # break the miss streak so a later stop event starts fresh escalation.
                st.exit_miss_count = 0
                st.exit_prev_net = 0.0
                return False
            slip = slip_stop if stopped else 0.0   # breakeven close sells AT the touch, never concedes
            q = round(min(self._long_qty(inv), self._avail(account.base_balance)), vol_dp)
            if q < self.exch_min:
                # Non-submit step — break the miss streak. A step that sends no order must not be
                # counted as an IOC miss, or repeated sub-lot/low-balance steps inflate the count
                # and the next real cut crosses wider than warranted (and slip can't fix a qty gap).
                st.exit_miss_count = 0
                st.exit_prev_net = 0.0
                return False
            # Track misses ONLY on the STOPPED path — a breakeven (aged_be) close that misses must NOT
            # pre-charge the catastrophe-stop escalation, so a later genuine stop starts fresh at 4bps.
            if stopped:
                st.exit_prev_net = abs(net)
            else:
                st.exit_miss_count, st.exit_prev_net = 0, 0.0
            self._cancel_all(response, account, book_id)
            px = round(best_bid * (1.0 - slip), self._price_decimals)
            self._submit_limit(response, book_id, OrderDirection.SELL, q, px, ioc=True, post_only=False)
        else:
            ts, _, px0, _ = inv.shorts[0]
            uw = (best_ask - px0) / px0 * 1e4 if px0 > 0 else 0.0
            stopped = uw >= stop_bps
            aged_be = (now - ts >= self.exit_giveup_ns) and uw <= 0.0   # timer + breakeven-or-better
            if not (stopped or aged_be):
                st.exit_miss_count = 0
                st.exit_prev_net = 0.0
                return False
            slip = slip_stop if stopped else 0.0
            buy_px = best_ask * (1.0 + slip)
            q_max = self._avail(account.quote_balance) / buy_px if buy_px > 0 else self._short_qty(inv)
            q = round(min(self._short_qty(inv), q_max), vol_dp)
            if q < self.exch_min:
                # Non-submit step — break the miss streak (see long branch).
                st.exit_miss_count = 0
                st.exit_prev_net = 0.0
                return False
            if stopped:                          # see long branch — misses tracked on STOPPED only
                st.exit_prev_net = abs(net)
            else:
                st.exit_miss_count, st.exit_prev_net = 0, 0.0
            self._cancel_all(response, account, book_id)
            px = round(best_ask * (1.0 + slip), self._price_decimals)
            self._submit_limit(response, book_id, OrderDirection.BUY, q, px, ioc=True, post_only=False,
                               settlement=self._loan_settlement(account))
        reason = "stop" if stopped else "age-be"
        if st.exit_miss_count >= 2:
            stage = "IOC-CROSS" if st.exit_miss_count >= 4 else "IOC-ESCALATE"
            bt.logging.info(
                f"[StableMaker uid={self.uid}] {stage} book={book_id} "
                f"miss={st.exit_miss_count} slip={slip*1e4:.0f}bps"
            )
        bt.logging.info(
            f"[StableMaker uid={self.uid}] MANAGED-EXIT book={book_id} reason={reason} "
            f"net={net:+.4f} q={q} @~{px} uw={uw:.1f}bps stop={stop_bps:.0f}bps "
            f"noise={st.noise_bps:.1f}bps"
        )
        return True

    # ------------------------------------------------------------------ activity backstop
    def _activity_close(
        self, response, book_id: int, account, inv: _Inv, net: float,
        best_bid: float, best_ask: float, vol_dp: int,
    ) -> bool:
        long_q = self._long_qty(inv)
        short_q = self._short_qty(inv)
        # FLAT book → do nothing (idle). The validator free-drops 0-RT books (kappa=None) and never
        # applies the activity factor to a None book, so forcing a round-trip on a dead/edgeless book
        # only manufactures a guaranteed small loss → a negative-kappa scored book that drags the
        # median. So we never force-seed a flat book; it rests at None (free) and keeps quoting below.
        # Only a HELD position is force-closed here (the held-only activity backstop, a deep safety).
        if long_q < self.exch_min and short_q < self.exch_min:
            return False
        slip = FORCE_TRIM_SLIPPAGE_BPS / 1e4
        lot = max(self.quote_lot, self.exch_min)
        if long_q >= self.exch_min:
            q = round(min(long_q, self._avail(account.base_balance), lot), vol_dp)
            if q < self.exch_min:
                return False
            px = round(best_bid * (1.0 - slip), self._price_decimals)
            self._cancel_all(response, account, book_id)
            self._submit_limit(response, book_id, OrderDirection.SELL, q, px, ioc=True, post_only=False)
        else:
            buy_px = best_ask * (1.0 + slip)
            q_max = self._avail(account.quote_balance) / buy_px if buy_px > 0 else short_q
            q = round(min(short_q, lot, q_max), vol_dp)
            if q < self.exch_min:
                return False
            px = round(best_ask * (1.0 + slip), self._price_decimals)
            self._cancel_all(response, account, book_id)
            self._submit_limit(response, book_id, OrderDirection.BUY, q, px, ioc=True, post_only=False,
                               settlement=self._loan_settlement(account))
        bt.logging.info(f"[StableMaker uid={self.uid}] ACTIVITY-CLOSE book={book_id} net={net:+.4f}")
        return True

    # ------------------------------------------------------------------ quoting
    def _desired_quotes(
        self, validator: str, book_id: int, account, inv: _Inv, net: float,
        best_bid: float, best_ask: float, mid: float, maker_fee: float | None,
        volume_cap: float, now: int, gate_min_edge: float,
    ) -> dict[int, tuple[float, float]]:
        st = self._bstate(validator, book_id)
        tick = self._tick
        pdp, vdp = self._price_decimals, self._volume_decimals
        desired: dict[int, tuple[float, float]] = {}

        # Adaptive profit target: base OR 2× maker fee (ensures each RT covers both legs).
        fee_bps = (maker_fee * 1e4) if maker_fee is not None else 0.0
        tp_floor_bps = TP_FEE_MULT * fee_bps + (tick / mid) * 1e4
        base_target = max(self.tp_bps_base, tp_floor_bps) / 1e4
        stop_bps = self._stop_bps(st)            # vol-scaled stop floor for the resting reduce

        free_base = account.base_balance.free if account.base_balance else 0.0
        free_quote = account.quote_balance.free if account.quote_balance else 0.0
        base_avail = self._avail(account.base_balance)
        quote_avail = self._avail(account.quote_balance)

        spread = best_ask - best_bid       # inside-on-wide: step 1 tick inside ONLY when the spread is
        improve = tick if spread > 2 * tick else 0.0   # wide; best price → fills first → more RT
                                           # density (needed to clear the kappa lookback gate).
        bid_inside = round(best_bid + improve, pdp)
        ask_inside = round(best_ask - improve, pdp)
        if bid_inside >= ask_inside:
            bid_inside, ask_inside = round(best_bid, pdp), round(best_ask, pdp)

        if net >= self.exch_min:
            # Holding long → passive SELL reduce, priced off FIFO-next (oldest) lot.
            age = now - inv.longs[0][0]
            fifo_px = inv.longs[0][2]
            px = self._reduce_price(True, fifo_px, age, ask_inside, base_target,
                                    tp_floor_bps, stop_bps, pdp)
            q = round(min(self._long_qty(inv), base_avail), vdp)
            if q >= self.exch_min and px > 0:
                desired[OrderDirection.SELL] = (px, q)
        elif net <= -self.exch_min:
            # Holding short → passive BUY reduce, priced off FIFO-next (oldest) lot.
            age = now - inv.shorts[0][0]
            fifo_px = inv.shorts[0][2]
            px = self._reduce_price(False, fifo_px, age, bid_inside, base_target,
                                    tp_floor_bps, stop_bps, pdp)
            q_max = quote_avail / px if px > 0 else self._short_qty(inv)
            q = round(min(self._short_qty(inv), q_max), vdp)
            if q >= self.exch_min and px > 0:
                desired[OrderDirection.BUY] = (px, q)
        elif st.last_cut_ns > 0 and now - st.last_cut_ns < self.reentry_cooldown_ns:
            # Post-cut cooldown: don't re-enter immediately after a managed exit.
            pass
        elif (self._entry_ok(validator, book_id, st, now, volume_cap)
              and self._gate_ok(best_bid, best_ask, mid, maker_fee, gate_min_edge)):
            # Flat, budget gates clear, AND the best-K-by-edge gate passes → quote both sides. A book below
            # the gate floor IDLES (kappa=None, free-dropped up to 48) — standalone selectivity that can
            # never idle below MIN_ACTIVE_BOOKS (so the median can't collapse). Runs without AR's selector.
            q = round(self.quote_lot, vdp)
            if q >= self.exch_min and free_base >= q:
                desired[OrderDirection.SELL] = (ask_inside, q)
            if q >= self.exch_min and free_quote >= q * bid_inside:
                desired[OrderDirection.BUY] = (bid_inside, q)

        return desired

    def _reduce_price(
        self, is_long: bool, px0: float, age_ns: int, touch_inside: float,
        base_target: float, walk_floor_bps: float, stop_bps: float, pdp: int,
    ) -> float:
        """Walk the passive-reduce limit from the profit target toward BREAKEVEN with lot age — never
        to the touch. The uid-60 study showed walking to the touch gives away the spread: late fills
        realize a loss and tank both win-rate and the kappa mean. Walking only to breakeven means a
        late fill nets ~0 (covers fees), keeping the kappa downside sparse; positions that can't fill
        at breakeven are HELD for reversion and exit via _managed_exit only on a genuine stop/giveup.
        The hard stop floor remains as a backstop so a resting reduce never fills worse than the IOC."""
        w = self._exit_walk(age_ns)
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

    def _exit_walk(self, age_ns: int) -> float:
        if age_ns <= self.exit_walk_start_ns:
            return 0.0
        if age_ns >= self.exit_giveup_ns:
            return 1.0
        span = self.exit_giveup_ns - self.exit_walk_start_ns
        return (age_ns - self.exit_walk_start_ns) / span if span > 0 else 1.0

    def _entry_ok(
        self, validator: str, book_id: int,
        st: _BookState, now: int, volume_cap: float,
    ) -> bool:
        """Budget gate for new inventory: RT count + volume cap only (the EDGE GATE is separate)."""
        vol_ok = self._rolled_quote_volume(validator, book_id, now) < volume_cap
        rt_ok = self._rt_count(st, now) < RT_MAX
        return vol_ok and rt_ok

    def _net_edge_bps(
        self, best_bid: float, best_ask: float, mid: float, maker_fee: float | None,
    ) -> float | None:
        """The TRUE per-round-trip economics in bps: full_spread − 2*maker_fee − ADVERSE_SEL_BPS. A winning
        RT enters at one touch and reduces toward the opposite touch (_reduce_price), banking ~the FULL
        spread (not a single half-spread); you pay the maker fee on BOTH legs; adverse selection / partial
        capture eats ~2.5bps. So ne=0 at full_spread = 2*fee + 2.5 — matching the measured breakeven (gross
        +10.99 → net +0.02 over 5,350 RTs at fee ~4.25bps ⇒ full_spread ~11bps). >=0 means genuinely +EV.
        Returns None on an invalid book or UNKNOWN fee (fail-safe: an unrankable book is excluded from the
        +EV set and the floor — it idles until the fee is known, transient at startup)."""
        if mid <= 0 or best_ask <= best_bid or maker_fee is None:
            return None
        full_spread_bps = (best_ask - best_bid) / mid * 1e4
        fee_bps = maker_fee * 1e4
        return full_spread_bps - 2.0 * fee_bps - ADVERSE_SEL_BPS

    def _gate_ok(
        self, best_bid: float, best_ask: float, mid: float, maker_fee: float | None,
        gate_min_edge: float,
    ) -> bool:
        """BEST-K-by-edge ENTRY gate. Quote a flat book iff its net_edge clears this step's floor
        (gate_min_edge = min(0, edge of the MIN_ACTIVE_BOOKS-th best book), from _compute_gate_min_edge).
        So every +EV book trades, and in a fee spike we still admit the least-bad books down to the floor —
        never idling below MIN_ACTIVE_BOOKS, so the kappa/pnl median can never collapse. Gates ENTRY only."""
        if not EDGE_GATE_ENABLED:
            return True
        ne = self._net_edge_bps(best_bid, best_ask, mid, maker_fee)
        return ne is not None and ne >= gate_min_edge

    def _reconcile_quotes(
        self, response, account, book_id: int, desired: dict[int, tuple[float, float]],
    ) -> None:
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

    # ------------------------------------------------------------------ events
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

    # ------------------------------------------------------------------ FIFO state
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
            f"[StableMaker uid={self.uid}] priceDecimals={price_decimals} tick={self._tick} "
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

    def _net_qty(self, inv: _Inv) -> float:
        return self._long_qty(inv) - self._short_qty(inv)

    @staticmethod
    def _avail(balance) -> float:
        if balance is None:
            return 0.0
        return (balance.free or 0.0) + (balance.reserved or 0.0)

    def _book_equity(self, account, mid: float) -> float:
        return self._avail(account.quote_balance) + self._avail(account.base_balance) * mid

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

    def _apply_fill(
        self, validator: str, book_id: int, is_buy: bool, qty: float, price: float,
        fee: float, ts: int,
    ) -> None:
        inv = self._inv(validator, book_id)
        realized, rtv, matched_ts, gross = self._match_fifo(inv, is_buy, qty, price, fee, ts)
        if rtv > 0:
            st = self._bstate(validator, book_id)
            kappa_before = st.kappa3
            rt_window_n = self._rt_count(st, ts)
            st.last_rt_ns = ts
            self._record_rt_close(validator, book_id, ts, realized)
            self._log_rt(
                validator=validator, book_id=book_id, ts=ts,
                hold_s=(ts - matched_ts) / _NS if matched_ts is not None else None,
                side="buy" if is_buy else "sell", exit_px=price, rtv=rtv,
                gross_pnl=gross, net_pnl=realized,
                kappa_before=kappa_before, kappa_after=st.kappa3,
                rt_window_n=rt_window_n, st=st,
            )

    def _match_fifo(
        self, inv: _Inv, is_buy: bool, qty: float, price: float, fee: float, ts: int,
    ) -> tuple[float, float, int | None, float]:
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

    def _record_rt_close(self, validator: str, book_id: int, ts: int, net_pnl: float) -> None:
        st = self._bstate(validator, book_id)
        self._prune_rt_events(st, ts)
        st.rt_events.append((ts, net_pnl))
        self._refresh_book_kappa(validator, book_id, ts)

    def _global_rt_timestamps(self, validator: str, now: int) -> list[int]:
        cutoff = now - self.kappa_rt_history_ns
        ts_set: set[int] = set()
        for st in self.books_state.get(validator, {}).values():
            for ts, _ in st.rt_events:
                if ts >= cutoff:
                    ts_set.add(ts)
        return sorted(ts_set)

    def _book_pnl_series(
        self, validator: str, book_id: int, now: int,
        timestamps: list[int] | None = None,
    ) -> list[float]:
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
        mid = len(s) // 2
        return s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])

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
        # kappa3 is a LOGGING-ONLY proxy logged solely on MAIN_VALIDATOR — skip the O(B²·E) cross-book scan
        # entirely on every other validator (pure per-step overhead otherwise; never read by a decision).
        if validator != MAIN_VALIDATOR:
            return
        st = self._bstate(validator, book_id)
        timestamps = self._global_rt_timestamps(validator, now)
        if len(timestamps) < 2 or timestamps[-1] - timestamps[0] < self.kappa_min_lookback_ns:
            st.kappa3 = None
            return
        st.kappa3 = self._kappa3_raw(self._book_pnl_series(validator, book_id, now, timestamps))

    def _rt_count(self, st: _BookState, now: int) -> int:
        cutoff = now - self.rt_window_ns
        return sum(1 for ts, _ in st.rt_events if ts >= cutoff)

    # ------------------------------------------------------------------ RT logging
    @staticmethod
    def _rt_log_enabled(validator: str) -> bool:
        return validator == MAIN_VALIDATOR

    @staticmethod
    def _fmt_kappa(before: float | None, after: float | None) -> str:
        if before is None and after is None:
            return "n/a"
        if before is None:
            return f"n/a->{after:.4f}"
        if after is None:
            return f"{before:.4f}->n/a"
        delta = after - before
        return f"{before:.4f}->{after:.4f} ({'+' if delta >= 0 else ''}{delta:.4f})"

    def _fmt_rt_pnl_list(self, st: _BookState, now: int) -> str:
        cutoff = now - self.rt_window_ns
        pnls = [p for ts, p in st.rt_events if ts >= cutoff]
        return "[" + ", ".join(f"{p:+.4f}" for p in pnls) + "]" if pnls else "[]"

    def _log_rt(
        self, *, validator: str, book_id: int, ts: int, hold_s: float | None,
        side: str, exit_px: float, rtv: float, gross_pnl: float, net_pnl: float,
        kappa_before: float | None, kappa_after: float | None, rt_window_n: int,
        st: _BookState,
    ) -> None:
        if not self._rt_log_enabled(validator):
            return
        hold_str = f"{hold_s:.2f}" if hold_s is not None else "n/a"
        bt.logging.info(
            f"[StableMaker uid={self.uid} RT] book={book_id} close={side} "
            f"rtv={rtv:.4f} exit={exit_px:.4f} hold_s={hold_str} "
            f"gross={gross_pnl:+.4f} net={net_pnl:+.4f} "
            f"kappa={self._fmt_kappa(kappa_before, kappa_after)} "
            f"rt_n={rt_window_n} pnls={self._fmt_rt_pnl_list(st, ts)}"
        )

    # ------------------------------------------------------------------ helpers
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

    def _cancel_all(self, response, account, book_id: int) -> None:
        if account.orders:
            response.cancel_orders(book_id, [o.id for o in account.orders])

    def _submit_limit(
        self, response, book_id: int, direction: int, qty: float, price: float,
        *, post_only: bool = True, ioc: bool = False,
        settlement: LoanSettlementOption = LoanSettlementOption.NONE,
    ) -> None:
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
    launch(StableMakerAgent)
