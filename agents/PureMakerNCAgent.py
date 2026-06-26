# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
PureMakerNCAgent — EXPERIMENT fork of PureMakerAgent (DON'T touch the live version). All-book two-sided
NEVER-CUT maker for Subnet 79 (τaos). Tests the leaderboard thesis: the top makers (kappa 0.1+) do NOT
cut — they hold losers for reversion and realize a stream of small positives, so Σ(realized loss³) ≈ 0.
Our live tight-cut PureMaker is NEGATIVE-kappa precisely because each 10bps cut contributes 10³=1000 to
the cubed downside (and trips the outlier penalty).

DESIGN RATIONALE (kappa = per-book Sortino-3, denominator ∝ (Σ realized loss³)^(1/3) → ONE big realized
loss craters the book + trips the outlier penalty; an OPEN underwater position contributes NOTHING):

  1) NEVER-CUT, BOUNDED. While holding, quote ONLY the reducing side, walked to BREAKEVEN (never below)
     and HELD for reversion — most maker books mean-revert, so the realized close is a win/breakeven.
     NO tight stop, NO time-cut. Losses are realized only by two BOUNDED backstops:
       (a) CATASTROPHE stop ~30-40bps (a genuine trend; rare → negligible Σloss³), and
       (b) a global BAGGED-BOOK cap: if > MAX_BAGGED_BOOKS are held underwater at once, the OLDEST are
           force-closed at their (catastrophe-bounded) loss. This defuses the never-cut failure mode —
           a pile of stuck books that lock capital AND exceed the 37.5% inactive limit (injecting hard
           0.0s into the kappa median). In a trend the agent gives up GRACEFULLY (many small bounded
           losses) instead of bagging 100+ books or taking a few catastrophic ones.

  2) IDLE FLAT BOOKS — never force-seed a flat book (a forced RT on an edgeless book is a guaranteed
     small loss → a negative-kappa scored book that drags the median). Activity comes from natural
     two-sided quoting on the reverting majority; the held-underwater minority free-drops (kept under
     the inactive limit by the bagged cap). The held-only activity backstop closes a held position ONLY
     at breakeven-or-better (IOC at entry) — it NEVER realizes a loss for activity.

  3) NO REPRICE CUSHION — keep-band is tick/2 (repeg fast). A wider keep-band holds quotes resting in
     place as price drifts, so they get run over by book-sweeps → a catastrophic-walk loss tail. tick/2
     repegs out of the way and dodges them.

QUOTING: inside-on-wide (improve = tick if spread > 2·tick) — best price → fills first → more RT
  density (needed to clear the kappa lookback gate).

BOOK SELECTION — none: quote ALL books, always. No spread/fee gate, no idle, no backoff — those are
  AdaptiveRouter's responsibility; this is the pure execution leg. Standalone (no gate) it bleeds the
  maker fee on thin books in a maker-pays regime; the gate is what makes it earn. Risk is bounded
  purely on the EXIT side (tight managed-exit / vol-stop / reentry cooldown).

MECHANICS (per book, each step):
  prune/kappa/noise → managed exit → risk guard → activity backstop (held-only) → quote → reconcile.
  Managed exit runs FIRST so a stop is never blocked by the risk guard's early return.

  * Managed exit → NEVER-CUT: IOC-close the held side ONLY on a bounded backstop — underwater >= the
            CATASTROPHE band (_stop_bps, 30-40bps) OR the book is over the bagged cap (evicted). NO
            time-cut. Slippage-capped, with 4→8→18bps escalation if an IOC keeps missing.
  * Flat  → idle if no edge (no force-seed; rests at kappa=None, free-dropped); else quote both sides
            inside-on-wide, gated only by the RT/volume budget.
  * Hold  → only the reducing side; reduce walks target→breakeven, then HELD for reversion (never below).
  * Activity → a HELD position that outlives the window is closed ONLY at NET-breakeven-or-better
            (covers both fee legs) — never realizes a loss for activity. FLAT books are never force-traded.
  * FIFO inventory mirrors the validator's _match_trade_fifo exactly (oldest-lot matching).

Provenance: EXPERIMENT never-cut fork of the live tight-cut PureMakerAgent — tests the leaderboard thesis
  that top makers HOLD losers for reversion (Σ realized loss³ ≈ 0). Bounded by the catastrophe stop
  (30-40bps) + a global bagged-book cap; NOT yet proven — run the live A/B before adopting.
"""

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
MAX_INVENTORY_LOTS = 1.5           # small cap (NC): reduce-only-when-holding keeps ≤~1 lot/book so this
                                   # rarely binds; it bounds _risk_trim's worst-case realized loss and
                                   # the per-book inventory. We never average into the bag.
MAX_INVENTORY_EQUITY_FRAC = 0.10
RISK_TRIM_SLIPPAGE_BPS = 6.0

# ---- book gate: NONE ----
# No spread/fee book selector — we make on ALL books (rationale in the module docstring). Risk is
# handled purely on the EXIT side (managed-exit / vol-stop / reentry cooldown).

# ---- profit target ----
# Profit-target base: take profit quickly. NOTE: in the maker-PAYS regime the fee floor dominates
# this base — effective TP = max(TP_BPS_BASE, TP_FEE_MULT × maker_fee_bps + a tick) ≈ 2×9bps ≈ 18bps,
# so a round-trip always covers both maker-fee legs. TP_BPS_BASE only binds if the regime flips to a
# maker-rebate (then we take the small 8bps capture rather than wait for more).
TP_BPS_BASE = 8.0
TP_FEE_MULT = 2.0                  # floor = 2× maker_fee (covers both legs + small buffer)
QUOTE_EXPIRY_S = 12.0

# ---- managed exit: NEVER-CUT, two bounded backstops only (NC experiment) ----
# kappa-3 CUBES realized downside (LPM3), and an OPEN underwater position contributes NOTHING. So the
# best move is to NOT realize losses: hold the loser, work the reduce at breakeven, wait for reversion
# (most maker books revert). NO tight stop, NO time-cut. Losses are realized only by two bounded
# backstops that defuse the never-cut failure mode (a pile of stuck books):
#   * CATASTROPHE stop band 30-40bps: only a genuine trend trips it; rare → negligible Σloss³.
#   * GLOBAL bagged-book cap (MAX_BAGGED_BOOKS): if too many books are held underwater at once, the
#     OLDEST are force-closed at their (catastrophe-bounded) loss — bounds capital AND keeps the
#     inactive-book count under the 37.5% (0.0-injection) limit. Graceful give-up in a trend.
#   * the reduce still walks only to BREAKEVEN, never the touch (see _reduce_price) — a late passive
#     fill nets ~0; non-reverting lots are simply HELD (not cut) until a backstop fires.
EXIT_WALK_START_S = 30.0           # start walking reduce from target toward breakeven after 30s
EXIT_GIVEUP_S = 150.0              # WALK-completion time only (reduce reaches breakeven by here, then
                                   #   HOLDS at breakeven). NOT a cut anymore — never-cut holds for revert.
EXIT_STOP_LOSS_BPS = 30.0          # FLOOR of the CATASTROPHE stop band (was 10 tight-cut) — only a real
EXIT_STOP_CAP_BPS = 40.0           # CAP of the catastrophe band (was 14) — hard ceiling on a single loss
EXIT_STOP_NOISE_MULT = 6.0         # stop ≈ MULT × per-book per-step mid-noise(bps), clamped to band
MAX_BAGGED_BOOKS = 24              # NC: max books held UNDERWATER at once; beyond this the OLDEST are
                                   #   force-closed (graceful give-up). NB: this bounds ONLY the held-
                                   #   underwater subset of kappa=None books — flat-quiet books with <3
                                   #   RTs are a SEPARATE None population. 24 leaves ~24-book headroom
                                   #   under the 48-book (37.5%) inactive cliff for them; monitor the
                                   #   live None count and lower this further if it approaches 48.
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
ACTIVITY_DEADLINE_S = 510.0        # NC: when a HELD position has outlived the window, try to close it
                                   # ONLY at breakeven-or-better (IOC at entry) to register the RT —
                                   # NEVER realizes a loss for activity (a never-cut book is held, not
                                   # cut, until it reverts or a backstop fires). FLAT books idle at None.
RT_MAX = 15                        # cap on round-trips per RT_WINDOW per book (anti-overtrade)

# ---- quoting: tight tick/2 repeg ----
REPRICE_KEEP_TICKS = 0.5           # keep-band = tick/2 (0.5×tick): repeg fast. A wider keep-band would
                                   # hold quotes resting in place as price drifts -> run over by
                                   # book-sweeps -> a catastrophic-walk loss tail that craters kappa.

# ---- volume cap ----
CAPITAL_TURNOVER_CAP = 10.0
VOLUME_SAFETY = 0.8
VOLUME_ASSESSMENT_NS = 86_400_000_000_000

# ---- kappa-3 (validator-faithful) ----
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


class PureMakerNCAgent(FinanceSimulationAgent):

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
        self._evict_set: dict[str, set[int]] = {}   # NC: per-validator books over the bagged cap → force-close
        self._active_validator: str | None = None

        bt.logging.info(
            f"[PureMakerNC uid={self.uid}] PURE-MAKER NEVER-CUT lot={QUOTE_LOT} exch_min={self.exch_min} "
            f"gate=NONE(all 128 books) backoff=NONE "
            f"tp_base={self.tp_bps_base:.1f}bps tp_floor={TP_FEE_MULT}×fee "
            f"exit_walk={EXIT_WALK_START_S:.0f}-{giveup_s:.1f}s(->breakeven, then HOLD) "
            f"NEVER-CUT catastrophe_band=[{EXIT_STOP_LOSS_BPS:.0f},{EXIT_STOP_CAP_BPS:.0f}]bps×{EXIT_STOP_NOISE_MULT:.0f}noise "
            f"bagged_cap={MAX_BAGGED_BOOKS} "
            f"inv_cap={MAX_INVENTORY_LOTS}lot/{MAX_INVENTORY_EQUITY_FRAC:.0%}eq "
            f"activity={activity_s:.0f}s(breakeven-only) rt_max={RT_MAX} rt_log={MAIN_VALIDATOR[:8]}"
        )

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
            f"[PureMakerNC uid={self.uid}] new simulation: {validator[:8]} sim_id={simulation_id}"
        )

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        validator = state.dendrite.hotkey
        cfg = self.simulation_config
        self._sync_precision(cfg.priceDecimals, cfg.volumeDecimals)

        vol_dp = cfg.volumeDecimals
        volume_cap = CAPITAL_TURNOVER_CAP * cfg.miner_wealth * VOLUME_SAFETY
        now = state.timestamp

        # NC bagged-book cap: if more than MAX_BAGGED_BOOKS are held UNDERWATER at once, mark the OLDEST
        # excess for force-close this step (graceful give-up — bounds capital + keeps inactive < 37.5%).
        self._evict_set[validator] = self._bagged_evictions(validator, state, now)

        for book_id in sorted(self.accounts.keys()):
            book = state.books.get(book_id)
            account = self.accounts.get(book_id) if book else None
            if book is None or account is None:
                continue
            try:
                self._step_book(response, validator, book_id, book, account, vol_dp, volume_cap, now)
            except Exception as ex:
                bt.logging.warning(f"[PureMakerNC uid={self.uid}] step {book_id}: {ex}")

        return response

    # ------------------------------------------------------------------ per-book
    def _step_book(
        self, response, validator: str, book_id: int, book, account,
        vol_dp: int, volume_cap: float, now: int,
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
        # No book gate — we make on EVERY book (uid-145's all-book posture).

        # 1) MANAGED EXIT — stop-loss / giveup takes priority over everything, including the
        #    inventory risk guard. Without this ordering, _risk_trim's early-return blocks
        #    _managed_exit on any over-cap book, leaving underwater positions with no stop-loss
        #    protection until the trim IOC finally fills (which can take hundreds of steps on
        #    illiquid books, letting losses grow to hundreds of bps).
        if self._managed_exit(response, validator, book_id, account, inv, net, best_bid, best_ask, vol_dp, now, st):
            st.last_cut_ns = now
            return

        # 2) RISK GUARD — drains breached inventory after stop-loss has had its turn.
        if self._risk_trim(response, book_id, account, net, mid, vol_dp):
            return

        # 3) ACTIVITY BACKSTOP (held-only) — if a HELD position has outlived the window, IOC-close it
        #    ONLY at NET-breakeven-or-better to register the round-trip (NC: never realizes a loss; an
        #    underwater held lot is left alone, held for reversion). FLAT books return False here (idle
        #    at kappa=None, free-dropped) — this agent never force-seeds. 510s < the validator's 600s grace.
        activity_ref = st.last_rt_ns if st.last_rt_ns > 0 else st.seen_ns
        if (now - activity_ref) >= self.activity_deadline_ns:
            if self._activity_close(response, book_id, account,
                                    inv, net, best_bid, best_ask, vol_dp):
                return

        # 4) DESIRED QUOTES — reduce-only when holding; two-sided entry when the budget gates in
        #    _entry_ok clear (RT count + volume cap only — no book gate, no streak/backoff pause).
        desired = self._desired_quotes(
            validator, book_id, account, inv, net,
            best_bid, best_ask, mid, maker_fee, volume_cap, now,
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
            f"[PureMakerNC uid={self.uid}] RISK-TRIM book={book_id} net={net:+.4f} trim={trim}"
        )
        return True

    # ------------------------------------------------------------------ vol-scaled stop band
    def _stop_bps(self, st: _BookState) -> float:
        """Vol-scaled CATASTROPHE stop within [FLOOR, CAP] = [30, 40]bps (NC). NEVER-CUT holds through
        noise and normal drawdowns; only a genuine adverse TREND past this band realizes a loss. FLOOR
        is the calm-book default; CAP is the hard ceiling on a single realized loss. Also serves as the
        resting-reduce price floor in _reduce_price (a passive reduce never rests worse than this)."""
        scaled = EXIT_STOP_NOISE_MULT * st.noise_bps
        return min(EXIT_STOP_CAP_BPS, max(EXIT_STOP_LOSS_BPS, scaled))

    # ------------------------------------------------------------------ managed exit
    def _managed_exit(
        self, response, validator: str, book_id: int, account, inv: _Inv, net: float,
        best_bid: float, best_ask: float, vol_dp: int, now: int, st: _BookState,
    ) -> bool:
        """NC NEVER-CUT: realize a loss ONLY on a bounded backstop — a CATASTROPHE move (uw past the
        30-40bps band) or eviction (book over the global bagged cap). NO time-cut: a non-reverting lot
        inside the band is HELD (returns False here) for reversion, working a breakeven reduce quote in
        _desired_quotes. This is the whole experiment — keep Σ(realized loss³) ≈ 0."""
        if abs(net) < self.exch_min:
            # Flat — clear escalation state so the NEXT position starts a fresh miss streak.
            st.exit_miss_count = 0
            st.exit_prev_net = 0.0
            return False
        stop_bps = self._stop_bps(st)            # vol-scaled: hold through noise, cut on trend
        # Escalate the concession on consecutive IOC-cut misses: a fixed-price IOC that doesn't cross
        # on a fast/wide book re-fires every step while the position bleeds. Escalating 4→8→18bps caps
        # the loss window; the final stage is a wide LIMIT (not a market order) to bound gap fills.
        if st.exit_prev_net > 0:
            if abs(net) >= st.exit_prev_net - self._flat_eps:
                st.exit_miss_count += 1          # |net| didn't shrink => the last IOC missed
            else:
                st.exit_miss_count = 0           # partial/full fill => streak broken
                st.exit_prev_net = 0.0
        if st.exit_miss_count >= 4:
            slip = EXIT_CUT_CROSS_BPS / 1e4
        elif st.exit_miss_count >= 2:
            slip = EXIT_CUT_ESCALATE_BPS / 1e4
        else:
            slip = EXIT_CUT_SLIPPAGE_BPS / 1e4
        if net > 0:
            _, _, px0, _ = inv.longs[0]
            uw = (px0 - best_bid) / px0 * 1e4 if px0 > 0 else 0.0
            evict = book_id in self._evict_set.get(validator, ())   # NC: over the bagged cap → graceful force-close
            stopped = uw >= stop_bps             # NC: stop_bps is now CATASTROPHE-only (~30-40bps)
            if not (evict or stopped):
                # Position recovered without a fill — break the miss streak so a later stop event
                # starts fresh escalation instead of inheriting a stale (premature) count.
                st.exit_miss_count = 0
                st.exit_prev_net = 0.0
                return False
            q = round(min(self._long_qty(inv), self._avail(account.base_balance)), vol_dp)
            if q < self.exch_min:
                # Non-submit step — break the miss streak. A step that sends no order must not be
                # counted as an IOC miss, or repeated sub-lot/low-balance steps inflate the count
                # and the next real cut crosses wider than warranted (and slip can't fix a qty gap).
                st.exit_miss_count = 0
                st.exit_prev_net = 0.0
                return False
            st.exit_prev_net = abs(net)
            self._cancel_all(response, account, book_id)
            px = round(best_bid * (1.0 - slip), self._price_decimals)
            self._submit_limit(response, book_id, OrderDirection.SELL, q, px, ioc=True, post_only=False)
        else:
            _, _, px0, _ = inv.shorts[0]
            uw = (best_ask - px0) / px0 * 1e4 if px0 > 0 else 0.0
            evict = book_id in self._evict_set.get(validator, ())   # NC: over the bagged cap → graceful force-close
            stopped = uw >= stop_bps             # NC: stop_bps is now CATASTROPHE-only (~30-40bps)
            if not (evict or stopped):
                st.exit_miss_count = 0
                st.exit_prev_net = 0.0
                return False
            buy_px = best_ask * (1.0 + slip)
            q_max = self._avail(account.quote_balance) / buy_px if buy_px > 0 else self._short_qty(inv)
            q = round(min(self._short_qty(inv), q_max), vol_dp)
            if q < self.exch_min:
                # Non-submit step — break the miss streak (see long branch).
                st.exit_miss_count = 0
                st.exit_prev_net = 0.0
                return False
            st.exit_prev_net = abs(net)
            self._cancel_all(response, account, book_id)
            px = round(best_ask * (1.0 + slip), self._price_decimals)
            self._submit_limit(response, book_id, OrderDirection.BUY, q, px, ioc=True, post_only=False,
                               settlement=self._loan_settlement(account))
        reason = "evict+catastrophe" if (evict and stopped) else ("evict" if evict else "catastrophe")
        if st.exit_miss_count >= 2:
            stage = "IOC-CROSS" if st.exit_miss_count >= 4 else "IOC-ESCALATE"
            bt.logging.info(
                f"[PureMakerNC uid={self.uid}] {stage} book={book_id} "
                f"miss={st.exit_miss_count} slip={slip*1e4:.0f}bps"
            )
        bt.logging.info(
            f"[PureMakerNC uid={self.uid}] MANAGED-EXIT book={book_id} reason={reason} "
            f"net={net:+.4f} q={q} @~{px} uw={uw:.1f}bps stop={stop_bps:.0f}bps "
            f"noise={st.noise_bps:.1f}bps"
        )
        return True

    # ------------------------------------------------------------------ activity backstop
    def _activity_close(
        self, response, book_id: int, account, inv: _Inv, net: float,
        best_bid: float, best_ask: float, vol_dp: int,
    ) -> bool:
        """NC: register an RT on an aged HELD position ONLY if it can close at NET-BREAKEVEN-or-better.
        The close limit is the entry px shifted by a 2×maker-fee buffer so the realized RT covers BOTH
        fee legs and never nets a loss for activity — an underwater (or only-gross-breakeven) held book
        is left alone (held for reversion). FLAT books are never force-seeded (idle at kappa=None)."""
        long_q = self._long_qty(inv)
        short_q = self._short_qty(inv)
        if long_q < self.exch_min and short_q < self.exch_min:
            return False
        lot = max(self.quote_lot, self.exch_min)
        fee = self._maker_fee_rate(account) or 0.0
        buf = max(2.0 * fee, 0.0)                    # cover both fee legs -> truly NET-breakeven
        if long_q >= self.exch_min:
            px0 = inv.longs[0][2]
            be = px0 * (1.0 + buf)                   # net-breakeven sell price
            if px0 <= 0 or best_bid < be:            # can't exit >= net-breakeven -> never realize a loss
                return False
            q = round(min(long_q, self._avail(account.base_balance), lot), vol_dp)
            if q < self.exch_min:
                return False
            px = round(be, self._price_decimals)     # IOC fills only at >= net-breakeven
            self._cancel_all(response, account, book_id)
            self._submit_limit(response, book_id, OrderDirection.SELL, q, px, ioc=True, post_only=False)
        else:
            px0 = inv.shorts[0][2]
            be = px0 * (1.0 - buf)                   # net-breakeven buy price
            if px0 <= 0 or best_ask > be:            # can't cover <= net-breakeven -> hold
                return False
            buy_px = be
            q_max = self._avail(account.quote_balance) / buy_px if buy_px > 0 else short_q
            q = round(min(short_q, lot, q_max), vol_dp)
            if q < self.exch_min:
                return False
            px = round(be, self._price_decimals)     # IOC fills only at <= net-breakeven
            self._cancel_all(response, account, book_id)
            self._submit_limit(response, book_id, OrderDirection.BUY, q, px, ioc=True, post_only=False,
                               settlement=self._loan_settlement(account))
        bt.logging.info(f"[PureMakerNC uid={self.uid}] ACTIVITY-CLOSE(breakeven) book={book_id} net={net:+.4f}")
        return True

    # ------------------------------------------------------------------ NC bagged-book cap
    def _bagged_evictions(self, validator: str, state, now: int) -> set[int]:
        """Books to force-close this step because the count of held-UNDERWATER books exceeds
        MAX_BAGGED_BOOKS — the OLDEST excess are evicted (graceful give-up). Bounds capital and keeps
        the inactive-book count under the validator's 37.5% (0.0-injection) limit. The realized loss on
        an evicted book is bounded by the catastrophe stop, so Σloss³ stays controlled even in a trend."""
        inv_map = self.inv.get(validator)
        if not inv_map:
            return set()
        cand: list[tuple[int, int]] = []   # (age_ns, book_id) for held-underwater books
        for bid, inv in inv_map.items():
            net = self._net_qty(inv)
            if abs(net) < self.exch_min:
                continue
            book = state.books.get(bid)
            if book is None or not book.bids or not book.asks:
                continue
            best_bid, best_ask = book.bids[0].price, book.asks[0].price
            if net > 0:
                ts0, _, px0, _ = inv.longs[0]
                uw = (px0 - best_bid) / px0 * 1e4 if px0 > 0 else 0.0
            else:
                ts0, _, px0, _ = inv.shorts[0]
                uw = (best_ask - px0) / px0 * 1e4 if px0 > 0 else 0.0
            if uw > 0.0:                    # underwater only (a winner-held book isn't "bagged")
                cand.append((now - ts0, bid))
        if len(cand) <= MAX_BAGGED_BOOKS:
            return set()
        cand.sort(reverse=True)            # oldest (largest age) first
        return {bid for _, bid in cand[:len(cand) - MAX_BAGGED_BOOKS]}

    # ------------------------------------------------------------------ quoting
    def _desired_quotes(
        self, validator: str, book_id: int, account, inv: _Inv, net: float,
        best_bid: float, best_ask: float, mid: float, maker_fee: float | None,
        volume_cap: float, now: int,
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
        elif self._entry_ok(validator, book_id, st, now, volume_cap):
            # Flat and the budget gates clear → quote both sides.
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
        at breakeven are HELD for reversion and exit via _managed_exit only on catastrophe or eviction
        (NC: no time-cut). The catastrophe floor remains so a resting reduce never rests below the band."""
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
        """Gate new inventory: RT budget + volume cap only. No book gate, no streak/backoff pause."""
        vol_ok = self._rolled_quote_volume(validator, book_id, now) < volume_cap
        rt_ok = self._rt_count(st, now) < RT_MAX
        return vol_ok and rt_ok

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
            f"[PureMakerNC uid={self.uid}] priceDecimals={price_decimals} tick={self._tick} "
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
            f"[PureMakerNC uid={self.uid} RT] book={book_id} close={side} "
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
    launch(PureMakerNCAgent)
