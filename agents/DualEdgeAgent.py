# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
DualEdgeAgent — dual-mode (maker / taker) FIFO-aware, activity-guaranteed engine for subnet 79.

This is KappaMaker's proven maker engine (verbatim) PLUS a TakerScalper-style taker mode,
routed per book by the live fee regime:

  * MAKER mode (default; most books): KappaMaker's two-sided spread-capture with FIFO no-loss
    reduce-only exits, capped-slippage managed cut, and an activity floor. Used wherever taking
    is not paid for.
  * TAKER mode (deep-rebate books): when the taker fee is a rebate large enough that crossing
    the spread is +EV (rebate >= half-spread + margin), run TakerScalper's discipline instead —
    open ONE small clip in the microprice-bias direction, exit on TP / SL / max-hold within
    seconds. Tiny, short exposure bounds the worst case (the proven rank-leading behavior).

Both modes share ONE FIFO inventory (validator-faithful _match_fifo), so round-trips, kappa, and
activity accounting are identical regardless of which mode opened the position. A mode switch is
committed only when the book is flat, so a held position is always closed by the engine that
opened it (the taker never inherits a maker bag, and vice versa).

Why this design (reverse-engineered from taos/im/validator/reward.py, utils/kappa.py and
the validator's _match_trade_fifo):
  * The live score collapses to  trading_score ≈ 0.79 · kappa_score, where
    kappa_score = median over books of normalized per-book Kappa-3.
  * Kappa-3 = (mean−τ)/cbrt(LPM3) on MAD-normalized REALIZED round-trip P&L, τ=0, over a
    3h window, ≥3 round-trips/book. Downside is CUBED, so a single losing round-trip hurts
    ~cubically while a winner helps ~linearly. Only REALIZED P&L counts.
  * CRITICAL — realized P&L and round-trip volume are computed by the validator via FIFO
    against the OLDEST open lot (validator._match_trade_fifo), NOT against a VWAP average.
    => to guarantee a positive realized round-trip, an exit must clear the OLDEST lot's
       price (+fees), so we track FIFO lots locally and price exits off the oldest lot.
  * ACTIVITY — the activity factor decays for any book that does not generate round-trip
    volume each sampling window, and round-trip volume is produced ONLY by a CLOSING trade
    (a fill that matches against opposing inventory). => every book MUST close something
    every window. Activity 1.0 is a hard requirement and is never sacrificed here.

Strategy (mirrors the top miners: tiny tight inventory, fast recycling, cut losers small):
  Capture spread as a MAKER and only REALIZE round-trips that are FIFO-positive when the market
  cooperates — but the moment a position goes stale or underwater, cut it FAST and SMALL. The
  cubic downside penalty makes one big forced loss far worse than many tiny ones, so we never
  bag a loser waiting for a bounce.
    * Flat        -> quote BOTH sides (postOnly) inside the touch, fee-aware.
    * Holding     -> work ONLY the reducing side (no averaging into the bag). Size the reduce off
                     the TOTAL side (partial-fill dust lots bundle in via FIFO) and price off the
                     worst lot so every consumed lot is FIFO-positive; the price WALKS from the
                     profit target down to the touch as the lot ages, so it keeps filling instead
                     of resting at an unreachable entry price.
    * Managed exit-> IOC-cut the WHOLE reducible side once its oldest lot is too old (~20s) OR
                     underwater beyond a hard stop (~25bps). Capped slippage -> each loss bounded.
    * Dust        -> a sub-min residual (|net| < exch_min) from partial fills cannot be closed by
                     one order; it is treated as flat so two-sided quoting recycles it fast.
    * Activity    -> ultimate floor at ~480s: if a book still has not closed, force a one-lot
                     close (or seed a flat/idle book) so round-trip volume keeps activity at 1.0.
    * Risk guard  -> if inventory breaches lot/equity caps, trim the excess with a
                     capped-slippage marketable (IOC) order.

Per book each step:
  prune/refresh kappa -> risk-guard trim -> managed exit (cut) -> activity backstop close
            -> desired quotes (oldest-lot-aware) -> reconcile resting orders.
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

# Exchange floor for any BASE order (sim `minOrderSize`, not exposed to agents).
EXCHANGE_MIN_ORDER_SIZE = 0.25
# Per-side quote lot. Small so fills accumulate smoothly (we embrace partial fills).
QUOTE_LOT = 0.25

# ---- inventory bounds (allow passive reduce to work; trim only real breaches) ----
MAX_INVENTORY_LOTS = 3.0           # hard per-book lot cap; excess is risk-trimmed
MAX_INVENTORY_EQUITY_FRAC = 0.15   # hard per-book notional cap as a fraction of book equity
RISK_TRIM_SLIPPAGE_BPS = 6.0       # max adverse slippage allowed on a forced (loss) trim

# ---- exit / round-trip economics ----
TP_BPS = 13.0                      # initial profit target over the OLDEST lot, in bps
TP_FEE_MULT = 2.2                  # require target >= TP_FEE_MULT × maker_fee (both legs + buffer)
MAX_ENTRY_MAKER_FEE = 0.0015       # skip opening a side while the maker fee exceeds this
QUOTE_EXPIRY_S = 12.0              # GTT expiry; reconcile refreshes each step; match fast recycle

# ---- managed exit: recycle FAST and cut losers SMALL (top miners flip in seconds, never bag) ----
# Kappa-3 downside is CUBED: one big forced loss dwarfs many tiny ones. So instead of resting a
# reduce at the entry price (which never fills when underwater) and dumping it at a far-away cliff,
# we (a) walk the passive reduce price from the profit target down to the touch as the lot ages,
# and (b) IOC-cut the oldest lot once it is too old OR underwater beyond a hard stop — bounding the
# loss on each cut and keeping the realized-P&L distribution tight.
EXIT_WALK_START_S = 10.0           # below this oldest-lot age, rest the reduce at the full profit target
EXIT_GIVEUP_S = 20.0               # by this age the reduce has walked to the touch; then IOC-cut the lot
EXIT_STOP_LOSS_BPS = 16.0          # IOC-cut immediately if the oldest lot is underwater beyond this
EXIT_CUT_SLIPPAGE_BPS = 6.0        # capped adverse slippage on a forced IOC cut
REENTRY_COOLDOWN_S = 75.0          # after a forced cut, pause fresh entries so we don't re-bag a trend

# ---- activity: guarantee a round-trip close on every book every window ----
# Validator samples round-trip volume every 600s; we keep this internal RT-budget/display
# window just under that so per-window RT counts line up with the scorer.
RT_WINDOW_S = 570.0
ACTIVITY_DEADLINE_S = 480.0        # ultimate floor: force a close this long since the last close
RT_MAX = 10                        # max RTs per book per window (≥1 RT/10m via activity backstop)
FORCE_TRIM_SLIPPAGE_BPS = 6.0      # slippage cap for an activity cross-close

# ---- volume cap (avoid the capital_turnover_cap ceiling) ----
CAPITAL_TURNOVER_CAP = 10.0
VOLUME_SAFETY = 0.8
VOLUME_ASSESSMENT_NS = 86_400_000_000_000

# ---- Kappa-3 (validator-faithful; 3h history for score visibility) ----
KAPPA_TAU = 0.0
KAPPA_MIN_OBS = 3
KAPPA_MIN_LOOKBACK_S = 5400.0      # 90 min
KAPPA_RT_HISTORY_S = 10_800.0      # 3h

# ---- TAKER mode (TakerScalper port: deep-rebate books only) ----
# Route a book to TAKER when crossing is paid for: rebate >= half-spread + margin AND
# rebate >= the gate (>= SL/2 so a stop/time exit is cushioned toward break-even).
ROUTER_TAKER_MARGIN_BPS = 1.0
TAKER_REBATE_GATE_BPS = 2.5
TAKER_EDGE_MARGIN_BPS = 1.0        # min +EV round-trip estimate (bps of notional)
TAKER_TP_BPS = 2.5                 # take-profit (gross, vs side avg)
TAKER_SL_BPS = 4.0                 # stop-loss (gross) — rebate-cushioned by the gate
TAKER_MAX_HOLD_S = 4.0             # hard hold cap (bounds adverse exposure)
TAKER_MIN_HOLD_S = 1.5             # min dwell before any exit
TAKER_REOPEN_GAP_S = 4.0           # throttle between a close and the next open

MODE_MAKER = "maker"
MODE_TAKER = "taker"

# RT logs only for the scoring validator.
MAIN_VALIDATOR = "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF"


@dataclass
class _Inv:
    """FIFO inventory mirroring the validator's open_positions: oldest-first deques of
    lots (ts, qty, price, fee). Net position = sum(longs) − sum(shorts)."""
    longs: deque = field(default_factory=deque)
    shorts: deque = field(default_factory=deque)


@dataclass
class _BookState:
    last_rt_ns: int = 0                 # last close that generated round-trip volume
    last_cut_ns: int = 0                # last forced (managed-exit) cut; gates re-entry cooldown
    seen_ns: int = 0                    # first-seen ts; activity clock before the first RT
    rt_events: list[tuple[int, float]] = field(default_factory=list)  # (ts, realized_pnl)
    kappa3: float | None = None
    vol_log: list[tuple[int, float]] = field(default_factory=list)    # (ts, traded quote vol)
    # dual-mode routing / taker
    mode: str = MODE_MAKER
    taker_open_ns: int = 0              # when the current taker clip was opened
    last_close_ns: int = 0             # last taker close (reopen throttle)


class DualEdgeAgent(FinanceSimulationAgent):
    # ------------------------------------------------------------------ setup
    def initialize(self) -> None:
        bt.logging.set_info()

        self.quote_lot = QUOTE_LOT
        self.exch_min = EXCHANGE_MIN_ORDER_SIZE
        self._flat_eps = 0.5 * 10 ** (-4)      # overwritten by _sync_precision on first respond
        self._price_decimals: int | None = None
        self._volume_decimals: int | None = None
        self._tick = 0.01
        self.volume_assessment_ns = VOLUME_ASSESSMENT_NS

        # Per-UID jitter so a fleet does not act in lockstep.
        jitter = ((self.uid * 2654435761) % 1000) / 1000.0
        self.tp_bps = TP_BPS * (0.9 + 0.2 * jitter)
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
        self._step_ts_ns: int = 0
        self._active_validator: str | None = None

        bt.logging.info(
            f"[DualEdge uid={self.uid}] DUAL-MODE FIFO lot={QUOTE_LOT} exch_min={self.exch_min} "
            f"| MAKER tp={self.tp_bps:.1f}bps inv_cap={MAX_INVENTORY_LOTS}lot/{MAX_INVENTORY_EQUITY_FRAC:.0%}eq "
            f"exit_walk={EXIT_WALK_START_S:.0f}-{giveup_s:.0f}s stop={EXIT_STOP_LOSS_BPS:.0f}bps "
            f"max_entry_maker_fee={MAX_ENTRY_MAKER_FEE} "
            f"| TAKER gate={TAKER_REBATE_GATE_BPS:.1f}bps tp={TAKER_TP_BPS:.1f}/sl={TAKER_SL_BPS:.1f}bps "
            f"hold<={TAKER_MAX_HOLD_S:.0f}s reopen={TAKER_REOPEN_GAP_S:.0f}s "
            f"| activity_deadline={activity_s:.0f}s rt_window={RT_WINDOW_S / 60:.0f}m rt_log={MAIN_VALIDATOR[:8]}"
        )

    # --------------------------------------------------------------- lifecycle
    def update(self, state: MarketSimulationStateUpdate) -> None:
        self._step_ts_ns = int(state.timestamp)
        self._active_validator = state.dendrite.hotkey
        # Reset before super().update() so the new sim's first fills don't hit stale state.
        self._ensure_simulation(self._active_validator, state.config.simulation_id)
        super().update(state)

    def _ensure_simulation(self, validator: str, simulation_id: str | None) -> None:
        """Drop per-validator state when the validator starts a new simulation."""
        if self._sim_id.get(validator) == simulation_id:
            return
        self.inv.pop(validator, None)
        self.books_state.pop(validator, None)
        if simulation_id is not None:
            self._sim_id[validator] = simulation_id
        else:
            self._sim_id.pop(validator, None)
        bt.logging.info(
            f"[DualEdge uid={self.uid}] new simulation: {validator[:8]} sim_id={simulation_id}"
        )

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        validator = state.dendrite.hotkey
        cfg = self.simulation_config
        self._sync_precision(cfg.priceDecimals, cfg.volumeDecimals)

        vol_dp = cfg.volumeDecimals
        volume_cap = CAPITAL_TURNOVER_CAP * cfg.miner_wealth * VOLUME_SAFETY
        now = state.timestamp

        for book_id in sorted(self.accounts.keys()):
            book = state.books.get(book_id)
            account = self.accounts.get(book_id) if book else None
            if book is None or account is None:
                continue
            try:
                self._step_book(response, validator, book_id, book, account, vol_dp, volume_cap, now)
            except Exception as ex:
                bt.logging.warning(f"[DualEdge uid={self.uid}] step {book_id}: {ex}")

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
        if self._prune_rt_events(st, now):
            self._refresh_book_kappa(validator, book_id, now)

        net = self._net_qty(inv)

        # 1) RISK GUARD — trim hard breaches of the inventory caps (may realize a loss).
        if self._risk_trim(response, validator, book_id, account, inv, net, mid, vol_dp):
            return

        # 1b) ROUTE — pick the mode for this book from the live fee regime. Commit a switch only
        # when flat & with no resting orders, so a held position is always closed by the engine
        # that opened it (taker never inherits a maker bag and vice versa).
        flat = abs(net) < self.exch_min
        want_mode = self._desired_mode(account, best_bid, best_ask, mid)
        if want_mode != st.mode and flat:
            if account.orders:
                self._cancel_all(response, account, book_id)
                return
            bt.logging.info(
                f"[DualEdge uid={self.uid}] MODE {st.mode}->{want_mode} book={book_id} "
                f"taker_fee={self._taker_fee_rate(account)} spread_bps="
                f"{(best_ask - best_bid) / mid * 1e4:.1f}"
            )
            st.mode = want_mode

        if st.mode == MODE_TAKER:
            self._taker_step(response, validator, book_id, book, account, inv, net,
                             best_bid, best_ask, mid, volume_cap, vol_dp, now)
            return

        # 2) MANAGED EXIT — IOC-cut the oldest lot if it is too old or underwater past the stop.
        if self._managed_exit(response, book_id, account, inv, net, best_bid, best_ask, vol_dp, now):
            st.last_cut_ns = now
            return

        # 3) ACTIVITY BACKSTOP — ultimate floor: guarantee a round-trip close each window.
        if self._activity_elapsed(st, now) >= self.activity_deadline_ns:
            if self._activity_close(response, validator, book_id, book, account, inv, net,
                                    best_bid, best_ask, vol_dp):
                return

        # 4) Desired quotes (oldest-lot-aware), then reconcile resting orders.
        desired = self._desired_quotes(
            validator, book_id, book, account, inv, net, best_bid, best_ask, mid, volume_cap, now,
        )
        self._reconcile_quotes(response, account, book_id, desired)

    # ------------------------------------------------------------------ risk guard
    def _risk_trim(
        self, response, validator: str, book_id: int, account, inv: _Inv, net: float,
        mid: float, vol_dp: int,
    ) -> bool:
        """Trim inventory that breaches the lot/equity caps with a capped-slippage IOC order.
        This (and the activity backstop) are the only paths that may realize a loss."""
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
        self._cancel_all(response, account, book_id)
        if net > 0:
            limit_px = round(mid * (1.0 - slip), self._price_decimals)
            # _cancel_all (above) frees base reserved in our resting orders this step.
            trim = round(min(trim, self._avail(account.base_balance)), vol_dp)
            if trim < self.exch_min:
                return False
            self._submit_limit(response, book_id, OrderDirection.SELL, trim, limit_px,
                               ioc=True, post_only=False)
        else:
            limit_px = round(mid * (1.0 + slip), self._price_decimals)
            self._submit_limit(response, book_id, OrderDirection.BUY, trim, limit_px,
                               ioc=True, post_only=False, settlement=self._loan_settlement(account))
        bt.logging.info(
            f"[DualEdge uid={self.uid}] RISK-TRIM book={book_id} net={net:+.4f} "
            f"trim={trim} @~{limit_px} (lot_cap={lot_cap} notional_cap={notional_cap:.0f})"
        )
        return True

    # ------------------------------------------------------------------ managed exit
    def _managed_exit(
        self, response, book_id: int, account, inv: _Inv, net: float,
        best_bid: float, best_ask: float, vol_dp: int, now: int,
    ) -> bool:
        """Force a marketable (IOC) cut of the WHOLE reducible side when its oldest lot is either
        too old (the passive reduce had its full walk-to-touch window and still did not fill) or
        underwater beyond the hard stop. Cutting fast mirrors the top miners (who recycle in
        seconds and never bag a loser) and bounds each realized loss, so the cubic Kappa downside
        stays tiny.

        Sizes off the TOTAL side quantity (capped by spendable balance), NOT the oldest lot, so
        partial-fill dust lots below exch_min are bundled in via FIFO instead of stranding the
        position until the activity cliff. A genuinely sub-min residual (|net| < exch_min) cannot
        be closed by a single order and is left for two-sided quoting to recycle."""
        if abs(net) < self.exch_min:
            return False
        slip = EXIT_CUT_SLIPPAGE_BPS / 1e4
        if net > 0:
            ts, _, px0, _ = inv.longs[0]
            underwater_bps = (px0 - best_bid) / px0 * 1e4 if px0 > 0 else 0.0
            aged = now - ts >= self.exit_giveup_ns
            stopped = underwater_bps >= EXIT_STOP_LOSS_BPS
            if not (aged or stopped):
                return False
            q = round(min(self._long_qty(inv), self._avail(account.base_balance)), vol_dp)
            if q < self.exch_min:
                return False
            self._cancel_all(response, account, book_id)
            px = round(best_bid * (1.0 - slip), self._price_decimals)
            self._submit_limit(response, book_id, OrderDirection.SELL, q, px, ioc=True, post_only=False)
        else:
            ts, _, px0, _ = inv.shorts[0]
            underwater_bps = (best_ask - px0) / px0 * 1e4 if px0 > 0 else 0.0
            aged = now - ts >= self.exit_giveup_ns
            stopped = underwater_bps >= EXIT_STOP_LOSS_BPS
            if not (aged or stopped):
                return False
            q = round(self._short_qty(inv), vol_dp)
            if q < self.exch_min:
                return False
            self._cancel_all(response, account, book_id)
            px = round(best_ask * (1.0 + slip), self._price_decimals)
            self._submit_limit(response, book_id, OrderDirection.BUY, q, px, ioc=True, post_only=False,
                               settlement=self._loan_settlement(account))
        reason = "stop" if (stopped and not aged) else ("age" if aged and not stopped else "age+stop")
        bt.logging.info(
            f"[DualEdge uid={self.uid}] MANAGED-EXIT book={book_id} reason={reason} "
            f"net={net:+.4f} q={q} @~{px} underwater={underwater_bps:.0f}bps"
        )
        return True

    # ------------------------------------------------------------------ activity backstop
    def _activity_close(
        self, response, validator: str, book_id: int, book, account, inv: _Inv, net: float,
        best_bid: float, best_ask: float, vol_dp: int,
    ) -> bool:
        """Force a round-trip close so the book generates round-trip volume within the 600s
        activity sampling window and keeps its activity factor at 1.0. Even a partial fill
        counts as round-trip volume, so a marketable (IOC) cross at the touch is enough.

        Closes ONE lot at a time (sized off the total position, so a sub-min oldest lot never
        blocks the close); balances are taken as free+reserved because _cancel_all (issued
        first) frees our resting orders on the exchange this step. If genuinely flat, seed a
        small long so a close follows on the next step (last_rt_ns is unchanged by the seed,
        so the backstop re-fires and closes it within seconds — well inside the 600s window).
        With managed exits cutting by ~20s, holding books rarely reach this 480s floor;
        when they do, a tiny capped loss is acceptable versus losing activity."""
        slip = FORCE_TRIM_SLIPPAGE_BPS / 1e4
        self._cancel_all(response, account, book_id)
        long_q = self._long_qty(inv)
        short_q = self._short_qty(inv)
        lot = max(self.quote_lot, self.exch_min)
        base_avail = self._avail(account.base_balance)
        quote_avail = self._avail(account.quote_balance)

        if long_q >= self.exch_min:
            q = round(min(long_q, base_avail, lot), vol_dp)
            if q < self.exch_min:
                return False
            px = round(best_bid * (1.0 - slip), self._price_decimals)
            self._submit_limit(response, book_id, OrderDirection.SELL, q, px, ioc=True, post_only=False)
        elif short_q >= self.exch_min:
            q = round(min(short_q, lot), vol_dp)
            if q < self.exch_min:
                return False
            px = round(best_ask * (1.0 + slip), self._price_decimals)
            self._submit_limit(response, book_id, OrderDirection.BUY, q, px, ioc=True, post_only=False,
                               settlement=self._loan_settlement(account))
        else:
            # Flat & idle: seed a long so the next step can close it for round-trip volume.
            q = lot
            if quote_avail < q * best_ask * (1.0 + slip):
                return False
            px = round(best_ask * (1.0 + slip), self._price_decimals)
            self._submit_limit(response, book_id, OrderDirection.BUY, q, px, ioc=True, post_only=False)
        bt.logging.info(f"[DualEdge uid={self.uid}] ACTIVITY-CLOSE book={book_id} net={net:+.4f}")
        return True

    # ------------------------------------------------------------------ quoting
    def _desired_quotes(
        self, validator: str, book_id: int, book, account, inv: _Inv, net: float,
        best_bid: float, best_ask: float, mid: float, volume_cap: float, now: int,
    ) -> dict[int, tuple[float, float]]:
        """Return {direction: (price, qty)} we want resting.

        Holding inventory -> work ONLY the reducing side (no averaging into the bag); the
        reduce is priced off the OLDEST lot and WALKS from the profit target toward the touch
        with lot age (see _reduce_price), so it keeps filling rather than resting forever at an
        unreachable entry. Flat -> quote BOTH sides inside the touch for spread capture
        (fee / turnover / RT-budget gated)."""
        st = self._bstate(validator, book_id)
        maker_fee = self._maker_fee_rate(account)
        tick = self._tick
        pdp, vdp = self._price_decimals, self._volume_decimals
        desired: dict[int, tuple[float, float]] = {}

        # Base profit target over the oldest lot: at least tp_bps, floored above fees+tick.
        fee_bps = (maker_fee * 1e4) if maker_fee is not None else 0.0
        floor_bps = TP_FEE_MULT * fee_bps + (tick / mid) * 1e4
        base_target = max(self.tp_bps, floor_bps) / 1e4

        # free = spendable now (for opening); avail = free+reserved (for maintaining a reduce
        # order, whose own base/quote is reserved in it — sizing off free would churn it).
        free_base = account.base_balance.free if account.base_balance else 0.0
        free_quote = account.quote_balance.free if account.quote_balance else 0.0
        base_avail = self._avail(account.base_balance)
        quote_avail = self._avail(account.quote_balance)

        spread = best_ask - best_bid
        improve = tick if spread > 2 * tick else 0.0
        bid_inside = round(best_bid + improve, pdp)
        ask_inside = round(best_ask - improve, pdp)
        if bid_inside >= ask_inside:                # locked/degenerate -> join the touch
            bid_inside, ask_inside = round(best_bid, pdp), round(best_ask, pdp)

        if net >= self.exch_min:
            # Holding long -> passive SELL reduce of the WHOLE long. Size off TOTAL long (so
            # partial-fill dust lots bundle in via FIFO, never stranding the position) and price
            # off the WORST (highest) lot so every lot the fill consumes is FIFO-positive; the
            # price walks from the target down to the touch with the oldest lot's age so it keeps
            # filling rather than resting at an unreachable entry (the old bag trap).
            age = now - inv.longs[0][0]
            worst_px = max(p for _, _, p, _ in inv.longs)
            px = self._reduce_price(True, worst_px, age, ask_inside, base_target, pdp)
            q = round(min(self._long_qty(inv), base_avail), vdp)
            if q >= self.exch_min and px > 0:
                desired[OrderDirection.SELL] = (px, q)
        elif net <= -self.exch_min:
            # Holding short -> passive BUY reduce of the WHOLE short; price off the worst (lowest)
            # short entry; walk up to the touch with age.
            age = now - inv.shorts[0][0]
            worst_px = min(p for _, _, p, _ in inv.shorts)
            px = self._reduce_price(False, worst_px, age, bid_inside, base_target, pdp)
            q = round(self._short_qty(inv), vdp)
            if q >= self.exch_min and px > 0 and quote_avail >= q * px:
                desired[OrderDirection.BUY] = (px, q)
        elif now - st.last_cut_ns < self.reentry_cooldown_ns:
            # Just cut a loser on this book -> pause fresh entries so we don't re-bag a trend.
            pass
        elif self._entry_ok(maker_fee, validator, book_id, st, now, volume_cap):
            # Flat OR sub-min dust (|net| < exch_min, uncloseable by a single order) -> quote both
            # sides inside the touch. A fill absorbs the dust into a closeable position or closes
            # it (generating round-trip volume), recycling it fast instead of stranding it.
            q = round(self.quote_lot, vdp)
            if q >= self.exch_min and free_base >= q:
                desired[OrderDirection.SELL] = (ask_inside, q)
            if q >= self.exch_min and free_quote >= q * bid_inside:
                desired[OrderDirection.BUY] = (bid_inside, q)
        return desired

    def _reduce_price(
        self, is_long: bool, px0: float, age_ns: int, touch_inside: float,
        base_target: float, pdp: int,
    ) -> float:
        """Passive-reduce limit price for the oldest lot. Starts at the profit target (above
        the oldest long / below the oldest short) and, as the lot ages, walks linearly toward
        the touch so it keeps filling instead of resting forever at an unreachable entry price.
        Once the lot is too old or too far underwater, _managed_exit IOC-cuts it outright."""
        w = self._exit_walk(age_ns)
        if is_long:
            ideal = max(touch_inside, px0 * (1.0 + base_target))   # seek profit above market
            return round(ideal + (touch_inside - ideal) * w, pdp)  # ... walk down to the ask
        ideal = min(touch_inside, px0 * (1.0 - base_target))       # seek profit below market
        return round(ideal + (touch_inside - ideal) * w, pdp)      # ... walk up to the bid

    def _exit_walk(self, age_ns: int) -> float:
        """0 while the lot is young (rest at the full profit target), ramping linearly to 1 by
        the give-up age (rest at the touch). Drives _reduce_price's walk toward the market."""
        if age_ns <= self.exit_walk_start_ns:
            return 0.0
        if age_ns >= self.exit_giveup_ns:
            return 1.0
        span = self.exit_giveup_ns - self.exit_walk_start_ns
        return (age_ns - self.exit_walk_start_ns) / span if span > 0 else 1.0

    def _entry_ok(self, maker_fee, validator, book_id, st, now, volume_cap) -> bool:
        """Gate opening new inventory on fee regime, turnover cap and per-window RT budget."""
        fee_ok = maker_fee is None or maker_fee <= MAX_ENTRY_MAKER_FEE
        vol_ok = self._rolled_quote_volume(validator, book_id, now) < volume_cap
        rt_ok = self._rt_count(st, now) < RT_MAX
        return fee_ok and vol_ok and rt_ok

    def _reconcile_quotes(
        self, response, account, book_id: int, desired: dict[int, tuple[float, float]],
    ) -> None:
        """Keep resting orders that already match a desired (side, price, ~qty); cancel the
        rest; post any desired side still missing. Preserves time priority on stable quotes
        and respects the 5-instructions/book budget."""
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
                and abs(o.price - want[0]) < self._tick / 2
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
                # Attach loan settlement only when we truly lack the base to sell (opening a
                # short), not when it is merely reserved in the prior reduce order.
                short_sale = self._avail(account.base_balance) < qty
                self._submit_limit(
                    response, book_id, OrderDirection.SELL, qty, px, post_only=True,
                    settlement=self._loan_settlement(account) if short_sale else LoanSettlementOption.NONE,
                )

    # ------------------------------------------------------------------ events
    def onTrade(self, event: TradeEvent, validator: str | None = None) -> None:
        """Route both maker and taker fills into FIFO accounting. is_buy mirrors the
        validator: taker+BUY or maker+SELL-aggressor both mean WE bought."""
        if event.bookId is None:
            return
        validator = validator or self._active_validator
        if validator is None:
            return
        if self.uid == event.takerAgentId:
            is_buy = event.side == OrderDirection.BUY
            fee = event.takerFee
        elif self.uid == event.makerAgentId:
            is_buy = event.side == OrderDirection.SELL   # aggressor sold -> our resting BUY filled
            fee = event.makerFee
        else:
            return
        ts_ns = int(event.timestamp) if event.timestamp else self._step_ts_ns
        self._record_trade_volume(validator, event.bookId, event.quantity, event.price, ts_ns)
        self._apply_fill(validator, event.bookId, is_buy, event.quantity, event.price, fee, ts_ns)

    # ------------------------------------------------------------------ state / FIFO
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
            f"[DualEdge uid={self.uid}] priceDecimals={price_decimals} tick={self._tick} "
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
    def _activity_elapsed(st: _BookState, now: int) -> int:
        ref = st.last_rt_ns if st.last_rt_ns > 0 else st.seen_ns
        return now - ref

    @staticmethod
    def _avail(balance) -> float:
        """Quantity that becomes spendable this step once our resting orders are cancelled:
        free + reserved (reserved is locked only in our own orders, freed by _cancel_all)."""
        if balance is None:
            return 0.0
        return (balance.free or 0.0) + (balance.reserved or 0.0)

    def _book_equity(self, account, mid: float) -> float:
        q = account.quote_balance
        b = account.base_balance
        quote = (q.free + (q.reserved or 0.0)) if q else 0.0
        base = (b.free + (b.reserved or 0.0)) if b else 0.0
        return quote + base * mid

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
        """Apply a fill through FIFO matching (mirrors validator._match_trade_fifo). A fill
        that closes opposing inventory realizes P&L and round-trip volume; record it for
        kappa and log the round-trip. Residual quantity opens a new lot."""
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
                hold_s=(ts - matched_ts) / _NS if matched_ts else None,
                side=("buy" if is_buy else "sell"), exit_px=price, rtv=rtv,
                gross_pnl=gross, net_pnl=realized, kappa_before=kappa_before,
                kappa_after=st.kappa3, rt_window_n=rt_window_n, st=st,
            )

    def _match_fifo(
        self, inv: _Inv, is_buy: bool, qty: float, price: float, fee: float, ts: int,
    ) -> tuple[float, float, int | None, float]:
        """FIFO match a fill against opposing lots. Returns
        (realized_pnl_net_of_fees, roundtrip_volume, oldest_matched_ts, gross_pnl)."""
        close_book = inv.shorts if is_buy else inv.longs   # buying closes shorts; selling closes longs
        open_book = inv.longs if is_buy else inv.shorts
        realized = 0.0
        gross = 0.0
        rtv = 0.0
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
            open_fee = fee * remaining * qinv
            open_book.append((ts, remaining, price, open_fee))
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

    def _book_pnl_series(self, validator: str, book_id: int, now: int) -> list[float]:
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
        """Validator-faithful per-book Kappa-3 on MAD-normalized realized P&L."""
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

    # ------------------------------------------------------------------ RT logging
    @staticmethod
    def _rt_log_enabled(validator: str) -> bool:
        return validator == MAIN_VALIDATOR

    @staticmethod
    def _fmt_kappa_pair(before: float | None, after: float | None) -> str:
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
        self, *, validator: str, book_id: int, ts: int, hold_s: float | None, side: str,
        exit_px: float, rtv: float, gross_pnl: float, net_pnl: float,
        kappa_before: float | None, kappa_after: float | None, rt_window_n: int, st: _BookState,
    ) -> None:
        if not self._rt_log_enabled(validator):
            return
        hold_str = f"{hold_s:.2f}" if hold_s is not None else "n/a"
        bt.logging.info(
            f"[DualEdge uid={self.uid} RT] book={book_id} mode={st.mode} close={side} "
            f"rtv={rtv:.4f} exit={exit_px:.4f} hold_s={hold_str} "
            f"gross_pnl={gross_pnl:+.4f} net_pnl={net_pnl:+.4f} "
            f"kappa={self._fmt_kappa_pair(kappa_before, kappa_after)} "
            f"close_rt_n={rt_window_n} close_rt_pnl={self._fmt_rt_pnl_list(st, ts)}"
        )

    # ================================================================== TAKER mode
    def _desired_mode(self, account, best_bid: float, best_ask: float, mid: float) -> str:
        """Route a book to TAKER only when crossing the spread is PAID FOR: the taker fee is a
        rebate at least as large as the half-spread plus a margin, and at least the rebate gate
        (>= SL/2, so a stop/time exit is cushioned toward break-even). Otherwise MAKER.

        This is the heart of the dual edge: on deep-rebate books the proven TakerScalper clip
        (tiny size, seconds-long hold, hard TP/SL) out-scores passive making; everywhere else the
        KappaMaker spread-capture engine runs."""
        taker_fee = self._taker_fee_rate(account)
        if taker_fee is None or taker_fee >= 0.0:
            return MODE_MAKER
        rebate_bps = -taker_fee * 1e4
        half_spread_bps = (best_ask - best_bid) / mid * 0.5 * 1e4 if mid > 0 else 1e9
        if rebate_bps >= TAKER_REBATE_GATE_BPS and rebate_bps >= half_spread_bps + ROUTER_TAKER_MARGIN_BPS:
            return MODE_TAKER
        return MODE_MAKER

    def _taker_step(
        self, response, validator, book_id, book, account, inv, net,
        best_bid, best_ask, mid, volume_cap, vol_dp, now,
    ) -> None:
        """TakerScalper discipline on a shared FIFO inventory: hold -> TP/SL/max-hold exit;
        flat -> scalp the rebate edge (throttled), else keep activity alive."""
        st = self._bstate(validator, book_id)
        if abs(net) >= self.exch_min:
            self._taker_exit(response, book_id, account, inv, net, best_bid, best_ask, vol_dp, now, st)
            return
        throttled = st.last_close_ns and (now - st.last_close_ns) < int(TAKER_REOPEN_GAP_S * _NS)
        if not throttled and self._taker_open(
            response, validator, book_id, book, account, best_bid, best_ask, mid, volume_cap, now, st, vol_dp,
        ):
            return
        if self._activity_elapsed(st, now) >= self.activity_deadline_ns:
            self._activity_close(response, validator, book_id, book, account, inv, net,
                                 best_bid, best_ask, vol_dp)

    def _taker_exit(
        self, response, book_id, account, inv, net, best_bid, best_ask, vol_dp, now, st,
    ) -> None:
        """Close the whole clip with a marketable order on TP / SL / max-hold. Exposure is one
        small lot held <= a few seconds, so market slippage is negligible and each realized loss
        is bounded (the cubic Kappa downside stays tiny)."""
        if net > 0:
            avg = self._side_avg(inv.longs)
            ts0 = inv.longs[0][0]
            gross_bps = (best_bid - avg) / avg * 1e4 if avg > 0 else 0.0
        else:
            avg = self._side_avg(inv.shorts)
            ts0 = inv.shorts[0][0]
            gross_bps = (avg - best_ask) / avg * 1e4 if avg > 0 else 0.0
        held = now - ts0
        if held < int(TAKER_MIN_HOLD_S * _NS):
            return
        if gross_bps >= TAKER_TP_BPS:
            reason = "tp"
        elif gross_bps <= -TAKER_SL_BPS:
            reason = "sl"
        elif held >= int(TAKER_MAX_HOLD_S * _NS):
            reason = "time"
        else:
            return
        self._cancel_all(response, account, book_id)
        if net > 0:
            q = round(min(abs(net), self._avail(account.base_balance)), vol_dp)
            if q < self.exch_min:
                return
            response.market_order(book_id, OrderDirection.SELL, q, stp=STP.CANCEL_OLDEST)
        else:
            q = round(abs(net), vol_dp)
            if q < self.exch_min:
                return
            response.market_order(book_id, OrderDirection.BUY, q, stp=STP.CANCEL_OLDEST,
                                  settlement_option=self._loan_settlement(account))
        st.last_close_ns = now
        st.taker_open_ns = 0
        bt.logging.info(
            f"[DualEdge uid={self.uid}] TAKER-EXIT book={book_id} reason={reason} "
            f"net={net:+.4f} q={q} gross={gross_bps:+.1f}bps held_s={held / _NS:.2f}"
        )

    def _taker_open(
        self, response, validator, book_id, book, account,
        best_bid, best_ask, mid, volume_cap, now, st, vol_dp,
    ) -> bool:
        """Open ONE clip in the microprice-bias direction when the rebate round-trip is +EV.
        Shares the turnover and per-window RT-budget gates with the maker engine."""
        if self._rolled_quote_volume(validator, book_id, now) >= volume_cap:
            return False
        if self._rt_count(st, now) >= RT_MAX:
            return False
        taker_fee = self._taker_fee_rate(account)
        if taker_fee is None:
            return False
        rebate_bps = -taker_fee * 1e4
        half_spread_bps = (best_ask - best_bid) / mid * 0.5 * 1e4 if mid > 0 else 1e9
        # +EV estimate: rebate earned on both legs minus the spread we cross twice.
        est_bps = 2.0 * rebate_bps - 2.0 * half_spread_bps
        if est_bps < TAKER_EDGE_MARGIN_BPS:
            return False
        direction = self._taker_bias(book, mid)
        q = round(self.quote_lot, vol_dp)
        if q < self.exch_min:
            return False
        if direction == OrderDirection.BUY:
            if self._avail(account.quote_balance) < q * best_ask:
                return False
            response.market_order(book_id, OrderDirection.BUY, q, stp=STP.CANCEL_OLDEST,
                                  settlement_option=self._loan_settlement(account))
        else:
            if self._avail(account.base_balance) < q:    # never naked-short; sell base we hold
                return False
            response.market_order(book_id, OrderDirection.SELL, q, stp=STP.CANCEL_OLDEST)
        st.taker_open_ns = now
        bt.logging.info(
            f"[DualEdge uid={self.uid}] TAKER-OPEN book={book_id} "
            f"{'BUY' if direction == OrderDirection.BUY else 'SELL'} q={q} "
            f"rebate={rebate_bps:.1f}bps est={est_bps:+.1f}bps"
        )
        return True

    @staticmethod
    def _taker_bias(book, mid: float) -> int:
        """microprice vs mid -> directional lean; tie -> long (mirrors TakerScalper)."""
        bid, ask = book.bids[0], book.asks[0]
        denom = bid.quantity + ask.quantity
        micro = (ask.price * bid.quantity + bid.price * ask.quantity) / denom if denom > 0 else mid
        return OrderDirection.SELL if micro < mid else OrderDirection.BUY

    @staticmethod
    def _side_avg(lots) -> float:
        tot = sum(q for _, q, _, _ in lots)
        return sum(q * p for _, q, p, _ in lots) / tot if tot > 0 else 0.0

    def _taker_fee_rate(self, account) -> float | None:
        fees = getattr(account, "fees", None)
        rate = getattr(fees, "taker_fee_rate", None) if fees is not None else None
        try:
            return float(rate) if rate is not None else None
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------ market helpers
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
            "book_id": book_id,
            "direction": direction,
            "quantity": qty,
            "price": price,
            "stp": STP.CANCEL_OLDEST,
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
    launch(DualEdgeAgent)
