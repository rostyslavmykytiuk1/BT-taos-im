# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
PureMakerAgent — breakeven-gated two-sided maker for Subnet 79 (τaos).

A maker SPECIALIST: the goal is to be TOP when the market is maker-favorable (rich spreads, mean
reversion), NOT to be all-weather robust — AdaptiveRouter covers trending regimes. So it quotes
aggressively (symmetric 10/10 economics, breakeven coverage, high frequency) and carries ONE
reactive safety net (_regime_backoff) that idles a book only once making there is clearly losing.

Design derived from analysis of UID 165 (top maker) trade history vs validator scoring:

BREAKEVEN GATE — the primary book selector:
  Quote on any book where capturing the spread covers both maker-fee legs: spread >= 2*maker_fee*mid
  (evaluated live per book). A wide-spread book is +EV even at a high fee — the top makers (109/165)
  make on ALL 128 books, 165 even at +11.5bps — and a thin-spread book loses even at a low fee. This
  replaces the old fixed 12bps fee ceiling, which idled profitable wide-spread high-fee books and
  quoted unprofitable thin-spread ones. Sub-breakeven books idle and fall inside the 37.5%
  inactive-book tolerance (48 free), dropping from the median at no cost.

PROFIT MECHANISM on active books:
  Posting passive quotes captures the spread plus the mean-reversion of heavy aggressive flow:
  fill on the way down, exit on the bounce. Hold window typically ~1-2 min. Per-RT win target >=
  the stop (positive skew) so the cubed downside in kappa-3 stays small.

KEY FAILURE MODE OF UID 165 (and what we fix):
  165 ran no stop-loss. Books 116 (−106 PnL) and 106 (−48 PnL) held positions 500-660s as
  the price trended without reverting. We add a stop (EXIT_STOP_LOSS_BPS) plus a reduce-price
  floor, a per-book loss-streak cooldown, and the reactive regime backoff below.

MECHANICS (per book, each step):
  prune/kappa → managed exit → risk guard → breakeven gate → activity backstop → quote → reconcile.
  Managed exit runs FIRST so a stop is never blocked by the risk guard's early return.

  * Managed exit → IOC-cut the held side if underwater >= EXIT_STOP_LOSS_BPS OR lot age >=
            EXIT_GIVEUP_S. Slippage capped. Bounds each realized loss.
  * Breakeven gate → take new inventory only where spread >= 2*maker_fee*mid. Held inventory on a
            book that drops sub-breakeven is still managed DOWN (never force-dumped), so a volatile
            spread can't churn the book active/idle.
  * Flat  → quote both sides inside the touch, gated by breakeven + cooldowns + RT/volume budget.
  * Hold  → ONLY the reducing side (never average into the bag). Reduce price walks from the profit
            target toward the touch with lot age, FLOORED at the stop so a reduce fill never loses
            more than the IOC cut would.
  * Loss streak → STREAK_LIMIT consecutive losing RTs → pause new entries for STREAK_COOLDOWN_S.
  * Regime backoff → window net realized PnL < -(BACKOFF_NET_WINS × avg win) → idle the book for
            BACKOFF_COOLDOWN_S, then re-test. The maker specialist's safety net; free in good regimes.
  * Activity → an active book always closes >= 1 RT per window (activity factor 1.0), even while
            backed off; only sub-breakeven books go idle -> kappa=None -> dropped (37.5% free budget).
  * FIFO inventory mirrors the validator's _match_trade_fifo exactly (oldest-lot matching).
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
MAX_INVENTORY_LOTS = 2.0           # tighter than DualEdge: maker runs smaller book
MAX_INVENTORY_EQUITY_FRAC = 0.10
RISK_TRIM_SLIPPAGE_BPS = 6.0

# ---- breakeven gate: make on any book where the spread covers both maker-fee legs ----
# Condition is spread >= 2*maker_fee*mid (computed live per book), NOT a fixed fee ceiling: a fixed
# ceiling wrongly idled profitable wide-spread high-fee books (UID 165 makes at +11.5bps) and quoted
# unprofitable thin-spread low-fee ones. The tight stop + reduce floor keep high-fee books kappa-safe.

# ---- profit target ----
# Target >= stop on purpose. Kappa-3 CUBES the downside, so the realized win must be at least as
# big as the realized loss (non-negative skew) — the opposite of the old PnL-tuned 8bps target,
# which sat BELOW the 15bps stop and guaranteed losses bigger than wins (negative skew that drags
# kappa even at a high win rate). Now matches the symmetric 10/10 economics proven on the router.
# Adaptive floor: max(TP_BPS_BASE, TP_FEE_MULT × maker_fee_bps) ensures two-leg cost coverage.
TP_BPS_BASE = 10.0
TP_FEE_MULT = 2.0                  # floor = 2× maker_fee (covers both legs + small buffer)
QUOTE_EXPIRY_S = 12.0

# ---- managed exit: cut fast and small ----
# mean-reverting books bounce within ~1-1.5 min; if they haven't by EXIT_GIVEUP_S, they're trending.
# Stop tightened 15->10bps: the dominant kappa lever is keeping every realized loss small (the cube),
# and the reduce-price FLOOR (=stop) plus this stop cap the loss tail. Smaller stop also = positive
# skew vs the ~10bps target. Size (clip + inventory cap), not a wide stop, absorbs noise.
EXIT_WALK_START_S = 30.0           # start walking reduce toward touch after 30s
EXIT_GIVEUP_S = 90.0              # IOC-cut at 1.5min if still not reverted (was 120)
EXIT_STOP_LOSS_BPS = 10.0          # immediate IOC-cut: price moved 10bps against oldest lot (was 15)
EXIT_CUT_SLIPPAGE_BPS = 4.0        # tighter cut concession (was 5)
REENTRY_COOLDOWN_S = 30.0         # after a managed cut, re-quote fast (was 120): kappa rewards
                                   # FREQUENCY (coverage^2/3); long pauses starve a book's RT count.

# ---- per-book loss-streak cooldown ----
# If the last STREAK_LIMIT round-trips are ALL losses on a book: halt new entries for
# STREAK_COOLDOWN_S. Prevents the "keep re-entering on a trending book" disaster.
# (165's book 116: 98.9% loss rate → -106 total PnL. This gate would have paused after 4 RTs.)
STREAK_LIMIT = 4
STREAK_COOLDOWN_S = 300.0           # shorter toxic-book pause (was 600) so the book rejoins coverage

# ---- reactive regime backoff: the maker specialist's safety net ----
# PureMaker MAXIMIZES maker-favorable regimes (rich spreads, mean reversion) — that is its job, not
# all-weather robustness (AdaptiveRouter handles trends). But the validator scores continuously, so
# when a book's regime turns hostile, making must not keep bleeding. Signal = recent NET realized PnL
# relative to the book's OWN average win (scale-free across price levels, and — unlike a win-rate
# trigger — it catches a SKEW bleed: a 65%-win book whose few losses are 2-3x the wins still nets
# negative). When the net loss over the window exceeds BACKOFF_NET_WINS average wins, idle the book
# and re-test. FREE in good regimes: a net-positive book never trips it, so it never costs upside.
BACKOFF_WINDOW_S = 600.0           # rolling window of realized RTs used to judge the book's regime
BACKOFF_MIN_RTS = 6                # need >= this many RTs in the window before judging (enough signal)
BACKOFF_NET_WINS = 3.0             # back off when window net PnL < -(this × avg win); conservative so
                                   # good-regime upside is never sacrificed to a false positive
BACKOFF_COOLDOWN_S = 180.0         # idle the book this long, then re-test the regime

# ---- activity (active books only) ----
RT_WINDOW_S = 570.0
ACTIVITY_DEADLINE_S = 480.0
RT_MAX = 15                        # more RTs than KappaMaker on quality books
FORCE_TRIM_SLIPPAGE_BPS = 5.0

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
    # per-book loss-streak cooldown
    loss_streak: int = 0                   # consecutive losing RT count
    streak_cooldown_until_ns: int = 0      # no new entries before this timestamp


class PureMakerAgent(FinanceSimulationAgent):

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
        self.streak_cooldown_ns = int(STREAK_COOLDOWN_S * _NS)
        self.backoff_window_ns = int(BACKOFF_WINDOW_S * _NS)
        self.backoff_cooldown_ns = int(BACKOFF_COOLDOWN_S * _NS)
        self.kappa_rt_history_ns = int(KAPPA_RT_HISTORY_S * _NS)
        self.kappa_min_lookback_ns = int(KAPPA_MIN_LOOKBACK_S * _NS)

        self.inv: dict[str, dict[int, _Inv]] = {}
        self.books_state: dict[str, dict[int, _BookState]] = {}
        self._sim_id: dict[str, str] = {}
        self._step_ts_ns: int = 0
        self._active_validator: str | None = None

        bt.logging.info(
            f"[PureMaker uid={self.uid}] PURE-MAKER lot={QUOTE_LOT} exch_min={self.exch_min} "
            f"gate=breakeven(spread>=2*fee) "
            f"tp_base={self.tp_bps_base:.1f}bps tp_floor={TP_FEE_MULT}×fee "
            f"exit_walk={EXIT_WALK_START_S:.0f}-{giveup_s:.1f}s stop={EXIT_STOP_LOSS_BPS}bps "
            f"reentry={REENTRY_COOLDOWN_S}s "
            f"streak(limit={STREAK_LIMIT} cooldown={STREAK_COOLDOWN_S}s) "
            f"inv_cap={MAX_INVENTORY_LOTS}lot/{MAX_INVENTORY_EQUITY_FRAC:.0%}eq "
            f"activity={activity_s:.0f}s rt_max={RT_MAX} rt_log={MAIN_VALIDATOR[:8]}"
        )

    # ------------------------------------------------------------------ lifecycle
    def update(self, state: MarketSimulationStateUpdate) -> None:
        self._step_ts_ns = int(state.timestamp)
        self._active_validator = state.dendrite.hotkey
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
            f"[PureMaker uid={self.uid}] new simulation: {validator[:8]} sim_id={simulation_id}"
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
                bt.logging.warning(f"[PureMaker uid={self.uid}] step {book_id}: {ex}")

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
        maker_fee = self._maker_fee_rate(account)
        active = self._is_active_book(maker_fee, best_ask - best_bid, mid)
        holding = abs(net) >= self.exch_min

        # 1) MANAGED EXIT — stop-loss / giveup takes priority over everything, including the
        #    inventory risk guard. Without this ordering, _risk_trim's early-return blocks
        #    _managed_exit on any over-cap book, leaving underwater positions with no stop-loss
        #    protection until the trim IOC finally fills (which can take hundreds of steps on
        #    illiquid books, letting losses grow to hundreds of bps).
        if self._managed_exit(response, book_id, account, inv, net, best_bid, best_ask, vol_dp, now):
            st.last_cut_ns = now
            return

        # 2) RISK GUARD — drains breached inventory after stop-loss has had its turn.
        if self._risk_trim(response, validator, book_id, account, inv, net, mid, vol_dp):
            return

        # 3) BREAKEVEN GATE — a sub-breakeven book takes NO new inventory. If flat, it idles here.
        #    If it still HOLDS inventory from a prior active window (or the spread just tightened
        #    under it), fall through and MANAGE it down gracefully (managed exit + passive reduce)
        #    rather than force-dumping via IOC. Because the spread is volatile, the gate is checked
        #    only for new entries — never to trigger a forced exit — so a book hovering at breakeven
        #    cannot churn IOC losses by flickering active/idle.
        if not active and not holding:
            return

        # 4) ACTIVITY BACKSTOP — guarantee >=1 RT per window so the activity factor stays 1.0 on
        #    EVERY book we make on, INCLUDING a backed-off one. The backoff only blocks voluntary
        #    quoting (the bleed); this floor keeps activity alive. It cannot starve the factor: the
        #    backoff (180s) is shorter than this deadline (480s), which is shorter than the
        #    validator's 600s decay grace. Only sub-breakeven books (active=False) are allowed to go
        #    idle -> kappa=None -> dropped from the median for free (the 37.5% inactive budget).
        if active and self._activity_elapsed(st, now) >= self.activity_deadline_ns:
            if self._activity_close(response, validator, book_id, book, account,
                                    inv, net, best_bid, best_ask, vol_dp):
                return

        # 5) DESIRED QUOTES — reduce-only when holding (any book); two-sided entry only when the
        #    breakeven gate in _entry_ok clears. A held sub-breakeven book reduces but never re-opens.
        desired = self._desired_quotes(
            validator, book_id, book, account, inv, net,
            best_bid, best_ask, mid, maker_fee, volume_cap, now,
        )
        self._reconcile_quotes(response, account, book_id, desired)

    # ------------------------------------------------------------------ breakeven gate
    def _is_active_book(self, maker_fee: float | None, spread: float, mid: float) -> bool:
        """BREAKEVEN gate: quote only when capturing the spread covers both maker-fee legs,
        i.e. spread >= 2*maker_fee*mid. A wide-spread book is +EV even at a HIGH fee (UID 165
        makes profitably at +11.5bps); a thin-spread book loses even at a LOW fee. This replaces a
        fixed fee ceiling, which wrongly idled profitable wide-spread high-fee books and quoted
        unprofitable thin-spread low-fee ones. The tight stop + reduce floor bound the per-trade
        loss, so high-fee (potentially adverse) books stay kappa-safe."""
        if maker_fee is None:
            return True   # unknown fee → optimistic; other entry gates still apply
        if mid <= 0:
            return False
        return spread >= 2.0 * maker_fee * mid

    # ------------------------------------------------------------------ risk guard
    def _risk_trim(
        self, response, validator: str, book_id: int, account, inv: _Inv, net: float,
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
        self._cancel_all(response, account, book_id)
        if net > 0:
            trim = round(min(trim, self._avail(account.base_balance)), vol_dp)
            if trim < self.exch_min:
                return False
            px = round(mid * (1.0 - slip), self._price_decimals)
            self._submit_limit(response, book_id, OrderDirection.SELL, trim, px,
                               ioc=True, post_only=False)
        else:
            px = round(mid * (1.0 + slip), self._price_decimals)
            self._submit_limit(response, book_id, OrderDirection.BUY, trim, px,
                               ioc=True, post_only=False, settlement=self._loan_settlement(account))
        bt.logging.info(
            f"[PureMaker uid={self.uid}] RISK-TRIM book={book_id} net={net:+.4f} trim={trim}"
        )
        return True

    # ------------------------------------------------------------------ managed exit
    def _managed_exit(
        self, response, book_id: int, account, inv: _Inv, net: float,
        best_bid: float, best_ask: float, vol_dp: int, now: int,
    ) -> bool:
        """IOC-cut the whole reducible side when the oldest lot is too old (EXIT_GIVEUP_S, the
        mean-reversion window) or too underwater (EXIT_STOP_LOSS_BPS). Either path bounds the loss."""
        if abs(net) < self.exch_min:
            return False
        slip = EXIT_CUT_SLIPPAGE_BPS / 1e4
        if net > 0:
            ts, _, px0, _ = inv.longs[0]
            uw = (px0 - best_bid) / px0 * 1e4 if px0 > 0 else 0.0
            aged = now - ts >= self.exit_giveup_ns
            stopped = uw >= EXIT_STOP_LOSS_BPS
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
            uw = (best_ask - px0) / px0 * 1e4 if px0 > 0 else 0.0
            aged = now - ts >= self.exit_giveup_ns
            stopped = uw >= EXIT_STOP_LOSS_BPS
            if not (aged or stopped):
                return False
            q = round(self._short_qty(inv), vol_dp)
            if q < self.exch_min:
                return False
            self._cancel_all(response, account, book_id)
            px = round(best_ask * (1.0 + slip), self._price_decimals)
            self._submit_limit(response, book_id, OrderDirection.BUY, q, px, ioc=True, post_only=False,
                               settlement=self._loan_settlement(account))
        reason = "stop" if stopped and not aged else ("age" if aged and not stopped else "age+stop")
        bt.logging.info(
            f"[PureMaker uid={self.uid}] MANAGED-EXIT book={book_id} reason={reason} "
            f"net={net:+.4f} q={q} @~{px} uw={uw:.1f}bps"
        )
        return True

    # ------------------------------------------------------------------ activity backstop
    def _activity_close(
        self, response, validator: str, book_id: int, book, account, inv: _Inv, net: float,
        best_bid: float, best_ask: float, vol_dp: int,
    ) -> bool:
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
            # Flat: seed a tiny long so the next step can close it for round-trip volume.
            q = lot
            if quote_avail < q * best_ask * (1.0 + slip):
                return False
            px = round(best_ask * (1.0 + slip), self._price_decimals)
            self._submit_limit(response, book_id, OrderDirection.BUY, q, px, ioc=True, post_only=False)
        bt.logging.info(f"[PureMaker uid={self.uid}] ACTIVITY-CLOSE book={book_id} net={net:+.4f}")
        return True

    # ------------------------------------------------------------------ quoting
    def _desired_quotes(
        self, validator: str, book_id: int, book, account, inv: _Inv, net: float,
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

        free_base = account.base_balance.free if account.base_balance else 0.0
        free_quote = account.quote_balance.free if account.quote_balance else 0.0
        base_avail = self._avail(account.base_balance)
        quote_avail = self._avail(account.quote_balance)

        spread = best_ask - best_bid
        improve = tick if spread > 2 * tick else 0.0
        bid_inside = round(best_bid + improve, pdp)
        ask_inside = round(best_ask - improve, pdp)
        if bid_inside >= ask_inside:
            bid_inside, ask_inside = round(best_bid, pdp), round(best_ask, pdp)

        if net >= self.exch_min:
            # Holding long → passive SELL reduce, priced off worst lot, walking toward touch.
            age = now - inv.longs[0][0]
            worst_px = max(p for _, _, p, _ in inv.longs)
            px = self._reduce_price(True, worst_px, age, ask_inside, base_target, pdp)
            q = round(min(self._long_qty(inv), base_avail), vdp)
            if q >= self.exch_min and px > 0:
                desired[OrderDirection.SELL] = (px, q)
        elif net <= -self.exch_min:
            # Holding short → passive BUY reduce.
            age = now - inv.shorts[0][0]
            worst_px = min(p for _, _, p, _ in inv.shorts)
            px = self._reduce_price(False, worst_px, age, bid_inside, base_target, pdp)
            q = round(self._short_qty(inv), vdp)
            if q >= self.exch_min and px > 0 and quote_avail >= q * px:
                desired[OrderDirection.BUY] = (px, q)
        elif st.last_cut_ns > 0 and now - st.last_cut_ns < self.reentry_cooldown_ns:
            # Post-cut cooldown: don't re-enter immediately after a managed exit.
            pass
        elif self._entry_ok(maker_fee, spread, mid, validator, book_id, st, now, volume_cap):
            # Flat and all gates clear → quote both sides.
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
        """Walk the passive-reduce limit price from the profit target toward the touch with lot age.
        FLOOR the realized loss at the stop: a resting reduce quote must never fill worse than the
        managed-exit IOC would. Otherwise an aged underwater lot's reduce walks all the way to the
        touch and fills there on a gap, locking a loss many times the stop (kappa-3 cubes that tail).
        Anything below the floor is left to _managed_exit's bounded IOC instead."""
        w = self._exit_walk(age_ns)
        stop = EXIT_STOP_LOSS_BPS / 1e4
        if is_long:
            ideal = max(touch_inside, px0 * (1.0 + base_target))
            px = ideal + (touch_inside - ideal) * w
            return round(max(px, px0 * (1.0 - stop)), pdp)
        ideal = min(touch_inside, px0 * (1.0 - base_target))
        px = ideal + (touch_inside - ideal) * w
        return round(min(px, px0 * (1.0 + stop)), pdp)

    def _exit_walk(self, age_ns: int) -> float:
        if age_ns <= self.exit_walk_start_ns:
            return 0.0
        if age_ns >= self.exit_giveup_ns:
            return 1.0
        span = self.exit_giveup_ns - self.exit_walk_start_ns
        return (age_ns - self.exit_walk_start_ns) / span if span > 0 else 1.0

    def _entry_ok(
        self, maker_fee: float | None, spread: float, mid: float, validator: str, book_id: int,
        st: _BookState, now: int, volume_cap: float,
    ) -> bool:
        """Gate new inventory: breakeven gate, streak cooldown, RT budget, volume cap."""
        if not self._is_active_book(maker_fee, spread, mid):
            return False
        if now < st.streak_cooldown_until_ns:
            return False
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
        ts_ns = int(event.timestamp) if event.timestamp else self._step_ts_ns
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
            f"[PureMaker uid={self.uid}] priceDecimals={price_decimals} tick={self._tick} "
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
        inv = self._inv(validator, book_id)
        realized, rtv, matched_ts, gross = self._match_fifo(inv, is_buy, qty, price, fee, ts)
        if rtv > 0:
            st = self._bstate(validator, book_id)
            kappa_before = st.kappa3
            rt_window_n = self._rt_count(st, ts)
            st.last_rt_ns = ts
            self._record_rt_close(validator, book_id, ts, realized)
            self._update_streak(st, book_id, realized, ts)
            self._regime_backoff(st, book_id, ts)
            self._log_rt(
                validator=validator, book_id=book_id, ts=ts,
                hold_s=(ts - matched_ts) / _NS if matched_ts else None,
                side="buy" if is_buy else "sell", exit_px=price, rtv=rtv,
                gross_pnl=gross, net_pnl=realized,
                kappa_before=kappa_before, kappa_after=st.kappa3,
                rt_window_n=rt_window_n, st=st,
            )

    def _update_streak(self, st: _BookState, book_id: int, realized_pnl: float, now_ns: int) -> None:
        """Track consecutive losses. After STREAK_LIMIT in a row, pause new entries."""
        if realized_pnl > 0:
            st.loss_streak = 0
        else:
            st.loss_streak += 1
            if st.loss_streak >= STREAK_LIMIT:
                st.streak_cooldown_until_ns = max(
                    st.streak_cooldown_until_ns, now_ns + self.streak_cooldown_ns)
                bt.logging.info(
                    f"[PureMaker uid={self.uid}] STREAK-COOLDOWN book={book_id} "
                    f"streak={st.loss_streak} cooldown={STREAK_COOLDOWN_S}s"
                )

    def _regime_backoff(self, st: _BookState, book_id: int, now_ns: int) -> None:
        """Idle a book whose recent making has turned net-LOSING. Trigger = window net realized PnL
        below -(BACKOFF_NET_WINS × avg win), i.e. the losses have eaten more than a few wins' worth.
        Net-based (not win-rate) so it catches a skew bleed (few but oversized losses). Sets the same
        cooldown the streak guard uses, so _entry_ok blocks new entries while existing inventory is
        still managed down. Re-tests after the cooldown. A net-positive book never trips this."""
        cutoff = now_ns - self.backoff_window_ns
        recent = [p for t, p in st.rt_events if t >= cutoff]
        if len(recent) < BACKOFF_MIN_RTS:
            return
        net = sum(recent)
        if net >= 0:
            return
        wins = [p for p in recent if p > 0]
        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        if net < -BACKOFF_NET_WINS * avg_win:   # avg_win==0 (all losses) => any net<0 trips it
            until = now_ns + self.backoff_cooldown_ns
            if until > st.streak_cooldown_until_ns:
                st.streak_cooldown_until_ns = until
                bt.logging.info(
                    f"[PureMaker uid={self.uid}] REGIME-BACKOFF book={book_id} "
                    f"net={net:+.3f} avg_win={avg_win:.3f} n={len(recent)} idle={BACKOFF_COOLDOWN_S:.0f}s"
                )

    def _match_fifo(
        self, inv: _Inv, is_buy: bool, qty: float, price: float, fee: float, ts: int,
    ) -> tuple[float, float, int | None, float]:
        close_book = inv.shorts if is_buy else inv.longs
        open_book = inv.longs if is_buy else inv.shorts
        realized = 0.0; gross = 0.0; rtv = 0.0; remaining = qty
        matched_ts: int | None = None
        qinv = 1.0 / qty if qty > 0 else 0.0

        while remaining > self._flat_eps and close_book:
            o_ts, o_qty, o_px, o_fee = close_book[0]
            if matched_ts is None:
                matched_ts = o_ts
            take = min(o_qty, remaining)
            price_pnl = (o_px - price) * take if is_buy else (price - o_px) * take
            if o_qty <= remaining + self._flat_eps:
                close_fee = fee * o_qty * qinv; open_fee = o_fee
                close_book.popleft()
            else:
                close_fee = fee * take * qinv; open_fee = o_fee * (take / o_qty)
                close_book[0] = (o_ts, o_qty - take, o_px, o_fee - open_fee)
            realized += price_pnl - open_fee - close_fee
            gross += price_pnl; rtv += take; remaining -= take

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
            f"[PureMaker uid={self.uid} RT] book={book_id} close={side} "
            f"rtv={rtv:.4f} exit={exit_px:.4f} hold_s={hold_str} "
            f"gross={gross_pnl:+.4f} net={net_pnl:+.4f} "
            f"kappa={self._fmt_kappa(kappa_before, kappa_after)} "
            f"streak={st.loss_streak} "
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
    launch(PureMakerAgent)
