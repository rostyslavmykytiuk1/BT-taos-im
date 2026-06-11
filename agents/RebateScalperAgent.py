"""
RebateScalperAgent — rebate round-trip + activity-maintenance engine for subnet 79.

Two open modes, one fee router. Activity factor MUST stay 1.0 on every book, so
every book completes >=1 round-trip per validator sampling window no matter what.

Tick flow
---------
1. Close all open legs (TP / SL / max_hold).  [unchanged exit logic]
2. Scan BOOKS_PER_STEP books per tick (round-robin over all books).
3. Per flat book, decide via the fee router (`_manage_open`). Two concerns, in order:

   PROFIT ENGINE  (runs every visit; source of the rebate income)
     paid := (taker_fee <= 0 AND est_pnl > 0)        # genuinely paid to take, +EV
     if paid AND kappa OK AND vol < cap AND rt_n < RT_MAX:
         -> TAKER open, microprice side ("kappa")    # scalp; the gate opens freely
                                                      # for the first few RTs, so
                                                      # start-up trades flow through here
     (if the engine opens nothing, fall through to the activity backstop)

   ACTIVITY BACKSTOP  (guarantees >=1 RT per book per validator window)
     T := seconds since the last completed RT (anchored to first-sight at start-up,
          so we never see a huge "overdue" gap before any RT exists)
     Phase A   T <  MAKER_START (~427-450s)   -> idle (a recent RT still counts)
     Phase B   MAKER_START <= T < DEADLINE    -> if paid: TAKER open ("taker_force")
                                                 else:    rest one post-only BUY @ bid
                                                          (maker), re-quote if buried
     Phase C   T >= DEADLINE (~509-525s)      -> MUST cross now, profitable or not:
                                                 TAKER ("taker_force" if paid, else
                                                 BUY "scratch_taker"). DEADLINE is kept
                                                 below RT_WINDOW (570s) so the RT always
                                                 lands inside the activity window.

4. At most ONE of {resting maker quote, in-flight taker, open position} per book.
   Logging (RT close + scratch events) is emitted for the main validator only.
"""

import math
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

MIN_ORDER_SIZE = 0.255

# Hold / exit (from uid 215 RT study: tail losses at ~6s hold dominate LPM3; 6bps SL was noise)
MIN_HOLD_S = 1.5
MAX_HOLD_S = 4.0
MIN_GROSS_TP_BPS = 2.5
MAX_GROSS_SL_BPS = 4.0

# Margin leverage used for a SELL open when free base inventory is insufficient.
SHORT_LEVERAGE = 1.0

# RT window for opens (validator activity sampling = 10m)
RT_WINDOW_S = 570.0                # ~10 min
RT_MAX = 20                        # max RTs per book in window

# Scratch (activity-maintenance) timing, measured as seconds since the last RT.
# All thresholds stay safely below the validator's ~600s sampling window so a
# completed RT is always registered in time. Jittered per-uid in initialize().
SCRATCH_MAKER_START_S = 450.0      # begin resting a maker quote for activity
SCRATCH_TAKER_DEADLINE_S = 525.0   # hard backstop: cross with a taker to guarantee the RT
SCRATCH_REQUOTE_THROTTLE_S = 12.0  # min gap between maker re-quotes (preserve queue priority)

# Kappa-3 (3h history for score projection)
KAPPA_TAU = 0.0
KAPPA_MIN_OBS = 3
KAPPA_MIN_LOOKBACK_S = 5400.0      # 90 min
KAPPA_RT_HISTORY_S = 10_800.0      # 3h RT history kept for kappa
KAPPA_PROJ_IMPROVE = 0.003
KAPPA_PROJ_TOLERANCE = 0.008

BOOKS_PER_STEP = 10

# Volume cap (kappa path only)
CAPITAL_TURNOVER_CAP = 10.0
VOLUME_SAFETY = 0.8
VOLUME_ASSESSMENT_NS = 86_400_000_000_000

# RT logs only for the scoring validator (per-validator rt_events otherwise mislead).
MAIN_VALIDATOR = "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF"


@dataclass
class _Position:
    qty: float = 0.0
    avg: float = 0.0
    entry_ts: int = 0
    entry_fee: float = 0.0


@dataclass
class _BookState:
    last_rt_ns: int = 0
    pending_open: bool = False
    rt_events: list[tuple[int, float]] = field(default_factory=list)
    kappa3: float | None = None
    vol_log: list[tuple[int, float]] = field(default_factory=list)
    window_anchor_ns: int = 0          # first-sight ts; activity clock anchor before any RT
    scratch_reposted_ns: int = 0       # last maker (re)quote ts, for throttle


@dataclass
class _RtLogCtx:
    """Open snapshot stashed at submit; close_reason added at close submit; logged at RT fill."""
    open_reason: str = "?"
    side: str = "?"
    open_rt_window_n: int = 0
    open_rt_pnl_list: str = "[]"
    est_pnl: float | None = None
    kappa_at_open: float | None = None
    kappa_proj: float | None = None
    taker_bps: float | None = None
    close_reason: str = "fill"


class RebateScalperAgent(FinanceSimulationAgent):
    def initialize(self) -> None:
        bt.logging.set_info()

        self.min_order_size = MIN_ORDER_SIZE
        self._min_qty = MIN_ORDER_SIZE / 2
        self._volume_decimals: int | None = None
        self.volume_assessment_ns = VOLUME_ASSESSMENT_NS

        jitter = ((self.uid * 2654435761) % 1000) / 1000.0
        max_hold_s = MAX_HOLD_S * (0.92 + 0.16 * jitter)

        self.min_hold_ns = int(MIN_HOLD_S * _NS)
        self.max_hold_ns = int(max_hold_s * _NS)
        self.rt_window_ns = int(RT_WINDOW_S * _NS)
        self.kappa_rt_history_ns = int(KAPPA_RT_HISTORY_S * _NS)
        self.kappa_min_lookback_ns = int(KAPPA_MIN_LOOKBACK_S * _NS)

        # Scratch timing, jittered per-uid so 10 miners don't act in lockstep.
        # Deadline stays < RT_WINDOW_S so the maintenance RT always lands in time.
        scratch_start_s = SCRATCH_MAKER_START_S * (0.95 + 0.05 * jitter)      # ~427-450s
        scratch_deadline_s = SCRATCH_TAKER_DEADLINE_S * (0.97 + 0.03 * jitter)  # ~509-525s
        self.scratch_maker_start_ns = int(scratch_start_s * _NS)
        self.scratch_taker_deadline_ns = int(scratch_deadline_s * _NS)
        self.scratch_requote_throttle_ns = int(SCRATCH_REQUOTE_THROTTLE_S * _NS)

        self.positions: dict[str, dict[int, _Position]] = {}
        self.books_state: dict[str, dict[int, _BookState]] = {}
        self._sim_id: dict[str, str] = {}
        self._open_step: dict[str, int] = {}
        self._rt_log: dict[tuple[str, int], _RtLogCtx] = {}
        self._step_ts_ns: int = 0
        self._active_validator: str | None = None

        bt.logging.info(
            f"[RebateScalper uid={self.uid}] lot={MIN_ORDER_SIZE} "
            f"hold={MIN_HOLD_S}-{max_hold_s:.1f}s tp={MIN_GROSS_TP_BPS}bps sl={MAX_GROSS_SL_BPS}bps "
            f"books/step={BOOKS_PER_STEP} "
            f"rt_window={RT_WINDOW_S / 60:.0f}m max={RT_MAX} "
            f"scratch_maker={scratch_start_s:.0f}s taker={scratch_deadline_s:.0f}s "
            f"rt_log={MAIN_VALIDATOR[:8]}"
        )

    def update(self, state: MarketSimulationStateUpdate) -> None:
        self._step_ts_ns = int(state.timestamp)
        self._active_validator = state.dendrite.hotkey
        self._ensure_simulation(self._active_validator, state.config.simulation_id)
        super().update(state)

    def _ensure_simulation(self, validator: str, simulation_id: str | None) -> None:
        """Drop per-validator state when the validator starts a new simulation."""
        if self._sim_id.get(validator) == simulation_id:
            return
        self._book_positions(validator).clear()
        self.books_state.pop(validator, None)
        self._rt_log = {k: v for k, v in self._rt_log.items() if k[0] != validator}
        self._open_step.pop(validator, None)
        if simulation_id is not None:
            self._sim_id[validator] = simulation_id
        else:
            self._sim_id.pop(validator, None)
        bt.logging.info(
            f"[RebateScalper uid={self.uid}] new simulation: {validator[:8]} "
            f"sim_id={simulation_id}"
        )

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        validator = state.dendrite.hotkey
        cfg = self.simulation_config
        self._sync_order_size(cfg.volumeDecimals)

        vol_dp = cfg.volumeDecimals
        volume_cap = CAPITAL_TURNOVER_CAP * cfg.miner_wealth * VOLUME_SAFETY
        now = state.timestamp

        for book_id in self._books_to_close(state, validator):
            book = state.books.get(book_id)
            if book is None:
                continue
            try:
                self._handle_close(response, validator, book_id, book, vol_dp, now)
            except Exception as ex:
                bt.logging.warning(f"[RebateScalper uid={self.uid}] close {book_id}: {ex}")

        open_step = self._open_step.get(validator, 0)
        for book_id in self._open_book_batch(open_step):
            book = state.books.get(book_id)
            account = self.accounts.get(book_id) if book else None
            if book is None or account is None:
                continue
            try:
                self._manage_open(
                    response, validator, book_id, book, account, vol_dp, volume_cap, now,
                )
            except Exception as ex:
                bt.logging.warning(f"[RebateScalper uid={self.uid}] open {book_id}: {ex}")

        self._open_step[validator] = open_step + 1
        return response

    # ------------------------------------------------------------------ events
    def onOrderRejected(self, event: OrderPlacementEvent) -> None:
        if event.bookId is None or not self._active_validator:
            return
        validator = self._active_validator
        self._bstate(validator, event.bookId).pending_open = False
        self._rt_log.pop((validator, event.bookId), None)

    def onTrade(self, event: TradeEvent, validator: str | None = None) -> None:
        """Route our fills into position accounting. The traded direction is the
        aggressor side for our taker fills, and its inverse for our maker fills."""
        if event.bookId is None:
            return
        validator = validator or self._active_validator
        if validator is None:
            return

        is_taker = self.uid == event.takerAgentId
        if is_taker:
            direction = event.side
        elif self.uid == event.makerAgentId:
            direction = OrderDirection.SELL if event.side == OrderDirection.BUY else OrderDirection.BUY
        else:
            return

        ts_ns = int(event.timestamp) if event.timestamp else self._step_ts_ns
        self._record_trade_volume(validator, event.bookId, event.quantity, event.price, ts_ns)
        fee = event.takerFee if is_taker else event.makerFee
        self._apply_fill(validator, event.bookId, direction, event.quantity, event.price, fee, ts_ns)

    # ------------------------------------------------------------------ state
    @staticmethod
    def _effective_min_order_size(volume_decimals: int) -> float:
        return max(MIN_ORDER_SIZE, 10 ** (-volume_decimals))

    def _sync_order_size(self, volume_decimals: int) -> None:
        if volume_decimals == self._volume_decimals:
            return
        self._volume_decimals = volume_decimals
        lot = round(self._effective_min_order_size(volume_decimals), volume_decimals)
        self.min_order_size = lot
        self._min_qty = lot / 2
        bt.logging.info(
            f"[RebateScalper uid={self.uid}] volumeDecimals={volume_decimals} effective_min={lot}"
        )

    def _book_positions(self, validator: str) -> dict[int, _Position]:
        return self.positions.setdefault(validator, {})

    def _bstate(self, validator: str, book_id: int) -> _BookState:
        return self.books_state.setdefault(validator, {}).setdefault(book_id, _BookState())

    @staticmethod
    def _clear_position(pos: _Position) -> None:
        pos.qty = pos.avg = pos.entry_fee = 0.0
        pos.entry_ts = 0

    def _record_trade_volume(
        self, validator: str, book_id: int, qty: float, price: float, ts_ns: int,
    ) -> None:
        vol = float(qty) * float(price)
        if vol <= 0:
            return
        self._bstate(validator, book_id).vol_log.append((ts_ns, vol))

    def _prune_vol_log(self, st: _BookState, now_ns: int) -> None:
        cutoff = now_ns - self.volume_assessment_ns
        st.vol_log = [(t, v) for t, v in st.vol_log if t >= cutoff]

    def _rolled_quote_volume(self, validator: str, book_id: int, now_ns: int) -> float:
        st = self._bstate(validator, book_id)
        self._prune_vol_log(st, now_ns)
        return sum(v for _, v in st.vol_log)

    def _apply_fill(
        self,
        validator: str,
        book_id: int,
        direction: OrderDirection,
        qty: float,
        price: float,
        trade_fee: float,
        ts: int,
    ) -> None:
        """Update the per-book position from a fill (taker or maker leg).

        Opening/adding accumulates the volume-weighted entry and entry fee; a fill
        that reduces the position realizes a clean per-RT PnL (gross minus allocated
        open+close fees), records it for kappa, and logs the round-trip.
        """
        pos = self._book_positions(validator).setdefault(book_id, _Position())
        signed = qty if direction == OrderDirection.BUY else -qty
        prev = pos.qty
        entry_avg = pos.avg

        # Same side (or opening from flat): grow the position, blend the entry price.
        if prev == 0 or (prev > 0) == (signed > 0):
            total = abs(prev) + qty
            pos.avg = (pos.avg * abs(prev) + price * qty) / total if total > 0 else price
            pos.qty = prev + signed
            if prev == 0:
                pos.entry_ts = ts
                pos.entry_fee = trade_fee
                self._bstate(validator, book_id).pending_open = False
            else:
                pos.entry_fee += trade_fee
            return

        closed_qty = min(qty, abs(prev))
        if closed_qty >= self._min_qty and entry_avg > 0:
            rpnl = (price - entry_avg) * closed_qty if prev > 0 else (entry_avg - price) * closed_qty
            open_fee, close_fee = self._allocate_close_fees(
                pos.entry_fee, abs(prev), closed_qty, qty, trade_fee,
            )
            net_pnl = rpnl - open_fee - close_fee
            st = self._bstate(validator, book_id)
            entry_ts = pos.entry_ts
            kappa_before = st.kappa3
            rt_window_n = self._rt_count(st, ts)
            st.last_rt_ns = ts
            self._record_rt_close(validator, book_id, ts, net_pnl)
            self._log_rt(
                validator=validator,
                book_id=book_id,
                ts=ts,
                hold_s=(ts - entry_ts) / _NS if entry_ts else None,
                entry_avg=entry_avg,
                exit_px=price,
                gross_pnl=rpnl,
                open_fee=open_fee,
                close_fee=close_fee,
                net_pnl=net_pnl,
                kappa_before=kappa_before,
                kappa_after=st.kappa3,
                rt_window_n=rt_window_n,
                st=st,
            )
            pos.entry_fee -= open_fee

        pos.qty = prev + signed
        if abs(pos.qty) < 1e-12:
            self._clear_position(pos)
        elif (prev > 0) != (pos.qty > 0):
            pos.avg, pos.entry_ts = price, ts
            pos.entry_fee = trade_fee * (abs(pos.qty) / qty) if qty > 0 else 0.0

    def _prune_rt_events(self, st: _BookState, now: int) -> bool:
        cutoff = now - self.kappa_rt_history_ns
        before = len(st.rt_events)
        st.rt_events = [(t, p) for t, p in st.rt_events if t >= cutoff]
        return len(st.rt_events) != before

    def _record_rt_close(self, validator: str, book_id: int, ts: int, net_pnl: float) -> None:
        st = self._bstate(validator, book_id)
        self._prune_rt_events(st, ts)
        st.rt_events.append((ts, net_pnl))
        st.window_anchor_ns = ts          # reset the activity clock at each completed RT
        self._refresh_book_kappa(validator, book_id, ts)

    def _sync_book_rt_state(self, validator: str, book_id: int, now: int) -> _BookState:
        st = self._bstate(validator, book_id)
        if self._prune_rt_events(st, now):
            self._refresh_book_kappa(validator, book_id, now)
        return st

    # ------------------------------------------------------------------ kappa-3
    def _global_rt_timestamps(self, validator: str, now: int) -> list[int]:
        cutoff = now - self.kappa_rt_history_ns
        ts_set: set[int] = set()
        for st in self.books_state.get(validator, {}).values():
            for ts, _ in st.rt_events:
                if ts >= cutoff:
                    ts_set.add(ts)
        return sorted(ts_set)

    def _book_pnl_series(
        self,
        validator: str,
        book_id: int,
        now: int,
        extra: tuple[int, float] | None = None,
    ) -> list[float]:
        timestamps = self._global_rt_timestamps(validator, now)
        if extra is not None:
            extra_ts, _ = extra
            if extra_ts not in timestamps:
                timestamps = sorted(timestamps + [extra_ts])
        if not timestamps:
            return []

        cutoff = now - self.kappa_rt_history_ns
        by_ts = {t: p for t, p in self._bstate(validator, book_id).rt_events if t >= cutoff}
        if extra is not None:
            by_ts[extra[0]] = extra[1]
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
        if not pnl_series:
            return None
        if sum(1 for x in pnl_series if x != 0.0) < KAPPA_MIN_OBS:
            return None

        med = cls._median(pnl_series)
        mad = max(cls._median(abs(x - med) for x in pnl_series), 1e-6)
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

    @staticmethod
    def _estimate_rt_pnl(taker_rate: float, book, qty: float) -> float:
        """Conservative taker RT: buy at ask, sell at bid, fees on both legs."""
        if not book.bids or not book.asks:
            return 0.0
        bid = book.bids[0].price
        ask = book.asks[0].price
        if bid <= 0 or ask <= 0:
            return 0.0
        gross = (bid - ask) * qty
        return gross - taker_rate * (ask + bid) * qty

    def _project_kappa(
        self, validator: str, book_id: int, now: int, estimated_pnl: float,
    ) -> float | None:
        close_ts = now + self.min_hold_ns
        return self._kappa3_raw(
            self._book_pnl_series(validator, book_id, now, extra=(close_ts, estimated_pnl)),
        )

    def _kappa_open_ok(self, st: _BookState, validator: str, book_id: int, estimated: float, now: int) -> bool:
        if estimated <= 0.0:
            return False

        projected = self._project_kappa(validator, book_id, now, estimated)
        rt_n = self._rt_count(st, now, self.kappa_rt_history_ns)
        if projected is None:
            return rt_n < KAPPA_MIN_OBS

        current = st.kappa3
        if current is None:
            return projected >= 0.0
        if current < 0.05:
            return projected >= current + KAPPA_PROJ_IMPROVE
        return projected >= current - KAPPA_PROJ_TOLERANCE

    # ------------------------------------------------------------------ RT logging
    @staticmethod
    def _rt_log_enabled(validator: str) -> bool:
        return validator == MAIN_VALIDATOR

    @staticmethod
    def _fmt_pnl(value: float | None) -> str:
        return "n/a" if value is None else f"{value:+.4f}"

    @staticmethod
    def _fmt_kappa_pair(before: float | None, after: float | None) -> str:
        if before is None and after is None:
            return "n/a"
        if before is None:
            return f"n/a->{after:.4f}"
        if after is None:
            return f"{before:.4f}->n/a"
        delta = after - before
        sign = "+" if delta >= 0 else ""
        return f"{before:.4f}->{after:.4f} ({sign}{delta:.4f})"

    def _fmt_rt_pnl_list(self, st: _BookState, now: int, window_ns: int | None = None) -> str:
        cutoff = now - (window_ns if window_ns is not None else self.rt_window_ns)
        pnls = [p for ts, p in st.rt_events if ts >= cutoff]
        if not pnls:
            return "[]"
        body = ", ".join(f"{p:+.4f}" for p in pnls)
        return f"[{body}]"

    def _stash_rt_open(
        self,
        validator: str,
        book_id: int,
        book,
        account,
        direction: OrderDirection,
        now: int,
        open_reason: str | None,
    ) -> None:
        if not self._rt_log_enabled(validator):
            return
        st = self._bstate(validator, book_id)
        rate = self._taker_fee_rate(account)
        est_pnl = self._estimate_rt_pnl(rate, book, self.min_order_size) if rate is not None else 0.0
        self._rt_log[(validator, book_id)] = _RtLogCtx(
            open_reason=open_reason or "?",
            side="long" if direction == OrderDirection.BUY else "short",
            open_rt_window_n=self._rt_count(st, now),
            open_rt_pnl_list=self._fmt_rt_pnl_list(st, now),
            est_pnl=est_pnl,
            kappa_at_open=st.kappa3,
            kappa_proj=self._project_kappa(validator, book_id, now, est_pnl),
            taker_bps=(rate * 1e4) if rate is not None else None,
        )

    def _set_rt_close_reason(self, validator: str, book_id: int, close_reason: str) -> None:
        if not self._rt_log_enabled(validator):
            return
        ctx = self._rt_log.get((validator, book_id))
        if ctx is not None:
            ctx.close_reason = close_reason

    def _log_rt(
        self,
        *,
        validator: str,
        book_id: int,
        ts: int,
        hold_s: float | None,
        entry_avg: float,
        exit_px: float,
        gross_pnl: float,
        open_fee: float,
        close_fee: float,
        net_pnl: float,
        kappa_before: float | None,
        kappa_after: float | None,
        rt_window_n: int,
        st: _BookState,
    ) -> None:
        if not self._rt_log_enabled(validator):
            self._rt_log.pop((validator, book_id), None)
            return
        ctx = self._rt_log.pop((validator, book_id), _RtLogCtx())
        hold_str = f"{hold_s:.2f}" if hold_s is not None else "n/a"
        taker_bps_str = f"{ctx.taker_bps:.2f}" if ctx.taker_bps is not None else "n/a"

        bt.logging.info(
            f"[RebateScalper uid={self.uid} RT] "
            f"book={book_id} "
            f"open={ctx.open_reason}/{ctx.side} "
            f"open_rt_n={ctx.open_rt_window_n} open_rt_pnl={ctx.open_rt_pnl_list} "
            f"est_pnl={self._fmt_pnl(ctx.est_pnl)} "
            f"kappa_open={self._fmt_kappa_pair(ctx.kappa_at_open, ctx.kappa_proj)} "
            f"taker_bps={taker_bps_str} "
            f"close={ctx.close_reason} hold_s={hold_str} "
            f"entry={entry_avg:.4f} exit={exit_px:.4f} "
            f"gross_pnl={gross_pnl:+.4f} open_fee={open_fee:+.4f} close_fee={close_fee:+.4f} "
            f"net_pnl={net_pnl:+.4f} "
            f"kappa_close={self._fmt_kappa_pair(kappa_before, kappa_after)} "
            f"close_rt_n={rt_window_n} close_rt_pnl={self._fmt_rt_pnl_list(st, ts)}"
        )

    def _log_scratch_post(
        self, validator: str, book_id: int, price: float, taker_rate: float,
        account, elapsed_ns: int,
    ) -> None:
        if not self._rt_log_enabled(validator):
            return
        maker_rate = self._maker_fee_rate(account)
        maker_bps = f"{maker_rate * 1e4:.2f}" if maker_rate is not None else "n/a"
        bt.logging.info(
            f"[RebateScalper uid={self.uid} SCR] book={book_id} post BUY@{price:.4f} "
            f"T={elapsed_ns / _NS:.0f}s maker_bps={maker_bps} taker_bps={taker_rate * 1e4:.2f}"
        )

    def _log_scratch_taker(self, validator: str, book_id: int, elapsed_ns: int) -> None:
        if not self._rt_log_enabled(validator):
            return
        bt.logging.info(
            f"[RebateScalper uid={self.uid} SCR] book={book_id} taker BUY (deadline) "
            f"T={elapsed_ns / _NS:.0f}s"
        )

    # ------------------------------------------------------------------ open gates
    def _rt_count(self, st: _BookState, now: int, window_ns: int | None = None) -> int:
        cutoff = now - (self.rt_window_ns if window_ns is None else window_ns)
        return sum(1 for ts, _ in st.rt_events if ts >= cutoff)

    def _open_book_batch(self, step: int) -> list[int]:
        book_ids = sorted(self.accounts.keys())
        if not book_ids:
            return []

        batch_count = (len(book_ids) + BOOKS_PER_STEP - 1) // BOOKS_PER_STEP
        start = (step % batch_count) * BOOKS_PER_STEP
        return book_ids[start:start + BOOKS_PER_STEP]

    def _manage_open(
        self,
        response,
        validator: str,
        book_id: int,
        book,
        account,
        vol_dp: int,
        volume_cap: float,
        now: int,
    ) -> None:
        """Fee router for a single book. Holds at most one in-flight action per book
        and guarantees a maintenance RT before the activity window lapses."""
        pos = self._book_positions(validator).setdefault(book_id, _Position())
        self._reconcile_position(account, pos, vol_dp)

        st = self._bstate(validator, book_id)
        if st.window_anchor_ns == 0:
            st.window_anchor_ns = now

        # We already hold a position -> the close path manages it; never stack opens.
        if abs(pos.qty) >= self._min_qty:
            return

        st = self._sync_book_rt_state(validator, book_id, now)
        # A taker open is in flight (submitted, fill not yet confirmed) -> wait.
        if st.pending_open:
            return

        rate = self._taker_fee_rate(account)
        direction, mid = self._book_bias(book)
        if mid is None or rate is None:
            return

        resting = self._resting_order(account)
        rt_n = self._rt_count(st, now)
        est_pnl = self._estimate_rt_pnl(rate, book, self.min_order_size)
        paid = rate <= 0.0 and est_pnl > 0.0   # genuinely paid to take, +EV round-trip

        # Activity clock: seconds since the last completed RT, anchored to first-sight
        # before any RT exists (so start-up never sees a huge "overdue" gap).
        ref = st.last_rt_ns if st.last_rt_ns > 0 else st.window_anchor_ns
        elapsed_ns = now - ref

        # ---------------------------------- PROFIT ENGINE (runs every visit) --------
        # The rebate income comes from here: when genuinely paid to take, scalp as many
        # kappa-gated round-trips as quality/volume allow. The gate opens freely for the
        # first few RTs, so start-up trades flow through here as normal profitable opens.
        if paid:
            if resting is not None:
                # Regime favours taking; drop any stale maker quote first.
                self._cancel_scratch(response, validator, book_id, resting)
                return
            if (
                rt_n < RT_MAX
                and self._rolled_quote_volume(validator, book_id, now) < volume_cap
                and self._kappa_open_ok(st, validator, book_id, est_pnl, now)
            ):
                if self._taker_open(response, validator, book_id, account, book, direction):
                    self._prune_vol_log(st, now)
                    self._stash_rt_open(validator, book_id, book, account, direction, now, "kappa")
                return
            # Paid but kappa paused / capped -> fall through to the activity backstop,
            # which still crosses for us (we are not flat on activity time).

        # ---------------------------------- ACTIVITY BACKSTOP (anchored phases) -----
        # Reached only when the profit engine did not open. Guarantees >=1 RT per book
        # per validator window, regardless of fee regime.

        # Phase A: a recent RT still satisfies activity -> idle, hold no stray quote.
        if elapsed_ns < self.scratch_maker_start_ns:
            if resting is not None:
                self._cancel_scratch(response, validator, book_id, resting)
            return

        # Phase C: deadline reached -> MUST complete the RT this window; cross now,
        # profitable or not (losing the activity factor is far costlier than the spread).
        if elapsed_ns >= self.scratch_taker_deadline_ns:
            if resting is not None:
                # Drop the quote this tick; the taker fires next visit when truly flat.
                self._cancel_scratch(response, validator, book_id, resting)
                return
            side = direction if paid else OrderDirection.BUY
            if self._taker_open(response, validator, book_id, account, book, side):
                tag = "taker_force" if paid else "scratch_taker"
                self._stash_rt_open(validator, book_id, book, account, side, now, tag)
                if not paid:
                    self._log_scratch_taker(validator, book_id, elapsed_ns)
            return

        # Phase B (maker window): cross if we are paid to (kappa just paused the open);
        # otherwise rest a single post-only BUY @ bid and re-quote only if buried.
        if paid:
            if self._taker_open(response, validator, book_id, account, book, direction):
                self._stash_rt_open(validator, book_id, book, account, direction, now, "taker_force")
            return
        if resting is None:
            self._post_scratch_maker(response, validator, book_id, book, account, st, now, rate, elapsed_ns)
        else:
            self._maybe_reprice_scratch(response, validator, book_id, book, st, resting, now)

    def _resting_order(self, account):
        """Our only resting orders are scratch maker quotes (all other opens are
        market orders), so the first open order, if any, is that quote."""
        orders = getattr(account, "orders", None)
        if not orders:
            return None
        return orders[0]

    def _post_scratch_maker(
        self, response, validator: str, book_id: int, book, account, st: _BookState,
        now: int, rate: float, elapsed_ns: int,
    ) -> None:
        if not book.bids:
            return
        bid_px = book.bids[0].price
        qty = self.min_order_size
        if bid_px <= 0 or account.quote_balance.free < qty * bid_px:
            return
        response.limit_order(
            book_id=book_id,
            direction=OrderDirection.BUY,
            quantity=qty,
            price=bid_px,
            currency=OrderCurrency.BASE,
            stp=STP.CANCEL_OLDEST,
            timeInForce=TimeInForce.GTC,
            postOnly=True,
        )
        st.scratch_reposted_ns = now
        self._stash_rt_open(
            validator, book_id, book, account, OrderDirection.BUY, now, "scratch_maker",
        )
        self._log_scratch_post(validator, book_id, bid_px, rate, account, elapsed_ns)

    def _maybe_reprice_scratch(
        self, response, validator: str, book_id: int, book, st: _BookState, resting, now: int,
    ) -> None:
        if not book.bids:
            return
        best_bid = book.bids[0].price
        # Keep queue priority unless the market has left our quote behind.
        if resting.price is None or resting.price >= best_bid:
            return
        if (now - st.scratch_reposted_ns) < self.scratch_requote_throttle_ns:
            return
        # Cancel now; a fresh quote is posted next visit once flat with no resting order.
        self._cancel_scratch(response, validator, book_id, resting)

    def _cancel_scratch(self, response, validator: str, book_id: int, resting) -> None:
        response.cancel_order(book_id=book_id, order_id=resting.id)
        if self._rt_log_enabled(validator):
            self._rt_log.pop((validator, book_id), None)

    # ------------------------------------------------------------------ market
    @staticmethod
    def _allocate_close_fees(
        entry_fee: float, pos_qty: float, closed_qty: float, trade_qty: float, trade_fee: float,
    ) -> tuple[float, float]:
        if pos_qty <= 0 or closed_qty <= 0:
            return 0.0, 0.0
        open_fee = entry_fee * (closed_qty / pos_qty)
        close_fee = trade_fee if closed_qty >= trade_qty else trade_fee * (closed_qty / trade_qty)
        return open_fee, close_fee

    @staticmethod
    def _mid(book) -> float | None:
        if not book.bids or not book.asks:
            return None
        return 0.5 * (book.bids[0].price + book.asks[0].price)

    @staticmethod
    def _microprice(book) -> float | None:
        if not book.bids or not book.asks:
            return None
        bid, ask = book.bids[0], book.asks[0]
        denom = bid.quantity + ask.quantity
        if denom <= 0:
            return 0.5 * (bid.price + ask.price)
        return (ask.price * bid.quantity + bid.price * ask.quantity) / denom

    @classmethod
    def _book_bias(cls, book) -> tuple[OrderDirection, float | None]:
        """microprice vs mid → direction; tie → long."""
        mid = cls._mid(book)
        micro = cls._microprice(book)
        if mid is None or micro is None:
            return OrderDirection.BUY, mid
        direction = OrderDirection.SELL if micro < mid else OrderDirection.BUY
        return direction, mid

    @staticmethod
    def _exit_gross_bps(pos: _Position, bid: float | None, ask: float | None, mid: float) -> float:
        if pos.qty > 0:
            exit_px = bid if bid and bid > 0 else mid
            return (exit_px - pos.avg) / pos.avg * 1e4
        exit_px = ask if ask and ask > 0 else mid
        return (pos.avg - exit_px) / pos.avg * 1e4

    @staticmethod
    def _loan_settlement(account) -> LoanSettlementOption:
        quote_loan = getattr(account, "quote_loan", 0.0) or 0.0
        return LoanSettlementOption.FIFO if quote_loan > 0 else LoanSettlementOption.NONE

    def _taker_fee_rate(self, account) -> float | None:
        return self._fee_rate(account, "taker_fee_rate")

    def _maker_fee_rate(self, account) -> float | None:
        return self._fee_rate(account, "maker_fee_rate")

    @staticmethod
    def _fee_rate(account, attr: str) -> float | None:
        fees = getattr(account, "fees", None)
        if fees is None:
            return None
        rate = getattr(fees, attr, None)
        if rate is None:
            return None
        try:
            return float(rate)
        except (TypeError, ValueError):
            return None

    def _books_to_close(self, state: MarketSimulationStateUpdate, validator: str) -> list[int]:
        ids = set(state.books.keys())
        for book_id, pos in self._book_positions(validator).items():
            if abs(pos.qty) >= self._min_qty:
                ids.add(book_id)
        return sorted(ids)

    def _handle_close(
        self, response, validator: str, book_id: int, book, vol_dp: int, now: int,
    ) -> None:
        """Exit an open position after the min hold on TP / SL / max-hold, whichever
        triggers first. Shared by every open mode (rebate, kappa, scratch)."""
        mid = self._mid(book)
        if mid is None or mid <= 0:
            return
        account = self.accounts.get(book_id)
        if account is None:
            return

        pos = self._book_positions(validator).setdefault(book_id, _Position())
        self._reconcile_position(account, pos, vol_dp)

        if abs(pos.qty) < self._min_qty or pos.avg <= 0:
            return

        hold_ns = (now - pos.entry_ts) if pos.entry_ts else 0
        if hold_ns < self.min_hold_ns:
            return

        bid = book.bids[0].price if book.bids else None
        ask = book.asks[0].price if book.asks else None
        gross_bps = self._exit_gross_bps(pos, bid, ask, mid)
        if gross_bps >= MIN_GROSS_TP_BPS:
            exit_reason = "tp"
        elif gross_bps <= -MAX_GROSS_SL_BPS:
            exit_reason = "sl"
        elif hold_ns >= self.max_hold_ns:
            exit_reason = "time"
        else:
            return

        self._set_rt_close_reason(validator, book_id, exit_reason)
        self._close_position(response, account, book_id, pos, vol_dp)

    def _taker_open(
        self,
        response,
        validator: str,
        book_id: int,
        account,
        book,
        direction: OrderDirection,
    ) -> bool:
        """Cross the spread for one lot. BUY needs quote balance; SELL uses base
        inventory, falling back to a margin short. Sets pending_open and returns
        whether the order was submitted."""
        qty = self.min_order_size
        st = self._bstate(validator, book_id)

        if direction == OrderDirection.BUY:
            ask_px = book.asks[0].price if book.asks else 0.0
            if ask_px <= 0 or account.quote_balance.free < qty * ask_px:
                return False
            self._submit_market(response, book_id, OrderDirection.BUY, qty)
        else:
            bid_px = book.bids[0].price if book.bids else 0.0
            if bid_px <= 0:
                return False
            free_base = account.base_balance.free if account.base_balance else 0.0
            if free_base >= qty:
                self._submit_market(response, book_id, OrderDirection.SELL, qty)
            else:
                quote_loan = getattr(account, "quote_loan", 0.0) or 0.0
                self._submit_market(
                    response,
                    book_id,
                    OrderDirection.SELL,
                    qty,
                    leverage=0.0 if quote_loan > 0 else SHORT_LEVERAGE,
                    settlement=self._loan_settlement(account),
                )

        st.pending_open = True
        return True

    def _close_position(
        self, response, account, book_id: int, pos: _Position, vol_dp: int,
    ) -> None:
        # Flatten the entire position in one market order so a partial-fill stack
        # (partial maker fill + deadline taker) never leaves base dust behind.
        qty = round(abs(pos.qty), vol_dp)
        if qty < self._min_qty:
            return

        if pos.qty > 0:
            free = account.base_balance.free if account.base_balance else 0.0
            qty = round(min(qty, free), vol_dp)
            if qty < self._min_qty:
                return
            self._submit_market(response, book_id, OrderDirection.SELL, qty)
        else:
            self._submit_market(
                response,
                book_id,
                OrderDirection.BUY,
                qty,
                settlement=self._loan_settlement(account),
            )

    @staticmethod
    def _submit_market(
        response,
        book_id: int,
        direction: OrderDirection,
        qty: float,
        *,
        leverage: float = 0.0,
        settlement: LoanSettlementOption = LoanSettlementOption.NONE,
    ) -> None:
        kwargs: dict[str, Any] = {
            "book_id": book_id,
            "direction": direction,
            "quantity": qty,
            "currency": OrderCurrency.BASE,
            "stp": STP.CANCEL_OLDEST,
        }
        if leverage > 0:
            kwargs["leverage"] = leverage
        if settlement != LoanSettlementOption.NONE:
            kwargs["settlement_option"] = settlement
        response.market_order(**kwargs)

    def _reconcile_position(self, account, pos: _Position, vol_dp: int) -> None:
        # Clamp tracked size to real free base (covers external settlement drift). We do
        # NOT cap at one lot: a partial-fill stack can legitimately exceed it, and the
        # close path flattens the whole position so nothing is left as untracked dust.
        if pos.qty > 0 and account.base_balance is not None:
            free = account.base_balance.free
            if free < self._min_qty:
                self._clear_position(pos)
            else:
                pos.qty = round(min(pos.qty, free), vol_dp)


if __name__ == "__main__":
    launch(RebateScalperAgent)
