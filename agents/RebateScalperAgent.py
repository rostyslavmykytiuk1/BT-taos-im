"""
RebateScalperAgent — taker rebate round-trip engine for subnet 79.

Tick flow
---------
1. Close all open legs (TP / SL / max_hold).
2. Scan BOOKS_PER_STEP books per tick (round-robin over all books).
3. Per book in RT_WINDOW_S: activity open at rt_n==0, kappa if rebate and rt_n < RT_MAX,
   skip at cap. Volume cap on kappa path only.
4. Open side from order book: microprice vs mid → long (BUY) or short (SELL).
"""

import math
from dataclasses import dataclass, field
from typing import Any

import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol import FinanceAgentResponse, MarketSimulationStateUpdate
from taos.im.protocol.events import OrderPlacementEvent, TradeEvent
from taos.im.protocol.models import LoanSettlementOption, OrderCurrency, OrderDirection, STP

_NS = 1_000_000_000

MIN_ORDER_SIZE = 0.255

# Hold / exit (from uid 215 RT study: tail losses at ~6s hold dominate LPM3; 6bps SL was noise)
MIN_HOLD_S = 1.5
MAX_HOLD_S = 4.0
MIN_GROSS_TP_BPS = 2.5
MAX_GROSS_SL_BPS = 4.0

MIN_REOPEN_GAP_S = 4.0

# Open side: microprice vs mid (order book only)
SHORT_LEVERAGE = 1.0             # margin for SELL when base inventory insufficient

# RT window for opens (validator activity sampling = 10m)
RT_WINDOW_S = 570.0                # ~10 min
RT_MAX = 20                        # max RTs per book in window

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
        self.min_reopen_gap_ns = int(MIN_REOPEN_GAP_S * (0.9 + 0.2 * jitter) * _NS)
        self.rt_window_ns = int(RT_WINDOW_S * _NS)
        self.kappa_rt_history_ns = int(KAPPA_RT_HISTORY_S * _NS)
        self.kappa_min_lookback_ns = int(KAPPA_MIN_LOOKBACK_S * _NS)

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
                do_open, direction, open_reason = self._open_decision(
                    validator, book_id, book, account, vol_dp, volume_cap, now,
                )
                if (
                    do_open
                    and direction is not None
                    and self._taker_open(response, validator, book_id, account, book, direction)
                ):
                    self._prune_vol_log(self._bstate(validator, book_id), now)
                    self._stash_rt_open(
                        validator, book_id, book, account, direction, now, open_reason,
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
        pos = self._book_positions(validator).setdefault(book_id, _Position())
        signed = qty if direction == OrderDirection.BUY else -qty
        prev = pos.qty
        entry_avg = pos.avg

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

    # ------------------------------------------------------------------ open gates
    def _reopen_ok(self, st: _BookState, now: int) -> bool:
        return st.last_rt_ns == 0 or (now - st.last_rt_ns) >= self.min_reopen_gap_ns

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

    def _open_decision(
        self,
        validator: str,
        book_id: int,
        book,
        account,
        vol_dp: int,
        volume_cap: float,
        now: int,
    ) -> tuple[bool, OrderDirection | None, str | None]:
        pos = self._book_positions(validator).get(book_id, _Position())
        self._reconcile_position(account, pos, vol_dp)
        if abs(pos.qty) >= self._min_qty:
            return False, None, None

        st = self._sync_book_rt_state(validator, book_id, now)
        if st.pending_open or not self._reopen_ok(st, now):
            return False, None, None

        rate = self._taker_fee_rate(account)
        direction, mid = self._book_bias(book)
        if mid is None or rate is None:
            return False, None, None

        rt_n = self._rt_count(st, now)
        if rt_n >= RT_MAX:
            return False, None, None
        if rt_n == 0:
            return True, direction, "force"

        estimated = self._estimate_rt_pnl(rate, book, self.min_order_size)
        if (
            rate <= 0.0
            and self._rolled_quote_volume(validator, book_id, now) < volume_cap
            and self._kappa_open_ok(st, validator, book_id, estimated, now)
        ):
            return True, direction, "kappa"

        return False, None, None

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
        fees = getattr(account, "fees", None)
        if fees is None:
            return None
        rate = getattr(fees, "taker_fee_rate", None)
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
        qty = round(min(self.min_order_size, abs(pos.qty)), vol_dp)
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
        if pos.qty > 0 and account.base_balance is not None:
            free = account.base_balance.free
            if free < self._min_qty:
                self._clear_position(pos)
            else:
                pos.qty = round(min(pos.qty, free, self.min_order_size), vol_dp)


if __name__ == "__main__":
    launch(RebateScalperAgent)
