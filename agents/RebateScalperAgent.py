"""
RebateScalperAgent — taker rebate round-trip engine for subnet 79.

Tick flow
---------
1. Close all open legs (TP / SL / max_hold).
2. Scan 32 books per tick (0 RT in 10m first, then rotating window).
3. Per book in RT_WINDOW_S: force if 0 RTs, kappa if rebate and count < RT_MAX,
   skip if at cap. Volume cap on kappa path only.

Open reasons (telemetry action): open_kappa | open_force
"""

import math
from dataclasses import dataclass, field
from typing import Any

import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.telemetry import MinerTelemetry
from taos.im.protocol import FinanceAgentResponse, MarketSimulationStateUpdate
from taos.im.protocol.events import OrderPlacementEvent, SimulationStartEvent, TradeEvent
from taos.im.protocol.models import OrderCurrency, OrderDirection, STP

_NS = 1_000_000_000

MIN_ORDER_SIZE = 0.25

# Hold / exit
MIN_HOLD_S = 1.5
MAX_HOLD_S = 5.0
MIN_GROSS_TP_BPS = 2.5
MAX_GROSS_SL_BPS = 6.0

MIN_REOPEN_GAP_S = 4.0

# RT window for opens (validator activity sampling = 10m)
RT_WINDOW_S = 570.0                # ~10 min
RT_MAX = 3                         # max RTs per book in window

# Kappa-3 (3h history for score projection)
KAPPA_TAU = 0.0
KAPPA_MIN_OBS = 3
KAPPA_MIN_LOOKBACK_S = 5400.0      # 90 min
KAPPA_RT_HISTORY_S = 10_800.0      # 3h RT history kept for kappa
KAPPA_PROJ_IMPROVE = 0.003
KAPPA_PROJ_TOLERANCE = 0.008

BOOKS_PER_STEP = 32

# Volume cap (kappa path only)
CAPITAL_TURNOVER_CAP = 10.0
VOLUME_SAFETY = 0.8
VOLUME_ASSESSMENT_NS = 86_400_000_000_000


@dataclass
class _Position:
    qty: float = 0.0
    avg: float = 0.0
    entry_ts: int = 0
    entry_fee: float = 0.0


@dataclass
class _BookState:
    last_rt_ns: int = 0
    pending_buy: bool = False
    rt_events: list[tuple[int, float]] = field(default_factory=list)
    kappa3: float | None = None
    vol_log: list[tuple[int, float]] = field(default_factory=list)


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
        self._exit_reason: dict[tuple[str, int], str] = {}
        self._open_step: dict[str, int] = {}
        self._step_ts_ns: int = 0
        self._active_validator: str | None = None

        self.telemetry = MinerTelemetry.from_agent(self, agent_class="RebateScalperAgent")
        bt.logging.info(
            f"[RebateScalper uid={self.uid}] lot={MIN_ORDER_SIZE} "
            f"hold={MIN_HOLD_S}-{max_hold_s:.1f}s books/step={BOOKS_PER_STEP} "
            f"rt_window={RT_WINDOW_S / 60:.0f}m force=0 max={RT_MAX}"
        )

    def onStart(self, event: SimulationStartEvent) -> None:
        self.positions.clear()
        self.books_state.clear()
        self._sim_id.clear()
        self._exit_reason.clear()
        self._open_step.clear()

    def update(self, state: MarketSimulationStateUpdate) -> None:
        self._step_ts_ns = int(state.timestamp)
        self._active_validator = state.dendrite.hotkey
        super().update(state)

    def onOrderRejected(self, event: OrderPlacementEvent) -> None:
        if event.bookId is None:
            return
        if self._active_validator:
            self._bstate(self._active_validator, event.bookId).pending_buy = False
            return
        for v in self.books_state:
            self._bstate(v, event.bookId).pending_buy = False

    def onTrade(self, event: TradeEvent, validator: str | None = None) -> None:
        if event.bookId is None:
            return
        validator = validator or self._active_validator
        if validator is None:
            return

        is_taker = self.uid == event.takerAgentId
        if is_taker:
            direction = OrderDirection.BUY if event.side == OrderDirection.BUY else OrderDirection.SELL
        elif self.uid == event.makerAgentId:
            direction = OrderDirection.SELL if event.side == OrderDirection.BUY else OrderDirection.BUY
        else:
            return

        ts_ns = int(event.timestamp) if event.timestamp else self._step_ts_ns
        self._record_trade_volume(validator, event.bookId, event.quantity, event.price, ts_ns)
        fee = event.takerFee if is_taker else event.makerFee
        self._apply_fill(validator, event.bookId, direction, event.quantity, event.price, fee, ts_ns)

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        validator = state.dendrite.hotkey
        cfg = self.simulation_config
        self._sync_order_size(cfg.volumeDecimals)
        self._ensure_validator(validator, cfg.simulation_id)

        vol_dp = cfg.volumeDecimals
        volume_cap = CAPITAL_TURNOVER_CAP * cfg.miner_wealth * VOLUME_SAFETY
        now = state.timestamp
        self.telemetry.begin_step(state)
        instr_before = len(response.instructions)

        for book_id in self._books_to_close(state, validator):
            book = state.books.get(book_id)
            if book is None:
                continue
            try:
                self._handle_close(response, validator, book_id, book, vol_dp, now)
            except Exception as ex:
                bt.logging.warning(f"[RebateScalper uid={self.uid}] close {book_id}: {ex}")

        open_step = self._open_step.get(validator, 0)
        for book_id in self._open_book_batch(validator, open_step, now):
            book = state.books.get(book_id)
            account = self.accounts.get(book_id) if book else None
            if book is None or account is None:
                continue
            try:
                do_open, reason, meta = self._open_decision(
                    validator, book_id, book, account, vol_dp, volume_cap, now,
                )
                if do_open and self._taker_buy(response, validator, book_id, account, book):
                    self._record_open(
                        validator, book_id, book, account, now, reason, volume_cap, meta,
                    )
            except Exception as ex:
                bt.logging.warning(f"[RebateScalper uid={self.uid}] open {book_id}: {ex}")

        self._open_step[validator] = open_step + 1
        self.telemetry.end_step(state, instructions=len(response.instructions) - instr_before)
        return response

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

    def _ensure_validator(self, validator: str, simulation_id: str) -> None:
        if self._sim_id.get(validator) == simulation_id:
            return
        self._book_positions(validator).clear()
        self.books_state.pop(validator, None)
        self._exit_reason = {k: v for k, v in self._exit_reason.items() if k[0] != validator}
        self._open_step[validator] = 0
        self._sim_id[validator] = simulation_id

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
        if vol > 0:
            self._bstate(validator, book_id).vol_log.append((ts_ns, vol))

    def _rolled_quote_volume(self, validator: str, book_id: int, now_ns: int) -> float:
        st = self._bstate(validator, book_id)
        cutoff = now_ns - self.volume_assessment_ns
        st.vol_log = [(t, v) for t, v in st.vol_log if t >= cutoff]
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
        entry_ts = pos.entry_ts

        if prev == 0 or (prev > 0) == (signed > 0):
            total = abs(prev) + qty
            pos.avg = (pos.avg * abs(prev) + price * qty) / total if total > 0 else price
            pos.qty = prev + signed
            if prev == 0:
                pos.entry_ts = ts
                pos.entry_fee = trade_fee
                self._bstate(validator, book_id).pending_buy = False
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
            st.last_rt_ns = ts
            self._record_rt_close(validator, book_id, ts, net_pnl)
            self.telemetry.record_round_trip(
                book_id=book_id,
                ts_close_ns=ts,
                side="long" if prev > 0 else "short",
                qty=closed_qty,
                entry_avg=entry_avg,
                exit_avg=price,
                realized_pnl=net_pnl,
                hold_s=(ts - entry_ts) / _NS if entry_ts else None,
                reason=self._exit_reason.pop((validator, book_id), "fill"),
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

    def _kappa_open_ok(self, st: _BookState, validator: str, book_id: int, estimated: float, now: int) -> bool:
        if estimated <= 0.0:
            return False

        close_ts = now + self.min_hold_ns
        projected = self._kappa3_raw(
            self._book_pnl_series(validator, book_id, now, extra=(close_ts, estimated)),
        )
        rt_n = self._rt_count(st, now, self.kappa_rt_history_ns)
        if projected is None:
            return rt_n < KAPPA_MIN_OBS

        current = st.kappa3
        if current is None:
            return projected >= 0.0
        if current < 0.05:
            return projected >= current + KAPPA_PROJ_IMPROVE
        return projected >= current - KAPPA_PROJ_TOLERANCE

    # ------------------------------------------------------------------ open gates
    def _reopen_ok(self, st: _BookState, now: int) -> bool:
        return st.last_rt_ns == 0 or (now - st.last_rt_ns) >= self.min_reopen_gap_ns

    def _rt_count(self, st: _BookState, now: int, window_ns: int | None = None) -> int:
        cutoff = now - (self.rt_window_ns if window_ns is None else window_ns)
        return sum(1 for ts, _ in st.rt_events if ts >= cutoff)

    def _books_needing_force(self, validator: str, now: int) -> list[int]:
        """Flat books with 0 RTs in the window — scanned first each tick."""
        stale: list[tuple[int, int]] = []
        for book_id in sorted(self.accounts.keys()):
            st = self._bstate(validator, book_id)
            pos = self._book_positions(validator).get(book_id, _Position())
            if pos.qty >= self._min_qty or st.pending_buy or self._rt_count(st, now) > 0:
                continue
            stale.append((now if st.last_rt_ns == 0 else now - st.last_rt_ns, book_id))
        stale.sort(reverse=True)
        return [book_id for _, book_id in stale]

    def _open_book_batch(self, validator: str, step: int, now: int) -> list[int]:
        book_ids = sorted(self.accounts.keys())
        if not book_ids:
            return []

        batch_count = (len(book_ids) + BOOKS_PER_STEP - 1) // BOOKS_PER_STEP
        start = (step % batch_count) * BOOKS_PER_STEP
        rotated = book_ids[start:start + BOOKS_PER_STEP]

        merged: list[int] = []
        seen: set[int] = set()
        for book_id in self._books_needing_force(validator, now) + rotated:
            if book_id in seen:
                continue
            seen.add(book_id)
            merged.append(book_id)
            if len(merged) >= BOOKS_PER_STEP:
                break
        return merged

    def _open_decision(
        self,
        validator: str,
        book_id: int,
        book,
        account,
        vol_dp: int,
        volume_cap: float,
        now: int,
    ) -> tuple[bool, str, dict[str, Any]]:
        pos = self._book_positions(validator).get(book_id, _Position())
        self._reconcile_position(account, pos, vol_dp)
        if pos.qty >= self._min_qty:
            return False, "", {}

        st = self._sync_book_rt_state(validator, book_id, now)
        if st.pending_buy or not self._reopen_ok(st, now):
            return False, "", {}

        rate = self._taker_fee_rate(account)
        if self._mid(book) is None or rate is None:
            return False, "", {}

        # rt_n = completed RT closes in the last RT_WINDOW_S
        rt_n = self._rt_count(st, now)
        estimated = self._estimate_rt_pnl(rate, book, self.min_order_size)
        meta: dict[str, Any] = {
            "taker_bps": rate * 1e4,
            "kappa3": st.kappa3,
            "rt_window_n": rt_n,
            "estimated_pnl": estimated,
        }

        if rt_n >= RT_MAX:
            return False, "", {}

        if rt_n == 0:
            return True, "force", meta

        if (
            rate <= 0.0
            and self._rolled_quote_volume(validator, book_id, now) < volume_cap
            and self._kappa_open_ok(st, validator, book_id, estimated, now)
        ):
            return True, "kappa", meta

        return False, "", {}

    def _record_open(
        self,
        validator: str,
        book_id: int,
        book,
        account,
        now: int,
        reason: str,
        volume_cap: float,
        meta: dict[str, Any],
    ) -> None:
        mid = self._mid(book)
        bid = book.bids[0].price if book.bids else None
        ask = book.asks[0].price if book.asks else None
        traded = self._rolled_quote_volume(validator, book_id, now)
        self.telemetry.snapshot(
            book_id=book_id,
            mid=mid,
            bid=bid,
            ask=ask,
            pos_qty=self.min_order_size,
            pos_avg=ask or mid,
            base_bal=account.base_balance.total if account.base_balance else None,
            quote_bal=account.quote_balance.total if account.quote_balance else None,
            traded_volume=traded,
            volume_cap=volume_cap,
            volume_remaining=max(0.0, volume_cap - traded),
            signals={
                "taker_bps": meta.get("taker_bps"),
                "kappa3": meta.get("kappa3"),
                "est_pnl": meta.get("estimated_pnl"),
                "rt_window_n": meta.get("rt_window_n"),
            },
            action=f"open_{reason}",
        )

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

        if pos.qty < -self._min_qty:
            self._exit_reason[(validator, book_id)] = "short_flatten"
            self._taker_sell(response, account, book_id, pos, vol_dp)
            return
        if pos.qty < self._min_qty or pos.avg <= 0:
            return

        hold_ns = (now - pos.entry_ts) if pos.entry_ts else 0
        if hold_ns < self.min_hold_ns:
            return

        bid = book.bids[0].price if book.bids else None
        exit_px = bid if bid and bid > 0 else mid
        gross_bps = (exit_px - pos.avg) / pos.avg * 1e4

        if gross_bps >= MIN_GROSS_TP_BPS:
            self._exit_reason[(validator, book_id)] = "tp"
        elif gross_bps <= -MAX_GROSS_SL_BPS:
            self._exit_reason[(validator, book_id)] = "sl"
        elif hold_ns >= self.max_hold_ns:
            self._exit_reason[(validator, book_id)] = "time"
        else:
            return

        self._taker_sell(response, account, book_id, pos, vol_dp)

    def _taker_buy(self, response, validator: str, book_id: int, account, book) -> bool:
        qty = self.min_order_size
        ask_px = book.asks[0].price if book.asks else 0.0
        if ask_px <= 0 or account.quote_balance.free < qty * ask_px:
            return False
        response.market_order(
            book_id=book_id,
            direction=OrderDirection.BUY,
            quantity=qty,
            currency=OrderCurrency.BASE,
            stp=STP.CANCEL_OLDEST,
        )
        self._bstate(validator, book_id).pending_buy = True
        return True

    def _taker_sell(self, response, account, book_id: int, pos: _Position, vol_dp: int) -> None:
        if pos.qty <= 0:
            return
        free = account.base_balance.free if account.base_balance else 0.0
        qty = round(min(self.min_order_size, pos.qty, free), vol_dp)
        if qty < self._min_qty:
            return
        response.market_order(
            book_id=book_id,
            direction=OrderDirection.SELL,
            quantity=qty,
            currency=OrderCurrency.BASE,
            stp=STP.CANCEL_OLDEST,
        )

    def _reconcile_position(self, account, pos: _Position, vol_dp: int) -> None:
        if account.base_balance is None:
            return
        free = account.base_balance.free
        if pos.qty > 0:
            if free < self._min_qty:
                self._clear_position(pos)
            else:
                pos.qty = round(min(pos.qty, free, self.min_order_size), vol_dp)


if __name__ == "__main__":
    launch(RebateScalperAgent)
