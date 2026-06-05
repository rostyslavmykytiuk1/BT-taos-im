"""
KalmanMomentumAgent — Subnet 79 (MVTRX / taos)

Momentum on Kalman slope (3s candles by default):

  • Kalman tracks hidden level + slope from microprice each candle.
  • Long when slope_bps >= gate and fair >= level; short when opposite.
  • Exit when slope sign flips (long while slope < 0, short while slope > 0).
  • All strategy orders are market (taker fee must be <= 0).
  • Activity ping after 9 min without a closed round-trip on a book.
"""

import traceback
from dataclasses import dataclass, field

import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.telemetry import MinerTelemetry
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse
from taos.im.protocol.events import TradeEvent, SimulationStartEvent
from taos.im.protocol.models import OrderDirection, OrderCurrency, STP

SLOPE_GATE_BPS = 10.0
KALMAN_PROCESS_VAR = 1e-5
KALMAN_MEAS_VAR = 1e-3
KALMAN_WARMUP_STEPS = 8
CANDLE_S = 3.0
CAPITAL_TURNOVER_CAP = 10.0
VOLUME_SAFETY = 0.5
VOLUME_WINDOW_S = 86_400.0
ACTIVITY_RT_IDLE_S = 540.0
ACTIVITY_PING_HOLD_S = 2.0


class _KalmanFair:
    """2-state Kalman: hidden level + velocity (price units per candle step)."""

    __slots__ = ("dt", "q", "r", "x0", "x1", "p00", "p01", "p11", "_ready")

    def __init__(self, dt: float = 1.0, process_var: float = 1e-5, meas_var: float = 1e-3):
        self.dt = max(dt, 1e-9)
        self.q = process_var
        self.r = max(meas_var, 1e-12)
        self.x0 = self.x1 = 0.0
        self.p00 = self.p11 = 1.0
        self.p01 = 0.0
        self._ready = False

    def update(self, obs: float) -> tuple[float, float, float]:
        if not self._ready:
            self.x0 = obs
            self._ready = True
            return obs, 0.0, 0.0

        dt, q = self.dt, self.q
        x0p = self.x0 + self.x1 * dt
        x1p = self.x1
        p00p = self.p00 + 2.0 * dt * self.p01 + dt * dt * self.p11 + 0.25 * dt ** 4 * q
        p01p = self.p01 + dt * self.p11 + 0.5 * dt ** 3 * q
        p11p = self.p11 + dt * dt * q

        innov = obs - x0p
        s = p00p + self.r
        if s <= 0:
            return x0p, x1p, 0.0
        k0, k1 = p00p / s, p01p / s
        self.x0 = x0p + k0 * innov
        self.x1 = x1p + k1 * innov
        self.p00 = (1.0 - k0) * p00p
        self.p01 = (1.0 - k0) * p01p
        self.p11 = p11p - k1 * p01p
        residual_bps = innov / self.x0 * 1e4 if self.x0 > 0 else 0.0
        return self.x0, self.x1, residual_bps


@dataclass
class Position:
    qty: float = 0.0
    avg: float = 0.0
    opened_ns: int = 0
    is_ping: bool = False


@dataclass
class BookState:
    kalman: _KalmanFair = field(default_factory=_KalmanFair)
    kalman_steps: int = 0
    ping_followup_at: int = 0
    ping_followup_is_buy: bool = False
    vol_log: list = field(default_factory=list)
    first_seen_ns: int = 0
    last_rt_ns: int = 0


class KalmanMomentumAgent(FinanceSimulationAgent):

    def initialize(self) -> None:
        bt.logging.set_info()
        self.quote_notional = self._param("quote_notional", 1800.0)
        self.min_order_size = self._param("min_order_size", 0.25)
        self.candle_s = self._param("candle_s", CANDLE_S)
        self.candle_ns = int(self.candle_s * 1e9) if self.candle_s >= 1.0 else 0
        self.slope_gate_bps = self._param("slope_gate_bps", SLOPE_GATE_BPS)
        self.kalman_process_var = self._param("kalman_process_var", KALMAN_PROCESS_VAR)
        self.kalman_meas_var = self._param("kalman_meas_var", KALMAN_MEAS_VAR)
        self.kalman_warmup = int(self._param("kalman_warmup", KALMAN_WARMUP_STEPS))
        self.max_taker_fee = self._param("max_taker_fee", 0.0)
        self.turnover_cap = self._param("capital_turnover_cap", CAPITAL_TURNOVER_CAP)
        self.volume_safety = self._param("volume_safety", VOLUME_SAFETY)
        self.volume_window_ns = int(VOLUME_WINDOW_S * 1e9)
        self.activity_rt_idle_ns = int(ACTIVITY_RT_IDLE_S * 1e9)

        jitter = ((self.uid * 2654435761) % 1000) / 1000.0
        self.slope_gate_bps *= 0.92 + 0.16 * jitter

        self.positions: dict[str, dict[int, Position]] = {}
        self.books: dict[str, dict[int, BookState]] = {}
        self._sim_id: dict[str, str] = {}
        self._exit_reason: dict[tuple[str, int], str] = {}
        self._candle_bucket: dict[str, int] = {}
        self._step_ts_ns: int = 0
        self.telemetry = MinerTelemetry.from_agent(self, agent_class="KalmanMomentumAgent")
        bt.logging.info(
            f"[KalmanMomentum uid={self.uid}] notional={self.quote_notional} "
            f"candle={self.candle_s:.0f}s slope_gate={self.slope_gate_bps:.1f}bps "
            f"exit=slope_flip ping_after={ACTIVITY_RT_IDLE_S / 60:.0f}min"
        )

    def _param(self, name: str, default: float) -> float:
        try:
            return float(getattr(self.config, name, default))
        except (TypeError, ValueError):
            return default

    def onStart(self, event: SimulationStartEvent) -> None:
        self.positions.clear()
        self.books.clear()
        self._sim_id.clear()
        self._exit_reason.clear()
        self._candle_bucket.clear()

    def update(self, state: MarketSimulationStateUpdate) -> None:
        self._step_ts_ns = int(state.timestamp)
        super().update(state)

    def _position(self, validator: str, book_id: int) -> Position:
        return self.positions.setdefault(validator, {}).setdefault(book_id, Position())

    def _state(self, validator: str, book_id: int) -> BookState:
        st = self.books.setdefault(validator, {}).setdefault(book_id, BookState())
        if st.kalman.dt != self.candle_s:
            st.kalman = _KalmanFair(self.candle_s, self.kalman_process_var, self.kalman_meas_var)
            st.kalman_steps = 0
        return st

    def onTrade(self, event: TradeEvent, validator: str = None) -> None:
        if event.bookId is None:
            return
        if self.uid == event.takerAgentId:
            direction = event.side
        elif self.uid == event.makerAgentId:
            direction = OrderDirection.SELL if event.side == OrderDirection.BUY else OrderDirection.BUY
        else:
            return
        ts = self._step_ts_ns or event.timestamp
        self._state(validator, event.bookId).vol_log.append((ts, event.quantity * event.price))
        self._apply_fill(validator, event.bookId, direction, event.quantity, event.price, ts)

    def _apply_fill(self, validator, book_id, direction, qty, price, ts) -> None:
        pos = self._position(validator, book_id)
        signed = qty if direction == OrderDirection.BUY else -qty
        prev, entry_avg, entry_ts = pos.qty, pos.avg, pos.opened_ns

        if prev == 0 or (prev > 0) == (signed > 0):
            total = abs(prev) + qty
            pos.avg = (pos.avg * abs(prev) + price * qty) / total if total > 0 else price
            pos.qty = prev + signed
            if prev == 0:
                pos.opened_ns = ts
            return

        closed_qty = min(qty, abs(prev))
        if closed_qty > 0 and entry_avg > 0:
            self._state(validator, book_id).last_rt_ns = ts
        if abs(prev + signed) < 1e-12 and abs(prev) >= self.min_order_size / 2 and entry_avg > 0:
            side = "long" if prev > 0 else "short"
            rpnl = (price - entry_avg) * closed_qty if prev > 0 else (entry_avg - price) * closed_qty
            self.telemetry.record_round_trip(
                book_id=book_id, ts_close_ns=ts, side=side, qty=closed_qty,
                entry_avg=entry_avg, exit_avg=price, realized_pnl=rpnl,
                hold_s=(ts - entry_ts) / 1e9 if entry_ts else None,
                reason=self._exit_reason.pop((validator, book_id), "fill"),
            )
            self._state(validator, book_id).last_rt_ns = ts
        pos.qty = prev + signed
        if abs(pos.qty) < 1e-12:
            self.positions[validator][book_id] = Position()
        elif (prev > 0) != (pos.qty > 0):
            pos.avg, pos.opened_ns, pos.is_ping = price, ts, False

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

    def _rolled_volume(self, st: BookState, now: int) -> float:
        st.vol_log = [(t, v) for t, v in st.vol_log if t >= now - self.volume_window_ns]
        return sum(v for _, v in st.vol_log)

    def _rt_idle(self, st: BookState, now: int) -> bool:
        if st.first_seen_ns <= 0 or now - st.first_seen_ns < self.activity_rt_idle_ns:
            return False
        ref = st.last_rt_ns if st.last_rt_ns > 0 else st.first_seen_ns
        return now - ref >= self.activity_rt_idle_ns

    def _on_candle_close(self, validator: str, ts_ns: int) -> bool:
        if self.candle_ns <= 0:
            return True
        bucket = ts_ns // self.candle_ns
        if self._candle_bucket.get(validator) == bucket:
            return False
        self._candle_bucket[validator] = bucket
        return True

    def _taker_ok(self, account) -> bool:
        fees = getattr(account, "fees", None)
        rate = getattr(fees, "taker_fee_rate", None) if fees else None
        try:
            return rate is not None and float(rate) <= self.max_taker_fee
        except (TypeError, ValueError):
            return False

    def _buy(self, response, book_id, qty) -> None:
        response.market_order(book_id=book_id, direction=OrderDirection.BUY, quantity=qty,
                              currency=OrderCurrency.BASE, stp=STP.CANCEL_OLDEST)

    def _sell(self, response, book_id, qty) -> None:
        response.market_order(book_id=book_id, direction=OrderDirection.SELL, quantity=qty,
                              currency=OrderCurrency.BASE, stp=STP.CANCEL_OLDEST)

    def _close_all(self, response, validator, book_id, pos, account, vol_dp, reason) -> str:
        if pos.qty > 0:
            qty = round(min(pos.qty, account.base_balance.free), vol_dp)
            if qty >= self.min_order_size:
                self._exit_reason[(validator, book_id)] = reason
                self._sell(response, book_id, qty)
        else:
            qty = round(-pos.qty, vol_dp)
            if qty >= self.min_order_size:
                self._exit_reason[(validator, book_id)] = reason
                self._buy(response, book_id, qty)
        return reason

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        validator = state.dendrite.hotkey
        cfg = self.simulation_config

        if self._sim_id.get(validator) != cfg.simulation_id:
            self.positions.pop(validator, None)
            self.books.pop(validator, None)
            self._exit_reason = {k: v for k, v in self._exit_reason.items() if k[0] != validator}
            self._candle_bucket.pop(validator, None)
            self._sim_id[validator] = cfg.simulation_id

        if not self._on_candle_close(validator, int(state.timestamp)):
            return response

        cap = self.turnover_cap * cfg.miner_wealth * self.volume_safety
        ping_book = self._pick_ping_book(validator, state, state.timestamp, cfg.volumeDecimals)
        self.telemetry.begin_step(state)
        n_instr = len(response.instructions)
        for book_id, book in state.books.items():
            try:
                self._handle_book(
                    response, validator, book_id, book,
                    cfg.volumeDecimals, cap, state.timestamp, ping_book,
                )
            except Exception as ex:
                bt.logging.warning(
                    f"[KalmanMomentum uid={self.uid}] book {book_id}: {ex}\n{traceback.format_exc()}"
                )
        self.telemetry.end_step(state, instructions=len(response.instructions) - n_instr)
        return response

    def _pick_ping_book(self, validator: str, state, now: int, vol_dp: int) -> int | None:
        half = self.min_order_size / 2
        for book_id, pos in sorted(self.positions.get(validator, {}).items()):
            if pos.is_ping and pos.qty > half:
                return book_id
        for book_id in sorted(state.books.keys()):
            if self._state(validator, book_id).ping_followup_at > 0:
                return book_id
        for book_id in sorted(state.books.keys()):
            st = self._state(validator, book_id)
            if not self._rt_idle(st, now):
                continue
            account = self.accounts.get(book_id)
            book = state.books.get(book_id)
            if account is None or book is None or not book.bids or not book.asks:
                continue
            if not self._taker_ok(account):
                continue
            mid = self._mid(book)
            qty = round(self.min_order_size, vol_dp)
            if mid is None or mid <= 0 or qty < self.min_order_size:
                continue
            if account.quote_balance.free < qty * (book.asks[0].price or mid):
                continue
            return book_id
        return None

    def _handle_book(self, response, validator, book_id, book, vol_dp, cap, now,
                     ping_book: int | None) -> None:
        mid = self._mid(book)
        fair = self._microprice(book) or mid
        if mid is None or mid <= 0 or fair is None:
            return
        account = self.accounts.get(book_id)
        if account is None:
            return

        st = self._state(validator, book_id)
        if st.first_seen_ns <= 0:
            st.first_seen_ns = now
        pos = self._position(validator, book_id)
        if abs(pos.qty) < self.min_order_size / 2:
            pos.is_ping = False

        level, slope, residual_bps = st.kalman.update(fair)
        st.kalman_steps += 1
        slope_bps = slope / level * 1e4 if level > 0 else 0.0
        bid = book.bids[0].price if book.bids else None
        ask = book.asks[0].price if book.asks else None
        half = self.min_order_size / 2

        if ping_book == book_id:
            action = self._activity_ping(response, validator, book_id, book, pos, st, account, mid, vol_dp, now)
            if action is not None:
                self._snap(st, book_id, mid, bid, ask, pos, account, level, slope_bps, residual_bps, action, cap, now)
                return

        if abs(pos.qty) >= half and not pos.is_ping:
            action = self._manage_slope(response, validator, book_id, pos, account, slope_bps, vol_dp)
            self._snap(st, book_id, mid, bid, ask, pos, account, level, slope_bps, residual_bps, action, cap, now)
            return

        if abs(pos.qty) < half:
            action = self._maybe_open(
                response, book_id, pos, st, account, mid, fair, level, slope_bps, cap, now, vol_dp,
            )
            self._snap(st, book_id, mid, bid, ask, pos, account, level, slope_bps, residual_bps, action, cap, now)

    def _manage_slope(self, response, validator, book_id, pos, account, slope_bps, vol_dp) -> str:
        if pos.qty > 0 and slope_bps < 0:
            return self._close_all(response, validator, book_id, pos, account, vol_dp, "close_slope")
        if pos.qty < 0 and slope_bps > 0:
            return self._close_all(response, validator, book_id, pos, account, vol_dp, "close_slope")
        return "manage"

    def _maybe_open(self, response, book_id, pos, st, account, mid, fair, level,
                    slope_bps, cap, now, vol_dp) -> str:
        if self._rolled_volume(st, now) >= cap:
            return "cap"
        if st.kalman_steps < self.kalman_warmup:
            return "warmup"
        if not self._taker_ok(account):
            return "flat"

        gate = self.slope_gate_bps
        long_ok = slope_bps >= gate and fair >= level
        short_ok = slope_bps <= -gate and fair <= level
        if not (long_ok or short_ok):
            return "flat"

        qty = round(self.quote_notional / mid, vol_dp)
        if qty < self.min_order_size:
            return "flat"
        if long_ok:
            if account.quote_balance.free < qty * mid:
                return "flat"
            self._buy(response, book_id, qty)
            return "open_long"
        if account.base_balance.free < qty:
            return "flat"
        self._sell(response, book_id, qty)
        return "open_short"

    def _activity_ping(self, response, validator, book_id, book, pos, st, account,
                       mid, vol_dp, now) -> str | None:
        half = self.min_order_size / 2
        hold_ns = int(ACTIVITY_PING_HOLD_S * 1e9)

        if pos.is_ping and pos.qty > half:
            if (now - pos.opened_ns) / 1e9 >= ACTIVITY_PING_HOLD_S:
                return self._close_all(response, validator, book_id, pos, account, vol_dp, "activity_ping")
            return "ping_hold"

        if st.ping_followup_at > 0:
            if now >= st.ping_followup_at and self._taker_ok(account):
                qty = round(self.min_order_size, vol_dp)
                if qty >= self.min_order_size:
                    self._exit_reason[(validator, book_id)] = "activity_ping"
                    if st.ping_followup_is_buy:
                        if account.quote_balance.free >= qty * (book.asks[0].price or mid):
                            self._buy(response, book_id, qty)
                    elif account.base_balance.free >= qty:
                        self._sell(response, book_id, qty)
                    st.ping_followup_at = 0
                    return "activity_ping"
            return "ping_followup_wait"

        if not self._rt_idle(st, now) or not self._taker_ok(account):
            return None
        qty = round(self.min_order_size, vol_dp)
        if qty < self.min_order_size or account.quote_balance.free < qty * (book.asks[0].price or mid):
            return None

        if pos.qty > half:
            if account.base_balance.free < qty:
                return None
            self._sell(response, book_id, qty)
            st.ping_followup_at = now + hold_ns
            st.ping_followup_is_buy = True
            return "ping_open"

        self._buy(response, book_id, qty)
        if abs(pos.qty) < half:
            pos.is_ping = True
            pos.opened_ns = now
        else:
            st.ping_followup_at = now + hold_ns
            st.ping_followup_is_buy = False
        return "ping_open"

    def _snap(self, st, book_id, mid, bid, ask, pos, account, level, slope_bps, residual_bps,
              action, cap, now) -> None:
        traded = self._rolled_volume(st, now)
        self.telemetry.snapshot(
            book_id=book_id, mid=mid, bid=bid, ask=ask,
            pos_qty=pos.qty, pos_avg=pos.avg,
            base_bal=account.base_balance.total if account.base_balance else None,
            quote_bal=account.quote_balance.total if account.quote_balance else None,
            traded_volume=traded, volume_cap=cap, volume_remaining=max(0.0, cap - traded),
            signals={"trend_bps": slope_bps, "flow": residual_bps, "imb": 0.0, "level": level},
            action=action,
        )


if __name__ == "__main__":
    launch(KalmanMomentumAgent)
