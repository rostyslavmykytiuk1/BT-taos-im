"""
MeanReversionAgent — Subnet 79 (MVTRX / taos)

Per order book:

  1. Mean reversion — buy when price is below its rolling average, sell when above,
     then close at take-profit, stop-loss, or max hold. Maker entries when possible.

  2. Activity ping — if a book has had no closed round-trip for 9 minutes, run one
     tiny market buy/sell when taker fees allow (keeps activity_factor alive).

Run (local proxy test):
  python MeanReversionAgent.py --port 8902 --agent_id 0 --params quote_notional=1800
"""

import math
import traceback
from collections import deque
from dataclasses import dataclass, field

import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.telemetry import MinerTelemetry
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse
from taos.im.protocol.events import TradeEvent, SimulationStartEvent
from taos.im.protocol.models import OrderDirection, OrderCurrency, STP, TimeInForce

# --- mean reversion (open long / open short vs rolling average) ---------------
AVERAGE_WINDOW_S = 300.0
TREND_WINDOW_S = 900.0
TREND_GATE_BPS = 25.0       # skip open long in slow downtrend / open short in uptrend
BAND_K = 1.8
MIN_BAND_BPS = 18.0
MAX_BAND_BPS = 120.0
CLOSE_TP_BPS = 14.0
CLOSE_STOP_BPS = 22.0
CLOSE_MAX_HOLD_S = 180.0
PAUSE_AFTER_STOP_S = 90.0
STILL_FALLING_BPS = 6.0     # skip new entries if last tick dropped this much
IMBALANCE_DEPTH = 5
IMBALANCE_GATE = 0.30

MIN_SAMPLES = 8
ENTRY_EXPIRY_S = 8.0
CAPITAL_TURNOVER_CAP = 10.0
VOLUME_SAFETY = 0.5
VOLUME_WINDOW_S = 86_400.0
MID_HISTORY_MAXLEN = 32

# --- activity ping ------------------------------------------------------------
ACTIVITY_RT_IDLE_S = 540.0
ACTIVITY_PING_HOLD_S = 2.0


@dataclass
class Position:
    """Our signed position on one book, rebuilt from fills (not from the exchange feed)."""
    qty: float = 0.0       # >0 long, <0 short, in BASE
    avg: float = 0.0       # volume-weighted entry price for the open leg
    opened_ns: int = 0     # sim timestamp when the current leg opened
    is_ping: bool = False  # True for the tiny activity round-trip only


@dataclass
class BookState:
    """Rolling stats for one book; all timestamps are simulation nanoseconds."""
    prices: deque = field(default_factory=lambda: deque(maxlen=2000))  # trade prints for average
    mids: deque = field(default_factory=lambda: deque(maxlen=MID_HISTORY_MAXLEN))  # per-step mid
    trend_ema: float = 0.0       # slow EMA of mid (downtrend / uptrend filter)
    pause_until: int = 0         # no new entries until this time (after a stop loss)
    ping_followup_at: int = 0    # when to place the second leg of a ping (book not flat)
    ping_followup_is_buy: bool = False  # second ping leg direction
    vol_log: list = field(default_factory=list)  # (ts, quote notional) for volume cap
    first_seen_ns: int = 0       # when we first saw this book (for ping grace period)
    last_rt_ns: int = 0          # last fully closed round-trip (for activity ping)


class MeanReversionAgent(FinanceSimulationAgent):
    """Mean-revert vs a rolling trade-price average; optional activity ping per book."""

    def initialize(self) -> None:
        """Load params, per-UID band jitter, and telemetry."""
        bt.logging.set_info()
        self.quote_notional = self._param("quote_notional", 1800.0)
        self.min_order_size = self._param("min_order_size", 0.25)

        self.average_window_ns = int(AVERAGE_WINDOW_S * 1e9)
        self.volume_window_ns = int(VOLUME_WINDOW_S * 1e9)
        self.trend_alpha = 1.0 - math.exp(-1.0 / TREND_WINDOW_S)
        self.activity_rt_idle_ns = int(ACTIVITY_RT_IDLE_S * 1e9)

        # Slightly different stretch bands per UID so fleet miners don't quote the same level.
        jitter = ((self.uid * 2654435761) % 1000) / 1000.0
        self.band_k = BAND_K * (0.92 + 0.16 * jitter)

        self.positions: dict[str, dict[int, Position]] = {}
        self.books: dict[str, dict[int, BookState]] = {}
        self._sim_id: dict[str, str] = {}
        self._exit_reason: dict[tuple[str, int], str] = {}
        self._step_ts_ns: int = 0

        self.telemetry = MinerTelemetry.from_agent(self, agent_class="MeanReversionAgent")
        bt.logging.info(
            f"[MeanReversion uid={self.uid}] notional={self.quote_notional} "
            f"activity ping after {ACTIVITY_RT_IDLE_S / 60:.0f}min idle"
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
        bt.logging.info(f"[MeanReversion uid={self.uid}] simulation start: reset state")

    def update(self, state: MarketSimulationStateUpdate) -> None:
        self._step_ts_ns = int(state.timestamp)
        super().update(state)

    def _position(self, validator: str, book_id: int) -> Position:
        return self.positions.setdefault(validator, {}).setdefault(book_id, Position())

    def _state(self, validator: str, book_id: int) -> BookState:
        return self.books.setdefault(validator, {}).setdefault(book_id, BookState())

    def onTrade(self, event: TradeEvent, validator: str = None) -> None:
        """Update position, volume, and round-trip telemetry when our order fills."""
        if event.bookId is None:
            return
        if self.uid == event.takerAgentId:
            direction = event.side
        elif self.uid == event.makerAgentId:
            # Maker fill is the opposite of the aggressor's side.
            direction = OrderDirection.SELL if event.side == OrderDirection.BUY else OrderDirection.BUY
        else:
            return
        ts = self._step_ts_ns or event.timestamp
        self._state(validator, event.bookId).vol_log.append((ts, event.quantity * event.price))
        self._apply_fill(validator, event.bookId, direction, event.quantity, event.price, ts)

    def _apply_fill(self, validator, book_id, direction, qty, price, ts) -> None:
        """Track qty/avg; emit telemetry round_trip when a leg closes to ~zero."""
        pos = self._position(validator, book_id)
        signed = qty if direction == OrderDirection.BUY else -qty
        prev, entry_avg, entry_ts = pos.qty, pos.avg, pos.opened_ns

        # Same direction as before (or flat): add to the open leg.
        if prev == 0 or (prev > 0) == (signed > 0):
            total = abs(prev) + qty
            pos.avg = (pos.avg * abs(prev) + price * qty) / total if total > 0 else price
            pos.qty = prev + signed
            if prev == 0:
                pos.opened_ns = ts
            return

        # Opposite direction: this fill closes part or all of the open leg.
        closed_qty = min(qty, abs(prev))
        if closed_qty > 0 and entry_avg > 0:
            self._state(validator, book_id).last_rt_ns = ts
        # Full close → one completed round-trip for dashboard / activity_factor.
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
    def _fair_price(book) -> float | None:
        """Microprice: lean toward the side with more size at the touch."""
        if not book.bids or not book.asks:
            return None
        bid, ask = book.bids[0], book.asks[0]
        denom = bid.quantity + ask.quantity
        if denom <= 0:
            return 0.5 * (bid.price + ask.price)
        return (ask.price * bid.quantity + bid.price * ask.quantity) / denom

    def _imbalance(self, book) -> float:
        """Bid vs ask depth imbalance in [-1, 1]; positive = more bid size."""
        bq = sum(l.quantity for l in book.bids[:IMBALANCE_DEPTH])
        aq = sum(l.quantity for l in book.asks[:IMBALANCE_DEPTH])
        denom = bq + aq
        return (bq - aq) / denom if denom > 0 else 0.0

    def _ingest(self, st: BookState, book, mid: float, now: int) -> None:
        """Append trade prints and mid; trim old prints; update trend EMA."""
        for e in book.events or []:
            if getattr(e, "type", None) == "t" and e.price > 0:
                st.prices.append((now, float(e.price)))
        while st.prices and st.prices[0][0] < now - self.average_window_ns:
            st.prices.popleft()
        st.mids.append((now, mid))
        st.trend_ema = mid if st.trend_ema <= 0 else st.trend_ema + self.trend_alpha * (mid - st.trend_ema)

    def _average_and_band(self, st: BookState) -> tuple[float | None, float]:
        """Rolling mean of trade prices and entry band (bps) from recent dispersion."""
        if len(st.prices) < MIN_SAMPLES:
            return None, MIN_BAND_BPS
        ps = [p for _, p in st.prices]
        mean = sum(ps) / len(ps)
        if mean <= 0:
            return None, MIN_BAND_BPS
        dispersion_bps = math.sqrt(sum((p - mean) ** 2 for p in ps) / len(ps)) / mean * 1e4
        band = max(MIN_BAND_BPS, min(MAX_BAND_BPS, self.band_k * dispersion_bps))
        return mean, band

    def _last_step_bps(self, st: BookState) -> float:
        """One-step mid return in bps; negative means price just ticked down."""
        if len(st.mids) < 2:
            return 0.0
        prev, cur = st.mids[-2][1], st.mids[-1][1]
        return (cur - prev) / prev * 1e4 if prev > 0 else 0.0

    def _rolled_volume(self, st: BookState, now: int) -> float:
        """Quote notional traded on this book in the rolling volume window."""
        st.vol_log = [(t, v) for t, v in st.vol_log if t >= now - self.volume_window_ns]
        return sum(v for _, v in st.vol_log)

    def _rt_idle(self, st: BookState, now: int) -> bool:
        """True if no full round-trip close for ACTIVITY_RT_IDLE_S (validator samples ~10 min)."""
        if st.first_seen_ns <= 0 or now - st.first_seen_ns < self.activity_rt_idle_ns:
            return False
        ref = st.last_rt_ns if st.last_rt_ns > 0 else st.first_seen_ns
        return now - ref >= self.activity_rt_idle_ns

    def _clear_stale_flags(self, pos: Position) -> None:
        """If size is effectively zero, drop ping flag so the next open is not blocked."""
        if abs(pos.qty) < self.min_order_size / 2:
            pos.is_ping = False

    def _ping_book_for_step(self, validator: str, state, now: int, vol_dp: int) -> int | None:
        """Pick at most one book this step for ping close or ping open (lowest book id first)."""
        half = self.min_order_size / 2
        # Finish an open ping long or a two-leg ping before starting a new one.
        for book_id, pos in sorted(self.positions.get(validator, {}).items()):
            if pos.is_ping and pos.qty > half:
                return book_id
        for book_id in sorted(state.books.keys()):
            st = self._state(validator, book_id)
            if st.ping_followup_at > 0:
                return book_id

        for book_id in sorted(state.books.keys()):
            st = self._state(validator, book_id)
            if now < st.pause_until:
                continue
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

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        """Main loop: one pass per book; reset state when simulation id changes."""
        response = FinanceAgentResponse(agent_id=self.uid)
        validator = state.dendrite.hotkey
        cfg = self.simulation_config

        if self._sim_id.get(validator) != cfg.simulation_id:
            self.positions.pop(validator, None)
            self.books.pop(validator, None)
            self._exit_reason = {k: v for k, v in self._exit_reason.items() if k[0] != validator}
            self._sim_id[validator] = cfg.simulation_id

        # Stop new entries when rolling quote volume on a book hits this cap.
        cap = CAPITAL_TURNOVER_CAP * cfg.miner_wealth * VOLUME_SAFETY
        ping_book = self._ping_book_for_step(validator, state, state.timestamp, cfg.volumeDecimals)
        self.telemetry.begin_step(state)
        instr_before = len(response.instructions)
        for book_id, book in state.books.items():
            try:
                self._handle_book(response, validator, book_id, book,
                                  cfg.priceDecimals, cfg.volumeDecimals, cap, state.timestamp,
                                  ping_book=ping_book)
            except Exception as ex:
                bt.logging.warning(
                    f"[MeanReversion uid={self.uid}] book {book_id} error: {ex}\n"
                    f"{traceback.format_exc()}"
                )
        self.telemetry.end_step(state, instructions=len(response.instructions) - instr_before)
        return response

    def _handle_book(self, response, validator, book_id, book,
                     price_dp, vol_dp, cap, now, ping_book: int | None = None) -> None:
        """Compute signals for this book, then manage an open leg or try to open one."""
        mid = self._mid(book)
        fair = self._fair_price(book) or mid
        if mid is None or mid <= 0 or fair is None:
            return
        account = self.accounts.get(book_id)
        if account is None:
            return

        st = self._state(validator, book_id)
        if st.first_seen_ns <= 0:
            st.first_seen_ns = now
        self._ingest(st, book, mid, now)
        pos = self._position(validator, book_id)
        self._clear_stale_flags(pos)
        average, band_bps = self._average_and_band(st)
        bid = book.bids[0].price if book.bids else None
        ask = book.asks[0].price if book.asks else None
        imb = self._imbalance(book)
        falling = self._last_step_bps(st) <= -STILL_FALLING_BPS

        # Average above slow EMA → uptrend; below → downtrend (used to block bad-way entries).
        trend_bps = (average - st.trend_ema) / st.trend_ema * 1e4 if (average and st.trend_ema > 0) else 0.0
        # Fair vs average: negative = cheap (candidate buy), positive = rich (candidate sell).
        stretch_bps = (fair - average) / average * 1e4 if average else 0.0

        half = self.min_order_size / 2
        ping_action = None
        if ping_book == book_id:
            ping_action = self._try_activity_ping(
                response, validator, book_id, book, pos, st, account,
                mid, vol_dp, now, falling,
            )
        if ping_action is not None:
            action = ping_action
        elif abs(pos.qty) >= half:
            action = self._manage(response, validator, book_id, pos, st, account, fair, now, vol_dp)
        else:
            action = self._maybe_open(response, validator, book_id, book, pos, st, account,
                                      mid, fair, average, band_bps, stretch_bps, trend_bps,
                                      imb, falling, cap, now, price_dp, vol_dp)

        self._snap(st, book_id, mid, bid, ask, pos, account, trend_bps, stretch_bps, imb, action, cap, now)

    def _try_activity_ping(self, response, validator, book_id, book, pos, st, account,
                           mid, vol_dp, now, falling) -> str | None:
        """9 min idle → min-size two-leg round-trip; works even if the book already has size."""
        half = self.min_order_size / 2
        hold_ns = int(ACTIVITY_PING_HOLD_S * 1e9)

        if pos.is_ping and pos.qty > half:
            return self._manage_ping(response, validator, book_id, pos, account, now, vol_dp)

        if st.ping_followup_at > 0:
            if now >= st.ping_followup_at and not falling:
                qty = round(self.min_order_size, vol_dp)
                if qty >= self.min_order_size and self._taker_ok(account):
                    self._exit_reason[(validator, book_id)] = "activity_ping"
                    if st.ping_followup_is_buy:
                        if account.quote_balance.free >= qty * (book.asks[0].price or mid):
                            self._buy(response, book_id, qty)
                    elif account.base_balance.free >= qty:
                        self._sell(response, book_id, qty)
                    st.ping_followup_at = 0
                    return "activity_ping"
            return "ping_followup_wait"

        if falling or not self._rt_idle(st, now):
            return None
        qty = round(self.min_order_size, vol_dp)
        if qty < self.min_order_size or not self._taker_ok(account):
            return None
        if account.quote_balance.free < qty * (book.asks[0].price or mid):
            return None

        was_flat = abs(pos.qty) < half
        was_short = pos.qty < -half
        was_long = pos.qty > half

        if was_long:
            if account.base_balance.free < qty:
                return None
            self._sell(response, book_id, qty)
            st.ping_followup_at = now + hold_ns
            st.ping_followup_is_buy = True
            return "ping_open"

        self._buy(response, book_id, qty)
        if was_flat:
            pos.is_ping = True
            pos.opened_ns = now
        else:
            st.ping_followup_at = now + hold_ns
            st.ping_followup_is_buy = False
        return "ping_open"

    def _manage(self, response, validator, book_id, pos, st, account, fair, now, vol_dp) -> str:
        """Exit rules for normal positions (not activity ping)."""
        if pos.is_ping and pos.qty > self.min_order_size / 2:
            return self._manage_ping(response, validator, book_id, pos, account, now, vol_dp)

        pnl_bps = ((fair - pos.avg) if pos.qty > 0 else (pos.avg - fair)) / pos.avg * 1e4
        held_s = (now - pos.opened_ns) / 1e9 if pos.opened_ns else 0.0
        if pnl_bps >= CLOSE_TP_BPS:
            return self._close_all(response, validator, book_id, pos, account, vol_dp, "close_tp")
        if pnl_bps <= -CLOSE_STOP_BPS:
            st.pause_until = now + int(PAUSE_AFTER_STOP_S * 1e9)
            return self._close_all(response, validator, book_id, pos, account, vol_dp, "close_sl")
        if held_s >= CLOSE_MAX_HOLD_S:
            return self._close_all(response, validator, book_id, pos, account, vol_dp, "close_time")
        return "manage"

    def _manage_ping(self, response, validator, book_id, pos, account, now, vol_dp) -> str:
        """Sell the ping long after ACTIVITY_PING_HOLD_S to complete the round-trip."""
        held_s = (now - pos.opened_ns) / 1e9 if pos.opened_ns else 0.0
        if held_s >= ACTIVITY_PING_HOLD_S:
            return self._close_all(response, validator, book_id, pos, account, vol_dp, "activity_ping")
        return "ping_hold"

    def _maybe_open(self, response, validator, book_id, book, pos, st, account,
                    mid, fair, average, band_bps, stretch_bps, trend_bps,
                    imb, falling, cap, now, price_dp, vol_dp) -> str:
        """open_long / open_short when price is far from average."""
        if now < st.pause_until:
            return "pause"
        if self._rolled_volume(st, now) >= cap:
            return "cap"

        if average is None:
            return "warmup"

        downtrend = trend_bps < -TREND_GATE_BPS
        uptrend = trend_bps > TREND_GATE_BPS
        qty = round(self.quote_notional / mid, vol_dp)
        if qty < self.min_order_size:
            return "flat"

        # Cheap vs average + filters → buy toward the mean.
        if stretch_bps <= -band_bps and not falling and imb >= -IMBALANCE_GATE and not downtrend:
            self._enter_maker(response, account, book_id, book, OrderDirection.BUY, qty, price_dp)
            return "open_long"
        # Rich vs average + filters → sell toward the mean.
        if stretch_bps >= band_bps and not falling and imb <= IMBALANCE_GATE and not uptrend:
            self._enter_maker(response, account, book_id, book, OrderDirection.SELL, qty, price_dp)
            return "open_short"

        if falling:
            return "falling"
        return "flat"

    def _taker_ok(self, account) -> bool:
        """Use market orders only when taker fee is zero or a rebate (DIS)."""
        fees = getattr(account, "fees", None)
        rate = getattr(fees, "taker_fee_rate", None) if fees else None
        try:
            return rate is not None and float(rate) <= 0.0
        except (TypeError, ValueError):
            return False

    def _buy(self, response, book_id, qty) -> None:
        response.market_order(book_id=book_id, direction=OrderDirection.BUY, quantity=qty,
                              currency=OrderCurrency.BASE, stp=STP.CANCEL_OLDEST)

    def _sell(self, response, book_id, qty) -> None:
        response.market_order(book_id=book_id, direction=OrderDirection.SELL, quantity=qty,
                              currency=OrderCurrency.BASE, stp=STP.CANCEL_OLDEST)

    def _enter_maker(self, response, account, book_id, book, direction, qty, price_dp) -> None:
        """Post at the touch when fees favour maker; otherwise market if taker is free."""
        if direction == OrderDirection.BUY:
            if account.quote_balance.free < qty * (book.asks[0].price or 0):
                return
            price = round(book.bids[0].price, price_dp)
        else:
            if account.base_balance.free < qty:
                return
            price = round(book.asks[0].price, price_dp)

        if self._taker_ok(account):
            response.market_order(book_id=book_id, direction=direction, quantity=qty,
                                  currency=OrderCurrency.BASE, stp=STP.CANCEL_OLDEST)
        else:
            response.limit_order(book_id=book_id, direction=direction, quantity=qty, price=price,
                                 postOnly=True, timeInForce=TimeInForce.GTT,
                                 expiryPeriod=int(ENTRY_EXPIRY_S * 1e9), stp=STP.CANCEL_OLDEST)

    def _close_all(self, response, validator, book_id, pos, account, vol_dp, reason) -> str:
        """Market out of the full leg; reason is stored for the round_trip row on fill."""
        if pos.qty > 0:
            qty = round(min(pos.qty, account.base_balance.free), vol_dp)
            if qty >= self.min_order_size:
                self._exit_reason[(validator, book_id)] = reason
                self._sell(response, book_id, qty)
        else:
            qty = round(-pos.qty, vol_dp)
            if qty >= self.min_order_size:
                self._exit_reason[(validator, book_id)] = reason
                response.market_order(book_id=book_id, direction=OrderDirection.BUY, quantity=qty,
                                      currency=OrderCurrency.BASE, stp=STP.CANCEL_OLDEST)
        return reason

    def _snap(self, st, book_id, mid, bid, ask, pos, account,
              trend_bps, stretch_bps, imb, action, cap, now) -> None:
        """Write one telemetry row per book per step (dashboard signals + action)."""
        traded = self._rolled_volume(st, now)
        self.telemetry.snapshot(
            book_id=book_id, mid=mid, bid=bid, ask=ask,
            pos_qty=pos.qty, pos_avg=pos.avg,
            base_bal=account.base_balance.total if account.base_balance else None,
            quote_bal=account.quote_balance.total if account.quote_balance else None,
            traded_volume=traded, volume_cap=cap, volume_remaining=max(0.0, cap - traded),
            signals={"trend_bps": trend_bps, "flow": stretch_bps, "imb": imb},
            action=action,
        )


if __name__ == "__main__":
    launch(MeanReversionAgent)
