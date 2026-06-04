"""
MeanReversionAgent — Subnet 79 (MVTRX / taos)

Two behaviours per order book, both tuned for Kappa-3 (consistent, low-downside
realized round-trips):

  1. Dip-buy rebound (the main edge).
     A sharp dump is reliably followed by a slow grind back up. Measured on the
     target validator tape: after a >=50 bps drop in 20s, the median forward
     return is +52 bps at 5 min, +75 bps at 10 min (81% positive), still rising
     out to 30 min. So we wait for the fall to settle, buy, and SCALE OUT over
     ~25 minutes instead of dumping the whole position at once. A wide safety
     stop only guards against a book that keeps collapsing.

  2. Range fade (the everyday behaviour).
     When price is stretched away from its rolling average without a dump, fade
     back toward the average with a normal take-profit / stop. Entries are
     maker-first to save the spread.

  3. Activity ping (keep books alive for scoring).
     Validator samples round-trip volume every 10 sim-minutes; a book with no
     recent RT can sit at activity_factor 0 and contribute nothing to Kappa.
     If a book has had no closed round-trip for 9 minutes, post one minimum-size
     market round-trip (buy then sell a few seconds later) when taker fees allow.

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

# --- dump detection (a sharp vertical drop) -----------------------------------
DUMP_DROP_BPS = 50.0       # >=0.5% drop within the window counts as a dump
DUMP_WINDOW_S = 20.0       # lookback used to measure the drop
SETTLE_WAIT_S = 30.0       # wait after a dump before buying the rebound
STILL_FALLING_BPS = 6.0    # per-step drop that means it is still dropping (wait)

# --- rebound long: scale out as the book grinds back up -----------------------
# (seconds since entry, fraction of the original rebound size to sell)
REBOUND_EXITS = [(300, 0.30), (600, 0.30), (900, 0.30), (1200, 0.05), (1500, 0.05)]
REBOUND_STOP_BPS = 80.0    # wide safety stop; recovery dips ~44 bps at the 25th pct
REBOUND_MAX_HOLD_S = 1800.0
REBOUND_TOPUP_MIN_S = 90.0  # min time held before a fresh dump may top the long up

# --- range fade (no dump) -----------------------------------------------------
AVERAGE_WINDOW_S = 300.0    # rolling fair-value window (kept large on purpose)
TREND_WINDOW_S = 900.0      # slow trend filter
TREND_GATE_BPS = 25.0       # |average - trend| above this = trending, fade less
BAND_K = 1.8               # stretch band = k * price dispersion
MIN_BAND_BPS = 18.0        # only fade real stretches: activity_impact=0, so extra
                           # round-trips add no score but dilute kappa (cubed downside)
MAX_BAND_BPS = 120.0
FADE_TP_BPS = 14.0         # take-profit (spread median ~5 bps, so this clears it)
FADE_STOP_BPS = 22.0
FADE_MAX_HOLD_S = 180.0
PAUSE_AFTER_STOP_S = 90.0  # longer pause after a stop to avoid re-churning a book
IMBALANCE_DEPTH = 5
IMBALANCE_GATE = 0.30      # do not fade into a one-sided book

# --- shared safety ------------------------------------------------------------
SHORT_BLOCK_AFTER_DUMP_S = 1800.0  # never short a book that just dumped (it grinds up)
MIN_SAMPLES = 8
ENTRY_EXPIRY_S = 8.0       # GTT on maker entries
CAPITAL_TURNOVER_CAP = 10.0
VOLUME_SAFETY = 0.5        # stay under 50% of the volume cap
VOLUME_WINDOW_S = 86_400.0

# --- activity ping (no RT on book → activity_factor stays 0) ------------------
ACTIVITY_RT_IDLE_S = 540.0       # 9 min; inside the 10 min validator RT sample window
ACTIVITY_PING_HOLD_S = 2.0       # hold the ping long briefly, then close


@dataclass
class Position:
    """Net position on one book, rebuilt from our own fills."""
    qty: float = 0.0           # signed BASE (>0 long, <0 short)
    avg: float = 0.0           # average entry price
    opened_ns: int = 0         # when the current exposure opened
    is_rebound: bool = False   # opened as a dip-buy after a dump
    is_ping: bool = False      # minimum-size activity round-trip (not strategy)
    rebound_size: float = 0.0  # original rebound size, for percentage scale-outs
    legs_done: int = 0         # how many scheduled scale-out legs have fired


@dataclass
class BookState:
    """Rolling per-book stats and timers, all on simulation time (ns)."""
    prices: deque = field(default_factory=lambda: deque(maxlen=2000))  # (ts, trade price)
    mids: deque = field(default_factory=lambda: deque(maxlen=120))     # (ts, mid)
    trend_ema: float = 0.0
    rebound_until: int = 0       # rebound window still active until this time
    settle_at: int = 0           # earliest time we may buy after the last dump
    short_block_until: int = 0   # shorts blocked until this time
    pause_until: int = 0         # paused after a stop until this time
    vol_log: list = field(default_factory=list)  # (ts, quote volume)
    first_seen_ns: int = 0       # first state update on this book (sim time)
    last_rt_ns: int = 0          # last closed round-trip on this book


class MeanReversionAgent(FinanceSimulationAgent):

    # ------------------------------------------------------------------ setup
    def initialize(self) -> None:
        bt.logging.set_info()
        self.quote_notional = self._param("quote_notional", 1800.0)
        self.min_order_size = self._param("min_order_size", 0.25)

        self.average_window_ns = int(AVERAGE_WINDOW_S * 1e9)
        self.dump_window_ns = int(DUMP_WINDOW_S * 1e9)
        self.volume_window_ns = int(VOLUME_WINDOW_S * 1e9)
        self.trend_alpha = 1.0 - math.exp(-1.0 / TREND_WINDOW_S)

        # Small per-UID jitter so a replicated fleet does not post identical
        # prices and trade against itself.
        jitter = ((self.uid * 2654435761) % 1000) / 1000.0
        self.band_k = BAND_K * (0.92 + 0.16 * jitter)

        self.positions: dict[str, dict[int, Position]] = {}
        self.books: dict[str, dict[int, BookState]] = {}
        self._sim_id: dict[str, str] = {}
        self._exit_reason: dict[tuple[str, int], str] = {}
        self._step_ts_ns: int = 0

        self.activity_rt_idle_ns = int(ACTIVITY_RT_IDLE_S * 1e9)

        self.telemetry = MinerTelemetry.from_agent(self, agent_class="MeanReversionAgent")
        bt.logging.info(
            f"[MeanReversion uid={self.uid}] notional={self.quote_notional} "
            f"dump>={DUMP_DROP_BPS}bps/{DUMP_WINDOW_S}s scale-out over "
            f"{REBOUND_EXITS[-1][0] // 60}min · activity ping after "
            f"{ACTIVITY_RT_IDLE_S / 60:.0f}min idle"
        )

    def _param(self, name: str, default: float) -> float:
        try:
            return float(getattr(self.config, name, default))
        except (TypeError, ValueError):
            return default

    # --------------------------------------------------------------- lifecycle
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

    # ------------------------------------------------------------- fill tracking
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
            pos.avg, pos.opened_ns, pos.is_rebound, pos.is_ping, pos.legs_done = price, ts, False, False, 0

    # ----------------------------------------------------------------- features
    @staticmethod
    def _mid(book) -> float | None:
        if not book.bids or not book.asks:
            return None
        return 0.5 * (book.bids[0].price + book.asks[0].price)

    @staticmethod
    def _fair_price(book) -> float | None:
        """Size-weighted top-of-book price (leans toward the heavier side)."""
        if not book.bids or not book.asks:
            return None
        bid, ask = book.bids[0], book.asks[0]
        denom = bid.quantity + ask.quantity
        if denom <= 0:
            return 0.5 * (bid.price + ask.price)
        return (ask.price * bid.quantity + bid.price * ask.quantity) / denom

    def _imbalance(self, book) -> float:
        bq = sum(l.quantity for l in book.bids[:IMBALANCE_DEPTH])
        aq = sum(l.quantity for l in book.asks[:IMBALANCE_DEPTH])
        denom = bq + aq
        return (bq - aq) / denom if denom > 0 else 0.0

    def _ingest(self, st: BookState, book, mid: float, now: int) -> None:
        """Record this step's trade prints and mid; maintain rolling windows."""
        for e in book.events or []:
            if getattr(e, "type", None) == "t" and e.price > 0:
                st.prices.append((now, float(e.price)))
        while st.prices and st.prices[0][0] < now - self.average_window_ns:
            st.prices.popleft()
        st.mids.append((now, mid))
        while st.mids and st.mids[0][0] < now - self.dump_window_ns:
            st.mids.popleft()
        st.trend_ema = mid if st.trend_ema <= 0 else st.trend_ema + self.trend_alpha * (mid - st.trend_ema)

    def _average_and_band(self, st: BookState) -> tuple[float | None, float]:
        """Rolling average price (fair value) and a volatility-scaled fade band."""
        if len(st.prices) < MIN_SAMPLES:
            return None, MIN_BAND_BPS
        ps = [p for _, p in st.prices]
        mean = sum(ps) / len(ps)
        if mean <= 0:
            return None, MIN_BAND_BPS
        dispersion_bps = math.sqrt(sum((p - mean) ** 2 for p in ps) / len(ps)) / mean * 1e4
        band = max(MIN_BAND_BPS, min(MAX_BAND_BPS, self.band_k * dispersion_bps))
        return mean, band

    def _dump_drop_bps(self, st: BookState, mid: float) -> float:
        """Drop from the recent window high to now, in bps (>=0)."""
        if not st.mids:
            return 0.0
        hi = max(m for _, m in st.mids)
        return max(0.0, (hi - mid) / hi * 1e4) if hi > 0 else 0.0

    def _last_step_bps(self, st: BookState) -> float:
        """Most recent mid-to-mid move in bps (negative = falling)."""
        if len(st.mids) < 2:
            return 0.0
        prev, cur = st.mids[-2][1], st.mids[-1][1]
        return (cur - prev) / prev * 1e4 if prev > 0 else 0.0

    def _rolled_volume(self, st: BookState, now: int) -> float:
        st.vol_log = [(t, v) for t, v in st.vol_log if t >= now - self.volume_window_ns]
        return sum(v for _, v in st.vol_log)

    def _rt_idle(self, st: BookState, now: int) -> bool:
        """True if this book has had no closed round-trip for ACTIVITY_RT_IDLE_S."""
        if st.first_seen_ns <= 0 or now - st.first_seen_ns < self.activity_rt_idle_ns:
            return False
        ref = st.last_rt_ns if st.last_rt_ns > 0 else st.first_seen_ns
        return now - ref >= self.activity_rt_idle_ns

    def _clear_stale_flags(self, pos: Position) -> None:
        """Drop entry flags when flat so an unfilled order does not block the next tick."""
        if abs(pos.qty) < self.min_order_size / 2:
            pos.is_rebound = False
            pos.is_ping = False

    def _ping_book_for_step(self, validator: str, state, now: int, vol_dp: int) -> int | None:
        """One book per step: close an open ping first, else open on the lowest idle book."""
        for book_id, pos in sorted(self.positions.get(validator, {}).items()):
            if pos.is_ping and abs(pos.qty) >= self.min_order_size / 2:
                return book_id

        for book_id in sorted(state.books.keys()):
            st = self._state(validator, book_id)
            pos = self._position(validator, book_id)
            if abs(pos.qty) >= self.min_order_size / 2:
                continue
            if now < st.pause_until or now < st.rebound_until:
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

    # ------------------------------------------------------------------ respond
    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        validator = state.dendrite.hotkey
        cfg = self.simulation_config

        if self._sim_id.get(validator) != cfg.simulation_id:
            self.positions.pop(validator, None)
            self.books.pop(validator, None)
            self._exit_reason = {k: v for k, v in self._exit_reason.items() if k[0] != validator}
            self._sim_id[validator] = cfg.simulation_id

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

        # Detect a fresh dump and arm the rebound window + safety blocks.
        drop_bps = self._dump_drop_bps(st, mid)
        falling = self._last_step_bps(st) <= -STILL_FALLING_BPS
        if drop_bps >= DUMP_DROP_BPS:
            st.rebound_until = now + int(REBOUND_MAX_HOLD_S * 1e9)
            st.settle_at = now + int(SETTLE_WAIT_S * 1e9)
            st.short_block_until = now + int(SHORT_BLOCK_AFTER_DUMP_S * 1e9)

        trend_bps = (average - st.trend_ema) / st.trend_ema * 1e4 if (average and st.trend_ema > 0) else 0.0
        stretch_bps = (fair - average) / average * 1e4 if average else 0.0

        if abs(pos.qty) >= self.min_order_size / 2 and pos.avg > 0:
            action = self._manage(response, validator, book_id, pos, st, account,
                                  fair, drop_bps, falling, now, vol_dp)
        else:
            action = self._maybe_open(response, validator, book_id, book, pos, st, account,
                                      mid, fair, average, band_bps, stretch_bps, trend_bps,
                                      imb, drop_bps, falling, cap, now, price_dp, vol_dp,
                                      ping_book=ping_book)

        self._snap(st, book_id, mid, bid, ask, pos, account, trend_bps, stretch_bps, imb, action, cap, now)

    # ---------------------------------------------------------- manage open pos
    def _manage(self, response, validator, book_id, pos, st, account,
                fair, drop_bps, falling, now, vol_dp) -> str:
        """Exit logic for an open position (rebound scale-out or fade TP/stop)."""
        if pos.is_ping:
            return self._manage_ping(response, validator, book_id, pos, account, now, vol_dp)
        if pos.is_rebound:
            return self._manage_rebound(response, validator, book_id, pos, st, account,
                                        fair, drop_bps, falling, now, vol_dp)

        pnl_bps = ((fair - pos.avg) if pos.qty > 0 else (pos.avg - fair)) / pos.avg * 1e4
        held_s = (now - pos.opened_ns) / 1e9 if pos.opened_ns else 0.0
        if pnl_bps >= FADE_TP_BPS:
            return self._close_all(response, validator, book_id, pos, account, vol_dp, "tp")
        if pnl_bps <= -FADE_STOP_BPS:
            st.pause_until = now + int(PAUSE_AFTER_STOP_S * 1e9)
            return self._close_all(response, validator, book_id, pos, account, vol_dp, "sl")
        if held_s >= FADE_MAX_HOLD_S:
            return self._close_all(response, validator, book_id, pos, account, vol_dp, "time")
        return "manage"

    def _manage_rebound(self, response, validator, book_id, pos, st, account,
                        fair, drop_bps, falling, now, vol_dp) -> str:
        """Scale out of a rebound long over time; widen-stop / top-up guards."""
        pnl_bps = (fair - pos.avg) / pos.avg * 1e4
        held_s = (now - pos.opened_ns) / 1e9 if pos.opened_ns else 0.0

        if pnl_bps <= -REBOUND_STOP_BPS:
            return self._close_all(response, validator, book_id, pos, account, vol_dp, "rebound_stop")
        if held_s >= REBOUND_MAX_HOLD_S:
            return self._close_all(response, validator, book_id, pos, account, vol_dp, "rebound_time")

        # A fresh dump while still early -> add back up to the original size and
        # restart the scale-out on the combined position (user's "buy more" idea).
        if drop_bps >= DUMP_DROP_BPS and not falling and held_s >= REBOUND_TOPUP_MIN_S:
            add = round(pos.rebound_size - pos.qty, vol_dp)
            if add >= self.min_order_size and self._taker_ok(account):
                self._buy(response, book_id, add)
                pos.opened_ns = now
                pos.legs_done = 0
                return "rebound_add"

        if pos.legs_done < len(REBOUND_EXITS):
            leg_time, frac = REBOUND_EXITS[pos.legs_done]
            if held_s >= leg_time:
                last_leg = pos.legs_done == len(REBOUND_EXITS) - 1
                want = pos.qty if last_leg else pos.rebound_size * frac
                qty = round(min(want, pos.qty, account.base_balance.free), vol_dp)
                if qty >= self.min_order_size:
                    pos.legs_done += 1
                    self._exit_reason[(validator, book_id)] = "rebound_tp"
                    self._sell(response, book_id, qty)
                    return f"rebound_exit_{pos.legs_done}"
        return "rebound_hold"

    def _manage_ping(self, response, validator, book_id, pos, account, now, vol_dp) -> str:
        """Close the minimum-size activity long after a short hold."""
        held_s = (now - pos.opened_ns) / 1e9 if pos.opened_ns else 0.0
        if held_s >= ACTIVITY_PING_HOLD_S:
            return self._close_all(response, validator, book_id, pos, account, vol_dp, "activity_ping")
        return "ping_hold"

    # ------------------------------------------------------------- open new pos
    def _maybe_open(self, response, validator, book_id, book, pos, st, account,
                    mid, fair, average, band_bps, stretch_bps, trend_bps,
                    imb, drop_bps, falling, cap, now, price_dp, vol_dp,
                    ping_book: int | None = None) -> str:
        if now < st.pause_until:
            return "pause"
        if self._rolled_volume(st, now) >= cap:
            return "cap"

        if ping_book == book_id and not falling and self._rt_idle(st, now):
            qty = round(self.min_order_size, vol_dp)
            if (
                qty >= self.min_order_size
                and self._taker_ok(account)
                and account.quote_balance.free >= qty * (book.asks[0].price or mid)
            ):
                self._buy(response, book_id, qty)
                pos.is_ping = True
                pos.is_rebound = False
                pos.opened_ns = now
                return "ping_open"

        in_rebound = now < st.rebound_until
        settled = now >= st.settle_at

        # 1) Dip-buy the rebound after a dump has settled.
        if in_rebound and settled and not falling:
            qty = round(self.quote_notional / mid, vol_dp)
            if qty >= self.min_order_size and account.quote_balance.free >= qty * (book.asks[0].price or mid):
                self._buy(response, book_id, qty)
                pos.is_rebound = True
                pos.rebound_size = qty
                pos.legs_done = 0
                return "rebound_open"
            return "rebound_wait"

        if average is None:
            return "warmup"

        downtrend = trend_bps < -TREND_GATE_BPS
        uptrend = trend_bps > TREND_GATE_BPS
        qty = round(self.quote_notional / mid, vol_dp)
        if qty < self.min_order_size:
            return "flat"

        # 2) Fade a stretch back toward the average (no dump in play).
        if stretch_bps <= -band_bps and not falling and imb >= -IMBALANCE_GATE and not downtrend:
            self._enter_maker(response, account, book_id, book, OrderDirection.BUY, qty, price_dp)
            return "fade_long"
        if stretch_bps >= band_bps and now >= st.short_block_until and imb <= IMBALANCE_GATE and not uptrend:
            self._enter_maker(response, account, book_id, book, OrderDirection.SELL, qty, price_dp)
            return "fade_short"

        if falling:
            return "falling"
        if in_rebound:
            return "rebound_wait"
        return "flat"

    # ------------------------------------------------------------------ orders
    def _taker_ok(self, account) -> bool:
        """Take only when the taker fee is a rebate or zero."""
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
        """Post-only entry at the near touch; take only when the fee favours it."""
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

    # --------------------------------------------------------------- telemetry
    def _snap(self, st, book_id, mid, bid, ask, pos, account,
              trend_bps, stretch_bps, imb, action, cap, now) -> None:
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
