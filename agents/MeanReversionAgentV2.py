"""
MeanReversionAgentV2
====================

Taker min-lot scalper aligned with top miners (#22, #234, #248) on sim
``20260606_1135``:

  * **100% taker** — no maker orders; skip book when ``taker_fee_rate > 0``.
  * **Fixed 0.25 BASE** clip per trade (same as top-miner tape).
  * **Fast round-trips** — hold ≤15s, exit at fee-aware TP/SL on **bid** (realistic).
  * **128-book parallel** — each book: flat → taker buy → taker sell → flat.
  * **When to open**: mean-reversion dip OR activity RT due (~520s window).
  * **Signal** still filters *discretionary* entries; activity RT keeps books alive.

Run: ``AGENT_NAME=MeanReversionAgentV2`` in ``.env.miner-2``, then pm2 restart.

v1 baseline remains in ``MeanReversionAgent.py``.
"""

import math
import traceback
from collections import deque
from dataclasses import dataclass, field

import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse
from taos.im.protocol.events import TradeEvent, SimulationStartEvent
from taos.im.protocol.models import OrderDirection, OrderCurrency, STP

# ---------------------------------------------------------------------------
# Strategy constants — top-miner taker scalper (#22 / #234 / #248 tape)
# ---------------------------------------------------------------------------

MIN_ORDER_SIZE = 0.25             # fixed clip — top miners use min lot only

# --- entry signal (when to open a discretionary scalp vs activity-only RT) ---
MEAN_WINDOW_S = 300.0
MIN_SAMPLES = 8
K_ENTRY = 2.0
MIN_BAND_BPS = 10.0
MAX_BAND_BPS = 120.0
IMBALANCE_DEPTH = 5
IMB_GATE = 0.35

# --- exit (fee-aware floors; actual TP/SL computed live per book) ---
TP_BUFFER_BPS = 2.0               # profit above breakeven after taker RT + spread
MIN_TP_BPS = 6.0                  # floor when fees are zero/rebate
MIN_SL_BPS = 8.0
ASSUMED_TAKER_BPS = 2.3           # fallback if account.fees missing
DEFAULT_SPREAD_BPS = 6.0          # fallback full spread if book empty
MAX_HOLD_S = 15.0                 # top miners: RT completes in seconds
COOLDOWN_S = 30.0                 # pause book after stop-out

# --- trend / crash filters (same tape asymmetry as v1) ---
TREND_WINDOW_S = 600.0
TREND_GATE_BPS = 20.0
CRASH_BPS = 35.0
CRASH_WINDOW_S = 20.0
RECOVERY_WINDOW_S = 300.0
RECOVERY_TP_MULT = 1.2
GRIND_DUMP_WINDOW_S = 300.0
GRIND_DUMP_BPS = 38.0
GRIND_RISE_MIN_BPS = 8.0
GRIND_LONG_MAX_DEV_BPS = 18.0
MID_DEQUE_MAXLEN = 360
KNIFE_MIN_DROP_BPS = 20.0
KNIFE_STEP_BPS = 8.0
KNIFE_BLOCK_S = 8.0

# --- activity round-trip cadence (under 600s validator window) ---
RT_INTERVAL_S = 520.0
ORDER_COOLDOWN_S = 1.0            # min gap between orders on same book

# --- taker-only (top miners: zero maker fee on chart) ---
MAX_TAKER_FEE = 0.0               # trade only when taker fee ≤ 0 (free/rebate)

# --- volume cap ---
CAPITAL_TURNOVER_CAP = 10.0
VOLUME_SAFETY = 0.35
VOLUME_ASSESSMENT_NS = 86_400_000_000_000

PRICES_DEQUE_MAXLEN = 3000


@dataclass
class _Position:
    """Per-book net position reconstructed from our own fills."""
    qty: float = 0.0          # signed BASE (>0 long, <0 short)
    avg: float = 0.0          # volume-weighted average entry price
    entry_ts: int = 0         # sim timestamp (ns) current exposure opened
    post_crash: bool = False  # entered during a post-crash recovery window


@dataclass
class _BookState:
    """Rolling per-book statistics, all on simulation time (ns)."""
    prices: deque = field(default_factory=lambda: deque(maxlen=PRICES_DEQUE_MAXLEN))
    mids: deque = field(default_factory=lambda: deque(maxlen=30))      # (ts, mid) — CRASH_WINDOW_S
    grind_mids: deque = field(default_factory=lambda: deque(maxlen=MID_DEQUE_MAXLEN))  # slow dump / grind
    ema_long: float = 0.0
    crash_until: int = 0      # post-crash recovery window end (ns)
    knife_until: int = 0        # block new longs during active dump (ns)
    cooldown_until: int = 0   # pause after a stop-loss (ns)
    last_rt_ns: int = 0
    last_order_ns: int = 0
    vol_log: list = field(default_factory=list)


class MeanReversionAgentV2(FinanceSimulationAgent):
    # ------------------------------------------------------------------ setup
    def initialize(self) -> None:
        bt.logging.set_info()

        self.min_order_size = MIN_ORDER_SIZE
        self.mean_window_s = MEAN_WINDOW_S
        self.min_samples = MIN_SAMPLES
        self.k_entry = K_ENTRY
        self.min_band_bps = MIN_BAND_BPS
        self.max_band_bps = MAX_BAND_BPS
        self.imbalance_depth = IMBALANCE_DEPTH
        self.imb_gate = IMB_GATE
        self.tp_buffer_bps = TP_BUFFER_BPS
        self.min_tp_bps = MIN_TP_BPS
        self.min_sl_bps = MIN_SL_BPS
        self.assumed_taker_bps = ASSUMED_TAKER_BPS
        self.default_spread_bps = DEFAULT_SPREAD_BPS
        self.max_hold_s = MAX_HOLD_S
        self.cooldown_s = COOLDOWN_S
        self.trend_window_s = TREND_WINDOW_S
        self.trend_gate_bps = TREND_GATE_BPS
        self.crash_bps = CRASH_BPS
        self.crash_window_s = CRASH_WINDOW_S
        self.recovery_window_s = RECOVERY_WINDOW_S
        self.recovery_tp_mult = RECOVERY_TP_MULT
        self.knife_min_drop_bps = KNIFE_MIN_DROP_BPS
        self.knife_block_s = KNIFE_BLOCK_S
        self.knife_step_bps = KNIFE_STEP_BPS
        self.grind_dump_window_s = GRIND_DUMP_WINDOW_S
        self.grind_dump_bps = GRIND_DUMP_BPS
        self.grind_rise_min_bps = GRIND_RISE_MIN_BPS
        self.grind_long_max_dev_bps = GRIND_LONG_MAX_DEV_BPS
        self.rt_interval_s = RT_INTERVAL_S
        self.order_cooldown_s = ORDER_COOLDOWN_S
        self.rt_interval_ns = int(self.rt_interval_s * 1e9)
        self.order_cooldown_ns = int(self.order_cooldown_s * 1e9)
        self.max_taker_fee = MAX_TAKER_FEE
        self.turnover_cap = CAPITAL_TURNOVER_CAP
        self.volume_safety = VOLUME_SAFETY
        self.volume_assessment_ns = VOLUME_ASSESSMENT_NS

        # Per-UID jitter (±8%): fleet miners get slightly different k_entry / crash_bps
        # so they don't all hit the same book at the same threshold.
        jitter = ((self.uid * 2654435761) % 1000) / 1000.0
        self.k_entry *= 0.92 + 0.16 * jitter
        self.crash_bps *= 0.92 + 0.16 * jitter
        self.grind_dump_bps *= 0.92 + 0.16 * jitter

        self.mean_window_ns = int(self.mean_window_s * 1e9)
        self.grind_dump_window_ns = int(self.grind_dump_window_s * 1e9)
        self.trend_alpha = 1.0 - math.exp(-1.0 / max(self.trend_window_s, 1.0))

        # runtime state, keyed by validator hotkey then book id
        self.positions: dict[str, dict[int, _Position]] = {}
        self.books_state: dict[str, dict[int, _BookState]] = {}
        self._sim_id: dict[str, str] = {}
        self._exit_reason: dict[tuple[str, int], str] = {}
        self._step_ts_ns: int = 0


        bt.logging.info(
            f"[MeanReversionV2 uid={self.uid}] taker_scalper lot={self.min_order_size} "
            f"hold={self.max_hold_s}s rt={self.rt_interval_s}s "
            f"k_entry={self.k_entry:.2f} max_taker_fee={self.max_taker_fee}"
        )

    # --------------------------------------------------------------- lifecycle
    def onStart(self, event: SimulationStartEvent) -> None:
        self.positions.clear()
        self.books_state.clear()
        self._sim_id.clear()
        self._exit_reason.clear()
        bt.logging.info(f"[MeanReversionV2 uid={self.uid}] simulation start: reset state")

    def update(self, state: MarketSimulationStateUpdate) -> None:
        # Stamp the current simulation time so fills are tracked on sim-time.
        self._step_ts_ns = int(state.timestamp)
        super().update(state)

    # ------------------------------------------------------------- fill tracking
    def onTrade(self, event: TradeEvent, validator: str = None) -> None:
        if event.bookId is None:
            return
        if self.uid == event.takerAgentId:
            direction = OrderDirection.BUY if event.side == OrderDirection.BUY else OrderDirection.SELL
        elif self.uid == event.makerAgentId:
            direction = OrderDirection.SELL if event.side == OrderDirection.BUY else OrderDirection.BUY
        else:
            return
        ts_ns = self._step_ts_ns or event.timestamp
        self._record_trade_volume(validator, event.bookId, event.quantity, event.price, ts_ns)
        self._apply_fill(validator, event.bookId, direction, event.quantity, event.price, ts_ns)

    def _book_positions(self, validator: str) -> dict[int, _Position]:
        return self.positions.setdefault(validator, {})

    def _bstate(self, validator: str, book_id: int) -> _BookState:
        return self.books_state.setdefault(validator, {}).setdefault(book_id, _BookState())

    def _record_trade_volume(self, validator, book_id, qty, price, ts_ns) -> None:
        vol = float(qty) * float(price)
        if vol <= 0:
            return
        self._bstate(validator, book_id).vol_log.append((ts_ns, vol))

    def _rolled_quote_volume(self, validator, book_id, now_ns) -> float:
        st = self._bstate(validator, book_id)
        if not st.vol_log:
            return 0.0
        cutoff = now_ns - self.volume_assessment_ns
        st.vol_log = [(t, v) for t, v in st.vol_log if t >= cutoff]
        return sum(v for _, v in st.vol_log)

    def _apply_fill(self, validator, book_id, direction, qty, price, ts) -> None:
        pos = self._book_positions(validator).setdefault(book_id, _Position())
        signed = qty if direction == OrderDirection.BUY else -qty
        prev = pos.qty
        entry_avg = pos.avg
        entry_ts = pos.entry_ts
        if prev == 0 or (prev > 0) == (signed > 0):
            # Open or add in the same direction -> blend the average.
            total = abs(prev) + qty
            pos.avg = (pos.avg * abs(prev) + price * qty) / total if total > 0 else price
            pos.qty = prev + signed
            if prev == 0:
                pos.entry_ts = ts
        else:
            # Reduce / close / flip -> realize a round-trip on the closed amount.
            # Partial closes (activity-ping slice) count as RTs for scoring.
            closed_qty = min(qty, abs(prev))
            if closed_qty >= self.min_order_size / 2 and entry_avg > 0:
                self._exit_reason.pop((validator, book_id), None)
                self._bstate(validator, book_id).last_rt_ns = ts
            pos.qty = prev + signed
            if abs(pos.qty) < 1e-12:
                pos.qty, pos.avg, pos.entry_ts, pos.post_crash = 0.0, 0.0, 0, False
            elif (prev > 0) != (pos.qty > 0):
                pos.avg, pos.entry_ts = price, ts

    # ----------------------------------------------------------------- features
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

    def _book_imbalance(self, book) -> float:
        bq = sum(l.quantity for l in book.bids[: self.imbalance_depth])
        aq = sum(l.quantity for l in book.asks[: self.imbalance_depth])
        denom = bq + aq
        return (bq - aq) / denom if denom > 0 else 0.0

    def _ingest(self, st: _BookState, book, mid: float, now: int) -> None:
        """Append this step's prints + mid; maintain windows and the long EMA."""
        for e in book.events or []:
            if getattr(e, "type", None) == "t" and e.price > 0:
                st.prices.append((now, float(e.price)))
        cutoff = now - self.mean_window_ns
        while st.prices and st.prices[0][0] < cutoff:
            st.prices.popleft()
        st.mids.append((now, mid))
        crash_cut = now - int(self.crash_window_s * 1e9)
        while st.mids and st.mids[0][0] < crash_cut:
            st.mids.popleft()
        st.grind_mids.append((now, mid))
        grind_cut = now - self.grind_dump_window_ns
        while st.grind_mids and st.grind_mids[0][0] < grind_cut:
            st.grind_mids.popleft()
        st.ema_long = mid if st.ema_long <= 0 else st.ema_long + self.trend_alpha * (mid - st.ema_long)

    def _ref_and_band(self, st: _BookState) -> tuple[float | None, float]:
        """Rolling mean (ref) and per-book entry band in bps from dispersion."""
        if len(st.prices) < self.min_samples:
            return None, self.min_band_bps
        ps = [p for _, p in st.prices]
        mean = sum(ps) / len(ps)
        if mean <= 0:
            return None, self.min_band_bps
        var = sum((p - mean) ** 2 for p in ps) / len(ps)
        disp_bps = (math.sqrt(var) / mean) * 1e4
        band = self.k_entry * disp_bps
        return mean, max(self.min_band_bps, min(self.max_band_bps, band))

    def _crash_drop_bps(self, st: _BookState, mid: float) -> float:
        """Drop from the CRASH_WINDOW_S high to now, in bps (>=0 means a drop)."""
        if not st.mids:
            return 0.0
        hi = max(m for _, m in st.mids)
        if hi <= 0:
            return 0.0
        return max(0.0, (hi - mid) / hi * 1e4)

    def _grind_drop_bps(self, st: _BookState, mid: float) -> float:
        """Drop from the GRIND_DUMP_WINDOW_S high to now, in bps (slow multi-minute cliff)."""
        if not st.grind_mids:
            return 0.0
        hi = max(m for _, m in st.grind_mids)
        if hi <= 0:
            return 0.0
        return max(0.0, (hi - mid) / hi * 1e4)

    def _grind_rise_bps(self, st: _BookState, mid: float) -> float:
        """Rise from the GRIND_DUMP_WINDOW_S low to now, in bps (grind-up leg)."""
        if not st.grind_mids:
            return 0.0
        lo = min(m for _, m in st.grind_mids)
        if lo <= 0:
            return 0.0
        return max(0.0, (mid - lo) / lo * 1e4)

    def _last_step_bps(self, st: _BookState) -> float:
        """Most recent mid-to-mid move in bps (negative = falling)."""
        if len(st.mids) < 2:
            return 0.0
        prev = st.mids[-2][1]
        cur = st.mids[-1][1]
        return (cur - prev) / prev * 1e4 if prev > 0 else 0.0

    # ------------------------------------------------------------------ respond
    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        validator = state.dendrite.hotkey
        cfg = self.simulation_config

        if self._sim_id.get(validator) != cfg.simulation_id:
            self._book_positions(validator).clear()
            self.books_state.pop(validator, None)
            self._exit_reason = {k: v for k, v in self._exit_reason.items() if k[0] != validator}
            self._sim_id[validator] = cfg.simulation_id

        price_dp = cfg.priceDecimals
        vol_dp = cfg.volumeDecimals
        cap = self.turnover_cap * cfg.miner_wealth * self.volume_safety

        for book_id, book in state.books.items():
            try:
                self._handle_book(response, validator, book_id, book,
                                  price_dp, vol_dp, cap, state.timestamp)
            except Exception as ex:
                bt.logging.warning(
                    f"[MeanReversionV2 uid={self.uid}] book {book_id} error: {ex}\n"
                    f"{traceback.format_exc()}"
                )
        return response

    def _reconcile_position(self, account, pos, vol_dp) -> None:
        """Clamp tracked qty to free base; cap at one min lot (top-miner clip)."""
        if account.base_balance is None:
            return
        free = account.base_balance.free
        if pos.qty > 0:
            if free < self.min_order_size / 2:
                pos.qty, pos.avg, pos.entry_ts, pos.post_crash = 0.0, 0.0, 0, False
            else:
                pos.qty = round(min(pos.qty, free, self.min_order_size), vol_dp)

    def _order_cooldown_ok(self, st: _BookState, now: int) -> bool:
        return (st.last_order_ns == 0) or ((now - st.last_order_ns) >= self.order_cooldown_ns)

    def _rt_due(self, st: _BookState, now: int) -> bool:
        return (st.last_rt_ns == 0) or ((now - st.last_rt_ns) >= self.rt_interval_ns)

    def _spread_bps(self, book, mid: float) -> float:
        if not book.bids or not book.asks or mid <= 0:
            return self.default_spread_bps
        return (book.asks[0].price - book.bids[0].price) / mid * 1e4

    def _taker_fee_bps(self, account) -> float:
        fees = getattr(account, "fees", None)
        rate = getattr(fees, "taker_fee_rate", None) if fees is not None else None
        if rate is None:
            return self.assumed_taker_bps
        try:
            return float(rate) * 1e4
        except (TypeError, ValueError):
            return self.assumed_taker_bps

    def _roundtrip_fee_bps(self, account) -> float:
        """Taker-in + taker-out fee cost in bps."""
        return 2.0 * self._taker_fee_bps(account)

    def _tp_sl_bps(self, account, book, mid: float, post_crash: bool) -> tuple[float, float]:
        """Fee + spread aware TP/SL for a full taker round-trip."""
        rt_fee = self._roundtrip_fee_bps(account)
        spread = self._spread_bps(book, mid)
        tp = max(self.min_tp_bps, rt_fee + spread + self.tp_buffer_bps)
        sl = max(self.min_sl_bps, rt_fee + spread * 0.5)
        if post_crash:
            tp *= self.recovery_tp_mult
        return tp, sl

    def _taker_allowed(self, account) -> bool:
        fees = getattr(account, "fees", None)
        rate = getattr(fees, "taker_fee_rate", None) if fees is not None else None
        if rate is None:
            return False
        try:
            return float(rate) <= self.max_taker_fee
        except (TypeError, ValueError):
            return False

    def _handle_book(self, response, validator, book_id, book,
                     price_dp, vol_dp, cap, now) -> None:
        mid = self._mid(book)
        fair = self._microprice(book) or mid
        if mid is None or mid <= 0 or fair is None:
            return
        account = self.accounts.get(book_id)
        if account is None:
            return
        if not self._taker_allowed(account):
            return

        st = self._bstate(validator, book_id)
        self._ingest(st, book, mid, now)
        ref, band_bps = self._ref_and_band(st)
        bid = book.bids[0].price if book.bids else None
        ask = book.asks[0].price if book.asks else None
        imb = self._book_imbalance(book)
        pos = self._book_positions(validator).setdefault(book_id, _Position())
        self._reconcile_position(account, pos, vol_dp)

        if pos.qty < -self.min_order_size / 2:
            self._exit_reason[(validator, book_id)] = "short_flatten"
            st.last_order_ns = now
            self._taker_sell(response, account, book_id, pos, vol_dp)
            return

        drop_bps = self._crash_drop_bps(st, mid)
        grind_drop_bps = self._grind_drop_bps(st, mid)
        grind_rise_bps = self._grind_rise_bps(st, mid)
        step_bps = self._last_step_bps(st)
        if drop_bps >= self.crash_bps or grind_drop_bps >= self.grind_dump_bps:
            st.crash_until = now + int(self.recovery_window_s * 1e9)
        if step_bps <= -self.knife_step_bps and drop_bps >= self.knife_min_drop_bps:
            st.knife_until = now + int(self.knife_block_s * 1e9)
        in_recovery = now < st.crash_until
        knife_active = now < st.knife_until
        trend_bps = ((ref - st.ema_long) / st.ema_long * 1e4) if (ref and st.ema_long > 0) else 0.0
        downtrend = trend_bps < -self.trend_gate_bps

        # ---- manage open min-lot long: taker sell on bid PnL ----
        if pos.qty >= self.min_order_size / 2 and pos.avg > 0:
            exit_px = bid if bid and bid > 0 else mid
            pnl_bps = (exit_px - pos.avg) / pos.avg * 1e4
            tp, sl = self._tp_sl_bps(account, book, mid, pos.post_crash)
            timed_out = (now - pos.entry_ts) >= int(self.max_hold_s * 1e9) if pos.entry_ts else False
            if pnl_bps >= tp:
                self._exit_reason[(validator, book_id)] = "tp"
                exit_action = "exit_tp"
                st.last_order_ns = now
                self._taker_sell(response, account, book_id, pos, vol_dp)
            elif pnl_bps <= -sl:
                self._exit_reason[(validator, book_id)] = "sl"
                st.cooldown_until = now + int(self.cooldown_s * 1e9)
                exit_action = "exit_sl"
                st.last_order_ns = now
                self._taker_sell(response, account, book_id, pos, vol_dp)
            elif timed_out:
                self._exit_reason[(validator, book_id)] = "time"
                exit_action = "exit_time"
                st.last_order_ns = now
                self._taker_sell(response, account, book_id, pos, vol_dp)
            else:
                exit_action = "manage"
            return

        # ---- flat: taker buy on signal or activity RT ----
        if ref is None or now < st.cooldown_until:
            label = "warmup" if ref is None else "cooldown"
            return
        if not self._order_cooldown_ok(st, now):
            return
        if self._rolled_quote_volume(validator, book_id, now) >= cap:
            return

        dev_bps = (fair - ref) / ref * 1e4
        fade_long = False
        grind_long = False
        if dev_bps <= -band_bps:
            if not knife_active and imb >= -self.imb_gate:
                if in_recovery or not downtrend:
                    fade_long = True
        elif (in_recovery and mid > st.ema_long and grind_rise_bps >= self.grind_rise_min_bps
              and dev_bps <= self.grind_long_max_dev_bps):
            if not knife_active and imb >= -self.imb_gate:
                grind_long = True
                fade_long = True

        rt_due = self._rt_due(st, now)
        if fade_long or rt_due:
            action = (
                "fade_long_grind" if grind_long
                else "fade_long_recover" if in_recovery and fade_long
                else "fade_long" if fade_long
                else "activity_rt"
            )
            if self._taker_buy(response, account, book_id, book, pos, st, now):
                return

        if knife_active:
            label = "knife"
        elif in_recovery and mid > st.ema_long and grind_rise_bps >= self.grind_rise_min_bps:
            label = "grind_up"
        elif in_recovery:
            label = "recover"
        elif rt_due:
            label = "rt_skip"
        else:
            label = "flat"

    def _taker_buy(self, response, account, book_id, book, pos, st, now) -> bool:
        """Market-buy exactly one min lot."""
        qty = self.min_order_size
        ask_px = book.asks[0].price if book.asks else 0.0
        if ask_px <= 0 or account.quote_balance.free < qty * ask_px:
            return False
        st.last_order_ns = now
        pos.post_crash = now < st.crash_until
        response.market_order(book_id=book_id, direction=OrderDirection.BUY,
                              quantity=qty, currency=OrderCurrency.BASE,
                              stp=STP.CANCEL_OLDEST)
        return True

    def _taker_sell(self, response, account, book_id, pos, vol_dp) -> None:
        """Market-sell full min-lot long."""
        if pos.qty <= 0:
            return
        free = account.base_balance.free if account.base_balance else 0.0
        qty = round(min(self.min_order_size, pos.qty, free), vol_dp)
        if qty < self.min_order_size / 2:
            return
        response.market_order(book_id=book_id, direction=OrderDirection.SELL,
                              quantity=qty, currency=OrderCurrency.BASE,
                              stp=STP.CANCEL_OLDEST)



if __name__ == "__main__":
    launch(MeanReversionAgentV2)
