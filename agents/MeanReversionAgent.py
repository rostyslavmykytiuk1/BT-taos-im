"""
MeanReversionAgent
==================

Contrarian range-fader for Subnet 79 (MVTRX / taos), tuned for **Kappa-3**.

Why this design
---------------
An empirical study of the target validator's tape (see ``MINER_STRATEGY_REPORT.md``
§3B) shows, across all 128 books:

  * Returns are **mean-reverting at every horizon** (1s-120s); momentum is
    essentially absent. Fading extremes beats chasing trends.
  * **123/128 books** show a **sharp dump (~20 bps/s) then a slow grind up
    (~0.3 bps/s)**. After a >=50 bps cliff the median forward return over the
    next ~5 min is **+63 bps** (97/109 books recover). The edge is therefore
    *asymmetric*: do not mirror long/short rules.

Scoring reality (verified): Kappa-3 rewards **consistent, low-downside realized
round-trip PnL across all books**; LPM3 cubes losses, so one blow-out hurts far
more than a win helps. Volume is **not** rewarded today (activity_impact = 0),
so we stay well under the cap and never churn — top miners use only ~35-45k of
the 500k/book cap and win on PnL-per-round-trip, not volume.

Strategy
--------
Per book, each step:

  fair  = microprice
  ref   = rolling mean of recent trade prices (local fair value)
  band  = k_entry * price_dispersion          (per-book, scales with volatility)
  trend = (ref - long_EMA) / long_EMA         (slow direction filter)
  crash = fast drop over crash_window          (sharp-dump detector)

  if holding -> exit on TP / SL / time  (asymmetric in post-crash recovery)
  elif flat and not over the volume cap:
      * during an active cliff -> DO NOT catch the knife (block new longs)
      * post-crash floor (below ref, drop stalled) -> fade LONG with wider TP /
        longer hold (recovery is slow but persistent)
      * normal over-extension -> fade back toward ref, gated by the trend filter
      * enter MAKER (post-only limit at the opposite top-of-book) to earn fee
        rebate + better price; taker fallback only when the fee regime pays
        takers and the signal is strong

Every position is closed with a market order to **realize** PnL.

Run (local proxy test):
  python MeanReversionAgent.py --port 8902 --agent_id 0 \
    --params quote_notional=1800 tp_bps=12 sl_bps=16 max_hold_s=150 \
             k_entry=1.4 mean_window_s=120 trend_window_s=600 \
             crash_bps=35 crash_window_s=20 recovery_window_s=300 \
             recovery_tp_mult=1.8 recovery_hold_mult=2.0 knife_block_s=8
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
    prices: deque = field(default_factory=lambda: deque(maxlen=600))   # (ts, price)
    mids: deque = field(default_factory=lambda: deque(maxlen=120))     # (ts, mid)
    ema_long: float = 0.0
    crash_until: int = 0      # post-crash recovery window end (ns)
    knife_until: int = 0      # block-new-longs (active cliff) end (ns)
    short_block_until: int = 0  # block new shorts after a crash (ns)
    cooldown_until: int = 0   # pause after a stop-loss (ns)
    vol_log: list = field(default_factory=list)  # (ts, quote_volume) round-trip cost


class MeanReversionAgent(FinanceSimulationAgent):
    # ------------------------------------------------------------------ setup
    def initialize(self) -> None:
        bt.logging.set_info()

        # --- sizing ---
        self.quote_notional = self._param("quote_notional", 1800.0)   # QUOTE per entry
        self.min_order_size = self._param("min_order_size", 0.25)     # BASE

        # --- core fade signal ---
        self.mean_window_s = self._param("mean_window_s", 120.0)      # ref mean window
        self.min_samples = int(self._param("min_samples", 8.0))
        self.k_entry = self._param("k_entry", 1.4)                    # band = k * dispersion
        self.min_band_bps = self._param("min_band_bps", 8.0)          # floor on entry band
        self.max_band_bps = self._param("max_band_bps", 120.0)        # cap on entry band
        self.imbalance_depth = int(self._param("imbalance_depth", 5.0))
        self.imb_gate = self._param("imb_gate", 0.30)                 # don't fade into a one-sided push

        # --- exits (symmetric defaults; recovery overrides below) ---
        self.tp_bps = self._param("tp_bps", 12.0)
        self.sl_bps = self._param("sl_bps", 16.0)
        self.max_hold_s = self._param("max_hold_s", 150.0)
        self.cooldown_s = self._param("cooldown_s", 30.0)             # pause book after a stop

        # --- trend filter ---
        self.trend_window_s = self._param("trend_window_s", 600.0)
        self.trend_gate_bps = self._param("trend_gate_bps", 25.0)    # |ref-EMA| above this = trending

        # --- crash / recovery asymmetry (the §3B.5 pattern) ---
        self.crash_bps = self._param("crash_bps", 35.0)              # drop to call a "cliff"
        self.crash_window_s = self._param("crash_window_s", 20.0)    # over this lookback
        self.recovery_window_s = self._param("recovery_window_s", 300.0)
        self.recovery_tp_mult = self._param("recovery_tp_mult", 1.8)   # wider TP for post-crash longs
        self.recovery_hold_mult = self._param("recovery_hold_mult", 2.0)  # longer hold post-crash
        self.knife_block_s = self._param("knife_block_s", 8.0)       # block longs while still dropping
        self.knife_step_bps = self._param("knife_step_bps", 8.0)     # per-step drop that = "still falling"
        self.recovery_short_sl_mult = self._param("recovery_short_sl_mult", 0.6)  # tighter stop on shorts in recovery
        # Backtest on the target tape: shorting a book that recently dumped is the
        # dominant Kappa-3 tail (the slow grind up runs the short over). Blocking
        # shorts for a while after any crash lifted win-rate 61->71% and the
        # per-book Kappa-3 proxy 0.58->0.88. This is the single most important guard.
        self.short_block_after_crash_s = self._param("short_block_after_crash_s", 1800.0)

        # --- maker / taker routing ---
        self.entry_expiry_s = self._param("entry_expiry_s", 8.0)     # GTT on maker entries
        self.max_taker_fee = self._param("max_taker_fee", 0.0)       # only take when fee <= this (<=0 => rebate)

        # --- volume cap awareness (volume is NOT rewarded; just a constraint) ---
        self.turnover_cap = self._param("capital_turnover_cap", 10.0)
        self.volume_safety = self._param("volume_safety", 0.5)       # stay under 50% of cap
        self.volume_assessment_ns = int(self._param("volume_assessment_ns", 86_400_000_000_000))

        # Small deterministic per-UID jitter so a replicated fleet does not
        # post identical prices and self-interfere (helps cross-book picture).
        jitter = ((self.uid * 2654435761) % 1000) / 1000.0
        self.k_entry *= 0.92 + 0.16 * jitter
        self.crash_bps *= 0.92 + 0.16 * jitter

        self.mean_window_ns = int(self.mean_window_s * 1e9)
        self.trend_alpha = 1.0 - math.exp(-1.0 / max(self.trend_window_s, 1.0))

        # runtime state, keyed by validator hotkey then book id
        self.positions: dict[str, dict[int, _Position]] = {}
        self.books_state: dict[str, dict[int, _BookState]] = {}
        self._sim_id: dict[str, str] = {}
        self._exit_reason: dict[tuple[str, int], str] = {}
        self._step_ts_ns: int = 0

        self.telemetry = MinerTelemetry.from_agent(self, agent_class="MeanReversionAgent")

        bt.logging.info(
            f"[MeanReversion uid={self.uid}] notional={self.quote_notional} "
            f"tp={self.tp_bps}bps sl={self.sl_bps}bps hold={self.max_hold_s}s "
            f"k_entry={self.k_entry:.2f} mean={self.mean_window_s}s "
            f"crash={self.crash_bps:.1f}bps/{self.crash_window_s}s "
            f"recovery={self.recovery_window_s}s tp_mult={self.recovery_tp_mult}"
        )

    def _param(self, name: str, default: float) -> float:
        val = getattr(self.config, name, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    # --------------------------------------------------------------- lifecycle
    def onStart(self, event: SimulationStartEvent) -> None:
        self.positions.clear()
        self.books_state.clear()
        self._sim_id.clear()
        self._exit_reason.clear()
        bt.logging.info(f"[MeanReversion uid={self.uid}] simulation start: reset state")

    def update(self, state: MarketSimulationStateUpdate) -> None:
        # Stamp the current simulation time so fills are tracked on sim-time
        # (aligns telemetry + round-trip times with trades.csv / the dashboard).
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
            closed_qty = min(qty, abs(prev))
            if abs(prev + signed) < 1e-12 and abs(prev) >= self.min_order_size / 2 and entry_avg > 0:
                if prev > 0:
                    rpnl = (price - entry_avg) * closed_qty
                    side = "long"
                else:
                    rpnl = (entry_avg - price) * closed_qty
                    side = "short"
                hold_s = (ts - entry_ts) / 1e9 if entry_ts else None
                reason = self._exit_reason.pop((validator, book_id), "fill")
                self.telemetry.record_round_trip(
                    book_id=book_id, ts_close_ns=ts, side=side, qty=closed_qty,
                    entry_avg=entry_avg, exit_avg=price, realized_pnl=rpnl,
                    hold_s=hold_s, reason=reason,
                )
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
        """Drop from the window's high to now, in bps (>=0 means a drop)."""
        if not st.mids:
            return 0.0
        hi = max(m for _, m in st.mids)
        if hi <= 0:
            return 0.0
        return max(0.0, (hi - mid) / hi * 1e4)

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

        self.telemetry.begin_step(state)
        instr_before = len(response.instructions)
        for book_id, book in state.books.items():
            try:
                self._handle_book(response, validator, book_id, book,
                                  price_dp, vol_dp, cap, state.timestamp)
            except Exception as ex:
                bt.logging.warning(
                    f"[MeanReversion uid={self.uid}] book {book_id} error: {ex}\n"
                    f"{traceback.format_exc()}"
                )
        self.telemetry.end_step(state, instructions=len(response.instructions) - instr_before)
        return response

    def _handle_book(self, response, validator, book_id, book,
                     price_dp, vol_dp, cap, now) -> None:
        mid = self._mid(book)
        fair = self._microprice(book) or mid
        if mid is None or mid <= 0 or fair is None:
            return
        account = self.accounts.get(book_id)
        if account is None:
            return

        st = self._bstate(validator, book_id)
        self._ingest(st, book, mid, now)
        ref, band_bps = self._ref_and_band(st)
        bid = book.bids[0].price if book.bids else None
        ask = book.asks[0].price if book.asks else None
        imb = self._book_imbalance(book)
        pos = self._book_positions(validator).setdefault(book_id, _Position())

        # --- crash / knife state machine (the asymmetric §3B.5 handling) ---
        drop_bps = self._crash_drop_bps(st, mid)
        step_bps = self._last_step_bps(st)
        if drop_bps >= self.crash_bps:
            st.crash_until = now + int(self.recovery_window_s * 1e9)
            st.short_block_until = now + int(self.short_block_after_crash_s * 1e9)
        if step_bps <= -self.knife_step_bps:           # still actively falling
            st.knife_until = now + int(self.knife_block_s * 1e9)
        in_recovery = now < st.crash_until
        knife_active = now < st.knife_until
        short_blocked = now < st.short_block_until

        # trend filter
        trend_bps = ((ref - st.ema_long) / st.ema_long * 1e4) if (ref and st.ema_long > 0) else 0.0
        uptrend = trend_bps > self.trend_gate_bps
        downtrend = trend_bps < -self.trend_gate_bps

        # ---- 1) manage an open position (market exit guarantees realization) ----
        if abs(pos.qty) >= self.min_order_size / 2 and pos.avg > 0:
            pnl_bps = ((fair - pos.avg) if pos.qty > 0 else (pos.avg - fair)) / pos.avg * 1e4
            tp = self.tp_bps
            sl = self.sl_bps
            hold_ns = self.max_hold_s * 1e9
            if pos.qty > 0 and pos.post_crash:
                # Post-crash longs: recovery is slow -> wider TP, longer hold.
                tp *= self.recovery_tp_mult
                hold_ns *= self.recovery_hold_mult
            if pos.qty < 0 and in_recovery:
                # Shorts during a recovery grind are dangerous -> tighter stop.
                sl *= self.recovery_short_sl_mult
            timed_out = (now - pos.entry_ts) >= hold_ns if pos.entry_ts else False
            if pnl_bps >= tp:
                self._exit_reason[(validator, book_id)] = "tp"
                exit_action = "exit_tp"
                self._flatten(response, account, book_id, pos, vol_dp)
            elif pnl_bps <= -sl:
                self._exit_reason[(validator, book_id)] = "sl"
                st.cooldown_until = now + int(self.cooldown_s * 1e9)
                exit_action = "exit_sl"
                self._flatten(response, account, book_id, pos, vol_dp)
            elif timed_out:
                self._exit_reason[(validator, book_id)] = "time"
                exit_action = "exit_time"
                self._flatten(response, account, book_id, pos, vol_dp)
            else:
                exit_action = "manage"
            self._snap(validator, book_id, mid, bid, ask, pos, account,
                       trend_bps, (fair - ref) / ref * 1e4 if ref else 0.0, imb,
                       exit_action, cap, now)
            return

        # ---- 2) flat: decide whether to fade ----
        action = "hold"
        if ref is None or now < st.cooldown_until:
            self._snap(validator, book_id, mid, bid, ask, pos, account,
                       trend_bps, 0.0, imb, "warmup" if ref is None else "cooldown", cap, now)
            return

        if self._rolled_quote_volume(validator, book_id, now) >= cap:
            self._snap(validator, book_id, mid, bid, ask, pos, account,
                       trend_bps, 0.0, imb, "cap", cap, now)
            return

        dev_bps = (fair - ref) / ref * 1e4    # >0 stretched above ref, <0 below
        fade_long = False
        fade_short = False

        if dev_bps <= -band_bps:
            # Over-extended DOWN -> buy the dip, expecting reversion up.
            # But not while the knife is still falling, and not in a downtrend
            # unless this is a post-crash recovery floor (then we *want* the dip).
            if not knife_active and imb >= -self.imb_gate:
                if in_recovery or not downtrend:
                    fade_long = True
        elif dev_bps >= band_bps:
            # Over-extended UP -> fade short. Never short into a recovery grind
            # or a book that recently dumped (the slow grind up is the dominant
            # short tail), and shrink/skip in an uptrend.
            if not in_recovery and not short_blocked and imb <= self.imb_gate and not uptrend:
                fade_short = True

        if not (fade_long or fade_short):
            if knife_active:
                label = "knife"
            elif in_recovery:
                label = "recover"
            elif short_blocked and dev_bps >= band_bps:
                label = "short_blocked"
            else:
                label = "flat"
            self._snap(validator, book_id, mid, bid, ask, pos, account,
                       trend_bps, dev_bps, imb, label, cap, now)
            return

        qty = round(self.quote_notional / mid, vol_dp)
        if qty < self.min_order_size:
            self._snap(validator, book_id, mid, bid, ask, pos, account,
                       trend_bps, dev_bps, imb, "too_small", cap, now)
            return

        take_ok = self._taker_allowed(account)
        if fade_long:
            action = "fade_long_recover" if in_recovery else "fade_long"
            self._enter(response, account, book_id, OrderDirection.BUY, qty,
                        book, price_dp, take_ok, mark_post_crash=in_recovery, pos=pos)
        else:
            action = "fade_short"
            self._enter(response, account, book_id, OrderDirection.SELL, qty,
                        book, price_dp, take_ok, mark_post_crash=False, pos=pos)

        self._snap(validator, book_id, mid, bid, ask, pos, account,
                   trend_bps, dev_bps, imb, action, cap, now)

    # ------------------------------------------------------------------ orders
    def _taker_allowed(self, account) -> bool:
        """Allow a taker entry only when the current taker fee is a rebate/cheap."""
        fees = getattr(account, "fees", None)
        rate = getattr(fees, "taker_fee_rate", None) if fees is not None else None
        if rate is None:
            return False
        try:
            return float(rate) <= self.max_taker_fee
        except (TypeError, ValueError):
            return False

    def _enter(self, response, account, book_id, direction, qty, book,
               price_dp, take_ok, mark_post_crash, pos) -> None:
        """Maker-first entry; taker fallback only when the fee regime pays takers."""
        expiry_ns = int(self.entry_expiry_s * 1e9)
        if direction == OrderDirection.BUY:
            price = round(book.bids[0].price, price_dp)
            if account.quote_balance.free < qty * (book.asks[0].price or price):
                return
        else:
            price = round(book.asks[0].price, price_dp)
            if account.base_balance.free < qty:
                return

        # Remember intent so the realized fill is tagged as post-crash for exit
        # asymmetry. Always set deterministically on a long entry so a stale
        # True can't carry over from an earlier unfilled maker order.
        if direction == OrderDirection.BUY:
            pos.post_crash = bool(mark_post_crash)

        if take_ok:
            response.market_order(book_id=book_id, direction=direction, quantity=qty,
                                  currency=OrderCurrency.BASE, stp=STP.CANCEL_OLDEST)
        else:
            response.limit_order(book_id=book_id, direction=direction, quantity=qty,
                                 price=price, postOnly=True, timeInForce=TimeInForce.GTT,
                                 expiryPeriod=expiry_ns, stp=STP.CANCEL_OLDEST)

    def _flatten(self, response, account, book_id, pos, vol_dp) -> None:
        if pos.qty > 0:
            qty = round(min(pos.qty, account.base_balance.free), vol_dp)
            if qty < self.min_order_size:
                return
            response.market_order(book_id=book_id, direction=OrderDirection.SELL,
                                  quantity=qty, currency=OrderCurrency.BASE,
                                  stp=STP.CANCEL_OLDEST)
        else:
            qty = round(-pos.qty, vol_dp)
            if qty < self.min_order_size:
                return
            response.market_order(book_id=book_id, direction=OrderDirection.BUY,
                                  quantity=qty, currency=OrderCurrency.BASE,
                                  stp=STP.CANCEL_OLDEST)

    # --------------------------------------------------------------- telemetry
    def _snap(self, validator, book_id, mid, bid, ask, pos, account,
              trend_bps, dev_bps, imb, action, cap, now) -> None:
        traded = self._rolled_quote_volume(validator, book_id, now)
        self.telemetry.snapshot(
            book_id=book_id, mid=mid, bid=bid, ask=ask,
            pos_qty=pos.qty, pos_avg=pos.avg,
            base_bal=account.base_balance.total if account.base_balance else None,
            quote_bal=account.quote_balance.total if account.quote_balance else None,
            traded_volume=traded, volume_cap=cap, volume_remaining=max(0.0, cap - traded),
            # dashboard columns: trend_bps / flow / imb. We map flow->deviation
            # (the primary fade signal) so the Signals tab stays meaningful.
            signals={"trend_bps": trend_bps, "flow": dev_bps, "imb": imb},
            action=action,
        )


if __name__ == "__main__":
    launch(MeanReversionAgent)
