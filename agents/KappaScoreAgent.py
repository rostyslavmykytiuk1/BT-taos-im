# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
KappaScoreAgent
===============

A pure SCORE-MAXIMIZING agent for Subnet 79 (taos), not a profit-maximizing
trader. The single objective is the validator's intraday Kappa-3 score, which
on this subnet today behaves as follows (verified in taos/im/validator/reward.py
and taos/im/utils/kappa.py):

  * Only REALIZED round-trip PnL feeds Kappa-3. Open inventory is invisible.
  * Returns are MAD-normalized PER BOOK, so absolute profit size barely matters.
    Many small, same-signed wins beat a few large ones.
  * LPM3 cubes downside, so ONE realized loss hurts far more than an equal win
    helps. Avoiding realized losses is the dominant lever.
  * Final score = MEDIAN of activity-weighted Kappa across all books, minus a
    cross-book outlier penalty. Uniformity across all 128 books wins.
  * A book's activity_factor is 0 until its first completed round-trip, and a
    book needs >= min_realized_observations (3) closes and >= min_lookback
    (90 min) of history before it scores. To keep activity_factor at 1.0 you
    must COMPLETE a round-trip on the book within each activity sampling
    interval (600 s / 10 min today).
  * Volume is NOT rewarded today (activity_impact = 0): never churn, stay well
    under the per-book turnover cap.

Design (per book, every step)
-----------------------------
  fair = microprice; ref = rolling mean of recent trades; band scales with vol.

  HOLDING: work a PASSIVE maker close at a fee-clearing take-profit, and harvest
    fully once held long enough while profitable. A very wide disaster stop bounds
    the extreme cubed tail. Otherwise underwater positions are held (their
    unrealized PnL is invisible to Kappa-3) until the maker close fills -- EXCEPT
    that activity always takes priority (see ACTIVITY PING below).

  FLAT: enter only high-conviction mean-reversion fades whose expected reversion
    clears the round-trip fee + a profit buffer, gated by a trend filter and
    crash / knife guards (do not catch a falling knife; do not short a freshly
    dumped book). Long-biased by default: the tape grinds up after dumps, so
    shorts are the dominant tail and are disabled unless ALLOW_SHORTS is on.

  ACTIVITY PING (overrides everything): activeness is the hard floor -- a book
    that does not complete a round-trip within the ~10-min activity window loses
    its activity_factor and scores 0 on that book. Two-step, one lot at a time:
      FLAT + ping due  -> market BUY  exactly MIN_ORDER_SIZE (open leg)
      HOLDING + ping due -> market SELL exactly MIN_ORDER_SIZE (close leg = RT)
    Ping uses taker orders on purpose: we cancel resting quotes every tick, so a
    maker probe would be cancelled before it fills. Normal fades still use maker
    entries; only the mandatory activity legs use market orders.

Normal entries and passive take-profit closes are POST-ONLY maker orders.

Run (local proxy test):
  python KappaScoreAgent.py --port 8904 --agent_id 0

Tune the constants below, then: pm2 restart <miner>
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
from taos.im.protocol.models import OrderDirection, OrderCurrency, STP, TimeInForce

# ===========================================================================
# Strategy constants - edit here, then pm2 restart the miner
# ===========================================================================

# --- position size (small: tiny per-book tail risk; size is not rewarded) ---
QUOTE_NOTIONAL = 800.0          # QUOTE notional per normal entry
MIN_ORDER_SIZE = 0.25           # smallest BASE order the sim accepts
# Activity ping uses MIN_ORDER_SIZE only (one min lot open -> one min slice close).
# Large ping notionals stacked repeated opens and caused burst close loops.

# --- entry signal: rolling mean (ref) + volatility band ---
MEAN_WINDOW_S = 300.0           # seconds of trade prints for the rolling mean
MIN_SAMPLES = 8                 # min prints before ref/band are valid
K_ENTRY = 2.0                   # entry band = K_ENTRY * dispersion; high = selective
MIN_BAND_BPS = 10.0             # never fade a stretch smaller than this (bps)
MAX_BAND_BPS = 120.0            # cap the required stretch (bps)
IMBALANCE_DEPTH = 5             # LOB levels for depth imbalance
IMB_GATE = 0.30                 # skip long if heavy sell pressure (imb < -gate)

# --- fee-aware exits (the "do not lose to fees" core) ---
MIN_TP_BPS = 8.0                # base take-profit; raised at runtime to clear fees
TP_BUFFER_BPS = 4.0            # profit margin required ON TOP of round-trip fees
BE_BUFFER_BPS = 1.0             # break-even harvest sits this far above fee cost
ASSUMED_FEE_BPS = 2.3           # fallback per-side fee (bps) if account.fees is missing
MAX_HOLD_S = 240.0              # after this, harvest if profitable; else keep waiting
DISASTER_BPS = 150.0            # only realize a loss if it gets this bad (tail cap)
COOLDOWN_S = 30.0               # pause new entries on a book after a disaster stop

# --- trend filter (do not fade into a strong one-way move) ---
TREND_WINDOW_S = 600.0          # EMA lookback for slow trend (s); > MEAN_WINDOW_S
TREND_GATE_BPS = 25.0           # |ref vs EMA| beyond this = trending book

# --- crash / recovery asymmetry (tape: sharp dump then slow grind up) ---
CRASH_BPS = 35.0                # drop over CRASH_WINDOW_S that flags a cliff
CRASH_WINDOW_S = 20.0           # lookback for the fast cliff (s)
RECOVERY_WINDOW_S = 300.0       # after a cliff, treat book as recovery this long
RECOVERY_TP_MULT = 1.6          # post-crash longs: let winners run a bit more
RECOVERY_HOLD_MULT = 2.0        # post-crash longs: hold longer (recovery is slow)
GRIND_DUMP_WINDOW_S = 300.0     # slow multi-minute cliff lookback
GRIND_DUMP_BPS = 38.0           # drop over the grind window that flags a slow dump
MID_DEQUE_MAXLEN = 360          # mid history points for the grind window

# --- knife (block longs into an active dump) ---
KNIFE_MIN_DROP_BPS = 20.0       # already dumped this much from the 20s high
KNIFE_STEP_BPS = 8.0            # and still falling at least this much this step
KNIFE_BLOCK_S = 8.0             # block new longs this long after a knife trigger

# --- shorts (off by default: the grind-up is the dominant short tail) ---
ALLOW_SHORTS = False            # set True only if you accept the grind-up risk
SHORT_BLOCK_AFTER_CRASH_S = 1800.0  # if shorts on, block them this long post-cliff
ROCKET_MIN_RISE_BPS = 20.0      # rocket: already ripped this much from the 20s low
ROCKET_STEP_BPS = 8.0           # and still ripping this much this step
ROCKET_BLOCK_S = 8.0            # block new shorts this long after a rocket trigger

# --- order routing ---
ENTRY_EXPIRY_S = 8.0            # maker entry GTT expiry (s)
CLOSE_EXPIRY_S = 8.0            # maker close GTT expiry (s)
MAX_TAKER_FEE = 0.0             # only take when taker fee <= this (0 = rebate/free)

# --- activity ping (MANDATORY: a book that does not round-trip within the
#     ~10-min activity window loses its activity_factor and scores 0 on that
#     book -- confirmed live, regardless of config defaults). Every book MUST
#     complete a round-trip every window, unconditionally, even at a small loss:
#     an inactive book (score 0) is the single worst outcome. ---
PING_INTERVAL_S = 480.0         # force a round-trip within this (well under the 600s window)
PING_SUBMIT_COOLDOWN_S = 5.0    # min gap between ping orders on the same book (anti-burst)

# --- volume cap (hard limit; volume not rewarded today) ---
CAPITAL_TURNOVER_CAP = 10.0     # max turnover = this * miner_wealth
VOLUME_SAFETY = 0.5             # use only this fraction of the cap
VOLUME_ASSESSMENT_NS = 86_400_000_000_000  # rolling 24h volume window (ns)

PRICES_DEQUE_MAXLEN = 3000      # max stored trade prints per book


@dataclass
class _Position:
    """Per-book net position reconstructed from our own fills."""
    qty: float = 0.0          # signed BASE (>0 long, <0 short)
    avg: float = 0.0          # volume-weighted average entry price
    entry_ts: int = 0         # sim time (ns) the current exposure opened
    post_crash: bool = False  # opened during a post-crash recovery window


@dataclass
class _BookState:
    """Rolling per-book statistics, all on simulation time (ns)."""
    prices: deque = field(default_factory=lambda: deque(maxlen=PRICES_DEQUE_MAXLEN))
    mids: deque = field(default_factory=lambda: deque(maxlen=30))
    grind_mids: deque = field(default_factory=lambda: deque(maxlen=MID_DEQUE_MAXLEN))
    ema_long: float = 0.0
    crash_until: int = 0
    knife_until: int = 0
    rocket_until: int = 0
    short_block_until: int = 0
    cooldown_until: int = 0
    last_rt_ns: int = 0           # sim time (ns) of the last completed round-trip
    last_ping_submit_ns: int = 0  # sim time (ns) of last ping order submit (anti-burst)
    ping_awaiting_open: bool = False  # open leg submitted; do not stack another until fill
    vol_log: list = field(default_factory=list)  # (ts, quote_volume)


class KappaScoreAgent(FinanceSimulationAgent):
    # ------------------------------------------------------------------ setup
    def initialize(self) -> None:
        bt.logging.set_info()

        self.quote_notional = QUOTE_NOTIONAL
        self.min_order_size = MIN_ORDER_SIZE
        self.mean_window_s = MEAN_WINDOW_S
        self.min_samples = MIN_SAMPLES
        self.k_entry = K_ENTRY
        self.min_band_bps = MIN_BAND_BPS
        self.max_band_bps = MAX_BAND_BPS
        self.imbalance_depth = IMBALANCE_DEPTH
        self.imb_gate = IMB_GATE
        self.min_tp_bps = MIN_TP_BPS
        self.tp_buffer_bps = TP_BUFFER_BPS
        self.be_buffer_bps = BE_BUFFER_BPS
        self.assumed_fee_bps = ASSUMED_FEE_BPS
        self.max_hold_s = MAX_HOLD_S
        self.disaster_bps = DISASTER_BPS
        self.cooldown_s = COOLDOWN_S
        self.trend_window_s = TREND_WINDOW_S
        self.trend_gate_bps = TREND_GATE_BPS
        self.crash_bps = CRASH_BPS
        self.crash_window_s = CRASH_WINDOW_S
        self.recovery_window_s = RECOVERY_WINDOW_S
        self.recovery_tp_mult = RECOVERY_TP_MULT
        self.recovery_hold_mult = RECOVERY_HOLD_MULT
        self.grind_dump_window_s = GRIND_DUMP_WINDOW_S
        self.grind_dump_bps = GRIND_DUMP_BPS
        self.knife_min_drop_bps = KNIFE_MIN_DROP_BPS
        self.knife_step_bps = KNIFE_STEP_BPS
        self.knife_block_s = KNIFE_BLOCK_S
        self.allow_shorts = ALLOW_SHORTS
        self.short_block_after_crash_s = SHORT_BLOCK_AFTER_CRASH_S
        self.rocket_min_rise_bps = ROCKET_MIN_RISE_BPS
        self.rocket_step_bps = ROCKET_STEP_BPS
        self.rocket_block_s = ROCKET_BLOCK_S
        self.entry_expiry_s = ENTRY_EXPIRY_S
        self.close_expiry_s = CLOSE_EXPIRY_S
        self.max_taker_fee = MAX_TAKER_FEE
        self.ping_interval_s = PING_INTERVAL_S
        self.ping_submit_cooldown_ns = int(PING_SUBMIT_COOLDOWN_S * 1e9)
        self.turnover_cap = CAPITAL_TURNOVER_CAP
        self.volume_safety = VOLUME_SAFETY
        self.volume_assessment_ns = VOLUME_ASSESSMENT_NS

        # Per-UID jitter (+/-8%) so a fleet does not hit the same threshold at
        # the same instant on the same book.
        jitter = ((self.uid * 2654435761) % 1000) / 1000.0
        self.k_entry *= 0.92 + 0.16 * jitter
        self.crash_bps *= 0.92 + 0.16 * jitter
        self.grind_dump_bps *= 0.92 + 0.16 * jitter

        self.mean_window_ns = int(self.mean_window_s * 1e9)
        self.grind_dump_window_ns = int(self.grind_dump_window_s * 1e9)
        self.ping_interval_ns = int(self.ping_interval_s * 1e9)
        self.trend_alpha = 1.0 - math.exp(-1.0 / max(self.trend_window_s, 1.0))

        # runtime state, keyed by validator hotkey then book id
        self.positions: dict[str, dict[int, _Position]] = {}
        self.books_state: dict[str, dict[int, _BookState]] = {}
        self._sim_id: dict[str, str] = {}
        self._exit_reason: dict[tuple[str, int], str] = {}
        self._step_ts_ns: int = 0


        bt.logging.info(
            f"[KappaScore uid={self.uid}] notional={self.quote_notional} "
            f"min_tp={self.min_tp_bps}bps buffer={self.tp_buffer_bps}bps "
            f"hold={self.max_hold_s}s disaster={self.disaster_bps}bps "
            f"k_entry={self.k_entry:.2f} shorts={self.allow_shorts} "
            f"ping={self.ping_interval_s}s"
        )

    # --------------------------------------------------------------- lifecycle
    def onStart(self, event: SimulationStartEvent) -> None:
        self.positions.clear()
        self.books_state.clear()
        self._sim_id.clear()
        self._exit_reason.clear()
        bt.logging.info(f"[KappaScore uid={self.uid}] simulation start: reset state")

    def update(self, state: MarketSimulationStateUpdate) -> None:
        # Stamp sim time so fills are tracked on sim-time.
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
            if direction == OrderDirection.BUY:
                self._bstate(validator, book_id).ping_awaiting_open = False
        else:
            # Reduce / close / flip -> realize a round-trip on the closed amount.
            # Partial closes (e.g. a 0.25 activity-ping slice) are real round-trips
            # for scoring; update last_rt_ns on every slice so we do not drain the
            # whole position one tick at a time while the ping window is open.
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
        if not st.mids:
            return 0.0
        hi = max(m for _, m in st.mids)
        if hi <= 0:
            return 0.0
        return max(0.0, (hi - mid) / hi * 1e4)

    def _pump_rise_bps(self, st: _BookState, mid: float) -> float:
        if not st.mids:
            return 0.0
        lo = min(m for _, m in st.mids)
        if lo <= 0:
            return 0.0
        return max(0.0, (mid - lo) / lo * 1e4)

    def _grind_drop_bps(self, st: _BookState, mid: float) -> float:
        if not st.grind_mids:
            return 0.0
        hi = max(m for _, m in st.grind_mids)
        if hi <= 0:
            return 0.0
        return max(0.0, (hi - mid) / hi * 1e4)

    def _last_step_bps(self, st: _BookState) -> float:
        if len(st.mids) < 2:
            return 0.0
        prev = st.mids[-2][1]
        cur = st.mids[-1][1]
        return (cur - prev) / prev * 1e4 if prev > 0 else 0.0

    # --------------------------------------------------------------- fee helpers
    def _maker_fee_bps(self, account) -> float:
        """Per-side maker fee in bps (negative = rebate)."""
        fees = getattr(account, "fees", None)
        rate = getattr(fees, "maker_fee_rate", None) if fees is not None else None
        if rate is None:
            return self.assumed_fee_bps
        try:
            return float(rate) * 1e4
        except (TypeError, ValueError):
            return self.assumed_fee_bps

    def _roundtrip_fee_bps(self, account) -> float:
        """Maker-in + maker-out fee cost in bps (negative if both rebate)."""
        return 2.0 * self._maker_fee_bps(account)

    def _tp_and_breakeven_bps(self, account) -> tuple[float, float]:
        """Take-profit and break-even thresholds, both clearing the fee cost."""
        rt_fee = self._roundtrip_fee_bps(account)
        tp = max(self.min_tp_bps, rt_fee + self.tp_buffer_bps)
        breakeven = rt_fee + self.be_buffer_bps   # min profit that is not a loss
        return tp, breakeven

    def _taker_allowed(self, account) -> bool:
        fees = getattr(account, "fees", None)
        rate = getattr(fees, "taker_fee_rate", None) if fees is not None else None
        if rate is None:
            return False
        try:
            return float(rate) <= self.max_taker_fee
        except (TypeError, ValueError):
            return False

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
                    f"[KappaScore uid={self.uid}] book {book_id} error: {ex}\n"
                    f"{traceback.format_exc()}"
                )
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
        self._reconcile_position(account, pos, vol_dp)

        # ALLOW_SHORTS=False: accidental shorts (oversell / stale pos) are toxic
        # for Kappa-3 — flatten immediately and do nothing else this tick.
        if not self.allow_shorts and pos.qty < -self.min_order_size / 2:
            self._exit_reason[(validator, book_id)] = "short_flatten"
            st.last_ping_submit_ns = now
            self._market_flatten(response, account, book_id, pos, vol_dp)
            return

        # Always start from a clean slate: cancel our resting orders so we never
        # stack toward max_open_orders and quotes never go stale.
        if account.orders:
            response.cancel_orders(book_id, [o.id for o in account.orders])

        # --- crash / knife / rocket state machine ---
        drop_bps = self._crash_drop_bps(st, mid)
        grind_drop_bps = self._grind_drop_bps(st, mid)
        rise_bps = self._pump_rise_bps(st, mid)
        step_bps = self._last_step_bps(st)
        if drop_bps >= self.crash_bps or grind_drop_bps >= self.grind_dump_bps:
            st.crash_until = now + int(self.recovery_window_s * 1e9)
            st.short_block_until = now + int(self.short_block_after_crash_s * 1e9)
        if step_bps <= -self.knife_step_bps and drop_bps >= self.knife_min_drop_bps:
            st.knife_until = now + int(self.knife_block_s * 1e9)
        if step_bps >= self.rocket_step_bps and rise_bps >= self.rocket_min_rise_bps:
            st.rocket_until = now + int(self.rocket_block_s * 1e9)
        in_recovery = now < st.crash_until
        knife_active = now < st.knife_until
        rocket_active = now < st.rocket_until
        short_blocked = now < st.short_block_until

        trend_bps = ((ref - st.ema_long) / st.ema_long * 1e4) if (ref and st.ema_long > 0) else 0.0
        uptrend = trend_bps > self.trend_gate_bps
        downtrend = trend_bps < -self.trend_gate_bps
        tp_bps, breakeven_bps = self._tp_and_breakeven_bps(account)

        inv_qty = self._inventory_long(account, pos, vol_dp)

        # ---- 1) HOLDING LONG: work a fee-clearing maker close ----
        if inv_qty >= self.min_order_size / 2 and pos.avg > 0:
            self._manage_position(response, validator, book_id, book, account, pos, st,
                                  mid, bid, ask, price_dp, vol_dp, trend_bps, ref, imb,
                                  in_recovery, tp_bps, breakeven_bps, cap, now, inv_qty)
            return

        # ---- 2) FLAT: mandatory activity open leg, then optional fade ----
        if self._ping_open_due(st, inv_qty, now):
            self._submit_ping_open(response, validator, book_id, account, pos, st,
                                   book, mid, bid, ask, trend_bps, imb, cap, now)
            return

        if ref is None or now < st.cooldown_until:
            return

        if self._rolled_quote_volume(validator, book_id, now) >= cap:
            return

        dev_bps = (fair - ref) / ref * 1e4    # >0 stretched above ref, <0 below
        # Only fade if the reversion back toward ref can clear fees + buffer.
        min_edge = max(band_bps, tp_bps + self.tp_buffer_bps)

        fade_long = False
        fade_short = False

        if dev_bps <= -min_edge:
            # Oversold -> buy the dip. Block knife; respect downtrend unless this
            # is a post-crash recovery floor (then we want the dip).
            if not knife_active and imb >= -self.imb_gate and (in_recovery or not downtrend):
                fade_long = True
        elif self.allow_shorts and dev_bps >= min_edge:
            if (not rocket_active and not in_recovery and not short_blocked
                    and not uptrend and imb <= self.imb_gate):
                fade_short = True

        if not (fade_long or fade_short):
            if knife_active:
                label = "knife"
            elif rocket_active:
                label = "rocket"
            elif in_recovery:
                label = "recover"
            elif short_blocked and dev_bps >= min_edge:
                label = "short_blocked"
            else:
                label = "flat"
            return

        qty = round(self.quote_notional / mid, vol_dp)
        if qty < self.min_order_size:
            qty = self.min_order_size
        take_ok = self._taker_allowed(account)

        if fade_long:
            action = "fade_long_recover" if in_recovery else "fade_long"
            self._enter(response, account, book_id, OrderDirection.BUY, qty, book,
                        price_dp, take_ok, mark_post_crash=in_recovery, pos=pos)
        else:
            action = "fade_short"
            self._enter(response, account, book_id, OrderDirection.SELL, qty, book,
                        price_dp, take_ok, mark_post_crash=False, pos=pos)


    # ---------------------------------------------------------- manage position
    def _manage_position(self, response, validator, book_id, book, account, pos, st,
                         mid, bid, ask, price_dp, vol_dp, trend_bps, ref, imb,
                         in_recovery, tp_bps, breakeven_bps, cap, now, inv_qty) -> None:
        """Work a passive maker close. Only realize a loss on a disaster move.

        For score, the close price must clear the round-trip fee cost, so we
        never lock a loss purely on fees. Underwater positions are held (their
        unrealized PnL is invisible to Kappa-3) until the maker close at the
        fee-clearing target fills or the price reverts.
        """
        if not book.bids or not book.asks:
            return
        if pos.avg <= 0:
            return
        best_bid = book.bids[0].price
        best_ask = book.asks[0].price
        pnl_bps = ((mid - pos.avg) if pos.qty > 0 else (pos.avg - mid)) / pos.avg * 1e4

        tp = tp_bps
        hold_ns = self.max_hold_s * 1e9
        if pos.qty > 0 and pos.post_crash:
            tp *= self.recovery_tp_mult
            hold_ns *= self.recovery_hold_mult
        timed_out = (now - pos.entry_ts) >= hold_ns if pos.entry_ts else False

        # Disaster stop: bound the cubed tail. Rare; only catastrophic moves.
        if pnl_bps <= -self.disaster_bps:
            self._exit_reason[(validator, book_id)] = "disaster"
            st.cooldown_until = now + int(self.cooldown_s * 1e9)
            self._market_flatten(response, account, book_id, pos, vol_dp)
            return

        # Harvest: held long enough AND already profitable past break-even ->
        # realize a guaranteed non-loss win with a market close to free capital
        # and keep the book active.
        if timed_out and pnl_bps >= breakeven_bps:
            self._exit_reason[(validator, book_id)] = "harvest"
            self._market_flatten(response, account, book_id, pos, vol_dp)
            return

        # MANDATORY activity round-trip (keeps the book scored WHILE holding).
        # A book that does not complete a round-trip within the activity window
        # loses its activity_factor and contributes 0 to the median -- which is
        # worse than a small realized loss. So when the ping window lapses we ALWAYS
        # bank a MINIMUM slice as a round-trip, in any case, keeping the rest of the
        # position riding. Minimum size keeps the per-book LPM3 hit tiny vs the
        # book's wins. (The disaster stop above already fully exits a catastrophic
        # move, which is itself a round-trip.)
        if self._ping_close_due(st, inv_qty, now):
            self._submit_ping_close(response, validator, book_id, account, pos, st,
                                    mid, bid, ask, trend_bps, imb, breakeven_bps,
                                    pnl_bps, vol_dp, cap, now, inv_qty)
            return

        # Otherwise work a passive maker close at the fee-clearing target.
        close_qty = round(inv_qty, vol_dp)
        if close_qty < self.min_order_size:
            return
        expiry_ns = int(self.close_expiry_s * 1e9)
        target = round(max(pos.avg * (1 + tp / 1e4), best_ask), price_dp)
        if account.base_balance.free >= close_qty:
            response.limit_order(book_id=book_id, direction=OrderDirection.SELL,
                                 quantity=close_qty, price=target, postOnly=True,
                                 timeInForce=TimeInForce.GTT, expiryPeriod=expiry_ns,
                                 stp=STP.CANCEL_OLDEST)
        self._exit_reason[(validator, book_id)] = "tp"

    # ----------------------------------------------------------- activity ping
    def _rt_due(self, st: _BookState, now: int) -> bool:
        return (st.last_rt_ns == 0) or ((now - st.last_rt_ns) >= self.ping_interval_ns)

    def _submit_cooldown_ok(self, st: _BookState, now: int) -> bool:
        return (st.last_ping_submit_ns == 0) or (
            (now - st.last_ping_submit_ns) >= self.ping_submit_cooldown_ns)

    def _ping_open_due(self, st: _BookState, inv_qty: float, now: int) -> bool:
        """Flat book needs the open leg of a mandatory activity round-trip."""
        if st.ping_awaiting_open:
            # Open market order in flight — do not stack a second buy.
            if (now - st.last_ping_submit_ns) > int(60e9):
                st.ping_awaiting_open = False
            else:
                return False
        return inv_qty < self.min_order_size / 2 and self._rt_due(st, now) and self._submit_cooldown_ok(st, now)

    def _ping_close_due(self, st: _BookState, inv_qty: float, now: int) -> bool:
        """Holding book needs the close leg of a mandatory activity round-trip."""
        return inv_qty >= self.min_order_size / 2 and self._rt_due(st, now) and self._submit_cooldown_ok(st, now)

    def _submit_ping_open(self, response, validator, book_id, account, pos, st, book,
                          mid, bid, ask, trend_bps, imb, cap, now) -> None:
        """Market-buy exactly one min lot (open leg). Survives cancel-all-orders."""
        qty = self.min_order_size
        ask_px = book.asks[0].price if book.asks else mid
        if account.quote_balance.free < qty * ask_px:
            return
        st.last_ping_submit_ns = now
        st.ping_awaiting_open = True
        self._exit_reason[(validator, book_id)] = "ping_open"
        response.market_order(book_id=book_id, direction=OrderDirection.BUY,
                              quantity=qty, currency=OrderCurrency.BASE,
                              stp=STP.CANCEL_OLDEST)

    def _submit_ping_close(self, response, validator, book_id, account, pos, st,
                           mid, bid, ask, trend_bps, imb, breakeven_bps, pnl_bps,
                           vol_dp, cap, now, inv_qty) -> None:
        """Market-sell exactly one min lot (close leg = round-trip for scoring)."""
        reason = "ping_harvest" if pnl_bps >= breakeven_bps else "ping_active"
        slice_qty = round(min(self.min_order_size, inv_qty), vol_dp)
        if slice_qty < self.min_order_size:
            return
        st.last_ping_submit_ns = now
        self._exit_reason[(validator, book_id)] = reason
        # Close exactly one lot; never drain the whole position in a burst.
        self._market_close(response, account, book_id, pos, slice_qty, vol_dp)

    def _inventory_long(self, account, pos, vol_dp) -> float:
        """Long inventory we opened (fill tracker), capped by free base."""
        self._reconcile_position(account, pos, vol_dp)
        if pos.qty < self.min_order_size / 2:
            return 0.0
        if account.base_balance is None:
            return round(pos.qty, vol_dp)
        free = account.base_balance.free
        if free < self.min_order_size / 2:
            return 0.0
        return round(min(pos.qty, free), vol_dp)

    def _reconcile_position(self, account, pos, vol_dp) -> None:
        """Clamp agent position to on-chain inventory so we never oversell into a short."""
        if account.base_balance is None:
            return
        free = account.base_balance.free
        if pos.qty > 0:
            if free < self.min_order_size / 2:
                pos.qty, pos.avg, pos.entry_ts, pos.post_crash = 0.0, 0.0, 0, False
            elif free < pos.qty - self.min_order_size / 4:
                pos.qty = round(free, vol_dp)

    # ------------------------------------------------------------------ orders
    def _enter(self, response, account, book_id, direction, qty, book,
               price_dp, take_ok, mark_post_crash, pos) -> None:
        """Maker-first entry; taker only when the fee regime pays takers."""
        expiry_ns = int(self.entry_expiry_s * 1e9)
        if direction == OrderDirection.BUY:
            price = round(book.bids[0].price, price_dp)
            if account.quote_balance.free < qty * (book.asks[0].price or price):
                return
        else:
            price = round(book.asks[0].price, price_dp)
            if account.base_balance.free < qty:
                return

        if direction == OrderDirection.BUY:
            pos.post_crash = bool(mark_post_crash)

        if take_ok:
            response.market_order(book_id=book_id, direction=direction, quantity=qty,
                                  currency=OrderCurrency.BASE, stp=STP.CANCEL_OLDEST)
        else:
            response.limit_order(book_id=book_id, direction=direction, quantity=qty,
                                 price=price, postOnly=True, timeInForce=TimeInForce.GTT,
                                 expiryPeriod=expiry_ns, stp=STP.CANCEL_OLDEST)

    def _market_flatten(self, response, account, book_id, pos, vol_dp) -> None:
        self._market_close(response, account, book_id, pos, abs(pos.qty), vol_dp)

    def _market_close(self, response, account, book_id, pos, qty, vol_dp) -> None:
        """Market-close up to `qty` BASE of the current position (partial allowed)."""
        if pos.qty > 0:
            free = account.base_balance.free if account.base_balance else 0.0
            q = round(min(qty, pos.qty, free), vol_dp)
            if q < self.min_order_size:
                return
            response.market_order(book_id=book_id, direction=OrderDirection.SELL,
                                  quantity=q, currency=OrderCurrency.BASE,
                                  stp=STP.CANCEL_OLDEST)
        else:
            q = round(min(qty, -pos.qty), vol_dp)
            if q < self.min_order_size:
                return
            response.market_order(book_id=book_id, direction=OrderDirection.BUY,
                                  quantity=q, currency=OrderCurrency.BASE,
                                  stp=STP.CANCEL_OLDEST)



if __name__ == "__main__":
    launch(KappaScoreAgent)

