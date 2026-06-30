"""
MeanReversionAgent
==================

Contrarian range-fader for Subnet 79 (MVTRX / taos), tuned for **Kappa-3**.
Simple, reliable, long-only maker-fade with a low-churn activity backstop. Per-validator
state + simulation-reset handling, FIFO-safe (always buys in-sim before selling).

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

  if holding -> exit on TP / SL / time  (wider TP / longer hold post-crash)
  elif flat and not over the volume cap:
      * activity backstop: only when no round-trip in ~25 min (min lot; separate from fades).
        The validator needs >=3 RTs/book inside the 3h kappa lookback to score a book, and the
        activity factor decays only ~2% over a 25-min gap, so we keep ~7 RTs/3h (>2x the floor)
        instead of churning every 10 min and injecting marginal fee-loss RTs into the kappa.
      * during an active cliff -> DO NOT catch the knife (block new longs)
      * post-crash floor / grind-up -> fade LONG with recovery asymmetry
      * normal dip below ref -> fade LONG (maker-first entry)

Every position is closed with a market order to **realize** PnL.

Run (local proxy test):
  python MeanReversionAgent.py --port 8902 --agent_id 0

Tune strategy here (pm2 restart picks up changes; no .env / --agent.params needed):
"""

import gc
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

# ---------------------------------------------------------------------------
# Strategy constants — edit here, then: pm2 restart miner-1 miner-2 miner-3
# ---------------------------------------------------------------------------

# --- position size ---
# Small clip, matched to the top single-mode makers (~0.30 BASE). A big clip does NOT help
# kappa (MAD-normalizes magnitude away) but it DOES hurt: market exits walk the book, so a
# large clip realizes meaningfully worse than mid (slippage drags realized PnL -> drags kappa),
# and it multiplies inventory risk if many books dip at once and never revert. Keep it tiny.
QUOTE_NOTIONAL = 105.0          # QUOTE per entry -> ~0.35 BASE at price 300 (was 1800 = 6.0 BASE)
MIN_ORDER_SIZE = 0.25           # smallest BASE order the sim accepts on a book
PING_LOT_SIZE = 0.26            # activity backstop open/close size (slightly above min lot)

# --- entry signal: "ref" + band ---
MEAN_WINDOW_S = 300.0           # seconds of trade prints for rolling mean (ref); matches dashboard blue 5m line
MIN_SAMPLES = 8                 # min prints in window before ref/band are valid
K_ENTRY = 1.5                   # entry band = K_ENTRY × volatility
MIN_BAND_BPS = 8.0              # never enter on less than this stretch vs ref (bps)
MAX_BAND_BPS = 120.0            # never require more than this stretch to enter (bps cap)
IMBALANCE_DEPTH = 5             # LOB levels used for bid/ask depth imbalance
IMB_GATE = 0.30                 # skip long if heavy sell pressure (imb < -gate)

# --- exit rules (from entry price, not ref) ---
TP_BPS = 12.0                   # take profit when open PnL reaches +12 bps
SL_BPS = 16.0                   # stop loss when open PnL reaches -16 bps
MAX_HOLD_S = 180.0              # close anyway after this many seconds (tape-tuned; was 210)
COOLDOWN_S = 30.0               # after a stop on a book, pause new entries there for N seconds

# --- trend filter (slow direction; blocks fading into strong trends) ---
TREND_WINDOW_S = 600.0          # EMA lookback for slow trend (seconds); should be > MEAN_WINDOW_S
TREND_GATE_BPS = 25.0           # if ref vs EMA exceeds ±25 bps, treat book as trending

# --- crash / recovery (sharp dump then slow grind up — tape asymmetry) ---
CRASH_BPS = 35.0                # drop from recent high over CRASH_WINDOW_S that flags a "cliff"
CRASH_WINDOW_S = 20.0           # lookback (seconds) for measuring that cliff drop
RECOVERY_WINDOW_S = 300.0       # after a cliff, treat book as "recovery" for this long (favor careful longs)
RECOVERY_TP_MULT = 1.8          # post-crash longs: multiply TP by this (recovery is slow, let winners run)
RECOVERY_HOLD_MULT = 2.0        # post-crash longs: multiply MAX_HOLD_S by this
GRIND_DUMP_WINDOW_S = 300.0      # slow-cliff lookback (matches mean window); catches multi-minute dumps
GRIND_DUMP_BPS = 38.0           # drop from GRIND_DUMP_WINDOW high → post-dump recovery (tape-tuned)
GRIND_RISE_MIN_BPS = 8.0        # off the grind floor by at least this much before grind-long
GRIND_LONG_MAX_DEV_BPS = 18.0   # in grind-up, long while fair only this much above ref (not chasing spikes)

MID_DEQUE_MAXLEN = 360          # mid history for slow dump / grind windows (~1 pt/s × 300s)

# --- knife (block longs into an active dump) ---
KNIFE_MIN_DROP_BPS = 20.0       # knife only after price has already dumped this much from the 20s high
KNIFE_STEP_BPS = 8.0            # …and this step still falls ≥8 bps (falling knife → block longs)
KNIFE_BLOCK_S = 8.0             # how long to block new longs after a knife trigger

# --- activity backstop (separate from fade entries; guarantees the book stays scored) ---
# The validator needs >= min_realized_observations (3) completed round-trips per book inside
# the kappa lookback (3h) for that book to receive a kappa at all; below 3 the book is dropped
# (kappa=None). The activity FACTOR decays very gently when idle (base ~0.9999 per 5s scoring
# tick, impact=0), so a ~25-min gap only costs ~2% on a book's factor -- there is NO need to
# churn every 10 min. We therefore only ping a flat book when it has gone PING_INTERVAL_S
# without a round-trip, which yields ~7 RTs/3h (>2x the floor of 3) while keeping the factor
# ~0.98+. Fewer forced market RTs also means fewer small fee-losses dragging the cubic kappa.
PING_INTERVAL_S = 1500.0        # 25 min: >=3 RTs/3h with margin, NOT 10-min churn (was 480)
PING_SUBMIT_COOLDOWN_S = 5.0    # min gap between ping orders on the same book

# --- order routing ---
ENTRY_EXPIRY_S = 8.0            # maker limit entries expire after N seconds (GTT)
ENTRY_REPOST_GAP_S = 8.0        # don't re-post a fade bid within this gap (anti-stack: one
                                # resting bid at a time, matches the GTT lifetime)
MAX_TAKER_FEE = 0.0             # only market/taker when fee ≤ this (0 = rebate or zero fee only)

# --- volume cap (hard constraint; volume does not boost score today) ---
CAPITAL_TURNOVER_CAP = 10.0     # max quote turnover = this × miner_wealth before stopping new entries
VOLUME_SAFETY = 0.5             # use only this fraction of cap (stay well under limit)
VOLUME_ASSESSMENT_NS = 86_400_000_000_000  # rolling 24h window for traded volume tally (nanoseconds)

PRICES_DEQUE_MAXLEN = 3000      # max stored trade prints per book (covers MEAN_WINDOW_S on busy books)


@dataclass
class _Position:
    """Per-book net position reconstructed from our own fills."""
    qty: float = 0.0          # signed BASE (>0 long, <0 short)
    avg: float = 0.0          # volume-weighted average entry price
    entry_ts: int = 0         # sim timestamp (ns) current exposure opened
    post_crash: bool = False  # entered during a post-crash recovery window (set on FILL)
    via_ping: bool = False    # opened by the activity backstop (closed by ping, not TP/SL)


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
    seen_ns: int = 0          # sim time (ns) this book was first handled (cold-start anchor)
    ping_phase_ns: int = 0    # per-book phase offset so books don't all ping on the same tick
    last_rt_ns: int = 0           # sim time (ns) of last completed round-trip
    last_ping_submit_ns: int = 0  # sim time (ns) of last ping order submit (anti-burst)
    last_entry_ns: int = 0        # sim time (ns) of last fade-entry submit (anti-stack)
    ping_awaiting_open: bool = False  # open ping leg in flight; block fade until fill/timeout
    vol_log: list = field(default_factory=list)  # (ts, quote_volume) round-trip cost


class MeanReversionAgent(FinanceSimulationAgent):
    # ------------------------------------------------------------------ setup
    def initialize(self) -> None:
        bt.logging.set_info()

        self.quote_notional = QUOTE_NOTIONAL
        self.min_order_size = MIN_ORDER_SIZE
        self.ping_lot_size = PING_LOT_SIZE
        self.mean_window_s = MEAN_WINDOW_S
        self.min_samples = MIN_SAMPLES
        self.k_entry = K_ENTRY
        self.min_band_bps = MIN_BAND_BPS
        self.max_band_bps = MAX_BAND_BPS
        self.imbalance_depth = IMBALANCE_DEPTH
        self.imb_gate = IMB_GATE
        self.tp_bps = TP_BPS
        self.sl_bps = SL_BPS
        self.max_hold_s = MAX_HOLD_S
        self.cooldown_s = COOLDOWN_S
        self.trend_window_s = TREND_WINDOW_S
        self.trend_gate_bps = TREND_GATE_BPS
        self.crash_bps = CRASH_BPS
        self.crash_window_s = CRASH_WINDOW_S
        self.recovery_window_s = RECOVERY_WINDOW_S
        self.recovery_tp_mult = RECOVERY_TP_MULT
        self.recovery_hold_mult = RECOVERY_HOLD_MULT
        self.knife_min_drop_bps = KNIFE_MIN_DROP_BPS
        self.knife_block_s = KNIFE_BLOCK_S
        self.knife_step_bps = KNIFE_STEP_BPS
        self.grind_dump_window_s = GRIND_DUMP_WINDOW_S
        self.grind_dump_bps = GRIND_DUMP_BPS
        self.grind_rise_min_bps = GRIND_RISE_MIN_BPS
        self.grind_long_max_dev_bps = GRIND_LONG_MAX_DEV_BPS
        self.ping_interval_s = PING_INTERVAL_S
        self.ping_submit_cooldown_s = PING_SUBMIT_COOLDOWN_S
        self.ping_interval_ns = int(self.ping_interval_s * 1e9)
        self.ping_submit_cooldown_ns = int(self.ping_submit_cooldown_s * 1e9)
        self.entry_expiry_s = ENTRY_EXPIRY_S
        self.entry_repost_gap_ns = int(ENTRY_REPOST_GAP_S * 1e9)
        self.max_taker_fee = MAX_TAKER_FEE
        self.turnover_cap = CAPITAL_TURNOVER_CAP
        self.volume_safety = VOLUME_SAFETY
        self.volume_assessment_ns = VOLUME_ASSESSMENT_NS

        # Never read self.history — skip framework deep-copy of the full 128-book state each tick.
        self.history_len = 0
        self._tune_gc()

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
        self._step_ts_ns: int = 0
        self._cur_validator: str | None = None  # validator currently being processed in update()


        bt.logging.info(
            f"[MeanReversion uid={self.uid}] notional={self.quote_notional} "
            f"tp={self.tp_bps}bps sl={self.sl_bps}bps hold={self.max_hold_s}s "
            f"k_entry={self.k_entry:.2f} mean={self.mean_window_s}s "
            f"crash={self.crash_bps:.1f}bps/{self.crash_window_s}s "
            f"grind={self.grind_dump_bps:.1f}bps/{self.grind_dump_window_s}s "
            f"recovery={self.recovery_window_s}s tp_mult={self.recovery_tp_mult} "
            f"ping={self.ping_interval_s}s long_only"
        )

    def _tune_gc(self) -> None:
        """Reduce GC pauses during handle() — same approach as AdaptiveRouterAgent."""
        try:
            gc.collect()
            gc.freeze()
            gc.set_threshold(50_000, 500, 500)
            bt.logging.info(
                f"[MeanReversion uid={self.uid}] gc tuned: frozen={gc.get_freeze_count()} "
                f"thresholds={gc.get_threshold()} history_len=0"
            )
        except Exception as ex:
            bt.logging.warning(f"[MeanReversion uid={self.uid}] gc tune skipped: {ex}")

    # --------------------------------------------------------------- lifecycle
    def _reset_validator(self, validator: str | None) -> None:
        """Drop all per-book state for ONE validator (new simulation). Other validators'
        live positions are left untouched -- critical when serving multiple validators."""
        if validator is None:
            return
        self.positions.pop(validator, None)
        self.books_state.pop(validator, None)

    def onStart(self, event: SimulationStartEvent) -> None:
        # onStart is dispatched from inside update() while processing ONE validator's state,
        # but the event carries no validator id. Use the validator update() stamped for us and
        # reset only that one -- never a global wipe (that would clear other validators' books).
        validator = self._cur_validator
        self._reset_validator(validator)
        if validator is not None:
            # Force respond()'s sim_id guard to re-arm for this validator on the next tick.
            self._sim_id.pop(validator, None)
        bt.logging.info(
            f"[MeanReversion uid={self.uid}] simulation start: reset state for validator={validator}"
        )

    def update(self, state: MarketSimulationStateUpdate) -> None:
        # Stamp the current sim time + validator so onStart/onTrade (dispatched inside
        # super().update()) act on the correct validator and on sim-time.
        self._step_ts_ns = int(state.timestamp)
        self._cur_validator = state.dendrite.hotkey
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

    def _prune_vol_log(self, st: _BookState, now_ns: int) -> None:
        cutoff = now_ns - self.volume_assessment_ns
        st.vol_log = [(t, v) for t, v in st.vol_log if t >= cutoff]

    def _rolled_quote_volume(self, validator, book_id, now_ns) -> float:
        st = self._bstate(validator, book_id)
        self._prune_vol_log(st, now_ns)
        return sum(v for _, v in st.vol_log)

    def _apply_fill(self, validator, book_id, direction, qty, price, ts) -> None:
        pos = self._book_positions(validator).setdefault(book_id, _Position())
        st = self._bstate(validator, book_id)
        signed = qty if direction == OrderDirection.BUY else -qty
        prev = pos.qty
        entry_avg = pos.avg
        entry_ts = pos.entry_ts
        if prev == 0 or (prev > 0) == (signed > 0):
            # Open or add in the same direction -> blend the average.
            opening = prev == 0
            was_ping = st.ping_awaiting_open  # capture BEFORE we clear it below
            total = abs(prev) + qty
            pos.avg = (pos.avg * abs(prev) + price * qty) / total if total > 0 else price
            pos.qty = prev + signed
            if opening:
                pos.entry_ts = ts
                # Bug1 fix: classify the position at FILL time, not at submit time (an unfilled
                # resting bid must never leave post_crash stuck on a later, unrelated entry).
                pos.post_crash = ts < st.crash_until
                # Bug2 fix: remember whether this open was the activity-ping leg, so the ping
                # closer only ever touches ping lots -- never a real fade position.
                pos.via_ping = bool(was_ping)
            if direction == OrderDirection.BUY:
                st.ping_awaiting_open = False
        else:
            # Reduce / close / flip -> realize a round-trip on the closed amount.
            closed_qty = min(qty, abs(prev))
            if closed_qty >= self.min_order_size / 2 and entry_avg > 0:
                st.last_rt_ns = ts
            pos.qty = prev + signed
            if abs(pos.qty) < 1e-12:
                pos.qty, pos.avg, pos.entry_ts = 0.0, 0.0, 0
                pos.post_crash, pos.via_ping = False, False
            elif (prev > 0) != (pos.qty > 0):
                pos.avg, pos.entry_ts = price, ts
                pos.post_crash, pos.via_ping = False, False

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
        bq = 0.0
        aq = 0.0
        for i, level in enumerate(book.bids):
            if i >= self.imbalance_depth:
                break
            bq += level.quantity
        for i, level in enumerate(book.asks):
            if i >= self.imbalance_depth:
                break
            aq += level.quantity
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
        self._prune_vol_log(st, now)

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

        # Authoritative per-validator reset: if this validator's simulation_id changed (new sim,
        # possibly without us seeing the ESS event), wipe ONLY this validator's state.
        if self._sim_id.get(validator) != cfg.simulation_id:
            self._reset_validator(validator)
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
                    f"[MeanReversion uid={self.uid}] book {book_id} error: {ex}\n"
                    f"{traceback.format_exc()}"
                )
        return response

    def _reconcile_position(self, account, pos, vol_dp) -> None:
        """Clamp tracked qty to free base so we never oversell on ping/exit."""
        if account.base_balance is None:
            return
        free = account.base_balance.free
        if pos.qty > 0:
            if free < self.min_order_size / 2:
                pos.qty, pos.avg, pos.entry_ts = 0.0, 0.0, 0
                pos.post_crash, pos.via_ping = False, False
            elif free < pos.qty - self.min_order_size / 4:
                pos.qty = round(free, vol_dp)

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
        pos = self._book_positions(validator).setdefault(book_id, _Position())
        self._reconcile_position(account, pos, vol_dp)

        if st.seen_ns == 0:
            # Cold-start anchor: a freshly-seen book is NOT immediately ping-due (prevents a
            # 128-book ping storm at sim start/refresh). A per-book phase (0..ping_interval)
            # also staggers the very first ping so quiet books don't all fire on one tick.
            st.seen_ns = now
            st.ping_phase_ns = ((book_id * 2654435761) % 1000) * self.ping_interval_ns // 1000

        # Flatten accidental shorts (should not happen on a long-only agent).
        if pos.qty < -self.min_order_size / 2:
            st.last_ping_submit_ns = now
            self._flatten(response, account, book_id, pos, vol_dp)
            return

        # ---- fast path: open long — only mid + timestamps (skip rolling features) ----
        if pos.qty >= self.min_order_size / 2 and pos.avg > 0:
            if self._ping_close_due(st, pos, now):
                if self._submit_ping_close(response, validator, book_id, account, pos, st,
                                           vol_dp, now):
                    return

            # Exit PnL on mid — matches the chart and avoids microprice/spread false stops.
            pnl_bps = (mid - pos.avg) / pos.avg * 1e4
            tp = self.tp_bps
            sl = self.sl_bps
            hold_ns = self.max_hold_s * 1e9
            if pos.post_crash:
                tp *= self.recovery_tp_mult
                hold_ns *= self.recovery_hold_mult
            timed_out = (now - pos.entry_ts) >= hold_ns if pos.entry_ts else False
            if pnl_bps >= tp:
                self._flatten(response, account, book_id, pos, vol_dp)
            elif pnl_bps <= -sl:
                st.cooldown_until = now + int(self.cooldown_s * 1e9)
                self._flatten(response, account, book_id, pos, vol_dp)
            elif timed_out:
                self._flatten(response, account, book_id, pos, vol_dp)
            return

        # ---- flat: ping open before feature ingest (activity leg needs no ref/band) ----
        if self._ping_open_due(st, pos.qty, now):
            self._submit_ping_open(response, validator, book_id, account, st, book, mid, now)
            return

        self._ingest(st, book, mid, now)
        ref, band_bps = self._ref_and_band(st)
        imb = self._book_imbalance(book)

        # --- crash / knife state (asymmetric §3B.5 handling) ---
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

        if ref is None or now < st.cooldown_until or st.ping_awaiting_open:
            return

        if self._rolled_quote_volume(validator, book_id, now) >= cap:
            return

        dev_bps = (fair - ref) / ref * 1e4    # >0 stretched above ref, <0 below
        fade_long = False
        grind_long = False

        if dev_bps <= -band_bps:
            # Over-extended DOWN -> buy the dip, expecting reversion up.
            # But not while the knife is still falling, and not in a downtrend
            # unless this is a post-crash recovery floor (then we *want* the dip).
            if not knife_active and imb >= -self.imb_gate:
                if in_recovery or not downtrend:
                    fade_long = True
        elif (in_recovery and mid > st.ema_long and grind_rise_bps >= self.grind_rise_min_bps
              and dev_bps <= self.grind_long_max_dev_bps):
            # Post-dump slow grind up: price may sit above ref — ride the rise, don't fade it.
            if not knife_active and imb >= -self.imb_gate:
                grind_long = True
                fade_long = True

        if not fade_long:
            return

        qty = round(self.quote_notional / mid, vol_dp)
        if qty < self.min_order_size:
            return

        # Anti-stack: a maker entry rests up to ENTRY_EXPIRY_S (GTT). Don't pile on another
        # bid while the previous one is still live, or we burn the 5-instruction/book budget
        # and build more inventory than one intended clip.
        if st.last_entry_ns and (now - st.last_entry_ns) < self.entry_repost_gap_ns:
            return

        take_ok = self._taker_allowed(account)
        # Bug4 fix: only arm the anti-stack timer if an order is actually submitted (an entry
        # aborted on insufficient balance must not block the book for ENTRY_REPOST_GAP_S).
        if self._enter(response, account, book_id, qty, book, price_dp, take_ok):
            st.last_entry_ns = now


    # ----------------------------------------------------------- activity ping
    def _rt_due(self, st: _BookState, now: int) -> bool:
        """True when this book needs a round-trip for the activity window. Before the first RT
        we anchor on seen_ns (+ per-book phase) so a freshly-seen book is not instantly due --
        this prevents the cold-start / sim-refresh ping storm across all 128 books."""
        if st.last_rt_ns == 0:
            anchor = st.seen_ns if st.seen_ns else now
            return (now - anchor) >= (self.ping_interval_ns + st.ping_phase_ns)
        return (now - st.last_rt_ns) >= self.ping_interval_ns

    def _submit_cooldown_ok(self, st: _BookState, now: int) -> bool:
        return (st.last_ping_submit_ns == 0) or (
            (now - st.last_ping_submit_ns) >= self.ping_submit_cooldown_ns)

    def _ping_open_due(self, st: _BookState, inv_qty: float, now: int) -> bool:
        """Flat book: open leg of a mandatory activity round-trip."""
        if st.ping_awaiting_open:
            if (now - st.last_ping_submit_ns) > int(60e9):
                st.ping_awaiting_open = False
            else:
                return False
        return (inv_qty < self.min_order_size / 2 and self._rt_due(st, now)
                and self._submit_cooldown_ok(st, now))

    def _ping_close_due(self, st: _BookState, pos: _Position, now: int) -> bool:
        """Close leg of a mandatory activity round-trip. ONLY for a ping-opened lot -- a real
        fade position is never sliced by the ping; it exits via TP/SL/time (which also records
        the round-trip), so fade books stay active without ping interference."""
        return (pos.via_ping and pos.qty >= self.ping_lot_size / 2
                and self._rt_due(st, now) and self._submit_cooldown_ok(st, now))

    def _submit_ping_open(self, response, validator, book_id, account, st, book,
                          mid, now) -> None:
        """Market-buy the ping lot (open leg). Separate from fade entries."""
        qty = self.ping_lot_size
        ask_px = book.asks[0].price if book.asks else mid
        if account.quote_balance.free < qty * ask_px:
            return
        st.last_ping_submit_ns = now
        st.ping_awaiting_open = True
        response.market_order(book_id=book_id, direction=OrderDirection.BUY,
                              quantity=qty, currency=OrderCurrency.BASE,
                              stp=STP.CANCEL_OLDEST)

    def _submit_ping_close(self, response, validator, book_id, account, pos, st,
                           vol_dp, now) -> bool:
        """Market-sell the WHOLE ping lot (it is a single min lot) to complete the round-trip.
        Closing the full lot avoids leaving an unmanaged dust remainder."""
        if pos.qty <= 0:
            return False
        free = account.base_balance.free if account.base_balance else pos.qty
        q = round(min(pos.qty, free), vol_dp)
        if q < self.min_order_size / 2:
            return False
        st.last_ping_submit_ns = now
        self._market_close_slice(response, account, book_id, pos, q, vol_dp)
        return True

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

    def _enter(self, response, account, book_id, qty, book, price_dp, take_ok) -> bool:
        """Maker-first long entry; taker fallback only when the fee regime pays takers.
        Returns True iff an order was actually submitted. post_crash / via_ping are classified
        on the resulting FILL (see _apply_fill), never here at submit time."""
        expiry_ns = int(self.entry_expiry_s * 1e9)
        price = round(book.bids[0].price, price_dp)
        if account.quote_balance.free < qty * (book.asks[0].price or price):
            return False
        if take_ok:
            response.market_order(book_id=book_id, direction=OrderDirection.BUY, quantity=qty,
                                  currency=OrderCurrency.BASE, stp=STP.CANCEL_OLDEST)
        else:
            response.limit_order(book_id=book_id, direction=OrderDirection.BUY, quantity=qty,
                                 price=price, postOnly=True, timeInForce=TimeInForce.GTT,
                                 expiryPeriod=expiry_ns, stp=STP.CANCEL_OLDEST)
        return True

    def _market_close_slice(self, response, account, book_id, pos, qty, vol_dp) -> None:
        """Market-close up to `qty` BASE of a long (partial allowed for activity ping)."""
        if pos.qty <= 0:
            return
        free = account.base_balance.free if account.base_balance else 0.0
        q = round(min(qty, pos.qty, free), vol_dp)
        if q < self.min_order_size:
            return
        response.market_order(book_id=book_id, direction=OrderDirection.SELL,
                              quantity=q, currency=OrderCurrency.BASE,
                              stp=STP.CANCEL_OLDEST)

    def _flatten(self, response, account, book_id, pos, vol_dp) -> None:
        """Market-close the full tracked position (long or accidental short)."""
        if pos.qty > 0:
            self._market_close_slice(response, account, book_id, pos, pos.qty, vol_dp)
        elif pos.qty < 0:
            qty = round(-pos.qty, vol_dp)
            if qty < self.min_order_size:
                return
            response.market_order(book_id=book_id, direction=OrderDirection.BUY,
                                  quantity=qty, currency=OrderCurrency.BASE,
                                  stp=STP.CANCEL_OLDEST)



if __name__ == "__main__":
    launch(MeanReversionAgent)
