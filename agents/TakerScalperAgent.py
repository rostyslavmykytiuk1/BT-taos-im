"""
TakerScalperAgent — pure-taker round-trip + activity engine for subnet 79.

Market orders only: they fill fully and immediately and leave nothing resting, so there
are no partial-fill quotes to manage. One position per book, strictly sequential.

Rules:
  1. Activity 1.0: every book completes >=1 RT per ~570s window. No forced RT during the
     first ~570s after agent/sim start; after that, force once ~500s have passed since the
     last RT (agent start time stands in for last RT until the first close).
  2. One position per book: a book is FLAT xor HELD; the open path runs only when flat.
  3. Wait for full fill: after any submit, st.pending_ns blocks the book until a fill
     resolves it (timeout-recovered).
  4. Kappa-3 + PnL: open a directional scalp only when the kappa gate clears; close on
     TP / SL / max-hold.

Per book each tick:
  reconcile -> pending? wait : held? close(TP/SL/time) : open(kappa | activity | idle)

RT logging -> main validator only.
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
)

_NS = 1_000_000_000

# Exchange floor for any BASE order (sim `minOrderSize`, not exposed to agents).
EXCHANGE_MIN_ORDER_SIZE = 0.25
# Entry lot, comfortably above the floor so a fill is always a sellable holding.
LOT = 0.3

# Hold / exit.
MIN_HOLD_S = 1.5
MAX_HOLD_S = 4.0
MIN_GROSS_TP_BPS = 2.5
MAX_GROSS_SL_BPS = 4.0

# Margin leverage for a SELL open when free base is insufficient.
SHORT_LEVERAGE = 1.0

RT_WINDOW_S = 570.0                # validator activity sampling window (~10 min)
RT_MAX = 30                        # max profit RTs per book per window

# Force a taker RT once this long since the last RT (kept under RT_WINDOW_S).
ACTIVITY_DEADLINE_S = 500.0
# Min gap between RT closes and the next profit open (per book; throttles churn).
MIN_REOPEN_GAP_S = 3.0
# After submitting, wait this long for the fill before assuming the order was lost.
PENDING_TIMEOUT_S = 5.0

# Kappa-3 (3h history for score projection).
KAPPA_TAU = 0.0
KAPPA_MIN_OBS = 3
KAPPA_MIN_LOOKBACK_S = 5400.0      # 90 min
KAPPA_RT_HISTORY_S = 10_800.0      # 3h
KAPPA_PROJ_IMPROVE = 0.003
KAPPA_PROJ_TOLERANCE = 0.008
KAPPA_EST_PNL_FLOOR = 0.03         # min projected RT edge to spend a kappa taker
KAPPA_MIN_REBATE_BPS = 1.0         # require taker rebate >= this (rate <= -1bp)

# Volume cap (kappa path only).
CAPITAL_TURNOVER_CAP = 10.0
VOLUME_SAFETY = 0.8
VOLUME_ASSESSMENT_NS = 86_400_000_000_000

# RT logs only for the scoring validator.
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
    pending_ns: int = 0                 # last order submit ts (rule 3: wait for the fill)
    pending_kind: str = ""              # tag of the in-flight open (for RT logging)
    rt_events: list[tuple[int, float]] = field(default_factory=list)
    kappa3: float | None = None
    vol_log: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class _RtLogCtx:
    """Open snapshot stashed at submit; close_reason set at close; logged at the RT fill."""
    open_reason: str = "?"
    side: str = "?"
    open_rt_window_n: int = 0
    open_rt_pnl_list: str = "[]"
    est_pnl: float | None = None
    kappa_at_open: float | None = None
    kappa_proj: float | None = None
    taker_bps: float | None = None
    close_reason: str = "fill"


class TakerScalperAgent(FinanceSimulationAgent):
    def initialize(self) -> None:
        bt.logging.set_info()

        self.min_order_size = LOT
        self._min_qty = LOT / 2             # RT-log gate (meaningful close)
        self.exch_min = EXCHANGE_MIN_ORDER_SIZE
        self._flat_eps = 0.5 * 10 ** (-4)  # overwritten by _sync_order_size on first respond
        self._volume_decimals: int | None = None
        self.volume_assessment_ns = VOLUME_ASSESSMENT_NS

        jitter = ((self.uid * 2654435761) % 1000) / 1000.0
        max_hold_s = MAX_HOLD_S * (0.92 + 0.16 * jitter)
        activity_s = ACTIVITY_DEADLINE_S * (0.92 + 0.08 * jitter)   # ~460-500s

        self.min_hold_ns = int(MIN_HOLD_S * _NS)
        self.max_hold_ns = int(max_hold_s * _NS)
        self.min_reopen_gap_ns = int(MIN_REOPEN_GAP_S * (0.9 + 0.2 * jitter) * _NS)
        self.rt_window_ns = int(RT_WINDOW_S * _NS)
        self.activity_deadline_ns = int(activity_s * _NS)
        self.pending_timeout_ns = int(PENDING_TIMEOUT_S * _NS)
        self.kappa_rt_history_ns = int(KAPPA_RT_HISTORY_S * _NS)
        self.kappa_min_lookback_ns = int(KAPPA_MIN_LOOKBACK_S * _NS)
        self.kappa_min_rebate_rate = -KAPPA_MIN_REBATE_BPS / 1e4

        self.positions: dict[str, dict[int, _Position]] = {}
        self.books_state: dict[str, dict[int, _BookState]] = {}
        self._sim_id: dict[str, str] = {}
        self._rt_log: dict[tuple[str, int], _RtLogCtx] = {}
        self._step_ts_ns: int = 0
        self._agent_start_ns: int = 0     # activity clock before the first RT (per sim)
        self._active_validator: str | None = None

        bt.logging.info(
            f"[TakerScalper uid={self.uid}] PURE-TAKER lot={LOT} exch_min={self.exch_min} "
            f"hold={MIN_HOLD_S}-{max_hold_s:.1f}s tp={MIN_GROSS_TP_BPS}bps sl={MAX_GROSS_SL_BPS}bps "
            f"reopen_gap={MIN_REOPEN_GAP_S}s rt_window={RT_WINDOW_S / 60:.0f}m max={RT_MAX} "
            f"activity_deadline={activity_s:.0f}s "
            f"kappa_gate(est>={KAPPA_EST_PNL_FLOOR},rebate>={KAPPA_MIN_REBATE_BPS}bps) "
            f"rt_log={MAIN_VALIDATOR[:8]}"
        )

    def update(self, state: MarketSimulationStateUpdate) -> None:
        self._step_ts_ns = int(state.timestamp)
        self._active_validator = state.dendrite.hotkey
        # Reset before super().update() so the new sim's first fills don't hit stale state.
        self._ensure_simulation(self._active_validator, state.config.simulation_id)
        if self._agent_start_ns == 0 and self._step_ts_ns > 0:
            self._agent_start_ns = self._step_ts_ns
        super().update(state)

    def _ensure_simulation(self, validator: str, simulation_id: str | None) -> None:
        """Drop per-validator state when the validator starts a new simulation."""
        if self._sim_id.get(validator) == simulation_id:
            return
        self._book_positions(validator).clear()
        self.books_state.pop(validator, None)
        self._rt_log = {k: v for k, v in self._rt_log.items() if k[0] != validator}
        self._agent_start_ns = 0
        if simulation_id is not None:
            self._sim_id[validator] = simulation_id
        else:
            self._sim_id.pop(validator, None)
        bt.logging.info(
            f"[TakerScalper uid={self.uid}] new simulation: {validator[:8]} sim_id={simulation_id}"
        )

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        validator = state.dendrite.hotkey
        cfg = self.simulation_config
        self._sync_order_size(cfg.volumeDecimals)

        vol_dp = cfg.volumeDecimals
        volume_cap = CAPITAL_TURNOVER_CAP * cfg.miner_wealth * VOLUME_SAFETY
        now = state.timestamp

        for book_id in sorted(self.accounts.keys()):
            book = state.books.get(book_id)
            account = self.accounts.get(book_id) if book else None
            if book is None or account is None:
                continue
            try:
                self._step_book(response, validator, book_id, book, account, vol_dp, volume_cap, now)
            except Exception as ex:
                bt.logging.warning(f"[TakerScalper uid={self.uid}] step {book_id}: {ex}")

        return response

    def _step_book(
        self, response, validator: str, book_id: int, book, account,
        vol_dp: int, volume_cap: float, now: int,
    ) -> None:
        """One sequential action per book: wait while an order is in flight, else close the
        held position or, when flat, decide whether to open."""
        pos = self._book_positions(validator).setdefault(book_id, _Position())
        self._reconcile_position(account, pos, vol_dp)

        st = self._bstate(validator, book_id)
        if self._agent_start_ns == 0 and now > 0:
            self._agent_start_ns = now
        if self._prune_rt_events(st, now):
            self._refresh_book_kappa(validator, book_id, now)

        # Rule 3: one order at a time -> wait for the fill (or a timeout) before acting.
        if st.pending_ns and (now - st.pending_ns) < self.pending_timeout_ns:
            return
        st.pending_ns = 0   # presumed filled / lost; re-derive from the position

        if abs(pos.qty) >= self._flat_eps:
            self._close(response, validator, book_id, book, account, pos, vol_dp, now)
        else:
            self._open(response, validator, book_id, book, account, volume_cap, now)

    # ------------------------------------------------------------------ events
    def onOrderRejected(self, event: OrderPlacementEvent) -> None:
        if event.bookId is None or not self._active_validator:
            return
        validator = self._active_validator
        st = self._bstate(validator, event.bookId)
        st.pending_ns = 0
        st.pending_kind = ""
        self._rt_log.pop((validator, event.bookId), None)

    def onTrade(self, event: TradeEvent, validator: str | None = None) -> None:
        """Route our taker fills into position accounting (we only ever take)."""
        if event.bookId is None or self.uid != event.takerAgentId:
            return
        validator = validator or self._active_validator
        if validator is None:
            return
        ts_ns = int(event.timestamp) if event.timestamp else self._step_ts_ns
        self._record_trade_volume(validator, event.bookId, event.quantity, event.price, ts_ns)
        self._apply_fill(
            validator, event.bookId, event.side, event.quantity, event.price, event.takerFee, ts_ns,
        )

    # ------------------------------------------------------------------ state
    def _sync_order_size(self, volume_decimals: int) -> None:
        if volume_decimals == self._volume_decimals:
            return
        self._volume_decimals = volume_decimals
        lot = round(max(LOT, 10 ** (-volume_decimals)), volume_decimals)
        self.min_order_size = lot
        self._min_qty = lot / 2
        self.exch_min = max(EXCHANGE_MIN_ORDER_SIZE, 10 ** (-volume_decimals))
        # Half a volume tick: below this a holding is rounding noise, treat as flat.
        self._flat_eps = 0.5 * 10 ** (-volume_decimals)
        bt.logging.info(
            f"[TakerScalper uid={self.uid}] volumeDecimals={volume_decimals} "
            f"lot={lot} exch_min={self.exch_min}"
        )

    def _book_positions(self, validator: str) -> dict[int, _Position]:
        return self.positions.setdefault(validator, {})

    def _bstate(self, validator: str, book_id: int) -> _BookState:
        return self.books_state.setdefault(validator, {}).setdefault(book_id, _BookState())

    @staticmethod
    def _clear_position(pos: _Position) -> None:
        pos.qty = pos.avg = pos.entry_fee = 0.0
        pos.entry_ts = 0

    @staticmethod
    def _side_label(direction: OrderDirection) -> str:
        return "long" if direction == OrderDirection.BUY else "short"

    def _activity_force_due(self, st: _BookState, now: int) -> bool:
        """True when a forced RT is needed to stay inside the activity window.

        Before the first RT on a book, agent/sim start time is the reference clock.
        The first full RT_WINDOW after start is grace — kappa can trade without a
        backstop. After that, force once ACTIVITY_DEADLINE has passed since the
        last RT (or since start if still none).
        """
        if self._agent_start_ns <= 0:
            return False
        if st.last_rt_ns == 0 and (now - self._agent_start_ns) < self.rt_window_ns:
            return False
        ref = st.last_rt_ns if st.last_rt_ns > 0 else self._agent_start_ns
        return (now - ref) >= self.activity_deadline_ns

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
        """Update the per-book position from a fill. Opening blends the entry; a reducing
        fill realizes a clean per-RT PnL, records it for kappa, and logs the round-trip."""
        pos = self._book_positions(validator).setdefault(book_id, _Position())
        st = self._bstate(validator, book_id)
        st.pending_ns = 0   # a fill resolved the in-flight order

        signed = qty if direction == OrderDirection.BUY else -qty
        prev = pos.qty
        entry_avg = pos.avg
        opening = abs(prev) < self._flat_eps   # sub-eps prior holding -> re-anchor cleanly

        # Same side (or opening from flat): grow the position, blend the entry price.
        if opening or (prev > 0) == (signed > 0):
            base = 0.0 if opening else abs(prev)
            total = base + qty
            pos.avg = (pos.avg * base + price * qty) / total if total > 0 else price
            pos.qty = (0.0 if opening else prev) + signed
            if opening:
                pos.entry_ts = ts
                pos.entry_fee = trade_fee
                self._ensure_rt_open_ctx(validator, book_id, direction, ts)
            else:
                pos.entry_fee += trade_fee
            return

        closed_qty = min(qty, abs(prev))
        new_qty = prev + signed
        final = abs(new_qty) < self._flat_eps
        if closed_qty >= self._min_qty and entry_avg > 0:
            rpnl = (price - entry_avg) * closed_qty if prev > 0 else (entry_avg - price) * closed_qty
            open_fee, close_fee = self._allocate_close_fees(
                pos.entry_fee, abs(prev), closed_qty, qty, trade_fee,
            )
            net_pnl = rpnl - open_fee - close_fee
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
                final=final,
            )
            pos.entry_fee -= open_fee
        elif final:
            # Closed, but this last leg was too small to log; drop the lingering context.
            self._rt_log.pop((validator, book_id), None)

        pos.qty = new_qty
        if abs(pos.qty) < self._flat_eps:
            self._clear_position(pos)
        elif (prev > 0) != (pos.qty > 0):
            # Net flip (should not occur with flatten-only exits): re-anchor the leg.
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
        self, validator: str, book_id: int, now: int, extra: tuple[int, float] | None = None,
    ) -> list[float]:
        timestamps = self._global_rt_timestamps(validator, now)
        if extra is not None and extra[0] not in timestamps:
            timestamps = sorted(timestamps + [extra[0]])
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
        if not pnl_series or sum(1 for x in pnl_series if x != 0.0) < KAPPA_MIN_OBS:
            return None

        med = cls._median(pnl_series)
        mad = max(cls._median([abs(x - med) for x in pnl_series]), 1e-6)
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
        bid, ask = book.bids[0].price, book.asks[0].price
        if bid <= 0 or ask <= 0:
            return 0.0
        return (bid - ask) * qty - taker_rate * (ask + bid) * qty

    def _project_kappa(
        self, validator: str, book_id: int, now: int, estimated_pnl: float,
    ) -> float | None:
        close_ts = now + self.min_hold_ns
        return self._kappa3_raw(
            self._book_pnl_series(validator, book_id, now, extra=(close_ts, estimated_pnl)),
        )

    def _kappa_open_ok(
        self, st: _BookState, validator: str, book_id: int, estimated: float, now: int,
    ) -> bool:
        if estimated <= 0.0:
            return False
        projected = self._project_kappa(validator, book_id, now, estimated)
        if projected is None:
            return self._rt_count(st, now, self.kappa_rt_history_ns) < KAPPA_MIN_OBS
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
        return f"{before:.4f}->{after:.4f} ({'+' if delta >= 0 else ''}{delta:.4f})"

    def _fmt_rt_pnl_list(self, st: _BookState, now: int) -> str:
        cutoff = now - self.rt_window_ns
        pnls = [p for ts, p in st.rt_events if ts >= cutoff]
        return "[" + ", ".join(f"{p:+.4f}" for p in pnls) + "]" if pnls else "[]"

    def _stash_rt_open(
        self, validator: str, book_id: int, book, account,
        direction: OrderDirection, now: int, open_reason: str,
    ) -> None:
        if not self._rt_log_enabled(validator):
            return
        st = self._bstate(validator, book_id)
        rate = self._taker_fee_rate(account)
        est_pnl = self._estimate_rt_pnl(rate, book, self.min_order_size) if rate is not None else 0.0
        self._rt_log[(validator, book_id)] = _RtLogCtx(
            open_reason=open_reason,
            side=self._side_label(direction),
            open_rt_window_n=self._rt_count(st, now),
            open_rt_pnl_list=self._fmt_rt_pnl_list(st, now),
            est_pnl=est_pnl,
            kappa_at_open=st.kappa3,
            kappa_proj=self._project_kappa(validator, book_id, now, est_pnl),
            taker_bps=(rate * 1e4) if rate is not None else None,
        )

    def _ensure_rt_open_ctx(
        self, validator: str, book_id: int, direction: OrderDirection, ts: int,
    ) -> None:
        """Backfill a context on the opening fill if none was stashed, so a close never
        logs open=? (a richer stashed context is kept)."""
        if not self._rt_log_enabled(validator) or (validator, book_id) in self._rt_log:
            return
        st = self._bstate(validator, book_id)
        self._rt_log[(validator, book_id)] = _RtLogCtx(
            open_reason=st.pending_kind or "?",
            side=self._side_label(direction),
            open_rt_window_n=self._rt_count(st, ts),
            open_rt_pnl_list=self._fmt_rt_pnl_list(st, ts),
            kappa_at_open=st.kappa3,
        )

    def _log_rt(
        self, *, validator: str, book_id: int, ts: int, hold_s: float | None,
        entry_avg: float, exit_px: float, gross_pnl: float, open_fee: float, close_fee: float,
        net_pnl: float, kappa_before: float | None, kappa_after: float | None,
        rt_window_n: int, st: _BookState, final: bool = True,
    ) -> None:
        key = (validator, book_id)
        if not self._rt_log_enabled(validator):
            self._rt_log.pop(key, None)
            return
        # Keep the context across partial closes; pop only when fully flat.
        ctx = self._rt_log.pop(key, _RtLogCtx()) if final else self._rt_log.get(key, _RtLogCtx())
        hold_str = f"{hold_s:.2f}" if hold_s is not None else "n/a"
        taker_bps_str = f"{ctx.taker_bps:.2f}" if ctx.taker_bps is not None else "n/a"

        bt.logging.info(
            f"[TakerScalper uid={self.uid} RT] book={book_id} "
            f"open={ctx.open_reason}/{ctx.side} "
            f"open_rt_n={ctx.open_rt_window_n} open_rt_pnl={ctx.open_rt_pnl_list} "
            f"est_pnl={self._fmt_pnl(ctx.est_pnl)} "
            f"kappa_open={self._fmt_kappa_pair(ctx.kappa_at_open, ctx.kappa_proj)} "
            f"taker_bps={taker_bps_str} close={ctx.close_reason} hold_s={hold_str} "
            f"entry={entry_avg:.4f} exit={exit_px:.4f} "
            f"gross_pnl={gross_pnl:+.4f} open_fee={open_fee:+.4f} close_fee={close_fee:+.4f} "
            f"net_pnl={net_pnl:+.4f} "
            f"kappa_close={self._fmt_kappa_pair(kappa_before, kappa_after)} "
            f"close_rt_n={rt_window_n} close_rt_pnl={self._fmt_rt_pnl_list(st, ts)}"
        )

    # ------------------------------------------------------------------ open / close
    def _rt_count(self, st: _BookState, now: int, window_ns: int | None = None) -> int:
        cutoff = now - (self.rt_window_ns if window_ns is None else window_ns)
        return sum(1 for ts, _ in st.rt_events if ts >= cutoff)

    def _open(
        self, response, validator: str, book_id: int, book, account, volume_cap: float, now: int,
    ) -> None:
        """Flat book: scalp the directional edge when the kappa gate clears, else force an RT
        before the activity window lapses, else idle."""
        rate = self._taker_fee_rate(account)
        direction, mid = self._book_bias(book)
        if mid is None or rate is None:
            return

        st = self._bstate(validator, book_id)
        est_pnl = self._estimate_rt_pnl(rate, book, self.min_order_size)
        paid = rate <= 0.0 and est_pnl > 0.0   # genuinely paid to take, +EV round-trip
        reopen_ok = st.last_rt_ns == 0 or (now - st.last_rt_ns) >= self.min_reopen_gap_ns

        # Profit engine: scalp the directional edge, throttled per book, gated on kappa.
        if (
            paid
            and reopen_ok
            and self._rt_count(st, now) < RT_MAX
            and est_pnl >= KAPPA_EST_PNL_FLOOR
            and rate <= self.kappa_min_rebate_rate
            and self._rolled_quote_volume(validator, book_id, now) < volume_cap
            and self._kappa_open_ok(st, validator, book_id, est_pnl, now)
        ):
            self._try_open(response, validator, book_id, book, account, direction, now, "kappa", prune_vol=True)
            return

        # Activity backstop: guarantee >=1 RT per book per window.
        if self._activity_force_due(st, now):
            tag = "taker_force" if paid else "activity"
            self._try_open(response, validator, book_id, book, account, direction, now, tag)

    def _try_open(
        self, response, validator: str, book_id: int, book, account,
        direction: OrderDirection, now: int, tag: str, *, prune_vol: bool = False,
    ) -> None:
        """Submit one taker lot and stash RT context on success."""
        if not self._taker_open(response, validator, book_id, account, book, direction, tag):
            return
        st = self._bstate(validator, book_id)
        if prune_vol:
            self._prune_vol_log(st, now)
        self._stash_rt_open(validator, book_id, book, account, direction, now, tag)

    def _close(
        self, response, validator: str, book_id: int, book, account, pos: _Position,
        vol_dp: int, now: int,
    ) -> None:
        """Held position: taker flatten on TP / SL / max-hold, after the min hold."""
        mid = self._mid(book)
        if mid is None or mid <= 0 or pos.avg <= 0:
            return
        hold_ns = (now - pos.entry_ts) if pos.entry_ts else 0
        if hold_ns < self.min_hold_ns:
            return

        bid = book.bids[0].price if book.bids else None
        ask = book.asks[0].price if book.asks else None
        gross_bps = self._exit_gross_bps(pos, bid, ask, mid)
        if gross_bps >= MIN_GROSS_TP_BPS:
            reason = "tp"
        elif gross_bps <= -MAX_GROSS_SL_BPS:
            reason = "sl"
        elif hold_ns >= self.max_hold_ns:
            reason = "time"
        else:
            return

        if self._rt_log_enabled(validator):
            ctx = self._rt_log.get((validator, book_id))
            if ctx is not None:
                ctx.close_reason = reason
        self._close_position(response, validator, book_id, account, pos, vol_dp)

    # ------------------------------------------------------------------ market helpers
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
        """microprice vs mid -> direction; tie -> long."""
        mid = cls._mid(book)
        micro = cls._microprice(book)
        if mid is None or micro is None:
            return OrderDirection.BUY, mid
        return (OrderDirection.SELL if micro < mid else OrderDirection.BUY), mid

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
        rate = getattr(fees, "taker_fee_rate", None) if fees is not None else None
        try:
            return float(rate) if rate is not None else None
        except (TypeError, ValueError):
            return None

    def _taker_open(
        self, response, validator: str, book_id: int, account, book,
        direction: OrderDirection, kind: str,
    ) -> bool:
        """Cross the spread for one lot. BUY needs quote balance; SELL uses base inventory,
        falling back to a margin short. Returns whether the order was submitted."""
        qty = self.min_order_size
        if direction == OrderDirection.BUY:
            ask_px = book.asks[0].price if book.asks else 0.0
            if ask_px <= 0 or account.quote_balance.free < qty * ask_px:
                return False
            self._submit_market(response, validator, book_id, OrderDirection.BUY, qty)
        else:
            bid_px = book.bids[0].price if book.bids else 0.0
            if bid_px <= 0:
                return False
            free_base = account.base_balance.free if account.base_balance else 0.0
            if free_base >= qty:
                self._submit_market(response, validator, book_id, OrderDirection.SELL, qty)
            else:
                quote_loan = getattr(account, "quote_loan", 0.0) or 0.0
                self._submit_market(
                    response, validator, book_id, OrderDirection.SELL, qty,
                    leverage=0.0 if quote_loan > 0 else SHORT_LEVERAGE,
                    settlement=self._loan_settlement(account),
                )
        self._bstate(validator, book_id).pending_kind = kind
        return True

    def _close_position(
        self, response, validator: str, book_id: int, account, pos: _Position, vol_dp: int,
    ) -> None:
        """Flatten the whole position with one market order. Market fills are full, so a
        sub-min residue is unexpected; if one appears it cannot be ordered, so drop its
        tracking (negligible dust) rather than spam rejected orders."""
        qty = round(abs(pos.qty), vol_dp)
        if pos.qty > 0:
            free = account.base_balance.free if account.base_balance else 0.0
            qty = round(min(qty, free), vol_dp)
            direction, settlement = OrderDirection.SELL, LoanSettlementOption.NONE
        else:
            direction, settlement = OrderDirection.BUY, self._loan_settlement(account)

        if qty < self.exch_min:
            self._clear_position(pos)
            return
        self._submit_market(response, validator, book_id, direction, qty, settlement=settlement)

    def _submit_market(
        self, response, validator: str, book_id: int, direction: OrderDirection, qty: float,
        *, leverage: float = 0.0, settlement: LoanSettlementOption = LoanSettlementOption.NONE,
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
        self._bstate(validator, book_id).pending_ns = self._step_ts_ns   # rule 3: now in flight

    def _reconcile_position(self, account, pos: _Position, vol_dp: int) -> None:
        # Clamp tracked size to the real held base (settlement drift); clear when ~flat.
        if pos.qty > 0 and account.base_balance is not None:
            bal = account.base_balance
            held = bal.free + (bal.reserved or 0.0)
            if held < self._flat_eps:
                self._clear_position(pos)
            else:
                pos.qty = round(min(pos.qty, held), vol_dp)


if __name__ == "__main__":
    launch(TakerScalperAgent)
