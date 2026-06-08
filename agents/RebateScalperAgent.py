"""
RebateScalperAgent — taker rebate round-trip engine for subnet 79.

Tick flow
---------
1. Close all open legs (TP / SL / max_hold).
2. Scan 10 books per tick (rotating window); per book, open only if gates pass.
3. Open decision uses cached per-book Kappa-3 (3h realized PnL) on the normal path
   only when fee is good (rebate/zero → profitable RT estimate) and kappa keeps
   or improves. Activity overrides at 520s+ / 570s+ since last RT. Volume cap on
   kappa path only (24h rolling quote volume).

Open reasons (telemetry action): open_kappa | open_activity | open_force
"""

from dataclasses import dataclass, field
import math
from typing import Any

import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.telemetry import MinerTelemetry
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse
from taos.im.protocol.events import OrderPlacementEvent, TradeEvent, SimulationStartEvent
from taos.im.protocol.models import OrderDirection, OrderCurrency, STP


MIN_ORDER_SIZE = 0.25

# Hold / exit — shared close path
MIN_HOLD_S = 1.5
MAX_HOLD_S = 5.0
MIN_GROSS_TP_BPS = 1.5
MAX_GROSS_SL_BPS = 10.0

# Activity overrides (validator decay_grace_period = 600s)
ACTIVITY_RT_INTERVAL_S = 520.0     # skip kappa gate; open if fee ≤ 0
FORCE_ACTIVITY_RT_S = 570.0        # open anyway (still subject to RT caps + reopen gap)
MIN_REOPEN_GAP_S = 4.0

# Kappa-3 + RT caps (validator scoring.kappa defaults)
KAPPA_TAU = 0.0
KAPPA_MIN_OBS = 3
KAPPA_MIN_LOOKBACK_S = 5400.0      # 90 min
RT_LOOKBACK_S = 10_800.0           # 3h
RT_LOOKBACK_10M_S = 600.0
RT_CAP_3H = 18
RT_CAP_10M = 3

BOOKS_PER_STEP = 10                # open window per tick; closes scan all books

# Volume cap (kappa path only; activity/force bypass)
CAPITAL_TURNOVER_CAP = 10.0
VOLUME_SAFETY = 0.8
VOLUME_ASSESSMENT_NS = 86_400_000_000_000


@dataclass
class _Position:
    qty: float = 0.0
    avg: float = 0.0
    entry_ts: int = 0
    entry_fee: float = 0.0           # positive=cost, negative=rebate (validator FIFO)


@dataclass
class _BookState:
    last_rt_ns: int = 0
    pending_buy: bool = False                      # buy submitted, fill not yet seen
    rt_events: list = field(default_factory=list)    # (close_ts_ns, net_pnl) rolling 3h
    kappa3: float | None = None
    vol_log: list = field(default_factory=list)      # (ts, quote vol) for 24h turnover cap


class RebateScalperAgent(FinanceSimulationAgent):
    def initialize(self) -> None:
        bt.logging.set_info()

        self.min_order_size = MIN_ORDER_SIZE
        self._min_qty = MIN_ORDER_SIZE / 2
        self._volume_decimals: int | None = None
        self.volume_assessment_ns = VOLUME_ASSESSMENT_NS

        jitter = ((self.uid * 2654435761) % 1000) / 1000.0
        self.min_hold_ns = int(MIN_HOLD_S * 1e9)
        self.max_hold_ns = int(MAX_HOLD_S * (0.92 + 0.16 * jitter) * 1e9)
        self.min_reopen_gap_ns = int(MIN_REOPEN_GAP_S * (0.9 + 0.2 * jitter) * 1e9)
        self.activity_rt_interval_ns = int(ACTIVITY_RT_INTERVAL_S * 1e9)
        self.force_activity_rt_ns = int(FORCE_ACTIVITY_RT_S * 1e9)
        self.rt_lookback_ns = int(RT_LOOKBACK_S * 1e9)
        self.rt_lookback_10m_ns = int(RT_LOOKBACK_10M_S * 1e9)
        self.kappa_min_lookback_ns = int(KAPPA_MIN_LOOKBACK_S * 1e9)

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
            f"hold={MIN_HOLD_S}-{MAX_HOLD_S * (0.92 + 0.16 * jitter):.1f}s "
            f"books/step={BOOKS_PER_STEP} rt_cap={RT_CAP_10M}/10m {RT_CAP_3H}/3h"
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
        validator = self._active_validator
        if validator is not None:
            self._bstate(validator, event.bookId).pending_buy = False
        else:
            for v in self.books_state:
                self._bstate(v, event.bookId).pending_buy = False

    def onTrade(self, event: TradeEvent, validator: str = None) -> None:
        if event.bookId is None:
            return
        is_taker = self.uid == event.takerAgentId
        if is_taker:
            direction = OrderDirection.BUY if event.side == OrderDirection.BUY else OrderDirection.SELL
        elif self.uid == event.makerAgentId:
            direction = OrderDirection.SELL if event.side == OrderDirection.BUY else OrderDirection.BUY
        else:
            return
        ts_ns = self._step_ts_ns or event.timestamp
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
        for book_id in self._open_book_batch(open_step):
            book = state.books.get(book_id)
            account = self.accounts.get(book_id) if book else None
            if book is None or account is None:
                continue
            try:
                do_open, reason, meta = self._open_decision(
                    validator, book_id, book, account, vol_dp, volume_cap, now,
                )
                if do_open and self._taker_buy(response, validator, book_id, account, book):
                    self._record_open(validator, book_id, book, account, now, reason, volume_cap, meta)
            except Exception as ex:
                bt.logging.warning(f"[RebateScalper uid={self.uid}] open {book_id}: {ex}")

        self._open_step[validator] = open_step + 1
        self.telemetry.end_step(state, instructions=len(response.instructions) - instr_before)
        return response

    # ------------------------------------------------------------------ state
    @staticmethod
    def _effective_min_order_size(volume_decimals: int) -> float:
        """Match simulator: max(minOrderSize, 10^-volumeDecimals)."""
        return max(MIN_ORDER_SIZE, 10 ** (-volume_decimals))

    def _sync_order_size(self, volume_decimals: int) -> None:
        if volume_decimals == self._volume_decimals:
            return
        self._volume_decimals = volume_decimals
        lot = round(self._effective_min_order_size(volume_decimals), volume_decimals)
        self.min_order_size = lot
        self._min_qty = lot / 2
        bt.logging.info(
            f"[RebateScalper uid={self.uid}] volumeDecimals={volume_decimals} "
            f"effective_min={lot}"
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

    def _record_trade_volume(self, validator, book_id, qty, price, ts_ns) -> None:
        vol = float(qty) * float(price)
        if vol > 0:
            self._bstate(validator, book_id).vol_log.append((ts_ns, vol))

    def _rolled_quote_volume(self, validator, book_id, now_ns) -> float:
        st = self._bstate(validator, book_id)
        cutoff = now_ns - self.volume_assessment_ns
        st.vol_log = [(t, v) for t, v in st.vol_log if t >= cutoff]
        return sum(v for _, v in st.vol_log)

    def _apply_fill(self, validator, book_id, direction, qty, price, trade_fee, ts) -> None:
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
                book_id=book_id, ts_close_ns=ts,
                side="long" if prev > 0 else "short", qty=closed_qty,
                entry_avg=entry_avg, exit_avg=price, realized_pnl=net_pnl,
                hold_s=(ts - entry_ts) / 1e9 if entry_ts else None,
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
        cutoff = now - self.rt_lookback_ns
        before = len(st.rt_events)
        st.rt_events = [(t, p) for t, p in st.rt_events if t >= cutoff]
        return len(st.rt_events) != before

    def _record_rt_close(self, validator, book_id, ts, net_pnl) -> None:
        """Append RT PnL and refresh cached Kappa-3."""
        st = self._bstate(validator, book_id)
        self._prune_rt_events(st, ts)
        st.rt_events.append((ts, net_pnl))
        self._refresh_book_kappa(validator, book_id, ts)

    def _sync_book_rt_state(self, validator: str, book_id: int, now: int) -> _BookState:
        """Prune 3h RT history and refresh kappa when events drop off."""
        st = self._bstate(validator, book_id)
        if self._prune_rt_events(st, now):
            self._refresh_book_kappa(validator, book_id, now)
        return st

    # ------------------------------------------------------------------ kappa-3
    def _global_rt_timestamps(self, validator: str, now: int) -> list[int]:
        cutoff = now - self.rt_lookback_ns
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
        if extra is not None:
            extra_ts, _ = extra
            if extra_ts not in timestamps:
                timestamps = sorted(timestamps + [extra_ts])
        if not timestamps:
            return []

        cutoff = now - self.rt_lookback_ns
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
        """Mirror taos/im/utils/kappa.py per-book realized Kappa-3 ratio."""
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

    def _estimate_rt_pnl(self, taker_rate: float, book, qty: float) -> float:
        """Two taker legs: rebate/fees at mid, spread paid on entry+exit."""
        mid = self._mid(book)
        if mid is None or mid <= 0:
            return 0.0
        bid = book.bids[0].price if book.bids else mid
        ask = book.asks[0].price if book.asks else mid
        fee_pnl = -2.0 * taker_rate * mid * qty
        spread_cost = max(0.0, ask - bid) * qty
        return fee_pnl - spread_cost

    def _projected_close_ts(self, now: int) -> int:
        """Hypothetical RT close time for kappa projection (validator stamps at close)."""
        return now + self.min_hold_ns

    def _kappa_keeps_or_improves(self, validator: str, book_id: int, estimated_pnl: float, now: int) -> bool:
        """Whether adding a non-negative RT keeps or improves per-book Kappa-3."""
        if estimated_pnl < 0:
            return False
        st = self._bstate(validator, book_id)
        current = st.kappa3
        close_ts = self._projected_close_ts(now)
        projected = self._kappa3_raw(
            self._book_pnl_series(validator, book_id, now, extra=(close_ts, estimated_pnl)),
        )
        if projected is None:
            return len(st.rt_events) < KAPPA_MIN_OBS
        if current is None:
            return projected >= 0.0
        return projected >= current

    # ------------------------------------------------------------------ open gates
    def _rt_gap_ns(self, st: _BookState, now: int) -> int:
        if st.last_rt_ns == 0:
            return 0
        return now - st.last_rt_ns

    def _reopen_ok(self, st: _BookState, now: int) -> bool:
        return st.last_rt_ns == 0 or (now - st.last_rt_ns) >= self.min_reopen_gap_ns

    def _rt_count(self, st: _BookState, now: int, window_ns: int) -> int:
        cutoff = now - window_ns
        return sum(1 for ts, _ in st.rt_events if ts >= cutoff)

    def _at_rt_cap(self, st: _BookState, now: int) -> bool:
        return (
            self._rt_count(st, now, self.rt_lookback_10m_ns) >= RT_CAP_10M
            or self._rt_count(st, now, self.rt_lookback_ns) >= RT_CAP_3H
        )

    def _open_book_batch(self, step: int) -> list[int]:
        book_ids = sorted(self.accounts.keys())
        if not book_ids:
            return []
        batch_count = (len(book_ids) + BOOKS_PER_STEP - 1) // BOOKS_PER_STEP
        start = (step % batch_count) * BOOKS_PER_STEP
        return book_ids[start:start + BOOKS_PER_STEP]

    def _open_decision(
        self, validator, book_id, book, account, vol_dp, volume_cap, now,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Return (open?, reason, meta). reason: kappa | activity | force."""
        pos = self._book_positions(validator).get(book_id, _Position())
        self._reconcile_position(account, pos, vol_dp)
        if pos.qty >= self._min_qty:
            return False, "", {}

        st = self._sync_book_rt_state(validator, book_id, now)
        if st.pending_buy:
            return False, "", {}
        if not self._reopen_ok(st, now) or self._at_rt_cap(st, now):
            return False, "", {}

        mid = self._mid(book)
        rate = self._taker_fee_rate(account)
        if mid is None or mid <= 0 or rate is None:
            return False, "", {}

        gap_ns = self._rt_gap_ns(st, now)
        meta: dict[str, Any] = {
            "taker_bps": rate * 1e4,
            "kappa3": st.kappa3,
            "gap_s": gap_ns / 1e9,
        }

        if gap_ns >= self.force_activity_rt_ns:
            return True, "force", meta
        if gap_ns >= self.activity_rt_interval_ns:
            return (True, "activity", meta) if rate <= 0.0 else (False, "", {})

        if rate > 0.0:
            return False, "", {}
        if self._rolled_quote_volume(validator, book_id, now) >= volume_cap:
            return False, "", {}
        estimated = self._estimate_rt_pnl(rate, book, self.min_order_size)
        meta["estimated_pnl"] = estimated
        if estimated >= 0 and self._kappa_keeps_or_improves(validator, book_id, estimated, now):
            return True, "kappa", meta
        return False, "", {}

    def _record_open(
        self, validator, book_id, book, account, now, reason, volume_cap, meta: dict[str, Any],
    ) -> None:
        mid = self._mid(book)
        bid = book.bids[0].price if book.bids else None
        ask = book.asks[0].price if book.asks else None
        traded = self._rolled_quote_volume(validator, book_id, now)
        est = meta.get("estimated_pnl")
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
                "est_pnl": est if est is not None else meta.get("gap_s"),
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

    def _handle_close(self, response, validator, book_id, book, vol_dp, now) -> None:
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

    def _taker_buy(self, response, validator, book_id, account, book) -> bool:
        qty = self.min_order_size
        ask_px = book.asks[0].price if book.asks else 0.0
        if ask_px <= 0 or account.quote_balance.free < qty * ask_px:
            return False
        response.market_order(
            book_id=book_id, direction=OrderDirection.BUY,
            quantity=qty, currency=OrderCurrency.BASE, stp=STP.CANCEL_OLDEST,
        )
        self._bstate(validator, book_id).pending_buy = True
        return True

    def _taker_sell(self, response, account, book_id, pos, vol_dp) -> None:
        if pos.qty <= 0:
            return
        free = account.base_balance.free if account.base_balance else 0.0
        qty = round(min(self.min_order_size, pos.qty, free), vol_dp)
        if qty < self._min_qty:
            return
        response.market_order(
            book_id=book_id, direction=OrderDirection.SELL,
            quantity=qty, currency=OrderCurrency.BASE, stp=STP.CANCEL_OLDEST,
        )

    def _reconcile_position(self, account, pos, vol_dp) -> None:
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
