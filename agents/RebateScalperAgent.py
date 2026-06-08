"""
RebateScalperAgent
==================

Per-book taker round-trips on rebate books:

  * **Normal** — strong rebate (≥5 bps/side), deficit-weighted across 128 books.
  * **Activity** — no RT in 520s: open weak-rebate books (fee ≤ 0, not strong).
  * **Force** — no RT in 580s: open any book (even fee > 0) before 600s grace.
  * **Close** — standard TP / SL / max_hold (~5s); no special activity exit.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator

import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.telemetry import MinerTelemetry
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse
from taos.im.protocol.events import TradeEvent, SimulationStartEvent
from taos.im.protocol.models import OrderDirection, OrderCurrency, STP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_ORDER_SIZE = 0.25

MIN_HOLD_S = 1.5
MAX_HOLD_S = 5.0
MIN_GROSS_TP_BPS = 1.5
MAX_GROSS_SL_BPS = 10.0
COOLDOWN_S = 15.0                  # after SL; normal opens only

ACTIVITY_RT_INTERVAL_S = 520.0     # weak-rebate activity open
FORCE_ACTIVITY_RT_S = 580.0        # force open (20s before 600s grace)

MIN_RT_CYCLE_S = 4.0
ORDER_COOLDOWN_S = 0.5

MAX_TAKER_FEE = 0.0
STRONG_REBATE_BPS = -5.0
MAX_OPENS_PER_TICK = 6

# scheduler (normal pass)
KAPPA_LOOKBACK_S = 10_800.0
KAPPA_MIN_OBS = 3
KAPPA_TARGET_OBS = 6
SCORING_INTERVAL_S = 5.0
RT_TARGET_3H = 12
RT_CAP_3H = 24
BUCKET_COUNT = 16

PNL_EMA_ALPHA = 0.35
QUALITY_REF = 0.05
QUALITY_FREEZE = -0.3
TOXIC_PNL_FRAC = 0.003
TOXIC_CONSECUTIVE_SL = 3
RT_SURPLUS_BLOCK = 0.8
OUTLIER_Q1_MARGIN_FRAC = 0.001

CAPITAL_TURNOVER_CAP = 10.0
VOLUME_SAFETY = 0.35
VOLUME_ASSESSMENT_NS = 86_400_000_000_000


@dataclass
class _Position:
    qty: float = 0.0
    avg: float = 0.0
    entry_ts: int = 0
    entry_taker_bps: float = 0.0


@dataclass
class _BookState:
    last_rt_ns: int = 0
    last_order_ns: int = 0
    cooldown_until: int = 0
    vol_log: list = field(default_factory=list)
    rt_close_ts: list = field(default_factory=list)
    net_pnl_events: list = field(default_factory=list)
    net_pnl_ema: float = 0.0
    consecutive_sl: int = 0


class RebateScalperAgent(FinanceSimulationAgent):
    """Taker RT on rebate books with simple normal / activity / force open tiers."""

    def initialize(self) -> None:
        bt.logging.set_info()

        self.min_order_size = MIN_ORDER_SIZE
        self.min_hold_s = MIN_HOLD_S
        self.max_hold_s = MAX_HOLD_S
        self.min_gross_tp_bps = MIN_GROSS_TP_BPS
        self.max_gross_sl_bps = MAX_GROSS_SL_BPS
        self.cooldown_s = COOLDOWN_S
        self.min_rt_cycle_s = MIN_RT_CYCLE_S
        self.activity_rt_interval_s = ACTIVITY_RT_INTERVAL_S
        self.force_activity_rt_s = FORCE_ACTIVITY_RT_S
        self.order_cooldown_s = ORDER_COOLDOWN_S
        self.strong_rebate_bps = STRONG_REBATE_BPS
        self.max_opens_per_tick = MAX_OPENS_PER_TICK
        self.bucket_count = BUCKET_COUNT
        self.rt_target_3h = RT_TARGET_3H
        self.rt_cap_3h = RT_CAP_3H
        self.kappa_min_obs = KAPPA_MIN_OBS
        self.kappa_target_obs = KAPPA_TARGET_OBS
        self.turnover_cap = CAPITAL_TURNOVER_CAP
        self.volume_safety = VOLUME_SAFETY
        self.volume_assessment_ns = VOLUME_ASSESSMENT_NS

        self.min_hold_ns = int(self.min_hold_s * 1e9)
        self.max_hold_ns = int(self.max_hold_s * 1e9)
        self.min_rt_cycle_ns = int(self.min_rt_cycle_s * 1e9)
        self.activity_rt_interval_ns = int(self.activity_rt_interval_s * 1e9)
        self.force_activity_rt_ns = int(self.force_activity_rt_s * 1e9)
        self.order_cooldown_ns = int(self.order_cooldown_s * 1e9)
        self.cooldown_ns = int(self.cooldown_s * 1e9)
        self.kappa_lookback_ns = int(KAPPA_LOOKBACK_S * 1e9)
        self.scoring_interval_ns = int(SCORING_INTERVAL_S * 1e9)

        jitter = ((self.uid * 2654435761) % 1000) / 1000.0
        self.min_rt_cycle_s *= 0.9 + 0.2 * jitter
        self.max_hold_s *= 0.92 + 0.16 * jitter
        self.min_rt_cycle_ns = int(self.min_rt_cycle_s * 1e9)
        self.max_hold_ns = int(self.max_hold_s * 1e9)

        self.positions: dict[str, dict[int, _Position]] = {}
        self.books_state: dict[str, dict[int, _BookState]] = {}
        self._sim_id: dict[str, str] = {}
        self._exit_reason: dict[tuple[str, int], str] = {}
        self._step_ts_ns: int = 0

        self._fee_log_path = self._cfg_str("fee_log_path")
        self._fee_log_validator = self._cfg_str("fee_log_validator")
        self._fee_log_interval_s = float(getattr(self.config, "fee_log_interval_s", 30.0))
        self._fee_log_top_n = int(getattr(self.config, "fee_log_top_n", 20))
        self._fee_log_interval_ns = int(self._fee_log_interval_s * 1e9)
        self._fee_log_last_ns: int = 0

        self._sched_log_path = self._cfg_str("scheduler_log_path")
        self._sched_log_validator = self._cfg_str("scheduler_log_validator")
        self._sched_log_interval_s = float(getattr(self.config, "scheduler_log_interval_s", 60.0))
        self._sched_log_interval_ns = int(self._sched_log_interval_s * 1e9)
        self._sched_log_last_ns: int = 0
        self._last_open_log: dict[str, int] = {}

        self.telemetry = MinerTelemetry.from_agent(self, agent_class="RebateScalperAgent")
        bt.logging.info(
            f"[RebateScalper uid={self.uid}] lot={self.min_order_size} "
            f"hold={self.min_hold_s}-{self.max_hold_s}s "
            f"activity={self.activity_rt_interval_s}s force={self.force_activity_rt_s}s "
            f"strong={self.strong_rebate_bps}bps opens/tick={self.max_opens_per_tick}"
            + (f" fee_log={self._fee_log_path}" if self._fee_log_path else "")
            + (f" sched_log={self._sched_log_path}" if self._sched_log_path else "")
        )

    def _cfg_str(self, name: str) -> str | None:
        val = getattr(self.config, name, None)
        return str(val).strip() if val else None

    # --------------------------------------------------------------- lifecycle
    def onStart(self, event: SimulationStartEvent) -> None:
        self.positions.clear()
        self.books_state.clear()
        self._sim_id.clear()
        self._exit_reason.clear()

    def update(self, state: MarketSimulationStateUpdate) -> None:
        self._step_ts_ns = int(state.timestamp)
        super().update(state)

    # ------------------------------------------------------------- fills / state
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
        if vol > 0:
            self._bstate(validator, book_id).vol_log.append((ts_ns, vol))

    def _rolled_quote_volume(self, validator, book_id, now_ns) -> float:
        st = self._bstate(validator, book_id)
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
            total = abs(prev) + qty
            pos.avg = (pos.avg * abs(prev) + price * qty) / total if total > 0 else price
            pos.qty = prev + signed
            if prev == 0:
                pos.entry_ts = ts
        else:
            closed_qty = min(qty, abs(prev))
            if closed_qty >= self.min_order_size / 2 and entry_avg > 0:
                rpnl = (price - entry_avg) * closed_qty if prev > 0 else (entry_avg - price) * closed_qty
                side = "long" if prev > 0 else "short"
                reason = self._exit_reason.pop((validator, book_id), "fill")
                st = self._bstate(validator, book_id)
                st.last_rt_ns = ts
                exit_bps = self._taker_bps(self.accounts.get(book_id))
                net_pnl = self._net_rt_pnl(rpnl, closed_qty, entry_avg, price, pos.entry_taker_bps, exit_bps)
                self._record_rt_close(st, ts, net_pnl, reason)
                self.telemetry.record_round_trip(
                    book_id=book_id, ts_close_ns=ts, side=side, qty=closed_qty,
                    entry_avg=entry_avg, exit_avg=price, realized_pnl=rpnl,
                    hold_s=(ts - entry_ts) / 1e9 if entry_ts else None, reason=reason,
                )
            pos.qty = prev + signed
            if abs(pos.qty) < 1e-12:
                pos.qty, pos.avg, pos.entry_ts, pos.entry_taker_bps = 0.0, 0.0, 0, 0.0
            elif (prev > 0) != (pos.qty > 0):
                pos.avg, pos.entry_ts = price, ts
                pos.entry_taker_bps = 0.0

    # ----------------------------------------------------------------- fees
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

    def _taker_bps(self, account) -> float | None:
        rate = self._taker_fee_rate(account)
        return rate * 1e4 if rate is not None else None

    def _rebate_ok(self, account) -> bool:
        rate = self._taker_fee_rate(account)
        return rate is not None and rate <= MAX_TAKER_FEE

    def _strong_rebate(self, taker_bps: float | None) -> bool:
        return taker_bps is not None and taker_bps <= self.strong_rebate_bps

    def _account_book_ids(self) -> list[int]:
        return sorted(self.accounts.keys())

    def _fee_rows_all_books(self) -> list[tuple[float, float | None, int]]:
        rows = []
        for book_id in self._account_book_ids():
            bps = self._taker_bps(self.accounts[book_id])
            if bps is None:
                continue
            rows.append((bps, self._taker_fee_rate(self.accounts[book_id]), book_id))
        rows.sort()
        return rows

    def _books_to_close(self, state: MarketSimulationStateUpdate, validator: str) -> list[int]:
        ids = set(state.books.keys())
        for book_id, pos in self._book_positions(validator).items():
            if abs(pos.qty) >= self.min_order_size / 2:
                ids.add(book_id)
        return sorted(ids)

    # ---------------------------------------------------------- RT gap / tiers
    def _rt_gap_ns(self, st: _BookState, now: int) -> int:
        if st.last_rt_ns == 0:
            return self.force_activity_rt_ns
        return now - st.last_rt_ns

    def _rt_gap_s(self, st: _BookState, now: int) -> float:
        return self._rt_gap_ns(st, now) / 1e9

    def _activity_due(self, st: _BookState, now: int) -> bool:
        return st.last_rt_ns == 0 or self._rt_gap_ns(st, now) >= self.activity_rt_interval_ns

    def _force_due(self, st: _BookState, now: int) -> bool:
        return st.last_rt_ns == 0 or self._rt_gap_ns(st, now) >= self.force_activity_rt_ns

    def _in_activity_window(self, st: _BookState, now: int) -> bool:
        """520s ≤ gap < 580s — weak-rebate activity tier."""
        return self._activity_due(st, now) and not self._force_due(st, now)

    # ---------------------------------------------------------- scheduler (normal pass)
    @staticmethod
    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    @staticmethod
    def _rebate_notional(bps: float | None, notional: float) -> float:
        if bps is None or bps >= 0 or notional <= 0:
            return 0.0
        return abs(bps / 1e4) * notional

    def _net_rt_pnl(self, gross, qty, entry_avg, exit_px, entry_bps, exit_bps) -> float:
        return gross + self._rebate_notional(entry_bps, entry_avg * qty) + self._rebate_notional(exit_bps, exit_px * qty)

    def _prune_rolling(self, st: _BookState, now: int) -> None:
        cutoff = now - self.kappa_lookback_ns
        st.rt_close_ts = [t for t in st.rt_close_ts if t >= cutoff]
        st.net_pnl_events = [(t, p) for t, p in st.net_pnl_events if t >= cutoff]

    def _rt_count_3h(self, st: _BookState, now: int) -> int:
        self._prune_rolling(st, now)
        return len(st.rt_close_ts)

    def _net_pnl_3h(self, st: _BookState, now: int) -> float:
        self._prune_rolling(st, now)
        return sum(p for _, p in st.net_pnl_events)

    def _pnl_obs_3h(self, st: _BookState, now: int) -> int:
        self._prune_rolling(st, now)
        return len({ts // self.scoring_interval_ns for ts, p in st.net_pnl_events if p != 0.0})

    def _record_rt_close(self, st: _BookState, ts: int, net_pnl: float, reason: str) -> None:
        st.net_pnl_ema = net_pnl if not st.rt_close_ts else (1 - PNL_EMA_ALPHA) * st.net_pnl_ema + PNL_EMA_ALPHA * net_pnl
        st.rt_close_ts.append(ts)
        st.net_pnl_events.append((ts, net_pnl))
        st.consecutive_sl = st.consecutive_sl + 1 if reason == "sl" else 0

    def _book_capital(self, cfg) -> float:
        return cfg.miner_wealth / max(len(self._account_book_ids()), 1)

    def _fleet_q1(self, validator: str, now: int) -> float | None:
        vals = [self._net_pnl_3h(self._bstate(validator, b), now) for b in self._account_book_ids()
                if self._bstate(validator, b).rt_close_ts]
        if len(vals) < 4:
            return None
        s = sorted(vals)
        return s[len(s) // 4]

    def _book_tier(self, st: _BookState, now: int, book_capital: float, fleet_q1: float | None) -> str:
        net_3h = self._net_pnl_3h(st, now)
        if net_3h < -TOXIC_PNL_FRAC * book_capital or st.consecutive_sl >= TOXIC_CONSECUTIVE_SL:
            return "toxic"
        q = st.net_pnl_ema / QUALITY_REF
        if q < QUALITY_FREEZE:
            return "weak"
        if fleet_q1 is not None and net_3h < fleet_q1 - OUTLIER_Q1_MARGIN_FRAC * book_capital:
            return "outlier"
        return "ok"

    def _book_priority(self, st: _BookState, now: int, taker_bps: float, book_capital: float, fleet_q1: float | None) -> dict:
        rt_n = self._rt_count_3h(st, now)
        obs_n = self._pnl_obs_3h(st, now)
        gap_s = self._rt_gap_s(st, now)
        tier = self._book_tier(st, now, book_capital, fleet_q1)
        q = st.net_pnl_ema / QUALITY_REF
        obs_def = self._clamp01((self.kappa_target_obs - obs_n) / self.kappa_target_obs)
        rt_def = self._clamp01((self.rt_target_3h - rt_n) / self.rt_target_3h)
        rt_sur = self._clamp01((rt_n - self.rt_cap_3h) / self.rt_cap_3h)
        fee = self._clamp01((self.strong_rebate_bps - taker_bps) / 20.0)
        outlier = tier in ("toxic", "outlier")
        priority = (
            4.0 * obs_def + 2.5 * rt_def + 1.5 * fee + max(0.0, q)
            - 3.0 * rt_sur - (5.0 if outlier else 0.0)
        )
        return {
            "priority": priority, "gap_s": gap_s, "rt_count": rt_n, "pnl_obs": obs_n,
            "tier": tier, "rt_surplus": rt_sur, "quality": q,
            "obs_deficit": self._clamp01((self.kappa_min_obs - obs_n) / self.kappa_min_obs),
        }

    def _normal_ok(self, m: dict) -> bool:
        return m["tier"] != "toxic" and m["tier"] != "outlier" and m["rt_surplus"] <= RT_SURPLUS_BLOCK and m["quality"] >= QUALITY_FREEZE

    def _bucket_ok(self, book_id: int, now: int, st: _BookState, m: dict) -> bool:
        if self._activity_due(st, now) or m["obs_deficit"] >= 0.5:
            return True
        return (book_id % self.bucket_count) == ((now // 1_000_000_000) % self.bucket_count)

    def _iter_flat_books(
        self, state: MarketSimulationStateUpdate, validator: str,
    ) -> Iterator[tuple[int, object, object, float, _BookState]]:
        for book_id in self._account_book_ids():
            book = state.books.get(book_id)
            if book is None:
                continue
            account = self.accounts[book_id]
            taker_bps = self._taker_bps(account)
            if taker_bps is None:
                continue
            pos = self._book_positions(validator).get(book_id, _Position())
            if pos.qty >= self.min_order_size / 2:
                continue
            yield book_id, book, account, taker_bps, self._bstate(validator, book_id)

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

        vol_dp = cfg.volumeDecimals
        cap = self.turnover_cap * cfg.miner_wealth * self.volume_safety
        now = state.timestamp
        self.telemetry.begin_step(state)
        instr_before = len(response.instructions)
        opens = {"normal": 0, "activity": 0, "force": 0}

        # 1) Close — TP / SL / max_hold only.
        for book_id in self._books_to_close(state, validator):
            book = state.books.get(book_id)
            if book is None:
                continue
            try:
                self._handle_close(response, validator, book_id, book, vol_dp, cap, now)
            except Exception as ex:
                bt.logging.warning(f"[RebateScalper uid={self.uid}] close {book_id}: {ex}")

        book_capital = self._book_capital(cfg)
        fleet_q1 = self._fleet_q1(validator, now)
        normal_ranked: list[tuple[float, int, float, dict]] = []
        normal_picked: list[tuple[int, float, dict]] = []

        # 2) Normal — strong rebate, gap < 520s (not yet in activity window).
        for book_id, book, account, taker_bps, st in self._iter_flat_books(state, validator):
            if not self._rebate_ok(account) or not self._strong_rebate(taker_bps):
                continue
            if self._activity_due(st, now):
                continue
            m = self._book_priority(st, now, taker_bps, book_capital, fleet_q1)
            normal_ranked.append((m["priority"], book_id, taker_bps, m))

        normal_ranked.sort(key=lambda x: (-x[0], x[2]))
        for priority, book_id, taker_bps, m in normal_ranked:
            if opens["normal"] >= self.max_opens_per_tick:
                break
            if not self._normal_ok(m) or not self._bucket_ok(book_id, now, self._bstate(validator, book_id), m):
                continue
            st = self._bstate(validator, book_id)
            if not self._rt_cycle_ok(st, now):
                continue
            book = state.books.get(book_id)
            if book and self._try_open(response, validator, book_id, book, vol_dp, cap, now, taker_bps, "normal", opens):
                normal_picked.append((book_id, priority, m))

        # 3) Activity — 520s ≤ gap < 580s, weak rebate (fee ≤ 0, not strong).
        activity_list = [
            (self._rt_gap_s(st, now), taker_bps, book_id, book)
            for book_id, book, account, taker_bps, st in self._iter_flat_books(state, validator)
            if self._in_activity_window(st, now) and self._rebate_ok(account) and not self._strong_rebate(taker_bps)
        ]
        activity_list.sort(key=lambda x: (-x[0], x[1]))
        for _gap, taker_bps, book_id, book in activity_list:
            self._try_open(response, validator, book_id, book, vol_dp, cap, now, taker_bps, "activity", opens)

        # 4) Force — gap ≥ 580s, any fee; bypass cooldown + volume cap.
        force_list = [
            (taker_bps, book_id, book)
            for book_id, book, _account, taker_bps, st in self._iter_flat_books(state, validator)
            if self._force_due(st, now)
        ]
        force_list.sort()
        for taker_bps, book_id, book in force_list:
            self._try_open(response, validator, book_id, book, vol_dp, cap, now, taker_bps, "force", opens)

        self._maybe_log_opens(state, now, opens, normal_ranked, normal_picked, fleet_q1, book_capital)
        self._maybe_log_top_fees(state, now)
        self.telemetry.end_step(state, instructions=len(response.instructions) - instr_before)
        return response

    def _try_open(self, response, validator, book_id, book, vol_dp, cap, now, taker_bps, mode, opens: dict) -> bool:
        try:
            if self._handle_open(response, validator, book_id, book, vol_dp, cap, now, taker_bps, mode):
                opens[mode] += 1
                return True
        except Exception as ex:
            bt.logging.warning(f"[RebateScalper uid={self.uid}] {mode} {book_id}: {ex}")
        return False

    # -------------------------------------------------------------- open / close
    def _order_cooldown_ok(self, st: _BookState, now: int) -> bool:
        return st.last_order_ns == 0 or (now - st.last_order_ns) >= self.order_cooldown_ns

    def _rt_cycle_ok(self, st: _BookState, now: int) -> bool:
        return st.last_rt_ns == 0 or (now - st.last_rt_ns) >= self.min_rt_cycle_ns

    def _handle_close(self, response, validator, book_id, book, vol_dp, cap, now) -> None:
        mid = self._mid(book)
        if mid is None or mid <= 0:
            return
        account = self.accounts.get(book_id)
        if account is None:
            return

        bid = book.bids[0].price if book.bids else None
        pos = self._book_positions(validator).setdefault(book_id, _Position())
        st = self._bstate(validator, book_id)
        self._reconcile_position(account, pos, vol_dp)

        if pos.qty < -self.min_order_size / 2:
            self._exit_reason[(validator, book_id)] = "short_flatten"
            st.last_order_ns = now
            self._taker_sell(response, account, book_id, pos, vol_dp)
            return

        if pos.qty < self.min_order_size / 2 or pos.avg <= 0:
            return

        hold_ns = (now - pos.entry_ts) if pos.entry_ts else 0
        if hold_ns < self.min_hold_ns:
            return
        if not self._order_cooldown_ok(st, now):
            return

        exit_px = bid if bid and bid > 0 else mid
        gross_bps = (exit_px - pos.avg) / pos.avg * 1e4

        if gross_bps >= self.min_gross_tp_bps:
            self._exit_reason[(validator, book_id)] = "tp"
        elif gross_bps <= -self.max_gross_sl_bps:
            self._exit_reason[(validator, book_id)] = "sl"
            st.cooldown_until = now + self.cooldown_ns
        elif hold_ns >= self.max_hold_ns:
            self._exit_reason[(validator, book_id)] = "time"
        else:
            return

        st.last_order_ns = now
        self._taker_sell(response, account, book_id, pos, vol_dp)

    def _handle_open(self, response, validator, book_id, book, vol_dp, cap, now, taker_bps: float, mode: str) -> bool:
        account = self.accounts.get(book_id)
        if account is None or self._mid(book) is None:
            return False

        pos = self._book_positions(validator).setdefault(book_id, _Position())
        st = self._bstate(validator, book_id)
        self._reconcile_position(account, pos, vol_dp)
        if pos.qty >= self.min_order_size / 2:
            return False
        if not self._order_cooldown_ok(st, now):
            return False

        activity_mode = mode in ("activity", "force")
        force = mode == "force"

        if not force and not self._rebate_ok(account):
            return False
        if mode == "normal" and not self._strong_rebate(taker_bps):
            return False
        if mode == "activity" and self._strong_rebate(taker_bps):
            return False

        if not activity_mode:
            if now < st.cooldown_until:
                return False
            if self._rolled_quote_volume(validator, book_id, now) >= cap:
                return False
            if not self._rt_cycle_ok(st, now):
                return False

        return self._taker_buy(response, account, book_id, book, pos, st, now, taker_bps)

    def _taker_buy(self, response, account, book_id, book, pos, st, now, taker_bps: float) -> bool:
        qty = self.min_order_size
        ask_px = book.asks[0].price if book.asks else 0.0
        if ask_px <= 0 or account.quote_balance.free < qty * ask_px:
            return False
        st.last_order_ns = now
        pos.entry_taker_bps = taker_bps
        response.market_order(
            book_id=book_id, direction=OrderDirection.BUY,
            quantity=qty, currency=OrderCurrency.BASE, stp=STP.CANCEL_OLDEST,
        )
        return True

    def _taker_sell(self, response, account, book_id, pos, vol_dp) -> None:
        if pos.qty <= 0:
            return
        free = account.base_balance.free if account.base_balance else 0.0
        qty = round(min(self.min_order_size, pos.qty, free), vol_dp)
        if qty < self.min_order_size / 2:
            return
        response.market_order(
            book_id=book_id, direction=OrderDirection.SELL,
            quantity=qty, currency=OrderCurrency.BASE, stp=STP.CANCEL_OLDEST,
        )

    # --------------------------------------------------------------------- logs
    def _maybe_log_opens(self, state, now, opens, normal_ranked, normal_picked, fleet_q1, book_capital):
        validator = state.dendrite.hotkey
        total = sum(opens.values())
        if total > 0:
            last = self._last_open_log.get(validator, 0)
            if now - last >= 10_000_000_000:
                self._last_open_log[validator] = now
                sel = ",".join(str(b) for b, _, _ in normal_picked) or "-"
                bt.logging.debug(
                    f"[RebateScalper uid={self.uid}] opens normal={opens['normal']} "
                    f"activity={opens['activity']} force={opens['force']} sel=[{sel}]"
                )

        if not self._sched_log_path:
            return
        if self._sched_log_validator and validator != self._sched_log_validator:
            return
        if self._sched_log_last_ns and (now - self._sched_log_last_ns) < self._sched_log_interval_ns:
            return
        self._sched_log_last_ns = now

        max_gap = max((self._rt_gap_s(self._bstate(validator, b), now) for b in self._account_book_ids()), default=0.0)
        wall = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        val_short = validator[:12] + "..."
        lines = [
            f"# sched uid={self.uid} | {wall} | {val_short} sim={now/1e9:.1f}s",
            f"# opens normal={opens['normal']} activity={opens['activity']} force={opens['force']} max_gap={max_gap:.0f}s",
            "rank  book  pri   gap_s  rt3h obs  tier     sel",
        ]
        for rank, (pri, book_id, _bps, m) in enumerate(normal_ranked[:10], 1):
            sel = "Y" if any(b == book_id for b, _, _ in normal_picked) else "N"
            lines.append(f"{rank:4}  {book_id:4}  {pri:5.2f}  {m['gap_s']:5.0f}  {m['rt_count']:3}  {m['pnl_obs']:3}  {m['tier']:7}  {sel}")
        try:
            os.makedirs(os.path.dirname(self._sched_log_path) or ".", exist_ok=True)
            with open(self._sched_log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n\n")
        except OSError as ex:
            bt.logging.warning(f"[RebateScalper uid={self.uid}] sched log failed: {ex}")

    def _maybe_log_top_fees(self, state: MarketSimulationStateUpdate, now: int) -> None:
        if not self._fee_log_path:
            return
        validator = state.dendrite.hotkey
        if self._fee_log_validator and validator != self._fee_log_validator:
            return
        if self._fee_log_last_ns and (now - self._fee_log_last_ns) < self._fee_log_interval_ns:
            return
        self._fee_log_last_ns = now

        rows = self._fee_rows_all_books()
        wall = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        val_short = validator[:12] + "..."
        lines = [
            f"# fees uid={self.uid} | {wall} | {val_short} sim={now/1e9:.1f}s",
            f"# books={len(rows)} strong={sum(1 for b,_,_ in rows if self._strong_rebate(b))}",
            "rank  book  taker_bps  strong  rebate_ok",
        ]
        for rank, (bps, _rate, book_id) in enumerate(rows[: self._fee_log_top_n], 1):
            lines.append(f"{rank:4}  {book_id:4}  {bps:8.2f}  {'Y' if self._strong_rebate(bps) else 'N':6}  {'Y' if bps <= 0 else 'N'}")
        try:
            os.makedirs(os.path.dirname(self._fee_log_path) or ".", exist_ok=True)
            with open(self._fee_log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n\n")
        except OSError as ex:
            bt.logging.warning(f"[RebateScalper uid={self.uid}] fee log failed: {ex}")

    def _reconcile_position(self, account, pos, vol_dp) -> None:
        if account.base_balance is None:
            return
        free = account.base_balance.free
        if pos.qty > 0:
            if free < self.min_order_size / 2:
                pos.qty, pos.avg, pos.entry_ts, pos.entry_taker_bps = 0.0, 0.0, 0, 0.0
            else:
                pos.qty = round(min(pos.qty, free, self.min_order_size), vol_dp)


if __name__ == "__main__":
    launch(RebateScalperAgent)
