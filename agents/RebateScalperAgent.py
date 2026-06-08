"""
RebateScalperAgent
==================

Taker rebate round-trip engine modeled on top miner #22 (sim ``20260606_1135``):

  * **100% taker** on books where ``taker_fee_rate <= 0`` (free or rebate).
  * **0.25 BASE** clips; buy wave → hold ~2-5s → sell wave (separate ticks).
  * **Edge from rebates** (~20-30 bps/RT); gross price move kept small (TP ~2 bps).
  * **128-book parallel** — every eligible book handled each validator tick.
  * **Activity guard** — force RT if none completed in ~520s (under 600s window).
  * **Rebate priority** — flat books sorted by best (most negative) taker rate.
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

# ---------------------------------------------------------------------------
# Strategy constants — tuned from #22 trade tape + status table
# ---------------------------------------------------------------------------

MIN_ORDER_SIZE = 0.25              # #22 median lot

# --- hold / exit (#22: med hold ~4.2s, buy→sell gap ~1.8s) ---
MIN_HOLD_S = 1.5                   # don't sell same tick as buy
MAX_HOLD_S = 5.0                   # timeout exit; rebate already earned on buy leg
MIN_GROSS_TP_BPS = 1.5             # tiny price win; rebate is the main edge
MAX_GROSS_SL_BPS = 10.0            # gross loss cap; ~30 bps RT rebate still nets positive
COOLDOWN_S = 15.0                  # pause book after stop-out

# --- round-trip cadence ---
MIN_RT_CYCLE_S = 4.0               # min gap between RT starts (protect per-book MTR)
ACTIVITY_RT_INTERVAL_S = 520.0     # force RT before ~600s activity decay
ORDER_COOLDOWN_S = 0.5             # min gap between orders on same book

# --- rebate gate ---
MAX_TAKER_FEE = 0.0                # trade only when taker fee ≤ 0
STRONG_REBATE_BPS = -5.0           # prefer books with rebate ≥ 5 bps/side (rate ≤ -0.0005)

# --- volume cap ---
CAPITAL_TURNOVER_CAP = 10.0
VOLUME_SAFETY = 0.35
VOLUME_ASSESSMENT_NS = 86_400_000_000_000


@dataclass
class _Position:
    qty: float = 0.0
    avg: float = 0.0
    entry_ts: int = 0
    entry_taker_bps: float = 0.0     # taker fee rate × 1e4 at open (negative = rebate)


@dataclass
class _BookState:
    last_rt_ns: int = 0
    last_order_ns: int = 0
    cooldown_until: int = 0
    vol_log: list = field(default_factory=list)


class RebateScalperAgent(FinanceSimulationAgent):
    """Per-book taker RT machine gated on dynamic taker rebates."""

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
        self.order_cooldown_s = ORDER_COOLDOWN_S
        self.max_taker_fee = MAX_TAKER_FEE
        self.strong_rebate_bps = STRONG_REBATE_BPS
        self.turnover_cap = CAPITAL_TURNOVER_CAP
        self.volume_safety = VOLUME_SAFETY
        self.volume_assessment_ns = VOLUME_ASSESSMENT_NS

        self.min_hold_ns = int(self.min_hold_s * 1e9)
        self.max_hold_ns = int(self.max_hold_s * 1e9)
        self.min_rt_cycle_ns = int(self.min_rt_cycle_s * 1e9)
        self.activity_rt_interval_ns = int(self.activity_rt_interval_s * 1e9)
        self.order_cooldown_ns = int(self.order_cooldown_s * 1e9)
        self.cooldown_ns = int(self.cooldown_s * 1e9)

        # Per-UID jitter so fleet miners don't synchronize on every book.
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

        self.telemetry = MinerTelemetry.from_agent(self, agent_class="RebateScalperAgent")

        bt.logging.info(
            f"[RebateScalper uid={self.uid}] lot={self.min_order_size} "
            f"hold={self.min_hold_s}-{self.max_hold_s}s "
            f"tp={self.min_gross_tp_bps}bps sl={self.max_gross_sl_bps}bps "
            f"rt_cycle={self.min_rt_cycle_s:.1f}s activity={self.activity_rt_interval_s}s "
            f"max_taker_fee={self.max_taker_fee}"
        )

    # --------------------------------------------------------------- lifecycle
    def onStart(self, event: SimulationStartEvent) -> None:
        self.positions.clear()
        self.books_state.clear()
        self._sim_id.clear()
        self._exit_reason.clear()
        bt.logging.info(f"[RebateScalper uid={self.uid}] simulation start: reset state")

    def update(self, state: MarketSimulationStateUpdate) -> None:
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
                if prev > 0:
                    rpnl = (price - entry_avg) * closed_qty
                    side = "long"
                else:
                    rpnl = (entry_avg - price) * closed_qty
                    side = "short"
                hold_s = (ts - entry_ts) / 1e9 if entry_ts else None
                reason = self._exit_reason.pop((validator, book_id), "fill")
                self._bstate(validator, book_id).last_rt_ns = ts
                self.telemetry.record_round_trip(
                    book_id=book_id, ts_close_ns=ts, side=side, qty=closed_qty,
                    entry_avg=entry_avg, exit_avg=price, realized_pnl=rpnl,
                    hold_s=hold_s, reason=reason,
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

    def _taker_rebate_ok(self, account) -> bool:
        rate = self._taker_fee_rate(account)
        return rate is not None and rate <= self.max_taker_fee

    def _expected_rt_rebate_bps(self, account) -> float:
        """Estimated RT rebate in bps (positive number) from current taker rate."""
        bps = self._taker_bps(account)
        if bps is None or bps > 0:
            return 0.0
        return abs(2.0 * bps)

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

        # Pass 1 — sell wave: close open legs ready to exit.
        for book_id, book in state.books.items():
            try:
                self._handle_close(response, validator, book_id, book, vol_dp, cap, now)
            except Exception as ex:
                bt.logging.warning(
                    f"[RebateScalper uid={self.uid}] close book {book_id}: {ex}\n"
                    f"{traceback.format_exc()}"
                )

        # Pass 2 — buy wave: open on rebate books (best rebate first).
        buy_candidates: list[tuple[float, int, object]] = []
        for book_id, book in state.books.items():
            account = self.accounts.get(book_id)
            if account is None:
                continue
            taker_bps = self._taker_bps(account)
            if taker_bps is None:
                continue
            pos = self._book_positions(validator).get(book_id, _Position())
            if pos.qty >= self.min_order_size / 2:
                continue
            buy_candidates.append((taker_bps, book_id, book))

        buy_candidates.sort(key=lambda x: x[0])  # most negative = best rebate first

        for taker_bps, book_id, book in buy_candidates:
            try:
                self._handle_open(response, validator, book_id, book, vol_dp, cap, now, taker_bps)
            except Exception as ex:
                bt.logging.warning(
                    f"[RebateScalper uid={self.uid}] open book {book_id}: {ex}\n"
                    f"{traceback.format_exc()}"
                )

        self.telemetry.end_step(state, instructions=len(response.instructions) - instr_before)
        return response

    def _reconcile_position(self, account, pos, vol_dp) -> None:
        if account.base_balance is None:
            return
        free = account.base_balance.free
        if pos.qty > 0:
            if free < self.min_order_size / 2:
                pos.qty, pos.avg, pos.entry_ts, pos.entry_taker_bps = 0.0, 0.0, 0, 0.0
            else:
                pos.qty = round(min(pos.qty, free, self.min_order_size), vol_dp)

    def _order_cooldown_ok(self, st: _BookState, now: int) -> bool:
        return st.last_order_ns == 0 or (now - st.last_order_ns) >= self.order_cooldown_ns

    def _rt_cycle_ok(self, st: _BookState, now: int) -> bool:
        return st.last_rt_ns == 0 or (now - st.last_rt_ns) >= self.min_rt_cycle_ns

    def _activity_rt_due(self, st: _BookState, now: int) -> bool:
        return st.last_rt_ns == 0 or (now - st.last_rt_ns) >= self.activity_rt_interval_ns

    def _handle_close(self, response, validator, book_id, book, vol_dp, cap, now) -> None:
        mid = self._mid(book)
        if mid is None or mid <= 0:
            return
        account = self.accounts.get(book_id)
        if account is None:
            return

        bid = book.bids[0].price if book.bids else None
        ask = book.asks[0].price if book.asks else None
        pos = self._book_positions(validator).setdefault(book_id, _Position())
        st = self._bstate(validator, book_id)
        self._reconcile_position(account, pos, vol_dp)

        taker_bps = self._taker_bps(account)

        if pos.qty < -self.min_order_size / 2:
            self._exit_reason[(validator, book_id)] = "short_flatten"
            st.last_order_ns = now
            self._taker_sell(response, account, book_id, pos, vol_dp)
            self._snap(validator, book_id, mid, bid, ask, pos, account, taker_bps, "short_flatten", cap, now)
            return

        if pos.qty < self.min_order_size / 2 or pos.avg <= 0:
            return

        if not self._taker_rebate_ok(account):
            # Holding from prior rebate regime; still close to flat (exit always taker).
            pass

        hold_ns = (now - pos.entry_ts) if pos.entry_ts else 0
        if hold_ns < self.min_hold_ns:
            self._snap(validator, book_id, mid, bid, ask, pos, account, taker_bps, "hold", cap, now)
            return

        if not self._order_cooldown_ok(st, now):
            self._snap(validator, book_id, mid, bid, ask, pos, account, taker_bps, "order_cd", cap, now)
            return

        exit_px = bid if bid and bid > 0 else mid
        gross_bps = (exit_px - pos.avg) / pos.avg * 1e4
        timed_out = hold_ns >= self.max_hold_ns

        if gross_bps >= self.min_gross_tp_bps:
            self._exit_reason[(validator, book_id)] = "tp"
            action = "exit_tp"
        elif gross_bps <= -self.max_gross_sl_bps:
            self._exit_reason[(validator, book_id)] = "sl"
            st.cooldown_until = now + self.cooldown_ns
            action = "exit_sl"
        elif timed_out:
            self._exit_reason[(validator, book_id)] = "time"
            action = "exit_time"
        else:
            self._snap(validator, book_id, mid, bid, ask, pos, account, taker_bps, "manage", cap, now,
                       gross_bps=gross_bps, hold_s=hold_ns / 1e9)
            return

        st.last_order_ns = now
        self._taker_sell(response, account, book_id, pos, vol_dp)
        self._snap(validator, book_id, mid, bid, ask, pos, account, taker_bps, action, cap, now,
                   gross_bps=gross_bps, hold_s=hold_ns / 1e9)

    def _handle_open(self, response, validator, book_id, book, vol_dp, cap, now, taker_bps: float) -> None:
        mid = self._mid(book)
        if mid is None or mid <= 0:
            return
        account = self.accounts.get(book_id)
        if account is None:
            return

        bid = book.bids[0].price if book.bids else None
        ask = book.asks[0].price if book.asks else None
        pos = self._book_positions(validator).setdefault(book_id, _Position())
        st = self._bstate(validator, book_id)
        self._reconcile_position(account, pos, vol_dp)

        if pos.qty >= self.min_order_size / 2:
            return

        activity_due = self._activity_rt_due(st, now)
        rebate_ok = self._taker_rebate_ok(account)

        if not rebate_ok:
            if activity_due:
                self._snap(validator, book_id, mid, bid, ask, pos, account, taker_bps,
                           "activity_wait_rebate", cap, now)
            else:
                self._snap(validator, book_id, mid, bid, ask, pos, account, taker_bps,
                           "no_rebate", cap, now)
            return

        if now < st.cooldown_until:
            self._snap(validator, book_id, mid, bid, ask, pos, account, taker_bps, "cooldown", cap, now)
            return

        if not self._order_cooldown_ok(st, now):
            self._snap(validator, book_id, mid, bid, ask, pos, account, taker_bps, "order_cd", cap, now)
            return

        if self._rolled_quote_volume(validator, book_id, now) >= cap:
            self._snap(validator, book_id, mid, bid, ask, pos, account, taker_bps, "cap", cap, now)
            return

        cycle_ok = self._rt_cycle_ok(st, now)
        if not cycle_ok and not activity_due:
            self._snap(validator, book_id, mid, bid, ask, pos, account, taker_bps, "rt_cycle", cap, now)
            return

        if activity_due:
            action = "activity_rt"
        elif taker_bps <= self.strong_rebate_bps:
            action = "rebate_strong"
        else:
            action = "rebate_cycle"

        if self._taker_buy(response, account, book_id, book, pos, st, now, taker_bps):
            self._snap(validator, book_id, mid, bid, ask, pos, account, taker_bps, action, cap, now)

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

    def _snap(self, validator, book_id, mid, bid, ask, pos, account, taker_bps,
              action, cap, now, gross_bps: float = 0.0, hold_s: float = 0.0) -> None:
        traded = self._rolled_quote_volume(validator, book_id, now)
        exp_rebate = self._expected_rt_rebate_bps(account)
        self.telemetry.snapshot(
            book_id=book_id, mid=mid, bid=bid, ask=ask,
            pos_qty=pos.qty, pos_avg=pos.avg,
            base_bal=account.base_balance.total if account.base_balance else None,
            quote_bal=account.quote_balance.total if account.quote_balance else None,
            traded_volume=traded, volume_cap=cap, volume_remaining=max(0.0, cap - traded),
            signals={
                "taker_bps": taker_bps if taker_bps is not None else 0.0,
                "exp_rt_rebate_bps": exp_rebate,
                "gross_bps": gross_bps,
                "hold_s": hold_s,
            },
            action=action,
        )


if __name__ == "__main__":
    launch(RebateScalperAgent)
