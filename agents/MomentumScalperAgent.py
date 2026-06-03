# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
MomentumScalperAgent
====================

Directional taker scalper for Subnet 79 (MVTRX / taos).

Strategy
--------
Per book, each step we build a fast signal that requires AGREEMENT between two
independent views of short-horizon direction:

  1. Trend   : sign of the change across recent trade prices (this-step events).
  2. Pressure: top-of-book depth imbalance combined with signed trade flow.

Only when both agree and the trend magnitude clears `signal_bps` do we open a
small position in the trend direction with a market order. Every open position
is then closed (realizing PnL, which is the only thing Kappa-3 scores) on the
first of:

  * profit  >= tp_bps
  * loss    <= -sl_bps           (kept tight to keep the cubed-downside tail short)
  * holding >  max_hold_s        (recycle capital, keep books Kappa-eligible)

Design priorities: realize many small consistent round-trips on (almost) every
book, keep losers tiny, stay well under the volume cap, never let one book throw.

Run (local proxy test):
  python MomentumScalperAgent.py --port 8901 --agent_id 0 \
    --params quote_notional=2200 tp_bps=15 sl_bps=13 max_hold_s=90 \
             signal_bps=9 stretch_bps=12 mean_window_s=90 min_flow_mult=0.5 \
             cooldown_s=45 max_taker_fee=0.0003 imbalance_depth=5
"""

import traceback
from collections import deque
from dataclasses import dataclass

import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.telemetry import MinerTelemetry
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse
from taos.im.protocol.events import TradeEvent, SimulationStartEvent
from taos.im.protocol.models import OrderDirection, OrderCurrency, STP


@dataclass
class _Position:
    """Per-book net position reconstructed from our own fills."""
    qty: float = 0.0      # signed BASE (>0 long, <0 short)
    avg: float = 0.0      # volume-weighted average entry price of current exposure
    entry_ts: int = 0     # sim timestamp (ns) at which current exposure opened


class MomentumScalperAgent(FinanceSimulationAgent):
    # ------------------------------------------------------------------ setup
    def initialize(self) -> None:
        bt.logging.set_info()

        # --- strategy parameters (all overridable via --params) ---
        self.quote_notional = self._param("quote_notional", 2200.0)   # QUOTE per entry
        self.tp_bps = self._param("tp_bps", 15.0)                     # take-profit (bps)
        self.sl_bps = self._param("sl_bps", 13.0)                     # stop-loss (bps) — tight tail
        self.max_hold_s = self._param("max_hold_s", 90.0)            # time stop (sim sec)
        self.signal_bps = self._param("signal_bps", 9.0)             # min trend to act (bps)
        self.stretch_bps = self._param("stretch_bps", 12.0)           # skip entries into stretched moves
        self.mean_window_s = self._param("mean_window_s", 90.0)
        self.min_samples = int(self._param("min_samples", 6.0))
        self.min_flow_mult = self._param("min_flow_mult", 0.5)     # min signed flow vs min order
        self.cooldown_s = self._param("cooldown_s", 45.0)            # pause book after stop-loss
        self.max_taker_fee = self._param("max_taker_fee", 0.0003)    # skip taker entries above this
        self.imbalance_depth = int(self._param("imbalance_depth", 5.0))
        self.min_imbalance = self._param("min_imbalance", 0.12)    # required book skew
        self.min_order_size = self._param("min_order_size", 0.25)   # BASE
        self.turnover_cap = self._param("capital_turnover_cap", 10.0)
        self.volume_safety = self._param("volume_safety", 0.55)     # stay under validator cap
        # Default matches validator scoring.activity.trade_volume_assessment_period (24h sim).
        self.volume_assessment_ns = int(
            self._param("volume_assessment_ns", 86_400_000_000_000)
        )

        # Small per-UID jitter so replicated fleet UIDs are not perfectly
        # correlated (helps the cross-book outlier picture across a fleet).
        jitter = ((self.uid * 2654435761) % 1000) / 1000.0  # deterministic in [0,1)
        self.signal_bps *= 0.9 + 0.2 * jitter

        # --- runtime state ---
        self.mean_window_ns = int(self.mean_window_s * 1e9)
        self.cooldown_ns = int(self.cooldown_s * 1e9)

        self.positions: dict[str, dict[int, _Position]] = {}
        self._trade_prices: dict[str, dict[int, deque]] = {}
        self._cooldown_until: dict[tuple[str, int], int] = {}
        self._sim_id: dict[str, str] = {}
        self._exit_reason: dict[tuple[str, int], str] = {}
        # (sim_ts_ns, quote_volume) per validator/book — state.accounts lacks traded_volume.
        self._volume_log: dict[str, dict[int, list[tuple[int, float]]]] = {}
        self.telemetry = MinerTelemetry.from_agent(self, agent_class="MomentumScalperAgent")

        bt.logging.info(
            f"[MomentumScalper uid={self.uid}] notional={self.quote_notional} "
            f"tp={self.tp_bps}bps sl={self.sl_bps}bps hold={self.max_hold_s}s "
            f"signal={self.signal_bps:.2f}bps stretch={self.stretch_bps:.1f}bps "
            f"cooldown={self.cooldown_s}s max_taker_fee={self.max_taker_fee}"
        )

    def _param(self, name: str, default: float) -> float:
        """Read a numeric param, falling back to `default` on missing/invalid."""
        val = getattr(self.config, name, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    # --------------------------------------------------------------- lifecycle
    def onStart(self, event: SimulationStartEvent) -> None:
        """New simulation run for some validator -> clear all tracked state."""
        self.positions.clear()
        self._trade_prices.clear()
        self._cooldown_until.clear()
        self._sim_id.clear()
        self._exit_reason.clear()
        self._volume_log.clear()
        bt.logging.info(f"[MomentumScalper uid={self.uid}] simulation start: reset state")

    def _record_trade_volume(self, validator: str, book_id: int, qty: float, price: float, ts_ns: int) -> None:
        vol = float(qty) * float(price)
        if vol <= 0:
            return
        self._volume_log.setdefault(validator, {}).setdefault(book_id, []).append((ts_ns, vol))

    def _rolled_quote_volume(self, validator: str, book_id: int, now_ns: int) -> float:
        entries = self._volume_log.get(validator, {}).get(book_id, [])
        if not entries:
            return 0.0
        cutoff = now_ns - self.volume_assessment_ns
        kept = [(t, v) for t, v in entries if t >= cutoff]
        self._volume_log[validator][book_id] = kept
        return sum(v for _, v in kept)

    def onTrade(self, event: TradeEvent, validator: str = None) -> None:
        """Reconstruct our position + cost basis from each of our own fills."""
        if event.bookId is None:
            return
        if self.uid == event.takerAgentId:
            direction = OrderDirection.BUY if event.side == OrderDirection.BUY else OrderDirection.SELL
        elif self.uid == event.makerAgentId:
            # Maker is the opposite side of the aggressor.
            direction = OrderDirection.SELL if event.side == OrderDirection.BUY else OrderDirection.BUY
        else:
            return
        self._record_trade_volume(validator, event.bookId, event.quantity, event.price, event.timestamp)
        self._apply_fill(validator, event.bookId, direction, event.quantity, event.price, event.timestamp)

    def _book_positions(self, validator: str) -> dict[int, _Position]:
        return self.positions.setdefault(validator, {})

    def _apply_fill(self, validator, book_id, direction, qty, price, ts) -> None:
        pos = self._book_positions(validator).setdefault(book_id, _Position())
        signed = qty if direction == OrderDirection.BUY else -qty
        prev = pos.qty
        entry_avg = pos.avg
        entry_ts = pos.entry_ts
        if prev == 0 or (prev > 0) == (signed > 0):
            # Opening or adding in the same direction -> blend the average.
            total = abs(prev) + qty
            pos.avg = (pos.avg * abs(prev) + price * qty) / total if total > 0 else price
            pos.qty = prev + signed
            if prev == 0:
                pos.entry_ts = ts
        else:
            # Reducing, closing or flipping.
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
                    book_id=book_id,
                    ts_close_ns=ts,
                    side=side,
                    qty=closed_qty,
                    entry_avg=entry_avg,
                    exit_avg=price,
                    realized_pnl=rpnl,
                    hold_s=hold_s,
                    reason=reason,
                )
            pos.qty = prev + signed
            if abs(pos.qty) < 1e-12:
                pos.qty, pos.avg, pos.entry_ts = 0.0, 0.0, 0
            elif (prev > 0) != (pos.qty > 0):
                # Flipped through zero: remainder is a fresh position here.
                pos.avg, pos.entry_ts = price, ts

    # ----------------------------------------------------------------- helpers
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

    def _update_trade_history(self, validator: str, book_id: int, book, now: int) -> None:
        dq = self._trade_prices.setdefault(validator, {}).setdefault(
            book_id, deque(maxlen=500)
        )
        for event in book.events or []:
            if getattr(event, "type", None) == "t" and event.price > 0:
                dq.append((now, float(event.price)))
        cutoff = now - self.mean_window_ns
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def _rolling_mean(self, validator: str, book_id: int) -> float | None:
        dq = self._trade_prices.get(validator, {}).get(book_id)
        if not dq or len(dq) < self.min_samples:
            return None
        return sum(p for _, p in dq) / len(dq)

    def _on_cooldown(self, validator: str, book_id: int, now: int) -> bool:
        until = self._cooldown_until.get((validator, book_id), 0)
        return until > now

    def _book_imbalance(self, book) -> float:
        """(bid_qty - ask_qty)/(bid_qty + ask_qty) over the top `imbalance_depth`."""
        bq = sum(l.quantity for l in book.bids[: self.imbalance_depth])
        aq = sum(l.quantity for l in book.asks[: self.imbalance_depth])
        denom = bq + aq
        return (bq - aq) / denom if denom > 0 else 0.0

    @staticmethod
    def _trend_bps(book) -> tuple[float, float]:
        """Return (trend in bps, signed trade-flow) from this step's trade events."""
        if not book.events:
            return 0.0, 0.0
        trades = [e for e in book.events if getattr(e, "type", None) == "t"]
        if len(trades) < 2:
            flow = sum((t.quantity if t.side == OrderDirection.BUY else -t.quantity)
                       for t in trades)
            return 0.0, flow
        first, last = trades[0].price, trades[-1].price
        trend = (last - first) / first * 1e4 if first > 0 else 0.0
        flow = sum((t.quantity if t.side == OrderDirection.BUY else -t.quantity) for t in trades)
        return trend, flow

    # ------------------------------------------------------------------ respond
    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        validator = state.dendrite.hotkey
        cfg = self.simulation_config

        # Reset if this validator started serving a different simulation.
        if self._sim_id.get(validator) != cfg.simulation_id:
            self._book_positions(validator).clear()
            self._trade_prices.pop(validator, None)
            self._cooldown_until = {
                k: v for k, v in self._cooldown_until.items() if k[0] != validator
            }
            self._volume_log.pop(validator, None)
            self._sim_id[validator] = cfg.simulation_id

        price_dp = cfg.priceDecimals
        vol_dp = cfg.volumeDecimals
        hold_ns = self.max_hold_s * 1e9
        cap = self.turnover_cap * cfg.miner_wealth * self.volume_safety

        self.telemetry.begin_step(state)
        instructions_before = len(response.instructions)
        for book_id, book in state.books.items():
            try:
                self._handle_book(response, validator, book_id, book,
                                  price_dp, vol_dp, hold_ns, cap, state.timestamp)
            except Exception as ex:  # never let one book break the whole response
                bt.logging.warning(
                    f"[MomentumScalper uid={self.uid}] book {book_id} error: {ex}\n"
                    f"{traceback.format_exc()}"
                )
        self.telemetry.end_step(
            state,
            instructions=len(response.instructions) - instructions_before,
        )
        return response

    def _handle_book(self, response, validator, book_id, book,
                     price_dp, vol_dp, hold_ns, cap, now) -> None:
        mid = self._mid(book)
        mark = self._microprice(book) or mid
        if mid is None or mid <= 0 or mark is None:
            return

        self._update_trade_history(validator, book_id, book, now)

        account = self.accounts.get(book_id)
        if account is None:
            return

        pos = self._book_positions(validator).setdefault(book_id, _Position())
        bid = book.bids[0].price if book.bids else None
        ask = book.asks[0].price if book.asks else None
        imb = self._book_imbalance(book)
        trend, flow = self._trend_bps(book)
        action = "hold"

        # ---- 1) Manage an open position: exit on tp / sl / time ----
        if abs(pos.qty) >= self.min_order_size / 2 and pos.avg > 0:
            if pos.qty > 0:
                pnl_bps = (mark - pos.avg) / pos.avg * 1e4
            else:
                pnl_bps = (pos.avg - mark) / pos.avg * 1e4
            timed_out = (now - pos.entry_ts) >= hold_ns if pos.entry_ts else False
            if pnl_bps >= self.tp_bps or pnl_bps <= -self.sl_bps or timed_out:
                if pnl_bps >= self.tp_bps:
                    reason = "tp"
                elif pnl_bps <= -self.sl_bps:
                    reason = "sl"
                    self._cooldown_until[(validator, book_id)] = now + self.cooldown_ns
                else:
                    reason = "time"
                self._exit_reason[(validator, book_id)] = reason
                action = "exit"
                self._flatten(response, account, book_id, pos, vol_dp)
            self._telemetry_snapshot(
                validator, book_id, mid, bid, ask, pos, account, trend, flow, imb, action, cap, now
            )
            return  # one action per book per step while managing a position

        # ---- 2) Flat: look for an entry, but respect the volume cap ----
        traded = self._rolled_quote_volume(validator, book_id, now)
        if traded >= cap:
            self._telemetry_snapshot(
                validator, book_id, mid, bid, ask, pos, account, trend, flow, imb, "cap", cap, now
            )
            return

        if self._on_cooldown(validator, book_id, now):
            self._telemetry_snapshot(
                validator, book_id, mid, bid, ask, pos, account, trend, flow, imb, "cooldown", cap, now
            )
            return

        if abs(trend) < self.signal_bps:
            self._telemetry_snapshot(
                validator, book_id, mid, bid, ask, pos, account, trend, flow, imb, action, cap, now
            )
            return

        min_flow = self.min_flow_mult * self.min_order_size
        if abs(flow) < min_flow:
            self._telemetry_snapshot(
                validator, book_id, mid, bid, ask, pos, account, trend, flow, imb, "weak_flow", cap, now
            )
            return

        fees = account.fees
        if fees is not None and fees.taker_fee_rate > self.max_taker_fee:
            self._telemetry_snapshot(
                validator, book_id, mid, bid, ask, pos, account, trend, flow, imb, "taker_fee", cap, now
            )
            return

        mean_px = self._rolling_mean(validator, book_id)
        stretch_up = mean_px * (1 + self.stretch_bps / 1e4) if mean_px else None
        stretch_dn = mean_px * (1 - self.stretch_bps / 1e4) if mean_px else None

        long_ok = trend > 0 and imb >= self.min_imbalance and flow >= min_flow
        short_ok = trend < 0 and imb <= -self.min_imbalance and flow <= -min_flow
        if stretch_up is not None and mid > stretch_up:
            long_ok = False
        if stretch_dn is not None and mid < stretch_dn:
            short_ok = False
        if not (long_ok or short_ok):
            self._telemetry_snapshot(
                validator, book_id, mid, bid, ask, pos, account, trend, flow, imb, action, cap, now
            )
            return

        direction = OrderDirection.BUY if long_ok else OrderDirection.SELL
        qty = round(self.quote_notional / mid, vol_dp)
        if qty < self.min_order_size:
            self._telemetry_snapshot(
                validator, book_id, mid, bid, ask, pos, account, trend, flow, imb, action, cap, now
            )
            return

        if direction == OrderDirection.BUY:
            if account.quote_balance.free < qty * book.asks[0].price:
                self._telemetry_snapshot(
                    validator, book_id, mid, bid, ask, pos, account, trend, flow, imb, "no_quote", cap, now
                )
                return
        else:
            if account.base_balance.free < qty:
                self._telemetry_snapshot(
                    validator, book_id, mid, bid, ask, pos, account, trend, flow, imb, "no_base", cap, now
                )
                return

        action = "enter_long" if long_ok else "enter_short"
        response.market_order(
            book_id=book_id,
            direction=direction,
            quantity=qty,
            currency=OrderCurrency.BASE,
            stp=STP.CANCEL_OLDEST,
        )
        self._telemetry_snapshot(
            validator, book_id, mid, bid, ask, pos, account, trend, flow, imb, action, cap, now
        )

    def _telemetry_snapshot(
        self,
        validator: str,
        book_id: int,
        mid,
        bid,
        ask,
        pos,
        account,
        trend,
        flow,
        imb,
        action: str,
        cap: float,
        now: int,
    ) -> None:
        traded = self._rolled_quote_volume(validator, book_id, now)
        remaining = max(0.0, cap - traded)
        self.telemetry.snapshot(
            book_id=book_id,
            mid=mid,
            bid=bid,
            ask=ask,
            pos_qty=pos.qty,
            pos_avg=pos.avg,
            base_bal=account.base_balance.total if account.base_balance else None,
            quote_bal=account.quote_balance.total if account.quote_balance else None,
            traded_volume=traded,
            volume_cap=cap,
            volume_remaining=remaining,
            signals={"trend_bps": trend, "flow": flow, "imb": imb},
            action=action,
        )

    def _flatten(self, response, account, book_id, pos, vol_dp) -> None:
        """Close the whole position with a market order to realize PnL."""
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


if __name__ == "__main__":
    launch(MomentumScalperAgent)
