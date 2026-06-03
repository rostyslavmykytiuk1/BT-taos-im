# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
MeanReversionAgent
==================

Contrarian maker scalper for Subnet 79 (MVTRX / taos).

Strategy
--------
Per book we maintain a short rolling mean of trade prices. When the mid has
stretched at least `stretch_bps` away from that mean AND the trade flow is
exhausting in the direction of the move, we fade it:

  * mid >> mean  -> SELL  (expect snap-back down)
  * mid << mean  -> BUY   (expect snap-back up)

Entries are **post-only limit orders** placed at (or just inside) the best
opposite level, so we tend to trade as a MAKER (0% base maker fee / DIS rebates),
improving realized PnL per round-trip. Every position is closed to realize PnL
on the first of:

  * profit  >= tp_bps
  * loss    <= -sl_bps
  * holding >  max_hold_s

This book is largely uncorrelated with a momentum book, which helps the
cross-book consistency requirement when running a fleet.

Run (local proxy test):
  python MeanReversionAgent.py --port 8902 --agent_id 0 \
    --params quote_notional=1500 tp_bps=14 sl_bps=20 max_hold_s=180 \
             stretch_bps=18 mean_window_s=120
"""

import traceback
from collections import deque
from dataclasses import dataclass

import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse
from taos.im.protocol.events import TradeEvent, SimulationStartEvent
from taos.im.protocol.models import OrderDirection, OrderCurrency, STP, TimeInForce


@dataclass
class _Position:
    qty: float = 0.0      # signed BASE (>0 long, <0 short)
    avg: float = 0.0      # volume-weighted average entry price
    entry_ts: int = 0


class MeanReversionAgent(FinanceSimulationAgent):
    # ------------------------------------------------------------------ setup
    def initialize(self) -> None:
        bt.logging.set_info()

        self.quote_notional = self._param("quote_notional", 1500.0)
        self.tp_bps = self._param("tp_bps", 14.0)
        self.sl_bps = self._param("sl_bps", 20.0)
        self.max_hold_s = self._param("max_hold_s", 180.0)
        self.stretch_bps = self._param("stretch_bps", 18.0)     # min deviation to fade
        self.mean_window_s = self._param("mean_window_s", 120.0)  # rolling mean window
        self.min_samples = int(self._param("min_samples", 8.0))
        self.imbalance_depth = int(self._param("imbalance_depth", 5.0))
        self.min_order_size = self._param("min_order_size", 0.25)
        self.expiry_s = self._param("entry_expiry_s", 10.0)     # GTT entry expiry
        self.turnover_cap = self._param("capital_turnover_cap", 10.0)
        self.volume_safety = self._param("volume_safety", 0.6)

        jitter = ((self.uid * 2654435761) % 1000) / 1000.0
        self.stretch_bps *= 0.9 + 0.2 * jitter

        self.positions: dict[str, dict[int, _Position]] = {}
        self.prices: dict[str, dict[int, deque]] = {}
        self._sim_id: dict[str, str] = {}

        bt.logging.info(
            f"[MeanReversion uid={self.uid}] notional={self.quote_notional} "
            f"tp={self.tp_bps} sl={self.sl_bps} hold={self.max_hold_s}s "
            f"stretch={self.stretch_bps:.2f}bps window={self.mean_window_s}s"
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
        self.prices.clear()
        self._sim_id.clear()
        bt.logging.info(f"[MeanReversion uid={self.uid}] simulation start: reset state")

    def onTrade(self, event: TradeEvent, validator: str = None) -> None:
        if event.bookId is None:
            return
        if self.uid == event.takerAgentId:
            direction = OrderDirection.BUY if event.side == OrderDirection.BUY else OrderDirection.SELL
        elif self.uid == event.makerAgentId:
            direction = OrderDirection.SELL if event.side == OrderDirection.BUY else OrderDirection.BUY
        else:
            return
        self._apply_fill(validator, event.bookId, direction, event.quantity, event.price, event.timestamp)

    def _book_positions(self, validator: str) -> dict[int, _Position]:
        return self.positions.setdefault(validator, {})

    def _apply_fill(self, validator, book_id, direction, qty, price, ts) -> None:
        pos = self._book_positions(validator).setdefault(book_id, _Position())
        signed = qty if direction == OrderDirection.BUY else -qty
        prev = pos.qty
        if prev == 0 or (prev > 0) == (signed > 0):
            total = abs(prev) + qty
            pos.avg = (pos.avg * abs(prev) + price * qty) / total if total > 0 else price
            pos.qty = prev + signed
            if prev == 0:
                pos.entry_ts = ts
        else:
            pos.qty = prev + signed
            if abs(pos.qty) < 1e-12:
                pos.qty, pos.avg, pos.entry_ts = 0.0, 0.0, 0
            elif (prev > 0) != (pos.qty > 0):
                pos.avg, pos.entry_ts = price, ts

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _mid(book) -> float | None:
        if not book.bids or not book.asks:
            return None
        return 0.5 * (book.bids[0].price + book.asks[0].price)

    def _book_imbalance(self, book) -> float:
        bq = sum(l.quantity for l in book.bids[: self.imbalance_depth])
        aq = sum(l.quantity for l in book.asks[: self.imbalance_depth])
        denom = bq + aq
        return (bq - aq) / denom if denom > 0 else 0.0

    def _update_mean(self, validator, book_id, book, now) -> float | None:
        """Append this step's trades and return the rolling mean trade price."""
        buf = self.prices.setdefault(validator, {}).setdefault(book_id, deque())
        if book.events:
            for e in book.events:
                if getattr(e, "type", None) == "t":
                    buf.append((e.timestamp, e.price))
        cutoff = now - self.mean_window_s * 1e9
        while buf and buf[0][0] < cutoff:
            buf.popleft()
        if len(buf) < self.min_samples:
            return None
        return sum(p for _, p in buf) / len(buf)

    # ------------------------------------------------------------------ respond
    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        validator = state.dendrite.hotkey
        cfg = self.simulation_config

        if self._sim_id.get(validator) != cfg.simulation_id:
            self._book_positions(validator).clear()
            self.prices.setdefault(validator, {}).clear()
            self._sim_id[validator] = cfg.simulation_id

        price_dp = cfg.priceDecimals
        vol_dp = cfg.volumeDecimals
        hold_ns = self.max_hold_s * 1e9
        expiry_ns = int(self.expiry_s * 1e9)
        cap = self.turnover_cap * cfg.miner_wealth * self.volume_safety

        for book_id, book in state.books.items():
            try:
                self._handle_book(response, validator, book_id, book, price_dp,
                                  vol_dp, hold_ns, expiry_ns, cap, state.timestamp)
            except Exception as ex:
                bt.logging.warning(
                    f"[MeanReversion uid={self.uid}] book {book_id} error: {ex}\n"
                    f"{traceback.format_exc()}"
                )
        return response

    def _handle_book(self, response, validator, book_id, book, price_dp, vol_dp,
                     hold_ns, expiry_ns, cap, now) -> None:
        mean = self._update_mean(validator, book_id, book, now)
        mid = self._mid(book)
        if mid is None or mid <= 0:
            return

        account = self.accounts.get(book_id)
        if account is None:
            return

        pos = self._book_positions(validator).setdefault(book_id, _Position())

        # ---- 1) Manage an open position (market exit guarantees realization) ----
        if abs(pos.qty) >= self.min_order_size / 2 and pos.avg > 0:
            pnl_bps = ((mid - pos.avg) if pos.qty > 0 else (pos.avg - mid)) / pos.avg * 1e4
            timed_out = (now - pos.entry_ts) >= hold_ns if pos.entry_ts else False
            if pnl_bps >= self.tp_bps or pnl_bps <= -self.sl_bps or timed_out:
                self._flatten(response, account, book_id, pos, vol_dp)
            return

        # ---- 2) Flat: look for an over-extension to fade ----
        if mean is None:
            return
        traded = account.traded_volume or 0.0
        if traded >= cap:
            return

        dev_bps = (mid - mean) / mean * 1e4
        if abs(dev_bps) < self.stretch_bps:
            return

        imb = self._book_imbalance(book)
        # Fade only when the book is not still strongly pushing the same way:
        # the stretch is the primary signal, the imbalance gate just avoids
        # fading into a one-sided book that is likely to keep running.
        fade_short = dev_bps > 0 and imb <= 0.15   # over-extended up -> sell
        fade_long = dev_bps < 0 and imb >= -0.15   # over-extended down -> buy
        if not (fade_short or fade_long):
            return

        qty = round(self.quote_notional / mid, vol_dp)
        if qty < self.min_order_size:
            return

        if fade_long:
            # Buy passively at the best bid (post-only -> maker).
            price = round(book.bids[0].price, price_dp)
            if account.quote_balance.free < qty * price:
                return
            response.limit_order(book_id=book_id, direction=OrderDirection.BUY,
                                quantity=qty, price=price, postOnly=True,
                                timeInForce=TimeInForce.GTT, expiryPeriod=expiry_ns,
                                stp=STP.CANCEL_OLDEST)
        else:
            price = round(book.asks[0].price, price_dp)
            if account.base_balance.free < qty:
                return
            response.limit_order(book_id=book_id, direction=OrderDirection.SELL,
                                quantity=qty, price=price, postOnly=True,
                                timeInForce=TimeInForce.GTT, expiryPeriod=expiry_ns,
                                stp=STP.CANCEL_OLDEST)

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


if __name__ == "__main__":
    launch(MeanReversionAgent)
