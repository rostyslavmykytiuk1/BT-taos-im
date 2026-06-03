# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
AdaptiveMakerAgent
==================

Two-sided, inventory-flattened, DIS-fee-aware market maker for Subnet 79.

Strategy
--------
Per book the agent runs in one of two modes:

  * NEAR FLAT  -> quote BOTH sides just inside the spread with short-expiry GTT
                  limit orders, skewing quotes by book imbalance to lean against
                  building inventory. It will NOT post on a side while the maker
                  fee is above `max_maker_fee` (DIS-aware: prefer rebate regimes).

  * HOLDING    -> stop adding; work a passive CLOSING limit at `tp_bps` to bank
                  the spread as a maker, and hard-exit with a market order if the
                  position runs to -sl_bps or exceeds max_hold_s. Realizing these
                  round-trips is what feeds Kappa-3.

Stale quotes are cancelled and refreshed every step so the agent never drifts
away from the touch and never piles up toward `max_open_orders`.

This is the lowest-directional-risk of the three agents (smoothest Kappa) but
the most latency-sensitive — co-locate to avoid adverse selection.

Run (local proxy test):
  python AdaptiveMakerAgent.py --port 8903 --agent_id 0 \
    --params quote_notional=1200 tp_bps=10 sl_bps=16 max_hold_s=150 \
             max_maker_fee=0.0005 quote_expiry_s=5
"""

import traceback
from dataclasses import dataclass

import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse
from taos.im.protocol.events import TradeEvent, SimulationStartEvent
from taos.im.protocol.models import OrderDirection, OrderCurrency, STP, TimeInForce


@dataclass
class _Position:
    qty: float = 0.0
    avg: float = 0.0
    entry_ts: int = 0


class AdaptiveMakerAgent(FinanceSimulationAgent):
    # ------------------------------------------------------------------ setup
    def initialize(self) -> None:
        bt.logging.set_info()

        self.quote_notional = self._param("quote_notional", 1200.0)
        self.tp_bps = self._param("tp_bps", 10.0)
        self.sl_bps = self._param("sl_bps", 16.0)
        self.max_hold_s = self._param("max_hold_s", 150.0)
        self.quote_expiry_s = self._param("quote_expiry_s", 5.0)
        self.max_maker_fee = self._param("max_maker_fee", 0.0005)   # skip posting above this
        self.imbalance_depth = int(self._param("imbalance_depth", 5.0))
        self.inventory_band = self._param("inventory_band", 0.5)    # BASE; flat tolerance
        self.skew_strength = self._param("skew_strength", 0.5)      # 0..1 imbalance lean
        self.min_order_size = self._param("min_order_size", 0.25)
        self.turnover_cap = self._param("capital_turnover_cap", 10.0)
        self.volume_safety = self._param("volume_safety", 0.6)

        jitter = ((self.uid * 2654435761) % 1000) / 1000.0
        self.tp_bps *= 0.9 + 0.2 * jitter

        self.positions: dict[str, dict[int, _Position]] = {}
        self._sim_id: dict[str, str] = {}

        bt.logging.info(
            f"[AdaptiveMaker uid={self.uid}] notional={self.quote_notional} "
            f"tp={self.tp_bps:.2f} sl={self.sl_bps} hold={self.max_hold_s}s "
            f"max_maker_fee={self.max_maker_fee} expiry={self.quote_expiry_s}s"
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
        self._sim_id.clear()
        bt.logging.info(f"[AdaptiveMaker uid={self.uid}] simulation start: reset state")

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
    def _book_imbalance(self, book) -> float:
        bq = sum(l.quantity for l in book.bids[: self.imbalance_depth])
        aq = sum(l.quantity for l in book.asks[: self.imbalance_depth])
        denom = bq + aq
        return (bq - aq) / denom if denom > 0 else 0.0

    # ------------------------------------------------------------------ respond
    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        validator = state.dendrite.hotkey
        cfg = self.simulation_config

        if self._sim_id.get(validator) != cfg.simulation_id:
            self._book_positions(validator).clear()
            self._sim_id[validator] = cfg.simulation_id

        price_dp = cfg.priceDecimals
        vol_dp = cfg.volumeDecimals
        tick = 10 ** (-price_dp)
        hold_ns = self.max_hold_s * 1e9
        expiry_ns = int(self.quote_expiry_s * 1e9)
        cap = self.turnover_cap * cfg.miner_wealth * self.volume_safety

        for book_id, book in state.books.items():
            try:
                self._handle_book(response, validator, book_id, book, price_dp,
                                  vol_dp, tick, hold_ns, expiry_ns, cap, state.timestamp)
            except Exception as ex:
                bt.logging.warning(
                    f"[AdaptiveMaker uid={self.uid}] book {book_id} error: {ex}\n"
                    f"{traceback.format_exc()}"
                )
        return response

    def _handle_book(self, response, validator, book_id, book, price_dp, vol_dp,
                     tick, hold_ns, expiry_ns, cap, now) -> None:
        if not book.bids or not book.asks:
            return
        account = self.accounts.get(book_id)
        if account is None:
            return

        best_bid = book.bids[0].price
        best_ask = book.asks[0].price
        mid = 0.5 * (best_bid + best_ask)
        if mid <= 0:
            return

        pos = self._book_positions(validator).setdefault(book_id, _Position())

        # Refresh: cancel any of our resting orders so quotes never go stale and
        # we never accumulate toward max_open_orders.
        if account.orders:
            response.cancel_orders(book_id, [o.id for o in account.orders])

        qty = round(self.quote_notional / mid, vol_dp)
        if qty < self.min_order_size:
            return

        # ---- HOLDING: flatten inventory, banking the spread when possible ----
        if abs(pos.qty) > self.inventory_band and pos.avg > 0:
            pnl_bps = ((mid - pos.avg) if pos.qty > 0 else (pos.avg - mid)) / pos.avg * 1e4
            timed_out = (now - pos.entry_ts) >= hold_ns if pos.entry_ts else False
            if pnl_bps <= -self.sl_bps or timed_out:
                self._market_flatten(response, account, book_id, pos, vol_dp)
                return
            # Otherwise work a passive closing order at the take-profit level.
            close_qty = round(abs(pos.qty), vol_dp)
            if close_qty < self.min_order_size:
                return
            if pos.qty > 0:
                target = round(max(pos.avg * (1 + self.tp_bps / 1e4), best_ask), price_dp)
                if account.base_balance.free >= close_qty:
                    response.limit_order(book_id=book_id, direction=OrderDirection.SELL,
                                        quantity=close_qty, price=target, postOnly=True,
                                        timeInForce=TimeInForce.GTT, expiryPeriod=expiry_ns,
                                        stp=STP.CANCEL_OLDEST)
            else:
                target = round(min(pos.avg * (1 - self.tp_bps / 1e4), best_bid), price_dp)
                if account.quote_balance.free >= close_qty * target:
                    response.limit_order(book_id=book_id, direction=OrderDirection.BUY,
                                        quantity=close_qty, price=target, postOnly=True,
                                        timeInForce=TimeInForce.GTT, expiryPeriod=expiry_ns,
                                        stp=STP.CANCEL_OLDEST)
            return

        # ---- NEAR FLAT: quote both sides inside the spread (DIS-fee-aware) ----
        if account.fees and account.fees.maker_fee_rate > self.max_maker_fee:
            return  # maker fee too expensive right now -> sit out, stay flat
        traded = account.traded_volume or 0.0
        if traded >= cap:
            return

        imb = self._book_imbalance(book)
        # Place inside the spread when it is wide enough; otherwise join the touch.
        spread = best_ask - best_bid
        improve = tick if spread > 2 * tick else 0.0
        bid_price = round(best_bid + improve, price_dp)
        ask_price = round(best_ask - improve, price_dp)
        if bid_price >= ask_price:  # degenerate/locked book -> join the touch
            bid_price, ask_price = round(best_bid, price_dp), round(best_ask, price_dp)

        # Skew sizes against the imbalance: if book is bid-heavy (imb>0), lean to
        # quote more on the ask (sell) so we do not accumulate a long.
        lean = max(-1.0, min(1.0, imb)) * self.skew_strength
        bid_qty = round(qty * (1 - lean), vol_dp)
        ask_qty = round(qty * (1 + lean), vol_dp)

        if bid_qty >= self.min_order_size and account.quote_balance.free >= bid_qty * bid_price:
            response.limit_order(book_id=book_id, direction=OrderDirection.BUY,
                                quantity=bid_qty, price=bid_price, postOnly=True,
                                timeInForce=TimeInForce.GTT, expiryPeriod=expiry_ns,
                                stp=STP.CANCEL_OLDEST)
        if ask_qty >= self.min_order_size and account.base_balance.free >= ask_qty:
            response.limit_order(book_id=book_id, direction=OrderDirection.SELL,
                                quantity=ask_qty, price=ask_price, postOnly=True,
                                timeInForce=TimeInForce.GTT, expiryPeriod=expiry_ns,
                                stp=STP.CANCEL_OLDEST)

    def _market_flatten(self, response, account, book_id, pos, vol_dp) -> None:
        if pos.qty > 0:
            q = round(min(pos.qty, account.base_balance.free), vol_dp)
            if q >= self.min_order_size:
                response.market_order(book_id=book_id, direction=OrderDirection.SELL,
                                      quantity=q, currency=OrderCurrency.BASE,
                                      stp=STP.CANCEL_OLDEST)
        else:
            q = round(-pos.qty, vol_dp)
            if q >= self.min_order_size:
                response.market_order(book_id=book_id, direction=OrderDirection.BUY,
                                      quantity=q, currency=OrderCurrency.BASE,
                                      stp=STP.CANCEL_OLDEST)


if __name__ == "__main__":
    launch(AdaptiveMakerAgent)
