"""
QuantTradingAgent
=================

Blank starting point for a custom Subnet 79 (MVTRX) trading strategy.

Subclass ``FinanceSimulationAgent`` and fill in the hooks below step by step.
Telemetry is wired but optional — set ``TAOS_TELEMETRY_ENABLED=0`` to disable.

Run (local proxy test):
  python QuantTradingAgent.py --port 8900 --agent_id 0

Deploy:
  AGENT_NAME=QuantTradingAgent ./run_miner.sh
"""

from __future__ import annotations

import traceback

import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol import FinanceAgentResponse, MarketSimulationStateUpdate
from taos.im.protocol.events import SimulationStartEvent, TradeEvent
from taos.im.telemetry import MinerTelemetry


class QuantTradingAgent(FinanceSimulationAgent):
    """Custom quant strategy — implement your algorithm in the marked sections."""

    # ------------------------------------------------------------------ setup
    def initialize(self) -> None:
        bt.logging.set_info()

        # Example params (override via AGENT_PARAMS / --params key=value ...)
        self.quote_notional = self._param("quote_notional", 1800.0)

        self.telemetry = MinerTelemetry.from_agent(self, agent_class="QuantTradingAgent")
        self._step_ts_ns: int = 0

        bt.logging.info(
            f"[QuantTrading uid={self.uid}] ready quote_notional={self.quote_notional}"
        )

    def _param(self, name: str, default: float) -> float:
        raw = getattr(self.config, name, default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------- sim events
    def onStart(self, event: SimulationStartEvent) -> None:
        bt.logging.info(f"[QuantTrading uid={self.uid}] simulation start")

    def onTrade(self, event: TradeEvent, validator: str | None = None) -> None:
        """Called when one of our orders fills. Track positions / PnL here."""
        pass

    # ---------------------------------------------------------------- respond
    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        self._step_ts_ns = state.timestamp
        self.telemetry.begin_step(state)

        instr_before = len(response.instructions)
        for book_id, book in state.books.items():
            try:
                self._handle_book(response, state.dendrite.hotkey, book_id, book)
            except Exception as ex:
                bt.logging.warning(
                    f"[QuantTrading uid={self.uid}] book {book_id} error: {ex}\n"
                    f"{traceback.format_exc()}"
                )

        self.telemetry.end_step(state, instructions=len(response.instructions) - instr_before)
        return response

    def _handle_book(
        self,
        response: FinanceAgentResponse,
        validator: str,
        book_id: int,
        book,
    ) -> None:
        """
        Per-book strategy logic. Called once per simulation step.

        Available on ``book``:
          book.bids[0].price / book.asks[0].price  — top of book
          book.last_trade                             — latest trade
          book.events                                 — tick history since last update

        Available on ``self.accounts[book_id]``:
          base_balance / quote_balance, open orders, fees

        Submit orders via ``response``:
          response.limit_order(...)
          response.market_order(...)
          response.cancel_order(book_id, order_id)
        """
        if not book.bids or not book.asks:
            return

        mid = 0.5 * (book.bids[0].price + book.asks[0].price)
        account = self.accounts.get(book_id)
        if account is None:
            return

        # TODO: your signal / entry / exit logic here
        _ = (mid, account, validator)

        self.telemetry.snapshot(
            book_id=book_id,
            mid=mid,
            bid=book.bids[0].price,
            ask=book.asks[0].price,
            action="hold",
        )


if __name__ == "__main__":
    launch(QuantTradingAgent)
