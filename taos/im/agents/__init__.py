# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
import os
import json
import msgpack
import traceback
import time
import bittensor as bt
from threading import Thread
from abc import ABC, abstractmethod
from taos.common.agents import SimulationAgent
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse, FinanceEventNotification
from taos.im.protocol.events import *
from taos.im.protocol.models import *
from taos.im.utils import duration_from_timestamp

@dataclass
class RollingWindow:
    """
    Rolling window configuration for price sampling.
    Attributes:
        min (int): Minimum number of samples required before signals are considered reliable.
        max (int): Maximum length of the rolling buffer (in samples).
        samples (int): If max is not used or multi-scale approach
        num_windows (int): Multi-scale approaches
        sampling_interval (int): timestamps per sample (sec or nanosec)
    
    Parameter Tuning Guidelines:
        - TODO
    """
    min: int
    max: int
    samples: int
    num_windows: int
    sampling_interval: int = 1

@dataclass
class Thresholds:
    """
    Threshold configuration for generating trading signals.

    Attributes:
        signal (float): 
        tolerance (float): 
        model (float): 

    Parameter Tuning Guidelines:
        - Increase signal_threshold to reduce trading frequency, focus on strong trends.
        - Increase tolerance to reduce overtrading in noisy markets.
        - Adjust model_threshold to filter out unreliable signals.
    """
    signal: float
    tolerance: float
    model: float

@dataclass
class TimestampedPrice:
    """Container for midquote prices with timestamps."""
    price: float
    timestamp: int = 0

@dataclass
class Positions:
    """Container for positions for different books"""
    open: bool = False
    direction: OrderDirection = OrderDirection.BUY
    amount: int = 0
    qty: float = 0
    
class Signals(IntEnum):
    """
    Enum to represent signals coming from the model

    Attributes:
        ENTRY (int): Open a new position
        EXIT (int): Close current position 
        HOLD (int): Hold or extend current position
        NOISE (int): Ignore
        -- Simplified
        BULLISH (int): Rising
        BEARISH (int): Falling
    """
    REVERSION=0
    MOMENTUM=1
    HOLD=3
    NOISE=4
    BULLISH=5
    BEARISH=6


# Base class for agents operating in intelligent market simulations
class FinanceSimulationAgent(SimulationAgent):
    def __init__(self, uid : int, config : object, log_dir : str | None = None) -> None:
        """
        Initializer method that sets up the agent's unique ID and configuration, and initializes common objects for storing agent data.

        Args:
            uid (int): The UID of the agent in the subnet.
            config (obj): Config object for the agent.

        Returns:
            None
        """
        if config is None:
            from types import SimpleNamespace
            config = SimpleNamespace()
        self.history = []
        # State snapshots kept for handlers that read self.history (e.g. last
        # bid/ask). 0 keeps none — set it on agents that never read history to
        # avoid retaining full state copies.
        self.history_len = int(getattr(config, "history_len", 10))
        self.accounts = {}
        # Optional market-snapshot emitter (OFF unless `market_snapshot_dir` param is set).
        # Dumps a compact per-book mid/spread/fee snapshot for ALL books each tick (throttled),
        # read by tools/market_monitor.py to decide the taker/maker regime from live
        # microstructure (leading) rather than validator scores (which lag ~20h).
        self._snapshot_dir = getattr(config, "market_snapshot_dir", None)
        self._snapshot_interval = float(getattr(config, "market_snapshot_interval_s", 5.0))
        self._snapshot_last : dict[str, float] = {}
        if not hasattr(config, "lazy_load"):
            config.lazy_load = False
        else:
            config.lazy_load = bool(config.lazy_load)
        super().__init__(uid, config, log_dir)

    def handle(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        return super().handle(state)    

    def process(self, notification: FinanceEventNotification) -> FinanceEventNotification:
        """
        Method to handle a new event notification.
        """
        notification.acknowledged = True
        if notification.event.type == 'EVENT_SIMULATION_END':
            self.onEnd(notification.event)
        return notification
    
    def update(self, state : MarketSimulationStateUpdate) -> None:
        """
        Method to update the stored agent data, print relevant state information and trigger handlers for reported events.

        Args:
            state (taos.im.protocol.MarketSimulationStateUpdate): The UID of the agent in the subnet.

        Returns:
            None
        """
        if self.history_len:
            self.history.append(state.model_copy())
            self.history = self.history[-self.history_len:]
        else:
            self.history = []
        self.simulation_config = state.config
        self.accounts = state.accounts[self.uid]
        self.events = state.notices[self.uid]
        
        simulation_ended = False
        update_text = ''
        update_text += "\n" + '-' * 50 + "\n"
        update_text += f'VALIDATOR : {state.dendrite.hotkey} | SIMULATION TIME : {duration_from_timestamp(state.timestamp)} (T={state.timestamp})' + "\n"
        update_text += '-' * 50 + "\n"
        if len(self.events) > 0:
            global_events = False
            for event in self.events:
                match event.type:
                    case"RESET_AGENTS" | "RA":
                        if not global_events:
                            update_text += 'GLOBAL EVENTS' + "\n"
                            update_text += '-' * 50 + "\n"                            
                            global_events = True
                        update_text += f"{event}" + "\n"
                    case "EVENT_SIMULATION_START" | "ESS":
                        if not global_events:
                            update_text += 'GLOBAL EVENTS' + "\n"
                            update_text += '-' * 50 + "\n"                            
                            global_events = True
                        update_text += f"{event}" + "\n"
                        self.onStart(event)
                    case "EVENT_SIMULATION_END" | "ESE":
                        if not global_events:
                            update_text += 'GLOBAL EVENTS' + "\n"
                            update_text += '-' * 50 + "\n"                            
                            global_events = True
                        update_text += f"{event}" + "\n"
                        simulation_ended = True
                    case _:
                        pass
            if global_events:                
                update_text += '-' * 50 + "\n"
        debug_text = update_text
        for book_id in range(self.simulation_config.book_count):
            debug_text += f"BOOK {book_id}" + "\n"            
            debug_text += '-' * 50 + "\n"
            debug_text += 'EVENTS' + "\n"
            debug_text += '-' * 50 + "\n"
            for event in self.events:
                if hasattr(event, 'bookId') and event.bookId == book_id:
                    if not event.type in ["EVENT_TRADE", "ET"]:
                        debug_text += f"{event}" + "\n"
                        update_text += f"BOOK {book_id} : {event}" + "\n"
                    match event.type:
                        case "RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT" | "RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET" | "RDPOL" | "RDPOM":
                            self.onOrderAccepted(event)
                        case "ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT" | "ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET" | "ERDPOL" | "ERDPOM":
                            self.onOrderRejected(event)
                        case "RESPONSE_DISTRIBUTED_CANCEL_ORDERS" | "RDCO":
                            for cancellation in event.cancellations:
                                self.onOrderCancelled(cancellation)
                        case "ERROR_RESPONSE_DISTRIBUTED_CANCEL_ORDERS" | "ERDCO":
                            for cancellation in event.cancellations:
                                self.onOrderCancellationFailed(cancellation)
                        case "RESPONSE_DISTRIBUTED_CLOSE_POSITIONS" | "RDCP":
                            for close in event.closes:
                                self.onPositionClosed(close)
                        case "ERROR_RESPONSE_DISTRIBUTED_CLOSE_POSITIONS" | "ERDCP":
                            for close in event.closes:
                                self.onPositionCloseFailed(close)
                        case "EVENT_TRADE" | "ET":
                            role = "taker" if self.uid == event.takerAgentId else "maker"
                            trade_text = f"{'BUY ' if event.side == 0 else 'SELL'} TRADE #{event.tradeId} : YOUR {'AGGRESSIVE' if role=='taker' else 'PASSIVE'} " + \
                                f"ORDER #{event.takerOrderId if role=='taker' else event.makerOrderId} (AGENT {event.takerAgentId if role=='taker' else event.makerAgentId}) " + \
                                f"MATCHED AGAINST #{event.makerOrderId if role=='taker' else event.takerOrderId} (AGENT {event.makerAgentId if role=='taker' else event.takerAgentId}) " + \
                                f"FOR {event.quantity}@{event.price} AT {duration_from_timestamp(event.timestamp)} (T={event.timestamp})"
                            debug_text += f"{trade_text}" + "\n"
                            update_text += f"BOOK {book_id} : {trade_text}" + "\n"
                            self.onTrade(event,state.dendrite.hotkey)
                        case _:
                            bt.logging.warning(f"Unknown event : {event}")
            if len(self.events) == 0: 
                debug_text += "NO EVENTS\n"
            debug_text += '-' * 50 + "\n"
            if not self.config.lazy_load:
                account= self.accounts[book_id]
                debug_text += f"TOP LEVELS" + "\n"
                debug_text += '-' * 50 + "\n"
                debug_text += ' | '.join([f"{level.quantity:.4f}@{level.price}" for level in reversed(state.books[book_id].bids[:5])]) + '||' + ' | '.join([f"{level.quantity:.4f}@{level.price}" for level in state.books[book_id].asks[:5]]) + "\n"
                debug_text += '-' * 50 + "\n"
                debug_text += 'BALANCES' + "\n"
                debug_text += '-' * 50 + "\n"
                debug_text += f"BASE  : TOTAL={account.base_balance.total:.8f} FREE={account.base_balance.free:.8f} RESERVED={account.base_balance.reserved:.8f} | LOAN={account.base_loan:.8f} COLLATERAL={account.base_collateral}" + "\n"
                debug_text += f"QUOTE : TOTAL={account.quote_balance.total:.8f} FREE={account.quote_balance.free:.8f} RESERVED={account.quote_balance.reserved:.8f} | LOAN={account.quote_loan:.8f} COLLATERAL={account.quote_collateral}" + "\n"
                if len(account.orders) > 0:
                    debug_text += '-' * 50 + "\n"
                    debug_text += 'ORDERS' + "\n"
                    debug_text += '-' * 50 + "\n"
                    for order in sorted(account.orders, key=lambda x: x.timestamp):
                        debug_text += f"#{order.id} : {'BUY ' if order.side == 0 else 'SELL'} {f'{1+order.leverage:.2f}x' if order.leverage > 0 else ''}{order.quantity}@{order.price} [PLACED AT {duration_from_timestamp(order.timestamp)} (T={order.timestamp})]" + "\n"
                if len(account.loans) > 0:
                    debug_text += '-' * 50 + "\n"
                    debug_text += 'LOANS' + "\n"
                    debug_text += '-' * 50 + "\n"
                    for order_id, loan in account.loans.items():
                        debug_text += f"#{order_id} : {loan}\n"
                if account.fees:
                    debug_text += '-' * 50 + "\n"
                    debug_text += f'FEES : TRADED {account.fees.volume_traded} | MAKER {account.fees.maker_fee_rate * 100}% | TAKER {account.fees.taker_fee_rate * 100}%' + "\n"
                    debug_text += '-' * 50 + "\n"
                debug_text += '-' * 50 + "\n"
        if simulation_ended:
            update_text += f"{event}" + "\n"
            update_text += '-' * 50 + "\n"
            self.onEnd(event)
        bt.logging.debug("." + debug_text)
        if bt.logging.current_state_value == 'Info':
            bt.logging.info("." + update_text)
        try:
            self._emit_market_snapshot(state)
        except Exception as _snap_ex:
            bt.logging.debug(f"market snapshot emit skipped: {_snap_ex}")

    def _emit_market_snapshot(self, state: MarketSimulationStateUpdate) -> None:
        """Write a compact per-book snapshot (mid / spread_bps / taker_fee_bps / maker_fee_bps)
        for ALL books to `market_snapshot_dir`, throttled to `market_snapshot_interval_s`.

        No-op unless `market_snapshot_dir` was passed in --agent.params, so every agent that does
        not opt in pays only a single getattr per tick. One atomically-written file per validator
        (`market_snapshot.<hotkey8>.json`); the latest from each validator's own sim. Consumed by
        tools/market_monitor.py for a LEADING taker/maker regime call (live fees+spread) instead of
        lagging ~20h validator scores.
        """
        out_dir = getattr(self, "_snapshot_dir", None)
        if not out_dir:
            return
        validator = state.dendrite.hotkey
        now_wall = time.time()
        if now_wall - self._snapshot_last.get(validator, 0.0) < self._snapshot_interval:
            return
        self._snapshot_last[validator] = now_wall

        books_out = []
        for book_id, book in state.books.items():
            try:
                if not book.bids or not book.asks:
                    continue
                bid = book.bids[0].price
                ask = book.asks[0].price
                if bid <= 0 or ask <= 0:
                    continue
                mid = 0.5 * (bid + ask)
                spread_bps = (ask - bid) / mid * 1e4 if mid > 0 else 0.0
                acct = self.accounts.get(book_id)
                fees = getattr(acct, "fees", None) if acct is not None else None
                tf = getattr(fees, "taker_fee_rate", None) if fees is not None else None
                mf = getattr(fees, "maker_fee_rate", None) if fees is not None else None
                books_out.append([
                    book_id, round(mid, 6), round(spread_bps, 3),
                    round(tf * 1e4, 4) if tf is not None else None,
                    round(mf * 1e4, 4) if mf is not None else None,
                ])
            except Exception:
                continue

        payload = {
            "ts_ns": int(state.timestamp),
            "wall": now_wall,
            "validator": validator,
            "book_count": getattr(self.simulation_config, "book_count", len(state.books)),
            "cols": ["book", "mid", "spread_bps", "taker_fee_bps", "maker_fee_bps"],
            "books": books_out,
        }
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"market_snapshot.{validator[:8]}.json")
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)

    # Handler functions for various simulation events, to be overridden in agent implementations.
    def onStart(self, event : SimulationStartEvent) -> None:
        """
        Handler for simulation start event.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.SimulationStartEvent): The event class representing start of simulation.

        Returns:
            None
        """
        pass

    def onOrderAccepted(self, event : OrderPlacementEvent) -> None:
        """
        Handler for event where order is accepted to the book by simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.OrderPlacementEvent): The event class representing order placement.

        Returns:
            None
        """
        pass

    def onOrderRejected(self, event : OrderPlacementEvent) -> None:
        """
        Handler for event where order is rejected for placement by simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.OrderPlacementEvent): The event class representing order rejection.

        Returns:
            None
        """
        match event.message:
            case 'EXCEEDING_MAX_ORDERS':
                bt.logging.warning(f"FAILED TO PLACE {'BUY' if event.side == 0 else 'SELL'} ORDER FOR {event.quantity}@{event.price if event.type.endswith('L') else 'MARKET'} ON BOOK {event.bookId} : You already have the maximum allowed number of open orders ({self.simulation_config.max_open_orders}) on this book.  You will not be able to place any more orders until you either cancel existing orders, or they are traded.")
            case 'EXCEEDING_LOAN':
                bt.logging.warning(f"FAILED TO PLACE {'BUY' if event.side == 0 else 'SELL'} ORDER FOR {event.quantity}@{event.price if event.type.endswith('L') else 'MARKET'} ON BOOK {event.bookId} : You have exceeded the maximum allowed loan value ({self.simulation_config.max_loan}) on this book.  You need to close positions using the close_position method to repay some of the loan amount before you can place more leveraged orders.")
            case 'DUAL_POSITION':
                bt.logging.warning(f"FAILED TO PLACE {'BUY' if event.side == 0 else 'SELL'} ORDER FOR {event.quantity}@{event.price if event.type.endswith('L') else 'MARKET'} ON BOOK {event.bookId} : You already have a margin position in the opposite direction.  You must close this position before you can take leverage on this side of the book.")
            case _:
                pass

    def onOrderCancelled(self, event : OrderCancellationEvent) -> None:
        """
        Handler for event where order is cancelled in the simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.OrderCancellationEvent): The event class representing order cancellation.

        Returns:
            None
        """
        pass

    def onOrderCancellationFailed(self, event : OrderCancellationEvent) -> None:
        """
        Handler for event where order cancellation request is rejected by the simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.OrderCancellationEvent): The event class representing order cancellation.

        Returns:
            None
        """
        pass
    
    def onPositionClosed(self, event : ClosePositionEvent) -> None:
        """
        Handler for event where position is closed in the simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.ClosePositionEvent): The event class representing position closure.

        Returns:
            None
        """
        pass

    def onPositionCloseFailed(self, event : ClosePositionEvent) -> None:
        """
        Handler for event where close position request is rejected by the simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.ClosePositionEvent): The event class representing position closure.

        Returns:
            None
        """
        pass

    def onTrade(self, event : TradeEvent, validator: str = None) -> None:
        """
        Handler for event where an order is traded in the simulator.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.TradeEvent): The event class representing a trade.
            validator: Validator identifier (optional)
        Returns:
            None
        """
        pass

    def onEnd(self, event : SimulationEndEvent) -> None:
        """
        Handler for simulation end event.  To be implemented by subclasses.

        Args:
            event (taos.im.protocol.events.SimulationEndEvent): The event class representing end of simulation.

        Returns:
            None
        """
        pass

    def respond(self, state : MarketSimulationStateUpdate) -> FinanceAgentResponse:
        """
        Abstract method for handling generation of response to new state update.  To be implemented by subclasses.

        Args:
            state (taos.im.protocol.MarketSimulationStateUpdate): The class representing the latest state of the simulation.

        Returns:
            taos.im.protocol.FinanceAgentResponse : The response which will be attached to the synapse for return to the querying validator.
        """
        ...

    def report(self, state : MarketSimulationStateUpdate, response : FinanceAgentResponse) -> None:
        """
        Method for reporting the latest simulation state and the response generated by the agent.

        Args:
            state (taos.im.protocol.MarketSimulationStateUpdate): The class representing the latest state of the simulation.
            response (taos.im.protocol.FinanceAgentResponse): The class representing the response of the agent.

        Returns:
            None
        """
        update_text = '-' * 50 + "\n"
        if len(response.instructions) > 0:
            update_text += 'INSTRUCTIONS' + "\n"
            update_text += '-' * 50 + "\n"
            for instruction in response.instructions:
                update_text += f"{instruction}" + "\n"
        else:
            update_text += 'NO INSTRUCTIONS TO SUBMIT' + "\n"
        update_text += '-' * 50
        bt.logging.info(".\n" + update_text)

from taos.im.utils.history import history, batch_history
class StateHistoryManager:
    """
    Manages the state history for market simulations, including reconstruction of L2 states,
    constructing and maintaining histories over multiple state updates, as well as
    saving and loading of history data for later retrieval.
    """

    def __init__(self, history_retention_mins: int, log_dir: str, depth: int = 21, parallel_workers: int = 0, save: bool = False):
        """
        Initialize the StateHistoryManager.

        Args:
            history_retention_mins (int): Retention period for history in minutes.
            log_dir (str): Directory to store history files.
            depth (int): Number of levels of the order book to retain in history.
            parallel_workers (int): Number of parallel processes for reconstruction; if 0, disables parallelization.
            save (bool): If True, persist history to file after each update.
        """
        self.history_retention_mins: int = history_retention_mins  # Retention duration for history data
        self.log_dir: str = log_dir  # Directory path for log and history files
        self.state_file: str = os.path.join(log_dir, "history.mp")  # Path to serialized state file

        self.gap: dict[str, dict[int, bool]] = {}  # Tracks gaps in validator/book histories
        self.depth: int = depth  # Number of book levels to store
        self.parallel_workers: int = parallel_workers  # Number of processes for parallel reconstruction
        self.parallel: bool = parallel_workers > 0  # Whether parallel processing is enabled

        self.should_save: bool = save  # Whether to automatically save history after updates
        self.saving: bool = False  # Flag: True if a save operation is in progress
        self.updating: bool = False  # Flag: True if an update operation is in progress

        self.last_snapshot: dict[str, dict[int, L2Snapshot]] = {}  # Last known snapshot per validator/book
        self.history: dict[str, dict[int, L2History]] = {}  # Full history per validator/book
        self.publish_interval: int = None # Publishing interval

        self.load()  # Attempt to load existing history from disk

    def update(self, state: MarketSimulationStateUpdate) -> None:
        """
        Update the history with a new market simulation state.

        Args:
            state (MarketSimulationStateUpdate): The latest simulation state to process.
        """
        self.updating = True
        self.publish_interval = state.config.publish_interval
        try:
            validator: str = state.dendrite.hotkey  # Validator identifier

            # Ensure internal structures for this validator exist
            if validator not in self.last_snapshot:
                self.last_snapshot[validator] = {}
            if validator not in self.history:
                self.history[validator] = {}
            if validator not in self.gap:
                self.gap[validator] = {}

            # Wait for any ongoing save operation to complete
            while self.saving:
                bt.logging.info("Waiting for history saving to complete...")
                time.sleep(0.5)

            snapshots: dict[int, L2Snapshot | None] = {}
            for book_id, book in state.books.items():
                snapshots[book_id] = self._prepare_snapshot(state, book)

            # If all snapshots were successfully prepared
            if all(snapshots.values()):
                bt.logging.info(f"Updating state history for {validator} at {duration_from_timestamp(state.timestamp)}...")
                start_time = time.time()

                # Parallel or sequential history reconstruction
                if self.parallel:
                    num_processes = min(self.parallel_workers, len(state.books))
                    if len(state.books) % num_processes != 0:
                        raise ValueError(f"`parallel_workers` ({self.parallel_workers}) must divide number of books ({len(state.books)}).")

                    # Divide books into batches for parallel processing
                    batch_size = len(state.books) // num_processes
                    batches = [list(state.books.keys())[i:i + batch_size] for i in range(0, len(state.books), batch_size)]
                    histories = batch_history(
                        {book_id: snapshot.model_dump() for book_id, snapshot in snapshots.items()},
                        {book_id: [event.model_dump() for event in book.events] for book_id, book in state.books.items()},
                        batches,
                        state.config.volumeDecimals
                    )
                else:
                    # Process sequentially
                    histories = {
                        book_id: history(
                            snapshot.model_dump(),
                            [event.model_dump() for event in state.books[book_id].events],
                            state.config.volumeDecimals
                        )
                        for book_id, snapshot in snapshots.items()
                    }

                # Validate and process histories
                processed_histories = {
                    book_id: state.books[book_id].process_history(
                        {t: L2Snapshot.model_validate(snapshot).sort(self.depth) for t, snapshot in hist.items()},
                        {t: TradeInfo.model_validate(trade) for t, trade in trades.items()},
                        state.timestamp,
                        state.config,
                        self.history_retention_mins,
                        self.depth
                    )
                    for book_id, (hist, trades) in histories.items()
                }

                # Update all books in the state
                for book_id, book in state.books.items():
                    history_obj, matched, discrepancies = processed_histories[book_id]
                    self._update_book_history(state, book, history_obj, matched, discrepancies)

                bt.logging.info(f"Updated State History ({time.time() - start_time:.2f}s)")

            if self.should_save:
                # Trigger asynchronous save
                self.save()

        except Exception as ex:
            bt.logging.error(f"Exception processing state update for {validator} at {duration_from_timestamp(state.timestamp)}: {ex}")
        finally:
            self.updating = False

    def update_async(self, state: MarketSimulationStateUpdate) -> None:
        """
        Update the history asynchronously in a separate thread.
        Allows non-blocking updates while other operations continue.

        Args:
            state (MarketSimulationStateUpdate): The latest simulation state to process.
        """
        if not self.updating:
            Thread(target=self.update, args=(state,), daemon=True, name=f'update_history_{state.timestamp}').start()

    def _prepare_snapshot(self, state: MarketSimulationStateUpdate, book: Book) -> L2Snapshot | None:
        """
        Prepare a snapshot for a specific book, handling gaps if necessary.

        Args:
            state (MarketSimulationStateUpdate): The current simulation state.
            book (Book): The book data to process.

        Returns:
            L2Snapshot | None: The prepared snapshot, or None if unavailable.
        """
        book_id: int = book.id
        validator: str = state.dendrite.hotkey
        snapshot: L2Snapshot | None = None

        try:
            # Detect gaps or inconsistencies in history
            has_gap = (
                (book_id not in self.gap[validator] or not self.gap[validator][book_id])
                and (
                    (book_id in self.history[validator] and self.history[validator][book_id].end != state.timestamp - state.config.publish_interval)
                    or (book_id in self.last_snapshot[validator] and self.last_snapshot[validator][book_id].timestamp != state.timestamp - state.config.publish_interval)
                )
            )
            if has_gap:
                # Clear outdated snapshots and handle small/large gaps
                if book_id in self.last_snapshot[validator]:
                    del self.last_snapshot[validator][book_id]
                if book_id in self.history[validator]:
                    if state.timestamp > self.history[validator][book_id].end and state.timestamp < self.history[validator][book_id].end + (self.history_retention_mins * 60_000_000_000) // 10:
                        self.gap[validator][book_id] = True
                        if book_id == 0:
                            bt.logging.warning(
                                f"VALI {validator}: Small gap detected in L2 history from {duration_from_timestamp(self.history[validator][book_id].end)} to {duration_from_timestamp(state.timestamp - state.config.publish_interval)}. Continuing history."
                            )
                    else:
                        self.gap[validator][book_id] = False
                        if book_id == 0:
                            bt.logging.warning(
                                f"VALI {validator}: Large gap detected in L2 history. Resetting history for book {book_id}."
                            )
                        del self.history[validator][book_id]

            # Determine appropriate snapshot to use
            if book_id in self.last_snapshot[validator]:
                if book_id not in self.history[validator]:
                    snapshot = self.last_snapshot[validator][book_id]
                else:
                    if self.history[validator][book_id].end == state.timestamp - state.config.publish_interval:
                        snapshot = list(self.history[validator][book_id].snapshots.values())[-1]
                    else:
                        snapshot = self.last_snapshot[validator][book_id]

            # Update last snapshot for this book
            self.last_snapshot[validator][book_id] = book.snapshot(state.timestamp)

        except Exception as ex:
            bt.logging.error(
                f"VALI {validator} BOOK {book_id}: Exception while processing state at {duration_from_timestamp(state.timestamp)} (T={state.timestamp}): {ex}\n{traceback.format_exc()}"
            )
        finally:
            return snapshot

    def _update_book_history(self, state: MarketSimulationStateUpdate, book: Book, history: L2History, matched: bool, discrepancies: list[str]) -> None:
        """
        Commit an updated book history to the state manager.

        Args:
            state (MarketSimulationStateUpdate): The current simulation state.
            book (Book): The book being updated.
            history (L2History): The reconstructed book history.
            matched (bool): Whether reconstructed snapshot matches published state.
            discrepancies (list[str]): List of discrepancies found during reconstruction.
        """
        book_id: int = book.id
        validator: str = state.dendrite.hotkey

        try:
            if book_id in self.last_snapshot[validator]:
                self.gap[validator][book_id] = False
                if book_id not in self.history[validator]:
                    self.history[validator][book_id] = history
                    if book_id == 0:
                        bt.logging.info(
                            f"VALI {validator}: Initialized new L2 history at {duration_from_timestamp(self.history[validator][book_id].end)} "
                            f"(Available: {duration_from_timestamp(self.history[validator][book_id].start)}-{duration_from_timestamp(self.history[validator][book_id].end)})"
                        )
                else:
                    if self.history[validator][book_id].end == state.timestamp - state.config.publish_interval:
                        self.history[validator][book_id] = self.history[validator][book_id].append(history)
                        if book_id == 0:
                            bt.logging.info(
                                f"VALI {validator}: Appended L2 history at {duration_from_timestamp(state.timestamp)} "
                                f"(Available: {duration_from_timestamp(self.history[validator][book_id].start)}-{duration_from_timestamp(self.history[validator][book_id].end)})"
                            )
                    else:
                        # Recover after history gap
                        self.history[validator][book_id] = self.history[validator][book_id].append(history)
                        if book_id == 0:
                            bt.logging.info(
                                f"VALI {validator}: Recovered after history gap at {duration_from_timestamp(state.timestamp)} "
                                f"(Available: {duration_from_timestamp(self.history[validator][book_id].start)}-{duration_from_timestamp(self.history[validator][book_id].end)})"
                            )

                if not matched:
                    bt.logging.error(
                        f"VALI {validator} BOOK {book_id}: Mismatch between reconstructed and published book state:\n" + "\n".join(discrepancies)
                    )

            # Always update the last snapshot
            self.last_snapshot[validator][book_id] = book.snapshot(state.timestamp)

        except Exception as ex:
            bt.logging.error(
                f"VALI {validator} BOOK {book_id}: Exception while processing state at {duration_from_timestamp(state.timestamp)} (T={state.timestamp}): {ex}\n{traceback.format_exc()}"
            )

    def __getitem__(self, validator: str) -> dict[int, L2History]:
        """
        Access the history for a specific validator.

        Args:
            validator (str): Validator identifier.

        Returns:
            dict[int, L2History]: Mapping of book IDs to their histories.
        """
        return self.history[validator]

    def __contains__(self, validator: str) -> bool:
        """
        Check if a validator has history stored.

        Args:
            validator (str): Validator identifier.

        Returns:
            bool: True if history exists, False otherwise.
        """
        return validator in self.history

    def serialize(self) -> dict[str, Any]:
        """
        Serialize the current state history into a dictionary.

        Returns:
            dict[str, Any]: Serialized representation of the state history.
        """
        return {
            "publish_interval" : self.publish_interval,
            "last_snapshot": {
                validator: {book_id: snapshot.model_dump() for book_id, snapshot in validator_snapshot.items()}
                for validator, validator_snapshot in self.last_snapshot.items()
            },
            "history": {
                validator: {
                    book_id: {
                        'snapshots': {t: snapshot.model_dump() for t, snapshot in history.snapshots.items()},
                        'trades': {t: trade.model_dump() for t, trade in history.trades.items()}
                    }
                    for book_id, history in validator_history.items()
                }
                for validator, validator_history in self.history.items()
            }
        }

    def populate(self, serialized: dict[str, Any]) -> None:
        """
        Populate the state history from a serialized dictionary.

        Args:
            serialized (dict[str, Any]): Serialized history data.
        """
        self.last_snapshot = {
            validator: {book_id: L2Snapshot.model_validate(snapshot) for book_id, snapshot in validator_snapshot.items()}
            for validator, validator_snapshot in serialized["last_snapshot"].items()
        }
        self.history = {
            validator: {
                book_id: L2History(
                    snapshots={t: L2Snapshot.model_validate(snapshot) for t, snapshot in history_data["snapshots"].items()},
                    trades={t: TradeInfo.model_validate(trade) for t, trade in history_data["trades"].items()},
                    retention_mins=self.history_retention_mins,
                    publish_interval=serialized["publish_interval"]
                )
                for book_id, history_data in validator_history.items()
            }
            for validator, validator_history in serialized["history"].items()
        }

    def _save(self) -> None:
        """
        Save the current state history to disk synchronously.
        """
        self.saving = True
        bt.logging.info("Saving history...")
        start_time = time.time()

        # Write serialized data to a temporary file
        with open(self.state_file + ".tmp", 'wb') as file:
            packed_data = msgpack.packb(self.serialize(), use_bin_type=True)
            file.write(packed_data)

        # Replace old state file with new one
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        os.rename(self.state_file + ".tmp", self.state_file)

        bt.logging.info(f"History saved to {self.state_file} ({time.time() - start_time:.2f}s)")
        self.saving = False

    def save(self) -> None:
        """
        Save the state history asynchronously in a separate thread.
        """
        if not self.saving:
            Thread(target=self._save, daemon=True, name='save_history').start()

    def load(self) -> None:
        """
        Load the state history from disk if available, otherwise initialize empty structures.
        """
        if os.path.exists(self.state_file):
            bt.logging.info(f"Loading history from {self.state_file}...")
            with open(self.state_file, 'rb') as file:
                byte_data = file.read()
            state_data = msgpack.unpackb(byte_data, use_list=False, strict_map_key=False)
            self.populate(state_data)
            bt.logging.success("Loaded history! Available data:")
            for validator, validator_history in self.history.items():
                for book_id, book_history in validator_history.items():
                    bt.logging.info(
                        f"VALI {validator} BOOK {book_id}: {duration_from_timestamp(book_history.start)} - {duration_from_timestamp(book_history.end)}"
                    )
        else:
            # No saved state found; start fresh
            bt.logging.info(f"No history file found at {self.state_file}. Initializing empty history.")
            self.last_snapshot = {}
            self.history = {}

# GenTRXAgent extends FinanceSimulationAgent with distributed training.
# Import is deferred to end of file to avoid the circular-import that would
# result from gentrx.py importing FinanceSimulationAgent from this module.
try:
    from taos.im.agents.gentrx import GenTRXAgent  # noqa: F401
except ImportError:
    pass  # GenTRX optional dependencies (torch, boto3, etc.) not installed
