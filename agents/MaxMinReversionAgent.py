# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Mean-reversion agent using rolling max/min price bands.

Strategy:
- Sample mid price every 5 simulation seconds.
- Track max/min over a 40-minute rolling window.
- Buy when price < min + 0.3 * (max - min).
- Sell when price > max - 0.3 * (max - min).
- Monitor every resting limit order individually.
- Cancel buy when mid < order_price - 1.5 * (max - min).
- Cancel sell when mid > order_price + 1.5 * (max - min).
- Resting limit orders expire after 40 minutes (GTT).
- Minimum 20 minutes between new placements per side/book; interval resets on cancel.
- When base balance is depleted, place opposite BUY limits and take profit once
  mark-to-market PnL on the recovery position exceeds fees.
"""

from __future__ import annotations

import math
import traceback
from collections import defaultdict, deque
from typing import Deque, Dict, NamedTuple, Optional, Set, Tuple

import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol import FinanceAgentResponse, MarketSimulationStateUpdate
from taos.im.protocol.events import (
    LimitOrderPlacementEvent,
    OrderCancellationEvent,
    TradeEvent,
)
from taos.im.protocol.instructions import STP, TimeInForce
from taos.im.protocol.models import OrderDirection
from taos.im.utils import duration_from_timestamp

NS = 1_000_000_000
ORDER_QUANTITY = 1.0
RECOVERY_CLIENT_ID_BASE = 1_000_000


class PriceSample(NamedTuple):
    timestamp: int
    price: float


class TrackedOrder:
    __slots__ = (
        "order_id",
        "book_id",
        "placed_price",
        "direction",
        "placed_ts",
        "is_recovery",
    )

    def __init__(
        self,
        order_id: int,
        book_id: int,
        placed_price: float,
        direction: OrderDirection,
        placed_ts: int,
        is_recovery: bool = False,
    ) -> None:
        self.order_id = order_id
        self.book_id = book_id
        self.placed_price = placed_price
        self.direction = direction
        self.placed_ts = placed_ts
        self.is_recovery = is_recovery


class RecoveryPosition:
    """Tracks an opposite-side recovery leg after inventory is depleted."""

    __slots__ = (
        "entry_price",
        "quantity",
        "entry_ts",
        "client_order_id",
        "order_id",
        "pending",
    )

    def __init__(
        self,
        entry_price: float,
        quantity: float,
        entry_ts: int,
        client_order_id: int,
        order_id: Optional[int] = None,
        pending: bool = False,
    ) -> None:
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_ts = entry_ts
        self.client_order_id = client_order_id
        self.order_id = order_id
        self.pending = pending


class MaxMinReversionAgent(FinanceSimulationAgent):
    def initialize(self) -> None:
        self.sampling_interval_ns: int = int(
            getattr(self.config, "sampling_interval_ns", 5 * NS)
        )
        self.lookback_ns: int = int(
            getattr(self.config, "lookback_ns", 40 * 60 * NS)
        )
        self.order_expiry_ns: int = int(
            getattr(self.config, "order_expiry_ns", 40 * 60 * NS)
        )
        self.placement_interval_ns: int = int(
            getattr(self.config, "placement_interval_ns", 20 * 60 * NS)
        )
        self.band_fraction: float = float(getattr(self.config, "band_fraction", 0.3))
        self.adverse_range_multiplier: float = float(
            getattr(self.config, "adverse_range_multiplier", 1.5)
        )

        self.window_max_samples: int = max(
            1, self.lookback_ns // self.sampling_interval_ns
        )
        default_min_samples = min(120, self.window_max_samples)
        self.min_samples: int = int(
            getattr(self.config, "min_samples", default_min_samples)
        )

        self.price_history: Dict[str, Dict[int, Deque[PriceSample]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.window_max_samples))
        )
        self.last_sample_ts: Dict[str, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.tracked_orders: Dict[str, Dict[int, TrackedOrder]] = defaultdict(dict)
        self.placement_interval_anchor: Dict[str, Dict[int, Dict[str, int]]] = (
            defaultdict(lambda: defaultdict(lambda: {"buy": 0, "sell": 0}))
        )
        self.startup_cancelled: Dict[str, bool] = defaultdict(bool)
        self.recovery_state: Dict[str, Dict[int, Optional[RecoveryPosition]]] = (
            defaultdict(dict)
        )
        self.last_validator_ts: Dict[str, int] = defaultdict(int)
        self._current_validator: str = ""

    def update(self, state: MarketSimulationStateUpdate) -> None:
        self._current_validator = state.dendrite.hotkey
        super().update(state)

    def _mid_price(self, book) -> float | None:
        if not book.bids or not book.asks:
            return None
        return (book.bids[0].price + book.asks[0].price) / 2.0

    def _base_order_qty(self, decimals: int) -> float:
        factor = 10**decimals
        return math.floor(ORDER_QUANTITY * factor) / factor

    def _clip_quantity(self, qty: float, decimals: int) -> float:
        factor = 10**decimals
        return math.floor(qty * factor) / factor

    def _round_price(self, price: float, decimals: int) -> float:
        factor = 10**decimals
        return round(price * factor) / factor

    def _recovery_client_order_id(self, book_id: int) -> int:
        return RECOVERY_CLIENT_ID_BASE + book_id

    def _side_key(self, direction: OrderDirection) -> str:
        return "buy" if direction == OrderDirection.BUY else "sell"

    def _reset_validator_state(self, validator: str, now_ts: int) -> None:
        self.price_history[validator].clear()
        self.last_sample_ts[validator].clear()
        self.tracked_orders[validator].clear()
        self.placement_interval_anchor[validator].clear()
        self.recovery_state[validator].clear()
        self.startup_cancelled[validator] = False
        self.last_validator_ts[validator] = now_ts

    def _handle_timestamp_rollback(
        self, validator: str, now_ts: int
    ) -> None:
        prev_ts = self.last_validator_ts.get(validator, 0)
        if prev_ts and now_ts < prev_ts:
            bt.logging.warning(
                f"Validator timestamp rollback ({prev_ts} -> {now_ts}), resetting state."
            )
            self._reset_validator_state(validator, now_ts)
        else:
            self.last_validator_ts[validator] = now_ts

    def _update_price_history(
        self, validator: str, book_id: int, timestamp: int, price: float
    ) -> None:
        last_ts = self.last_sample_ts[validator][book_id]
        if last_ts and timestamp - last_ts < self.sampling_interval_ns:
            return

        history = self.price_history[validator][book_id]
        history.append(PriceSample(timestamp=timestamp, price=price))
        self.last_sample_ts[validator][book_id] = timestamp

        cutoff = timestamp - self.lookback_ns
        while history and history[0].timestamp < cutoff:
            history.popleft()

    def _window_extremes(
        self, history: Deque[PriceSample]
    ) -> Tuple[float, float] | None:
        if len(history) < self.min_samples:
            return None
        prices = [sample.price for sample in history]
        return min(prices), max(prices)

    def _effective_price_range(
        self, history: Deque[PriceSample], mid: float
    ) -> float:
        if not history:
            return mid * 1e-6 if mid > 0 else 1e-6
        prices = [sample.price for sample in history]
        return max(max(prices) - min(prices), mid * 1e-6 if mid > 0 else 1e-6)

    def _sync_tracked_orders(
        self, validator: str, book_id: int, account
    ) -> None:
        open_ids = {order.id for order in account.orders}
        for order_id, tracked in list(self.tracked_orders[validator].items()):
            if tracked.book_id == book_id and order_id not in open_ids:
                del self.tracked_orders[validator][order_id]

        recovery = self._get_recovery(validator, book_id)
        recovery_order_id = recovery.order_id if recovery else None

        for order in account.orders:
            if order.price is None:
                continue
            if order.id not in self.tracked_orders[validator]:
                self.tracked_orders[validator][order.id] = TrackedOrder(
                    order_id=order.id,
                    book_id=book_id,
                    placed_price=order.price,
                    direction=OrderDirection(order.side),
                    placed_ts=order.timestamp,
                    is_recovery=order.id == recovery_order_id,
                )

    def _reset_placement_interval(
        self,
        validator: str,
        book_id: int,
        timestamp: int,
        direction: Optional[OrderDirection] = None,
    ) -> None:
        if direction is None:
            self.placement_interval_anchor[validator][book_id]["buy"] = timestamp
            self.placement_interval_anchor[validator][book_id]["sell"] = timestamp
            return
        self.placement_interval_anchor[validator][book_id][
            self._side_key(direction)
        ] = timestamp

    def _can_place_order(
        self,
        validator: str,
        book_id: int,
        now_ts: int,
        direction: OrderDirection,
    ) -> bool:
        anchor = self.placement_interval_anchor[validator][book_id][
            self._side_key(direction)
        ]
        if anchor == 0:
            return True
        return now_ts - anchor >= self.placement_interval_ns

    def _has_open_order(
        self,
        account,
        direction: OrderDirection,
        excluded_ids: Optional[Set[int]] = None,
    ) -> bool:
        excluded = excluded_ids or set()
        return any(
            order.side == direction and order.id not in excluded
            for order in account.orders
        )

    def _can_add_order(self, account, max_open_orders: Optional[int]) -> bool:
        if max_open_orders is None:
            return True
        return len(account.orders) < max_open_orders

    def _is_base_depleted(self, account, qty: float) -> bool:
        if account.base_balance.free >= qty:
            return False
        if self._has_open_order(account, OrderDirection.SELL):
            return False
        return True

    def _fee_buffer(self, account, mid: float, quantity: float) -> float:
        if not account.fees or quantity <= 0 or mid <= 0:
            return 0.0
        maker = abs(account.fees.maker_fee_rate)
        taker = abs(account.fees.taker_fee_rate)
        rate = max(maker, taker)
        return mid * quantity * rate * 2

    def _monitor_open_orders(
        self,
        response: FinanceAgentResponse,
        validator: str,
        book_id: int,
        account,
        mid: float,
        price_range: float,
        now_ts: int,
    ) -> Set[int]:
        adverse_delta = price_range * self.adverse_range_multiplier
        cancel_ids: list[int] = []

        for order in account.orders:
            if order.price is None:
                continue

            tracked = self.tracked_orders[validator].get(order.id)
            placed_price = tracked.placed_price if tracked else order.price

            if price_range > 0:
                if order.side == OrderDirection.SELL:
                    if mid > placed_price + adverse_delta:
                        cancel_ids.append(order.id)
                elif order.side == OrderDirection.BUY:
                    if mid < placed_price - adverse_delta:
                        cancel_ids.append(order.id)

            if now_ts - order.timestamp >= self.order_expiry_ns:
                cancel_ids.append(order.id)

        cancel_ids = sorted(set(cancel_ids))
        if cancel_ids:
            response.cancel_orders(book_id=book_id, order_ids=cancel_ids)
            for order_id in cancel_ids:
                tracked = self.tracked_orders[validator].pop(order_id, None)
                if tracked:
                    self._reset_placement_interval(
                        validator, book_id, now_ts, tracked.direction
                    )
            bt.logging.debug(
                f"BOOK {book_id}: cancelled orders {cancel_ids} "
                f"(mid={mid:.4f}, range={price_range:.4f})"
            )

        return set(cancel_ids)

    def _get_recovery(
        self, validator: str, book_id: int
    ) -> Optional[RecoveryPosition]:
        return self.recovery_state[validator].get(book_id)

    def _set_recovery(
        self,
        validator: str,
        book_id: int,
        recovery: Optional[RecoveryPosition],
    ) -> None:
        if recovery is None:
            self.recovery_state[validator].pop(book_id, None)
        else:
            self.recovery_state[validator][book_id] = recovery

    def _recovery_unrealized(
        self, recovery: RecoveryPosition, mid: float, fee_buffer: float
    ) -> float:
        return (mid - recovery.entry_price) * recovery.quantity - fee_buffer

    def _record_recovery_fill(
        self,
        validator: str,
        book_id: int,
        price: float,
        quantity: float,
        timestamp: int,
    ) -> None:
        recovery = self._get_recovery(validator, book_id)
        if recovery is None:
            return

        total_qty = recovery.quantity + quantity
        if total_qty > 0:
            recovery.entry_price = (
                (recovery.entry_price * recovery.quantity) + (price * quantity)
            ) / total_qty
        else:
            recovery.entry_price = price
        recovery.quantity = total_qty
        recovery.pending = False
        recovery.entry_ts = recovery.entry_ts or timestamp
        self._set_recovery(validator, book_id, recovery)

    def _is_recovery_trade(self, event: TradeEvent, recovery: RecoveryPosition) -> bool:
        if recovery.order_id is not None:
            if event.makerOrderId == recovery.order_id:
                return True
            if event.takerOrderId == recovery.order_id:
                return True
        if event.clientOrderId is not None:
            return event.clientOrderId == recovery.client_order_id
        return recovery.pending

    def _try_recovery_take_profit(
        self,
        response: FinanceAgentResponse,
        validator: str,
        book_id: int,
        account,
        mid: float,
        volume_decimals: int,
    ) -> bool:
        recovery = self._get_recovery(validator, book_id)
        if recovery is None or recovery.pending or recovery.quantity <= 0:
            return False

        fee_buffer = self._fee_buffer(account, mid, recovery.quantity)
        unrealized = self._recovery_unrealized(recovery, mid, fee_buffer)
        if unrealized <= 0:
            return False

        close_qty = self._clip_quantity(
            min(recovery.quantity, account.base_balance.free),
            volume_decimals,
        )
        if close_qty <= 0:
            return False

        if account.orders:
            response.cancel_orders(
                book_id=book_id,
                order_ids=[order.id for order in account.orders],
            )

        entry_price = recovery.entry_price
        response.market_order(
            book_id=book_id,
            direction=OrderDirection.SELL,
            quantity=close_qty,
        )
        recovery.quantity -= close_qty
        if recovery.quantity <= 1e-9:
            self._set_recovery(validator, book_id, None)
        else:
            self._set_recovery(validator, book_id, recovery)

        bt.logging.info(
            f"BOOK {book_id}: recovery take profit "
            f"unrealized={unrealized:.4f} qty={close_qty} entry={entry_price:.4f}"
        )
        return True

    def _cancel_non_recovery_buys(
        self,
        response: FinanceAgentResponse,
        validator: str,
        book_id: int,
        account,
        now_ts: int,
    ) -> Set[int]:
        recovery = self._get_recovery(validator, book_id)
        recovery_order_id = recovery.order_id if recovery else None
        cancel_ids = [
            order.id
            for order in account.orders
            if order.side == OrderDirection.BUY and order.id != recovery_order_id
        ]
        if cancel_ids:
            response.cancel_orders(book_id=book_id, order_ids=cancel_ids)
            for order_id in cancel_ids:
                tracked = self.tracked_orders[validator].pop(order_id, None)
                if tracked:
                    self._reset_placement_interval(
                        validator, book_id, now_ts, OrderDirection.BUY
                    )
        return set(cancel_ids)

    def _try_recovery_entry(
        self,
        response: FinanceAgentResponse,
        validator: str,
        book_id: int,
        account,
        book,
        qty: float,
        price_decimals: int,
        now_ts: int,
        max_open_orders: Optional[int],
    ) -> bool:
        recovery = self._get_recovery(validator, book_id)
        if recovery is not None and (recovery.pending or recovery.quantity > 0):
            return False

        cancelled = self._cancel_non_recovery_buys(
            response, validator, book_id, account, now_ts
        )
        excluded = cancelled
        if self._has_open_order(account, OrderDirection.BUY, excluded):
            return False
        if not self._can_add_order(account, max_open_orders):
            return False

        price = self._round_price(book.bids[0].price, price_decimals)
        if account.quote_balance.free < qty * price:
            return False

        client_order_id = self._recovery_client_order_id(book_id)
        response.limit_order(
            book_id=book_id,
            direction=OrderDirection.BUY,
            quantity=qty,
            price=price,
            clientOrderId=client_order_id,
            stp=STP.CANCEL_BOTH,
            timeInForce=TimeInForce.GTT,
            expiryPeriod=self.order_expiry_ns,
        )
        self._set_recovery(
            validator,
            book_id,
            RecoveryPosition(
                entry_price=price,
                quantity=0.0,
                entry_ts=now_ts,
                client_order_id=client_order_id,
                pending=True,
            ),
        )
        bt.logging.info(
            f"BOOK {book_id}: base depleted, placed recovery BUY {qty}@{price}"
        )
        return True

    def _in_recovery_mode(self, validator: str, book_id: int) -> bool:
        recovery = self._get_recovery(validator, book_id)
        return recovery is not None and (recovery.pending or recovery.quantity > 0)

    def onOrderAccepted(self, event: LimitOrderPlacementEvent) -> None:
        if (
            event.bookId is None
            or event.orderId is None
            or not event.success
            or not hasattr(event, "price")
        ):
            return

        validator = self._current_validator
        book_id = event.bookId
        recovery = self._get_recovery(validator, book_id)
        is_recovery = (
            recovery is not None
            and (
                event.clientOrderId == recovery.client_order_id
                or (
                    recovery.pending
                    and event.side == OrderDirection.BUY
                    and recovery.order_id is None
                )
            )
        )

        if is_recovery and recovery is not None:
            recovery.order_id = event.orderId
            recovery.pending = True
            self._set_recovery(validator, book_id, recovery)

        self.tracked_orders[validator][event.orderId] = TrackedOrder(
            order_id=event.orderId,
            book_id=book_id,
            placed_price=event.price,
            direction=OrderDirection(event.side),
            placed_ts=event.timestamp,
            is_recovery=is_recovery,
        )

    def onOrderCancelled(self, event: OrderCancellationEvent) -> None:
        if not event.success:
            return

        validator = self._current_validator
        tracked = self.tracked_orders[validator].pop(event.orderId, None)
        if tracked:
            self._reset_placement_interval(
                validator, event.bookId, event.timestamp, tracked.direction
            )

        recovery = self._get_recovery(validator, event.bookId)
        if recovery and (
            event.orderId == recovery.order_id
            or (recovery.pending and recovery.quantity <= 0)
        ):
            self._set_recovery(validator, event.bookId, None)

    def onTrade(self, event: TradeEvent, validator: str = None) -> None:
        if event.bookId is None:
            return
        if event.makerAgentId != self.uid and event.takerAgentId != self.uid:
            return

        validator = validator or self._current_validator
        book_id = event.bookId
        is_maker = event.makerAgentId == self.uid
        is_taker = event.takerAgentId == self.uid
        is_buy = (is_taker and event.side == 0) or (is_maker and event.side == 1)

        recovery = self._get_recovery(validator, book_id)
        if recovery is None or not is_buy:
            return
        if not self._is_recovery_trade(event, recovery):
            return

        self._record_recovery_fill(
            validator,
            book_id,
            event.price,
            event.quantity,
            event.timestamp,
        )

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        validator = state.dendrite.hotkey
        self._current_validator = validator
        now_ts = state.timestamp
        volume_decimals = state.config.volumeDecimals
        price_decimals = state.config.priceDecimals
        max_open_orders = state.config.max_open_orders

        self._handle_timestamp_rollback(validator, now_ts)

        if not self.startup_cancelled[validator]:
            for book_id in state.books:
                account = state.accounts.get(self.uid, {}).get(book_id)
                if account and account.orders:
                    response.cancel_orders(
                        book_id=book_id,
                        order_ids=[order.id for order in account.orders],
                    )
                    self._reset_placement_interval(validator, book_id, now_ts)
            self.tracked_orders[validator].clear()
            self.recovery_state[validator].clear()
            self.startup_cancelled[validator] = True

        for book_id, book in state.books.items():
            try:
                account = state.accounts.get(self.uid, {}).get(book_id)
                if not account:
                    continue

                mid = self._mid_price(book)
                if mid is None:
                    continue

                self._update_price_history(validator, book_id, now_ts, mid)
                self._sync_tracked_orders(validator, book_id, account)

                history = self.price_history[validator][book_id]
                price_range = self._effective_price_range(history, mid)

                cancelled_ids = self._monitor_open_orders(
                    response, validator, book_id, account, mid, price_range, now_ts
                )

                qty = self._base_order_qty(volume_decimals)

                if self._try_recovery_take_profit(
                    response, validator, book_id, account, mid, volume_decimals
                ):
                    continue

                if self._is_base_depleted(account, qty):
                    self._try_recovery_entry(
                        response,
                        validator,
                        book_id,
                        account,
                        book,
                        qty,
                        price_decimals,
                        now_ts,
                        max_open_orders,
                    )
                    continue

                if self._in_recovery_mode(validator, book_id):
                    continue

                extremes = self._window_extremes(history)
                if extremes is None:
                    continue

                window_min, window_max = extremes
                band_range = window_max - window_min
                if band_range <= 0:
                    continue

                buy_threshold = window_min + self.band_fraction * band_range
                sell_threshold = window_max - self.band_fraction * band_range

                if mid < buy_threshold and not self._has_open_order(
                    account, OrderDirection.BUY, cancelled_ids
                ):
                    if not self._can_place_order(
                        validator, book_id, now_ts, OrderDirection.BUY
                    ):
                        continue
                    if not self._can_add_order(account, max_open_orders):
                        continue
                    price = self._round_price(book.bids[0].price, price_decimals)
                    if account.quote_balance.free >= qty * price:
                        response.limit_order(
                            book_id=book_id,
                            direction=OrderDirection.BUY,
                            quantity=qty,
                            price=price,
                            stp=STP.CANCEL_BOTH,
                            timeInForce=TimeInForce.GTT,
                            expiryPeriod=self.order_expiry_ns,
                        )
                        self._reset_placement_interval(
                            validator, book_id, now_ts, OrderDirection.BUY
                        )
                        bt.logging.debug(
                            f"BOOK {book_id}: BUY limit {qty}@{price} "
                            f"(mid={mid:.4f} < {buy_threshold:.4f}, "
                            f"window=[{window_min:.4f}, {window_max:.4f}])"
                        )

                elif mid > sell_threshold and not self._has_open_order(
                    account, OrderDirection.SELL, cancelled_ids
                ):
                    if not self._can_place_order(
                        validator, book_id, now_ts, OrderDirection.SELL
                    ):
                        continue
                    if not self._can_add_order(account, max_open_orders):
                        continue
                    price = self._round_price(book.asks[0].price, price_decimals)
                    if account.base_balance.free >= qty:
                        response.limit_order(
                            book_id=book_id,
                            direction=OrderDirection.SELL,
                            quantity=qty,
                            price=price,
                            stp=STP.CANCEL_BOTH,
                            timeInForce=TimeInForce.GTT,
                            expiryPeriod=self.order_expiry_ns,
                        )
                        self._reset_placement_interval(
                            validator, book_id, now_ts, OrderDirection.SELL
                        )
                        bt.logging.debug(
                            f"BOOK {book_id}: SELL limit {qty}@{price} "
                            f"(mid={mid:.4f} > {sell_threshold:.4f}, "
                            f"window=[{window_min:.4f}, {window_max:.4f}])"
                        )

            except Exception as ex:
                bt.logging.error(
                    f"VALI {validator} BOOK {book_id}: Exception at "
                    f"{duration_from_timestamp(now_ts)} (T={now_ts}): {ex}\n"
                    f"{traceback.format_exc()}"
                )

        return response


if __name__ == "__main__":
    launch(MaxMinReversionAgent)