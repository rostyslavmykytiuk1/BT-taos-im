"""
DualEdgeAgent — fee-adaptive maker/taker agent for Subnet 79 (taos)

Scoring (from validator reward.py / _match_trade_fifo):
  * Kappa-3 rewards the SHAPE of realized round-trip PnL; LPM3 cubes downside, so
    one realized loss is very expensive. pnl.impact = 0 and activity.impact = 0:
    PnL size, deployed capital, and trade volume do not score.
  * One round-trip per book per ~600s bucket pins activity_factor to 1.0; missing
    the bucket decays the book toward 0 — far worse than a tiny forced loss.
  * Realized PnL is FIFO and net of fees, using the fee stored at fill time.
  * A sell with no prior in-sim buy opens a SHORT (0 PnL, 0 round-trip), so we
    keep a long-only FIFO ladder of our in-sim buys and never sell beyond it; the
    Pareto base endowment is left as reserve (selling it would open a short).
  * max_instructions_per_book = 5; we emit <= 2 per book per tick.

Strategy (per book, every tick):
  1. Reconcile the ladder down to on-chain base (never up).
  2. Router: taker if the rebate covers the spread, else maker. A switch is
     committed only when the book has no open position.
  3. Activity, two-stage so the deadline close is rarely a loss:
       soft — past ACTIVITY_SOFT_FRAC of the window, sell at the no-loss break-even
              via IOC the instant the bid covers it;
       hard — at the deadline, force a min-lot round-trip with a guaranteed market
              fill even at a small loss (losing activity_factor is worse).
  4. Maker (post-only, long-only): bid accumulates below mid and deepens as
     inventory fills; ask harvests the oldest FIFO lots at the worst per-lot
     break-even (every FIFO prefix closes >= 0). The ask is not repriced up on a
     rally (it gets lifted) — only to stay >= break-even or back down after a drop.
  5. Taker: market-buy a clip when the rebate makes it +EV; exit with an IOC sell
     at break-even, sent only when the bid covers it. Else hold.

Run (local proxy test):  python DualEdgeAgent.py --port 8904 --agent_id 0
Deploy: set AGENT_NAME=DualEdgeAgent in the miner env, then pm2 restart <miner>.
"""

import traceback
from collections import deque
from dataclasses import dataclass, field

import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol import FinanceAgentResponse, MarketSimulationStateUpdate
from taos.im.protocol.events import OrderPlacementEvent, SimulationStartEvent, TradeEvent
from taos.im.protocol.models import OrderCurrency, OrderDirection, STP, TimeInForce

_NS = 1_000_000_000

# ===========================================================================
# Tunables — edit here, then pm2 restart the miner
# ===========================================================================

# --- size ---
EXCHANGE_MIN_ORDER_SIZE = 0.25     # sim minOrderSize floor
LOT = 0.30                         # clip per quote/scalp (~0.3, matches top agents)
MAX_INVENTORY_LOTS = 5             # max long lots (smaller bag => cheaper, lower break-even)
MAKER_INVENTORY_SKEW_BPS = 30.0    # extra bid depth as inventory fills (cheaper scale-in)

# --- dynamic-fee router (all thresholds in bps; rebate = negative fee) ---
ROUTER_TAKER_MARGIN_BPS = 1.0      # need (-taker_fee) >= half_spread + this for TAKER
MAKER_FEE_DEFENSIVE_BPS = 12.0     # maker fee above this -> stop accumulating
ASSUMED_FEE_BPS = 2.3              # fallback per-side fee if account.fees missing

# --- maker engine ---
MAKER_MIN_HALF_SPREAD_BPS = 8.0    # min distance of a resting quote from mid
MAKER_EDGE_MARGIN_BPS = 4.0        # quote width on top of the maker fee
MAKER_HARVEST_BUFFER_BPS = 3.0     # cushion above break-even; absorbs fee rise before fill
MAKER_REPRICE_BPS = 8.0            # reprice a resting quote only past this drift
QUOTE_EXPIRY_S = 30.0              # GTT lifetime of resting maker quotes

# --- crash guard (do not accumulate into a fast dump) ---
CRASH_WINDOW_S = 20.0
CRASH_DROP_BPS = 30.0              # drop over the window that pauses accumulation

# --- taker engine ---
TAKER_REBATE_GATE_BPS = 1.0        # only take when taker rate <= -this (>=1bp rebate)
TAKER_EDGE_MARGIN_BPS = 1.0        # min +EV round-trip estimate (bps of notional)
TAKER_MIN_HOLD_S = 1.5             # min dwell before attempting the no-loss exit
TAKER_MIN_REOPEN_GAP_S = 4.0       # throttle: min gap between a close and next open
PENDING_TIMEOUT_S = 5.0            # assume a market order is lost after this

# --- activity (miss the ~600s bucket and the book decays; 480s leaves jitter margin) ---
PING_INTERVAL_S = 480.0
PING_SUBMIT_COOLDOWN_S = 5.0       # min gap between ping orders on a book
ACTIVITY_SOFT_FRAC = 0.6           # fraction of window before soft no-loss IOC banking

# --- logging ---
LOG_HEARTBEAT_S = 120.0           # per-book state heartbeat cadence

# --- volume cap (safety; volume is not scored here) ---
CAPITAL_TURNOVER_CAP = 10.0
VOLUME_SAFETY = 0.5
VOLUME_ASSESSMENT_NS = 86_400_000_000_000   # rolling 24h

MODE_MAKER = "maker"
MODE_TAKER = "taker"


@dataclass
class _Position:
    """FIFO ladder of in-sim buys. Each lot is [price, qty, fee] (fee at buy time,
    stored so the no-loss harvest is exact under dynamic fees)."""
    lots: deque = field(default_factory=deque)

    @property
    def qty(self) -> float:
        return sum(lot[1] for lot in self.lots)

    def add(self, price: float, qty: float, fee: float) -> None:
        if qty > 0:
            self.lots.append([float(price), float(qty), float(fee)])

    def reduce(self, qty: float, price: float = 0.0) -> tuple[float, float, float]:
        """FIFO-consume `qty` from the front, shrinking each lot's fee pro rata.
        Returns (closed_qty, price_pnl, open_fee); net realized = price_pnl -
        open_fee - close_fee."""
        remaining = qty
        closed = 0.0
        price_pnl = 0.0
        open_fee = 0.0
        while remaining > 1e-12 and self.lots:
            lot = self.lots[0]
            take = min(remaining, lot[1])
            fee_take = lot[2] * (take / lot[1]) if lot[1] > 0 else 0.0
            price_pnl += (price - lot[0]) * take
            open_fee += fee_take
            lot[2] -= fee_take
            lot[1] -= take
            remaining -= take
            closed += take
            if lot[1] <= 1e-12:
                self.lots.popleft()
        return closed, price_pnl, open_fee

    def harvest_bundle(self, min_qty: float) -> tuple[float, float]:
        """Bundle the oldest FIFO lots until cumulative qty >= `min_qty`. Returns
        (qty, worst_unit_cost), worst_unit_cost = max(price + fee/qty) over the
        bundle (the highest per-base break-even). Pricing a sell at this level keeps
        every FIFO prefix >= break-even even on a partial fill, and absorbs sub-min
        dust. Returns (0, 0) when inventory cannot form a min clip."""
        cum_q = 0.0
        worst_unit = 0.0
        for price, q, fee in self.lots:
            if q <= 0:
                continue
            unit = price + fee / q
            if unit > worst_unit:
                worst_unit = unit
            cum_q += q
            if cum_q >= min_qty - 1e-12:
                return cum_q, worst_unit
        return 0.0, 0.0


@dataclass
class _BookState:
    mode: str = MODE_MAKER
    last_rt_ns: int = 0                # last completed round-trip (activity clock)
    seen_ns: int = 0                   # first-seen ts (activity clock pre-first-RT)
    last_ping_submit_ns: int = 0       # anti-burst on ping orders
    ping_awaiting_open: bool = False   # open-leg market buy in flight
    pending_ns: int = 0                # market order in flight (sequencing)
    last_close_ns: int = 0             # last taker close (reopen throttle)
    taker_open_ns: int = 0             # when the current taker long was opened
    mids: deque = field(default_factory=lambda: deque(maxlen=64))  # (ts, mid)
    vol_log: list = field(default_factory=list)   # (ts, quote_volume) for the cap
    # logging / diagnostics
    rt_count: int = 0                  # completed round-trips on this book
    cum_pnl: float = 0.0               # cumulative realized PnL (net of fees)
    sell_tag: str = ""                 # reason tag for the next reducing fill
    last_log_ns: int = 0               # last heartbeat emitted


class DualEdgeAgent(FinanceSimulationAgent):
    # ------------------------------------------------------------------ setup
    def initialize(self) -> None:
        bt.logging.set_info()

        self.min_order_size = LOT
        self.exch_min = EXCHANGE_MIN_ORDER_SIZE
        self._flat_eps = LOT / 2
        self._volume_decimals: int | None = None

        # Per-UID jitter (+/-8%) so a fleet does not hit the same threshold on the
        # same book at the same instant.
        jitter = ((self.uid * 2654435761) % 1000) / 1000.0
        self._jit = 0.92 + 0.16 * jitter

        self.max_inventory = MAX_INVENTORY_LOTS * LOT
        self.ping_interval_ns = int(PING_INTERVAL_S * self._jit * _NS)
        self.activity_soft_ns = int(PING_INTERVAL_S * ACTIVITY_SOFT_FRAC * self._jit * _NS)
        self.ping_cooldown_ns = int(PING_SUBMIT_COOLDOWN_S * _NS)
        self.quote_expiry_ns = int(QUOTE_EXPIRY_S * _NS)
        self.crash_window_ns = int(CRASH_WINDOW_S * _NS)
        self.crash_drop_bps = CRASH_DROP_BPS * self._jit
        self.taker_min_hold_ns = int(TAKER_MIN_HOLD_S * _NS)
        self.taker_reopen_gap_ns = int(TAKER_MIN_REOPEN_GAP_S * _NS)
        self.pending_timeout_ns = int(PENDING_TIMEOUT_S * _NS)
        self.volume_assessment_ns = VOLUME_ASSESSMENT_NS

        self.positions: dict[str, dict[int, _Position]] = {}
        self.books_state: dict[str, dict[int, _BookState]] = {}
        self._sim_id: dict[str, str] = {}
        self._step_ts_ns: int = 0
        self._active_validator: str | None = None

        bt.logging.info(
            f"[DualEdge uid={self.uid}] lot={LOT} max_inv={self.max_inventory:.2f} "
            f"router(taker_margin={ROUTER_TAKER_MARGIN_BPS}bps maker_def={MAKER_FEE_DEFENSIVE_BPS}bps) "
            f"maker(width>=fee+{MAKER_EDGE_MARGIN_BPS}bps harvest_buf={MAKER_HARVEST_BUFFER_BPS}bps) "
            f"taker(rebate>={TAKER_REBATE_GATE_BPS}bps no-loss-IOC-exit) "
            f"ping={PING_INTERVAL_S:.0f}s"
        )

    # --------------------------------------------------------------- lifecycle
    def onStart(self, event: SimulationStartEvent) -> None:
        # Reset only the active validator's state so a restart on one validator
        # never wipes another's live books. _sim_id is owned by _ensure_simulation.
        v = self._active_validator
        if v is None:
            self.positions.clear()
            self.books_state.clear()
            self._sim_id.clear()
        else:
            self._reset_validator(v)
        bt.logging.info(
            f"[DualEdge uid={self.uid}] simulation start: reset {'ALL' if v is None else v[:8]}"
        )

    def onEnd(self, event) -> None:
        """Log a per-validator summary and reset that validator's state."""
        v = self._active_validator
        if v is None:
            return
        books = self.books_state.get(v, {})
        total_rt = sum(st.rt_count for st in books.values())
        total_pnl = sum(st.cum_pnl for st in books.values())
        bt.logging.info(
            f"[DualEdge uid={self.uid}] simulation end {v[:8]}: books={len(books)} "
            f"total_rt={total_rt} total_realized_pnl={total_pnl:+.3f} — reset state"
        )
        self._reset_validator(v)

    def update(self, state: MarketSimulationStateUpdate) -> None:
        self._step_ts_ns = int(state.timestamp)
        self._active_validator = state.dendrite.hotkey
        self._ensure_simulation(self._active_validator, state.config.simulation_id)
        super().update(state)

    def _reset_validator(self, validator: str) -> None:
        """Drop per-book ladders and state for one validator (recreated lazily)."""
        self.positions.pop(validator, None)
        self.books_state.pop(validator, None)

    def _ensure_simulation(self, validator: str, simulation_id: str | None) -> None:
        if self._sim_id.get(validator) == simulation_id:
            return
        self._reset_validator(validator)
        if simulation_id is not None:
            self._sim_id[validator] = simulation_id
        else:
            self._sim_id.pop(validator, None)
        bt.logging.info(f"[DualEdge uid={self.uid}] new simulation {validator[:8]} {simulation_id}")

    # ------------------------------------------------------------- fill tracking
    def onTrade(self, event: TradeEvent, validator: str | None = None) -> None:
        """Record each fill (maker or taker leg) into the FIFO ladder; store the
        per-fill fee on buys for the exact no-loss floor."""
        if event.bookId is None:
            return
        if self.uid == event.takerAgentId:
            direction = OrderDirection.BUY if event.side == OrderDirection.BUY else OrderDirection.SELL
            fee = float(getattr(event, "takerFee", 0.0) or 0.0)
        elif self.uid == event.makerAgentId:
            # As resting maker we are the opposite side of the aggressor.
            direction = OrderDirection.SELL if event.side == OrderDirection.BUY else OrderDirection.BUY
            fee = float(getattr(event, "makerFee", 0.0) or 0.0)
        else:
            return
        validator = validator or self._active_validator
        if validator is None:
            return
        ts_ns = self._step_ts_ns or int(event.timestamp)
        self._record_trade_volume(validator, event.bookId, event.quantity, event.price, ts_ns)
        self._apply_fill(validator, event.bookId, direction, event.quantity, event.price, fee, ts_ns)

    def _apply_fill(self, validator, book_id, direction, qty, price, fee, ts) -> None:
        pos = self._book_positions(validator).setdefault(book_id, _Position())
        st = self._bstate(validator, book_id)
        st.pending_ns = 0   # a fill resolves any in-flight market order
        if direction == OrderDirection.BUY:
            pos.add(price, qty, fee)
            st.ping_awaiting_open = False
        else:
            closed, price_pnl, open_fee = pos.reduce(qty, price)
            if closed > 1e-12:
                # Any reducing fill (incl. a partial) is a round-trip.
                st.last_rt_ns = ts
                close_fee = fee * (closed / qty) if qty > 0 else fee
                net = price_pnl - open_fee - close_fee
                avg_entry = price - price_pnl / closed
                st.rt_count += 1
                st.cum_pnl += net
                tag = st.sell_tag or st.mode
                st.sell_tag = ""
                msg = (
                    f"[DualEdge uid={self.uid} RT] book={book_id} {tag} "
                    f"qty={closed:.4f} entry~{avg_entry:.2f} exit={price:.2f} "
                    f"net={net:+.5f} (open_fee={open_fee:+.4f} close_fee={close_fee:+.4f}) "
                    f"rt_n={st.rt_count} cum={st.cum_pnl:+.3f}"
                )
                if net < 0:
                    bt.logging.warning(msg + " <LOSS>")
                else:
                    bt.logging.info(msg)

    def onOrderRejected(self, event: OrderPlacementEvent) -> None:
        """Log rejections and clear any pending/ping flag the order was holding."""
        if event.bookId is None or not self._active_validator:
            return
        st = self._bstate(self._active_validator, event.bookId)
        st.pending_ns = 0
        st.ping_awaiting_open = False
        side = "BUY" if getattr(event, "side", 0) == 0 else "SELL"
        msg = getattr(event, "message", "") or ""
        bt.logging.warning(
            f"[DualEdge uid={self.uid} REJECT] book={event.bookId} {side} "
            f"{getattr(event, 'quantity', '?')}@{getattr(event, 'price', 'MKT')} : {msg}"
        )

    # ----------------------------------------------------------------- helpers
    def _book_positions(self, validator: str) -> dict[int, _Position]:
        return self.positions.setdefault(validator, {})

    def _bstate(self, validator: str, book_id: int) -> _BookState:
        return self.books_state.setdefault(validator, {}).setdefault(book_id, _BookState())

    @staticmethod
    def _mid(book) -> float | None:
        if not book.bids or not book.asks:
            return None
        return 0.5 * (book.bids[0].price + book.asks[0].price)

    @staticmethod
    def _avail(balance) -> float:
        """Free + reserved (our resting orders are cancelled earlier in the same
        response, freeing their reserve for a market leg)."""
        if balance is None:
            return 0.0
        return (balance.free or 0.0) + (balance.reserved or 0.0)

    @staticmethod
    def _ceil_tick(px: float, price_dp: int) -> float:
        """Round up to the price grid so a break-even sell never rounds below cost."""
        tick = 10 ** (-price_dp)
        return round(-(-px // tick) * tick, price_dp)

    @staticmethod
    def _noloss_floor(worst_unit: float, close_rate_bps: float) -> float:
        """Lowest sell price netting >= 0 (plus buffer) on the worst lot after the
        close fee, so every FIFO prefix closes >= 0. close_rate_bps is the closing
        order's rate: maker for a resting ask, taker for an IOC (negative = rebate).
        net/unit = s*(1-c) - worst_unit >= 0  <=>  s >= worst_unit / (1-c)."""
        buffer = 1.0 + MAKER_HARVEST_BUFFER_BPS / 1e4
        return worst_unit * buffer / max(0.5, 1.0 - close_rate_bps / 1e4)

    def _maker_fee_bps(self, account) -> float:
        fees = getattr(account, "fees", None)
        rate = getattr(fees, "maker_fee_rate", None) if fees is not None else None
        try:
            return float(rate) * 1e4 if rate is not None else ASSUMED_FEE_BPS
        except (TypeError, ValueError):
            return ASSUMED_FEE_BPS

    def _taker_fee_bps(self, account) -> float:
        fees = getattr(account, "fees", None)
        rate = getattr(fees, "taker_fee_rate", None) if fees is not None else None
        try:
            return float(rate) * 1e4 if rate is not None else ASSUMED_FEE_BPS
        except (TypeError, ValueError):
            return ASSUMED_FEE_BPS

    # --------------------------------------------------------------- volume cap
    def _record_trade_volume(self, validator, book_id, qty, price, ts_ns) -> None:
        vol = float(qty) * float(price)
        if vol > 0:
            self._bstate(validator, book_id).vol_log.append((ts_ns, vol))

    def _rolled_quote_volume(self, st: _BookState, now_ns: int) -> float:
        cutoff = now_ns - self.volume_assessment_ns
        st.vol_log = [(t, v) for t, v in st.vol_log if t >= cutoff]
        return sum(v for _, v in st.vol_log)

    # --------------------------------------------------------------- features
    def _falling(self, st: _BookState, mid: float, now: int) -> bool:
        """True if mid dropped >= crash_drop_bps over the crash window (pause bids)."""
        cutoff = now - self.crash_window_ns
        hi = mid
        for t, m in st.mids:
            if t >= cutoff and m > hi:
                hi = m
        return hi > 0 and (hi - mid) / hi * 1e4 >= self.crash_drop_bps

    def _heartbeat(self, st: _BookState, book_id, inv, mid, maker_fee_bps,
                   taker_fee_bps, half_spread_bps, n_orders, now) -> None:
        """Throttled per-book diagnostic line."""
        if st.last_log_ns and (now - st.last_log_ns) < int(LOG_HEARTBEAT_S * _NS):
            return
        st.last_log_ns = now
        rt_age_s = (now - st.last_rt_ns) / _NS if st.last_rt_ns else -1.0
        bt.logging.info(
            f"[DualEdge uid={self.uid} HB] book={book_id} {st.mode} mid={mid:.2f} "
            f"inv={inv:.3f} orders={n_orders} fee(mk={maker_fee_bps:.1f}/tk={taker_fee_bps:.1f})bps "
            f"half_spread={half_spread_bps:.1f}bps last_rt={rt_age_s:.0f}s "
            f"rt_n={st.rt_count} cum_pnl={st.cum_pnl:+.3f}"
        )

    # --------------------------------------------------------------- reconcile
    def _reconcile(self, pos: _Position, account) -> None:
        """Clamp the ladder down to on-chain base so we never sell base we do not
        hold. Never seed up: the Pareto endowment is not an in-sim long (selling it
        would open a short)."""
        bal = account.base_balance
        if bal is None:
            return
        held = (bal.free or 0.0) + (bal.reserved or 0.0)
        tracked = pos.qty
        if tracked - held > self.exch_min:
            pos.reduce(tracked - held)

    # ------------------------------------------------------------------ respond
    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        validator = state.dendrite.hotkey
        cfg = self.simulation_config
        self._sync_order_size(cfg.volumeDecimals)

        price_dp = cfg.priceDecimals
        vol_dp = cfg.volumeDecimals
        cap = CAPITAL_TURNOVER_CAP * cfg.miner_wealth * VOLUME_SAFETY

        for book_id in sorted(self.accounts.keys()):
            book = state.books.get(book_id)
            account = self.accounts.get(book_id) if book is not None else None
            if book is None or account is None:
                continue
            try:
                self._handle_book(response, validator, book_id, book, account,
                                  price_dp, vol_dp, cap, int(state.timestamp))
            except Exception as ex:
                bt.logging.warning(
                    f"[DualEdge uid={self.uid}] book {book_id} error: {ex}\n{traceback.format_exc()}"
                )
        return response

    def _sync_order_size(self, volume_decimals: int) -> None:
        if volume_decimals == self._volume_decimals:
            return
        self._volume_decimals = volume_decimals
        lot = round(max(LOT, 10 ** (-volume_decimals)), volume_decimals)
        self.min_order_size = lot
        self.exch_min = max(EXCHANGE_MIN_ORDER_SIZE, 10 ** (-volume_decimals))
        self._flat_eps = lot / 2
        self.max_inventory = MAX_INVENTORY_LOTS * lot
        bt.logging.info(f"[DualEdge uid={self.uid}] volumeDecimals={volume_decimals} lot={lot}")

    # --------------------------------------------------------------- per book
    def _handle_book(self, response, validator, book_id, book, account,
                     price_dp, vol_dp, cap, now) -> None:
        mid = self._mid(book)
        if mid is None or mid <= 0 or not book.bids or not book.asks:
            return
        pos = self._book_positions(validator).setdefault(book_id, _Position())
        st = self._bstate(validator, book_id)
        if st.seen_ns == 0:
            st.seen_ns = now
        st.mids.append((now, mid))
        self._reconcile(pos, account)

        inv = pos.qty
        best_bid = book.bids[0].price
        best_ask = book.asks[0].price
        half_spread_bps = (best_ask - best_bid) / 2.0 / mid * 1e4
        maker_fee_bps = self._maker_fee_bps(account)
        taker_fee_bps = self._taker_fee_bps(account)

        pending = bool(st.pending_ns) and (now - st.pending_ns) < self.pending_timeout_ns
        no_position = inv < self._flat_eps and not pending

        # Router: commit a mode switch only when flat.
        self._heartbeat(st, book_id, inv, mid, maker_fee_bps, taker_fee_bps,
                        half_spread_bps, len(account.orders), now)

        desired = self._desired_mode(half_spread_bps, taker_fee_bps)
        if desired != st.mode:
            if no_position and not account.orders:
                bt.logging.info(
                    f"[DualEdge uid={self.uid}] book={book_id} mode {st.mode}->{desired} "
                    f"(maker_fee={maker_fee_bps:.1f}bps taker_fee={taker_fee_bps:.1f}bps "
                    f"half_spread={half_spread_bps:.1f}bps)"
                )
                st.mode = desired
            elif no_position and account.orders:
                # Flat but quotes still resting: pull them, switch next tick.
                response.cancel_orders(book_id, [o.id for o in account.orders])
                return

        # Activity hard deadline: guarantee one round-trip per window.
        if self._activity_due(st, now):
            if pending:
                return
            # Free reserve locked in resting quotes so the market leg can fill.
            if account.orders:
                response.cancel_orders(book_id, [o.id for o in account.orders])
            if inv < self.min_order_size:
                self._ping_open(response, st, account, book_id, best_ask, now)
            else:
                best_bid_qty = book.bids[0].quantity if book.bids else 0.0
                self._ping_close(response, st, account, pos, book_id, best_bid_qty, vol_dp, now)
            return

        # Activity soft window: bank a no-loss round-trip early via IOC at break-even
        # when the bid covers it, so the deadline (loss-bearing) close rarely fires.
        if (not pending and inv >= self.min_order_size
                and self._activity_soft_due(st, now)):
            b_qty, worst_unit = pos.harvest_bundle(self.min_order_size)
            if b_qty >= self.min_order_size and worst_unit > 0:
                limit_px = self._ceil_tick(self._noloss_floor(worst_unit, taker_fee_bps), price_dp)
                if best_bid >= limit_px:
                    if account.orders:   # free reserved base so the IOC can fill
                        response.cancel_orders(book_id, [o.id for o in account.orders])
                    qty = round(min(b_qty, self._avail(account.base_balance)), vol_dp)
                    if qty >= self.min_order_size:
                        st.sell_tag = "soft"
                        self._ioc_sell(response, st, book_id, qty, limit_px)
                    return

        # Trade in the committed mode.
        if st.mode == MODE_TAKER:
            self._taker_step(response, st, pos, account, book_id,
                             best_bid, best_ask, taker_fee_bps, half_spread_bps,
                             price_dp, vol_dp, cap, now, pending)
        else:
            allow_accumulate = (desired == MODE_MAKER) and not self._falling(st, mid, now)
            self._maker_step(response, st, pos, account, book_id,
                             mid, best_bid, best_ask, maker_fee_bps, allow_accumulate,
                             price_dp, vol_dp, cap, now)

    def _desired_mode(self, half_spread_bps, taker_fee_bps) -> str:
        # Taker only when the rebate more than pays for crossing the spread.
        if -taker_fee_bps >= half_spread_bps + ROUTER_TAKER_MARGIN_BPS \
                and -taker_fee_bps >= TAKER_REBATE_GATE_BPS:
            return MODE_TAKER
        return MODE_MAKER

    # ===================== MAKER engine (post-only, long-only) =====================
    def _maker_step(self, response, st, pos, account, book_id,
                    mid, best_bid, best_ask, maker_fee_bps, allow_accumulate,
                    price_dp, vol_dp, cap, now) -> None:
        inv = pos.qty
        tick = 10 ** (-price_dp)
        width_bps = max(MAKER_MIN_HALF_SPREAD_BPS, maker_fee_bps + MAKER_EDGE_MARGIN_BPS)

        # Accumulate only when allowed (stable, not dumping), the maker fee is not
        # punitive, inventory has room, and under the volume cap. Else harvest only.
        accumulate = (
            allow_accumulate
            and maker_fee_bps <= MAKER_FEE_DEFENSIVE_BPS
            and inv < self.max_inventory - self._flat_eps
            and self._rolled_quote_volume(st, now) < cap
        )

        bid_orders = [o for o in account.orders if o.side == OrderDirection.BUY]
        ask_orders = [o for o in account.orders if o.side == OrderDirection.SELL]

        # BID: accumulate a long clip below mid; bid deeper (cheaper) as inventory
        # fills, so later lots lower the average cost and accumulation self-throttles.
        if accumulate:
            bid_qty = round(min(LOT, self.max_inventory - inv), vol_dp)
            if bid_qty >= self.min_order_size:
                inv_frac = inv / self.max_inventory if self.max_inventory > 0 else 0.0
                bid_width_bps = width_bps + MAKER_INVENTORY_SKEW_BPS * inv_frac
                target_bid = round(min(mid * (1 - bid_width_bps / 1e4), best_ask - tick), price_dp)
                self._maintain_quote(response, book_id, OrderDirection.BUY, bid_orders,
                                     target_bid, bid_qty, mid,
                                     affordable=account.quote_balance.free >= bid_qty * target_bid)
        elif bid_orders:
            response.cancel_orders(book_id, [o.id for o in bid_orders])

        # ASK: harvest the oldest FIFO lots (bundled past dust) at the worst per-base
        # break-even, so every FIFO prefix closes >= 0.
        b_qty, worst_unit = pos.harvest_bundle(self.min_order_size)
        if b_qty >= self.min_order_size:
            ask_qty = round(b_qty, vol_dp)
            floor = self._noloss_floor(worst_unit, maker_fee_bps)
            target_ask = round(max(mid * (1 + width_bps / 1e4), floor, best_bid + tick), price_dp)
            self._maintain_quote(response, book_id, OrderDirection.SELL, ask_orders,
                                 target_ask, ask_qty, mid,
                                 affordable=account.base_balance.free >= ask_qty - self._flat_eps,
                                 floor_px=round(floor, price_dp))
        elif ask_orders:
            response.cancel_orders(book_id, [o.id for o in ask_orders])

    def _maintain_quote(self, response, book_id, direction, existing,
                        target_px, qty, mid, affordable, floor_px=0.0) -> None:
        """Keep exactly one resting post-only quote per side. Place when missing;
        reprice (cancel now, repost next tick) only when needed, to avoid a same-tick
        balance race and preserve queue priority. A partial fill keeps resting; a
        stray duplicate self-heals via the len>1 cancel.

        Reprice is direction-aware:
          * BID: symmetric drift — keep it near target.
          * ASK: do NOT reprice up on a rally (it should get lifted, not chase peaks
            on a mean-reverting book). Reprice only when forced — below the live
            break-even (`floor_px`, after a fee rise), or stale-high after a mid drop
            (move it back down so the next rebound fills it)."""
        if not existing:
            if affordable and target_px > 0:
                response.limit_order(book_id=book_id, direction=direction, quantity=qty,
                                     price=target_px, postOnly=True, timeInForce=TimeInForce.GTT,
                                     expiryPeriod=self.quote_expiry_ns, stp=STP.CANCEL_OLDEST)
            return
        if len(existing) > 1:
            response.cancel_orders(book_id, [o.id for o in existing])
            return
        resting_px = existing[0].price or target_px
        if direction == OrderDirection.SELL:
            under_floor = floor_px > 0 and resting_px < floor_px
            stale_high = (resting_px - target_px) / mid * 1e4 > MAKER_REPRICE_BPS
            if under_floor or stale_high:
                response.cancel_orders(book_id, [o.id for o in existing])
        else:
            drift_bps = abs(resting_px - target_px) / mid * 1e4
            if drift_bps > MAKER_REPRICE_BPS:
                response.cancel_orders(book_id, [o.id for o in existing])

    # ================ TAKER engine (rebate-cushioned long scalp) ================
    def _taker_step(self, response, st, pos, account, book_id,
                    best_bid, best_ask, taker_fee_bps, half_spread_bps,
                    price_dp, vol_dp, cap, now, pending) -> None:
        if pending:
            return
        st.pending_ns = 0
        inv = pos.qty

        if inv >= self._flat_eps:
            # HELD: no-loss exit. IOC sell at the bundle's worst break-even, sent only
            # when the bid already covers it, so every fill (and FIFO prefix on a
            # partial) nets >= 0. Else hold (unrealized PnL is invisible to Kappa) and
            # wait for recovery or the activity floor.
            held_ns = (now - st.taker_open_ns) if st.taker_open_ns else self.taker_min_hold_ns
            if held_ns < self.taker_min_hold_ns:
                return
            b_qty, worst_unit = pos.harvest_bundle(self.min_order_size)
            if b_qty < self.min_order_size or worst_unit <= 0:
                return
            limit_px = self._ceil_tick(self._noloss_floor(worst_unit, taker_fee_bps), price_dp)
            if best_bid >= limit_px:
                qty = round(min(b_qty, account.base_balance.free), vol_dp)
                if qty >= self.min_order_size:
                    self._ioc_sell(response, st, book_id, qty, limit_px)
                    st.last_close_ns = now
            return

        # FLAT: open a long clip only when the rebate makes the round-trip +EV.
        if st.last_close_ns and (now - st.last_close_ns) < self.taker_reopen_gap_ns:
            return
        if -taker_fee_bps < TAKER_REBATE_GATE_BPS:
            return
        est_bps = (-taker_fee_bps) * 2.0 - half_spread_bps * 2.0   # rebate both legs - spread
        if est_bps < TAKER_EDGE_MARGIN_BPS:
            return
        if self._rolled_quote_volume(st, now) >= cap:
            return
        if inv >= self.max_inventory - self._flat_eps:
            return
        qty = round(LOT, vol_dp)
        if best_ask <= 0 or account.quote_balance.free < qty * best_ask:
            return
        self._market(response, st, book_id, OrderDirection.BUY, qty)
        st.taker_open_ns = now

    # ================ ACTIVITY FLOOR (min-lot round-trip per window) ================
    def _activity_due(self, st: _BookState, now: int) -> bool:
        ref = st.last_rt_ns if st.last_rt_ns > 0 else st.seen_ns
        if ref == 0:
            return False
        if (now - ref) < self.ping_interval_ns:
            return False
        return (st.last_ping_submit_ns == 0) or ((now - st.last_ping_submit_ns) >= self.ping_cooldown_ns)

    def _activity_soft_due(self, st: _BookState, now: int) -> bool:
        """Past the soft fraction of the window with no round-trip yet — accept a
        break-even close to pre-empt the deadline dump. Strictly earlier than
        _activity_due."""
        ref = st.last_rt_ns if st.last_rt_ns > 0 else st.seen_ns
        if ref == 0:
            return False
        return (now - ref) >= self.activity_soft_ns

    def _ping_open(self, response, st, account, book_id, best_ask, now) -> None:
        """Open leg: market-buy one min lot (buy-first => a real in-sim long).
        Sized off free+reserved quote (resting bids were cancelled this response)."""
        if st.ping_awaiting_open and (now - st.last_ping_submit_ns) <= int(60 * _NS):
            return
        qty = self.min_order_size
        if best_ask <= 0 or self._avail(account.quote_balance) < qty * best_ask:
            return
        st.ping_awaiting_open = True
        st.last_ping_submit_ns = self._step_ts_ns
        bt.logging.info(f"[DualEdge uid={self.uid} PING] book={book_id} open qty={qty:.4f} ({st.mode})")
        self._market(response, st, book_id, OrderDirection.BUY, qty)

    def _ping_close(self, response, st, account, pos, book_id, best_bid_qty, vol_dp, now) -> None:
        """Last-resort activity close: the window stayed below break-even, so bank one
        round-trip or lose activity_factor (worse than a small loss). Market order for
        a guaranteed fill — the loss here is the price gap, not depth slippage. The
        clip is capped to the top-of-book size when that still forms a valid order so a
        thin book is not swept deep; the whole position is sold when a min slice would
        leave sub-min dust. Sized off free+reserved base (asks cancelled this response)."""
        avail = self._avail(account.base_balance)
        target = self.min_order_size
        if pos.qty - target < self.min_order_size:
            target = pos.qty   # avoid leaving un-sellable dust
        qty = round(min(target, avail, pos.qty), vol_dp)
        if qty < self.min_order_size:
            return
        # Cap to top-of-book so we do not sweep deep levels.
        if best_bid_qty >= self.min_order_size:
            qty = round(min(qty, best_bid_qty), vol_dp)
        st.sell_tag = "ping"
        st.last_ping_submit_ns = self._step_ts_ns
        bt.logging.info(f"[DualEdge uid={self.uid} PING] book={book_id} close qty={qty:.4f} ({st.mode})")
        self._market(response, st, book_id, OrderDirection.SELL, qty)

    # ------------------------------------------------------------------ orders
    def _market(self, response, st, book_id, direction, qty) -> None:
        if qty < self.exch_min:
            return
        response.market_order(book_id=book_id, direction=direction, quantity=qty,
                              currency=OrderCurrency.BASE, stp=STP.CANCEL_OLDEST)
        st.pending_ns = self._step_ts_ns

    def _ioc_sell(self, response, st, book_id, qty, price) -> None:
        """No-loss exit: an IOC limit sell crosses only bids >= `price` and cancels
        any remainder, so it never fills below break-even. Marks pending so the
        sequential taker cycle waits for the fill (or timeout), mirroring _market."""
        if qty < self.exch_min or price <= 0:
            return
        response.limit_order(book_id=book_id, direction=OrderDirection.SELL, quantity=qty,
                             price=price, postOnly=False, timeInForce=TimeInForce.IOC,
                             stp=STP.CANCEL_OLDEST)
        st.pending_ns = self._step_ts_ns


if __name__ == "__main__":
    launch(DualEdgeAgent)
