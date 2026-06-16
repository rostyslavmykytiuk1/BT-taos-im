"""
DualEdgeAgent — fee-adaptive maker/taker agent for Subnet 79 (taos)
===================================================================

Designed from the validator source (taos/im/validator/reward.py and
taos/im/neurons/validator.py ``_match_trade_fifo``), the active DynamicFeePolicy
(simulation_0.xml), and the live behaviour of the top agents (114 / 165).

Verified facts that drive every decision
----------------------------------------
  * Score = activity-weighted, MAD-normalized Kappa-3 per book, medianed across
    books with an outlier penalty. ``kappa.pnl.impact = 0`` and
    ``activity.impact = 0`` today, so:
      - Only the SHAPE of realized round-trip PnL matters (many small same-signed
        wins beat a few big ones; LPM3 cubes downside, so one realized loss is
        very expensive). Absolute PnL size and deployed capital do NOT score.
      - One COMPLETED round-trip per book per ~600s sampling bucket pins
        activity_factor to 1.0; trade VOLUME is irrelevant. Miss the bucket and
        the book decays toward 0 — far worse than a tiny forced loss.
  * Realized PnL is FIFO and NET OF FEES, using the fee STORED AT FILL TIME
    (fees are dynamic, so the open and close legs can carry different rates).
  * The validator FIFO only knows about in-sim trades: a SELL with no prior
    in-sim BUY opens a SHORT (0 PnL, 0 round-trip). Therefore every book must
    BUY before it SELLs. We hold a long-only FIFO ladder of our in-sim buys and
    never sell more than that ladder, so every sell closes a real long and the
    ladder always mirrors the validator's positions exactly. The Pareto base
    endowment the sim hands us is left as reserve (selling it would open shorts,
    and using it would not raise Kappa since capital is not rewarded).
  * max_instructions_per_book = 5: excess per-book instructions are dropped.
    We emit <= 2 per book per tick (a cancel-list counts as one instruction).

Strategy (per book, every tick) — the approved decision tree
------------------------------------------------------------
  1. Reconcile the ladder DOWN to on-chain base (never up): never sell base we
     do not hold; ignore the base endowment.
  2. ROUTER (a switch is COMMITTED only when the book has no open position):
         taker rebate covers the spread -> TAKER ; else -> MAKER.
  3. ACTIVITY FLOOR (overrides trading): if no round-trip closed within the
     window, cancel resting quotes (freeing the reserve) and force a min-lot
     taker round-trip. Buy first when flat, sell to close otherwise.
  4. MAKER: post-only, long-only.
         bid (accumulate) rests below mid by a width that beats the maker fee,
             only while inventory < cap, under the volume cap, not into a dump,
             and not winding down for a mode switch;
         ask (harvest) bundles the OLDEST FIFO lots up to >= one clip and sells
             at the WORST per-lot fee-clearing break-even in the bundle (using
             each lot's stored fee), so EVERY FIFO prefix — including a partial
             fill that consumes only the priciest oldest lots — closes >= 0, and
             partial-fill dust never strands. Quotes rest (GTT), reprice on a move.
  5. TAKER: one sequential clip cycle. Open a long (market buy) only when the
         rebate makes the round-trip +EV; exit with an IOC limit sell at the lots'
         worst break-even, sent ONLY when the book bid already covers it — so the
         exit fills at >= break-even or not at all (no market-dump slippage can
         realize a loss). Otherwise hold; the activity floor is the last resort.

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

# --- size (small per-book tail risk; volume is not rewarded) ---
EXCHANGE_MIN_ORDER_SIZE = 0.25     # sim minOrderSize floor
LOT = 0.30                         # clip per quote/scalp (matches 114/165 ~0.3)
MAX_INVENTORY_LOTS = 10            # max long inventory (114 rode ~3 BASE in a dip)

# --- dynamic-fee router (all thresholds in bps; rebate = negative fee) ---
ROUTER_TAKER_MARGIN_BPS = 1.0      # need (-taker_fee) >= half_spread + this for TAKER
MAKER_FEE_DEFENSIVE_BPS = 12.0     # maker fee above this -> stop accumulating
ASSUMED_FEE_BPS = 2.3              # fallback per-side fee if account.fees missing

# --- maker engine ---
MAKER_MIN_HALF_SPREAD_BPS = 8.0    # min distance of a resting quote from mid
MAKER_EDGE_MARGIN_BPS = 4.0        # quote width on top of the maker fee
MAKER_HARVEST_BUFFER_BPS = 3.0     # profit above the round-trip fee on the ask; also
                                   # cushions a same-tick maker-fee rise before a fill
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

# --- activity floor (UNCONDITIONAL: miss the ~600s bucket and the book decays
#     toward 0). 480s leaves margin against bucket alignment + UID jitter. ---
PING_INTERVAL_S = 480.0
PING_SUBMIT_COOLDOWN_S = 5.0       # anti-burst gap between ping orders on a book

# --- logging ---
LOG_HEARTBEAT_S = 120.0           # per-book state heartbeat cadence

# --- volume cap (self-imposed safety; volume is neither rewarded nor, here,
#     penalized — kept conservative to avoid the exchange turnover limit) ---
CAPITAL_TURNOVER_CAP = 10.0
VOLUME_SAFETY = 0.5
VOLUME_ASSESSMENT_NS = 86_400_000_000_000   # rolling 24h

MODE_MAKER = "maker"
MODE_TAKER = "taker"


@dataclass
class _Position:
    """Long-only FIFO ladder of our in-sim buys (mirrors the validator's FIFO
    longs deque). Each lot is [price, qty, fee] where fee is the QUOTE fee paid
    on that buy (stored so the no-loss harvest is exact under dynamic fees)."""
    lots: deque = field(default_factory=deque)

    @property
    def qty(self) -> float:
        return sum(lot[1] for lot in self.lots)

    @property
    def oldest_price(self) -> float:
        return self.lots[0][0] if self.lots else 0.0

    def add(self, price: float, qty: float, fee: float) -> None:
        if qty > 0:
            self.lots.append([float(price), float(qty), float(fee)])

    def reduce(self, qty: float, price: float = 0.0) -> tuple[float, float, float]:
        """Consume `qty` from the front (FIFO), shrinking each lot's fee pro rata
        to mirror the validator. Returns (closed_qty, price_pnl, open_fee) where
        price_pnl = sum((price - lot_price) * take) and open_fee is the stored buy
        fee consumed — so the net realized PnL is price_pnl - open_fee - close_fee."""
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
        """Oldest lots bundled (FIFO) until cumulative qty reaches `min_qty`.
        Returns (qty, worst_unit_cost) where worst_unit_cost = max over the bundled
        lots of (price + fee/qty), i.e. the HIGHEST per-base break-even in the
        bundle. Pricing the ask at worst_unit_cost / (1 - close_rate) (plus a
        buffer) makes EVERY FIFO prefix of the bundle close at >= its own
        break-even — so even a partial fill that consumes only the most expensive
        (oldest) lots cannot realize a loss (critical in a falling market, where
        the oldest lot is the priciest). Bundling also absorbs sub-min partial
        dust. Returns (0, 0) when total inventory cannot form a min clip."""
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

    def clear(self) -> None:
        self.lots.clear()


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
        # Dispatched inside super().update(), so _active_validator is the validator
        # whose simulation just (re)started. Reset ONLY that validator's state so a
        # restart on one validator never wipes another validator's live books. Do
        # not touch _sim_id here — _ensure_simulation owns it (and already ran this
        # tick); resetting positions/books_state also covers a reused simulation_id.
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
        """Log a per-validator summary at simulation end (investigation aid) and
        drop that validator's state so the next simulation starts clean."""
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
        """Drop all per-book ladders and state for one validator (used on sim
        change / start / end). _BookState and _Position are recreated lazily with
        clean defaults on next access, so nothing carries across simulations."""
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
        """Track each of our fills (maker or taker leg, incl. partial fills) into
        the FIFO ladder. Each TradeEvent is one fill slice, so partials accumulate
        naturally. The per-fill fee is stored on buys for the exact no-loss floor."""
        if event.bookId is None:
            return
        if self.uid == event.takerAgentId:
            direction = OrderDirection.BUY if event.side == OrderDirection.BUY else OrderDirection.SELL
            fee = float(getattr(event, "takerFee", 0.0) or 0.0)
        elif self.uid == event.makerAgentId:
            # As the resting maker we are the opposite side of the aggressor.
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
                # Any reducing fill (incl. a partial) completes a round-trip.
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
        """Surface rejections (post-only cross, insufficient balance, max open
        orders, instruction-budget drops) so we can diagnose them from the log,
        and clear any pending/ping flag the rejected order was holding."""
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
        """Spendable this step once our resting orders are cancelled: free +
        reserved (reserved is locked only in our own orders, which we cancel
        earlier in the same response before any market exit)."""
        if balance is None:
            return 0.0
        return (balance.free or 0.0) + (balance.reserved or 0.0)

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
        """True if the book dropped >= crash_drop_bps over the crash window —
        pause accumulation so we do not bid into a knife."""
        cutoff = now - self.crash_window_ns
        hi = mid
        for t, m in st.mids:
            if t >= cutoff and m > hi:
                hi = m
        return hi > 0 and (hi - mid) / hi * 1e4 >= self.crash_drop_bps

    def _heartbeat(self, st: _BookState, book_id, inv, mid, maker_fee_bps,
                   taker_fee_bps, half_spread_bps, n_orders, now) -> None:
        """Throttled per-book state line so a quiet book is still observable."""
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
        """Clamp the ladder DOWN to the real on-chain base (free + reserved) so we
        never try to sell base we do not hold. We never seed UP: the Pareto base
        endowment is not an in-sim long, so selling it would open a validator
        short — the ladder must contain only our in-sim buys."""
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

        # --- ROUTER: choose the mode, but COMMIT a switch only with no position. ---
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
                # No position but quotes still resting: pull them, switch next tick.
                response.cancel_orders(book_id, [o.id for o in account.orders])
                return

        # --- ACTIVITY FLOOR (overrides trading): guarantee one RT per window. ---
        if self._activity_due(st, now):
            if pending:
                return
            # Free any reserve locked in resting quotes so the market leg can fill.
            if account.orders:
                response.cancel_orders(book_id, [o.id for o in account.orders])
            if inv < self.min_order_size:
                self._ping_open(response, st, account, book_id, best_ask, now)
            else:
                self._ping_close(response, st, account, pos, book_id, vol_dp, now)
            return

        # --- trade in the committed mode ---
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
        # Taker is best only when the rebate more than pays for crossing the spread.
        if -taker_fee_bps >= half_spread_bps + ROUTER_TAKER_MARGIN_BPS \
                and -taker_fee_bps >= TAKER_REBATE_GATE_BPS:
            return MODE_TAKER
        return MODE_MAKER

    # =====================================================================
    # MAKER engine — post-only, long-only, FIFO no-loss harvest
    # =====================================================================
    def _maker_step(self, response, st, pos, account, book_id,
                    mid, best_bid, best_ask, maker_fee_bps, allow_accumulate,
                    price_dp, vol_dp, cap, now) -> None:
        inv = pos.qty
        tick = 10 ** (-price_dp)
        width_bps = max(MAKER_MIN_HALF_SPREAD_BPS, maker_fee_bps + MAKER_EDGE_MARGIN_BPS)

        # Accumulate only when allowed (mode stable, not dumping), the maker fee is
        # not punitive, inventory has room, and we are under the volume cap.
        # Otherwise "defensive": harvest what we hold, never add.
        accumulate = (
            allow_accumulate
            and maker_fee_bps <= MAKER_FEE_DEFENSIVE_BPS
            and inv < self.max_inventory - self._flat_eps
            and self._rolled_quote_volume(st, now) < cap
        )

        bid_orders = [o for o in account.orders if o.side == OrderDirection.BUY]
        ask_orders = [o for o in account.orders if o.side == OrderDirection.SELL]

        # ---- BID: accumulate a long clip below mid (post-only) ----
        if accumulate:
            bid_qty = round(min(LOT, self.max_inventory - inv), vol_dp)
            if bid_qty >= self.min_order_size:
                target_bid = round(min(mid * (1 - width_bps / 1e4), best_ask - tick), price_dp)
                self._maintain_quote(response, book_id, OrderDirection.BUY, bid_orders,
                                     target_bid, bid_qty, mid,
                                     affordable=account.quote_balance.free >= bid_qty * target_bid)
        elif bid_orders:
            response.cancel_orders(book_id, [o.id for o in bid_orders])

        # ---- ASK: harvest the OLDEST FIFO lots (bundled past dust) at their exact
        #      fee-clearing break-even (FIFO match guaranteed >= 0) ----
        b_qty, worst_unit = pos.harvest_bundle(self.min_order_size)
        if b_qty >= self.min_order_size:
            ask_qty = round(b_qty, vol_dp)
            maker_rate = maker_fee_bps / 1e4
            buffer = 1.0 + MAKER_HARVEST_BUFFER_BPS / 1e4
            # Cover the WORST per-base break-even in the bundle, grossed up for the
            # close fee: ask*(1-rate) >= worst_unit*buffer => every FIFO prefix >=0.
            floor = worst_unit * buffer / max(0.5, 1.0 - maker_rate)
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
        reprice (cancel now, repost next tick) only on a material drift, which
        avoids a same-tick balance race and preserves queue priority on stable
        quotes. A partially-filled order keeps resting (matched on price, not qty)
        so the remainder fills. A stray duplicate self-heals via the len>1 cancel.

        `floor_px` (sell asks only): the CURRENT no-loss break-even. Maker fees are
        dynamic, so a fee rise while the ask rests can push the live break-even above
        a quote priced at the old, lower fee — letting it fill below cost. If the
        resting ask is now under the floor we force a reprice regardless of drift, so
        a resting ask can never sit below the current break-even."""
        if not existing:
            if affordable and target_px > 0:
                response.limit_order(book_id=book_id, direction=direction, quantity=qty,
                                     price=target_px, postOnly=True, timeInForce=TimeInForce.GTT,
                                     expiryPeriod=self.quote_expiry_ns, stp=STP.CANCEL_OLDEST)
            return
        resting_px = existing[0].price or target_px
        under_floor = floor_px > 0 and resting_px < floor_px
        drift_bps = abs(resting_px - target_px) / mid * 1e4
        if under_floor or drift_bps > MAKER_REPRICE_BPS or len(existing) > 1:
            response.cancel_orders(book_id, [o.id for o in existing])

    # =====================================================================
    # TAKER engine — rebate-cushioned long scalp, one sequential cycle
    # =====================================================================
    def _taker_step(self, response, st, pos, account, book_id,
                    best_bid, best_ask, taker_fee_bps, half_spread_bps,
                    price_dp, vol_dp, cap, now, pending) -> None:
        if pending:
            return
        st.pending_ns = 0
        inv = pos.qty

        if inv >= self._flat_eps:
            # HELD: strictly no-loss exit. Post an IOC SELL limit at the bundle's
            # WORST per-lot break-even, but ONLY when the book bid already covers it.
            # The IOC matches resting bids from best down to the limit, so every fill
            # (and every FIFO prefix on a partial) clears break-even — no slippage can
            # realize a loss, unlike a market dump. When the bid is below break-even we
            # simply HOLD (unrealized PnL is invisible to Kappa) and wait for price to
            # recover or, as a last resort, the activity floor banks the round-trip.
            held_ns = (now - st.taker_open_ns) if st.taker_open_ns else self.taker_min_hold_ns
            if held_ns < self.taker_min_hold_ns:
                return
            b_qty, worst_unit = pos.harvest_bundle(self.min_order_size)
            if b_qty < self.min_order_size or worst_unit <= 0:
                return
            # Close fee here is a taker rebate (fee<0), so requiring bid >= worst_unit
            # is conservative: any fill at >= worst_unit nets strictly positive. Round
            # the limit UP to the price grid so the resting price never dips below
            # break-even; since best_bid is itself a grid price >= worst_unit, the
            # ceiled limit stays <= best_bid and therefore crosses.
            tick = 10 ** (-price_dp)
            limit_px = round(-(-worst_unit // tick) * tick, price_dp)
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
        est_bps = (-taker_fee_bps) * 2.0 - half_spread_bps * 2.0   # rebate both legs - full spread
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

    # =====================================================================
    # ACTIVITY FLOOR — unconditional min-lot taker round-trip per window
    # =====================================================================
    def _activity_due(self, st: _BookState, now: int) -> bool:
        ref = st.last_rt_ns if st.last_rt_ns > 0 else st.seen_ns
        if ref == 0:
            return False
        if (now - ref) < self.ping_interval_ns:
            return False
        return (st.last_ping_submit_ns == 0) or ((now - st.last_ping_submit_ns) >= self.ping_cooldown_ns)

    def _ping_open(self, response, st, account, book_id, best_ask, now) -> None:
        """Open leg: market-buy one min lot (buy-first => a real in-sim long).
        Sized off free+reserved quote because resting bids were cancelled in this
        same response."""
        if st.ping_awaiting_open and (now - st.last_ping_submit_ns) <= int(60 * _NS):
            return
        qty = self.min_order_size
        if best_ask <= 0 or self._avail(account.quote_balance) < qty * best_ask:
            return
        st.ping_awaiting_open = True
        bt.logging.info(f"[DualEdge uid={self.uid} PING] book={book_id} open qty={qty:.4f} ({st.mode})")
        self._market(response, st, book_id, OrderDirection.BUY, qty)

    def _ping_close(self, response, st, account, pos, book_id, vol_dp, now) -> None:
        """Close leg: market-sell a lot-aligned min slice = a completed round-trip.
        FIFO closes the oldest in-sim long; if underwater this banks a tiny loss,
        the accepted price of keeping activity_factor = 1.0. Sized off free+reserved
        base (asks were cancelled in this response); sells the whole position when a
        min slice would leave sub-min dust."""
        avail = self._avail(account.base_balance)
        target = self.min_order_size
        if pos.qty - target < self.min_order_size:
            target = pos.qty   # avoid leaving un-sellable dust
        qty = round(min(target, avail, pos.qty), vol_dp)
        if qty < self.min_order_size:
            return
        st.sell_tag = "ping"
        bt.logging.info(f"[DualEdge uid={self.uid} PING] book={book_id} close qty={qty:.4f} ({st.mode})")
        self._market(response, st, book_id, OrderDirection.SELL, qty)

    # ------------------------------------------------------------------ orders
    def _market(self, response, st, book_id, direction, qty) -> None:
        if qty < self.exch_min:
            return
        response.market_order(book_id=book_id, direction=direction, quantity=qty,
                              currency=OrderCurrency.BASE, stp=STP.CANCEL_OLDEST)
        st.pending_ns = self._step_ts_ns
        st.last_ping_submit_ns = self._step_ts_ns

    def _ioc_sell(self, response, st, book_id, qty, price) -> None:
        """Aggressive no-loss exit: an IOC limit sell crosses only resting bids at
        or above `price`, filling at their (>=price) levels and cancelling any
        remainder — so it can never rest or fill below break-even. Marks pending so
        the sequential taker cycle waits for the fill (or the timeout) before acting
        again, mirroring _market."""
        if qty < self.exch_min or price <= 0:
            return
        response.limit_order(book_id=book_id, direction=OrderDirection.SELL, quantity=qty,
                             price=price, postOnly=False, timeInForce=TimeInForce.IOC,
                             stp=STP.CANCEL_OLDEST)
        st.pending_ns = self._step_ts_ns


if __name__ == "__main__":
    launch(DualEdgeAgent)
