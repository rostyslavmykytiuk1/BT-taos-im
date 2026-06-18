# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
ApexTakerAgent — regime-aware positive-skew TAKER for Subnet 79 (taos). Built to beat UID 136.

WHAT 136 IS (dashboard_data/136_trades_top_taker_down_trends.csv + SUBNET79_AGENT_REPORT.md):
  100% taker, ALL 128 books, PAYS ~1.7bps fee on ~93% of fills (only ~7% rebate), peak abs inventory
  ~0.73 lot (near-flat), same-side run = 1 (FLIPS every trade), hold ~30-100s, ~10s gaps. He has
  ~ZERO directional skill (50% sell-fraction even on down-books, ~11% per-RT price win-rate). He wins
  on a POSITIVE-SKEW profile — cut losers tiny, let the occasional move RUN — whose many near-zero
  fee-paying RTs build a LARGE per-book MAD. Kappa-3 cubes the downside, so a large MAD normalizes his
  losses small and the cubic penalty barely bites. He rents kappa via VOLUME + BREADTH + activity=1.0.

HOW WE BEAT HIM (lift the per-book kappa MEDIAN, the 0.79-weighted scored term):
  1. KEEP his structural edge: high VOLUME + 128-book BREADTH + activity=1.0 + NEAR-FLAT (run=1) so a
     large per-book MAD cushions the cube. We do NOT idle fee books in a fee-heavy regime (that starves
     MAD and LOSES to 136) — we churn the VOLATILE ones permissively, leaning the entry direction.
  2. ADD a directional LEAN 136 lacks: enter with the trend (EMA drift) when present, else with order
     flow (microprice). A weak-but-better-than-coin-flip lean raises mean_r (the kappa numerator).
  3. ADD the REBATE edge 136 ignores (mirror 126): where 2*rebate covers the spread, scalp for the
     rebate — a round-trip is +EV with zero price move. Pure profit + MAD where PnL is positive.
  4. POSITIVE-SKEW exit carries the EV (not a per-trade cost hurdle): tiny stop cuts the many small
     fee-paying losers, while winners RIDE to a drift reversal / trailing give-back / max hold.
  5. IDLE only dead-calm no-edge books (no volatility to harvest, no rebate) -> kappa=None -> DROPPED
     from the median for free (48-book budget). And never bleed a genuinely toxic (all-loss) book.
  6. RISK: small fixed clips, hard near-flat inventory cap, tiny stops, an absolute circuit-breaker
     with a wide guaranteed-fill cut -> NO single round-trip can be a cubic-killing loss.

PER BOOK, EACH STEP:
  update signals (mid EMA drift + per-step volatility EMA + microprice) -> risk-trim over-cap ->
  manage open position (abs stop / lane stop / reversal / trail / target / max-hold) ->
  (flat) try open (rebate-scalp OR volatility-gated leaned churn) -> activity backstop.
  FIFO mirrors the validator's _match_trade_fifo exactly (oldest-lot matching) so kappa/PnL are faithful.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol import FinanceAgentResponse, MarketSimulationStateUpdate
from taos.im.protocol.events import TradeEvent
from taos.im.protocol.models import (
    LoanSettlementOption,
    OrderCurrency,
    OrderDirection,
    STP,
    TimeInForce,
)

_NS = 1_000_000_000

# ===================================================================== sizing
EXCHANGE_MIN_ORDER_SIZE = 0.25
CLIP = 0.30                        # ~matches 136's 0.30 clip (more volume => bigger MAD cushion); still well
                                   # above the 0.25 min so a fee-shaved BASE buy (the exchange takes the fee
                                   # out of base) still leaves >= 0.25 held => always closeable, bounded loss.

# ===================================================================== inventory bounds
MAX_INVENTORY_LOTS = 1.5           # hard per-book cap (near-flat, tighter than 136's ~1.6 p90). We open
MAX_INVENTORY_EQUITY_FRAC = 0.08   # only while flat (run=1, no pyramiding) so inventory is ~1 clip; the
RISK_TRIM_SLIPPAGE_BPS = 6.0       # cap is a safety net for multi-fill / carryover edge cases.

# ===================================================================== REBATE-SCALP lane (mirror 126)
# Open where the rebate covers the crossing: 2*rebate - spread >= REBATE_ENTER_BPS. Then a round-trip
# is +EV with zero price move (rebate on both legs minus the two crossings of a zero-alpha RT).
REBATE_ENTER_BPS = 1.0             # required net rebate edge (2*rebate - spread) to open a scalp
RB_TP_BPS = 2.5                    # small gross target; the rebate is the real profit
RB_SL_BPS = 4.0                    # gross stop; net stays positive while 2*rebate >= RB_SL (re-checked live)
RB_MIN_HOLD_S = 1.0
RB_MAX_HOLD_S = 60.0               # if the rebate scalp doesn't hit TP fast, HOLD for the revert instead of
                                   # dumping at 5s — those 579 five-second dumps were our biggest drag vs 136
RB_REOPEN_GAP_S = 1.5              # fast recycle between scalps (rebate is +EV, churn it)

# ===================================================================== CHURN lane (136's volume, leaned)
# On non-rebate (fee) books we churn the VOLATILE ones — frequent, microprice/drift-LEANED entries with
# an asymmetric POSITIVE-SKEW exit. We do NOT demand the move clear the fee up front (that starves MAD,
# the rebate-maximalist's failure mode); the tiny-stop / ride-winner skew + the occasional big move pays
# for the many small fee-paying losers, exactly as 136 does — but our lean beats his coin-flip.
VOL_FLOOR_BPS = 1.0               # per-step mid-volatility EMA must exceed this to churn a book. 136 is on
                                  # ALL 128 books; keep a LOW floor so anything with motion churns and only
                                  # dead-calm books (no revert to harvest) fall through to the activity backstop.
CH_MAX_SPREAD_BPS = 15.0         # 136 trades WIDE-spread books (~16bps median, 8-19) — he does NOT avoid
                                 #   them. The wide spread is the entry COST he recovers by HOLDING for the
                                 #   mean-reversion (below), not by finding cheap crossings. Match his range.
CH_DRIFT_DIR_BPS = 2.0           # |EMA drift| above this => lean WITH the trend; else lean by microprice
CH_SL_BPS = 15.0                 # WIDE move-stop: hold THROUGH the spread+noise so the position can revert.
                                 #   Cutting at 5bps locked in losers that would have come back; this ~matches
                                 #   136's typical loss size and the catastrophic tail is bounded by ABS_STOP.
CH_TRAIL_BPS = 8.0               # once armed, exit if the move gives back this much from its peak (ride far)
CH_ARM_BPS = 10.0                # arm trailing only after a real reversion move develops — let winners run
CH_MIN_HOLD_S = 12.0             # do NOT exit in the dead <12s zone (~11% positive); give it time to revert.
                                 #   Our OWN 40-90s holds are already positive-mean; that is where to live.
CH_MAX_HOLD_S = 120.0           # ride the reversion up to 2min (136 holds 30-100s and lets the 90s+ tail run)
CH_REOPEN_GAP_S = 5.0           # re-enter ~5s after going flat => ~always in market like 136 (100% time-in)
PENDING_OPEN_TIMEOUT_S = 8.0    # after submitting an open, treat the book as in-flight until the fill is
                                #   seen (or this elapses). A market fill can lag past the reopen gap, and
                                #   re-opening in that window stacks a second clip -> the over-cap churn we saw.

# ===================================================================== shared exit / circuit breaker
ABS_STOP_BPS = 30.0              # ABSOLUTE per-position circuit breaker (adverse MID move): only the
                                 # CATASTROPHIC tail — wide enough to let 136-style reversions happen, but
                                 # bounds a single position from riding a sustained trend. Small clip caps $.
EXIT_SLIPPAGE_BPS = 4.0          # max concession on a normal forced IOC exit (bounds slippage)
EXIT_SLIPPAGE_ABS_BPS = 20.0     # WIDER concession on the absolute-stop cut so it clears in ONE step on
                                 # a gap (guaranteed same-step exit; a few extra bps beats staying exposed)

# ===================================================================== toxic-book backoff (skew-safe)
# A positive-skew taker is net-negative over many windows BY DESIGN (many tiny losers, few big winners),
# so net<0 must NOT be read as toxic. We only idle a book whose recent RTs are ALL losses (a genuinely
# broken book the lean never gets right), then re-test. Everything else is left to the per-trade bounds.
BACKOFF_WINDOW_S = 600.0
BACKOFF_MIN_RTS = 8              # need a real sample of consecutive losses before idling
BACKOFF_COOLDOWN_S = 180.0

# ===================================================================== signals
EMA_FAST_N = 12                  # fast/slow EMA of mid -> signed drift (trend)
EMA_SLOW_N = 48
VOL_EMA_N = 24                   # EMA of per-step |mid change| in bps -> volatility (is there a move to harvest)

# ===================================================================== activity / volume / kappa
ACTIVITY_DEADLINE_S = 480.0      # force >=1 closing RT per book per window so activity stays 1.0
RT_WINDOW_S = 570.0
RT_MAX = 40                      # max profit-seeking opens per book per window (breadth + MAD)
FORCE_TRIM_SLIPPAGE_BPS = 5.0
CAPITAL_TURNOVER_CAP = 10.0
VOLUME_SAFETY = 0.8
VOLUME_ASSESSMENT_NS = 86_400_000_000_000
KAPPA_TAU = 0.0
KAPPA_MIN_OBS = 3
KAPPA_MIN_LOOKBACK_S = 5400.0
KAPPA_RT_HISTORY_S = 10_800.0

MAIN_VALIDATOR = "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF"

LANE_REBATE = "rebate"
LANE_CHURN = "churn"


@dataclass
class _Inv:
    longs: deque = field(default_factory=deque)
    shorts: deque = field(default_factory=deque)


@dataclass
class _BookState:
    seen_ns: int = 0
    last_rt_ns: int = 0
    last_open_ns: int = 0             # last open submit (reopen-gap throttle)
    pending_open_ns: int = 0          # open submitted, fill not yet seen (in-flight guard vs double-open)
    lane: str = ""                    # lane that opened the current position (exit policy)
    entry_mid: float = 0.0            # mid at entry — moves are measured vs this (NOT mark-to-bid, which
                                      #   reads a fresh taker entry as -spread and false-triggers the stop)
    peak_move_bps: float = 0.0        # best favourable mid-move while holding (for churn trailing)
    # signals
    prev_mid: float = 0.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    vol_ema: float = 0.0              # EMA of per-step |mid change| in bps
    # kappa / activity / backoff
    rt_events: list[tuple[int, float]] = field(default_factory=list)
    kappa3: float | None = None
    vol_log: list[tuple[int, float]] = field(default_factory=list)
    backoff_until_ns: int = 0


class ApexTakerAgent(FinanceSimulationAgent):

    # --------------------------------------------------------------- setup
    def initialize(self) -> None:
        bt.logging.set_info()
        self.clip = CLIP
        self.exch_min = EXCHANGE_MIN_ORDER_SIZE
        self._flat_eps = 0.5 * 10 ** (-4)
        self._price_decimals: int | None = None
        self._volume_decimals: int | None = None
        self._tick = 0.01
        self.volume_assessment_ns = VOLUME_ASSESSMENT_NS

        jitter = ((self.uid * 2654435761) % 1000) / 1000.0
        self._alpha_fast = 2.0 / (EMA_FAST_N + 1)
        self._alpha_slow = 2.0 / (EMA_SLOW_N + 1)
        self._alpha_vol = 2.0 / (VOL_EMA_N + 1)
        self.rb_min_hold_ns = int(RB_MIN_HOLD_S * _NS)
        self.rb_max_hold_ns = int(RB_MAX_HOLD_S * (0.9 + 0.2 * jitter) * _NS)
        self.rb_reopen_gap_ns = int(RB_REOPEN_GAP_S * (0.9 + 0.2 * jitter) * _NS)
        self.ch_min_hold_ns = int(CH_MIN_HOLD_S * _NS)
        self.ch_max_hold_ns = int(CH_MAX_HOLD_S * (0.9 + 0.2 * jitter) * _NS)
        self.ch_reopen_gap_ns = int(CH_REOPEN_GAP_S * (0.9 + 0.2 * jitter) * _NS)
        self.pending_open_timeout_ns = int(PENDING_OPEN_TIMEOUT_S * _NS)
        self.activity_deadline_ns = int(ACTIVITY_DEADLINE_S * (0.92 + 0.08 * jitter) * _NS)
        self.rt_window_ns = int(RT_WINDOW_S * _NS)
        self.backoff_window_ns = int(BACKOFF_WINDOW_S * _NS)
        self.backoff_cooldown_ns = int(BACKOFF_COOLDOWN_S * _NS)
        self.kappa_rt_history_ns = int(KAPPA_RT_HISTORY_S * _NS)
        self.kappa_min_lookback_ns = int(KAPPA_MIN_LOOKBACK_S * _NS)

        self.inv: dict[str, dict[int, _Inv]] = {}
        self.books_state: dict[str, dict[int, _BookState]] = {}
        self._sim_id: dict[str, str] = {}
        self._step_ts_ns: int = 0
        self._active_validator: str | None = None

        bt.logging.info(
            f"[ApexTaker uid={self.uid}] APEX-TAKER clip={CLIP} exch_min={self.exch_min} "
            f"rebate(enter={REBATE_ENTER_BPS}bps tp={RB_TP_BPS}/sl={RB_SL_BPS}) "
            f"churn(vol_floor={VOL_FLOOR_BPS} sl={CH_SL_BPS} trail={CH_TRAIL_BPS}/arm={CH_ARM_BPS} "
            f"hold<={CH_MAX_HOLD_S:.0f}s gap={CH_REOPEN_GAP_S:.0f}s) abs_stop={ABS_STOP_BPS}bps "
            f"inv_cap={MAX_INVENTORY_LOTS}lot ema={EMA_FAST_N}/{EMA_SLOW_N} vol_ema={VOL_EMA_N} "
            f"rt_max={RT_MAX} rt_log={MAIN_VALIDATOR[:8]}"
        )

    # --------------------------------------------------------------- lifecycle
    def update(self, state: MarketSimulationStateUpdate) -> None:
        self._step_ts_ns = int(state.timestamp)
        self._active_validator = state.dendrite.hotkey
        self._ensure_simulation(self._active_validator, state.config.simulation_id)
        super().update(state)

    def _ensure_simulation(self, validator: str, simulation_id: str | None) -> None:
        if self._sim_id.get(validator) == simulation_id:
            return
        self.inv.pop(validator, None)
        self.books_state.pop(validator, None)
        if simulation_id is not None:
            self._sim_id[validator] = simulation_id
        else:
            self._sim_id.pop(validator, None)
        bt.logging.info(f"[ApexTaker uid={self.uid}] new simulation: {validator[:8]} sim_id={simulation_id}")

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        validator = state.dendrite.hotkey
        cfg = self.simulation_config
        self._sync_precision(cfg.priceDecimals, cfg.volumeDecimals)
        vol_dp = cfg.volumeDecimals
        volume_cap = CAPITAL_TURNOVER_CAP * cfg.miner_wealth * VOLUME_SAFETY
        now = state.timestamp

        for book_id in sorted(self.accounts.keys()):
            book = state.books.get(book_id)
            account = self.accounts.get(book_id) if book else None
            if book is None or account is None:
                continue
            try:
                self._step_book(response, validator, book_id, book, account, vol_dp, volume_cap, now)
            except Exception as ex:
                bt.logging.warning(f"[ApexTaker uid={self.uid}] step {book_id}: {ex}")
        return response

    # --------------------------------------------------------------- per-book
    def _step_book(self, response, validator, book_id, book, account, vol_dp, volume_cap, now) -> None:
        if not book.bids or not book.asks:
            return
        best_bid = book.bids[0].price
        best_ask = book.asks[0].price
        mid = 0.5 * (best_bid + best_ask)
        if mid <= 0 or best_bid <= 0 or best_ask <= 0:
            return

        inv = self._inv(validator, book_id)
        st = self._bstate(validator, book_id)
        if st.seen_ns == 0:
            st.seen_ns = now
        if self._prune_rt_events(st, now):
            self._refresh_book_kappa(validator, book_id, now)
        self._update_signals(st, mid)

        net = self._net_qty(inv)
        holding = abs(net) >= self.exch_min
        taker_fee = self._taker_fee_rate(account)

        # 1) RISK TRIM — drain any inventory above the hard cap before anything else.
        if self._risk_trim(response, book_id, account, inv, net, best_bid, best_ask, mid, vol_dp):
            return

        # 2) MANAGE an open position — abs stop / lane stop / reversal / trailing / target / max-hold.
        if holding:
            self._manage_position(response, book_id, account, st, inv, net,
                                  best_bid, best_ask, mid, taker_fee, vol_dp, now)
            return

        # 3) FLAT — try to open (rebate-scalp or volatility-gated leaned churn), gated by backoff/budget.
        if self._try_open(response, validator, book_id, account, st, book,
                          best_bid, best_ask, mid, taker_fee, volume_cap, now, vol_dp):
            return

        # 4) ACTIVITY BACKSTOP — guarantee >=1 RT per window so activity stays 1.0 (even on idle books).
        if self._activity_elapsed(st, now) >= self.activity_deadline_ns:
            self._activity_close(response, validator, book_id, account, st, inv, net,
                                 best_bid, best_ask, mid, vol_dp)

    # --------------------------------------------------------------- signals
    def _update_signals(self, st: _BookState, mid: float) -> None:
        if st.ema_fast <= 0.0:
            st.ema_fast = mid
            st.ema_slow = mid
            st.prev_mid = mid
            return
        st.ema_fast += self._alpha_fast * (mid - st.ema_fast)
        st.ema_slow += self._alpha_slow * (mid - st.ema_slow)
        if st.prev_mid > 0.0:
            move_bps = abs(mid - st.prev_mid) / st.prev_mid * 1e4
            st.vol_ema += self._alpha_vol * (move_bps - st.vol_ema)
        st.prev_mid = mid

    def _drift_bps(self, st: _BookState) -> float:
        """Signed trend: fast EMA above slow => uptrend (+), below => downtrend (-)."""
        if st.ema_slow <= 0.0:
            return 0.0
        return (st.ema_fast - st.ema_slow) / st.ema_slow * 1e4

    @staticmethod
    def _imbalance_bps(book, mid: float) -> float:
        """Signed microprice imbalance: bid-heavy => + (up pressure), ask-heavy => - (down pressure)."""
        bid, ask = book.bids[0], book.asks[0]
        denom = bid.quantity + ask.quantity
        if denom <= 0 or mid <= 0:
            return 0.0
        microprice = (ask.price * bid.quantity + bid.price * ask.quantity) / denom
        return (microprice - mid) / mid * 1e4

    def _lean_direction(self, st: _BookState, book, mid: float) -> int:
        """Direction to open: lean WITH the EMA drift when it is meaningful, else with order flow
        (microprice). Always returns a side — this is a weak-but-better-than-136's-coinflip lean."""
        drift = self._drift_bps(st)
        if drift >= CH_DRIFT_DIR_BPS:
            return OrderDirection.BUY
        if drift <= -CH_DRIFT_DIR_BPS:
            return OrderDirection.SELL
        return OrderDirection.BUY if self._imbalance_bps(book, mid) >= 0 else OrderDirection.SELL

    # --------------------------------------------------------------- entry
    def _try_open(self, response, validator, book_id, account, st, book,
                  best_bid, best_ask, mid, taker_fee, volume_cap, now, vol_dp) -> bool:
        if now < st.backoff_until_ns:
            return False
        # In-flight guard: an open we already submitted may not have filled yet (a market fill can lag
        # past the reopen gap). Until we SEE the fill (which flips the book to "holding") or it times out,
        # do not submit another open — re-opening in that window is what stacked a second clip -> over-cap.
        if st.pending_open_ns and (now - st.pending_open_ns) < self.pending_open_timeout_ns:
            return False
        if not self._budget_ok(validator, book_id, st, now, volume_cap):
            return False

        spread_bps = (best_ask - best_bid) / mid * 1e4 if mid > 0 else 1e9
        rebate_bps = (-taker_fee * 1e4) if taker_fee is not None else 0.0
        rebate_edge = 2.0 * rebate_bps - spread_bps   # rebate covering both crossings of a zero-alpha RT

        # ---- REBATE lane: unconditional +EV scalp where the rebate covers the spread (mirror 126) ----
        if rebate_edge >= REBATE_ENTER_BPS:
            if st.last_open_ns and (now - st.last_open_ns) < self.rb_reopen_gap_ns:
                return False
            direction = OrderDirection.BUY if self._imbalance_bps(book, mid) >= 0 else OrderDirection.SELL
            return self._open(response, book_id, account, st, direction, best_bid, best_ask, mid, LANE_REBATE, vol_dp, now)

        # ---- CHURN lane: permissive, leaned, on VOLATILE fee books (positive-skew exit carries the EV) ----
        if st.vol_ema < VOL_FLOOR_BPS:
            return False                       # dead-calm + no rebate => no edge to harvest => idle
        if spread_bps > CH_MAX_SPREAD_BPS:
            return False                       # too wide to cross repeatedly as a taker => skip (would bleed)
        if st.last_open_ns and (now - st.last_open_ns) < self.ch_reopen_gap_ns:
            return False
        direction = self._lean_direction(st, book, mid)
        return self._open(response, book_id, account, st, direction, best_bid, best_ask, mid, LANE_CHURN, vol_dp, now)

    def _open(self, response, book_id, account, st, direction, best_bid, best_ask, mid, lane, vol_dp, now) -> bool:
        q = round(self.clip, vol_dp)
        if q < self.exch_min:
            return False
        if direction == OrderDirection.BUY:
            if self._avail(account.quote_balance) < q * best_ask:
                return False
            self._submit_market(response, book_id, OrderDirection.BUY, q,
                                settlement=self._loan_settlement(account))
        else:
            if self._avail(account.base_balance) < q:        # never naked-short from flat
                return False
            self._submit_market(response, book_id, OrderDirection.SELL, q)
        st.lane = lane
        st.entry_mid = mid                # measure the position's move vs the entry mid (see _manage_position)
        st.peak_move_bps = 0.0
        st.last_open_ns = now
        st.pending_open_ns = now          # in-flight until the fill is seen (guards against a stacked re-open)
        return True

    # --------------------------------------------------------------- exit
    def _manage_position(self, response, book_id, account, st, inv, net,
                         best_bid, best_ask, mid, taker_fee, vol_dp, now) -> bool:
        """Close the whole position this step if an exit fires. gross_bps = current unrealized move
        vs the FIFO-average entry, in the position's favour."""
        if net > 0:
            avg = self._side_avg(inv.longs)
            ts0 = inv.longs[0][0]
        else:
            avg = self._side_avg(inv.shorts)
            ts0 = inv.shorts[0][0]
        held = now - ts0
        # Measure the FAVOURABLE MID MOVE since entry — NOT mark-to-bid. A taker that just crossed reads
        # as ~spread underwater on a mark-to-bid gross, which trips the stop on the entry cost itself
        # (the death-by-abs-stop we saw). entry_mid==0 (carryover / pre-restart lot) falls back to the
        # mark-to-bid gross so the absolute circuit-breaker still protects it.
        if st.entry_mid > 0:
            move_bps = (mid - st.entry_mid) / st.entry_mid * 1e4
            if net < 0:
                move_bps = -move_bps
        elif avg > 0:
            move_bps = (best_bid - avg) / avg * 1e4 if net > 0 else (avg - best_ask) / avg * 1e4
        else:
            move_bps = 0.0
        st.peak_move_bps = max(st.peak_move_bps, move_bps)

        reason = self._exit_reason(st, move_bps, held, net, taker_fee)
        if reason is None:
            return False
        self._close_all(response, book_id, account, inv, net, best_bid, best_ask, vol_dp, reason)
        return True

    def _exit_reason(self, st, move_bps, held, net, taker_fee):
        # Absolute circuit breaker first — caps the cubic tail regardless of lane (adverse MID move).
        if move_bps <= -ABS_STOP_BPS:
            return "abs"
        if st.lane == LANE_REBATE:
            if held < self.rb_min_hold_ns:
                return None
            if move_bps >= RB_TP_BPS:
                return "rb_tp"
            # -RB_SL is a cushioned stop only while the rebate still covers it; else let ABS_STOP bound it.
            rebate_bps = (-taker_fee * 1e4) if taker_fee is not None else 0.0
            if move_bps <= -RB_SL_BPS and 2.0 * rebate_bps >= RB_SL_BPS:
                return "rb_sl"
            if held >= self.rb_max_hold_ns:
                return "rb_time"
            return None
        # churn lane: tiny stop (cut losers), ride winners (reversal / trailing give-back), max hold.
        if held < self.ch_min_hold_ns:
            return None
        if move_bps <= -CH_SL_BPS:
            return "ch_sl"
        drift = self._drift_bps(st)   # reversal: the trend we leaned into has flipped against us
        if (net > 0 and drift <= -CH_DRIFT_DIR_BPS) or (net < 0 and drift >= CH_DRIFT_DIR_BPS):
            return "ch_rev"
        if st.peak_move_bps >= CH_ARM_BPS and move_bps <= st.peak_move_bps - CH_TRAIL_BPS:
            return "ch_trail"
        if held >= self.ch_max_hold_ns:
            return "ch_time"
        return None

    def _close_all(self, response, book_id, account, inv, net, best_bid, best_ask, vol_dp, reason) -> None:
        slip = (EXIT_SLIPPAGE_ABS_BPS if reason == "abs" else EXIT_SLIPPAGE_BPS) / 1e4
        if net > 0:
            q = round(min(self._long_qty(inv), self._avail(account.base_balance)), vol_dp)
            if q < self.exch_min:
                return
            px = round(best_bid * (1.0 - slip), self._price_decimals)
            self._submit_limit(response, book_id, OrderDirection.SELL, q, px, ioc=True)
        else:
            q = round(self._short_qty(inv), vol_dp)
            if q < self.exch_min:
                return
            px = round(best_ask * (1.0 + slip), self._price_decimals)
            self._submit_limit(response, book_id, OrderDirection.BUY, q, px, ioc=True,
                               settlement=self._loan_settlement(account))
        if self._rt_log_enabled(self._active_validator or ""):
            bt.logging.info(f"[ApexTaker uid={self.uid}] CLOSE book={book_id} reason={reason} net={net:+.4f} q={q}")

    # --------------------------------------------------------------- risk / activity
    def _risk_trim(self, response, book_id, account, inv, net, best_bid, best_ask, mid, vol_dp) -> bool:
        qty = abs(net)
        if qty < self._flat_eps:
            return False
        lot_cap = MAX_INVENTORY_LOTS * self.clip
        equity = self._book_equity(account, mid)
        notional_cap = MAX_INVENTORY_EQUITY_FRAC * equity if equity > 0 else float("inf")
        excess = max(qty - lot_cap, (qty * mid - notional_cap) / mid if mid > 0 else 0.0)
        if excess <= self._flat_eps:
            return False
        trim = round(min(qty, max(excess, self.exch_min)), vol_dp)
        if trim < self.exch_min:
            return False
        slip = RISK_TRIM_SLIPPAGE_BPS / 1e4
        if net > 0:
            trim = round(min(trim, self._avail(account.base_balance)), vol_dp)
            if trim < self.exch_min:
                return False
            px = round(best_bid * (1.0 - slip), self._price_decimals)   # cross the BID so the trim FILLS
            self._submit_limit(response, book_id, OrderDirection.SELL, trim, px, ioc=True)
        else:
            px = round(best_ask * (1.0 + slip), self._price_decimals)   # lift the ASK so the trim FILLS
            self._submit_limit(response, book_id, OrderDirection.BUY, trim, px, ioc=True,
                               settlement=self._loan_settlement(account))
        bt.logging.info(f"[ApexTaker uid={self.uid}] RISK-TRIM book={book_id} net={net:+.4f} trim={trim}")
        return True

    def _activity_close(self, response, validator, book_id, account, st, inv, net,
                        best_bid, best_ask, mid, vol_dp) -> bool:
        """Force exactly one round-trip-producing close (or seed one clip if flat) so the book keeps
        activity=1.0. The seed leans WITH the EMA drift (down-trend => short) so a forced RT is not
        dead-wrong on a trending book."""
        slip = FORCE_TRIM_SLIPPAGE_BPS / 1e4
        long_q, short_q = self._long_qty(inv), self._short_qty(inv)
        base_avail = self._avail(account.base_balance)
        quote_avail = self._avail(account.quote_balance)
        q = round(self.clip, vol_dp)
        if long_q >= self.exch_min:
            qq = round(min(long_q, base_avail), vol_dp)
            if qq < self.exch_min:
                return False
            px = round(best_bid * (1.0 - slip), self._price_decimals)
            self._submit_limit(response, book_id, OrderDirection.SELL, qq, px, ioc=True)
        elif short_q >= self.exch_min:
            qq = round(short_q, vol_dp)
            if qq < self.exch_min:
                return False
            px = round(best_ask * (1.0 + slip), self._price_decimals)
            self._submit_limit(response, book_id, OrderDirection.BUY, qq, px, ioc=True,
                               settlement=self._loan_settlement(account))
        else:
            if q < self.exch_min:
                return False
            if self._drift_bps(st) >= 0.0:                       # uptrend (or flat) => seed long
                if quote_avail < q * best_ask * (1.0 + slip):
                    return False
                px = round(best_ask * (1.0 + slip), self._price_decimals)
                self._submit_limit(response, book_id, OrderDirection.BUY, q, px, ioc=True)
            else:                                                # downtrend => seed short
                if base_avail < q:
                    return False
                px = round(best_bid * (1.0 - slip), self._price_decimals)
                self._submit_limit(response, book_id, OrderDirection.SELL, q, px, ioc=True)
            st.lane = LANE_REBATE     # treat the seed like a fast scalp so the manager closes it quickly
            st.entry_mid = mid
            st.peak_move_bps = 0.0
        return True

    # --------------------------------------------------------------- events / FIFO (validator-faithful)
    def onTrade(self, event: TradeEvent, validator: str | None = None) -> None:
        if event.bookId is None:
            return
        validator = validator or self._active_validator
        if validator is None:
            return
        if self.uid == event.takerAgentId:
            is_buy = event.side == OrderDirection.BUY
            fee = event.takerFee
        elif self.uid == event.makerAgentId:
            is_buy = event.side == OrderDirection.SELL
            fee = event.makerFee
        else:
            return
        ts_ns = int(event.timestamp) if event.timestamp else self._step_ts_ns
        self._record_trade_volume(validator, event.bookId, event.quantity, event.price, ts_ns)
        self._apply_fill(validator, event.bookId, is_buy, event.quantity, event.price, fee, ts_ns)

    def _apply_fill(self, validator, book_id, is_buy, qty, price, fee, ts) -> None:
        inv = self._inv(validator, book_id)
        realized, rtv, matched_ts, gross = self._match_fifo(inv, is_buy, qty, price, fee, ts)
        st = self._bstate(validator, book_id)
        st.pending_open_ns = 0          # a fill arrived -> the in-flight open (if any) is resolved
        if rtv > 0:
            kappa_before = st.kappa3
            st.last_rt_ns = ts
            self._record_rt_close(validator, book_id, ts, realized)
            self._regime_backoff(st, book_id, ts)
            if abs(self._net_qty(inv)) < self.exch_min:   # now flat -> clear position bookkeeping
                st.lane = ""
                st.entry_mid = 0.0
                st.peak_move_bps = 0.0
            self._log_rt(validator, book_id, ts,
                         hold_s=(ts - matched_ts) / _NS if matched_ts else None,
                         side=("buy" if is_buy else "sell"), exit_px=price, rtv=rtv,
                         gross=gross, net=realized, kappa_before=kappa_before, kappa_after=st.kappa3)

    def _match_fifo(self, inv, is_buy, qty, price, fee, ts):
        close_book = inv.shorts if is_buy else inv.longs
        open_book = inv.longs if is_buy else inv.shorts
        realized = gross = rtv = 0.0
        remaining = qty
        matched_ts = None
        qinv = 1.0 / qty if qty > 0 else 0.0
        while remaining > self._flat_eps and close_book:
            o_ts, o_qty, o_px, o_fee = close_book[0]
            if matched_ts is None:
                matched_ts = o_ts
            take = min(o_qty, remaining)
            price_pnl = (o_px - price) * take if is_buy else (price - o_px) * take
            if o_qty <= remaining + self._flat_eps:
                close_fee = fee * o_qty * qinv
                open_fee = o_fee
                close_book.popleft()
            else:
                close_fee = fee * take * qinv
                open_fee = o_fee * (take / o_qty)
                close_book[0] = (o_ts, o_qty - take, o_px, o_fee - open_fee)
            realized += price_pnl - open_fee - close_fee
            gross += price_pnl
            rtv += take
            remaining -= take
        if remaining > self._flat_eps:
            open_book.append((ts, remaining, price, fee * remaining * qinv))
        return realized, rtv, matched_ts, gross

    # --------------------------------------------------------------- toxic-book backoff (skew-safe)
    def _regime_backoff(self, st, book_id, now_ns) -> None:
        cutoff = now_ns - self.backoff_window_ns
        recent = [p for t, p in st.rt_events if t >= cutoff]
        if len(recent) < BACKOFF_MIN_RTS:
            return
        # Only a GENUINELY broken book (every recent RT a loss) is idled. A positive-skew taker is
        # net-negative over many windows by design, so net<0 alone must never trigger a backoff.
        if all(p <= 0.0 for p in recent):
            until = now_ns + self.backoff_cooldown_ns
            if until > st.backoff_until_ns:
                st.backoff_until_ns = until
                bt.logging.info(
                    f"[ApexTaker uid={self.uid}] REGIME-BACKOFF book={book_id} all-loss "
                    f"n={len(recent)} net={sum(recent):+.3f} idle={BACKOFF_COOLDOWN_S:.0f}s")

    # --------------------------------------------------------------- kappa-3 (validator-faithful mirror)
    def _prune_rt_events(self, st, now) -> bool:
        cutoff = now - self.kappa_rt_history_ns
        before = len(st.rt_events)
        st.rt_events = [(t, p) for t, p in st.rt_events if t >= cutoff]
        return len(st.rt_events) != before

    def _record_rt_close(self, validator, book_id, ts, net_pnl) -> None:
        st = self._bstate(validator, book_id)
        self._prune_rt_events(st, ts)
        st.rt_events.append((ts, net_pnl))
        self._refresh_book_kappa(validator, book_id, ts)

    def _global_rt_timestamps(self, validator, now) -> list[int]:
        cutoff = now - self.kappa_rt_history_ns
        ts_set: set[int] = set()
        for st in self.books_state.get(validator, {}).values():
            for ts, _ in st.rt_events:
                if ts >= cutoff:
                    ts_set.add(ts)
        return sorted(ts_set)

    def _book_pnl_series(self, validator, book_id, now) -> list[float]:
        timestamps = self._global_rt_timestamps(validator, now)
        if not timestamps:
            return []
        cutoff = now - self.kappa_rt_history_ns
        by_ts = {t: p for t, p in self._bstate(validator, book_id).rt_events if t >= cutoff}
        return [by_ts.get(ts, 0.0) for ts in timestamps]

    @staticmethod
    def _median(values) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        m = len(s) // 2
        return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])

    @classmethod
    def _kappa3_raw(cls, pnl_series, tau=KAPPA_TAU):
        if not pnl_series or sum(1 for x in pnl_series if x != 0.0) < KAPPA_MIN_OBS:
            return None
        med = cls._median(pnl_series)
        mad = max(cls._median([abs(x - med) for x in pnl_series]), 1e-6)
        returns = [x / mad for x in pnl_series]
        n = len(returns)
        mean_r = sum(returns) / n
        lpm3 = sum(max(tau - r, 0.0) ** 3 for r in returns) / n
        upm3 = sum(max(r - tau, 0.0) ** 3 for r in returns) / n
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / n)
        reg = ((abs(mean_r) + std_r) * 0.1) ** 3
        eps = 1e-2 if mean_r > tau else 1e-6
        if lpm3 > eps:
            return (mean_r - tau) / ((lpm3 + reg) ** (1.0 / 3.0))
        if mean_r > tau:
            return (mean_r - tau) / ((upm3 + reg) ** (1.0 / 3.0))
        return 0.0

    def _kappa_history_ready(self, validator, now) -> bool:
        ts = self._global_rt_timestamps(validator, now)
        return len(ts) >= 2 and ts[-1] - ts[0] >= self.kappa_min_lookback_ns

    def _refresh_book_kappa(self, validator, book_id, now) -> None:
        st = self._bstate(validator, book_id)
        if not self._kappa_history_ready(validator, now):
            st.kappa3 = None
            return
        st.kappa3 = self._kappa3_raw(self._book_pnl_series(validator, book_id, now))

    def _rt_count(self, st, now) -> int:
        cutoff = now - self.rt_window_ns
        return sum(1 for ts, _ in st.rt_events if ts >= cutoff)

    # --------------------------------------------------------------- volume / activity
    def _record_trade_volume(self, validator, book_id, qty, price, ts_ns) -> None:
        vol = float(qty) * float(price)
        if vol > 0:
            self._bstate(validator, book_id).vol_log.append((ts_ns, vol))

    def _prune_vol_log(self, st, now_ns) -> None:
        cutoff = now_ns - self.volume_assessment_ns
        st.vol_log = [(t, v) for t, v in st.vol_log if t >= cutoff]

    def _rolled_quote_volume(self, validator, book_id, now_ns) -> float:
        st = self._bstate(validator, book_id)
        self._prune_vol_log(st, now_ns)
        return sum(v for _, v in st.vol_log)

    def _budget_ok(self, validator, book_id, st, now, volume_cap) -> bool:
        return (self._rt_count(st, now) < RT_MAX
                and self._rolled_quote_volume(validator, book_id, now) < volume_cap)

    @staticmethod
    def _activity_elapsed(st, now) -> int:
        ref = st.last_rt_ns if st.last_rt_ns > 0 else st.seen_ns
        return now - ref

    # --------------------------------------------------------------- state / precision / helpers
    def _inv(self, validator, book_id) -> _Inv:
        return self.inv.setdefault(validator, {}).setdefault(book_id, _Inv())

    def _bstate(self, validator, book_id) -> _BookState:
        return self.books_state.setdefault(validator, {}).setdefault(book_id, _BookState())

    @staticmethod
    def _long_qty(inv) -> float:
        return sum(q for _, q, _, _ in inv.longs)

    @staticmethod
    def _short_qty(inv) -> float:
        return sum(q for _, q, _, _ in inv.shorts)

    def _net_qty(self, inv) -> float:
        return self._long_qty(inv) - self._short_qty(inv)

    @staticmethod
    def _side_avg(lots) -> float:
        tot = sum(q for _, q, _, _ in lots)
        return sum(q * p for _, q, p, _ in lots) / tot if tot > 0 else 0.0

    def _sync_precision(self, price_decimals, volume_decimals) -> None:
        if price_decimals == self._price_decimals and volume_decimals == self._volume_decimals:
            return
        self._price_decimals = price_decimals
        self._volume_decimals = volume_decimals
        self._tick = 10 ** (-price_decimals)
        self.clip = round(max(CLIP, 10 ** (-volume_decimals)), volume_decimals)
        self.exch_min = max(EXCHANGE_MIN_ORDER_SIZE, 10 ** (-volume_decimals))
        self._flat_eps = 0.5 * 10 ** (-volume_decimals)
        bt.logging.info(
            f"[ApexTaker uid={self.uid}] priceDecimals={price_decimals} tick={self._tick} "
            f"volumeDecimals={volume_decimals} clip={self.clip} exch_min={self.exch_min}")

    @staticmethod
    def _avail(balance) -> float:
        if balance is None:
            return 0.0
        return (balance.free or 0.0) + (balance.reserved or 0.0)

    def _book_equity(self, account, mid) -> float:
        q, b = account.quote_balance, account.base_balance
        quote = ((q.free or 0.0) + (q.reserved or 0.0)) if q else 0.0
        base = ((b.free or 0.0) + (b.reserved or 0.0)) if b else 0.0
        return quote + base * mid

    @staticmethod
    def _loan_settlement(account) -> LoanSettlementOption:
        quote_loan = getattr(account, "quote_loan", 0.0) or 0.0
        return LoanSettlementOption.FIFO if quote_loan > 0 else LoanSettlementOption.NONE

    def _taker_fee_rate(self, account):
        fees = getattr(account, "fees", None)
        rate = getattr(fees, "taker_fee_rate", None) if fees is not None else None
        try:
            return float(rate) if rate is not None else None
        except (TypeError, ValueError):
            return None

    def _submit_market(self, response, book_id, direction, qty, *, settlement=LoanSettlementOption.NONE) -> None:
        kwargs: dict[str, Any] = {
            "book_id": book_id, "direction": direction, "quantity": qty,
            "currency": OrderCurrency.BASE, "stp": STP.CANCEL_OLDEST,
        }
        if settlement != LoanSettlementOption.NONE:
            kwargs["settlement_option"] = settlement
        response.market_order(**kwargs)

    def _submit_limit(self, response, book_id, direction, qty, price, *, ioc=True,
                      settlement=LoanSettlementOption.NONE) -> None:
        kwargs: dict[str, Any] = {
            "book_id": book_id, "direction": direction, "quantity": qty, "price": price,
            "stp": STP.CANCEL_OLDEST, "timeInForce": TimeInForce.IOC,
        }
        if not ioc:
            kwargs["timeInForce"] = TimeInForce.GTT
        if settlement != LoanSettlementOption.NONE:
            kwargs["settlement_option"] = settlement
        response.limit_order(**kwargs)

    # --------------------------------------------------------------- RT logging (scoring validator only)
    @staticmethod
    def _rt_log_enabled(validator) -> bool:
        return validator == MAIN_VALIDATOR

    @staticmethod
    def _fmt_kappa(before, after) -> str:
        if before is None and after is None:
            return "n/a"
        if before is None:
            return f"n/a->{after:.4f}"
        if after is None:
            return f"{before:.4f}->n/a"
        return f"{before:.4f}->{after:.4f}"

    def _log_rt(self, validator, book_id, ts, *, hold_s, side, exit_px, rtv, gross, net,
                kappa_before, kappa_after) -> None:
        if not self._rt_log_enabled(validator):
            return
        st = self._bstate(validator, book_id)
        hold_str = f"{hold_s:.2f}" if hold_s is not None else "n/a"
        bt.logging.info(
            f"[ApexTaker uid={self.uid} RT] book={book_id} lane={st.lane or '-'} close={side} "
            f"rtv={rtv:.4f} exit={exit_px:.4f} hold_s={hold_str} gross={gross:+.4f} net={net:+.4f} "
            f"kappa={self._fmt_kappa(kappa_before, kappa_after)}")


if __name__ == "__main__":
    launch(ApexTakerAgent)
