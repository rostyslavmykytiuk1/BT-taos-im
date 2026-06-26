"""
PatientTakerAgent — uid179-style PATIENT, rebate-funded mean-reversion taker for subnet 79.
(Derived from TakerScalperAgent; the ONLY behavioural change is the EXIT — patient reversion instead of
the fast TP/SL scalp. Sleep is ON by default and there are NO AGENT_PARAMS toggles.)

Modelled on the current #1 taker uid179 (live trade data, "weak taker market": 1751 RTs / 128 books / ~18min):
  * NOT a fast scalper. Median hold 36s (mean 88s, p90 255s, max 1159s); only ~20% of RTs < 2s.
  * PAYS the spread to enter+exit (gross −3.6bps mean, only 18% of RTs gross-positive); the deep REBATE
    (both legs, +5.3bps mean) funds a thin-POSITIVE net (+1.7bps/RT mean, 70% win) -> smooth -> high kappa.
  * Mechanism: enter on a rebate, then HOLD until the price REVERTS to near-entry, then exit and bank the
    rebate. NO tight stop — it holds adverse moves out and lets them mean-revert; only a rare genuine
    runaway realises a (bounded) loss. The patient exit is the proven differentiator: our fast-cut takers
    (TP2.5/SL4-12/hold3s) exit at adverse prices and bleed; uid179 waits for the rebate-funded reversion.

So this agent ENTERS like TakerScalper (rebate-gated, est_pnl>0, microprice-biased one lot) but EXITS
patiently: hold until gross reverts to within EXIT_REVERT_GROSS_BPS of entry (-> net positive via the
rebate), with a WIDE CATASTROPHE_GROSS_BPS backstop for a true trend and a long MAX_HOLD_S patience window.

SLEEP is FEE-ONLY with hysteresis (ALWAYS ON, no param): a book sleeps when its taker fee is POSITIVE
(it pays to take) and wakes once the fee is a rebate beyond SLEEP_WAKE_FEE_BPS; hard-capped at MAX_SLEEP
so we never overshoot the validator's ~48-inactive cliff (which injects hard 0.0s into the kappa median).
NB: reward.py::calculate_kappa_score is the real aggregation; the kappa helpers here are logging-only —
judge live on the endpoint, not the internal proxy.

Market orders only: full immediate fills, nothing resting. One position per book, strictly sequential.

Per book each tick (sleep/wake decided once up front in _update_sleep_states):
  reconcile -> pending? wait : held? close(revert/catastrophe/max-hold) : open(profit | activity) [asleep -> skip open]

RT logging -> main validator only.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol import FinanceAgentResponse, MarketSimulationStateUpdate
from taos.im.protocol.events import OrderPlacementEvent, TradeEvent
from taos.im.protocol.models import (
    LoanSettlementOption,
    OrderCurrency,
    OrderDirection,
    STP,
)

_NS = 1_000_000_000

# Exchange floor for any BASE order (sim `minOrderSize`, not exposed to agents).
EXCHANGE_MIN_ORDER_SIZE = 0.25
# Entry lot, comfortably above the floor so a fill is always a sellable holding.
LOT = 0.3

# Hold / exit: PATIENT mean-reversion (uid179 style). Enter on rebate, then HOLD for the price to revert
# to near-entry and exit banking the rebate — NOT a fast scalp. uid179 live: median hold 36s, exit gross
# ~−1.5bps, net +1.7bps/RT, 70% win, 128 books. No tight stop; only a wide catastrophe + long max-hold.
MIN_HOLD_S = 1.5                  # floor before any exit (avoids own-market-open-impact false exits)
MAX_HOLD_S = 300.0               # patience window: hold up to ~5min for a reversion (uid179 p90=255s,
                                 # median 36s). If still unreverted by here, give up and take the
                                 # rebate-cushioned exit at whatever gross (the bounded max-hold tail).
EXIT_REVERT_GROSS_BPS = -1.5      # EXIT once gross reverts to within 1.5bps of entry (or better). net =
                                 # gross + 2×rebate is then solidly positive. Matches uid179's ~−1.4bps
                                 # median exit. Starts at −spread after the taker cross, so this does NOT
                                 # fire immediately on a wide book — it waits for the reversion.
CATASTROPHE_GROSS_BPS = 30.0      # the ONLY hard cut: a genuine runaway that never reverted. WIDE so normal
                                 # reversion-holds (which wander to ~−10bps, uid179 gross p10) never trip it;
                                 # bounds the rare cube-downside tail (vs uid179's uncapped ~−70bps worst RT).

# FEE-BASED SLEEP with hysteresis (2026-06-23 rework). Spread is too noisy to drive the sleep SET (it
# would thrash tick-to-tick); the FEE is stable, so SLEEP keys on FEE only. A book SLEEPS when its taker
# fee is POSITIVE (it pays to take) and WAKES only once the fee is a rebate beyond SLEEP_WAKE_FEE_BPS
# (fee < -2bps); in the (-2bps, 0] band it KEEPS its current state (hysteresis). NOTE: spread is NOT
# ignored for OPENS — the open gate uses est_pnl>0 (spread-aware), only the sleep SET ignores spread.
SLEEP_WAKE_FEE_BPS = -2.0         # sleep -> wake only when fee drops below this (rebate > 2bps); enter = fee > 0
# Hard cap on simultaneously-sleeping books — LOAD-BEARING: the validator's reward.py injects hard 0.0
# entries into the kappa median once ~48 books are inactive, which CRATERS the score. 40 keeps a
# deliberate ~8-book margin below that cliff (40 sleep => >=88 of 128 active). When more books want to
# sleep than free slots, the WORST (highest-fee) ones sleep and the rest stay awake; once full, NO more.
MAX_SLEEP = 40

# Margin leverage for a SELL open when free base is insufficient.
SHORT_LEVERAGE = 1.0

RT_WINDOW_S = 570.0                # validator activity sampling window (~10 min)
RT_MAX = 40                        # max profit RTs per book per window

# Force a taker RT once this long since the last RT (kept under RT_WINDOW_S).
ACTIVITY_DEADLINE_S = 500.0
# Min gap between RT closes and the next profit open (per book; throttles churn).
MIN_REOPEN_GAP_S = 2.0
# After submitting, wait this long for the fill before assuming the order was lost.
PENDING_TIMEOUT_S = 5.0

# Kappa-3 (3h history; logging / projection only — NOT a gate).
KAPPA_TAU = 0.0
KAPPA_MIN_OBS = 3
KAPPA_MIN_LOOKBACK_S = 5400.0      # 90 min
KAPPA_RT_HISTORY_S = 10_800.0      # 3h
KAPPA_MIN_REBATE_BPS = 2.0         # open ONLY when taker rebate >= 2bps (rate <= -2bp); aligns with the -2bps wake bar

# Volume cap (profit path only).
CAPITAL_TURNOVER_CAP = 10.0
VOLUME_SAFETY = 0.8
VOLUME_ASSESSMENT_NS = 86_400_000_000_000

# RT logs only for the scoring validator.
MAIN_VALIDATOR = "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF"


@dataclass
class _Position:
    qty: float = 0.0
    avg: float = 0.0
    entry_ts: int = 0
    entry_fee: float = 0.0


@dataclass
class _BookState:
    last_rt_ns: int = 0
    pending_ns: int = 0                 # last order submit ts (rule 3: wait for the fill)
    pending_kind: str = ""              # tag of the in-flight open (for RT logging)
    rt_events: list[tuple[int, float]] = field(default_factory=list)
    kappa3: float | None = None
    vol_log: list[tuple[int, float]] = field(default_factory=list)
    sleeping: bool = False             # fee>0 (pays to take) -> skip RTs so it ages out to kappa=None
    # (no position_open flag — one-position-per-book is enforced by the exch_min flat-gate in _step_book)


@dataclass
class _RtLogCtx:
    """Open snapshot stashed at submit; close_reason set at close; logged at the RT fill."""
    open_reason: str = "?"
    side: str = "?"
    open_rt_window_n: int = 0
    open_rt_pnl_list: str = "[]"
    est_pnl: float | None = None
    kappa_at_open: float | None = None
    kappa_proj: float | None = None
    taker_bps: float | None = None
    close_reason: str = "fill"


class PatientTakerAgent(FinanceSimulationAgent):
    def initialize(self) -> None:
        bt.logging.set_info()

        self.min_order_size = LOT
        self._min_qty = LOT / 2             # RT-log gate (meaningful close)
        self.exch_min = EXCHANGE_MIN_ORDER_SIZE
        self._flat_eps = 0.5 / 1e4  # overwritten by _sync_order_size on first respond
        self._volume_decimals: int | None = None
        self.volume_assessment_ns = VOLUME_ASSESSMENT_NS

        jitter = ((self.uid * 2654435761) % 1000) / 1000.0
        max_hold_s = MAX_HOLD_S * (0.92 + 0.16 * jitter)
        activity_s = ACTIVITY_DEADLINE_S * (0.92 + 0.08 * jitter)   # ~460-500s

        self.min_hold_ns = int(MIN_HOLD_S * _NS)
        self.max_hold_ns = int(max_hold_s * _NS)
        self.min_reopen_gap_ns = int(MIN_REOPEN_GAP_S * (0.9 + 0.2 * jitter) * _NS)
        self.rt_window_ns = int(RT_WINDOW_S * _NS)
        self.activity_deadline_ns = int(activity_s * _NS)
        self.pending_timeout_ns = int(PENDING_TIMEOUT_S * _NS)
        self.kappa_rt_history_ns = int(KAPPA_RT_HISTORY_S * _NS)
        self.kappa_min_lookback_ns = int(KAPPA_MIN_LOOKBACK_S * _NS)
        self.kappa_min_rebate_rate = -KAPPA_MIN_REBATE_BPS / 1e4
        self.sleep_wake_rate = SLEEP_WAKE_FEE_BPS / 1e4   # wake a sleeper only when fee < this (rebate > 2bps)

        # SLEEP: ALWAYS ON (the no_sleep AGENT_PARAM was removed). Fee-based sleep with hysteresis — a
        # book sleeps when its taker fee is positive and wakes once the fee is a rebate beyond
        # SLEEP_WAKE_FEE_BPS, hard-capped at MAX_SLEEP so we never overshoot the validator inactivity
        # cliff. Kept as a fixed-False attribute so the downstream sleep guard (`if self.no_sleep`) is
        # unchanged — it is now a permanent no-op, so the fee-sleep logic always runs.
        self.no_sleep = False

        # EXIT: PATIENT reversion (no tight stop, no AGENT_PARAM). Hold until the price reverts to within
        # EXIT_REVERT_GROSS_BPS of entry, then exit and bank the rebate; only a WIDE CATASTROPHE_GROSS_BPS
        # runaway or the MAX_HOLD_S patience window forces an earlier exit. See _close.

        self.positions: dict[str, dict[int, _Position]] = {}
        self.books_state: dict[str, dict[int, _BookState]] = {}
        self._sim_id: dict[str, str] = {}
        self._rt_log: dict[tuple[str, int], _RtLogCtx] = {}
        self._step_ts_ns: int = 0
        self._agent_start_ns: dict[str, int] = {}
        self._active_validator: str | None = None

        bt.logging.info(
            f"[PatientTaker uid={self.uid}] PATIENT-TAKER(uid179-style) lot={LOT} exch_min={self.exch_min} "
            f"hold={MIN_HOLD_S}-{max_hold_s:.0f}s exit=revert@{EXIT_REVERT_GROSS_BPS}bps "
            f"catastrophe={CATASTROPHE_GROSS_BPS}bps (NO tight stop) "
            f"reopen_gap={MIN_REOPEN_GAP_S}s rt_window={RT_WINDOW_S / 60:.0f}m max={RT_MAX} "
            f"activity_deadline={activity_s:.0f}s "
            f"open_gate(rebate>={KAPPA_MIN_REBATE_BPS}bps AND est_pnl>0 [spread-aware]) "
            f"sleep=ON-fee(sleep fee>0, wake fee<{SLEEP_WAKE_FEE_BPS}bps, "
            f"max_sleep={MAX_SLEEP}) rt_log={MAIN_VALIDATOR[:8]}"
        )

    def update(self, state: MarketSimulationStateUpdate) -> None:
        self._step_ts_ns = int(state.timestamp)
        self._active_validator = state.dendrite.hotkey
        # Reset before super().update() so the new sim's first fills don't hit stale state.
        self._ensure_simulation(self._active_validator, state.config.simulation_id)
        if self._agent_start_ns.get(self._active_validator, 0) == 0 and self._step_ts_ns > 0:
            self._agent_start_ns[self._active_validator] = self._step_ts_ns
        super().update(state)

    def _ensure_simulation(self, validator: str, simulation_id: str | None) -> None:
        """Drop per-validator state when the validator starts a new simulation."""
        if self._sim_id.get(validator) == simulation_id:
            return
        self._book_positions(validator).clear()
        self.books_state.pop(validator, None)
        self._rt_log = {k: v for k, v in self._rt_log.items() if k[0] != validator}
        self._agent_start_ns.pop(validator, None)   # reset only THIS validator's activity clock
        if simulation_id is not None:
            self._sim_id[validator] = simulation_id
        else:
            self._sim_id.pop(validator, None)
        bt.logging.info(
            f"[PatientTaker uid={self.uid}] new simulation: {validator[:8]} sim_id={simulation_id}"
        )

    def _update_sleep_states(self, validator: str, now: int) -> None:
        """FEE-BASED sleep with hysteresis (FEE only — the spread is too noisy to drive sleep). A book
        SLEEPS when its taker fee is POSITIVE (it pays to take) and WAKES only once the fee is a rebate
        beyond SLEEP_WAKE_FEE_BPS (fee < -2bps); in the (-2bps, 0] band it keeps its state. The sleeping
        set is hard-capped at MAX_SLEEP: when more books want to sleep than there are free slots, the
        WORST (highest-fee) ones sleep and the rest stay awake; once full, NO more books sleep (the
        >48-inactive cliff in reward.py would crater the score). no_sleep disables all of this."""
        if self.no_sleep:
            return

        rates: dict[int, float] = {}
        for book_id, account in self.accounts.items():
            r = self._taker_fee_rate(account)
            if r is not None:
                rates[book_id] = r

        # 1) WAKE: any sleeper whose fee recovered to a real rebate (fee < -2bps).
        for book_id, r in rates.items():
            st = self._bstate(validator, book_id)
            if st.sleeping and r < self.sleep_wake_rate:
                st.sleeping = False
                if self._rt_log_enabled(validator):
                    bt.logging.info(f"[PatientTaker uid={self.uid}] WAKE book={book_id} (fee={r * 1e4:+.2f}bps)")

        # 2) SLEEP: awake books that PAY (fee > 0), WORST-first, into the remaining budget only.
        #    Never exceed MAX_SLEEP; once full, paying books stay awake (no displacement).
        # Count sleepers over the FULL per-validator state, NOT over `rates`: a sleeping book whose
        # account/fees is momentarily None drops out of `rates`, and counting only `rates` would
        # undercount `asleep`, overstate `slots`, and let the true sleeping set exceed MAX_SLEEP (and
        # the 48-inactive cliff). Counting state keeps the cap strict regardless of None fees.
        asleep = sum(1 for s in self.books_state.get(validator, {}).values() if s.sleeping)
        slots = MAX_SLEEP - asleep
        if slots > 0:
            payers = sorted(
                ((r, b) for b, r in rates.items() if r > 0.0 and not self._bstate(validator, b).sleeping),
                reverse=True,   # highest fee (worst) first
            )
            for r, book_id in payers[:slots]:
                st = self._bstate(validator, book_id)
                st.sleeping = True
                if self._rt_log_enabled(validator):
                    bt.logging.info(f"[PatientTaker uid={self.uid}] SLEEP book={book_id} (fee={r * 1e4:+.2f}bps)")

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        validator = state.dendrite.hotkey
        cfg = self.simulation_config
        self._sync_order_size(cfg.volumeDecimals)

        vol_dp = cfg.volumeDecimals
        volume_cap = CAPITAL_TURNOVER_CAP * cfg.miner_wealth * VOLUME_SAFETY
        now = state.timestamp

        # Fee-based sleep: books that PAY to take (fee > 0) sleep; hysteresis (wake at fee < -2bps) + budget cap.
        self._update_sleep_states(validator, now)

        for book_id in sorted(self.accounts.keys()):
            book = state.books.get(book_id)
            account = self.accounts.get(book_id) if book else None
            if book is None or account is None:
                continue
            try:
                self._step_book(response, validator, book_id, book, account, vol_dp, volume_cap, now)
            except Exception as ex:
                bt.logging.warning(f"[PatientTaker uid={self.uid}] step {book_id}: {ex}")

        return response

    def _step_book(
        self, response, validator: str, book_id: int, book, account,
        vol_dp: int, volume_cap: float, now: int,
    ) -> None:
        """One sequential action per book: wait while an order is in flight, else close the
        held position or, when flat, decide whether to open."""
        pos = self._book_positions(validator).setdefault(book_id, _Position())
        st = self._bstate(validator, book_id)
        if self._agent_start_ns.get(validator, 0) == 0 and now > 0:
            self._agent_start_ns[validator] = now
        if self._prune_rt_events(st, now):
            self._refresh_book_kappa(validator, book_id, now)

        # Rule 3: one order at a time -> wait for the fill (or a timeout) before acting.
        if st.pending_ns and (now - st.pending_ns) < self.pending_timeout_ns:
            return
        st.pending_ns = 0   # presumed filled / lost; re-derive from the position

        # Reconcile tracked size to the real held base AFTER the in-flight guard: while a close is in
        # flight the exchange hasn't debited the base yet, so reconciling then could clamp against a
        # stale mid-settlement balance. Only reconcile once no order is pending.
        self._reconcile_position(account, pos, vol_dp)

        # Hold vs flat by the EXCHANGE-MIN threshold (the AR's robust rule), NOT flat_eps: a sub-exch-min
        # remnant is UN-CLOSEABLE (below the 0.25 order minimum), so treat it as FLAT and open over it —
        # the next open ADDS it into a closeable lot which the close then resolves. (Using flat_eps here
        # is what made an earlier build try to "close" un-closeable dust → dust-abandon → orphan freeze.)
        if abs(pos.qty) >= self.exch_min:
            self._close(response, validator, book_id, book, account, pos, vol_dp, now)
        else:
            self._open(response, validator, book_id, book, account, volume_cap, now)

    # ------------------------------------------------------------------ events
    def onOrderRejected(self, event: OrderPlacementEvent) -> None:
        if event.bookId is None or not self._active_validator:
            return
        validator = self._active_validator
        st = self._bstate(validator, event.bookId)
        st.pending_ns = 0
        st.pending_kind = ""
        self._rt_log.pop((validator, event.bookId), None)

    def onTrade(self, event: TradeEvent, validator: str | None = None) -> None:
        """Route our taker fills into position accounting (we only ever take)."""
        if event.bookId is None or self.uid != event.takerAgentId:
            return
        validator = validator or self._active_validator
        if validator is None:
            return
        ts_ns = int(event.timestamp) if event.timestamp else self._step_ts_ns
        # Normalize the raw int side (0=buy/1=sell) to OrderDirection at the boundary so the value
        # matches the `direction: OrderDirection` hints downstream (was relying on IntEnum equality).
        direction = OrderDirection.BUY if event.side == 0 else OrderDirection.SELL
        self._record_trade_volume(validator, event.bookId, event.quantity, event.price, ts_ns)
        self._apply_fill(
            validator, event.bookId, direction, event.quantity, event.price, event.takerFee, ts_ns,
        )

    # ------------------------------------------------------------------ state
    def _sync_order_size(self, volume_decimals: int) -> None:
        if volume_decimals == self._volume_decimals:
            return
        self._volume_decimals = volume_decimals
        lot = round(max(LOT, 10 ** (-volume_decimals)), volume_decimals)
        self.min_order_size = lot
        self._min_qty = lot / 2
        self.exch_min = max(EXCHANGE_MIN_ORDER_SIZE, 10 ** (-volume_decimals))
        # Half a volume tick: below this a holding is rounding noise, treat as flat.
        self._flat_eps = 0.5 * 10 ** (-volume_decimals)
        bt.logging.info(
            f"[PatientTaker uid={self.uid}] volumeDecimals={volume_decimals} "
            f"lot={lot} exch_min={self.exch_min}"
        )

    def _book_positions(self, validator: str) -> dict[int, _Position]:
        return self.positions.setdefault(validator, {})

    def _bstate(self, validator: str, book_id: int) -> _BookState:
        return self.books_state.setdefault(validator, {}).setdefault(book_id, _BookState())

    @staticmethod
    def _clear_position(pos: _Position) -> None:
        pos.qty = pos.avg = pos.entry_fee = 0.0
        pos.entry_ts = 0

    @staticmethod
    def _side_label(direction: OrderDirection) -> str:
        return "long" if direction == OrderDirection.BUY else "short"

    def _activity_force_due(self, validator: str, st: _BookState, now: int) -> bool:
        """True when a forced RT is needed to stay inside the activity window."""
        start = self._agent_start_ns.get(validator, 0)
        if start <= 0:
            return False
        if st.last_rt_ns == 0 and (now - start) < self.rt_window_ns:
            return False
        ref = st.last_rt_ns if st.last_rt_ns > 0 else start
        return (now - ref) >= self.activity_deadline_ns

    def _record_trade_volume(
        self, validator: str, book_id: int, qty: float, price: float, ts_ns: int,
    ) -> None:
        vol = float(qty) * float(price)
        if vol <= 0:
            return
        self._bstate(validator, book_id).vol_log.append((ts_ns, vol))

    def _prune_vol_log(self, st: _BookState, now_ns: int) -> None:
        cutoff = now_ns - self.volume_assessment_ns
        st.vol_log = [(t, v) for t, v in st.vol_log if t >= cutoff]

    def _rolled_quote_volume(self, validator: str, book_id: int, now_ns: int) -> float:
        st = self._bstate(validator, book_id)
        self._prune_vol_log(st, now_ns)
        return sum(v for _, v in st.vol_log)

    def _apply_fill(
        self,
        validator: str,
        book_id: int,
        direction: OrderDirection,
        qty: float,
        price: float,
        trade_fee: float,
        ts: int,
    ) -> None:
        """Update the per-book position from a fill. Opening blends the entry; a reducing
        fill realizes a clean per-RT PnL, records it for kappa, and logs the round-trip."""
        pos = self._book_positions(validator).setdefault(book_id, _Position())
        st = self._bstate(validator, book_id)
        st.pending_ns = 0   # a fill resolved the in-flight order
        st.pending_kind = ""   # clear the in-flight open tag symmetrically with the lock

        signed = qty if direction == OrderDirection.BUY else -qty
        prev = pos.qty
        entry_avg = pos.avg
        opening = abs(prev) < self._flat_eps   # sub-eps prior holding -> re-anchor cleanly

        # Same side (or opening from flat): grow the position, blend the entry price.
        if opening or (prev > 0) == (signed > 0):
            base = 0.0 if opening else abs(prev)
            total = base + qty
            pos.avg = (pos.avg * base + price * qty) / total if total > 0 else price
            pos.qty = (0.0 if opening else prev) + signed
            if opening:
                pos.entry_ts = ts
                pos.entry_fee = trade_fee
                self._ensure_rt_open_ctx(validator, book_id, direction, ts)
            else:
                pos.entry_fee += trade_fee
            return

        closed_qty = min(qty, abs(prev))
        new_qty = prev + signed
        final = abs(new_qty) < self._flat_eps
        if closed_qty >= self._min_qty and entry_avg > 0:
            rpnl = (price - entry_avg) * closed_qty if prev > 0 else (entry_avg - price) * closed_qty
            open_fee, close_fee = self._allocate_close_fees(
                pos.entry_fee, abs(prev), closed_qty, qty, trade_fee,
            )
            net_pnl = rpnl - open_fee - close_fee
            entry_ts = pos.entry_ts
            kappa_before = st.kappa3
            rt_window_n = self._rt_count(st, ts)
            st.last_rt_ns = ts
            self._record_rt_close(validator, book_id, ts, net_pnl)
            self._log_rt(
                validator=validator,
                book_id=book_id,
                ts=ts,
                hold_s=(ts - entry_ts) / _NS if entry_ts else None,
                entry_avg=entry_avg,
                exit_px=price,
                gross_pnl=rpnl,
                open_fee=open_fee,
                close_fee=close_fee,
                net_pnl=net_pnl,
                kappa_before=kappa_before,
                kappa_after=st.kappa3,
                rt_window_n=rt_window_n,
                st=st,
                final=final,
            )
            pos.entry_fee -= open_fee
        elif final:
            self._rt_log.pop((validator, book_id), None)

        pos.qty = new_qty
        if abs(pos.qty) < self._flat_eps:
            self._clear_position(pos)
        elif (prev > 0) != (pos.qty > 0):
            # FLIP: this fill over-closed the prior side, leaving a residual on the OPPOSITE side. Under
            # single-open this is the dust-absorption path (open over a sub-exch-min opposite remnant),
            # so re-anchor the new leg AND give it a fresh RT open context so its eventual close logs a
            # real open_reason/side instead of a stale one (gated: only stashes if none exists).
            pos.avg, pos.entry_ts = price, ts
            pos.entry_fee = trade_fee * (abs(pos.qty) / qty) if qty > 0 else 0.0
            self._ensure_rt_open_ctx(
                validator, book_id,
                OrderDirection.BUY if pos.qty > 0 else OrderDirection.SELL, ts,
            )

    def _prune_rt_events(self, st: _BookState, now: int) -> bool:
        cutoff = now - self.kappa_rt_history_ns
        if not st.rt_events or st.rt_events[0][0] >= cutoff:
            return False
        before = len(st.rt_events)
        st.rt_events = [(t, p) for t, p in st.rt_events if t >= cutoff]
        return len(st.rt_events) != before

    def _record_rt_close(self, validator: str, book_id: int, ts: int, net_pnl: float) -> None:
        st = self._bstate(validator, book_id)
        self._prune_rt_events(st, ts)
        st.rt_events.append((ts, net_pnl))
        self._refresh_book_kappa(validator, book_id, ts)

    # ------------------------------------------------------------------ kappa-3
    def _global_rt_timestamps(self, validator: str, now: int) -> list[int]:
        cutoff = now - self.kappa_rt_history_ns
        ts_set: set[int] = set()
        for st in self.books_state.get(validator, {}).values():
            for ts, _ in st.rt_events:
                if ts >= cutoff:
                    ts_set.add(ts)
        return sorted(ts_set)

    def _book_pnl_series(
        self, validator: str, book_id: int, now: int, extra: tuple[int, float] | None = None,
    ) -> list[float]:
        timestamps = self._global_rt_timestamps(validator, now)
        if extra is not None and extra[0] not in timestamps:
            timestamps = sorted(timestamps + [extra[0]])
        if not timestamps:
            return []

        cutoff = now - self.kappa_rt_history_ns
        # SUM same-timestamp RTs on this book (matches the validator's per-bucket += ), don't overwrite.
        by_ts: dict[int, float] = {}
        for t, p in self._bstate(validator, book_id).rt_events:
            if t >= cutoff:
                by_ts[t] = by_ts.get(t, 0.0) + p
        if extra is not None:
            by_ts[extra[0]] = extra[1]
        return [by_ts.get(ts, 0.0) for ts in timestamps]

    @staticmethod
    def _median(values: list[float]) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        mid = len(s) // 2
        return s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])

    @classmethod
    def _kappa3_raw(cls, pnl_series: list[float]) -> float | None:
        tau = KAPPA_TAU
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

    def _kappa_history_ready(self, validator: str, now: int) -> bool:
        ts = self._global_rt_timestamps(validator, now)
        return len(ts) >= 2 and ts[-1] - ts[0] >= self.kappa_min_lookback_ns

    def _refresh_book_kappa(self, validator: str, book_id: int, now: int) -> None:
        st = self._bstate(validator, book_id)
        if not self._kappa_history_ready(validator, now):
            st.kappa3 = None
            return
        st.kappa3 = self._kappa3_raw(self._book_pnl_series(validator, book_id, now))

    @staticmethod
    def _estimate_rt_pnl(taker_rate: float, book, qty: float) -> float:
        """Conservative taker RT: buy at ask, sell at bid, fees on both legs."""
        if not book.bids or not book.asks:
            return 0.0
        bid, ask = book.bids[0].price, book.asks[0].price
        if bid <= 0 or ask <= 0:
            return 0.0
        return (bid - ask) * qty - taker_rate * (ask + bid) * qty

    def _project_kappa(
        self, validator: str, book_id: int, now: int, estimated_pnl: float,
    ) -> float | None:
        close_ts = now + self.min_hold_ns
        return self._kappa3_raw(
            self._book_pnl_series(validator, book_id, now, extra=(close_ts, estimated_pnl)),
        )

    # ------------------------------------------------------------------ RT logging
    @staticmethod
    def _rt_log_enabled(validator: str) -> bool:
        return validator == MAIN_VALIDATOR

    @staticmethod
    def _fmt_pnl(value: float | None) -> str:
        return "n/a" if value is None else f"{value:+.4f}"

    @staticmethod
    def _fmt_kappa_pair(before: float | None, after: float | None) -> str:
        if before is None and after is None:
            return "n/a"
        if before is None:
            return f"n/a->{after:.4f}"
        if after is None:
            return f"{before:.4f}->n/a"
        delta = after - before
        return f"{before:.4f}->{after:.4f} ({'+' if delta >= 0 else ''}{delta:.4f})"

    def _fmt_rt_pnl_list(self, st: _BookState, now: int) -> str:
        cutoff = now - self.rt_window_ns
        pnls = [p for ts, p in st.rt_events if ts >= cutoff]
        return "[" + ", ".join(f"{p:+.4f}" for p in pnls) + "]" if pnls else "[]"

    def _stash_rt_open(
        self, validator: str, book_id: int, book, account,
        direction: OrderDirection, now: int, open_reason: str,
    ) -> None:
        if not self._rt_log_enabled(validator):
            return
        st = self._bstate(validator, book_id)
        rate = self._taker_fee_rate(account)
        est_pnl = self._estimate_rt_pnl(rate, book, self.min_order_size) if rate is not None else 0.0
        self._rt_log[(validator, book_id)] = _RtLogCtx(
            open_reason=open_reason,
            side=self._side_label(direction),
            open_rt_window_n=self._rt_count(st, now),
            open_rt_pnl_list=self._fmt_rt_pnl_list(st, now),
            est_pnl=est_pnl,
            kappa_at_open=st.kappa3,
            kappa_proj=self._project_kappa(validator, book_id, now, est_pnl),
            taker_bps=(rate * 1e4) if rate is not None else None,
        )

    def _ensure_rt_open_ctx(
        self, validator: str, book_id: int, direction: OrderDirection, ts: int,
    ) -> None:
        if not self._rt_log_enabled(validator) or (validator, book_id) in self._rt_log:
            return
        st = self._bstate(validator, book_id)
        self._rt_log[(validator, book_id)] = _RtLogCtx(
            open_reason=st.pending_kind or "?",
            side=self._side_label(direction),
            open_rt_window_n=self._rt_count(st, ts),
            open_rt_pnl_list=self._fmt_rt_pnl_list(st, ts),
            kappa_at_open=st.kappa3,
        )

    def _log_rt(
        self, *, validator: str, book_id: int, ts: int, hold_s: float | None,
        entry_avg: float, exit_px: float, gross_pnl: float, open_fee: float, close_fee: float,
        net_pnl: float, kappa_before: float | None, kappa_after: float | None,
        rt_window_n: int, st: _BookState, final: bool = True,
    ) -> None:
        key = (validator, book_id)
        if not self._rt_log_enabled(validator):
            self._rt_log.pop(key, None)
            return
        ctx = self._rt_log.pop(key, _RtLogCtx()) if final else self._rt_log.get(key, _RtLogCtx())
        hold_str = f"{hold_s:.2f}" if hold_s is not None else "n/a"
        taker_bps_str = f"{ctx.taker_bps:.2f}" if ctx.taker_bps is not None else "n/a"

        bt.logging.info(
            f"[PatientTaker uid={self.uid} RT] book={book_id} "
            f"open={ctx.open_reason}/{ctx.side} "
            f"open_rt_n={ctx.open_rt_window_n} open_rt_pnl={ctx.open_rt_pnl_list} "
            f"est_pnl={self._fmt_pnl(ctx.est_pnl)} "
            f"kappa_open={self._fmt_kappa_pair(ctx.kappa_at_open, ctx.kappa_proj)} "
            f"taker_bps={taker_bps_str} close={ctx.close_reason} hold_s={hold_str} "
            f"entry={entry_avg:.4f} exit={exit_px:.4f} "
            f"gross_pnl={gross_pnl:+.4f} open_fee={open_fee:+.4f} close_fee={close_fee:+.4f} "
            f"net_pnl={net_pnl:+.4f} "
            f"kappa_close={self._fmt_kappa_pair(kappa_before, kappa_after)} "
            f"close_rt_n={rt_window_n} close_rt_pnl={self._fmt_rt_pnl_list(st, ts)}"
        )

    # ------------------------------------------------------------------ open / close
    def _rt_count(self, st: _BookState, now: int) -> int:
        cutoff = now - self.rt_window_ns
        return sum(1 for ts, _ in st.rt_events if ts >= cutoff)

    def _open(
        self, response, validator: str, book_id: int, book, account, volume_cap: float, now: int,
    ) -> None:
        """Flat book. Awake -> scalp the rebate densely (profit engine) with an activity backstop.
        Asleep (fee>0 payer) -> do NOTHING so the book ages out to kappa=None. Sleep/wake is decided per
        pass (fee-based + hysteresis) in _update_sleep_states."""
        st = self._bstate(validator, book_id)
        if st.sleeping:
            return   # slept (fee>0 payer) book: skip entirely so it ages out of the kappa window

        # ONE-POSITION-PER-BOOK is enforced STRUCTURALLY by the exch_min flat-gate in _step_book (we
        # only reach _open when abs(pos.qty) < exch_min), NOT by a stateful flag. The old `position_open`
        # flag was REMOVED 2026-06-25 (the AR's proven rule, adopted): on a carryover sim its self-heal
        # (raw held < flat_eps) could NEVER fire, so an orphaned flag wedged the book — and the activity
        # backstop below it — DORMANT (froze uid247 ~11h, uid73 ~3h). Treating a sub-exch-min remnant as
        # flat needs no flag and cannot wedge: an un-closeable dust is simply re-traded, and the next
        # open ABSORBS it into a closeable lot (_apply_fill add-branch) that the close then resolves.

        rate = self._taker_fee_rate(account)
        direction, mid = self._book_bias(book)
        if mid is None or rate is None:
            return

        # Expected RT PnL (rebate vs spread). The OPEN decision IS gated on this (est_pnl > 0) so we
        # never open a trade the spread would eat — spread matters PER-TRADE. (Only the SLEEP SET ignores
        # spread, because that set thrashes on spread noise; a stateless per-open spread check does not.)
        est_pnl = self._estimate_rt_pnl(rate, book, self.min_order_size)
        reopen_ok = st.last_rt_ns == 0 or (now - st.last_rt_ns) >= self.min_reopen_gap_ns

        # Profit engine: open on awake books, throttled per book. Gated on a real rebate
        # (rate <= -KAPPA_MIN_REBATE_BPS) AND POSITIVE expected PnL (est_pnl > 0 → the rebate beats the
        # spread). The PATIENT reversion exit (_close) then waits for the price to revert before flattening.
        if (
            reopen_ok
            and self._rt_count(st, now) < RT_MAX
            and rate <= self.kappa_min_rebate_rate
            and est_pnl > 0.0
            and self._rolled_quote_volume(validator, book_id, now) < volume_cap
        ):
            self._try_open(response, validator, book_id, book, account, direction, now, "profit", prune_vol=True)
            return

        # Activity backstop: guarantee >=1 RT per book per window (awake books only).
        if self._activity_force_due(validator, st, now):
            tag = "taker_force" if (rate <= 0.0 and est_pnl > 0.0) else "activity"
            self._try_open(response, validator, book_id, book, account, direction, now, tag)

    def _try_open(
        self, response, validator: str, book_id: int, book, account,
        direction: OrderDirection, now: int, tag: str, *, prune_vol: bool = False,
    ) -> bool:
        """Submit one taker lot and stash RT context on success. Returns whether it opened."""
        if not self._taker_open(response, validator, book_id, account, book, direction, tag):
            return False
        st = self._bstate(validator, book_id)
        if prune_vol:
            self._prune_vol_log(st, now)
        self._stash_rt_open(validator, book_id, book, account, direction, now, tag)
        return True

    def _close(
        self, response, validator: str, book_id: int, book, account, pos: _Position,
        vol_dp: int, now: int,
    ) -> None:
        """Held position: PATIENT (uid179-style) flatten. After the min hold, exit ONLY when the price has
        REVERTED to near-entry (gross >= EXIT_REVERT_GROSS_BPS -> the rebate makes net positive); otherwise
        keep HOLDING and waiting for the reversion. The only forced early exits are a WIDE catastrophe (a
        genuine runaway) and the long MAX_HOLD_S patience window. No tight stop -> we never realise the
        small adverse move that a fast cut would; the rebate-funded reversion is the edge."""
        mid = self._mid(book)
        if mid is None or mid <= 0 or pos.avg <= 0:
            return
        hold_ns = (now - pos.entry_ts) if pos.entry_ts else 0
        if hold_ns < self.min_hold_ns:
            return

        bid = book.bids[0].price if book.bids else None
        ask = book.asks[0].price if book.asks else None
        gross_bps = self._exit_gross_bps(pos, bid, ask, mid)
        if gross_bps >= EXIT_REVERT_GROSS_BPS:        # reverted to near-entry (or a small profit) -> bank rebate
            reason = "revert"
        elif gross_bps <= -CATASTROPHE_GROSS_BPS:     # genuine runaway, no reversion -> bounded catastrophe cut
            reason = "cut"
        elif hold_ns >= self.max_hold_ns:             # patience window elapsed -> give up, take the exit
            reason = "time"
        else:
            return                                    # keep HOLDING — wait for the reversion

        if self._rt_log_enabled(validator):
            ctx = self._rt_log.get((validator, book_id))
            if ctx is not None:
                ctx.close_reason = reason
        self._close_position(response, validator, book_id, account, pos, vol_dp)

    # ------------------------------------------------------------------ market helpers
    @staticmethod
    def _allocate_close_fees(
        entry_fee: float, pos_qty: float, closed_qty: float, trade_qty: float, trade_fee: float,
    ) -> tuple[float, float]:
        if pos_qty <= 0 or closed_qty <= 0:
            return 0.0, 0.0
        open_fee = entry_fee * (closed_qty / pos_qty)
        close_fee = trade_fee if closed_qty >= trade_qty else trade_fee * (closed_qty / trade_qty)
        return open_fee, close_fee

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

    @classmethod
    def _book_bias(cls, book) -> tuple[OrderDirection, float | None]:
        """microprice vs mid -> direction; tie -> long."""
        mid = cls._mid(book)
        micro = cls._microprice(book)
        if mid is None or micro is None:
            return OrderDirection.BUY, mid
        return (OrderDirection.SELL if micro < mid else OrderDirection.BUY), mid

    @staticmethod
    def _exit_gross_bps(pos: _Position, bid: float | None, ask: float | None, mid: float) -> float:
        if pos.qty > 0:
            exit_px = bid if bid and bid > 0 else mid
            return (exit_px - pos.avg) / pos.avg * 1e4
        exit_px = ask if ask and ask > 0 else mid
        return (pos.avg - exit_px) / pos.avg * 1e4

    @staticmethod
    def _loan_settlement(account) -> LoanSettlementOption:
        quote_loan = getattr(account, "quote_loan", 0.0) or 0.0
        return LoanSettlementOption.FIFO if quote_loan > 0 else LoanSettlementOption.NONE

    def _taker_fee_rate(self, account) -> float | None:
        fees = getattr(account, "fees", None)
        rate = getattr(fees, "taker_fee_rate", None) if fees is not None else None
        try:
            return float(rate) if rate is not None else None
        except (TypeError, ValueError):
            return None

    def _taker_open(
        self, response, validator: str, book_id: int, account, book,
        direction: OrderDirection, kind: str,
    ) -> bool:
        """Cross the spread for one lot. BUY needs quote balance; SELL uses base inventory,
        falling back to a margin short. Returns whether the order was submitted."""
        qty = self.min_order_size
        if direction == OrderDirection.BUY:
            ask_px = book.asks[0].price if book.asks else 0.0
            if ask_px <= 0 or account.quote_balance.free < qty * ask_px:
                return False
            self._submit_market(response, validator, book_id, OrderDirection.BUY, qty)
        else:
            bid_px = book.bids[0].price if book.bids else 0.0
            if bid_px <= 0:
                return False
            free_base = account.base_balance.free if account.base_balance else 0.0
            if free_base >= qty:
                self._submit_market(response, validator, book_id, OrderDirection.SELL, qty)
            else:
                quote_loan = getattr(account, "quote_loan", 0.0) or 0.0
                self._submit_market(
                    response, validator, book_id, OrderDirection.SELL, qty,
                    leverage=0.0 if quote_loan > 0 else SHORT_LEVERAGE,
                    settlement=self._loan_settlement(account),
                )
        self._bstate(validator, book_id).pending_kind = kind
        return True

    def _close_position(
        self, response, validator: str, book_id: int, account, pos: _Position, vol_dp: int,
    ) -> None:
        """Flatten the whole position with one market order."""
        qty = round(abs(pos.qty), vol_dp)
        if pos.qty > 0:
            # Size from held = free + reserved (the SAME view _reconcile_position uses to keep the
            # position alive). Clamping to free alone could under-size the flatten and strand a real
            # long whenever any base is momentarily reserved -> orphan + RT-log/pending leak.
            bal = account.base_balance
            held = (bal.free + (bal.reserved or 0.0)) if bal else 0.0
            qty = round(min(qty, held), vol_dp)
            direction, settlement = OrderDirection.SELL, LoanSettlementOption.NONE
        else:
            direction, settlement = OrderDirection.BUY, self._loan_settlement(account)

        if qty < self.exch_min:
            self._clear_position(pos)
            return
        self._submit_market(response, validator, book_id, direction, qty, settlement=settlement)

    def _submit_market(
        self, response, validator: str, book_id: int, direction: OrderDirection, qty: float,
        *, leverage: float = 0.0, settlement: LoanSettlementOption = LoanSettlementOption.NONE,
    ) -> None:
        kwargs: dict[str, Any] = {
            "book_id": book_id,
            "direction": direction,
            "quantity": qty,
            "currency": OrderCurrency.BASE,
            "stp": STP.CANCEL_OLDEST,
        }
        if leverage > 0:
            kwargs["leverage"] = leverage
        if settlement != LoanSettlementOption.NONE:
            kwargs["settlement_option"] = settlement
        response.market_order(**kwargs)
        self._bstate(validator, book_id).pending_ns = self._step_ts_ns   # rule 3: now in flight

    def _reconcile_position(self, account, pos: _Position, vol_dp: int) -> None:
        # Clamp tracked size to the real held base (settlement drift); clear when ~flat.
        if pos.qty > 0 and account.base_balance is not None:
            bal = account.base_balance
            held = bal.free + (bal.reserved or 0.0)
            if held < self._flat_eps:
                self._clear_position(pos)
            else:
                pos.qty = round(min(pos.qty, held), vol_dp)


if __name__ == "__main__":
    launch(PatientTakerAgent)
