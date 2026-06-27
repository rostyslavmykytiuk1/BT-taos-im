# FINAL IMPLEMENTATION PLAN — AdaptiveRouterV2Agent → regime-adaptive trend/momentum router ("AR V2.1-M")

Target file: `/root/sn-79/agents/AdaptiveRouterV2Agent.py` (class `AdaptiveRouterV2Agent`). All line numbers below are against the live file as read this session.

---

## 1. Overview & design principle

Add a **trend GATE that sits ABOVE the existing fee-based `_route()`** plus a new `_MomentumMode`. The gate is the single decision point; everything new keys off **one** latch field, `st.trend_on`. Two independent kill layers (a module flag `TREND_ENABLED` AND momentum membership in a `self._allowed` set built in `initialize()`) make the agent **byte-for-byte the current V2.1** when off.

Core invariants preserved from the current architecture:
- Routing happens **only when flat** (line 435) — unchanged. A trend detected mid-position changes nothing until the position closes via its own mode (never-cut maker / taker scalp). The one-position-per-book rule holds.
- **PnL-backoff is checked first** (line 438) and outranks the gate — unchanged.
- The fee-based `_route()` body (lines 492–534) and `_step_book` flat-branch commit block (449–479) stay character-identical, with the gate prepended.

**Three decisions forced by the adversarial reviews (these override the original component specs):**

1. **Primary signal is vol-normalized drift-z, NOT raw drift/path.** All three reviewers independently showed drift/path (`dpr`) is volatility-dominated: on a pure random walk of N=6 its mean ≈ 0.45 and P(dpr ≥ 0.62) ≈ 0.30 — so raw `dpr ≥ 0.62` latches "trend" on ~30% of *ranging* steps, and `dpr`/`move_norm`/EMA-cross are mutually correlated (~0.68), so the "3 independent gates" collapse to ~1. The fix is a genuinely orthogonal statistic: `z = net_drift / (sigma * sqrt(N))`, where `sigma` is the per-step mid stdev. On a random walk `z ~ |N(0,1)|`, so a `|z| ≥ 2` cut gives a true ~5% per-window false rate **independent of volatility**. We keep `dpr` only as a cheap secondary confirm (path-coherence), not the primary bar.

2. **A detected trend NEVER force-idles a book.** The original component-3 "force-IDLE on non-viable trend" relocates the idle-cliff bug and pulls maker-favorable books out of the proven never-cut maker. Fix: a detected trend **suppresses the MAKER branch only** (`suppress_maker=True`) and otherwise falls through to the existing taker/idle decision verbatim. Trend → momentum (if viable) OR existing taker/idle, never a new forced idle.

3. **Momentum is small, tight, single-lot, IOC-exited.** kappa is a per-book Sortino-3 consistency ratio (cubes downside). So: hard wrong-way cut via **IOC limit with capped slip** (never a market order — gap-through would manufacture a cube bigger than maker's 15bps); **MM_PYRAMID_MAX_LOTS = 1** for the first A/B (a 3-lot stack-out through one gapped exit is ~27× the cube-units of a 1-lot cut); and a **tight, uniform win profile** (small TP / tight trail) for a smooth positive stream, not "let winners run."

---

## 2. Trend detector

### 2.1 Signal math (per book, once per step)

Computed in `_step_book` **immediately after the existing `spread_ema_bps` block (after line 420), inside `if TREND_ENABLED:`**. Uses `now = state.timestamp` (the sim clock already in `_step_book`).

Maintain a bounded deque `mid_hist` of `(ts, mid)` pairs, `maxlen = TREND_WIN = 7` (→ 6 diffs). Append every step. Compute only when `len(mid_hist) == TREND_WIN` (full window — a 3–4 point window is trivially straight).

```
mids   = [m for _, m in mid_hist]            # length 7
diffs  = [mids[i+1]-mids[i] for i in range(6)]
net    = mids[-1] - mids[0]                   # signed displacement
path   = sum(abs(d) for d in diffs)
# --- PRIMARY: vol-normalized drift-z (orthogonal to volatility) ---
sigma  = stdev(diffs)                          # per-step mid stdev (population)
z      = net / (sigma * sqrt(6)) if sigma > 0 else 0.0   # ~N(0,1) under random walk
# --- SECONDARY: path coherence (cheap dpr confirm) ---
dpr    = abs(net) / path if path > 0 else 0.0
# --- DEGENERATE-PATH guard (one-tick staircase on a stale book) ---
max_frac = max(abs(d) for d in diffs) / path if path > 0 else 1.0
nonzero  = sum(1 for d in diffs if abs(d) > 0)
# --- magnitude in bps (for the momentum viability gate, NOT for detection) ---
move_bps = abs(net) / mids[0] * 1e4
sign     = 1 if net > 0 else (-1 if net < 0 else 0)
```

Smooth `z` with a **dt-based EMA** (matching the existing `spread_ema` decay convention, NOT a fixed alpha — fixed alpha is not cadence-robust under the sim's uneven ~7.3 sim-s/real-min):
```
alpha = 1 - 0.5 ** (dt / TREND_Z_HALFLIFE_NS)   # dt = now - st.trend_z_ns
st.trend_z_ema = z if first else st.trend_z_ema + alpha*(z - st.trend_z_ema)
```
Direction = `sign` of `net` (the EMA-cross from the original design is dropped: once you have a vol-z it adds nothing and is the cadence-fragile part).

### 2.2 Confirmation / latch state machine

`raw_strong` this step requires **all** of:
- `abs(st.trend_z_ema) >= TREND_Z_ENTER` (= 2.0) — the orthogonal core bar (~5% RW false rate, then debounced below)
- `dpr >= TREND_DPR_CONFIRM` (= 0.55) — path-coherence secondary
- `max_frac <= TREND_MAX_DIFF_FRAC` (= 0.6) AND `nonzero >= TREND_WIN-3` (= 4) — reject one-tick staircase / stale book
- `sign != 0`

Latch (asymmetric hysteresis + confirm streak; consecutive windows are autocorrelated since the deque overlaps, so the streak is a *weak* debounce — that is why the primary bar is already a true 5% test):
```
if raw_strong and (trend_confirm == 0 or sign == last_confirm_dir):
    trend_confirm += 1;  last_confirm_dir = sign
else:
    trend_confirm = 0                          # any non-confirm or sign flip resets
if not trend_on:
    if trend_confirm >= TREND_CONFIRM_N:       # = 3
        trend_on = True; trend_dir = sign; trend_on_since_ns = now
else:
    weak = (abs(trend_z_ema) < TREND_Z_EXIT)   # = 1.2  (clearly below RW, wide dead-band)
    dwell_ok = (now - trend_on_since_ns) >= TREND_MIN_DWELL_NS   # 120s
    if weak and dwell_ok:
        trend_on = False; trend_confirm = 0
    elif raw_strong and sign == -trend_dir and trend_confirm >= TREND_CONFIRM_N:
        trend_dir = sign; trend_on_since_ns = now    # confirmed flip
```
Warmup: gated behind `len(mid_hist) == TREND_WIN`; `trend_on` defaults False. Earliest latch = `TREND_WIN + TREND_CONFIRM_N` ≈ 10 steps after first seeing a book → restart-safe.

### 2.3 New `_BookState` fields (add at ~line 261, beside `spread_ema_bps`)

```python
# --- trend detector (V2.1-M); inert unless TREND_ENABLED ---
mid_hist: deque = field(default_factory=lambda: deque(maxlen=TREND_WIN))
trend_z_ema: float = 0.0
trend_z_ns: int = 0
trend_dir: int = 0                 # +1/-1/0  (latched direction)
trend_move_bps: float = 0.0        # |net| over the window in bps (momentum viability gate reads this)
trend_confirm: int = 0
last_confirm_dir: int = 0
trend_on: bool = False             # THE single latch every new branch reads
trend_on_since_ns: int = 0
mm_mfe_bps: float = 0.0            # peak favorable excursion of the current momentum position
mm_suppress_until_ns: int = 0     # after PnL-backoff trips on momentum: no re-latch into momentum
```
Store `st.trend_move_bps = move_bps` each compute (read later by `_route` viability check; errs toward not-entering when stale-low).

### 2.4 `_step_book` insertion (after line 420)

```python
if TREND_ENABLED:
    self._trend_update(st, mid, now)     # appends mid_hist, updates z_ema, runs the latch
```
`_trend_update` is a new pure-bookkeeping method (no orders). With `TREND_ENABLED=False` it is never called; even if called it only writes `trend_*` fields nothing consumes.

---

## 3. Momentum mode (`_MomentumMode`)

New class, peer of `_TakerMode`, registered only when `TREND_ENABLED`. Mirrors `_TakerMode.step` control flow so FIFO/RT-logging/activity accounting is identical. **All exits are IOC-limit with capped slip (never market orders).** Clip = shared `agent.clip` (0.26) so the per-book realized-RT MAD stays consistent.

### 3.1 `step`
```
if abs(net) >= exch_min:
    if not _manage(...):           # risk/cut/trail/fade/time exit (IOC)
        _maybe_pyramid(...)        # no-op when MM_PYRAMID_MAX_LOTS == 1
    return
if throttled (now - last_close_ns < tk_reopen_gap_ns): pass
elif _open(...): return
if _activity_due(st, now) and st.trend_on and st.trend_dir != 0:
    # only force a directional RT while trend is STILL live & fresh; if faded, let it idle out
    _activity_close(..., direction = BUY if trend_dir>0 else SELL)
```
If the trend has faded, do **not** force a wrong-way activity RT (a forced cross in a takers-pay regime is a guaranteed small loss every window) — the gate will route the book out of momentum on the next flat step; a None book is free-dropped, a forced lossy RT is not.

### 3.2 `_open` — economic viability (takers PAY now, ~3bps; rebate gate is structurally closed)
```
if not _budget_ok(...): return False
if not st.trend_on or st.trend_dir == 0: return False
fee_bps = taker_fee*1e4  (fallback MM_TAKER_FEE_FALLBACK_BPS = 3.0 if account.fees missing)
half_spread_bps = max(st.spread_ema_bps, 0.5)
rt_cost_bps = 2*fee_bps + 2*half_spread_bps                 # cross+fee, both legs
exp_move_bps = st.trend_move_bps                            # realized window move (proxy for continuation)
if exp_move_bps < MM_MIN_RUN_MULT * rt_cost_bps: return False   # MM_MIN_RUN_MULT = 1.8
direction = BUY if trend_dir>0 else SELL    # GO WITH the trend
# balance-checked, never naked-short from flat; ONE market clip for ENTRY only
# (entry market order is acceptable; the kappa-safety constraint is on the LOSS exit)
reset mm_mfe_bps = 0; stash_open(mode=momentum, reason="mom", side)
```
`MM_MIN_RUN_MULT` raised 1.6→**1.8** (review: realized past move is highest right after a run = exhaustion; demand a stronger paying continuation to reduce buy-the-top entries).

### 3.3 `_manage` — wrong-way fast cut (IOC) + tight uniform win
Compute `gross_bps` from FIFO side-avg vs the exit-side touch (`best_bid` for long, `best_ask` for short), exactly like `_TakerMode._exit`.
```
held = now - ts0
# HARD wrong-way cut is EXEMPT from min-hold (a -4bps adverse move is real, not own-impact noise):
if gross_bps <= -MM_HARD_SL_BPS:                # -4.0 == RT_LOSS_CAP_BPS
    -> exit "mm_sl" via IOC limit, slip capped at RT_LOSS_CAP_BPS (see 3.4)
# everything below requires min-hold (avoid self-impact false stops):
if held < tk_min_hold_ns: return False
mm_mfe_bps = max(mm_mfe_bps, gross_bps)
if gross_bps >= MM_TP_BPS:                       # 8.0 (was 20) — tight uniform win for Sortino
    -> exit "mm_tp" (IOC limit at touch)
elif mm_mfe_bps >= MM_TRAIL_ARM_BPS and gross_bps <= mm_mfe_bps - MM_TRAIL_BPS:
    -> exit "mm_trail"                           # ARM 4.0 / TRAIL 3.0 (was 6) — smoother stream
elif not st.trend_on:                            # trend latch cleared
    -> exit "mm_fade"
elif held >= MM_MAX_HOLD_S*_NS:                  # 60s absolute backstop -> dense RT cadence
    -> exit "mm_time"
```
**MM_TP_BPS 20→8 and MM_TRAIL_BPS 6→3** (review: a lumpy many-zeros/rare-+20 stream raises per-book variance and *depresses* Sortino-3; winning momentum miners earn kappa from cadence/breadth, not magnitude). These are flagged as the primary A/B knobs (§8).

### 3.4 Exit execution — IOC limit, slip-capped (the kappa-safety core)
Every momentum exit (cut included) uses an IOC limit priced at `touch * (1 ∓ slip)`, `slip = RT_LOSS_CAP_BPS/1e4`, mirroring `_activity_close`/`_managed_exit`. **No `_submit_market` on exit.** Worst realized adverse move is bounded by the limit price, not by available liquidity — so a wrong-way momentum RT lands in the same ≤4bps tail the rest of the agent produces (`4³=64` vs maker `15³=3375`), and a gap-through simply **misses** the IOC (position rides one more step at the same-side stack) rather than realizing a catastrophic fill. Add a dry-run assertion: no momentum exit is a market order; no exit limit implies > `RT_LOSS_CAP_BPS` adverse from FIFO avg.

### 3.5 `_maybe_pyramid`
`MM_PYRAMID_MAX_LOTS = 1` ⇒ this is a **no-op** for the first A/B (returns immediately). Code path retained, gated behind `MM_PYRAMID_MAX_LOTS > 1`, add-only-when-already-ahead ≥ `MM_PYRAMID_MIN_PROFIT_BPS` and direction == trend_dir, for a later experiment.

### 3.6 Momentum loss-streak circuit-breaker (review: maker has one, momentum had none)
Reuse the existing `mk_loss_streak`/`mk_streak_cooldown_until_ns` machinery for momentum: on a losing momentum close increment the streak; ≥ `MM_LOSS_STREAK_LIMIT` (3) sets a cooldown during which the gate will not route momentum (book falls through to fee-route). Symmetric to maker's, stops re-chasing chop-disguised-as-trend faster than the slow 10-min PnL-backoff.

### 3.7 Dispatch registration (`initialize`, ~line 313)
```python
self._modes = {MODE_TAKER:_TakerMode(), MODE_MAKER:_MakerMode(), MODE_IDLE:_IdleMode()}
if TREND_ENABLED:
    self._modes[MODE_MOMENTUM] = _MomentumMode()
    self._allowed = set(ALLOWED_MODES) | {MODE_MOMENTUM}
else:
    self._allowed = set(ALLOWED_MODES)
```
**`_route` and `default_mode` and the gate MUST read `self._allowed`**, not the module constant `ALLOWED_MODES` (lines 309/513/518/532 currently read the constant — they keep working since `self._allowed ⊇ ALLOWED_MODES`; the new gate's membership test reads `self._allowed`).

---

## 4. Routing integration — `_route` control flow

The gate is prepended to `_route`; the entire current body is preserved verbatim below it. `_route` gains one computed input, `global_trend_frac` (see §5.3, computed once at top of `respond()` like `idle_count`, passed through `_step_book`).

```python
def _route(self, st, account, best_bid, best_ask, mid, *,
           fallback_maker=False, cliff=False, global_trend_frac=0.0):
    # ===================== TREND GATE (above fee routing) =====================
    suppress_maker = False
    if (TREND_ENABLED and MODE_MOMENTUM in self._allowed
            and st.trend_on and st.trend_dir != 0
            and st.mm_suppress_until_ns <= /*now*/):     # backoff re-bleed guard
        taker_fee = self._taker_fee_rate(account)
        fee_bps = (taker_fee*1e4) if taker_fee is not None else MM_TAKER_FEE_FALLBACK_BPS
        half = st.spread_ema_bps if st.spread_ema_bps > 0 else <inst half>
        rt_cost = 2*half + 2*max(fee_bps, 0.0)
        viable = st.trend_move_bps >= MM_MIN_RUN_MULT*rt_cost + MOM_EDGE_MARGIN_BPS
        if viable:
            return MODE_MOMENTUM
        # CONFIRMED TREND, NOT VIABLE: do NOT force-idle (would relocate the cliff bug and pull
        # maker-favorable books out of never-cut maker). Suppress ONLY the maker branch and fall
        # through to the EXISTING taker/idle decision.
        suppress_maker = True
    # ===================== existing fee-based routing (verbatim) ===============
    cur = st.mode
    ... (lines 492–520 unchanged) ...
    maker_ok = (... and not suppress_maker)      # <-- the ONLY change inside the original body
    ... (lines 522–534 unchanged) ...
```

Two further surgical guards required by the reviews so the gate cannot recreate the idle cliff via the fallback path:
- **A trending book is never eligible for fallback-maker / cliff promotion.** Add `and not st.trend_on` to the `fallback_maker`/`cliff` maker-edge relaxation (lines 506–508, 517). A force-suppressed trending book must not be promoted back into the adverse-selection maker trap to "rescue" the cliff — pull a *non-trending* borderline book into the active slot instead.
- `global_trend_frac` is **not** used to loosen any per-book bar in V1 (the original "0.60→0.52 loose bar" is dropped: it sits only ~0.11 above the RW baseline and is the single biggest fleet-wide false-positive amplifier). It is computed and **logged only**, as a diagnostic for the A/B (metric M5).

**Composition with dwell / emergency-flip:**
- TREND-ENTER (cur∈{taker,maker,idle} → momentum): **respect the normal 300s dwell** — do NOT add it to the emergency bypass. Real trends last hours (regime-cadence memory), so 300s latency is immaterial, and a 3-confirm latch is a weaker churn guard than the dwell. (Reverses the original component-3 proposal.)
- TREND-EXIT / collapse (cur==MOMENTUM, `trend_on` False): add `(st.mode == MODE_MOMENTUM and not st.trend_on)` to the `emergency` OR so a collapsed trend can flip back to fee-route without waiting 300s — this is the safe direction (out of a directional bet into the fee-correct mode), already debounced by the latch decay + only-when-flat.

---

## 5. Regime-preservation safeguards

### 5.1 Inertness with `TREND_ENABLED=False` (the hard requirement)
1. `self._modes` has no momentum entry; `self._allowed == ALLOWED_MODES` → dispatch (line 481) can never resolve MODE_MOMENTUM.
2. The gate's first clause is `TREND_ENABLED and …` → Python short-circuits on the constant `False`, never reads `st.trend_on`, never sets `suppress_maker` → control flows into the **byte-identical** original `_route` body (the only inserted token, `and not suppress_maker`, is appended to `maker_ok` where `suppress_maker` is provably `False`).
3. `_trend_update` is called only inside `if TREND_ENABLED:` → `trend_*` fields stay default; even if written, no consumer reads them.
4. Equivalence is provable by `git diff` confined to (a) flag-guarded blocks and (b) inert bookkeeping, **plus** the §8 dry-run that diffs `_route` output against a frozen V2.1 snapshot across a fee/spread/cur-mode grid.

### 5.2 Inertness with flag ON but regime not trending (the live ~80% case)
`st.trend_on` stays False unless `|z_ema| ≥ 2.0` AND `dpr ≥ 0.55` AND staircase-reject AND 3 same-sign confirms AND warmed. A **maker-favorable ranging oscillation** fails on the orthogonal `z`: large `path`, small `net`, so `z ≈ |N(0,1)|` → `P(|z|≥2) ≈ 5%` per window, and 3 same-sign autocorrelated confirms plus the 1.2 exit dead-band drive the steady-state latch rate well under 1% — and even a latched book only **suppresses maker and falls through to the same taker/idle decision**, never a new idle. The never-cut maker, taker scalp, PnL-backoff, dwell/hysteresis, 48-idle cliff machinery all run on today's code.

### 5.3 `global_trend_frac` computation
Compute once at the **top of `respond()`** from prior-step `st.trend_on` (exactly like `idle_count` at lines 372–380), never mid-loop (avoids the one-step intra-loop inconsistency the reviews flagged). Used for logging/metric M5 only in V1.

### 5.4 PnL-backoff re-bleed guard
When `_pnl_backoff_check` trips while `st.mode == MODE_MOMENTUM`, set `st.mm_suppress_until_ns = now + 2 * PNL_BACKOFF_COOLDOWN_S`. While set, the gate's momentum clause is skipped → the book can only return to the fee-route, never re-latch momentum, until it shows a clean window. Backoff keeps priority over the gate (it is the outer `if` at 438) — unchanged.

### 5.5 False-positive guards summary (all must pass to latch)
orthogonal vol-z `|z|≥2` (true ~5% RW) · path-coherence `dpr≥0.55` · staircase reject (`max_frac≤0.6`, `nonzero≥4`) · 3 same-sign confirms · warmup (full 7-window) · 1.2/2.0 exit/enter dead-band · 120s min-dwell. Failure in *any* → `trend_on=False` → behave as today. Every error direction is "behave as V2.1."

### 5.6 Mode-transition state hygiene (review gap)
When a book switches into momentum, reset stale maker IOC-escalation state (`mk_ioc_miss_count`, `mk_ioc_prev_net`) so a later maker re-entry doesn't inherit a corrupt miss count. Verify `st.mode == MODE_MOMENTUM` persists across the fill-reconcile latency so the closing RT is attributed to momentum in `_log_rt`/`_record_rt_close` (mode is read at fill time in `_apply_fill`; momentum only switches when flat, so a closing fill always reconciles under the mode that opened it).

---

## 6. Full new/changed parameter table

| Name | Value | Rationale |
|---|---|---|
| `TREND_ENABLED` | `False` | Global kill; default OFF = current V2.1 exactly. Flip True only in the A/B miner's source. One-line rollback. |
| `MODE_MOMENTUM` | `"momentum"` | New mode key; added to `self._modes`/`self._allowed` only when enabled. Never `default_mode`. |
| `TREND_WIN` | `7` | mid_hist maxlen → 6 diffs. Compute only when full. |
| `TREND_Z_ENTER` | `2.0` | Primary bar on vol-normalized drift-z; ~5% per-window false rate on a random walk, **independent of volatility** (the orthogonal fix for the dpr/move correlation). |
| `TREND_Z_EXIT` | `1.2` | Exit/dead-band floor; well below 2.0 and below the RW |z| mean (~0.8) region we tolerate, so a decayed book unambiguously clears. |
| `TREND_Z_HALFLIFE_NS` | `int(20*_NS)` | dt-based EMA half-life on z (sim-seconds), matching the spread_ema decay convention (cadence-robust). |
| `TREND_DPR_CONFIRM` | `0.55` | Secondary path-coherence confirm; not the primary bar. |
| `TREND_MAX_DIFF_FRAC` | `0.6` | Reject a window where one diff dominates the path (one-tick gap / stale book). |
| `TREND_CONFIRM_N` | `3` | Consecutive same-sign confirming windows to latch; sign flip resets. Ranging books rarely reach 3-in-a-row. |
| `TREND_MIN_DWELL_NS` | `int(120*_NS)` | Min latch lifetime before it may clear; composes with the 300s route dwell. |
| `MM_ENTER` is governed by `trend_on` | — | Mode entry keys off the single latch, no separate strength constant. |
| `MM_MIN_RUN_MULT` | `1.8` | Expected continuation ÷ round-trip taker cost; >1 = strictly +EV after the fee+spread we PAY, 1.8 margins out exhaustion entries. |
| `MOM_EDGE_MARGIN_BPS` | `1.0` | Extra cushion above frictions before momentum vs fall-through. |
| `MM_TAKER_FEE_FALLBACK_BPS` | `3.0` | Assumed taker fee when `account.fees` missing (current takers-pay regime). |
| `MM_HARD_SL_BPS` | `4.0` | Wrong-way cut == `RT_LOSS_CAP_BPS`. Exempt from min-hold. Caps every momentum loss to the agent's existing tiny tail. **Executed as IOC limit, never market.** |
| `MM_TP_BPS` | `8.0` | Tight uniform take (was 20) for a smooth positive stream → higher Sortino-3. **A/B knob.** |
| `MM_TRAIL_ARM_BPS` | `4.0` | Arm trail only after ≥4bps MFE → trailed exit ≥ ~breakeven-after-fee, never a manufactured small loss. |
| `MM_TRAIL_BPS` | `3.0` | Trail giveback (was 6). **A/B knob.** |
| `MM_MAX_HOLD_S` | `60.0` | Absolute hold backstop → dense RT cadence (kappa frequency). |
| `MM_MIN_HOLD_S` | reuse `tk_min_hold_ns` (1.5s) | Min-hold for TP/trail only; hard SL exempt. |
| `MM_REOPEN_GAP_S` | reuse `tk_reopen_gap_ns` (1.5s) | Throttle between close and next open. |
| `MM_PYRAMID_MAX_LOTS` | `1` | **No pyramiding in V1** (3-lot stack-out through one gapped exit ≈ 27× cube-units). Path retained behind `>1` for a later experiment. |
| `MM_PYRAMID_MIN_PROFIT_BPS` | `5.0` | (Dormant in V1) add-only-into-winners when pyramiding is later enabled. |
| `MM_LOSS_STREAK_LIMIT` | `3` | Consecutive momentum losses → cooldown (no momentum routing) — symmetric to maker's circuit-breaker. |
| `clip` | `agent.clip` (0.26) | Shared notional → consistent per-book MAD / faithful cubic downside. |

Dropped from the original specs: `TREND_ENTER_DR_LOOSE`, `TREND_GLOBAL_FRAC` loosening (global frac is log-only), the absolute-8bps move filter, and the original raw-dpr enter bar (0.58/0.60/0.62) as the *primary* signal.

---

## 7. Risks & mitigations (incl. adversarial findings)

| Risk (source) | Mitigation in this plan |
|---|---|
| **drift/path is volatility-dominated; ~30% false latch in chop; guards correlated (all 3 reviewers, critical)** | Primary signal switched to **vol-normalized drift-z** (`z=net/(σ√N)`, true ~5% RW false rate, orthogonal to vol). dpr demoted to secondary confirm. |
| **Force-IDLE relocates the 48-idle cliff & pulls maker-favorable books out of never-cut maker (rev1+rev2, critical)** | Detected trend **suppresses maker only**, falls through to existing taker/idle; trending books excluded from fallback-maker/cliff promotion. No new forced idle. |
| **Market exit gap-through → loss worse than maker 15bps, defeats the kappa-safety premise (rev2, critical)** | All momentum exits are **IOC limit, slip-capped at `RT_LOSS_CAP_BPS`**. Gap-through misses the IOC (position rides), never a catastrophic fill. Dry-run asserts no market exit. |
| **3-lot pyramid = single 3× cube-bomb on reversal (rev2, high)** | `MM_PYRAMID_MAX_LOTS=1` in V1. |
| **Let-winners-run raises variance, lowers Sortino-3 (rev2, high)** | `MM_TP_BPS` 20→8, `MM_TRAIL_BPS` 6→3 (tight uniform wins); flagged as A/B knobs; re-derive winning-miner profile first (§8). |
| **PnL-backoff ↔ momentum re-bleed loop (rev1, high)** | `mm_suppress_until_ns = now + 2×cooldown` after backoff trips on momentum + `MM_LOSS_STREAK_LIMIT=3` circuit-breaker. |
| **Field/param/kill-switch inconsistency across components (all, high)** | One canonical field set (§2.3) and param table (§6); `TREND_ENABLED` flag AND `self._allowed`; gate reads `self._allowed`. |
| **Fixed-alpha EMAs not cadence-robust (rev3, high)** | z EMA is dt-based (`1-0.5**(dt/halflife)`), matching spread_ema; EMA-cross dropped. |
| **Exhaustion entries (buy the top) (rev2, medium)** | `MM_MIN_RUN_MULT` 1.6→1.8; tight TP/trail bank quickly; mm_fade exits a dead trend. |
| **Forced activity RT in a faded trend = guaranteed small loss (rev2, medium)** | Force activity only while `trend_on` still True; else let book idle out (free-dropped). |
| **spread_ema mis-scaled at restart inflates move filter (rev1, medium)** | Detection no longer divides by spread_ema (z uses σ of diffs); spread_ema only floors the *viability* gate at 0.5bps. |
| **One-tick staircase / stale book reads dpr=1.0 (rev3, medium)** | `max_frac≤0.6` + `nonzero≥4` reject. |
| **Calibration stale: numbers from earlier MAKER-trending session; live regime now TAKER (rev3, critical-flagged)** | Re-measure z-separation on current bleeding books before flipping `TREND_ENABLED` on (§8 step 1); thresholds are constants; M5 alarms on false positives. |
| **Held-at-onset maker rides to 15bps stop (rev1, missing)** | Out of scope of the gate (flat-only routing); bounded by the existing 15bps stop; quantify frequency in the A/B but do not change never-cut maker. |

---

## 8. Validation & rollout

**Step 0 — freeze a V2.1 snapshot** `git show HEAD:agents/AdaptiveRouterV2Agent.py` → reference for the equivalence diff.

**Step 1 — re-measure the signal on the CURRENT regime (mandatory before flag-on).** From `agents/data/*.csv` and our live books, compute per-book vol-z and dpr on a rolling 7-mid window; confirm ≥2σ separation between books where AR currently bleeds (maker-cut-heavy) and a verified-ranging null. If z does not separate, do not enable. Store the measured split.

**Step 2 — deterministic dry-run** new `tests/adaptiverouter_v2_momentum_dryrun.py` (template: `tests/puremaker_v3_dryrun.py`). Assert:
- (a) **flag-off equivalence**: `_route` output byte-identical to the V2.1 snapshot across a grid of `(taker_fee, maker_fee, spread, cur_mode)`.
- (b) **no false latch on synthetic ranging mids** (random-walk + mean-reverting paths): `trend_on` stays False ≥ ~99% of steps; momentum-book count ~0.
- (c) **latch on synthetic trends** (linear + noisy-drift paths): `trend_on` latches within ~10 steps.
- (d) **kappa-safety**: no momentum exit is a market order; every exit limit implies ≤ `RT_LOSS_CAP_BPS` adverse from FIFO avg; momentum never sets `st.mode=MODE_IDLE`; idle_count never increased by momentum.

**Step 3 — single-miner A/B (staggered, no fleet exposure).** Enable `TREND_ENABLED=True` on **miner-31** (already provisioned per `.env.miner-31`); keep AR-v1 controls (e.g. uid199 benchmark 0.031, and a V2.1 flag-off control). Confirm via the startup log line that the running process actually has `TREND_ENABLED True` and `modes={…,momentum}` (`.env` ≠ running agent — baked args). Watch ≥3h.

**Live metrics (use the live kappa endpoint, NOT the stale CSV / hand proxy):**
- M1 per-book kappa-3 median vs controls
- M2 momentum cut-rate (mm_sl share) vs maker baseline
- M3 idle_count (must stay < 46)
- M4 momentum win-run profile (mean/MAD of momentum RTs)
- M5 momentum-book count vs `tools/market_regime_state.json` regime

**Go criteria** (after ≥3h, ideally spanning a regime sample): M1 ≥ control AND M2 ≤ maker baseline AND M3 < 46 AND M5 ≈ 0 in any confirmed ranging/maker window.

**Rollback triggers** (any): live kappa < control after 3h · M2 > maker cut-rate baseline · M3 ≥ 46 · M5 spikes (>~15 momentum books) while regime=MAKER/ranging. **Rollback = set `TREND_ENABLED=False` (one line) and restart that miner** → byte-identical V2.1.

**Fleet promotion** only after the single-miner A/B clears go-criteria across at least one regime transition; stagger restarts (restart costs a small transient kappa dip).

---

Key files: target `/root/sn-79/agents/AdaptiveRouterV2Agent.py`; new test `/root/sn-79/tests/adaptiverouter_v2_momentum_dryrun.py`; A/B config `/root/sn-79/.env.miner-31`; regime reference `/root/sn-79/tools/market_regime_state.json` (currently `regime=TAKER`). No code has been changed — this is the plan only.
