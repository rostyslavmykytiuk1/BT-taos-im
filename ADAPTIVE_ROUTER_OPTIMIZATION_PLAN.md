# AdaptiveRouterAgent Optimization Plan

Generated: 2026-06-19 | Last updated: 2026-06-19 (v4 — maker data added, #3/#4 premise corrected)

Current state: 6 critical bugs fixed (flat-seed, position sizing, giveup/cooldown timers).
Agent is correct and stable. This plan covers performance improvements only.

Scoring: `score = 0.79 × kappa + 0.21 × pnl`. Kappa-3 cubes downside.

**Design principle**: AdaptiveRouter is the regime-robust agent. Keep changes
mechanism-improving and regime-neutral — do not bias it toward taker just because takers
look good this hour.

---

## Key findings from fleet RT data (miners 1–10)

- **Kappa mirror is effectively dead**: 52,669 kappa=n/a vs 402 numeric across the fleet.
  99.2% of RTs log no usable kappa. The rare numeric values are ~0.0000/−0.0001.
  Confirmed NOT restart immaturity — miners 1–10 are up 153–524 min (miner-1 up 8.7h),
  all well past the ~90 min maturity gate. The orphaned closes are the real cause.
- **24,217 orphaned open=?/? closes**: No known open context at scale. Corrupted RT
  accounting is the likely root cause of the dead kappa mirror — the mirror is computed
  from clean RT history and 24k orphans corrupt that history.
- **Taker TP rate is 1.2%**: TP=282, SL=15,537, time=6,927 across the fleet.
  68% SL, 31% time. Taker value is rebate-driven, not directional. Example:
  gross=−0.22, net=+0.012 on a rebate book — the rebate turns a losing gross into a
  small net positive. Directional tuning (TP/SL values) is low-leverage on this.
- **Maker closes: 92% passive fills, 8% forced cuts**: fill=14,161, cut=1,239.
  The passive fills are net negative: −0.0061/RT, 44% win rate, −86 QUOTE total.
  Maker bleeds via **adverse selection on passive fills** — reduce quotes get hit when
  price is moving against the position. Not a fill-rate problem; fills already dominate.
  Note: dedicated PureMakers (place 23–40) are beating AdaptiveRouter's maker mode —
  the adverse-selection gap is likely the reason.

---

## Phase 1: Do now — no prerequisites

### #2. IOC price escalation in managed exit

**File**: `agents/AdaptiveRouterAgent.py` — `_BookState`, `_managed_exit()`

**Problem**: `_managed_exit()` fires an IOC at `best_bid × (1 − 4bps)` every step on a
miss. In a fast/wide market this repeats indefinitely — the position bleeds step-by-step
with no escalation. Confirmed same bleed pattern in PureMaker and ApexTaker.

**Change**:
- Add `mk_ioc_miss_count: int = 0` to `_BookState`
- Increment each time `_managed_exit()` fires and position has not reduced since last step
- Reset on any reduction in `abs(net)`
- Escalation schedule:
  - Miss 0–1: `4bps` (current `MK_IOC_SLIPPAGE_BPS`)
  - Miss 2–3: `8bps`
  - Miss 4+: wide-limit cross at `~18bps` (NOT a market order — gap fills on
    uncapped market orders can produce catastrophic outcomes, e.g. −7bps → −770bps
    observed on ApexTaker)
- Add constants: `MK_IOC_ESCALATE_BPS = 8.0`, `MK_IOC_CROSS_BPS = 18.0`

---

### #10. Emergency mode flip (bypass dwell guard)

**File**: `agents/AdaptiveRouterAgent.py` — routing block in `_step_book()` (~line 400)

**Problem**: Mode is locked for 180s (`ROUTE_MIN_DWELL_S`). If fee regime flips sharply
mid-dwell (e.g., rebate +3bps → −2bps), we stay in taker crossing the spread for up to
3 more minutes.

**Change**:
- Inside the `if flat:` block, before the dwell guard, add override:
  - `mode=taker` AND `rebate_bps < -1.0` → flip to idle immediately
  - `mode=maker` AND `maker_edge_bps < -3.0` → flip to idle immediately
- Reset `mode_since_ns` on emergency flip
- New constants: `EMERGENCY_TAKER_EXIT_BPS = -1.0`, `EMERGENCY_MAKER_EXIT_BPS = -3.0`
- Safe: only fires when flat; only on clearly-negative regimes

---

## Phase 2: Investigate — gates everything kappa-related

### Step 0.5 (NEW): Diagnose open=?/? orphans and dead kappa mirror

**This is root-cause work, not a code change yet.**

**Why it's foundational**: 24,217 orphaned closes across miners 1–10 means the kappa
mirror's input history is heavily corrupted. The kappa mirror being 99.2% n/a is almost
certainly downstream of this. Fixing orphans likely unblocks #1, #8-quartile, and gives
a cleaner signal for #9. Immaturity is ruled out — miners are up 153–524 min.

**Investigation tasks**:
1. **Identify the main orphan source** — is it post-restart (FIFO wiped, stash gone),
   multi-level partial fills (stash consumed by first fill, rest log ?/?), or something
   else? Look at the timing distribution of orphan closes relative to process restarts.
2. **Check why kappa=n/a**: confirm in `AdaptiveRouterAgent.py` exactly where `st.kappa3`
   is computed and when it can be None. Is it None because the RT history is empty/corrupt,
   or because the kappa refresh (`_refresh_book_kappa`) hasn't been called?
3. **Assess fix options for orphans**:
   - Stash persistence across partial fills (track a stack, not a single slot)?
   - If source is restarts: minimize downtime; or find validator position state
     in the `account` object for seeding (requires API investigation).
   - **Do not seed FIFO at mark price / zero fee** — confirmed bad: injects wrong
     realized PnL that further corrupts the kappa mirror.
4. Rerun #8 kappa-quartile analysis after fix to confirm mirror is alive.

**Original #7 (FIFO seeding at mark price) is not the fix** — it was the right symptom
but the wrong approach.

---

## Phase 3: Data analysis — run non-kappa parts now

### #8. RT log analysis (partial — kappa-quartile blocked)

**Run immediately, skip the kappa-quartile breakdown until Step 0.5 is resolved.**

**Taker analysis — run now**:
1. Filter `mode=taker` closes from miners 1–10 RT logs
2. Group by exit reason: `close=tp`, `close=sl`, `close=time`
3. Compute per-group: average net PnL, win rate
4. **Key question**: is taker mode net-positive only via rebates?
   (Already suggested by TP=1.2%, gross=−0.22/net=+0.012 examples)
5. Check: are `close=time` exits net-positive? If yes, `TK_MAX_HOLD_S=4.0` is
   the binding constraint, not TP/SL values

**Maker analysis — run now (required for #3 and #4)**:
1. Filter `mode=maker` closes from miners 1–10 RT logs
2. Group by close type: `close=fill` (passive) vs `close=cut` (forced IOC)
3. Compute per-group: average net PnL, win rate, average hold_s
4. **Key question for #3**: does the fill group's net PnL improve at lower TP targets?
   (If adverse selection is the mechanism, shorter time-in-market = fewer toxic fills)
5. **Key question for #4**: what fraction of total PnL loss comes from cuts (8%) vs
   fills (92%)? This determines how much leverage SL reduction actually has.

**Blocked until Step 0.5**:
- Kappa-quartile breakdown (99.2% n/a, can't run)

**Current constants to validate**: `TK_TP_BPS=2.5`, `TK_SL_BPS=4.0`, `TK_MAX_HOLD_S=4.0`,
`MK_TP_BPS=10.0`, `MK_STOP_LOSS_BPS=10.0`

---

## Phase 4: After Phase 2+3 data

### #9. Churn lane for deep-rebate books

**File**: `agents/AdaptiveRouterAgent.py` — new branch in `_TakerMode.step()`

**Why elevated**: Given that taker value is rebate-driven (1.2% TP rate, gross losers
turned net-positive only by rebates), the rebate-harvest lane is likely the highest-value
taker improvement. Confirmed +EV on ApexTaker's `lane=churn`.

**Prerequisite check**: Measure how often `rebate_bps > 4bps` books exist in the current
fee distribution. If < 5% of books qualify, build cost exceeds benefit.

**Plan (if frequency check passes)**:
- Add `CHURN_REBATE_BPS = 4.0` constant
- In `_TakerMode.step()`: if `rebate_bps > CHURN_REBATE_BPS` → churn sub-mode:
  skip TP/SL, max hold = 1.5–2s, reopen gap = 0.5s (vs current 1.5s),
  accept any fill that covers the spread
- Stash `open_reason = "churn"` for traceability

---

### #3. Reduce `MK_TP_BPS` from 10 → 5–6bps

**File**: `agents/AdaptiveRouterAgent.py` — line ~159

**HOLD until #8 maker data is in. Premise corrected below.**

**Corrected premise**: The original plan said "closes concentrate in forced cuts." That was
wrong — fleet data shows 92% of maker closes are passive fills, only 8% are cuts. Lowering
TP to "improve passive fill rate" solves a non-problem; fills already dominate.

**Revised rationale**: Lowering TP may still help, but via a different mechanism —
**shrinking the adverse-selection window**. A reduce quote resting at 10bps above entry
sits in the book longer, giving more time for adverse price movement to hit it. Lowering
to 5bps means the quote fills sooner (at a smaller gain) and exits the adverse-selection
window faster. Whether this net-improves fill-group PnL is an empirical question — #8's
maker analysis will answer it before we touch this constant.

**Change (pending #8 data)**: `MK_TP_BPS = 10.0` → `5.0`. Floor `max(MK_TP_BPS, 2×fee + tick)`
protects low-fee books. A/B test on one miner.

---

### #4. Reduce `MK_STOP_LOSS_BPS` from 10 → 6bps

**File**: `agents/AdaptiveRouterAgent.py` — line ~166

**HOLD until #8 maker data is in.**

**Caveat from data**: Cuts are only 8% of maker closes. Even if tighter SL eliminates all
cut losses, the lever is small — 92% of the bleeding happens in passive fills (adverse
selection), which SL does not touch. The cubed-downside math (6³ vs 10³ ≈ 4× cheaper) is
still correct for the cut tail, but the total impact is limited.

**Trade-off**: Tighter stops also cut more would-be mean-reverts → fewer positive RTs.
A/B test and watch **win-rate**, not just loss magnitude. Let #8 cut-group data confirm
how much of the loss tail is actually reducible before committing.

**Change (pending #8 data)**: `MK_STOP_LOSS_BPS = 10.0` → `6.0`. Review
`MK_REENTRY_COOLDOWN_S=120s` if cut frequency increases.

---

### Deeper maker lever: adverse selection (future work)

The data points to a deeper problem than TP/SL constants: the 14,161 passive fills are
net −0.0061/RT and 44% win rate. This is adverse-selection — reduce quotes get hit
preferentially when price is moving against the position. TP/SL tuning is low-leverage
on this; the real fix is in entry edge and how fast resting quotes pull when the book moves.

This is likely why dedicated PureMakers (place 23–40) outperform AdaptiveRouter's maker
mode — they have more refined quote-pull logic. Worth a separate investigation after the
kappa mirror is alive and #8's maker data is in hand.

---

### #1. Kappa gate in taker entry (gate only)

**BLOCKED until Step 0.5 resolves the dead kappa mirror.**

When kappa mirror is alive:
- Add projected-kappa gate in `_TakerMode._open()`: only enter if opening here is
  expected to improve `st.kappa3`
- Do NOT bias direction by kappa — kappa is not a directional signal (it measures
  whether past RTs were profitable, not which way price moves next)
- Direction stays from `_bias(book, mid)` (microprice)
- Add `TK_KAPPA_ENTRY_MIN = 0.05` constant

---

## Removed from plan

| Item | Reason |
|---|---|
| #5 Kappa-aware routing | Kappa is not directional; threshold (0.3) never fires; shelved |
| #6 Adaptive hold by kappa | Same kappa-direction misread; shelved |
| #7 FIFO seeding at mark price | Corrupts kappa mirror — wrong approach; replaced by Step 0.5 investigation |

---

## Implementation order

| Priority | Item | Condition |
|---|---|---|
| **Now** | #2 IOC escalation | No prerequisites |
| **Now** | #10 Emergency mode flip | No prerequisites |
| **Now** | #8 taker + maker analysis | Run immediately; skip kappa-quartile |
| **Next** | Step 0.5 orphan diagnosis | Before any kappa item |
| **After data** | #9 churn lane | Frequency check first; likely high value |
| **After #8 maker data** | #3 MK_TP_BPS 10→5 | Premise corrected; verify with fill-group PnL first |
| **After #8 maker data** | #4 MK_STOP_LOSS 10→6 | Low leverage (8% of closes); confirm cut-tail impact |
| **Blocked** | #1 kappa gate | Only after Step 0.5 fixes kappa mirror |
| **Blocked** | #8 kappa-quartile | Only after Step 0.5 |
