# RebateScalper — Kappa Maximization Plan (taker + rebate)

Goal: maximize validator score for the **taker rebate** approach (no market-making yet),
given the validator/reward logic. Keep **activity factor = 1.0** on every book at all times.

## Scoring facts that drive the design (verified in code)

- Final score = `0.79·kappa_score + 0.21·pnl_score` (`taos/im/config/__init__.py`).
- **Per-book Kappa-3** (`taos/im/utils/kappa.py`): `kappa = mean_t(r) / cbrt(LPM3 + reg)`,
  with `r = pnl/MAD` on a **per-step grid** (every step is a timestamp, zero-filled for
  books that didn't close), `tau = 0`. **Downside is CUBED** → the largest losses dominate.
- `kappa_score` (`reward.py`): normalize per book to `(kappa+2.5)/5` clamped [0,1] (kappa 0 → 0.5),
  multiply by `combined_factor = activity(≤1.33) × pnl_factor(≈1)` with **asymmetric weighting**:
  - raw kappa > 0 (norm > 0.5): volume **multiplies** score (`combined·norm`).
  - raw kappa ≤ 0 (norm ≤ 0.5): volume **hurts** (`(2−combined)·norm`).
  Then take the **median** across books, minus a **left-tail outlier penalty** (IQR rule).
  Up to 37.5% books may be inactive (None) and excluded — but we keep all active (activity=1.0).
- Realized PnL is **strict FIFO** per book (`validator.py _match_trade_fifo`). Keep every RT a
  clean, fully-flat close (the 0.255 lot guarantees this) so FIFO PnL = clean per-RT PnL.

## Diagnosis from live data (uid 215, 2,468 real RTs)

- Win rate **89.8%**, mean net **+0.06** — entry/rebate engine is already healthy.
- **99.4% of the LPM3 (kappa damage) comes from just 6 trades** (0.24% of all RTs).
  Signature: `kappa/long`, held the full ~6s, hit by a fast downward move (e.g. book 84
  330.00→314.01, −4.85% in 6s = a fundamental jump). Agent has **no stop** → rides the full move.
- Each catastrophic loss also poisons that book's MAD for the whole 3h window → drags the
  median AND triggers the outlier penalty.

Conclusion: the bottleneck is the **unbounded loss tail**, not win rate or entry quality.
For a cubed-downside metric we must **bound loss size**, not frequency.

## Plan (priority order)

### 1. Bounded stop-loss (dominant lever) — IMPLEMENT NOW
- Exit immediately when the unrealized move crosses `STOP_LOSS_BPS` (gross, vs entry),
  bypassing min-hold so it fires the moment a jump hits.
- `STOP_LOSS_BPS = 30` — set *beyond* the rebate cushion (~6–16 bps/RT) so it never fires on
  normal rebate-covered noise, only on real adverse runs (the tail is 30–290 bps).
- Expected: LPM3 ~11.5 → ~0.1 (≈100×) → `cbrt` ~5× smaller → per-book kappa ~4–5× →
  kappa 0.29 → ~0.5+ (top-tier band) at the same win rate and activity 1.0.

### 2. Cap exposure across jumps — IMPLEMENT NOW
- Shorten `MAX_HOLD_S` 5.0 → 4.0 (less time exposed to an adverse run).
- After a stop, cooldown before re-opening that book (`STOP_COOLDOWN_S = 10`) to avoid
  re-entering into momentum continuation.
- Minimal 0.255 size already limits per-jump damage.

### 3. Concentrate volume on positive-kappa books (later)
- Heavy volume where raw kappa > 0 (activity multiplies up to ×1.33); maintenance-only volume
  on weak books (keep activity = 1.0 but don't pile in, since volume there is penalized).

### 4. Clean FIFO closes (already satisfied)
- Fully flatten each RT (0.255 lot) → no stale-basis contamination.

### 5. Uniform breadth → penalty = 0 (mostly a consequence of #1)
- Same disciplined scalp on all books so none becomes a left-tail outlier.

## Constraints / safety
- **activity = 1.0 must hold** — none of the above lets a book go inactive.
- Do not modify running miner processes during implementation; A/B on a warming-up miner first.

## Status
- [ ] #1 stop-loss
- [ ] #2 shorter max_hold + stop cooldown
- [ ] A/B vs uid 215 before wider rollout
- [ ] #3 volume concentration (later)
