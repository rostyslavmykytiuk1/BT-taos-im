# Spec — Plan item #1: REBATE-CHURN mode (upgrade `_TakerMode`)

Companion to `ADAPTIVE_ROUTER_OPTIMIZATION_PLAN.md` (v6). Implementation spec only — review before any code.

## Goal
Make AdaptiveRouter's taker behave like **uid120** (current #1, kappa 0.637): a **deep-rebate, two-sided, near-flat, high-frequency churner**. Harvest the rebate on both legs; price direction is irrelevant.

### Target behavior (uid120, verified)
- Trade only **deep-rebate books (≥5 bps)**; ignore thin ones (83% of its volume on ≥5 bps).
- **Two-sided, near-flat** (49/51 BUY/SELL) — open a clip, close it fast, reopen; do **not** build a directional position.
- **~1 s hold**, continuous. Net per RT ≈ −13 bps gross + 18 bps round-trip rebate = **+5 bps**, 66.6% win, tight distribution → high kappa.

## Current behavior & the gap (`_TakerMode`, lines 1004–1131)
| Aspect | Now | uid120 | Fix |
|---|---|---|---|
| routing gate | `TAKER_REBATE_ENTER_BPS=1.5` → **47% of taker books are <5 bps** | ≥5 bps | raise gate |
| inventory | pyramids same-side up to `TK_MAX_INVENTORY_LOTS=3` (directional build) | stays ~1 lot, flat | **no pyramid** |
| hold | `TK_MIN_HOLD_S=1.5` / `TK_MAX_HOLD_S=4.0` (med 2.0 s observed) | ~1 s | shorten |
| reopen gap | `TK_REOPEN_GAP_S=1.5` | ~0.5–1 s | shorten |
| direction | `_bias(book, mid)` microprice (fine — balances over many RTs) | balanced | keep |

The thin-rebate books (gross loss the rebate can't cover) produce inconsistent net → kappa-3 cubes that downside → score collapses. That's the 24× gap.

## Changes

### A. Routing gate (`_route`, ~line 508; constants ~110)
Concentrate taker routing on deep-rebate books only.
- `TAKER_REBATE_ENTER_BPS` **1.5 → 5.0** (rename `REBATE_CHURN_ENTER_BPS`)
- `TAKER_REBATE_EXIT_BPS` **0.75 → 3.5** (hysteresis band stays ~1.5 bps)
- No change to the maker/idle branches. Books that fall out of taker route to MAKER (if edge) or IDLE.

### B. Execution → flat two-sided churn (`_TakerMode`)
- **Disable pyramiding:** `TK_MAX_INVENTORY_LOTS` **3 → 1**. `_maybe_add()` already early-returns when `≤1` (line 1102) — no code change, just the constant. Keeps us near-flat like uid120.
- **Faster turnover:** `TK_MIN_HOLD_S` **1.5 → 0.8**, `TK_MAX_HOLD_S` **4.0 → 1.2**. The `close=time` path (line 1047) becomes the primary exit — lock the rebate on a short timer regardless of price.
- **Faster reopen:** `TK_REOPEN_GAP_S` **1.5 → 0.5**.
- **Keep** `TK_TP_BPS=2.5` (early exit if price gifts it) and `TK_SL_BPS=4.0` (safety cut if price gaps against during the ~1 s hold). Rebate still cushions the stop to net-positive.
- **Keep** `_open()`'s `est_bps = 2·rebate − 2·half_spread > 0` viability check (line 1078) — it already refuses to cross when the spread eats the rebate.

### C. Throttle / cap review (may bind the churn)
- `RT_MAX = 30` profit RTs/book/window (570 s). uid120 ran ~54 RTs on its top book. **Raise `RT_MAX` for deep-rebate books** (e.g. 60, or make it rebate-scaled) so we don't cap below uid120's frequency. *Validate against the volume cap first.*
- `CAPITAL_TURNOVER_CAP = 10× wealth` (`_budget_ok`) — two-sided churn is high-volume; this is the real ceiling. If we hit it, churn self-throttles (acceptable). Measure headroom on the A/B miner before raising `RT_MAX`.

### D. Guards (already satisfied — confirm, don't add)
- **No wash / self-volume:** `_open`/`_exit` use `_submit_market` (aggressive only); the churn book has **no resting quotes**, so aggressive BUY/SELL match *other* agents, never our own. ✓
- **No naked short:** `_open` already blocks SELL-from-flat without base balance (line 1091). ✓

## Sequencing caveat (important)
Raising the gate to 5 bps **moves the 1.5–5 bps books out of taker** → they fall to MAKER or IDLE → **idle goes up** until **#2 FEE-CHURN** exists to catch the *volatile* thin-rebate books. Therefore:
- **Deploy #1 together with the hard-idle-cap** (the cap absorbs the extra idle and keeps us under the 48 cliff).
- Consider a **staged gate** (e.g. 4.0 first, then 5.0) to limit the coverage shock and watch kappa vs idle.

## A/B validation (one miner, e.g. miner-5 or miner-10)
Before fleet rollout, on one AR miner measure over ~90 min (post warm-up):
1. **Rebate concentration:** % of taker RTs on ≥5 bps books → target ↑ toward uid120's 83%.
2. **Win rate:** target ↑ from 54% toward ~66%.
3. **Hold:** median → ~1 s.
4. **kappa & placement:** must improve vs an unchanged control miner.
5. **Idle count:** stays ≤45 (idle-cap holds).
6. **Volume cap:** `_budget_ok` rejections — confirm we're not slamming the ceiling.
Compare the miner's RT log profile directly against `dashboard_data/120_trades.csv`.

## Constant change summary
| Constant | Old | New |
|---|---|---|
| `TAKER_REBATE_ENTER_BPS` (→`REBATE_CHURN_ENTER_BPS`) | 1.5 | 5.0 |
| `TAKER_REBATE_EXIT_BPS` (→`REBATE_CHURN_EXIT_BPS`) | 0.75 | 3.5 |
| `TK_MAX_INVENTORY_LOTS` | 3 | 1 |
| `TK_MIN_HOLD_S` | 1.5 | 0.8 |
| `TK_MAX_HOLD_S` | 4.0 | 1.2 |
| `TK_REOPEN_GAP_S` | 1.5 | 0.5 |
| `RT_MAX` (deep-rebate only) | 30 | ~60 (after volume-cap check) |
| `TK_TP_BPS`, `TK_SL_BPS` | 2.5 / 4.0 | unchanged |

## Risks
- **Higher volume** → may hit `CAPITAL_TURNOVER_CAP`. Mitigate: measure on A/B; `_budget_ok` self-throttles.
- **Coverage dip** until FEE-CHURN (#2). Mitigate: deploy with idle-cap; staged gate.
- **Flat churn still pays the spread** on books where rebate < spread — the `est_bps>0` check (kept) refuses those, so no change in that safety.
- All changes are **constants + one gate value** in the existing open→hold→close→reopen loop — low structural risk, easy to A/B and revert.
