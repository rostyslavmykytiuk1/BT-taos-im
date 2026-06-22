# AdaptiveRouterAgent Optimization Plan

Generated 2026-06-19 · **Rewritten 2026-06-22 (v6 — rebate-churn redesign; directional dropped; confirmed by uid120 trade data + our-fleet gap diagnostic)**

## Strategic direction (v6)
The dominant edge on this subnet **right now** is **deep-rebate two-sided CHURN** — not direction, not maker. The current #1 (uid120, kappa **0.637**) is a rebate-harvesting churner. Our takers gate rebate far too low (1.0–1.5 bps vs uid120's effective ≥5 bps) and hold too long, so we capture **~4%** of its edge. The market **oscillates**, so we keep regime-aware fallbacks (fee-churn, maker) — but **rebate-churn is mode #1**.

Scoring: `kappa-3` (79%, per-book **median**, **cubes downside**) + `pnl` (21%, ~floored). Kappa rewards **consistency, frequency, and breadth/coverage** — exactly what direction-independent rebate churn delivers (tight positive net across many books).

---

## 1. Evidence

### uid120 (current #1) — verified against `120_trades.csv`
- **100% taker**, **83% of trades AND volume** on books with **≥5 bps rebate**; barely touches fee-paying books (1–3 trades each).
- **Two-sided churn, NOT directional**: top-10 most-traded books are all balanced (dominant-side 50–57%); overall 1213 BUY / 1253 SELL = **49/51**. (My earlier "directional" read was a small-sample artifact on thin books — corrected.)
- **Engine**: cross both ways → lose ~13 bps gross on price → collect ~18 bps round-trip rebate → **net ~+5 bps, 66.6% net-win** → tight, positive, direction-independent → **kappa 0.637**.
- Rebate depths it hammers: book 70 @ 15.4 bps, 55 @ 14.5, 100 @ 11.9, 118 @ 11.2.

### Our gap — confirmed on the fleet
- **AdaptiveRouter (miner-1):** **47% of its taker books are thin (<5 bps)**, down to 2 bps (median 5.6). Hold **2.0 s** (uid120 ~1 s). **Win 54%** (vs 66.6%). Gate = `TAKER_REBATE_ENTER_BPS=1.5`.
- **TakerScalper:** `KAPPA_MIN_REBATE_BPS=1.0` — enters on **1 bp** rebate. Even more dilutive.
- Thin-rebate RTs produce breakeven/losing net that — because kappa-3 cubes downside — collapses the score. **This is the 24× gap.**

### Reference library (`dashboard_data/`) — winners by regime
| File | Strategy | Regime |
|---|---|---|
| **120** (now) | deep-rebate two-sided churn | rebate-rich (current) |
| 126 | rebate-scalp taker | rebate-rich |
| 136 | **fee-paying** balanced churn (volume+breadth, MAD-kappa) | volatile downtrend, **no rebate** |
| 109 / 149 / 165 / 66 | makers | calm / wide-spread |

**The market oscillates:** rebate-rich → rebate-churn; volatile-no-rebate → fee-churn; calm → maker. Design must keep all three, with rebate-churn primary.

---

## 2. Already implemented — DONE
- ✅ 6 correctness bugs (flat-seed, position sizing, giveup/cooldown timers).
- ✅ IOC price escalation in managed exit (`MK_IOC_ESCALATE_BPS=8`, `MK_IOC_CROSS_BPS=18`, `mk_ioc_miss_count`).
- ✅ Emergency mode flip bypassing dwell on negative fee regime (`EMERGENCY_TAKER_EXIT_BPS=-1`, `EMERGENCY_MAKER_EXIT_BPS=-3`).
- ✅ Fee-regime router (taker/maker/idle with hysteresis + 180 s dwell).
- ✅ Hard idle cap `HARD_IDLE_BOOKS=45` (coded, reviewed, **pending deploy** — independent, can ship anytime).

---

## 3. Modes — current vs target

### Current (3 modes)
| Mode | Behavior | Gate |
|---|---|---|
| TAKER (rebate-scalp) | cross spread, small clips, ~2–4 s hold, somewhat directional | rebate ≥ **1.5 bps** |
| MAKER | passive two-sided quotes | maker_edge ≥ 1.5 bps |
| IDLE | nothing | neither |

### Target (4 modes) — 1 upgraded, 1 new, directional dropped
| Mode | Source | What it is |
|---|---|---|
| **REBATE-CHURN** | uid120 | **upgrade of TAKER**: gate **≥5 bps**, **continuous two-sided churn** (~1 s hold, max clip, direction-independent), concentrate frequency on deepest-rebate books |
| **FEE-CHURN** | uid136 | **NEW**: volatile regime with **no deep rebate** — balanced high-freq churn that *pays* the fee, wins via volume+breadth (MAD-kappa) |
| **MAKER** | — | kept but **parked** in taker regimes; used only when calm + no rebate |
| **IDLE** | — | truly-dead books only, bounded by the hard-idle-cap |

> **Dropped:** "directional momentum taker" (v5). None of the top takers (120/126/136) is directional. **Direction is not the edge; churn is.**

---

## 4. Routing — rebate depth PRIMARY, volatility SECONDARY
Per book, every step, in order:
1. `taker_rebate ≥ REBATE_CHURN_BPS (~5)` → **REBATE-CHURN**  ← dominant, direction-independent
2. else if `volatility high` → **FEE-CHURN**  ← 136-style; no rebate but vol pays the churn
3. else if `calm AND maker_edge ≥ thresh` → **MAKER**
4. else → **IDLE**

Primary signal is the **fee regime** (AR already computes `taker_fee`) — we mostly **raise the threshold + change execution to churn**. Volatility is only the secondary axis (fee-churn vs maker). Fast routing via existing per-step re-eval + hysteresis + a **regime-shift trigger** that bypasses the 180 s dwell when the classification changes; shorter dwell for taker↔taker sub-mode switches (no inventory straddle).

---

## 5. Key changes
- **Raise rebate gate** `TAKER_REBATE_ENTER_BPS` 1.5 → **~5.0** (`REBATE_CHURN_ENTER_BPS`); hysteresis exit ~3.5.
- **Churn execution**: continuous two-sided aggressive (both BUY+SELL), hold ~1 s (was 2–4 s), reopen gap ~0.5 s; harvest hardest where rebate is deepest.
- **Add FEE-CHURN mode** + a volatility gate (rolling stdev of mid returns over a short window).
- **Apply the same rebate-gate fix to TakerScalper** fleet: `KAPPA_MIN_REBATE_BPS` 1.0 → ~5.0.
- **Guards**: churn must stay **pure taker** (no resting quotes on the churned book → no wash / self-volume flag); respect `CAPITAL_TURNOVER_CAP` volume cap (two-sided churn is high-volume).

---

## 6. Implementation order
| # | Item | Note |
|---|---|---|
| 1 | **REBATE-CHURN**: raise gate to ~5 bps + two-sided churn execution | the #1 lever; A/B one miner vs the uid120 profile |
| 2 | **FEE-CHURN** mode + volatility gate | for no-rebate volatile regimes (uid136) |
| 3 | **Fast routing**: regime-shift trigger + dwell tuning | |
| 4 | **Deploy idle-cap** | independent; can ship anytime |
| 5 | **Coverage tightening** | after the churn modes exist |
| — | **TakerScalper rebate-gate fix** (1.0→5.0) | quick parallel win on the standalone takers |

---

## 7. Secondary / carried-forward (lower priority)
- Maker adverse-selection (quote-pull logic; why PureMakers beat AR maker) — only matters in maker regimes.
- Maker TP/SL tuning (`MK_TP_BPS` 10→5, `MK_STOP_LOSS_BPS` 10→6) — A/B; low leverage.
- Internal kappa mirror / `open=?/?` orphans — only needed if we gate entries on internal kappa.

---

## Design principle (v6)
**Regime-aware, rebate-first.** The edge is harvesting the fee *structure* (deep rebate) with high-frequency two-sided churn — **not predicting direction**. Concentrate where the rebate is deepest; fall back to fee-churn when volatile-no-rebate, maker when calm; idle only dead books. Detect the regime cheaply (rebate depth + volatility), switch fast but with hysteresis.
