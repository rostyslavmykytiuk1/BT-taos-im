# SN-79 Plan — improve the working taker toward the leaders

_Last updated: 2026-06-24. Owner decisions marked **[NEEDS YOU]**. Do nothing live without explicit go._

---

## 0. The goal & how we're scored (the target)

- Validator score = **0.79 × kappa + 0.21 × pnl**. Right now `pnl_score = 0` for **every** miner (that weight is effectively off this round), so in practice **score ≈ 0.79 × kappa**.
- **kappa is a per-book Sortino-3 ratio**, verified from `taos/im/utils/kappa.py` + `reward.py`:
  - `kappa = (mean realized PnL per round-trip) ÷ cube-root(downside)`, computed per book, then medianed across active books.
  - Per-book series is MAD-normalized → **volume and trade size don't matter**. Only the *shape* matters.
  - **Numerator = mean per-RT profit → you MUST be thin-positive on the typical trade** (mean ≤ 0 → kappa ≤ 0 → 0.50 normalized = the floor where we sit).
  - **Denominator cubes the losses → one fat loss wrecks the score.** You must be smooth.
  - Needs ≥3 non-zero round-trips per book in a 3-sim-hour window, and >~48 idle books crater the median.
- **Translation: kappa rewards a smooth, consistently-small-positive stream of round-trips across many books.** Not big wins, not volume — consistency of a small edge.

---

## 1. Where we are (live, 2026-06-24 ~11:28)

Our 5 test agents:

| miner | uid | agent | uptime | activity | kappa | place | realized |
|---|---|---|---|---|---|---|---|
| 26 | 73 | TakerScalperV1 | 12h | 1 | 0.0031 | 172 | +74 |
| **27** | 192 | **TakerScalperV3 no-sleep** | 3.75h | 1 | **0.0093** | **142** | **+202** |
| 28 | 162 | TakerScalperV3 sleep | 7.8h | 1 | −0.0016† | 214 | −69 (rising) |
| 29 | 80 | PureMakerV1 | 7h | 0 | None | 248 | −83 |
| 30 | 99 | PureMakerV2 | 3h | 0 | None | 227 | −29 |

> ⚠️ **PERMANENT NOTE — uid162 (miner-28, V3-sleep): its negative kappa/PnL is a BAD-START ARTIFACT, do not judge it on the snapshot.** It ran **wrong code initially**, then was **restarted onto correct code**. Its negative numbers are the wrong-code period still inside the validator's ~3-sim-hour kappa window; its realized PnL is **RISING** (−79 → −69) and activity recovered (0 → 1) since the fix. It needs the bad-start data to age out of the window before its kappa is fair to judge. **The user has stated this repeatedly — always carry this caveat when citing uid162.** Judge it by its trend, not its absolute number.

Reference (all = operator 38.127.44.98):
- **uid148** taker — kappa **0.120, place 0 (#1)**, ARV 228k (huge volume)
- **uid60 / uid84** makers — 0.077 / 0.072, place 1 / 2
- uid145 maker — 0.026, place 58 (mid-pack; the operator's lower-tier maker)

Field: median kappa ≈ 0.008, leaders ≈ 0.07–0.12.

**Read (compare on KAPPA, not PnL):**
1. **Our takers register kappa; our makers don't.** Both taker variants that aren't sleeping score kappa (V3-no-sleep 0.0091, V1 0.0032) and sit above the field-median floor. Both PureMakers are activity=0 / kappa=None — they don't complete enough round-trips across books to register. The maker edge needs adverse-selection defense + fill frequency we don't have.
2. **No confirmed taker winner yet.** On the current kappa snapshot V3-no-sleep (0.0091) is higher than V1 (0.0032), but BOTH are near-floor and at **very unequal run-times (3.75h+2 restarts vs 12h+0)** — not apples-to-apples (the 3h rolling-kappa window covers different periods). Needs equal stable run-time + the kappa TREND to call. Do NOT rank by realized-PnL amount.
3. **V3-sleep (uid162): do NOT judge — bad-start artifact (see permanent note above).** Its negative is wrong-code-before-restart data still in the kappa window; realized PnL is rising (−79→−69), activity recovered (0→1). Verdict on the sleep design must wait until the bad-start data ages out and it's run clean.
4. **The #1 miner is a high-volume taker (uid148, 0.120).** So the *top* is reachable via taker + breadth + rebate, and that's the engine we can actually run.

---

## 2. Diagnosis — why we floor, and why the taker is the path

- **Side is not the lever; execution is.** Field maker/taker are at parity at the top (~0.076). We score ~0 on maker, taker, and router alike — no consistent per-RT edge.
- **Our maker has TWO distinct failures (don't lump them):**
  - **AdaptiveRouter maker-mode — bags on exit:** entry is fine (captures +$1,840 spread), but losers persist to ~1,016s in a trending mid → −$1,639 gross, 86% adverse selection. NOTE: the code already tries to cut at 90s, so the >900s bags mean **the cut order isn't FILLING** (re-quote walks to breakeven, gets picked off, re-accumulates) — an exit-fill-failure, NOT a missing hold cap. A cap won't fix a cut that can't fill.
  - **PureMakerV1/V2 — dormant:** activity 0, not completing round-trips at all → a fill-FREQUENCY problem (different from the bagging above).
  - Both **parked**, non-blocking. When un-parked they need different fixes (force the exit fill + no re-accumulation; vs. raise fill frequency).
- **Our taker (TakerScalper):** thin-positive and scoring. **The gap to uid148 (0.120) is the per-RT MEAN (the numerator), NOT breadth.** Our V3-no-sleep already runs activity 1.0 (broadly active) and still floors at 0.009 → going 0.009→0.120 (~13×) can only come from raising the average per-RT profit; you can't cbrt the denominator into a 13× gain. Our takers scratch at ≈0 mean; uid148 lands a genuinely positive per-RT edge. **WHERE that edge comes from (book/side selection? entry timing? which fills it takes?) is unknown until we see uid148's tape (Step 2).**
- **The rebate is the only structurally positive term we know of.** Short-hold taker net per RT ≈ `2×rebate − spread ± drift`; the price leg is ~breakeven (operator takers win <30% on price). So rebate capture is necessary — but it's clearly **not sufficient** (we likely already get some rebate and still scratch). uid148's tape must reveal the extra edge.

---

## 3. Direction

**Improve the working short-hold rebate-taker toward uid148's profile; keep the side-adaptive engine as the end state so we never hand-restart on a regime change.**

- **Now:** make the taker better — specifically raise its **per-RT mean** (the numerator), with the lever read from uid148's tape (Step 2). It already works (positive realized PnL on the dashboard); the job is the ~13× from ~0.009 to leader-class.
- **End state:** one engine inside AdaptiveRouter that reads per-book live fees every tick and posts on the rebated side (taker where takers are rebated, maker where makers are rebated *and* we can execute it), with the short-hold discipline baked in. Deploy once → it follows regime flips on its own → **no manual market-watching, no regime restarts.** Only future restarts are code deploys.

---

## 4. Steps (do one by one)

**Step 0 — Landmine defuse. ✅ DONE 2026-06-24.**
`.env.miner-11..24` had `AGENT_NAME=TakerScalperV2Agent` (missing file; baked process also `TakerScalperAgent`, also missing) → any restart/reboot would crash-loop all 14. **Fixed:** edited all 14 `.env` → `AGENT_NAME=TakerScalperV1Agent` (exists, compiles, the +EV short-hold taker). Edit-only — verified all 14 still online, no process touched. Reversible.

**Step 1 — Confirm rebate capture. ✅ DONE 2026-06-24 (workflow w5eis6qcf).**
- **Regime:** takers rebated on **124/128 books**, median taker rebate **−6.8 bps**; makers pay (+9.1). Taking is the rebated side now.
- **Capture: PASS.** Our takers get the rebate on **99.5–100%** of fills (verified 3 ways: sign distribution, magnitude-match vs live `/metrics/books`, agent code). Fee-sleep works — **no meaningful paying-side leak**.
- **SCORED result: thin-POSITIVE and rising** (this is what pays). kappa is computed in `kappa.py` from the validator's own `realized_pnl_history` (completed trades), NOT a fill-price FIFO. Validator-scored: **uid192 (V3 no-sleep) kappa +0.0093 / realized +214 / placement 142→108 climbing**; uid73 (V1) +0.0029/+70; only uid162 (*sleep*) is negative. kappa>0 ⟹ the scored per-RT mean is positive — the original premise ("thin-positive and scoring") was correct.
- **A fill-price FIFO reads ~0/slightly-neg** (rebate +5.4bps covers ~93–98% of a −5.8bps spread-cross), but that is **NOT the scored quantity** and is likely a FIFO/side-assumption artifact. Do not conclude "net-negative" from it — the validator says positive.
- **Honest nuance:** the positive is TINY (kappa ~0.009 = near floor). The climb to uid148 (0.120) still needs a bigger per-RT edge — so the lever below is about **GROWING a positive, not flipping a negative.**
- **Implication:** the 13× gap to uid148 is **NOT a rebate problem** — rebate is maxed and available to everyone. It's a **price/selection edge**: crossing less spread (better entry timing), capturing favorable excursion, or book/regime selection — i.e. moving the **−5.8 bps price leg toward zero/positive**. That's exactly what uid148's tape must reveal. **Step 1 confirms uid148's tape is the gate for Step 3.**
- **One safe minor tweak available (stage, don't ship):** harden fee-sleep to drop a book faster when its taker rate flips ≥0 (only 4 books flipped; tiny leak — low priority).

**Step 2 — DONE 2026-06-24 (workflow wqz8p0pt3, + parallel w6bhj3y4o).** Reverse-engineered uid148 (indep taker, 154.54.100.203, kappa 0.138/#1) + operator takers 184/215 + our books. **SELECTION IS REFUTED** (uid148 trades all 127/128 books, full ~19bps spread; 0/128 of our books pass 2×rebate≥spread; selection closes ≤5-10% of the gap, floor-blocked). The lever is **EXIT DISCIPLINE.** Full detail in [[sn79-taker-edge]].

**Step 3 — Build: improve the taker. The lever is found.**
**#1 (high-confidence, BOTH analyses agree): REMOVE/WIDEN our 2bps stop-loss — stop cutting.** Our exit (SL 2.0/TP 2.5/3s) fires on ~96% of RTs → realizes a constant stream of small losses → high per-book downside → the kappa-3 cube tanks us. It also cuts the rare winners. uid148 never cuts. Removing the tight SL (→ catastrophe-only ~−15/−20bps) shrinks the downside (kappa denominator) AND keeps the positive-skew tail (numerator). Keep holds SHORT (~1-3s, uid148 median 1s), full 128-book breadth, no sleep. **This is the minimal change.**
- **#2 (RESOLVED 2026-06-24): STOP WALKING — cap order SIZE to the touch depth + marketable-limit price-cap.** Source-checked `Book.cpp`: taker fills at touch, walks WORSE on size, **no midpoint/price-improvement** → near-mid entry structurally dead; exit-passive dead (uid148 100% taker). MEASURED root cause of the cross gap: the costly walks (>50bps, ~6% of orders) are **0.6-1.2-sized MULTI-LOT orders** (the 0.3 lot fills at touch), ~50% filled at touch, and **88% would've been avoided by an order ≤0.25.** So we accumulate 2-4-lot bags and dump them in one oversized market order that walks (likely the SL force-closing). Fix: cap per-order size to ≤touch-depth (~0.25-0.3) / split large closes, + a marketable-limit price-cap (Book.cpp `minPrice/maxPrice`) as backstop, + likely stop accumulating multi-lot inventory. (The other advisor's "shrink LOT to 0.1" is the wrong target — infeasible at the 0.25 floor, and the 0.3 lots don't walk; the 0.6-1.2 oversized orders do.)
- **#3 — runner tail: WITHDRAWN.** Clean re-derivation shows uid148's edge is the **<3s bulk** (62% win, +1.6 med); its 10-30s bucket is its *worst* and the ≥30s tail is 8 noisy RTs (incl. a −58bps loser). **Keep MAX_HOLD short (~2-3s), do NOT extend.** (Concedes to the other advisor.)
- **Optional Tier-2 (the other advisor's idea, good):** per-book guard — after ~2 catastrophic-SL hits on a book, force it activity-only, to cap the fat tail the wider stop admits.
- **What NOT to do:** no tight-spread selection (refuted), no volume chase, no sleep, no 25-30s holds, don't touch the rebate (maxed).

**CONVERGED PARAMS (on V3-no-sleep, config-toggled):** **#1** `MAX_GROSS_SL_BPS 2.0 → ~12` (catastrophe-only — THE lever; also kills the SL forced-dumps that walk thin books up to 1489bps = the real cross-cost + downside). **#2** STOP WALKING: cap per-order SIZE to ≤touch-depth (~0.25-0.3, split large closes, don't dump 0.6-1.2 bags) + a **marketable-limit a few bps through the touch** backstop (Book.cpp `minPrice/maxPrice`). 88% of our >50bps walks are avoidable this way. Keep `MAX_HOLD_S ~2-3s`, 128 breadth, no-sleep, activity-backstop; likely also stop accumulating multi-lot inventory. Realistic target **~0.05-0.10**; 0.138 uncertain.

**Step 4 — A/B test it. [NEEDS YOU: how many UIDs]**
- **Control:** current **V3-no-sleep (miner-27 / uid192)** — highest-kappa of our live takers so far (near-floor, not a confirmed winner; let it stabilize for a like-for-like baseline).
- **Treatment:** **no-cut + no-walk taker** — (#1) widen 2bps SL → ~12 catastrophe-only, (#2) cap per-order size to ≤touch-depth + marketable-limit price-cap (no oversized dumps), keep short holds (1-3s) + 128 breadth + no sleep. Config-toggles on V3 (like `no_sleep`), one file. On a spare UID. (Optional separate arm: #1-only vs #1+#2 to isolate the walk-cap's contribution.)
- **Judge at 12–24h** on the live endpoint (raw `kappa` trend + placement, never the home-rolled proxy), **dual gate: per-book median realized edge ≥ +1bp AND per-book downside < control.** Interim win = uid145-band (kappa 0.03-0.06); stretch = uid148 0.138.

**Step 5 — Staggered roll on a clean pass.** One miner at a time, confirm clean rediscovery + kappa, then the next. Never all at once.

**End state — drop the winning engine into AdaptiveRouter** for autonomous per-book side-switching.

---

## 5. The hold-length question (your dashboard observation)

You see realized PnL rising stably and ask: should we hold longer even with the rebate?

- **The rebate is fixed per round-trip regardless of hold time.** Holding longer does *not* grow the rebate — it only adds price-drift exposure during the hold.
- For kappa (Sortino), added drift = added downside variance. On the operator's taker tape (uid184), downside rose ~6× from <5s to 30-60s holds while the mean *fell* — **longer holds lowered the risk-adjusted score.** Optimal was <5s.
- Our own live evidence agrees: V3-no-sleep holds short (~3s) and is our best scorer; the rising PnL **is** short-hold rebate-scalping working.
- **When longer holds *would* help:** only if we have a *directional entry edge* to capture during the hold. The operator's takers don't (28% price-win) — they're pure rebate-harvest. We haven't shown one either.
- **Verdict:** keep holds short; grow kappa via **breadth + rebate capture + consistency**, not duration. But it's cheap to settle directly — include one **longer-hold arm** in the Step-4 A/B so the data decides on *our* engine in *this* regime, not just on the operator's tape.

---

## 6. Risks & contingencies

- **State it plainly: this is a well-reasoned BET, not a sure path.** We are at kappa ~0.009; the target is ~0.120 (~13×). We do NOT yet know the lever, or whether it's even replicable — uid148 could win on something we can't copy. **uid148's tape (Step 2) is the linchpin that turns this from a bet into a plan.** Until we've seen it, every Step-3 lever is a hypothesis.
- **The lever is the per-RT mean, not breadth** (we're already activity 1.0 and still floored). Don't spend the build on "more books."
- **Rebate dependence / regime.** The taker edge needs takers to be rebated; the per-book fee sign flips intraday. The side-adaptive end state is chosen precisely so we don't have to call the regime — but until then, the taker assumes the rebate holds (Step 1 checks it).
- **Maker not abandoned, just parked — and don't carry a misdiagnosis forward.** The AR-maker bag is an **exit-fill-failure** (the 90s cut doesn't fill), so the fix is forcing the exit to fill + not re-accumulating — NOT a hold cap. PureMaker dormancy is a separate fill-frequency problem. Different fixes, both later.
- **Kappa is slow** (~90 sim-min to compute, 3h rolling) — judge at 12–24h, not early. Makers 29/30 may be partly warmup, but activity=0 after 7h (uid80) is a real low-frequency problem, not just warmup.

---

## 7. Open items / decisions

- **✅ Step 0 defuse — DONE** (edit-only, 14 files → TakerScalperV1Agent).
- **[YOU, in progress]** pulling uid84/60 **and uid148** tape. uid148 (the #1 taker) is the linchpin — Step 3 is gated on it.
- **[NEEDS YOU]** Step 4 UID budget (2 = treat+control; reuse miner-28 for treatment).
- **[ME, on go]** Step 1 rebate-capture check; then — AFTER uid148's tape — rank the Step-3 levers and draft the taker diff for your review before any deploy.

---

## Appendix — facts settled this round (so we stop relitigating)

- kappa = per-book Sortino-3; `pnl_impact=0`; score ≈ 0.79×kappa. Consistency of a thin-positive edge, size-invariant.
- No taker regime flip; "elite takers 0.18-0.22" was a phantom number. Field is MIXED, with the operator holding #1 (taker uid148) and #2/#3 (makers uid60/84).
- "Retune A" (taker-out exit to bank rebate) = net-negative (−2.6 bps/RT); the winner never takes exits. Dead.
- Our maker is gross-negative from adverse-selection bagging; PureMakerV1/V2 are dormant live. Maker parked.
- Our takers register kappa (V3-no-sleep 0.009, V1 0.003 — near-floor, no confirmed winner at unequal run-times); makers don't. Compare on kappa trend, not PnL. Sleep version dormant on kappa now (caveat: wrong-code start + warmup) — watch before verdict.
- Operator identity: uid84, uid60, uid148, uid145 are all 38.127.44.98 (uid84/60/145 makers, uid148 taker).
