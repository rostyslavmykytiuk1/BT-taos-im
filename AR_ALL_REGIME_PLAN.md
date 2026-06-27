# FINAL ALL-REGIME AdaptiveRouter — Engineering Plan (v3)

Target file: `/root/sn-79/agents/AdaptiveRouterV2Agent.py`. All line numbers verified against the live 1389-line file. This plan supersedes the two raw design components and folds in every adversarial-review fix.

## 1. Core idea (3 sentences)

The entire measured loss is the one-sided held maker lot caught by a trend and realized at the 15bps catastrophe stop (uid64: 1548 cuts = −258 net / −9.11 kappa, wiping the +8.48 kappa from 12k good fills; 128/128 books, corr(px_range,cut)=+0.44); so the single all-regime fix is to **never hold a one-sided maker lot into a directional move** — keep the maker continuously two-sided up to a tiny inventory cap in calm books, and route confirmed-directional books to a patient rebate-funded taker (uid62 archetype) or idle instead of opening a maker lot. The regime is detected per-book by **one-sidedness (drift_ratio), not raw amplitude** — the discriminator the diagnosis proved (corr(max-jump,net)≈0; one-sidedness is the signal) — so high-amplitude *mean-reverting* chop stays with the (now two-sided) maker while genuine *trends* divert to the taker. Scale-out is **demoted from a kappa lever to a fill-texture/price-walk detail** (the validator's MAD-normalized, tau=0, zero-padded kappa makes proportional fragmentation scale-invariant and even cross-book-harmful), so it is applied only where fragments fill at genuinely different prices and never on losing exits.

## 2. Per-book regime detector — ONE classifier (drift-based, amplitude as magnitude gate)

Reconciles the two conflicting proposals per all three reviews: **drift_ratio is the discriminator, amplitude is the necessary-but-not-sufficient magnitude gate.** A single source of truth wired once into `maker_ok`.

### Signal & math (event-driven path integral — fixes the 12s-cadence critical)
Measured live step cadence is **~12 sim-s/step** (median 11.99s, n≈28k), and 64% of winning maker RTs close *inside one step* (median hold 3.93s). Sampling mid once/step understates the oscillating path → biases drift_ratio high → false-DIRECTIONAL on the exact smooth winners. **Fix: accumulate the path integral from every mid seen in `onTrade` between steps, not from per-step snapshots.**

Maintain on `_BookState`:
- `mid_prev: float`, `path_sum: float` (Σ|Δmid| across every onTrade tick), `mid_win_open: float` (window-anchor mid), `mid_hi/mid_lo: float`, `char: str="unknown"`, `char_since_ns: int`, `char_window_open_ns: int`.

Per **onTrade** (fill granularity, where mid updates): `path_sum += abs(mid - mid_prev); mid_prev = mid; mid_hi=max(mid_hi,mid); mid_lo=min(mid_lo,mid)`.

Per **step** (window roll every `CHAR_WINDOW_S`): compute and reset the accumulators.
- **DRIFT_RATIO** = `abs(mid - mid_win_open) / max(path_sum, tiny)` — ~0 = wandered/reverted (two-sided, maker-capturable); ~1 = every tick same direction (trend). Scale-free, fee-agnostic, volatility-agnostic. Guard `path_sum < tiny` → treat SMOOTH (dead-flat is fine for maker).
- **RANGE_VS_SPREAD** = `(mid_hi - mid_lo)/mid * 1e4 / max(half_spread_ema_bps*2, SPREAD_FLOOR_BPS)`. **SPREAD_FLOOR_BPS=4.0** floors the denominator so tight books (live min spread 0.3-0.7bps) don't explode the ratio (fixes the tight-book false-positive critical).
- **ADVERSE_EXCURSION_BPS** = `(mid - mid_win_open)/mid * 1e4` — signed; the cut precursor magnitude.

### Classification (3 buckets, hysteresis, dwell)
- **WARMUP → UNKNOWN**: until window spans ≥`CHAR_MIN_SPAN_S` with ≥`CHAR_MIN_SAMPLES` onTrade ticks. UNKNOWN is **fail-CLOSED for maker when instantaneous half-spread or first-tick |move| is already elevated** (≥`SPREAD_FLOOR_BPS` amplitude), else permissive (fixes the restart correlated-cut low/medium finding). Seed `path_sum`/`mid_hi/lo` from the first ticks to shorten warmup.
- **DIRECTIONAL** (one-sided trend — enter strict, ALL must hold): `DRIFT_RATIO ≥ DIRT_ENTER(0.65)` AND `RANGE_VS_SPREAD ≥ RVS_ENTER(3.0)` AND `abs(ADVERSE_EXCURSION_BPS) ≥ EXC_ENTER_BPS(8.0)`. EXC raised to 8 (≈0.5×stop, > typical 13bps single-RT capture floor refined live) so a normal wide-spread oscillation never trips it.
- **CHOP** (high-amplitude mean-reverting): `RANGE_VS_SPREAD ≥ RVS_ENTER` AND `DRIFT_RATIO < DIRT_ENTER`. **Routes to the two-sided never-cut maker (§3), NOT idle and NOT taker** — the cycler captures the swings; this is the regime the amp-only veto would have wrongly killed (books 49/119: 34-57 RTs, 0 cuts).
- **SMOOTH**: everything else (low amplitude). Maker home.
- **EXIT to SMOOTH (loose)**: `DRIFT_RATIO ≤ DIRT_EXIT(0.45)` OR `RANGE_VS_SPREAD ≤ RVS_EXIT(1.8)`. The 0.65/0.45 and 3.0/1.8 gaps are the anti-flip bands.
- **DIRECTIONAL sticky** ≥`CHAR_MIN_DWELL_S(90s)` (rides the aftershock; mirrors ROUTE_MIN_DWELL).
- **Cut-rate guard**: never flag a book DIRECTIONAL whose recent realized maker cut-count is 0 (the winning books have 0 cuts by construction; belt-and-suspenders against detector false-positives).

### Hooks
- `_BookState` (after line 252): add the fields above.
- Constants (near line 140): `CHAR_WINDOW_S=120, CHAR_MIN_SPAN_S=72, CHAR_MIN_SAMPLES=6, CHAR_MIN_DWELL_S=90, DIRT_ENTER=0.65, DIRT_EXIT=0.45, RVS_ENTER=3.0, RVS_EXIT=1.8, EXC_ENTER_BPS=8.0, SPREAD_FLOOR_BPS=4.0`. Precompute `_ns` variants in `initialize()` (line ~300).
- `onTrade` path-accumulate (wherever mid is refreshed on a trade event).
- New `_update_char(self, st, mid, hspread_ema_bps, now)` called in `_step_book` right after the spread-EMA update (~line 428).

## 3. Range response — continuous two-sided never-cut maker + (texture-only) scale-out

Rewrites `_desired_quotes` (lines 1306-1334) so the maker quotes **both sides while holding**, up to a small cap, instead of going one-sided-reduce-only the instant it is filled (the structural cause of the held lot → cut).

### Behavior (long branch shown; short mirrors)
```
inv_lots = abs(net)/agent.clip
add_ok = (inv_lots < MK_SOFT_INVENTORY_LOTS                    # 1.5, below the 2.0 risk_trim hard cap
          and st.char != "directional"                        # SHARED detector, re-checked EVERY step
          and abs(net) <= abs(st.prev_net) + 1e-9             # net did NOT grow last step (no intra-step ramp)
          and not _in_streak_cooldown(st, now)
          and not _in_reentry_cooldown(st, now)               # just-cut book does not re-add
          and agent._budget_ok(...))
# REDUCE side ALWAYS rests (breakeven floor), regardless of add_ok:
rpx = _reduce_price(True, fifo_px, age, ask_inside, base_target, pdp, agent)   # unchanged never-cut walk
if rq >= exch_min and rpx>0: desired[SELL] = (rpx, rq)        # rq = full side qty (see scale-out note)
if add_ok and free_quote >= clip*bid_inside:                  # NEW: keep BUY (entry) side live while long
    desired[BUY] = (bid_inside, round(clip, vol_dp))
st.prev_net = net
```
Key asymmetry (review-mandated): **stop adding into a trend, but never stop resting the breakeven reduce.** The re-add (`add_ok`) is gated on the live `char != DIRECTIONAL` *re-evaluated every step* and on `net not having grown since last step` — so a book that turns directional *while holding* immediately stops re-adding and only bleeds out (fixes the "ratchet inventory up / bigger cube-bomb on a 1.5-lot position" medium finding and the in-mode-bypass hole). The flat branch is **unchanged**.

`MK_STOP_LOSS_BPS=15` is **untouched** — the rare backstop. With continuous cycling the lot mean-reverts before 15bps in chop, so the cut fires far less than 128×16; we keep the stop as the only defense for the irreducible fast gap (the detector cannot save an in-flight lot — see §9).

### Scale-out (texture only, NOT a kappa lever)
The maker reduce already fills **partially** as price ticks through the resting limit (FIFO partial, lines 636-639), at genuinely different prices — that is the *only* fragmentation that changes the return *distribution shape* (the sole thing MAD-normalized kappa responds to). So: **rely on organic partial-fill fragmentation of the resting reduce; do NOT add an explicit `_scaleout_qty` splitter to the maker** (it's a near-noop while clip 0.26 ≈ exch_min 0.25, and adds residual-inventory risk). The reduce quote stays full-side-qty; the book paces it.

## 4. Directional response — PatientTaker (uid62-style 8s rebate scale-out poller)

New `_PatientTakerMode` (MODE_PTAKER), the instrument for DIRECTIONAL books. **Shipped as a SELECTABLE mode** alongside the existing fast `_TakerMode` (keep both in `ALLOWED_MODES` for A/B, mirroring how AR_v0 was kept over the V1 rewrite). The fast TP2.5/SL4/4s scalp is the *wrong* archetype for trends (forced noise exits, 15× too short).

Timer-driven per book (uses the **sim `now`** passed into `_step_book`, never `time.time_ns()` — fixes the clock finding):
1. **Cadence gate**: if `now - st.pt_last_tick_ns < pt_tick_ns (8s, jittered)` → return. Set `pt_last_tick_ns` only on an action, so quiet books naturally stretch the gap.
2. **Catastrophe-only backstop (the ONLY stop)**: oldest FIFO lot underwater ≥ `PT_CATASTROPHE_BPS`, scaled DOWN as inventory grows (`PT_CATASTROPHE_BPS / max(1, inv_lots)`), IOC-close oldest lot only. Bounds max single-RT cubed loss regardless of accumulation (fixes the "5-lot×30bps cubes hard" medium finding).
3. **Holding** → TILT then act, ONE clip:
   - `bias = agent._bias(book, mid)`; `S = sign(net)`; `hold_age = now - oldest_ts`.
   - REDUCE one clip (scale-out, near-touch marketable IOC, slip `PT_REDUCE_SLIP_BPS=2`) if `hold_age ≥ PT_MIN_HOLD_S(45, jittered → median ~59s)` OR `inv_lots ≥ PT_SOFT_INV_LOTS(3)` OR **`bias == -S` (price moving against the lot → bleed WITH the move)**.
   - ACCUMULATE one clip on the bias side only if `hold_age < PT_MIN_HOLD` AND `inv_lots < PT_SOFT_INV_LOTS` AND `bias == S` (press the lean only when the move favors us). Hard cap `PT_MAX_INV_LOTS=3` (down from 5 per review — uid62 *median* peak is 0.49, not its p90 4.9).
4. **Flat** → open one clip on the bias side, subject to the entry gate.
5. **Activity backstop** (`agent._activity_due`/`_activity_close`) unchanged.

### Regime-agnostic entry gate (fixes "dead in maker-rebated regime")
Replace the rebate-only `est_bps = 2*rebate - 2*half_spread > 0` (line 1071) with a net-cost gate:
`rt_cost_bps = 2*half_spread_bps - 2*rebate_bps`.
- Taker-rebated (`rebate_bps>0`): `rt_cost` small/negative → pass freely (uid62 home).
- Taker-pays: open only if `rt_cost_bps ≤ PT_MAX_RT_COST_BPS(3.0)` AND book is tight. Costlier directional books → IDLE, not a churned cross.

**Scale-out on the LOSS path is forbidden** (fixes the high-severity "paced SL increases downside variance"): the reduce-per-tick is the *patient profit/breakeven bleed*; the catastrophe path closes the oldest lot in one IOC, never paced across worsening prices.

Hooks: `_BookState` add `pt_last_tick_ns, pt_last_reduce_ns, prev_net` (~line 261); constants block (replace nothing — add PT_* alongside TK_*); `initialize()` add `pt_tick_ns`/`pt_min_hold_ns` (sim-clock, jittered); `_modes` dict add `MODE_PTAKER`; new `_PatientTakerMode(_Mode)` class reusing `_bias/_submit_market/_submit_limit/_stash_open/_budget_ok/_taker_fee_rate/_avail`. FIFO/`_apply_fill`/`_record_rt_close` path unchanged — each scale-out IOC auto-produces a realized RT.

## 5. Shared scale-out exit primitive — SCOPED DOWN

The reviews proved scale-out is **not** an independent kappa lever (MAD-norm + tau=0 → proportional fragmentation is scale-invariant; the `2*(x/2)^3<x^3` cut-split argument is wrong because MAD rescales too; and new timestamps dilute *every other book's* kappa with zeros). Therefore:

- **DROP** `_scaleout_qty` / `SCALEOUT_FRAGS` / `SCALEOUT_CUT_FRAGS` as a kappa mechanism. **No splitting of the catastrophe cut** (it doesn't help and adds an extra step of cliff exposure).
- **KEEP** scale-out only where fragments genuinely fill at *different prices over time*: (a) the **maker resting reduce** (organic partial fills, §3 — zero new code), and (b) the **PatientTaker one-clip-per-8s-tick bleed** (§4 — intrinsic to the timer, and these books would otherwise be idle/lumpy, so the added global timestamps are net-additive not dilutive).
- **RT-budget fix** (high-severity regression on the smooth home case): since PatientTaker and any multi-fill close consume `_rt_count` toward `RT_MAX=30` (lines 756-780), **count an economic close as ONE budget RT**: tag fragment fills of the same close and exclude all-but-first from `_rt_count`; re-base `PNL_BACKOFF_MIN_RTS` on economic closes. (For the maker, organic partials already happen today, so this is mainly a PatientTaker guard.) Alternatively raise `RT_MAX` proportionally — but the tag approach is safer.

## 6. `_route` integration (preserves the smooth-range case)

`_route` runs only when FLAT (line 446); switching cancels resting orders first. Add **one** maker gate from the §2 classifier, leaving all fee-regime logic intact:

```
maker_char_ok = (st.char != "directional")            # CHOP and SMOOTH both allow maker (two-sided cycler handles chop)
maker_ok = (MODE_MAKER in ALLOWED_MODES and maker_fee_ok
            and maker_char_ok and maker_edge_bps >= maker_min_edge)
```
Fall-through, **regime-aware** (fixes the maker-rebated-idle gap):
- maker blocked + **taker-rebated** → `MODE_PTAKER` (uid62 survives trends via tilt).
- maker blocked + **maker-rebated regime** (taker not rebated, but maker rebate funds hold-for-revert) → route to the **two-sided never-cut maker** with cap+cooldown (§3), NOT idle — only the *first FLAT entry into a forming trend* is what we throttle; an already-cycling maker in a maker-rebated trend is the leaders' behavior.
- maker blocked + **both legs unrebated / `rt_cost > PT_MAX_RT_COST`** → IDLE (free-drop). Prefer IDLE over the fast scalper in the current MIXED/wide-spread (11.6bps) regime (crossing 11.6bps for a 2.5bps TP is −EV).
- **In-mode bypass** (mirrors the existing fee bypass at line 528): an already-holding maker book is never ejected mid-position by a fresh DIRECTIONAL flag (`_route` only runs flat anyway; the §3 re-add gate already stops adding into the trend).

**No `_route` emergency-bypass for amp** — the reviews proved that clause is *unreachable* (emergency block is inside `if flat:`, but the cut is on a held lot). The detector reduces cut **frequency** (future opens); the in-flight lot is defended only by `MK_STOP_LOSS_BPS` + the §3 stop-adding-and-bleed logic. If faster held-lot ejection is later wanted, it goes in `_MakerMode.step`/`_managed_exit`, not `_route`.

Smooth preservation: on a calm book `char=SMOOTH`, `maker_char_ok=True`, `maker_ok` byte-for-byte today's behavior; detector inert.

## 7. Full parameter table

| Param | Value | Rationale |
|---|---|---|
| CHAR_WINDOW_S | 120 | Path-integral window; sees a trend form, < kappa window so no smear. |
| CHAR_MIN_SPAN_S / CHAR_MIN_SAMPLES | 72 / 6 | Recalibrated to live ~12s cadence (≥6 ticks). |
| DIRT_ENTER / DIRT_EXIT | 0.65 / 0.45 | One-sidedness discriminator + anti-flip band. |
| RVS_ENTER / RVS_EXIT | 3.0 / 1.8 | Amplitude in full-spread units; magnitude gate, not sole veto. |
| EXC_ENTER_BPS | 8.0 | ≈0.5×stop, > typical single-RT capture; classify before the 15bps cut, no false trip on normal oscillation. |
| SPREAD_FLOOR_BPS | 4.0 | Floors RVS denominator so tight books don't explode → false DIRECTIONAL. |
| CHAR_MIN_DWELL_S | 90 | DIRECTIONAL stickiness through aftershock. |
| MK_SOFT_INVENTORY_LOTS | 1.5 | Re-add stops here; nests under risk_trim 2.0. |
| MK_MAX_INVENTORY_LOTS | 2.0 (unchanged) | Hard risk_trim backstop. |
| MK_STOP_LOSS_BPS | 15 (unchanged) | Rare catastrophe backstop; do not retune. |
| MK_REENTRY/STREAK params | 120/5/240 (unchanged) | Now gate the re-ADD side (was flat entries). |
| PT_TICK_S | 8.0 | uid62 measured median inter-fill 8.0s; sim-clock, jittered. |
| PT_MIN_HOLD_S | 45 (jitter→~59 median) | uid62 median hold 59s, 49% >60s. |
| PT_CATASTROPHE_BPS | 30, /inv_lots | Wide tail stop, scaled down by inventory → bounded single-RT cube. |
| PT_SOFT_INV_LOTS / PT_MAX_INV_LOTS | 3 / 3 | Near uid62 *typical*, not p90 (cube-tail safety). |
| PT_REDUCE_SLIP_BPS | 2.0 | Patient near-touch bleed; tiny per-fragment variance. |
| PT_MAX_RT_COST_BPS | 3.0 | Taker-pays gate; keeps per-RT cost thin & consistent (cube-safe). |
| RT_MAX | 30 + economic-close tagging | Fragments count as 1 economic RT (no budget starvation). |
| TARGET_CLIP | 0.26 (unchanged) | Do NOT raise for scale-out (no kappa benefit, raises cube-tail). |
| **DROPPED** | — | SCALEOUT_FRAGS / SCALEOUT_MIN_CLIP / SCALEOUT_CUT_FRAGS / AMP_MAKER_VETO / AMP_MAKER_REARM / amp emergency-bypass. |

## 8. Why this wins in EACH regime

- **Smooth-range (~80%, must-not-regress)**: `char=SMOOTH` → maker path identical to today (+128 net fill engine, 57% win); two-sided-while-holding adds entry-side re-fills that bank fresh spread instead of only walking to breakeven (uid84/60: 95-96% positive RTs at +18-19bps). Detector inert. No RT-budget regression (economic-close counting). No scale-out splitter to mislabel fast two-sided books.
- **Directional/disrupted (~20%, the entire gap)**: `char=DIRECTIONAL` blocks opening a one-sided maker lot → PatientTaker bleeds WITH the move in small rebate-funded clips (uid62 book-119 downtrend) or IDLE. Removes the 1548-cut / −258-net / −9.11-kappa loss at its source. Already-held lots stop re-adding and bleed at breakeven; only the irreducible fast gap hits the 15bps stop.
- **High-amplitude chop**: `char=CHOP` (high RVS, low drift) → stays with the **two-sided never-cut maker** which cycles the swings (the winners 49/119), NOT vetoed off into breakeven taker (the amp-only-veto regression) and NOT idled.
- **Taker-rebated**: PatientTaker gate passes freely; both legs rebated → thin-positive RTs; maker still owns the calm rebated books.
- **Maker-rebated**: directional books route to the never-cut two-sided maker (hold-for-revert, leader behavior), not idle → preserves activity_factor and the idle budget. Maker fee ceiling/cliff/free-quote logic untouched.

## 9. Risks & mitigations (from the reviews)

| Risk | Mitigation (in this plan) |
|---|---|
| Scale-out gives no kappa (MAD/tau=0) & dilutes other books with zeros | Dropped as a lever; kept only as organic-partial maker texture + PatientTaker timer (net-additive timestamps on otherwise-idle books). |
| RT_MAX/pnl-backoff starve winning smooth books | Economic-close counting (1 budget-RT per close); re-base PNL_BACKOFF on closes. |
| 12s cadence blinds drift_ratio / false-positive on fast two-sided winners | Event-driven path integral from onTrade ticks; recalibrated SPAN/SAMPLES; cut-rate=0 guard. |
| Tight/wide spread mis-calibration | SPREAD_FLOOR_BPS=4 denominator floor; EXC_ENTER=8 > single-RT capture. |
| Fast cliff fires stop before detector classifies | Accepted: detector cuts *frequency* (opens); MK_STOP_LOSS_BPS + per-lot PT_CATASTROPHE remain the fast-gap backstop. Stated, not overclaimed. |
| Two-sided maker ratchets inventory on a trend → bigger cube | add_ok requires char!=DIRECTIONAL (every step) AND net-not-grown-since-last-step; soft 1.5 / hard 2.0 caps. |
| PT 5-lot×30bps single cube loss | Cap → 3 lots; PT_CATASTROPHE scaled /inv_lots; accumulate only when bias favors. |
| Paced SL widens downside variance | No scale-out on loss path; catastrophe = single IOC. |
| Restart fail-open maker burst | Warmup fail-CLOSED to idle when instantaneous spread/amplitude elevated; seed accumulators from first ticks; stagger fleet restarts. |
| Two routing proposals conflict on chop | Reconciled to ONE drift-based 3-bucket classifier; amp-only veto deleted. |
| Sim-clock vs wall-clock for PT timer | All PT gates use sim `now` from `_step_book`; A/B not judged for ≥70 real-min (restart grace). |

## 10. Validation & rollout

**Unit/offline (before any deploy):**
1. Replay a recorded 128-book pnl stream through the real `_kappa3_raw` + `_book_pnl_series` (global zero-padding) to confirm: (a) the two-sided-maker + PatientTaker variant raises **aggregate** kappa (not single-book), (b) no cross-book zero-dilution regression. This directly tests the two critical review findings.
2. Classifier dry-run on `dashboard_data/` CSVs + live exchange log: per-book label distribution; assert winning books 49/119/1/41 classify SMOOTH/CHOP (never DIRECTIONAL), wide-range losers 3/10/123 classify DIRECTIONAL. Verify ≥6 onTrade ticks/window at live cadence.
3. Harness 20/20 on the new mode (precision sync, budget, FIFO).

**Live A/B (per the no-deploy-without-confirmation memory — get explicit owner approval first):**
- **Arms**: 1 miner = full v3 (detector + two-sided maker + PatientTaker); 1 miner = detector + two-sided maker only (fast taker for directional, isolates PatientTaker); keep current AR_v2 controls. Stagger restarts.
- **Primary metric**: live-endpoint kappa/rank (NOT the hand-rolled proxy — per memory). **Secondary**: maker cut-rate/book (target ≪16; from the agent's own RT log, the faithful per-mode source), maker-fill kappa contribution, DIRECTIONAL-book entries-avoided, PatientTaker median hold (~59s) & RT count vs RT_MAX, idle-book count (≤48 budget), activity_factor (=1.0).
- **Grace**: ignore first ~70 real-min (taker activity-clock grace) and ~13-min AR cold-start.
- **Go**: full-v3 arm kappa > AR_v2 control by a clear margin over ≥3h across a maker AND a taker regime, cut-rate dropped, no activity/idle-budget breach, smooth-book kappa not regressed.
- **Rollback**: any arm's kappa < control after grace, OR cut-rate not improved, OR activity_factor < 1.0, OR idle budget breached → revert (code+config, loads next restart). All knobs are config-gated; fast scalp stays selectable so PatientTaker can be disabled independently.

Key file: `/root/sn-79/agents/AdaptiveRouterV2Agent.py` — detector in `_step_book`/`_update_char` (~428) + `onTrade`; maker `_desired_quotes` (1306-1334); new `_PatientTakerMode`; `_route` gate (529-543); RT-count tagging in `_apply_fill`/`_rt_count` (603/756). Source archetypes: `dashboard_data/62_*.csv` (PatientTaker), `{84,60}_maker_*.csv` (two-sided maker). Kappa/budget anchors verified: `_kappa3_raw` L708, `_book_pnl_series` L691, `_rt_count`/`_budget_ok` L756/779, `_sync_precision` L790.