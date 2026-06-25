# AdaptiveRouter V2 — Development Plan (consolidated 2026-06-25)

> **Status:** PLAN — no V2 code written, nothing deployed. Target = a NEW file
> `agents/AdaptiveRouterV2Agent.py`. The live `agents/AdaptiveRouterAgent.py` (V1) stays **untouched**
> as the A/B control. Every change ships behind an independent toggle defaulting to V1-equivalent, so
> Phase 0 deploys as a byte-equivalent no-op.
>
> This consolidates and supersedes the earlier layered drafts. It is grounded in: a 6-subsystem code
> map of the live AR, the uid199 churn diagnosis, a 13-proposal adversarial-verification pass, three
> independent design reviews that converged, and live validator data. Constants/line refs are from the
> current HEAD and may drift.

---

## 0. The one-paragraph thesis

The live AdaptiveRouter routes each of 128 books to taker/maker by the live fee+spread regime — a good
idea executed with one fatal flaw: it **flips modes on spread noise**. Because kappa scores each book
over a **3-sim-hour window ≈ ~20–24 real hours** (sim runs ~8× slower than real), and the validator
pools *all* of a book's fills into one per-book Sortino series **with zero mode-awareness**, a mode flip
keeps that book's *scored* kappa contaminated until those RTs age out over the window (and a sustained
ping-pong keeps it saturated). So churn isn't a minor inefficiency — it's the dominant
score-killer. **V2's whole job: route to the right mode automatically (so you stop restarting miners by
hand), but switch only as often as the regime actually changes (~once/day), never on noise.** Plus two
proven execution legs (TakerScalperV4, PureMakerV4) and a thin idle floor for the rare both-bad book.

---

## 1. Goal (what success looks like)

1. **No manual market-watching / restarts.** A book that's genuinely taker-favorable runs taker
   automatically; genuinely maker-favorable runs maker; neither → idle. The router replaces the
   owner's manual ~once/day taker↔maker restart.
2. **Stable, not flippy.** It adapts to a real regime shift (hours→a day) but does **not** ping-pong
   intra-day on spread wiggle.
3. **Robust routing on the right signals** — fee as the stable anchor, spread as a heavily-smoothed
   modifier ("spread changes more often than fee").

---

## 2. Scoring reality (what we're optimizing — verify live before judging)

- **Score = 0.79·kappa_score + 0.21·pnl_score.** Live: `pnl_score = 0` for all miners (the MVTRX 0.5.0
  PnL fix is not on the scoring validator yet), `gentrx_score` is dormant (1/257), and ~80% of emission
  burns. ⇒ **today, score ≈ 0.79·kappa. Judge everything on kappa, on the live endpoint**
  (`http://84.32.70.8:9001/metrics/miner`, Prometheus text keyed by `agent_id`), not the stale CSV or a
  hand-rolled proxy.
- **kappa = per-book Sortino-3:** `(mean_realized − τ)/cbrt(LPM3 downside)`, τ=0, **MAD-normalized per
  book** (so size/volume-invariant — volume is irrelevant), then **medianed over active books**. A book
  scores only with **≥3 non-zero RTs** (`KAPPA_MIN_OBS=3`) spanning **≥90 sim-min**
  (`KAPPA_MIN_LOOKBACK_S=5400`); else `kappa3=None` and it's **dropped from the median**.
- **The cube is the lever.** LPM3 cubes the downside, so a recurring small-loss tail (or two mixed
  return-shapes) blows up the denominator far more than it moves the mean. **Consistency ≫ magnitude.**
  This is why (a) churn is lethal and (b) every realized loss must be **small and bounded**.
- **Idle budget:** up to **48** zero-RT books (`kappa3=None`) are dropped **free**; past 48 they inject
  **0.0** into the median → a hard crater. `MAX_IDLE_BOOKS≈40` keeps margin.

---

## 3. The problem: routing churn (the V2 diagnosis)

From a multi-day no-restart run of the current AR (uid199 logs):
- **~443 mode transitions in one slice, 161 of them EMERGENCY-FLIP** (128 maker→taker + 33 taker→maker);
  individual books flip up to **11×**. maker→taker ≈ taker→maker → **ping-pong, not migration**.
- **40% of maker→taker emergency flips fired at <6 bps spread, 22% at <2 bps** → pure tick noise.
- **Mechanism:** routing consumes the **instantaneous** spread (via `maker_edge = half_spread −
  maker_fee` and the `spread_viable` gate), and the EMERGENCY-FLIP **bypasses the 180 s dwell**. With
  makers paying ~9 bps, any spread tick below ~12 bps trips the maker→taker emergency *instantly*.
  **Asymmetry:** maker→taker flips instantly (spread emergency); taker→maker waits the 180 s dwell (the
  fee rarely inverts) — so a transient blip buys a 3-minute taker lock-in.
- **Why it craters kappa:** the validator pools every fill of a book into one per-book series with **no
  mode awareness** (verified). So dragging a clean maker book into taker mixes a loss-cutting taker
  stream into the maker's smooth-positive stream → the cubed downside explodes.
- **⚠️ The diagnosis above is regime-specific (now stale). Live today there are TWO churn axes.** This
  161-flip slice was a *taker-favorable* regime. The **current** live regime (sim 20260621_1813) is
  **~99% maker**, and the dominant churn has moved to **maker↔idle** (~200–240 ROUTE + ~50–70 PnL-backoff
  per miner; maker↔taker emergencies now rare, 7–17). So the anti-flip layer must handle **both** axes.

> **📖 In plain English — two kinds of churn, and why the cheap one still bites.**
> - **maker↔taker flip = expensive.** It mixes two trade-styles in the 20h grade-book → the cube-poison
>   above. This dominates when *takers are favored*.
> - **maker↔idle flip = cheaper but not free.** "Idle" = *don't trade this book right now*, so it adds
>   **no trades** → it doesn't mix styles, no poison. BUT a book needs **≥3 trades in the window** just to
>   get a grade. If it keeps flipping in/out of idle it may never string 3 together → it gets **no grade
>   → dropped from your scorecard.** This dominates *now* (a maker-favored regime).
> *Example:* Book 5 is a perfectly fine maker, but its spread keeps dipping just below the bar, so it goes
> maker→idle→maker→idle all hour. It never completes 3 trades in the window → kappa=None → a *good* book
> silently vanishes from your median, for no reason. So we damp maker↔idle too (don't treat it as free).

---

## 4. The reframe that drives everything: cadence = the kappa window

**The kappa window and the regime cadence are the same timescale.**
- `KAPPA_RT_HISTORY_S = 10800 sim-s` = **3 sim-hours ≈ 20–24 real hours** (the sim advances ~0.12–0.15×
  real, i.e. **~1 sim-s ≈ 7–8 real-s**).
- The fee/spread regime flips **~once/real-day (sometimes within hours)** — i.e. **roughly one regime
  per kappa window**. (The owner historically restarted miners ~1×/day to switch.)

Two consequences dominate the design:

1. **A flip is expensive for ~a full window, not 180s.** A flip's RTs stay in the book's scored series
   until they age out over the ~20–24h window, and a *sustained* ping-pong (the observed 11×) keeps it
   saturated with two mixed return-shapes. So **churn costs ~60× what the original diagnosis assumed in
   sim-units (the 3-sim-h window vs the 180-s dwell) — hundreds× in real-time → mode STABILITY is the
   #1 lever**, ahead of every per-trade refinement.
2. **Brief sleep/idle does NOT clean a book.** Its earlier RTs persist in the ~20h window, so it stays
   scored on aging data; a true free-drop needs **~20h of no-RTs** (or a never-traded book). So idle's
   real, narrow job is *"stop adding fresh negative RTs to a no-edge book"* — a **minor** lever.

> **📖 In plain English — why one flip poisons a book for a whole day.** Kappa grades each book on how
> *steady* its per-trade results have been over the last ~20 hours, and it **cubes losses** (so a few
> big losses hurt far more than many small wins help). Crucially it looks at *all* the book's trades
> together — it can't tell a "maker trade" from a "taker trade."
> *Example:* Book 5 has been a calm maker earning ~+0.1 every trade for 20h → smooth, high grade. You
> flip it to taker for 10 minutes and it takes three −0.5 hits. Those three big losses land in the
> **same 20-hour grade-book** as the calm +0.1s, and because losses are cubed they crater the grade —
> and they keep dragging it for the **full ~20h until they age out**. So a 10-minute detour poisons the
> book's score for ~a day. *That's why we flip rarely, and only on a real regime change.*

### 4.1 ⏱ All routing constants are SIM-TIME (the implementation-critical answer)

Every routing time-constant below is **sim-time** — measured against `now = state.timestamp`. **Not
wall-clock.** Route on sim-time because the thing being protected — the validator's kappa window — is
sim-time (the validator scores over a simulation-timestamp lookback). ⚠️ Note the agent's OWN local
kappa estimate was deliberately switched to **wall-clock** (`kappa_events` stamped with `time.time_ns()`)
to fix a cold-start activation lag — **do NOT mirror that for routing;** routing uses sim-time.
- **Best implemented as fractions of `KAPPA_RT_HISTORY_S`** (10800 sim-s). The window and the
  contamination cost are both sim-time, so sim-time constants directly fraction the window — and stay
  correctly sized even as the **sim↔real rate drifts** (~0.12–0.15×, i.e. ~7–8×, with sim load).
- **The trap:** read literally as wall-clock these would be **~7–8× too fast** — e.g. a 30-*real*-min
  dwell ≈ ~3.75 sim-min ≈ today's already-too-fast 180 s (= 3 sim-min) — re-introducing the churn.
- **Only the routing decision is slowed. Execution (per-trade open/close gates) still reads the RAW
  tick.**

> **📖 In plain English — the sim runs in slow-motion.** The simulation counts time ~8× slower than your
> wall clock: **~8 real seconds pass for every 1 "sim-second."** Kappa's ~20-hour memory is counted in
> *sim* time, so our wait-timers must be too.
> *Example:* You want "wait 30 minutes before this book may switch modes." If you set 30 minutes on a
> **wall clock**, only ~3.75 *sim*-minutes pass in that time — basically no wait at all in the game's
> eyes, so the book keeps flipping. You must say "wait 30 **sim**-minutes" (≈ 4 real hours) to actually
> slow it down. That's why every timer here is sim-time, ideally written as a fraction of the 10800-sim-s
> window so it stays right even if the sim speeds up or slows down.

### 4.2 Cadence-matched constants (sim-time; starting points to tune)

| Constant | live V1 | V2 (sim-time) | ≈ real (~8×) | rationale |
|---|---|---|---|---|
| routing spread EMA half-life | n/a (raw tick) | **~5 sim-min** | ~40 min | filter sub-minute noise, track the hours-scale regime |
| per-book ROUTE dwell | 180 s | **~30 sim-min** | ~4 h | min time in a mode before any non-fee flip |
| MAKER sticky dwell | n/a | **~45–60 sim-min** | ~6–8 h | maker is home; hold through intra-hour spread wiggle |
| spread-driven flip persistence | n/a | **~15 sim-min** | ~2 h | smoothed edge must hold past the bar this long |
| fee-inversion confirm | 0 (instant) | **~3 sim-min** | ~24 min | fee is stable → may respond faster, but still confirm |
| reverse-flip cooldown | n/a | **~45 sim-min, fee-bypassable** | ~6 h | block NOISE oscillation, but a genuine fee inversion always overrides (see tension note) |
| per-book flip budget | (none) | **~3–4 / 3-sim-h window** | ~per real day | a ~daily regime needs ~daily flips, not the ~60/day/book the 180 s dwell permits |

These supersede the old fast (~4 s / 180 s / 25 s) draft values. **Tune via the census + uid199 replay
before locking;** Phase 1 judges on flip-rate, not kappa (kappa won't reflect a routing change until a
full ~20h window passes).

> **⚖️ The one genuinely-open tension (verified, don't pretend it's settled):** sim-time dwell *for
> contamination* vs *catching a real regime flip*. Contamination is sim-time (→ long sim-dwell keeps a
> book mode-pure in the kappa window). But a regime can flip in a few **real** hours, and a long
> *sim*-time dwell maps to long *real* time (×~8), so it could lock a book on the wrong side through a
> genuine flip. **Resolution: the genuine-regime signal is the FEE, and a fee inversion ALWAYS bypasses
> dwell + reverse-cooldown** (the §6.2 fee-axis emergency) — so a real regime change is never locked
> out; the sim-time dwells/cooldowns only damp *spread-noise* reverse-flips. If replay still shows
> sim-time overshooting real regimes, measure the **regime-catching** timers (fee-confirm,
> reverse-cooldown) in **wall-clock** (the codebase already does this for `kappa_events`) while keeping
> the **contamination** timers (dwell, persistence, EMA) in sim-time. Decide empirically in Phase 2.

---

## 5. The design

### 5.1 Three modes + the routing rule
**TAKER / MAKER / IDLE only.** No proactive sleep, no force-activity, no heartbeat. Per-book decision
(fee = stable anchor, smoothed spread = noisy modifier):

```
maker_viable = (smoothed_half_spread − maker_fee) ≥ MAKER_EDGE_ENTER      # spread covers fee + adverse
taker_viable = (taker_fee ≤ −REBATE_BAR) AND (rebate beats smoothed spread, est_pnl > 0)

  maker_viable   (sticky, ~45–60 sim-min dwell)   → MAKER   # higher-kappa where spread is rich
  else taker_viable                               → TAKER   # rebate-funded default; trade-all
  else (neither)                                  → IDLE    # thin floor; ~0 books in today's regime
```
Maker is the sticky home; taker is the rebate-funded default; idle is the rare both-bad fallback.

### 5.2 Taker leg = TakerScalperV4 logic (into `_TakerMode`)
- **🔧 CORRECTED 2026-06-25 — SL is a config TOGGLE, NOT hardcoded 12.** An earlier draft (and the user's
  "use SL=12") carried the *pre-A/B* no-cut optimism that our own matured A/B later reversed. The honest
  record: at the 19:27 calm snapshot, **CUT SL≈2 (uid192) beat no-cut SL=12 (uid247) ~3.7×** (0.0092 vs
  0.0025) — no-cut realized a *fatter* cubed tail (a −12bps stop that didn't revert), so the "no-cut =
  lower downside" thesis was **falsified for a rebated taker** (whose 2bps cut is ≈rebate-covered, not a
  loss-stream). **But it's regime-dependent and non-durable:** live *now* the magnitudes have collapsed
  and flipped — uid192 (cut) **−0.0023/p76**, uid247 (no-cut) **+0.0002/p70**, both ≈0 (field
  compressed). ⇒ **keep `max_gross_sl_bps` as a toggle; default to the *tighter* cut (~2bps) as the
  calm-regime/dominant-base case, flip to 12 in a confirmed fee-adverse regime.** Do NOT hard-set
  either. (Reverses the prior "remove the parameter / hardcode 12" — confirm before building.)
- TP 2.5 / hold ~3 s / open gate = `rebate AND est_pnl>0`.
- **Session-local single-open flag → no pyramiding.** Also removes a real risk: AR V1's taker stacks to
  3 lots, which at a *wide* SL could realize ~36 bps if the rebate evaporates mid-hold; one lot/book
  bounds it. (With the default tight cut this risk is small anyway.)
- **No internal sleep** (AR owns routing).
- **⚠️ Load-bearing fix:** `RT_LOSS_CAP_BPS` is shared and hardcoded inside `_activity_close` with no
  mode awareness; **split it per-mode and thread `mode` into `_activity_close`** (3 callers + base def,
  §5.6e), or the forced-close path silently re-imposes a wrong-mode cap.
- *Unifying rule: bounded cut on BOTH legs — but the taker's bound is regime-set (~2bps calm / ~12bps
  adverse), the maker's is the vol-band 10–14bps.*

> **📖 In plain English — why a *tight* stop is right for a rebated taker (the opposite of intuition).**
> On a rebated book the rebate roughly **covers the cost of crossing the spread plus a *small* adverse
> move**, so a normal round-trip nets ≈ **break-even** (our reverse-engineered rebated takers run
> ~−0.3 to +1.3 bps/RT net — close to zero). A **tight 2bps stop** keeps the adverse move small, so each
> stopped-out trade stays near that break-even — a smooth stream, good grade. A **wide 12bps "no-cut"
> stop** lets a bad trade drift far; when it *doesn't* bounce back, the loss **outgrows what the rebate
> covers** → a real **several-bps net loss** on that RT — and kappa **cubes** big losses, so a few of
> those crater the grade.
> *Example (our matured A/B):* tight cut **uid192 = 0.0092/p47 (rising)** *beat* no-cut **uid247 =
> 0.0025/p88 (falling)** — the wide stop's occasional big-loss tail, cubed, sank it. (This **flips** in a
> *fee-adverse* regime where the rebate shrinks and a wider stop can win — hence a **toggle**, not a
> hard default.)

### 5.3 Maker leg = PureMakerV4 logic (into `_MakerMode`)
The real delta over AR's current maker is **5 things** (an earlier note said 6 — AR *already*
fee-floors TP at 2×fee via `MK_TP_FEE_MULT=2.0`, L1231-1232, identical to PMV4, so that's not a delta;
only the inert TP base 10-vs-8 differs, moot under the fee floor):
1. **Reprice cushion** (`REPRICE_KEEP_TICKS=1.5`) — quotes rest & get hit instead of churning.
2. **Reduce-walk to breakeven**, not the touch — a late passive fill nets ~0 instead of giving away spread.
3. **Vol-scaled stop band (10–14 bps)** vs fixed 10 — tighter on calm books, bounded on volatile.
4. **150 s giveup** (vs 90 s).
5. **Smaller inventory cap (1.5 vs 2.0 lots).**

All tight-cut, all V1-derived and **unproven inside AR**. So **A/B = AR-with-current-maker vs
AR-with-PMV4-maker** (same router/taker/sim; toggle only the maker internals) — don't assume the
refinements win; confirm. **Never run PMV4 standalone** (its standalone lineage V1 ≈ 0.001; it bleeds
the fee without the gate — the gate is what makes it earn).

> Why tight-cut both legs: PMV4's own A/B found **tight-cut beats never-cut for the maker** (V1 tight
> +0.0012 > V2 never-cut −0.0002) — same reason as the taker: kappa-3 cubes the tail, so a rare wide
> stop dominates LPM3. The system-wide rule is **bounded-small loss everywhere.**

### 5.4 Sleep / idle — DEMOTE, do not delete
- **Remove/neuter (named against AR's *actual* code):** the `_activity_close` **force-activity calls on
  FLAT books** in `_TakerMode`/`_MakerMode` (don't force-trade a dead book); and simplify the
  `fallback_maker` idle-budget promotion (L342) down to the thin floor + `MAX_IDLE_BOOKS` cap. **NB: AR
  V1 has only taker/maker/idle modes** — there is NO heartbeat/dormant or fee-sleep machinery to remove
  (those live only in the standalone TakerScalper, not in the router).
- **Keep:** a **thin idle floor** — a book idles only when *not* taker-rebated-enough **AND** smoothed
  maker_edge < 0 (both-axes-negative) — plus the reactive PnL-backoff ("this book is bleeding → rest
  it"). Keep the **`MAX_IDLE_BOOKS≈40` cap from day one** (cheap, load-bearing: a severe regime could
  push many books both-bad; never idle >48 or the median craters — trade the least-bad instead).
- **Why keep it (the rebate-vs-fee asymmetry):** taker RTs are *rebate-funded*, so trading every viable
  book adds dense, slightly-positive kappa → **breadth wins, never sleep taker books**; maker RTs *pay
  the fee*, so force-trading a dead book manufactures a negative-kappa book → idle the both-bad ones.
- **Right-sized:** under the ~20h window idle can't fast-clean a book, so it's a **minor** lever (just
  "stop adding fresh losses"). It is **not a transition tool** — on a genuine regime shift you *switch*
  to the now-correct mode and accept transient mixing until the old RTs age out; you don't idle through
  the transition. In today's regime (takers rebated on 124/128) the floor fires on **~0 books**, so the
  census confirms "no sleep in practice" empirically.

### 5.5 Architecture — build on V1, toggle-gated (not a fresh compose)
Keep the **proven router skeleton** (`_route`, `_step_book`, FIFO `_match_trade_fifo`,
`_ensure_simulation` per-sim reset, `onTrade` maker+taker routing) **and the proven maker as the A/B
baseline**. **Embed** the V4/PMV4 leg logic into the mode classes over AR's shared FIFO/kappa — do
**not** import the standalone agents wholesale. Every change behind an independent toggle defaulting to
V1-equivalent, so each is A/B-isolable and **Phase 0 ships as a byte-equivalent no-op clone**.

### 5.6 Implementation spec (the buildable detail — added per review)
AR V1 reads **no config** (every param is a module constant; `getattr(self.config, …)` = 0 hits), so the
toggle plumbing and three mechanisms below must be built; the design above assumes them.

- **(a) Toggle registry.** Read each via `getattr(self.config, name, <V1_default>)` (the V4/PMV4
  pattern; baked into AGENT_PARAMS at launch). All default V1-equivalent → Phase 0 is a no-op. Toggles
  by phase: `ROUTE_SPREAD_EMA` (P1); `ROUTE_CADENCE` = dwells/persistence/cooldown/flip-budget (P2);
  `EMERGENCY_FEE_AXIS_ONLY` + `MAKER_STICKY` + `FASTSWITCH_TO_IDLE` + `FLIP_BUDGET` (P3); `LEG_TAKER_V4`
  + `LEG_MAKER_PMV4` (P4); `IDLE_FLOOR_THIN` + `CENSUS` (P5).
- **(b) FIFO unification — the legs are RE-IMPLEMENTED, not imported.** AR's two-deque `_Inv(longs,
  shorts)` + `_match_fifo` stays canonical; do **NOT** import V4's `_Position`/`_apply_fill` or PMV4
  internals (incompatible inventory models). Map: V4 single-open flag → a net-flat check on
  `_net_qty`; V4 entry avg/age → `_side_avg(inv.longs)` / `inv.longs[0][0]`; the SL (the regime toggle
  per §5.2, default ~2bps) checked on `gross_bps` vs `_side_avg`. PMV4 maker: re-express breakeven-walk + vol-band stop against AR's existing
  `_reduce_price`/`_managed_exit` (same shape); add `noise_bps`/`last_mid` to `_BookState` + a per-step
  mid-noise update in `_step_book` (AR tracks no `last_mid` today).
- **(c) Routing FSM (deterministic).** New `_BookState` fields: `route_candidate`, `candidate_since_ns`,
  `last_flip_ns`, `last_flip_dir`, `flip_count`, `flip_window_start_ns`, `ema_val`, `ema_last_ns`.
  Evaluation order per step (**commit only when FLAT**): fee-emergency-confirm → reverse-flip-cooldown
  latch → flip-budget check → mode-specific dwell (MAKER sticky > base) → spread-persistence timer (for
  spread-driven flips) → commit. Timers ARM while the candidate condition holds, reset when it clears.
  *Routing/execution boundary:* `REBATE_BAR` = the existing `TAKER_REBATE_ENTER` (keep ENTER/EXIT
  hysteresis); `est_pnl>0` stays in `_TakerMode._open` (execution, raw tick) — **not** a routing input.
  - **⚠️ UPDATE vs COMMIT (verified gap, L388):** AR's whole routing block sits inside `if flat:`
    (L388-438), and `mode_since_ns` is only written when flat — so a *continuously-HELD* book would never
    advance a new timer placed there. **Advance the EMA + all candidate/dwell/persistence/budget timers
    EVERY step** (above the flat-gate ~L383, or inside `mode.step()` which always runs); gate only the
    **COMMIT** on flat. (Existing maker timers are safe — they live in `_MakerMode.step`, run uncond.)
  - **⚠️ COLD-START SEEDING (verified gap):** every new ns field defaults to 0, and `now − 0 ≥ X` is
    always true → a bare cooldown/budget check fires-immediately / no-ops the first window after every
    restart. **Seed new fields to `now` on first sight** (AR already backdates `mode_since_ns` to
    `now−dwell−1` at L374-379 and guards `last_rt_ns>0`) **or guard every read with `> 0`.** (They
    auto-reset per sim via `_ensure_simulation` — that part is fine.)

  > **📖 In plain English — two subtle timer bugs to avoid.**
  > **(1) Tick the clock even while holding.** The router only *decides whether to switch* when a book is
  > flat (no open position). If we also *count* our wait-timers only when flat, a book that holds a
  > position for 40 min never advances its clocks — then the instant it flattens, a timer that should read
  > "40 min elapsed" still reads "0," so it acts on stale info. **Fix: advance the clocks every step;
  > only *commit* a switch when flat.**
  > **(2) Don't start a timer at 0.** Our "enough time passed?" check is `now − timer ≥ wait`. `now` is a
  > giant number (nanoseconds since 1970), so `now − 0` is *always* bigger than any wait → the check is
  > "yes!" the instant after a restart, making every cooldown a no-op exactly when we want the book
  > stable. **Fix: when we first see a book, set its timer to `now`, not 0.**

- **(d) Time-decayed EMA (not fixed alpha).** AR fires once per published state at a *variable* sim-Δt,
  so a fixed per-step alpha (PMV4's `NOISE_EWMA_ALPHA`) yields a cadence-dependent half-life. Use
  `alpha = 1 − exp(−(now − ema_last_ns)/tau_ns)`, `tau = halflife/ln2` (halflife ~5 sim-min); seed
  `ema_val` = instantaneous on first sample; warm-up gate = suppress non-fee flips until ≥1 half-life of
  sim-span has accumulated.
- **(e) `_activity_close` per-mode fix (1 base def + 3 callers — verified count).** Thread `mode` into
  `_activity_close` (base def L877) + its 3 callers: `_IdleMode` L939 (residual drain → use the opening
  mode's cap), `_TakerMode` L966, `_MakerMode` L1099. Split `RT_LOSS_CAP_BPS` → `TK_RT_LOSS_CAP_BPS` (=
  the taker's regime SL, §5.2) / `MK_RT_LOSS_CAP_BPS` (keep). This SLIP ceiling is distinct from the
  taker `_exit` SL **trigger** and the maker `MK_IOC_SLIPPAGE_BPS` (=4) — set all consistently.
- **(f) Census denominator (verified gap, L362).** `_step_book` early-returns on empty book (`not
  book.bids or not book.asks`, L362) and bad mid (L367) *before* mode classification, and `respond()`
  skips `book is None` (L347) — so a taker+maker+idle census won't sum to 128 on steps with empty books.
  **Add a 4th `unseen` bucket** = 128 − (books classified this step), so `taker+maker+idle+unseen = 128`.

---

## 6. Anti-flip mechanisms (the concrete changes, layered)

1. **Smooth the spread before routing** — feed `maker_edge`, `spread_viable`, and the emergency check a
   **per-book EMA (~5 sim-min half-life)**, not the raw tick. Keep the raw tick for the *execution*
   gate only. (Kills the 40%-on-noise flips at the source.) Warm-up gate: suppress flips until the EMA
   has enough samples.
2. **Restrict the dwell-bypassing EMERGENCY-FLIP to the stable fee axis** — only a taker-rebate-sign
   inversion (read off `account.fees`, ~3 sim-min confirm) may bypass dwell. **Remove the maker-edge
   (spread) emergency**; a spread-driven maker→taker switch must clear the **~15 sim-min persistence**
   gate or the full dwell. Add an **anti-oscillation latch** (no reverse flip within the ~45 sim-min
   cooldown, §4.2 — fee-bypassable).
3. **Make MAKER the sticky home state** — `MAKER` dwell ~45–60 sim-min (> the ~30 sim-min base);
   require an extra rebate cushion before leaving maker for taker; symmetric hysteresis on both axes.
4. **Fix the `_recent_pnl_bad` fast-switch** — deepen the evidence window and route a bad maker book to
   **IDLE**, not straight to TAKER (don't convert a dipping maker into a loss-cutting taker).
5. **Flip circuit-breaker** — per-book flip budget **~3–4 per 3-sim-h window**; pin a thrashing book.
6. **No-cut/bounded-cut legs** (§5.2/5.3) — so even a residual flip injects a *bounded* stream, not a
   4 bps cut-storm.
7. **Dampen the maker↔idle axis too — it's the CURRENT dominant churn.** Live AR logs (2026-06-25,
   miner-1/13/19) show this regime is **~99% maker**, with the churn now on **maker↔idle (~200+ ROUTE
   transitions/miner)** while maker↔taker emergency-flips are **rare (7–17)**. maker↔idle does NOT
   contaminate the per-book stream (idle adds no RTs), so it's cheaper than a maker↔taker flip — **but**
   it still churns resting orders and, worse, a book oscillating in/out of idle can fall below the
   ≥3-RT/5400s gate and **lose a scorable positive book from the median**. So route the maker↔idle gate
   off the **same spread-EMA** and give maker→idle a short exit-stickiness (don't idle a maker that's
   fine on a brief edge dip). **The dominant churn axis is regime-dependent** — maker↔taker in
   taker-favorable regimes (the uid199 161-flip slice), maker↔idle in maker-favorable ones (now); the
   anti-flip layer must cover both. (When emergency-flips *do* fire now, still ~43% at <6bps — the
   noise-driven mechanism is intact, just dormant in this regime.)

---

## 7. What's removed / retired (and why it's safe)

*(Named against AR V1's actual components — verified by grep, not the stale "modes" memory.)*

| Removed / changed | Why |
|---|---|
| `_activity_close` **force-activity on FLAT books** — the `_TakerMode` (L966) & `_MakerMode` (L1099) call-sites | Force-trading a dead book manufactures negative kappa; let flat books free-drop. **Keep** the `_IdleMode` drain call (L939) for residual inventory only |
| `fallback_maker` idle-budget promotion (L342) | Collapse to the thin both-bad floor + the existing `MAX_IDLE_BOOKS≈40` cap |
| RT-density floor (old P1-4 — a *proposal*, never in code) | Unnecessary under "trade-if-edge-else-idle": a low-fill book free-drops to None (no penalty) instead of being force-fed a negative RT; held positions still close via the maker's 4→8→18 bps managed-exit |
| ~~heartbeat / dormant / fee-sleep~~ | **N/A — not in AR V1.** Modes are already only taker/maker/idle (`_modes` L280); these concepts exist only in the standalone TakerScalper. Nothing to delete. |

**DO NOT CHANGE:** the live `AdaptiveRouterAgent.py` (the control); the FIFO accounting (`_match_fifo`);
the per-sim reset (`_ensure_simulation`); the proven maker kept as the A/B baseline.

---

## 8. Observability — the churn census
Emit a periodic per-validator census (**wall-clock throttle**, ~60 s): mode counts `{taker, maker,
idle}` (**must sum to 128**), route-flip and emergency-flip counts (split by direction), books that
flipped ≥3× this interval, current median local `kappa3`, and `idle_count`. This turns flip-rate from
forensics into a live dashboard and is the operational half of Goal #1 (watch it adapt without manual
checks). Reset counters in `_ensure_simulation`.

---

## 9. Rollout — phased, toggle-gated, A/B-isolable

- **Phase 0 — parity (differential harness).** Land `AdaptiveRouterV2Agent.py` with every toggle at
  V1-equivalent. Build a **differential** harness (not the hand-asserted v3 dryrun): instantiate V1 and
  V2 with the **same uid** (→ same jitter), feed both an identical scripted `(state, onTrade)` sequence,
  and assert **equal sorted instruction tuples per step**. That is the operational meaning of
  "byte-identical routing."
- **Phase 1 — spread-EMA (cadence-matched).** Enable on ONE miner vs a control sibling. **Acceptance bar
  is flip-rate, not kappa:** emergency-flips → ~0, **<6bps-spread flips → 0**, per-book route-flips ≤ the
  flip budget per 3-sim-h window (vs V1's up-to-11×). Judge kappa only after a full ~20h window; give
  ~15 min post-restart (AR cold-start ≈ 13 min). The **"uid199 replay" is an explicit Phase-1
  deliverable** — a tool that parses uid199's ROUTE/EMERGENCY-FLIP states and re-runs routing under V1
  vs V2 constants to count per-book flips *offline*, before going live.
- **Phase 2 — cadence constants.** Dwells, persistence, reverse-cooldown, flip budget (the §4.2 table).
- **Phase 3 — fee-axis-only emergency + sticky maker + fast-switch→idle + circuit-breaker.**
- **Phase 4 — leg swaps, each A/B'd:** V4 taker (SL = regime toggle defaulting to **tight cut ~2bps**
  per §5.2, single-open, `_activity_close` per-mode fix); then PMV4 maker (AR-current-maker vs
  AR-PMV4-maker). **Pre-Phase-4: restart the stale `miner-1` (uid78)** off its Jun-23 4-mode build onto
  the current 3-mode source first (it's the lone divergent process — see §10).
- **Phase 5 — idle floor + cap + census on, then retire the removed machinery.**

Stagger fleet restarts (transient dip + ~10–20 min re-discovery; `.env` ≠ running agent — args are
baked at launch, so changing a toggle needs a `run_miner.sh` relaunch, not just pm2 restart). Each phase
gated on explicit user confirmation. **Judge routing changes over ≥24 h real windows** (a flip's kappa
effect isn't visible until the contaminated stream ages out).

---

## 10. Open questions / tuning
1. **Cadence constants (§4.2) are starting points** — tune via the census + uid199 replay before locking.
2. **Taker cut width (corrected §5.2)** — make `max_gross_sl_bps` a toggle, **default the tighter cut
   (~2bps)** as the calm-regime base case, A/B 12 in fee-adverse. The matured A/B's "cut wins" was
   regime-dependent and is **not durable** (live now: cut uid192 −0.0023 vs no-cut uid247 +0.0002, both
   ≈0). Don't hard-set either.
3. **Does PMV4 beat AR's maker inside the router?** — A/B; don't assume.
4. **Severe-regime idle cap** — the `MAX_IDLE_BOOKS≈40` cap handles >48-both-bad; revisit if it ever
   binds.
5. **🟢 The stability thesis is validated by our own fleet — and `uid199` is OURS, not a competitor.**
   uid199 (kappa **0.025/p18**) is one of *our* 6 stable AR miners (same wallet), running ~2.4–2.6× our
   *restarted* local AR (uid44 0.010, uid78 0.0095) — largely because it's been **stable for days** vs
   the Jun-24 restarts (maturity over the full ~20h window; some build difference confounds it). Two
   takeaways: **(a)** this is the live proof that *stability is the #1 lever* (the plan's core) — so
   **stop churn-restarting the AR fleet** and let books mature; **(b)** the one concrete divergence is
   the **stale `miner-1` (uid78)** still on the Jun-23 4-mode (heartbeat/dormant) build — restart it
   onto the current source (the "consolidate divergent versions" point reduces to this single ops fix;
   "3 versions" was over-counted — it's 2).
6. **Dominant churn axis is regime-dependent (live-verified)** — maker↔taker in taker-favorable regimes
   (the uid199 161-flip slice), **maker↔idle now** (the `fallback_maker` idle guard + PnL-backoff,
   ~200+/miner). Phase-1 acceptance must measure the *currently-dominant* axis, not just emergency-flips
   (§6 item 7).
7. **MVTRX 0.5.0 (forward-looking)** — when live on a scoring validator: native `stop_loss`/`take_profit`
   /`max_slippage` (capability-probed) can replace hand-rolled exits; the `pnl_score` term (0.21)
   activates and rewards our bounded-PnL streams. Watch for `pnl_score` going non-zero on the endpoint.

---

## 11. Appendix — parameter map (live V1 → V2)

| Param | live V1 | V2 |
|---|---|---|
| routing spread input | instantaneous tick | **EMA, ~5 sim-min half-life** |
| `ROUTE_MIN_DWELL_S` | 180 s | **~30 sim-min** |
| MAKER sticky dwell | — | **~45–60 sim-min** |
| spread-flip persistence | — | **~15 sim-min** |
| fee-inversion confirm | 0 | **~3 sim-min** |
| reverse-flip cooldown | — | **~45 sim-min, fee-bypassable** |
| per-book flip budget | — | **~3–4 / 3-sim-h window** |
| EMERGENCY (spread/maker-edge) | bypasses dwell | **removed** (fee-axis only) |
| taker SL | 4 bps (+ pyramiding) | **toggle, default tight ~2 bps (calm) / 12 (adverse), single-open** |
| `RT_LOSS_CAP_BPS` | shared, hardcoded in `_activity_close` | **split per-mode + `mode` threaded** |
| maker stop | fixed 10 bps | **vol-band 10–14 bps** |
| maker giveup | 90 s | **150 s** |
| maker reprice keep | tick/2 | **1.5 ticks** |
| maker reduce-walk target | touch | **breakeven** |
| maker inventory cap | 2.0 lots | **1.5 lots** |
| modes | taker/maker/idle (source already 3-mode; `fallback_maker` idle-budget; force-activity via `_activity_close`) | **taker/maker/idle (thin floor, `MAX_IDLE_BOOKS≈40`; no force-activity)** |
| force-activity | on | **off** |
| census | — | **on (sum=128, wall-clock throttle)** |

*All time constants are SIM-time (against `state.timestamp`), ideally coded as fractions of
`KAPPA_RT_HISTORY_S`. Execution gates remain on the raw tick.*
