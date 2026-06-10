# Subnet 79 (MVTRX / τaos) — Miner Strategy Report

> Working document to design a competitive trading agent for netuid 79.
> Combines repo/code analysis with live on-chain + dashboard observations
> **plus an empirical price-path study of the target validator's books** (§3B).
> All "current values" reflect simulation `20260528_1007`.
> **Scoring config can change weekly — re-verify the Scoring Config table
> before every serious deployment.**

> ### Scope decision (locked)
> **We optimize for ONE validator only:**
> `5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF`.
> It carries effectively all the stake/weight, so aligning to its simulation
> and scoring config is the entire game. Our miner still answers other
> validators, but **all tuning targets this one**. (This is the hotkey shown as
> "UID 0" on the Grafana Validators page — the owner/aggregator validator.)

> ### TL;DR — the findings that change everything
> An empirical study of all **128 books** on this validator's tape
> (sim `20260528_1007`, 178k trades) shows:
> 1. Returns are **mean-reverting at every horizon** (1 s–120 s); momentum is
>    essentially absent → **do not run a momentum scalper**.
> 2. **123/128 books** show **sharp dumps (~20 bps/s) and slow grinds up
>    (~0.3 bps/s)**; after a ≥50 bps cliff, median **+63 bps recovery in ~5 min**.
>    The next agent must treat this **asymmetrically** (§3B.5, §5.3).
> Recommended pivot: **mean-reversion range-fader** with explicit **crash–recovery**
> rules — start from `MeanReversionAgent.py`, not `MomentumScalperAgent.py`.

---

## 1. What the subnet is

MVTRX runs many simultaneous **simulated limit-order-book markets** (currently
**128 books**). Miners are **trading agents**: each state update they receive
the L2/L3 book state and respond with orders/cancels. Validators host a C++
simulator + background-agent population, score miners on **risk-adjusted
realized profit**, and set weights.

- Prices are **not** purely miner-driven. Each book has a large background
  population (noise, stylized, HFT, ALGO, futures traders) plus stochastic
  **FundamentalPrice** and **FuturesSignal** processes seeded from live
  Coinbase/Binance feeds by the validator. Miners trade *into* this market and
  do have market impact, but they don't set the price alone.
- A second, **opt-in** incentive pool (**GenTRX**, ~5%) rewards distributed
  model-training. **Currently uncontested** (see §6).

### Key simulation parameters (current)

| Param | Value | Notes |
|-------|-------|-------|
| Books | 128 | Must trade across (almost) all of them |
| Duration | 1 day sim time | New config ~weekly (Wed ~17:00 UTC mainnet) |
| Init/grace period | 10 min sim | No miner orders accepted before this |
| Publish interval | 1 s sim | You act once per sim-second; strategies operate ≥1s |
| Init price | 300 | |
| Miner wealth | 50,000 QUOTE **per book** | Basis for volume cap + PnL reference |
| Capital type | pareto | Miners get randomized starting capital |
| Decimals | price 2, vol 4, base 4, quote 10 | Round orders accordingly |
| Min order size | 0.25 BASE | |
| Max open orders | 100 / book | |
| Max leverage | 4 | |

### Fee policy (DIS — Dynamic Incentive Structure)

| Param | Value |
|-------|-------|
| Target MTR | 40% |
| Base maker rate | 0.00% |
| Base taker rate | 0.02% |
| Max rebate / max fee | 1.50% |

Fees float per book with the **maker-taker ratio (MTR)**. If a book is too
maker-heavy vs target, **makers pay and takers get rebates** (observed in the
tape); if too taker-heavy, the reverse. Fees are realized into PnL, so fee
regime per book matters but is secondary to directional edge right now.

---

## 2. How scoring actually works (verified from code)

Source: `taos/im/validator/reward.py`, `taos/im/utils/kappa.py`,
`taos/im/config/__init__.py`.

```
final_score (per UID) = trading_score          (when GenTRX not run by miner)
trading_score        = kappa_weight * kappa_score + pnl_weight * pnl_score
on-chain weight      = slow EMA of final_score  (moving_average_alpha)
```

### Current weights

| Component | Weight | Current effect |
|-----------|--------|----------------|
| Kappa3 Score | **0.79** | Dominant driver |
| PnL Score | **0.21** | Secondary; after normalization contributes little |
| GenTRX | 0.05 pool share | Unused → returns to trading |

### Kappa-3 (the thing that matters most)

Per book, over a rolling **3-hour** realized-PnL window
(`scoring.kappa.lookback`):

```
K3 = (mean_return - tau) / cbrt(LPM3(tau))      tau = 0
```

- **Returns = realized P&L from completed round-trips only.** Open/unrealized
  inventory does **not** feed Kappa-3.
- Returns are **MAD-normalized per book** (scale-invariant — absolute PnL size
  matters less than *consistency and sign*).
- `LPM3` = mean of cubed **downside** deviations → **losses are cubed**. One
  bad round-trip hurts far more than an equal win helps.
- Need **≥ 3 realized observations** per book (`min_realized_observations`),
  and **≥ 90 min** of history (`min_lookback`) before a book scores.
- Per-book K3 is normalized to [0,1] from the range **[-2.5, 2.5]**
  (`normalization_min/max`), then aggregated by **median across books**.

### Activity factor — impact off, multiplier still gates each book

```
activity_factor = min(1 + (roundtrip_volume / volume_cap) * activity_impact, 2.0)
weighted_kappa  = activity_factor × pnl_factor × normalized_kappa   (per book)
```

**`activity_impact = 0.0` today does *not* mean activity is ignored.** It means
there is **no volume boost** (active books cap at **1.0×**, not up to 2.0×).
It does **not** remove the activity multiplier from scoring.

How factors are set (`taos/im/validator/reward.py`):

| Book state | `activity_impact = 0` | `activity_impact > 0` |
|------------|------------------------|------------------------|
| Round-trip in the last **10 min** sampling interval | `activity_factor → 1.0` | `1.0` … `2.0` by volume |
| Never traded / factor still **0.0** | **`weighted_kappa = 0`** on that book | same zero until first RT |
| Idle after a prior `1.0`, `decay_rate = 0` | Stays **1.0** (decay skipped) | Decay pulls factor below 1 |

**Critical:** A book whose `activity_factor` is **0** contributes **no score**
on that book (`0 × kappa = 0`), even with good Kappa-3. You must get at least
one realized round-trip per book so the factor moves off zero. Top miners show
**Activity = 1.0** on traded books for this reason.

**What `activity_impact = 0` changes:** no extra reward for high volume; no
decay acceleration when `decay_rate = 0`. Volume is still capped at
`capital_turnover_cap × miner_wealth` per book per 24h assessment window.

**Cadence (when impact is turned on later):**
`trade_volume_sampling_interval` = **600 s (10 sim-minutes)** — trade each book
at least once per interval to stay in the “active” branch. Assessment window =
**24 sim-hours**.

### PnL factor / PnL score

- `kappa.pnl.impact = 0.0` → the multiplicative PnL factor is neutral (1.0).
- The additive `pnl_score` (weight 0.21) uses per-book daily-return normalized
  to [-0.5, 0.5], median across books. Empirically the top miners' trading
  score ≈ `0.79 × kappa_score`, so PnL score is currently a small tie-breaker.

### Cross-book outlier penalty (consistency enforcement)

- Uses the **1.5×IQR rule** on per-book activity-weighted Kappas.
- Books significantly **below** your median (and < 0.5) trigger a penalty
  subtracted from your median score.
- Tight, consistent per-book performance → low/zero penalty (top miners show
  **Penalty = 0.0000**).
- **Lesson: a few terrible books can drag your whole score. Uniformity wins.**

### Inactive-book tolerance

- `max_inactive_books_ratio` allows some books to have no Kappa without
  penalty; **excess** inactive books count as **0.0** and pull down the median.
- You must be active and profitable on the large majority of all 128 books.

### Volume cap (hard limit, not a score input)

- `capital_turnover_cap = 10` × `miner_wealth (50,000)` = **500,000 QUOTE per
  book per rolling 24 sim-hours**.
- Enforced in `taos/im/validator/query.py`: once hit on a book, **only cancels
  accepted** until rolling volume drops below cap.
- **Volume is NOT reset between simulations** (24h rolling window can span sims).

### Deregistration / reset

- On deregistration + re-registration of a UID, the account's balances and
  positions reset to configured starting values.
- **Kappa/score rolling windows are NOT cleared at sim start** — past
  performance still feeds the EMA weight.

---

## 3. Live competitive landscape (on-chain + dashboard)

### On-chain (Finney netuid 79, ~block 8,321,067)

- 256 UIDs, **194 miners earning incentive**.
- **UID 0 ≈ 80% of incentive** — this is the **owner / GenTRX aggregator**
  (stake ~3.7M, dividends 0.93, vTrust 1.0). **Not a copyable trading
  strategy**; exclude it from competitive analysis.
- Real miner competition splits the remaining ~20%, fairly flat:
  top miner ~7.8% of the *miner* pool, then a long tail.

### Operator clustering (important)

The leaderboard is **a few operators running fleets of UIDs**, co-located:

| Coldkey (prefix) | Top-40 UIDs controlled | Hosting |
|------------------|------------------------|---------|
| `5Gh2Kpjf…` | 57, 120, 245, 220, 144, 56, 249, 163 | mostly `38.127.44.98` |
| `5Do2v6Gb…` | 67, 30, 101, 90, 207, 32, 133, 228, 126 | |
| `5HEoMd1u…` | 75, 54, 4, 26, 51, 48, 229, 74 | |
| `5HQK9RcY…` | 125, 82, 164, 137, 11, 187 | |
| `5EKrj5pY…` | 127, 184, 19, 143, 158, 227, 237 | |

- **`38.127.44.98` alone hosts 18 of the top-40 miner UIDs.**
- Co-location → **low latency** → smaller execution delay/slippage (the sim
  delays your fills by your response time). Latency is a real edge here.

### Dashboard leaderboard (sim 20260528_1007, Kappa window 3h)

| Pos | UID | 24H Vol | 24H RT | Realized PnL | Med Kappa3 | Kappa3 Score | Trading Score | Penalty | Activity |
|-----|-----|---------|--------|--------------|------------|--------------|---------------|---------|----------|
| 0 | 57 | 37,540 | 7,514 | 37,658 | 0.0937 | 0.5187 | 0.4098 | 0 | 1.0 |
| 1 | 75 | 38,681 | 7,364 | 39,556 | 0.0949 | 0.5190 | 0.4100 | 0 | 1.0 |
| 2 | 67 | 34,903 | 7,255 | 37,941 | 0.0884 | 0.5177 | 0.4090 | 0 | 1.0 |
| 3 | 30 | 36,164 | 7,293 | 35,995 | 0.0825 | 0.5165 | 0.4080 | 0 | 1.0 |
| 4 | 54 | 35,916 | 7,931 | 34,913 | 0.0892 | 0.5178 | 0.4091 | 0 | 1.0 |
| 5 | 32 | 40,310 | **15,110** | **14,665** | 0.0878 | 0.5176 | 0.4089 | 0 | 1.0 |
| 6 | 245 | 35,998 | 7,849 | 38,707 | 0.0884 | 0.5177 | 0.4090 | 0 | 1.0 |
| 7 | 144 | 45,102 | **17,961** | **15,296** | 0.0856 | 0.5171 | 0.4085 | 0 | 1.0 |
| 8 | 127 | 39,796 | 8,112 | 39,401 | 0.0837 | 0.5167 | 0.4082 | 0 | 1.0 |

### What the data proves

1. **Volume is not the differentiator.** Top miners use only ~35–45k of the
   **500k** cap. Activity = 1.0 for all (impact=0).
2. **High realized PnL per round-trip wins.** UIDs 32 & 144 do ~2× the
   round-trip volume but earn **less than half** the realized PnL and **rank
   lower**. Churning is pure cost (fees + downside risk) with no reward.
3. **The top cluster is one replicated strategy** (near-identical Kappa3 ≈
   0.08–0.095, Trading Score ≈ 0.409). Matches the coldkey/IP fleets.
4. **Penalty = 0 everywhere** → they trade *consistently across all 128 books*.
5. Top miners hold a **net long BASE inventory** (~140 BASE/book) while price
   trended up (313–345 vs init 300) — a directional tilt that's been working,
   but **only realized closes feed Kappa-3**.

---

## 3B. Empirical price-path analysis — this validator's 128 books

Source data: `agents/data/189/5EWwdZB7…/20260528_1007/trades.csv` (the actual
market tape this validator produced), **178,485 trades across all 128 books**.
Method: per book, reconstruct last-trade price series, resample to fixed bars,
measure direction, range, volatility, trend-vs-chop, and **return
autocorrelation** (the momentum-vs-mean-reversion test).

### 3B.1 Per-book summary (distribution across 128 books)

| Metric | Min | Median | Max | Meaning |
|--------|-----|--------|-----|---------|
| Trades / book | 620 | 1,371 | 2,586 | Plenty of fills to round-trip everywhere |
| Total return (day) | **−13.4%** | +0.9% | **+7.7%** | Net daily drift, both signs |
| Range (hi−lo)/p0 | 0.8% | 3.6% | 18.1% | How much room a fader has |
| 1 s log-ret vol | 0.03% | 0.09% | 0.52% | Per-second move size |
| Trendiness¹ | 0.04 | 0.58 | 1.00 | 1 = clean trend, 0 = round-trips to start |
| Max drawdown | 0.1% | 2.3% | 18.0% | Worst peak→trough |
| Max run-up | 0.8% | 3.0% | 12.1% | Best trough→peak |
| Trades / hour | 417 | 920 | 1,710 | Liquidity is high on every book |

¹ trendiness = |p_end − p_start| / (hi − lo): close to 1 means the move was
one-directional; near 0 means price wandered and came back.

### 3B.2 Pattern buckets (shape × volatility regime)

Classified by net direction, trendiness, and volatility tercile:

| Shape | Vol regime | # books | Example books |
|-------|-----------|---------|---------------|
| mixed (drift + chop) | lo-vol | 32 | 5, 6, 9, 12, 21, 28, 29, 33 |
| trend up | lo-vol | 27 | 2, 4, 13, 14, 20, 25, 26, 31 |
| choppy range | lo-vol | 24 | 3, 7, 16, 43, 46, 48, 50, 55 |
| choppy range | hi-vol | 15 | 1, 8, 11, 15, 17, 18, 24, 44 |
| trend down | hi-vol | 14 | 0, 19, 22, 23, 27, 36, 38, 65 |
| mixed | hi-vol | 12 | 10, 40, 47, 63, 76, 87, 88, 99 |
| trend down | lo-vol | 2 | 54, 73 |
| trend up | hi-vol | 2 | 30, 32 |

**Shape totals:** mixed **44**, choppy-range **39**, trend-up **29**,
trend-down **16**.

Read this as: **~65% of books are range/chop or mixed** (no clean trend), and
trends split both directions. There is **no single dominant trend** to ride —
which already argues against a pure momentum approach.

### 3B.3 The decisive test — return autocorrelation by timeframe

Lag-1 autocorrelation of bar returns, across all 128 books:

| Bar size | Median ACF | Mean-reverting books (ACF<−0.03) | Momentum books (ACF>+0.03) |
|----------|-----------|----------------------------------|----------------------------|
| 1 s | −0.018 | 53 | 0 |
| 5 s | −0.016 | 43 | 2 |
| 15 s | −0.022 | 57 | 5 |
| 30 s | −0.022 | 57 | 7 |
| 60 s | −0.055 | 81 | 11 |
| 120 s | **−0.095** | **92** | 11 |

Plus: the **average return in the step right after a +1σ jump is negative**
(≈ −0.02σ, median across books) → **spikes tend to reverse**, exactly the
"sudden dump then recover" behavior seen on the book-0 chart.

**Interpretation:**
- **Momentum is statistically absent** (0–11 books out of 128 at any horizon).
- **Mean-reversion is pervasive and grows with horizon** (53 → 92 books).
- This is consistent with the market's construction (§ background agents):
  **fundamentalist stylized traders + the ALGO trader pull price back toward
  FP**, and **HFT market-makers** dampen momentum. Price oscillates around a
  slowly-moving FP anchor rather than trending tick-to-tick.

### 3B.4 Why this contradicts our current agent

`MomentumScalperAgent` enters **only when trend + LOB imbalance + flow all
agree in the same direction**, i.e. it **buys strength / sells weakness**. On a
tape where the next move after a push is, on average, a **pullback**, that
entry rule systematically enters **right before reversion** — paying taker fees
to do it. That is the worst quadrant for Kappa-3 (LPM3 cubes the resulting
losing round-trips).

### 3B.5 The “sudden dump, slow rise” pattern (explicit check)

This is the shape you called out on book 0 (stair-step up, vertical cliff, then
gradual recovery). We **did not** bucket it as its own label in the first pass,
but a dedicated asymmetry test on the same tape confirms it is **the dominant
microstructure on this validator**, not an occasional outlier.

Per book (1 s resampled last-trade prices, sim `20260528_1007`):

| Measure | Result |
|---------|--------|
| Median **drop speed** (largest drop per second of wall time) | **~20 bps/s** |
| Median **rise speed** (largest rise per second of wall time) | **~0.3 bps/s** |
| Median rise-speed / drop-speed ratio | **~0.015** (rises are ~60× slower) |
| Books matching “sharp dump + slower grind up” heuristic | **123 / 128** |
| Books with ≥50 bps drop inside 60 s at least once | **109 / 128** |
| **After** such a drop: median return over next ~300 s | **+63 bps** (97/109 books positive) |

**What this means mechanically:**
- **Dumps** = aggressive taker sell sweeps through the LOB (last-print cliff).
  They are **fast** and often **overshoot** fair value.
- **Recovery** = background ALGO/stylized flow + HFT makers **pull price back
  toward FP** over many small steps → **slow grind**, not a V-reversal in one tick.
- It is **not** symmetric: you cannot mirror long/short rules; the edge is
  **asymmetric fade timing**.

**Examples from the tape (book 0–style):**

| Book | Max drop (60 s window) | Best rise window | Drop speed | Rise speed |
|------|------------------------|------------------|------------|------------|
| 0 | ~702 bps in 5 s | ~340 bps in 300 s | ~140 bps/s | ~1.1 bps/s |
| 19 | ~1462 bps in 5 s | ~169 bps in 600 s | ~292 bps/s | ~0.3 bps/s |
| 22 | ~743 bps in 5 s | ~76 bps in 300 s | ~149 bps/s | ~0.3 bps/s |

So the Grafana book-0 chart and the FP chart diverge **because FP never takes
the cliff** — it is the slow anchor; the **trade-price** series takes the sweep
and then mean-reverts on a longer clock.

### 3B.6 What the patterns imply for strategy routing

- **Choppy-range + mixed (≈83 books):** core money-makers for a **fade**
  strategy — sell upper band / buy lower band, target reversion to a rolling
  mean/microprice, tight stop on band break.
- **Trend books (≈45, both directions):** a fader must avoid being run over.
  Use a **trend filter** (e.g. slope of a longer EMA / position vs FP-proxy):
  in a confirmed trend, only fade **against** the minor counter-swings, or
  widen the entry band and shrink size, or skip.
- **Volatility regime** sets **band width and size**: hi-vol books → wider
  entry offsets, smaller size, same risk budget; lo-vol → tighter bands,
  more round-trips.

> One adaptive strategy can cover all books if its **band width and trend
> filter are computed per book from live stats** — we do **not** need separate
> code per pattern, just per-book parameters derived at runtime.

---

## 4. The register → profit → deregister cycle (your observation)

Operators run **many UIDs**. Behavior pattern: spike to the top, extract
emissions, and as a UID's rolling window decays / capital is spent / a fresh
config favors a reset, they **let weak UIDs deregister and re-register** (which
resets balances to the clean starting allocation). Why this works under the
current rules:

- **Balances reset on re-registration** → a fresh clean book to start
  round-tripping from, no accumulated bad inventory.
- **Volume cap is rolling 24h and not reset by sim** → a fresh UID isn't
  carrying volume baggage.
- **Activity impact = 0** → no need to maintain continuous volume to avoid
  decay; you can run hard while ranked, then cycle.
- Running **N UIDs** diversifies variance: some books/sims go badly, but the
  fleet median stays high, and losers get recycled.

**Strategic takeaways for us:**
- Single-UID miners are at a structural disadvantage vs fleets (variance,
  latency, recycling). Consider running a **small fleet** (e.g. 3–8 UIDs) with
  **per-UID jitter** so they don't self-interfere on identical signals.
- Treat **registration cost vs expected emission window** as an explicit
  economic calc: enter when your strategy is hot, recycle UIDs that have
  decayed.
- Because re-registration resets balances, **avoid accumulating un-closeable
  inventory** — keep positions round-trippable so a UID stays productive.

---

## 5. Target strategy — **mean-reversion range-fader** (recommended)

Two facts decide the design:
1. **Scoring pays** consistent, low-downside **realized round-trip PnL across
   all 128 books** (Kappa-3, LPM3 cubes losses; volume is not rewarded today).
2. **The tape mean-reverts** at every horizon (§3B). Fade extremes, don't chase.

So the edge is: **systematically sell local highs and buy local lows around a
slowly-moving fair value, capture the reversion, close fast, repeat — on every
book, with tight risk.** This is the opposite of the current momentum agent.

### 5.1 Core loop (one adaptive strategy, per-book parameters)

```
for each book (read everything from state.config / live stats):
  fair      = microprice                 # (bid*askVol + ask*bidVol)/(askVol+bidVol)
  ref       = EMA(fair, mean_window)     # slow anchor ≈ local fair value
  sigma     = rolling stdev of fair (or ATR-like band) on this book
  band      = k_entry * sigma            # per-book, scales with volatility
  trend     = slope of a longer EMA      # trend filter (see 5.3)

  if holding a position:
      exit when fair reverts to ref  (take-profit)         # realize PnL
      OR price breaks band by k_stop*sigma (stop)          # cap LPM3 tail
      OR max_hold elapsed                                   # recycle capital
  elif flat and |fair - ref| >= band and trend filter allows:
      fade it:  price above ref -> SELL toward ref
                price below ref -> BUY  toward ref
      prefer a MAKER limit at/just inside the band edge (earn rebate, better fill px)
      fall back to a small taker only if fee regime favors taking (5.4)
  keep >=3 realized round-trips / book / 3h; size for low per-book PnL variance
  stay < ~50% of the 500k/book volume cap
```

### 5.2 Why maker-first here (big change vs current agent)

- Taker round-trip cost ≈ 2 × 2.3 bps = **~4.6 bps**; median 1 s move is ~9 bps,
  so a taker fader's edge is thin. **Posting limits at the band edge** both
  (a) earns a **better entry price** and (b) can collect a **maker rebate** when
  the book is taker-heavy (DIS). This widens the per-round-trip margin that
  Kappa-3 rewards.
- Mean-reversion entries are **naturally patient** — you *want* to be the
  resting liquidity that gets hit when price overshoots, then reverts.
- Risk: maker limits may not fill. Mitigate with **short GTT expiries**,
  re-quoting each tick, and a taker fallback only when the signal is strong and
  the fee regime is favorable.

### 5.3 Crash–recovery asymmetry (must be explicit)

The §3B.5 pattern is **exactly** what a mean-reversion fader should exploit,
but only with **asymmetric rules** — not symmetric TP/SL/hold on long and short.

| Phase | What the tape does | What the agent should do |
|-------|-------------------|-------------------------|
| **Pre-cliff rip** | Slow grind up, then stretched above mean | **Fade shorts** (sell above ref) when stretch + flow not still aggressively buying (`imb` gate, as in `MeanReversionAgent`) |
| **Active cliff** | Vertical dump in seconds | **Do not catch the knife** — no new longs while 1 s return < −X bps or while sell-flow still dominant; optional **stand down** until volatility of last N s drops |
| **Post-cliff floor** | Overshoot below mean; median +63 bps in ~5 min | **Fade longs** (buy below ref) with **wider TP / longer `max_hold`** than normal — recovery is slow, so exits at ref too early leave money on the table |
| **Slow grind up** | Price creeps back; short squeeze risk for early shorts | If still short from pre-cliff fade: **tighter stop / time stop** — do not assume a fast snap-back down; the rise leg is slow but persistent |

Concrete parameters to tune in implementation (not hardcoded):
- `crash_bps` — e.g. 30–50 bps in ≤30–60 s → enter “post-crash” mode for that book
- `recovery_hold_mult` — e.g. 1.5–2.5× normal `max_hold_s` for longs entered post-crash
- `recovery_tp_bps` — slightly wider than standard `tp_bps` (grind gives time)
- `knife_catch_block_s` — block new longs for N seconds after crash unless imbalance flips
- `short_stop_tight_bps` — tighter stop on shorts during recovery grind (asymmetric vs long)

`MeanReversionAgent` already has the right **skeleton** (fade stretch + imbalance
gate + maker entry + realize on TP/SL/time). The **gap** vs production-ready
crash-awareness is: no **crash detector**, no **post-crash hold/TP asymmetry**, and
no **knife-catch block**. Those are the first additions when we implement §5.

### 5.4 Trend filter (so trend books don't run you over)

≈45 books trend (both directions). A fader must not keep buying a falling book.
Gate entries with a **longer-horizon slope** (e.g. EMA over minutes, or sign of
cumulative drift vs an FP-proxy built from the slow EMA):
- **Range/mixed book (slope ≈ 0):** fade both sides freely.
- **Confirmed up-trend:** only fade **dips** (buy below ref); skip/Shrink shorts.
- **Confirmed down-trend:** only fade **rips** (sell above ref); skip/Shrink longs.
- **Strong trend + hi-vol:** widen band, cut size, or stand down on that book.

This makes a single code path adapt across all 8 pattern buckets in §3B.

### 5.5 Exploit per-book fee regime (DIS)

Read `accounts[book].fees` each tick:
- **Takers being rebated** → a taker fade is *paid to enter*; use taker for
  speed/fill certainty.
- **Makers being rebated** (book taker-heavy) → strongly prefer posting limits.
Route order type to whichever side is currently subsidized.

### 5.6 Risk discipline (Kappa-3 is unforgiving)

- **LPM3 cubes losses** → one blow-out round-trip can sink a book's Kappa. Hard
  per-trade stop at `k_stop*sigma`; never average down past a fixed inventory.
- **Always realize.** Unrealized inventory scores nothing and dies on
  reset/cap. Enter only with a pre-set exit; ensure ≥3 closes/book/3h.
- **Uniformity across books** to avoid the IQR outlier penalty — same logic
  everywhere, per-book params from live stats; don't let books rot to 0.
- **Latency.** Keep `respond()` fast (parallel books, light per-tick math); the
  sim delays fills by your response time, and faders care about fill price.

### 5.7 Implementation status (DONE) + offline backtest

`agents/MeanReversionAgent.py` now implements all of §5/§5.3:
microprice band fade, per-book dispersion band, trend filter, crash detector,
post-crash long asymmetry (wider TP / longer hold), knife-catch block,
**short-block after crash**, maker-first entries with fee-aware taker fallback,
volume-cap tracking, and per-UID jitter.

**Offline backtest** on this validator's tape (mid-fill replay, all 128 books,
sim `20260528_1007`) — signal quality only, not the full matching engine:

| Config | Round-trips | Win-rate | Mean / RT | Per-book Kappa-3 proxy¹ |
|--------|-------------|----------|-----------|--------------------------|
| Fade only (no crash guards) | 2,628 | 61.5% | +10.2 bps | +0.578 |
| **+ block shorts on recently-crashed books** | 1,906 | **70.7%** | **+14.5 bps** | **+0.884** |

¹ proxy = mean / cbrt(LPM3), median across books; not the validator's exact
normalization, but the same downside-cubing shape — directionally meaningful.

**Key learnings baked into the agent:**
- **Shorting a freshly-dumped book is the dominant tail.** The slow grind up
  (§3B.5) runs shorts over. Blocking shorts for `short_block_after_crash_s`
  (default 30 min) both *raised* win-rate and *cut* churn (fewer, better RTs) —
  exactly what Kappa-3 rewards.
- Fewer round-trips with higher quality **beats** more volume (consistent with
  the leaderboard: top miners ~7-8k RT, not the 15-18k churners).
- Residual catastrophic losses are **gap-throughs within one publish interval**
  on a cliff — unavoidable by stop level alone; mitigated by *not being
  positioned into cliffs* (knife-block for longs, short-block for shorts).

### 5.8 Score-awareness vs the leaderboard (target operating point)

From §3 (this validator, sim `20260528_1007`): top miners run
**24H Vol ≈ 35-45k**, **24H RT ≈ 7-8k**, Activity = 1.0 (impact 0), Penalty = 0,
Trading Score ≈ 0.41 (Kappa3 Score ≈ 0.52, raw median Kappa3 ≈ 0.08-0.095).

Our configured operating point (defaults in `.env`):
- `volume_safety=0.5` → cap ourselves at **250k/book** (we use far less; ~20 RT/
  book/day × ~1.8k notional ≈ well under cap). **Do not chase the 15-18k RT
  churn** — it loses now.
- Aim for **consistent positive realized PnL on every book** (all 128 scored in
  backtest) to keep the **cross-book IQR penalty at 0** and the median Kappa
  high.
- Re-check weekly: if `activity_impact` goes > 0, volume becomes a lever and we
  raise `volume_safety` / round-trip frequency deliberately.

### 5.9 Concrete next experiment

We already have `MeanReversionAgent.py` in the repo as a starting point. Plan:
1. **Repurpose toward this design** (microprice band fade + trend filter +
   maker-first + per-book sigma) rather than extending the momentum agent.
2. **Backtest via `agents/proxy/run`** against `simulation_0.xml`; measure
   per-book realized-PnL distribution and simulated Kappa-3, **not** just total
   PnL.
3. Compare head-to-head vs `MomentumScalperAgent` on the same offline tape; the
   §3B autocorrelation result predicts the fader wins on realized round-trips.

### Anti-patterns to avoid

- ❌ **Momentum entries** (buy strength / sell weakness) — fights the tape (§3B).
- ❌ Volume farming (UIDs 32/144 prove it loses now).
- ❌ Passive limits that never fill (no realized PnL → no Kappa) — use short GTT
  + re-quote + taker fallback.
- ❌ Accumulating directional inventory you can't close.
- ❌ One great book + several bad books (IQR penalty + inactive-book zeros).
- ❌ Slow `respond()` → timeouts or large slippage.

---

## 6. GenTRX — free, uncontested incentive right now

- Every top miner shows **GenTRX Score = 0.0** → nobody is training.
- The ~5% pool currently **returns to the trading pool** (unused).
- An agent subclassing `GenTRXAgent` that submits acceptable gradients each
  round could capture this **additive, uncontested** reward with **no trade-off
  to trading** (training runs in a background thread).
- Requires: an R2/Hippius bucket, on-chain read-key commit, a GPU
  (6–8 GB VRAM min), and pulling the latest checkpoint each round
  (see `doc/gentrx/miner_setup.md`). Watch for version-mismatch rejections.
- **Cost/benefit:** worth it if you already have a GPU host; it diversifies
  your income away from the crowded trading pool.

---

## 7. Open questions / things to re-verify each week

- Is `activity_impact` still 0? (If it goes > 0, volume becomes a lever again —
  strategy must add controlled volume.)
- Is `kappa.pnl.impact` still 0 and `pnl.weight` still 0.21?
- Did `capital_turnover_cap`, `miner_wealth`, book count, or fee params change?
- Kappa window (currently 3h), `min_realized_observations` (3),
  `max_inactive_books_ratio` — all affect how many closes/books you need.
- GenTRX `simulation_share` and whether competitors start training.

---

## 8. Immediate next steps

1. **Pivot from momentum to mean-reversion.** Build the §5 range-fader
   (microprice band + per-book sigma + trend filter + maker-first + strict
   round-trip/exit discipline + all-book uniformity). Start from
   `MeanReversionAgent.py`, not `MomentumScalperAgent.py`.
2. **Backtest locally** via `agents/proxy/run` against the live
   `simulation_0.xml` background model before any chain deployment; report
   per-book realized-PnL distribution + simulated Kappa-3, and a head-to-head
   vs the momentum agent on the same tape.
3. **Deploy on testnet (netuid 366)** to validate latency, success rate, and
   that every book scores (≥3 round-trips/book/3h, Penalty ~0).
4. **Tune for Kappa-3**: track per-book realized-PnL distribution; cut left-tail
   losses; verify Trading Score approaches/exceeds the ~0.41 top cluster.
5. **Consider a small jittered fleet** + a **GenTRX-enabled** variant to grab
   the uncontested training pool.
6. **Re-pull the dashboard Scoring Config weekly** and re-tune.

---

## 9. Simulation parameters — history, where to look, manual vs automatic

### Focus on the owner validator (dashboard “UID 0”)

On the Validators dashboard, **UID 0** is the **validator** (hotkey
`5EWwdZB7…`), not a trading miner. It is the subnet owner / GenTRX aggregator
(~80% on-chain incentive, vTrust 1.0). The **Agents** table on that page lists
**miner** UIDs (57, 75, 67, …) scored inside *that validator’s* simulation.

**Why focus here:** This validator is the reference deployment others try to
align with. Your miner still receives state from **every** validator that
queries you, but tuning against this dashboard + its sim config matches the
highest-trust environment.

**Caveat (FAQ §10):** Each validator runs its **own stochastic instance** of
the same XML. Paths differ run-to-run and host-to-host even when parameters
match. You cannot replicate one validator’s exact prices — only the same
rules.

### Two different things that “change”

| Layer | What it is | Example | How often |
|-------|------------|---------|-----------|
| **Simulation run** | One execution of a config | `simulation_id = 20260528_1007` | New run every time the sim restarts (often daily+ on a host) |
| **Simulation config** | Rules from XML (`simulation_0.xml` etc.) | 128 books, 50k wealth, 10m grace, DIS fees | Intended ~**weekly** (Wed ~17:00 UTC mainnet per FAQ) |

- **Same config, new run:** `book_count`, `miner_wealth`, decimals, fees stay
  the same; **prices, background flow, and `simulation_id` change** (new RNG).
- **New config deployed:** Structural fields can change (books, wealth, agent
  counts, grace, duration, fee policy). Validators pull new XML from the repo.

There is **no public “parameter history API”** in the repo. History is
inferred from:

1. **Grafana** — Validators page → **Simulation Config** + **Scoring Config**
   (current run only; change **Simulation ID** / time range if the UI allows).
2. **Git** — `simulate/trading/run/config/simulation_0.xml` (and any future
   `simulation_1.xml`, …) + commit history / release notes / Discord.
3. **Per-run logs** — Validator log dirs named by timestamp → `simulation_id`
   (e.g. `20260528_1007`).

### Current live config vs repo file (snapshot you pasted)

Your dashboard (sim `20260528_1007`) matches the repo XML in spirit; scoring
is validator-side:

| Field | Live (UID-0 validator) | Repo `simulation_0.xml` |
|-------|------------------------|-------------------------|
| Books | 128 | 8×16 = 128 |
| Duration | 1 day | 86400e9 ns |
| Init period | 10 min | gracePeriod 600e9 ns |
| Publish interval | 1 s | step 1e9 ns |
| Miner wealth | 50,000 QUOTE | wealth 50000 |
| Init price | 300 | 300 |
| Capital type | pareto | pareto |
| Max 24H vol (scoring) | 500,000 | 10 × 50,000 |

**Scoring config** (Kappa 79%, PnL 21%, activity impact **0**, 3h window) is
**not** in the XML — it lives in the validator process / dashboard only.

### Does the config change much between simulations?

**Between runs of the same XML:** Market **behavior** changes a lot
(stochastic background + external seeds), but **rules** are stable. Your agent
logic should not need a code change — only handle a fresh sim via `onStart`.

**Between weekly deploys:** Can change materially when owners adjust book
count, wealth, background agent counts, fees, or grace. That is why the
protocol sends full config every state update and uses `config.label()` on the
validator to reset internal state when the **structure** changes.

Typical weekly changes (when announced): book count, `miner_wealth`, fee/DIS
params, background agent population — not every constant every week.

### Where your agent gets parameters (automatic)

Every state update includes `state.config` as `MarketSimulationConfig`
(parsed from the validator’s XML). The framework sets:

```text
self.simulation_config = state.config   # every tick in FinanceSimulationAgent.update()
```

Use at runtime (no hardcoding):

- `state.config.book_count` — loop all books
- `state.config.miner_wealth` — volume cap = 10 × this (if cap unchanged)
- `state.config.priceDecimals`, `volumeDecimals`, `baseDecimals`, `quoteDecimals`
- `state.config.publish_interval`, `grace_period`, `duration`
- `state.config.max_open_orders`, `max_leverage`, `minOrderSize` (via sim rules)
- `state.config.simulation_id` — **which run** you are in
- `state.config.fee_policy` — DIS maker/taker baselines

Also in state: `state.accounts[book_id]`, `state.books`, `state.notices` (incl.
`SimulationStartEvent` / `SimulationEndEvent`).

### What is NOT sent to miners (monitor manually)

**Scoring parameters** are validator-local and appear on the dashboard, not in
`state.config`:

- Kappa3 weight / PnL weight / GenTRX share
- `activity_impact`, `activity_decay_rate`
- Kappa assessment window, min observations
- `capital_turnover_cap`, `max_instructions_per_book`

If these change (e.g. `activity_impact` back to 0.33), your **strategy
tuning** may need to change even though agent code still runs. Check the
**Scoring Config** table weekly — no automatic hook in the miner protocol
today.

### Do you need to update agent code every time?

| Event | Change agent code? | What to do |
|-------|-------------------|------------|
| New **simulation run** (new `simulation_id`) | **No** (if config-driven) | Override `onStart()` — clear per-book state, re-init history |
| New **XML config** (books/wealth/decimals) | **No** (if you read `state.config`) | Same; verify loops use `book_count`, rounding uses decimals |
| New **scoring rules** on validator | **Maybe** (strategy params) | Read dashboard; adjust thresholds / volume target |
| New **subnet code** (validator/miner release) | **Yes** | `git pull`, `pip install -e .`, restart miner |

**Anti-pattern:** Hardcoding `128`, `50000`, `500000` volume cap, or `0.35`
imbalance thresholds in source — breaks on the next config deploy.

**Good pattern:**

```python
def respond(self, state):
    cfg = state.config
    for book_id in range(cfg.book_count):
        qty = round(size, cfg.volumeDecimals)
        price = round(px, cfg.priceDecimals)
    # volume_cap = 10 * cfg.miner_wealth  # confirm cap multiplier still 10 via dashboard
```

**Detect new run** (optional but recommended):

```python
def onStart(self, event):
    self._positions.clear()
    self._history.clear()
    # reset anything keyed by sim time or book

def respond(self, state):
    sid = state.config.simulation_id
    if sid != getattr(self, "_last_sim_id", None):
        self._last_sim_id = sid
        # first tick after id change — treat like warm-up if needed
```

`RevengAgent` already resets studies on `state.config.simulation_id` change.

### Practical workflow for you

1. **Before deploy:** Open owner validator dashboard → copy **Simulation
   Config** + **Scoring Config** into notes (or this report §7 checklist).
2. **Agent code:** Read everything from `state.config`; implement `onStart`
   reset; never assume book count or wealth.
3. **After weekly update:** Diff `simulation_0.xml` on GitHub; re-check
   dashboard Scoring Config; adjust **strategy params** (`--agent.params`) only
   if needed — usually not the Python file structure.
4. **Local test:** Point proxy at the same XML path validators use
   (`agents/proxy/run --sim-xml …/simulation_0.xml`) before mainnet.

---

*Generated from analysis of the sn-79 repo (`taos/im/validator/reward.py`,
`kappa.py`, `config/__init__.py`, `simulation_0.xml`, `agents/`), the Finney
netuid-79 metagraph, the validator `5EWwdZB7…` Grafana dashboard, and an
**empirical study of that validator's full trade tape** (§3B:
`agents/data/189/5EWwdZB7…/20260528_1007/trades.csv`, 178,485 trades, 128
books — direction, range, volatility, and return-autocorrelation per book).*
