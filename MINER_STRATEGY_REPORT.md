# Subnet 79 (MVTRX / τaos) — Miner Strategy Report

> Working document to design a competitive trading agent for netuid 79.
> Combines repo/code analysis with live on-chain + dashboard observations.
> All "current values" reflect simulation `20260528_1007` as seen on the
> UID-0 validator dashboard. **Scoring config can change weekly — re-verify
> the Scoring Config table before every serious deployment.**

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

### Activity factor — CURRENTLY NEUTRALIZED

```
activity_factor = min(1 + (roundtrip_volume / volume_cap) * activity_impact, 2.0)
```

- `activity_impact = 0.0` right now → **activity_factor = 1.0 for everyone.**
- `activity_decay_rate = 0.0` → no decay penalty for pausing.
- **Implication: trading more volume gives ZERO score boost today.** Volume is
  only a *constraint* (the cap), not a lever. This is the biggest lever-change
  vs the README narrative, which assumes activity weighting is on.

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

## 5. Target strategy profile (how to beat them)

Optimize for **what the scoring actually pays today**: consistent, low-downside
**realized round-trip PnL across all 128 books** — not volume.

### Design principles

1. **Maximize realized PnL per round-trip, minimize losing round-trips.**
   LPM3 cubes losses — a strategy with a high *win-rate / small controlled
   losses* beats a higher-average strategy with fat left-tail losses, even at
   equal mean. Tight stop discipline; never let a round-trip blow out.
2. **Always close positions to realize PnL.** Unrealized inventory ≠ score.
   Enter with a predefined exit; prefer trades you can round-trip within the
   3h window. Aim for **≥ 3 realized closes per book** within the window so
   every book scores.
3. **Be uniform across all 128 books.** Run the same logic on every book;
   keep per-book Kappa tight to avoid the IQR outlier penalty. Don't let a few
   books rot (they become 0.0 beyond the inactive tolerance).
4. **Don't chase volume.** Stay well under 500k/book. Extra volume = extra
   fees + downside exposure for zero score (while impact=0). Reassess instantly
   if `activity_impact` becomes > 0 in a future config.
5. **Latency matters.** Faster responses = less fill delay/slippage. Co-locate
   near major validators; keep `respond()` fast (consider `lazy_load=1`,
   parallel book processing, avoid heavy per-tick work).
6. **Exploit fee regime per book.** Under DIS, in books where takers get
   rebates, taking is *cheaper than free*; where makers get rebates, posting is
   subsidized. Read `accounts[book].fees.maker_fee_rate/taker_fee_rate` each
   tick and route order type to the side currently being paid.
7. **Mild, controlled directional tilt** can help in trending books, but it
   must be expressed through **closed round-trips**, not buy-and-hold.

### Signal candidates (start, then differentiate)

- Short-horizon **microstructure** signals: LOB imbalance (microprice),
  trade-sign autocorrelation, queue dynamics — cheap, fast, per-tick.
- A small **online predictor** (cf. `SimpleRegressorAgent`) on OHLC + trade/
  order imbalance to predict next-interval return; trade only when |signal| is
  high and you have a defined exit.
- **Regime gating** (cf. `HybridTrainingAgent`): flat→quote, strong signal→
  enter, holding→manage+stop. But the stock template self-interferes if copied
  — replace the signal and tune thresholds per UID.

### Concrete scaffolding to build

```
for each book:
  1. update fast features (imbalance, microprice, short return, fee state)
  2. if holding a position:
       - exit at target or stop (realize PnL); never widen risk
  3. elif strong, high-confidence signal AND no position:
       - enter sized small, with a pre-set close plan
  4. else:
       - optionally post inside-spread on the fee-favored side, GTT short expiry
  5. keep ≥3 round-trips/book/3h; keep per-book PnL variance low
  6. stay < ~50% of volume cap as a safety margin
```

### Risk / anti-patterns to avoid

- ❌ Volume farming (UIDs 32/144 prove it loses now).
- ❌ Passive limits that never fill (no realized PnL → no Kappa).
- ❌ Accumulating directional inventory you can't close (dies on reset/cap, and
  shows as unrealized, not scored).
- ❌ One great book + several bad books (IQR penalty + inactive-book zeros).
- ❌ Slow `respond()` → timeouts (no instructions submitted) or large slippage.

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

1. **Build a baseline agent** on the scaffolding in §5 (microstructure signal +
   strict round-trip/exit discipline + all-book uniformity).
2. **Backtest locally** via `agents/proxy/run` against the live
   `simulation_0.xml` background model before any chain deployment.
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
netuid-79 metagraph, and the UID-0 validator Grafana dashboard
(sim 20260528_1007).*
