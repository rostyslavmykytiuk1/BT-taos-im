# Miner Agent Development Plan — Subnet 79 (MVTRX / τaos)

> Companion to `MINER_STRATEGY_REPORT.md`. That file explains *why* the scoring
> works the way it does; this file explains *how* we turn that into deployable
> agents and ships three ready-to-run implementations.
>
> **Re-verify the scoring config on the UID-0 validator dashboard before every
> serious deployment.** Everything below assumes the currently-observed config
> (`activity_impact = 0.0`, Kappa weight ≈ 0.79, 3h Kappa window, 128 books,
> 500k QUOTE/book volume cap). GenTRX is intentionally out of scope.

---

## 1. What actually earns emission (the only 4 things that matter)

From `taos/im/validator/reward.py` + `taos/im/utils/kappa.py`, with the current config:

1. **Kappa-3 of realized round-trip PnL, per book, median across books (~79%).**
   Only *closed* round-trips count. Open inventory is invisible to the score.
   Downside is **cubed** (LPM₃) — one big loss hurts ~ as much as several wins help.
2. **Daily-return PnL score (~21%)** — secondary tie-breaker, same median-across-books shape.
3. **Cross-book consistency** — the 1.5×IQR outlier penalty subtracts from your
   median if a few books badly underperform. Top miners run **Penalty = 0**.
4. **Coverage** — you must be active+profitable on the large majority of all 128 books;
   excess inactive books score 0 and drag your median down.

**Things that do NOT help right now:** raw volume beyond getting each book’s
`activity_factor` to **1.0** (`activity_impact=0` → no boost above 1.0×), churning
round-trips (spread + Kappa tail risk), holding a big unrealized winner (not scored
until closed). **Books left at `activity_factor=0` score zero on that book** — see
report §2 Activity factor.

### The design target

> Produce **many small, consistently-positive realized round-trips on (almost)
> every one of the 128 books**, with a **short, thin downside tail**, while
> staying well under the volume cap.

That single sentence drives every parameter default in the three agents.

---

## 2. Engineering rules (apply to every agent)

These are baked into the shipped code; keep them if you write your own.

- **Subclass `FinanceSimulationAgent`** and implement only `initialize()` and
  `respond(state)`. The base class auto-populates `self.simulation_config`
  (`state.config`), `self.accounts` (`{book_id: Account}`), and `self.events`,
  and dispatches `onStart` / `onTrade` *before* `respond` is called.
- **Read everything from `state.config` at runtime** — `book_count`,
  `priceDecimals`, `volumeDecimals`, `publish_interval`, `grace_period`,
  `miner_wealth`, `max_open_orders`. Never hard-code market params; the config
  is redeployed ~weekly and the agent must adapt with zero code changes.
- **Keep `respond()` fast and allocation-light.** You are called once per
  publish interval for *all* books; a slow response adds execution delay
  (the sim lags your fills by your response time → slippage). Use only top-of-book
  math, no per-step model inference, no blocking I/O.
- **Reset per-simulation state in `onStart()`** and also when `simulation_id`
  changes, so a new sim never inherits stale positions/averages.
- **Track your own position + cost basis from fills (`onTrade`)**, not from
  balance snapshots — it's exact and lets you compute realized targets.
- **Respect hard limits:** `max_open_orders` per book, min order size 0.25 BASE,
  round prices to `priceDecimals` and quantities to `volumeDecimals`, and stop
  *opening* (still allow closing) once a book nears the volume cap
  (`capital_turnover_cap × miner_wealth`).
- **Always check affordability** against `account.*_balance.free` before sending.
- **React to DIS fees** via `account.fees.maker_fee_rate / taker_fee_rate`
  (these float per book and can become rebates).
- **Fail safe:** wrap each book in try/except so one bad book can never take
  down the whole response.

---

## 3. The three shipped agents

All three live in `agents/`, are self-contained, and share the same robust
plumbing (param parsing, fill-based position tracker, volume-cap guard,
config-driven rounding, per-book exception isolation).

| File | Style | Entry | Edge it harvests | Best regime |
|------|-------|-------|------------------|-------------|
| `MomentumScalperAgent.py` | Directional taker scalper | Market | Short-horizon trend + trade-flow imbalance | Trending books (matches current leaders) |
| `MeanReversionAgent.py` | Contrarian maker scalper | Post-only limit | Over-extension snap-back, earns maker rebates | Range-bound / choppy books |
| `AdaptiveMakerAgent.py` | Two-sided inside-spread maker | Limit both sides | Spread capture + DIS rebates, inventory-flattened | Calm, tight-spread books |

Every agent enforces the **same exit discipline** — that's what protects Kappa:
- Take profit at a small `tp_bps`.
- Hard stop at `sl_bps` (kept tight to keep the cubed-downside tail short).
- Time stop at `max_hold_s` so capital recycles and books stay Kappa-eligible.
- Flatten to flat → realize PnL → become eligible for the next round-trip.

### 3.1 MomentumScalperAgent — the "match the leaders" baseline
Computes a fast signal per book = sign-agreement between (a) very short trend in
recent trade prices and (b) top-of-book + trade-flow imbalance. On a strong
aligned signal it takes a small market position in the trend direction, then
exits on tp/sl/time. This is the closest analogue to the current top cluster
(net directional tilt, high realized-PnL-per-RT, low churn).

### 3.2 MeanReversionAgent — the diversifier
Fades books that have stretched a configurable number of bps away from a short
rolling mean of trade prices (and where flow is exhausting). Enters with
**post-only** limits so it tends to be a *maker* (0% base maker fee / rebates),
improving per-RT PnL. Same strict exits. Uncorrelated with the momentum book →
good for the cross-book consistency requirement if you run a fleet.

### 3.3 AdaptiveMakerAgent — the spread harvester
Quotes both sides just inside the spread with short-expiry GTT orders, skews
quotes by imbalance, and *refuses to post on a side whose maker fee is above a
threshold* (DIS-aware), preferring the rebate side. When a quote fills it
immediately works a closing order at `tp_bps` and arms a stop, keeping inventory
near flat. Lowest directional risk → smoothest Kappa, but needs low latency to
avoid adverse selection.

---

## 4. How to run / test

Local offline test (see `agents/proxy/README.md` for the simulator + proxy):

```bash
# from repo root, with the pyenv 3.10.9 env active
python agents/MomentumScalperAgent.py --port 8901 --agent_id 0 \
  --params quote_notional=1500 tp_bps=12 sl_bps=18 max_hold_s=120 \
           signal_bps=6 imbalance_depth=5

python agents/MeanReversionAgent.py --port 8902 --agent_id 0 \
  --params quote_notional=1500 tp_bps=14 sl_bps=20 max_hold_s=180 \
           stretch_bps=18 mean_window_s=120

python agents/AdaptiveMakerAgent.py --port 8903 --agent_id 0 \
  --params quote_notional=1200 tp_bps=10 sl_bps=16 max_hold_s=150 \
           max_maker_fee=0.0005 quote_expiry_s=5
```

> `--params` are `key=value`; numeric values are auto-parsed to float. **Always
> pass at least one param** (the framework requires a params namespace).

Live: register the UID, point your miner runner at the chosen agent file, and
watch the validator dashboard. Tune from observed Med-Kappa3 / Penalty, not from
local PnL alone.

---

## 5. Tuning checklist (in priority order)

1. **Penalty must be ~0.** If not, your worst books are outliers → widen the
   coverage (lower entry thresholds so more books trade) or tighten stops on the
   books that bleed. Uniformity > peak performance.
2. **Med-Kappa3 up, not 24H volume up.** If volume climbs but Kappa doesn't,
   you're churning — raise `signal_bps` / `stretch_bps` to be more selective.
3. **Keep the downside tail short.** If realized PnL is positive but Kappa is
   weak, your losers are too big → tighten `sl_bps` and/or `max_hold_s`.
4. **Stay < ~60% of the volume cap** per book; the cap is a constraint, not a goal.
5. **Latency:** co-locate near the validator; response delay directly worsens
   taker fills. The maker agent is most latency-sensitive.
6. **Re-read the scoring config weekly.** If `activity_impact` ever turns
   positive, volume becomes a real lever and the maker agent's higher turnover
   gets rewarded — revisit sizing then.

---

## 6. Fleet note (how the current leaderboard is actually built)

The top of the board is a few operators running **near-identical agents across
many co-located UIDs** (see report §3). One robust, consistent agent replicated
across UIDs/books beats one clever agent on a single UID, because the score
rewards *median-across-books consistency*, not a single hero book. Run the same
tuned agent on every UID; optionally split styles (momentum vs reversion vs
maker) across UIDs to decorrelate drawdowns.
