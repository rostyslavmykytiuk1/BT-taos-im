# Subnet 79 (MVTRX) — Adaptive Trading Agent Report

A practical field report on building competitive trading agents for Bittensor Subnet 79.
Written from hands-on iteration (TakerScalper, DualEdge, KappaMaker) plus reverse-engineering
of the validator scoring code and analysis of top competitors' live trades.

---

## 1. The Subnet Situation

### 1.1 What miners are scored on

Subnet 79 runs an agent-based limit-order-book **simulation** (the τaos C++ simulator). Each
miner is a trading agent that receives market state updates per book and replies with order
instructions. The validator runs a faithful FIFO accounting of every fill and scores miners on
the **quality and profitability of completed round-trips**.

The final trading score is a fixed blend (from `taos/im/config/__init__.py`):

```
trading_score = 0.79 * kappa_score + 0.21 * pnl_score
```

- `kappa_score` (weight **0.79**) — risk-adjusted round-trip quality, the dominant term.
- `pnl_score`  (weight **0.21**) — absolute realized profitability, normalized to a daily return.

Both are computed **per book** and then aggregated by **median** across books. This median
aggregation is the single most important structural fact about the subnet — see §1.5.

### 1.2 Kappa-3: the dominant scoring term

Kappa-3 measures risk-adjusted realized PnL on completed round-trips. Per book:

1. Collect the series of **realized round-trip PnLs** over the lookback window (3 hours;
   `kappa.lookback = 10800e9 ns`). A book needs at least **3** round-trips
   (`kappa.min_realized_observations = 3`) to be scored at all.
2. Normalize the series by its **MAD** (median absolute deviation): `returns = pnl / MAD`.
3. Compute, with `tau = 0`:
   - `mean_r` = mean of normalized returns
   - `LPM3`   = mean of `max(tau - r, 0)^3`  → the **cubed downside**
   - `UPM3`   = mean of `max(r - tau, 0)^3`  → the cubed upside
4. Kappa-3 ≈ `(mean_r - tau) / cbrt(LPM3 + regularizer)`.
   When downside is negligible and mean is positive, it falls back to a UPM3-based branch.

The raw kappa is then clamped/normalized to roughly `[0, 1]`
(`normalization_min = -2.5`, `normalization_max = 2.5`).

**The critical property: downside is CUBED.** A single losing round-trip contributes to `LPM3`
by the cube of its (MAD-normalized) magnitude, while a winner only helps the numerator roughly
linearly. Consequences that drive every design decision:

- **One big loss is catastrophic.** A −15 round-trip on a book where typical PnLs are ±0.05
  can collapse that book's kappa to ~0 by itself. Many tiny losses are far cheaper than one
  large one.
- **MAD normalization is a double-edged sword.** On a book where you trade tightly (small,
  consistent PnLs), MAD is small, so any loss looks *huge* in normalized units and is punished
  cubically. On a book with large MAD (high-variance, high-volume trading), the same absolute
  loss is normalized down and barely dents kappa. This is precisely why high-volume traders can
  "absorb" losses that would destroy a low-volume agent (see §2 and §1.5).

### 1.3 Activity factor: a gate, not a lever

The activity factor multiplies a book's normalized kappa, ranging 0–2x. Two facts dominate:

- `activity.impact = 0.0`. This means trading **more** than the threshold gives **zero** extra
  boost. The formula `1 + (volume/volume_cap) * impact` collapses to exactly **1.0** for any book
  with at least one counted round-trip in the sampling window.
- The sampling window is **600 s** (`trade_volume_sampling_interval`), and the decay grace
  period is also **600 s**. If a book produces **no closing round-trip within a window**, its
  activity factor begins an accelerating exponential **decay** below 1.0 — which directly scales
  down that book's kappa contribution.

**Practical rule:** every book you touch must close **≥ 1 round-trip every ~570 s, forever**, or
it bleeds score through decay. But there is **no reward for volume beyond that** through the
activity channel. Volume only helps via the PnL term (§1.4), and only if it is profitable.

Round-trip volume is produced **only by a closing trade** (a fill that reduces opposing
inventory). Opening trades and resting orders do nothing for activity. This is why every agent
here carries an explicit **activity backstop** that forces a small closing round-trip before the
window lapses.

### 1.4 The PnL term

`pnl_score` (weight 0.21) is the **median per-book daily-return**, mapped to `[-0.5, 0.5]`. It is
computed from realized PnL over the same 3-hour lookback, scaled to a 24h-equivalent return
against per-book allocated capital. Notes:

- It is **realized** PnL only — open positions don't count.
- `kappa.pnl.impact = 0.0` by default, so the *separate* per-book "PnL factor" that can multiply
  kappa is currently **neutral (1.0)**. The 0.21-weighted `pnl_score` is the live PnL channel.
- Because it is also a **median across books**, a few very profitable books do **not** rescue a
  portfolio of break-even/losing books.

### 1.5 FIFO accounting and why it is symmetric

The validator matches every fill against the **oldest open opposing lot** (`_match_trade_fifo`),
exactly like a real exchange. Realized PnL = price difference on the matched quantity **minus
both legs' fees**. Key implications:

- **FIFO, not VWAP.** To guarantee a positive realized round-trip, an exit must clear the
  **oldest** lot's price + fees — not the average. Agents that want no-loss exits must track FIFO
  lots locally and price exits off the worst (oldest) lot.
- **Shorts are first-class.** Selling without prior inventory opens a SHORT; buying back closes
  it and is scored identically to a long round-trip. Two-sided market-making is fully supported.
- **Round-trip volume = closing volume.** Only the matched (closing) quantity counts toward the
  activity sampling, never the opening leg alone.

### 1.6 The realized-PnL rolling window (why cumulative PnL "flatlines")

A subtle but important operational fact: the dashboard's realized PnL is **not a forever
cumulative sum** — it is the PnL realized **within the rolling 3-hour lookback**. As time
advances, older profits roll **out** of the window. So a miner can look like its realized PnL
"stopped increasing" or even fell, while its lifetime trades were strongly positive — the good
early hours simply aged out of the 3h window while a weaker recent hour replaced them.

When diagnosing "PnL stopped going up," always separate:
1. **Window roll-off** (good past hours leaving the 3h window), versus
2. **Genuine recent degradation** (the last hour is actually losing).
Both can happen at once.

### 1.7 Dynamic fees and the changing maker/taker regime

Fees are **not fixed**. The simulator sets maker and taker fees **per book** based on the book's
recent passive/aggressive volume ratio. The agent reads them live from
`account.fees.maker_fee_rate` and `account.fees.taker_fee_rate`. Crucially:

- When a book is **starved of passive liquidity**, the exchange **pays takers a rebate**
  (`taker_fee_rate < 0`). Crossing the spread can then be **+EV** even before any price move,
  because you are *paid* to take.
- When a book is **flooded with aggressive flow**, taking is **expensive** and making (posting
  passive quotes) earns the spread + maker rebate.

**This is the heart of the "sometimes maker, sometimes taker" situation.** No single style wins
on every book at every moment:

- On **rebate-taker** books, a pure-taker scalper that opens and closes fast collects the rebate
  on both legs and barely cares about small adverse price moves — the rebate cushions stop-losses
  into net wins.
- On **maker-favorable** books, taking bleeds fees; the edge is in posting passive quotes and
  capturing spread — but only if you can avoid **adverse selection** (your passive quote gets
  filled precisely when the market is about to run through it).
- The regime on a given book **shifts over time**, so an agent ideally **detects the live fee
  regime per book and routes its behavior accordingly**, while never letting either style realize
  a large loss (because of the cubic kappa penalty).

### 1.8 What actually matters to win (summary)

1. **Never take a large loss.** The cubic downside means one −15 round-trip outweighs hundreds
   of small wins. Bounded, fast stop-losses are mandatory.
2. **Hold activity = 1.0 on every book** with ≥1 round-trip per ~570 s. There is no prize for
   more volume through the activity channel.
3. **Be profitable per book, measured by the median.** Consistency across books beats brilliance
   on a few. A handful of bleeding books drags the median down.
4. **Route by the live fee regime.** Take when paid a rebate; make when paid the spread; idle
   (minimum activity) when neither is +EV — rather than forcing trades that lose.
5. **Volume only helps through PnL, and only if it's profitable.** High volume also inflates MAD,
   which *cushions* the cubic penalty — this is the high-volume takers' real edge (§2).

---

## 2. How the Top Miners Win (Maker and Taker)

### 2.1 Top taker archetype (reference: `126_trades_top_taker.csv`)

This is **one** of the strong takers (not necessarily the best), but it scores **above** our
TakerScalper, so its trade log is a useful reference. Quantitative analysis of its 5,217 trades:

| Metric | Value | Reading |
|---|---|---|
| Role | **100% TAKER** | Pure taker; never posts passive quotes |
| Rebate fills | **99.9%** negative fees | It only takes when **paid a rebate** |
| Median fee | **−9.8 bps** | Deep rebate per leg — both legs earn |
| Clip size | **0.25–0.30** base | Small, uniform clips |
| Books traded | **128** | Extremely broad — nearly every book |
| Trades / book (median) | **42** | High recycling per book |
| Same-side run length (median) | **2** (max 6) | **Builds small inventory** — averages in 2–3 clips before flipping |
| Peak abs inventory / book | **~1.0 lot median, ~2 p90** | Holds modest directional inventory, not strictly flat |
| Inter-trade gap (median) | **~1 s** | Very fast cycling |

**What makes it score well:**

1. **Rebate-first selection.** 99.9% of its fills are rebate fills. It simply does **not take**
   on books where taking costs fees. The rebate (~9.8 bps/leg, ~19.6 bps round-trip) means a
   round-trip is profitable even if the price drifts a few bps against it.
2. **Breadth.** Trading 128 books means its **median** per-book kappa and PnL are computed over a
   wide, well-populated base. Activity is trivially 1.0 everywhere.
3. **High volume → large MAD → cushioned downside.** Because it trades a lot per book with
   meaningful clip sizes, each book's PnL series has a **large MAD**. The occasional losing
   round-trip, normalized by that large MAD, is small in normalized units, so the **cubic
   downside penalty barely bites**. This is the structural advantage high-volume takers hold over
   a low-volume agent: *they earn the right to take losses cheaply.*
4. **Modest inventory building.** It is not strictly one-in-one-out. It stacks 2–3 same-side
   clips (median run = 2), letting it press a directional rebate edge while keeping peak inventory
   ~1–2 lots so no single unwind is large.

**The takeaway for beating it:** match its rebate discipline, but **trade more books and recycle
faster** (more profitable rebate round-trips → higher PnL term), and allow **small bounded
inventory building** rather than strict single-clip flipping — without ever letting inventory
grow enough to risk a large unwind.

### 2.2 Top maker archetype (reference: prior UID 165 analysis)

The strongest makers are **two-sided market makers**, not no-loss long-only quoters:

- Predominantly **maker** fills, posting passive bids and asks around a reservation price.
- They **do** hold inventory and **do** take losses — roughly **59% of round-trips are small
  losses** (median loss ~2.8 bps) balanced by fewer larger wins; long holds (median ~177 s).
- The reason they survive the cubic penalty is again **MAD cushioning**: a very high round-trip
  count across many books builds a large MAD per book, diluting each small loss's normalized
  magnitude.

**The trap (what we learned the hard way):** a naive "no-loss maker" that refuses to ever realize
a loss is **fragile**, not safe. It accumulates underwater inventory that either (a) must be
force-dumped by the activity backstop (a concentrated loss), or (b) sits as a low-MAD book where
the *few* forced losses are punished cubically. The top makers' apparent "messiness" (many tiny
losses) is actually what **protects** them — it keeps MAD healthy and avoids the single big dump.

### 2.3 The synthesis

| | Top taker | Top maker |
|---|---|---|
| When it wins | Book pays taker rebate | Book pays maker spread/rebate, mean-reverting |
| Core edge | Rebate on both legs | Captured spread |
| Inventory | Small, bounded (~1–2 lots) | Two-sided, mean-reverting, bounded |
| Loss handling | Rebate cushions fast stops | Many tiny losses, large MAD cushions them |
| Failure mode | Rebate shrinks → stops go net-negative | Adverse selection on trending books |
| Activity | Trivial (fast cycling) | Must force a close each window |

A genuinely top-ranked agent needs **both** playbooks and must **route by live fee regime per
book**, because the regime changes over time and across books. That is the thesis behind
DualEdge (§3.2) — which is sound in principle but not yet working in practice.

---

## 3. What We Tried (and What Each Taught Us)

Three agents, each a different point in the design space. For each: the idea, how it behaved in
which market regime, and the concrete mistakes to **not repeat**.

### 3.1 TakerScalperAgent — pure-taker rebate scalper ✅ (best so far, still not top)

**File:** `agents/TakerScalperAgent.py`

**Idea.** Market orders only. One position per book, strictly sequential (flat → open → close →
flat). Open a small directional clip (`LOT = 0.3`) in the microprice-bias direction **only when
the book pays a taker rebate** and a Kappa-3 projection gate says the trade won't hurt the book's
kappa. Close on **TP +2.5 bps / SL −4 bps / max-hold 4 s**. An activity backstop forces a
round-trip if a book is about to lapse the ~570 s window. RT logging gated to the main validator.

Key parameters: `MIN_HOLD 1.5s`, `MAX_HOLD 4s`, `TP 2.5bps`, `SL 4bps`, `RT_MAX 20/window`,
`MIN_REOPEN_GAP 4s`, `KAPPA_MIN_REBATE_BPS 1.0`, kappa projection gate.

**How it behaved — rebate-rich market (good):**
- Win rate **85–91%**, strongly positive PnL (e.g. +31/hour in one observed window).
- Most exits are **`close=sl`** by tag, but they are **net positive** because the taker rebate on
  both legs (~9–12 bps) exceeds the ~4 bps gross stop. This is the whole trick: a "stop-loss" on a
  deep-rebate book is still a net win.
- Activity 1.0 everywhere, kappa steadily positive. Ranked at/near the top **of our fleet**.

**How it behaved — rebate degrades (bad):**
- Observed UID 119 (miner-5) decay over ~2 hours: win rate **90% → 35%**, hourly PnL **+31 → −7**.
- Mechanism: rebate dipped (~10.5 → ~8.5 bps) **and** gross stop losses widened slightly. SL
  exits that were **+0.047 net (91% positive)** became **−0.020 net (37% positive)**. The rebate
  cushion thinned below the gross loss, so the *same* strategy started bleeding.
- ~95% of closes were `close=sl`, almost **no take-profits** — in a trending/adverse window the
  price rarely reaches +2.5 bps before the 4 s clock or −4 bps stop.

**Why it's still not top.** Higher-volume takers (the §2.1 archetype) beat it by:
- trading **far more books** (128 vs our subset),
- **building small inventory** (2–3 clip runs) instead of strict one-in-one-out,
- recycling faster — more profitable rebate round-trips → higher PnL term and larger MAD cushion.

**Mistakes to avoid / lessons:**
- **Do not treat `close=sl` count as a problem by itself.** On rebate books, net-positive stops
  are the expected, healthy behavior. Judge by **net PnL**, not close-reason tags.
- **The edge is the rebate, not the prediction.** When rebate shrinks, the agent must trade
  **less** (tighter gate), not more. A fixed strategy silently flips from +EV to −EV as fees move.
- **`RT_MAX = 20` is a real ceiling on hot books** (we observed `close_rt_n` pinned at 19). It
  caps PnL on exactly the best books. Raising it (e.g. 28) + tightening reopen gap (4s → 3s) is
  the clean way to add profitable volume — **but only on rebate books**, never by loosening the
  kappa/rebate gate.
- **Don't loosen the kappa projection gate to get volume.** That is how a sibling miner (UID 47)
  ran much higher volume but **worse** PnL/kappa than UID 119 — marginal entries become net-loss
  stops that the cubic penalty punishes.

### 3.2 DualEdgeAgent — dual-mode router ❌ (failed so far, needs major work)

**File:** `agents/DualEdgeAgent.py`

**Idea.** One agent that **routes per book by live fee regime**: run the proven KappaMaker maker
engine by default, and switch a book to a TakerScalper-style taker clip **only when the taker fee
is a deep rebate** (rebate ≥ half-spread + margin, and ≥ 2.5 bps). Both modes share **one
validator-faithful FIFO inventory**; a mode switch commits **only when flat** so a held position
is always closed by the engine that opened it.

**How it behaved — live test on miner-1 (the test bed):** mixed-to-bad.

| Mode | RTs | Net PnL | Avg/RT | Loss % | Worst RT |
|---|---|---|---|---|---|
| **taker** | 175 | **+1.00** | +0.0057 | 36% | −0.168 |
| **maker** | 964 | **−9.68** | −0.0100 | 71% | −0.210 |
| **total** | 1139 | **−8.67** | −0.0076 | 66% | −0.210 |

**What worked:**
- **The catastrophe is gone.** Worst single round-trip went from **−15.2** (an earlier DualEdge
  build) to **−0.21**. The capped managed-exit (from the KappaMaker engine) bounds every loss.
- **Taker mode is profitable** (+1.0, 36% loss). The TakerScalper port behaves as designed when
  it only engages on deep-rebate books (switch fees median −9.6 bps).

**What failed:**
- **Maker mode bleeds: −9.68 net, 71% loss rate.** Passive spread-capture is **−EV** on these
  books: the reduce quotes don't fill at profit (the market moves away), then the managed-exit /
  activity backstop force many small losses. The losses are now *bounded* (no −15) but still
  *net-negative in aggregate* — death by a thousand cuts, dragging the per-book kappa median down.
- **Mode flip-flop.** ~2005 mode switches in 52 minutes because many books' taker fee hovers right
  at the rebate boundary. Switches only happen when flat (so not directly harmful), but it
  indicates the router has no **hysteresis/dwell** and is indecisive at the boundary.

**Root cause.** The maker mode inherited KappaMaker's assumption that **active spread-capture is
+EV**. Live, on the books DualEdge was routing to maker, it is **not** — adverse selection plus
maker fees being **costs** (not rebates) make passive quoting lose. DualEdge correctly *bounds*
the loss but does not *avoid* it.

**Mistakes to avoid / lessons:**
- **Don't default to active making.** Making is only +EV when the book actually pays for it
  (maker rebate or wide enough spread to clear fees + adverse-selection margin). The default for a
  non-rebate, non-wide book should be **minimum activity only** (1 forced RT/window) and otherwise
  **flat** — convert "actively bleeding" books into "roughly flat" books.
- **A bounded loss is still a loss.** Fixing the −15 catastrophe (essential) did **not** make the
  maker profitable. Bounding losses and *avoiding* losing trades are different problems; DualEdge
  solved the first, not the second.
- **Add router hysteresis.** A book should not be allowed to change modes more than once per N
  seconds; require the fee regime to clear the threshold by a margin and persist before switching.
- **Backtests lied.** An offline backtest of the two-sided maker predicted **+0.88 median kappa**;
  live it was deeply negative. The offline fill model assumed fills at quoted prices and
  mean-reversion; live reality is **adverse selection** (you get filled precisely when the market
  runs through your quote) and **trends**. *Never trust a maker backtest that fills passively at
  the quote.*

### 3.3 KappaMakerAgent — maker-primary FIFO no-loss engine ⚠️ (weak in maker-good markets)

**File:** `agents/KappaMakerAgent.py`

**Idea.** A complete, validator-faithful maker. Quote both sides inside the touch when flat; when
holding, work **only the reducing side** (never average into the bag), pricing the reduce off the
**FIFO worst lot** so every consumed lot is round-trip-positive, walking the price from the profit
target toward the touch as the lot ages. A **capped-slippage managed exit** IOC-cuts a lot that
is too old or beyond a hard stop (bounds each loss). An activity floor forces a close each window.
Risk guard trims inventory breaches.

This engine is genuinely well-built and is what **fixed DualEdge's catastrophe** — its loss
bounding works. But:

**How it behaved — maker-favorable market (disappointing):**
- Score was **not good** even when the market "should" suit a maker. The same mechanism that hurt
  DualEdge's maker mode applies: on books with **adverse selection**, passive quotes get filled
  adversely, the no-loss reduce often **can't fill at profit**, and the agent ends up taking many
  small managed-exit losses. On low-MAD books, those few losses are **cubically punished**.
- Its design embodies the **"no-loss maker" assumption** that proved fragile (§2.2): refusing to
  realize losses doesn't prevent them — it defers them into forced exits and starves MAD.

**Mistakes to avoid / lessons:**
- **"No-loss" passive making is a myth under adverse selection.** A maker that only wants
  FIFO-positive exits will, on a trending or toxic book, simply fail to exit at profit and then be
  forced out at a loss anyway — now concentrated and on a low-MAD book (worst case for kappa).
- **Reuse its loss-bounding, not its profit thesis.** The capped managed-exit and FIFO lot
  tracking are excellent and worth keeping in any agent. The *premise* that posting passive quotes
  earns money on arbitrary books is what fails.
- **It needs a regime filter.** KappaMaker would be much stronger if it **only quoted on books
  that pay the maker** (positive maker rebate or spread > fees + adverse-selection margin) and did
  minimum-activity-only elsewhere — the same fix DualEdge's maker mode needs.

### 3.4 Cross-cutting lessons (true for every agent here)

1. **Edge comes from the fee regime, not from price prediction.** Every profitable behavior we
   observed (ours and competitors') is fundamentally **getting paid** to trade — taker rebate or
   maker spread/rebate — with price moves as noise the fee must cover. When the fee stops paying,
   *stop trading that book*, don't trade harder.
2. **The cubic downside dictates everything.** Bounded, small losses always; never one big loss;
   prefer high volume (large MAD) on books you're profitable on, which further cushions the rare
   loss.
3. **Median across books punishes bleeding books.** It is better to be **flat** on a hard book
   than to actively lose small amounts on it — a flat book (minimum activity) contributes a
   neutral kappa, a bleeding book pulls the median down.
4. **Activity is a floor to maintain, not a lever to pull.** `impact = 0.0`. Hit 1 RT/window per
   book and stop; spend all other effort on PnL quality.
5. **Live ≠ backtest for makers.** Passive-fill backtests are systematically optimistic because
   they ignore adverse selection. Taker backtests are more trustworthy (market orders fill at the
   touch, as live).

---

## 4. Where To Go Next (open recommendations)

- **TakerScalper:** make the rebate/kappa gate **adaptive to the live rebate level** so it trades
  more when rebates are deep and backs off as they thin (the UID 119 decay was a fixed strategy
  meeting a thinning rebate). Consider trading **more books** and allowing **bounded inventory
  building** (2–3 clips) to approach the §2.1 archetype, plus `RT_MAX 28` + `reopen 3s` on
  rebate-rich books.
- **DualEdge:** change the maker mode from "active spread-capture by default" to **"make only when
  the book pays the maker; otherwise minimum activity + flat."** Add router **hysteresis**. This
  alone should move maker books from −9.7 toward break-even and let the profitable taker mode carry
  the score.
- **KappaMaker:** add the same **maker-regime filter**; keep its loss-bounding engine, drop its
  default-quote-everywhere premise.
- **General:** treat **miner-1 as the live test bed** (its score is not cared about) and validate
  every change there against **net PnL split by mode** and **worst-RT** before fleet rollout.

---

*Report generated from hands-on iteration and validator source analysis. The numbers cited are
from live `pm2` logs (UID 119 / miner-5, miner-1 DualEdge test) and competitor trade exports in
`other agents data/`. Re-run the per-mode PnL and rolling-3h-window analyses periodically — the
fee regime shifts, and a strategy that is +EV today can quietly turn −EV as rebates move.*
