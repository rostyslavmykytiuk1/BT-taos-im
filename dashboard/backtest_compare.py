#!/usr/bin/env python3
"""Offline mid-fill backtest: KappaScoreAgent vs MeanReversionAgent on validator tape.

Uses the same trades.csv path as the dashboard. Not a full matching engine — assumes
fills at the 1s resampled mid — but compares signal quality and Kappa-3 proxy fairly.
"""

from __future__ import annotations

import csv
import math
import statistics as st
from collections import defaultdict, deque
from dataclasses import dataclass

from chart_data import sim_seconds_from_duration
from taos.im.telemetry.paths import data_root

VALIDATOR = "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF"
SIM_ID = "20260528_1007"
UID = "189"


def _tape_path() -> str:
    return str(data_root() / UID / VALIDATOR / SIM_ID / "trades.csv")


def is_buy(side_raw: str) -> bool:
    return str(side_raw).strip() in ("0", "BUY", "buy")


@dataclass
class Bar:
    t: int
    mid: float
    trend_bps: float
    flow: float


def load_books(path: str) -> dict[int, list[Bar]]:
    """Per-book 1s bars: last trade price, intra-second trend, signed flow."""
    per_sec: dict[int, dict[int, list[tuple[float, float, bool]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                b = int(row["bookId"])
                p = float(row["price"])
                q = float(row["quantity"])
                t = sim_seconds_from_duration(row["timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            per_sec[b][t].append((p, q, is_buy(row["side"])))

    books: dict[int, list[Bar]] = {}
    for b, secs in per_sec.items():
        if not secs:
            continue
        lo, hi = min(secs), max(secs)
        bars: list[Bar] = []
        last_mid = secs[lo][0][0]
        for s in range(lo, hi + 1):
            if s not in secs:
                bars.append(Bar(s, last_mid, 0.0, 0.0))
                continue
            pts = secs[s]
            prices = [x[0] for x in pts]
            last_mid = prices[-1]
            first = prices[0]
            trend = (last_mid - first) / first * 1e4 if first > 0 else 0.0
            flow = sum(q if buy else -q for _, q, buy in pts)
            bars.append(Bar(s, last_mid, trend, flow))
        books[b] = bars
    return books


def kappa3_proxy(rets: list[float]) -> float | None:
    if len(rets) < 3:
        return None
    m = sum(rets) / len(rets)
    lpm3 = -sum(min(0.0, r) ** 3 for r in rets) / len(rets)
    if lpm3 <= 0:
        return 5.0 if m > 0 else 0.0
    return m / (lpm3 ** (1 / 3))


def run_mean_reversion(bars: list[Bar]) -> list[float]:
    """MeanReversionAgent constants (incl. grind dump, grind-long, soft short-block)."""
    if len(bars) < 300:
        return []

    MEAN_W, K = 300, 1.5
    MINBAND, MAXBAND = 8.0, 120.0
    TP, SL, HOLD = 12.0, 16.0, 180.0
    CRASH_BPS, CRASH_W, REC_W = 35.0, 20, 300
    GRIND_BPS, GRIND_W = 38.0, 300
    REC_TP, REC_HOLD = 1.8, 2.0
    KNIFE_STEP, KNIFE_DROP, KNIFE_S = 8.0, 20.0, 8
    ROCKET_STEP, ROCKET_RISE, ROCKET_S = 8.0, 20.0, 8
    GRIND_RISE, GRIND_DEV, STRONG_SHORT = 8.0, 18.0, 1.25
    TREND_W, TREND_GATE = 600.0, 25.0
    SHORT_BLOCK_S = 1800
    COOL_S = 30

    prices: deque = deque()
    mids: deque = deque()
    grind: deque = deque()
    ema = 0.0
    pos = 0
    avg = 0.0
    ent = 0
    postc = False
    crash_until = 0
    knife_until = 0
    rocket_until = 0
    short_block_until = 0
    cool = 0
    rts: list[float] = []

    for i, bar in enumerate(bars):
        t, m = bar.t, bar.mid
        prices.append((t, m))
        while prices and prices[0][0] < t - MEAN_W:
            prices.popleft()
        mids.append((t, m))
        while mids and mids[0][0] < t - CRASH_W:
            mids.popleft()
        grind.append((t, m))
        while grind and grind[0][0] < t - GRIND_W:
            grind.popleft()
        ema = m if ema <= 0 else ema + (1 - math.exp(-1 / TREND_W)) * (m - ema)
        if len(prices) < 8:
            continue

        ps = [p for _, p in prices]
        ref = sum(ps) / len(ps)
        var = sum((p - ref) ** 2 for p in ps) / len(ps)
        disp = (math.sqrt(var) / ref) * 1e4 if ref > 0 else 0.0
        band = max(MINBAND, min(MAXBAND, K * disp))

        hi = max(p for _, p in mids)
        lo = min(p for _, p in mids)
        ghi = max(p for _, p in grind)
        glo = min(p for _, p in grind)
        drop = max(0.0, (hi - m) / hi * 1e4) if hi > 0 else 0.0
        gdrop = max(0.0, (ghi - m) / ghi * 1e4) if ghi > 0 else 0.0
        rise = max(0.0, (m - lo) / lo * 1e4) if lo > 0 else 0.0
        grise = max(0.0, (m - glo) / glo * 1e4) if glo > 0 else 0.0
        step = (m - mids[-2][1]) / mids[-2][1] * 1e4 if len(mids) >= 2 else 0.0
        if drop >= CRASH_BPS or gdrop >= GRIND_BPS:
            crash_until = t + REC_W
            short_block_until = t + SHORT_BLOCK_S
        if step <= -KNIFE_STEP and drop >= KNIFE_DROP:
            knife_until = t + KNIFE_S
        if step >= ROCKET_STEP and rise >= ROCKET_RISE:
            rocket_until = t + ROCKET_S

        in_rec = t < crash_until
        knife = t < knife_until
        rocket = t < rocket_until
        short_blocked = t < short_block_until
        trend = (ref - ema) / ema * 1e4 if ema > 0 else 0.0
        up = trend > TREND_GATE
        dn = trend < -TREND_GATE
        dev = (m - ref) / ref * 1e4 if ref > 0 else 0.0
        strong_short = dev >= band * STRONG_SHORT
        grind_short_blocked = m > ema and grise >= GRIND_RISE and not strong_short

        if pos != 0:
            pnl = ((m - avg) if pos > 0 else (avg - m)) / avg * 1e4
            tp_ = TP * (REC_TP if (pos > 0 and postc) else 1.0)
            sl_ = SL * (0.6 if (pos < 0 and in_rec) else 1.0)
            hold_ = HOLD * (REC_HOLD if (pos > 0 and postc) else 1.0)
            if pnl >= tp_ or pnl <= -sl_ or (t - ent) >= hold_:
                rts.append(pnl / 1e4)
                if pnl <= -sl_:
                    cool = t + COOL_S
                pos = 0
                avg = 0.0
                ent = 0
                postc = False
            continue

        if t < cool:
            continue
        if dev <= -band and not knife and (in_rec or not dn):
            pos = 1
            avg = m
            ent = t
            postc = in_rec
        elif in_rec and m > ema and grise >= GRIND_RISE and dev <= GRIND_DEV and not knife:
            pos = 1
            avg = m
            ent = t
            postc = True
        elif (dev >= band and not in_rec and not short_blocked and not rocket
              and not grind_short_blocked and not up):
            pos = -1
            avg = m
            ent = t
            postc = False

    return rts


def run_kappa_score(bars: list[Bar]) -> list[float]:
    """KappaScoreAgent constants: fee-aware TP, no SL (disaster only), long-biased, ping."""
    if len(bars) < 300:
        return []

    MEAN_W, K = 300, 2.0
    MINBAND, MAXBAND = 10.0, 120.0
    MIN_TP, TP_BUF, BE_BUF = 8.0, 4.0, 1.0
    RT_FEE = 2.0 * 2.3  # assumed maker-in + maker-out (bps)
    TP = max(MIN_TP, RT_FEE + TP_BUF)
    BE = RT_FEE + BE_BUF
    HOLD = 240.0
    DISASTER = 150.0
    CRASH_BPS, CRASH_W, REC_W = 35.0, 20, 300
    GRIND_BPS, GRIND_W = 38.0, 300
    REC_TP, REC_HOLD = 1.6, 2.0
    KNIFE_STEP, KNIFE_DROP, KNIFE_S = 8.0, 20.0, 8
    TREND_W, TREND_GATE = 600.0, 25.0
    COOL_S = 30
    PING_S, PING_DEV = 540, 6.0

    prices: deque = deque()
    mids: deque = deque()
    grind: deque = deque()
    ema = 0.0
    pos = 0
    avg = 0.0
    ent = 0
    postc = False
    crash_until = 0
    knife_until = 0
    cool = 0
    last_rt = 0
    rts: list[float] = []

    min_edge = max(MINBAND, TP + TP_BUF)

    for bar in bars:
        t, m = bar.t, bar.mid
        prices.append((t, m))
        while prices and prices[0][0] < t - MEAN_W:
            prices.popleft()
        mids.append((t, m))
        while mids and mids[0][0] < t - CRASH_W:
            mids.popleft()
        grind.append((t, m))
        while grind and grind[0][0] < t - GRIND_W:
            grind.popleft()
        ema = m if ema <= 0 else ema + (1 - math.exp(-1 / TREND_W)) * (m - ema)
        if len(prices) < 8:
            continue

        ps = [p for _, p in prices]
        ref = sum(ps) / len(ps)
        var = sum((p - ref) ** 2 for p in ps) / len(ps)
        disp = (math.sqrt(var) / ref) * 1e4 if ref > 0 else 0.0
        band = max(MINBAND, min(MAXBAND, K * disp))

        hi = max(p for _, p in mids)
        ghi = max(p for _, p in grind)
        drop = max(0.0, (hi - m) / hi * 1e4) if hi > 0 else 0.0
        gdrop = max(0.0, (ghi - m) / ghi * 1e4) if ghi > 0 else 0.0
        step = (m - mids[-2][1]) / mids[-2][1] * 1e4 if len(mids) >= 2 else 0.0
        if drop >= CRASH_BPS or gdrop >= GRIND_BPS:
            crash_until = t + REC_W
        if step <= -KNIFE_STEP and drop >= KNIFE_DROP:
            knife_until = t + KNIFE_S

        in_rec = t < crash_until
        knife = t < knife_until
        trend = (ref - ema) / ema * 1e4 if ema > 0 else 0.0
        dn = trend < -TREND_GATE
        dev = (m - ref) / ref * 1e4 if ref > 0 else 0.0
        edge = max(band, min_edge)

        if pos != 0:
            pnl = ((m - avg) if pos > 0 else (avg - m)) / avg * 1e4
            tp_ = TP * (REC_TP if (pos > 0 and postc) else 1.0)
            hold_ = HOLD * (REC_HOLD if (pos > 0 and postc) else 1.0)
            timed_out = (t - ent) >= hold_

            if pnl <= -DISASTER:
                rts.append(pnl / 1e4)
                last_rt = t
                cool = t + COOL_S
                pos = 0
                avg = 0.0
                ent = 0
                postc = False
            elif timed_out and pnl >= BE:
                rts.append(pnl / 1e4)
                last_rt = t
                pos = 0
                avg = 0.0
                ent = 0
                postc = False
            elif pnl >= tp_:
                rts.append(pnl / 1e4)
                last_rt = t
                pos = 0
                avg = 0.0
                ent = 0
                postc = False
            continue

        if t < cool:
            continue

        need_ping = (last_rt == 0) or ((t - last_rt) >= PING_S)
        fade_long = False
        is_ping = False

        if dev <= -edge and not knife and (in_rec or not dn):
            fade_long = True
        elif need_ping and not knife and not dn and not in_rec and dev <= PING_DEV:
            fade_long = True
            is_ping = True

        if fade_long:
            pos = 1
            avg = m
            ent = t
            postc = in_rec and not is_ping

    return rts


def run_momentum(bars: list[Bar]) -> list[float]:
    """MomentumScalperAgent defaults on 1s bars; trade-flow imbalance proxy for LOB."""
    if len(bars) < 300:
        return []

    SIGNAL_BPS = 9.0
    STRETCH_BPS = 12.0
    MEAN_W = 90
    MIN_SAMPLES = 6
    MIN_FLOW = 0.5 * 0.25  # min_flow_mult * min_order_size
    MIN_IMB = 0.12
    IMB_W = 5  # seconds of trade flow for imbalance proxy
    TP, SL, HOLD = 15.0, 13.0, 90.0
    COOL_S = 45

    prices: deque = deque()
    flow_win: deque = deque()  # (t, signed_qty)
    pos = 0
    avg = 0.0
    ent = 0
    cool = 0
    rts: list[float] = []

    for bar in bars:
        t, m = bar.t, bar.mid
        prices.append((t, m))
        while prices and prices[0][0] < t - MEAN_W:
            prices.popleft()
        flow_win.append((t, bar.flow))
        while flow_win and flow_win[0][0] < t - IMB_W:
            flow_win.popleft()

        if pos != 0:
            pnl = ((m - avg) if pos > 0 else (avg - m)) / avg * 1e4
            if pnl >= TP or pnl <= -SL or (t - ent) >= HOLD:
                rts.append(pnl / 1e4)
                if pnl <= -SL:
                    cool = t + COOL_S
                pos = 0
                avg = 0.0
                ent = 0
            continue

        if t < cool or len(prices) < MIN_SAMPLES:
            continue

        trend = bar.trend_bps
        flow = bar.flow
        if abs(trend) < SIGNAL_BPS or abs(flow) < MIN_FLOW:
            continue

        bq = sum(q for _, q in flow_win if q > 0)
        aq = sum(-q for _, q in flow_win if q < 0)
        denom = bq + aq
        imb = (bq - aq) / denom if denom > 0 else 0.0

        ref = sum(p for _, p in prices) / len(prices)
        stretch_up = ref * (1 + STRETCH_BPS / 1e4)
        stretch_dn = ref * (1 - STRETCH_BPS / 1e4)

        long_ok = trend > 0 and imb >= MIN_IMB and flow >= MIN_FLOW and m <= stretch_up
        short_ok = trend < 0 and imb <= -MIN_IMB and flow <= -MIN_FLOW and m >= stretch_dn

        if long_ok:
            pos = 1
            avg = m
            ent = t
        elif short_ok:
            pos = -1
            avg = m
            ent = t

    return rts


@dataclass
class Summary:
    name: str
    rts: int
    win_pct: float
    mean_bps: float
    med_bps: float
    worst_bps: float
    best_bps: float
    losses_lt_200bps: int
    books_k3: int
    med_k3: float
    rts_per_book: float
    est_quote_vol: float


def summarize(name: str, all_rts: list[float], n_books: int, quote_notional: float) -> Summary:
    wins = sum(1 for r in all_rts if r > 0)
    n = len(all_rts)
    return Summary(
        name=name,
        rts=n,
        win_pct=100.0 * wins / n if n else 0.0,
        mean_bps=st.mean(all_rts) * 1e4 if n else 0.0,
        med_bps=st.median(all_rts) * 1e4 if n else 0.0,
        worst_bps=min(all_rts) * 1e4 if n else 0.0,
        best_bps=max(all_rts) * 1e4 if n else 0.0,
        losses_lt_200bps=sum(1 for r in all_rts if r < -0.02),
        books_k3=0,
        med_k3=0.0,
        rts_per_book=n / n_books if n_books else 0.0,
        est_quote_vol=n * 2 * quote_notional,
    )


def main() -> None:
    path = _tape_path()
    books = load_books(path)
    print(f"Tape: {path}")
    print(f"Books with bars: {len(books)}\n")

    strategies = [
        ("KappaScoreAgent", run_kappa_score, 800.0),
        ("MeanReversionAgent", run_mean_reversion, 1800.0),
        ("MomentumScalperAgent", run_momentum, 2200.0),
    ]

    rows: list[tuple[Summary, list[float]]] = []
    per_book_k: dict[str, list[float]] = {}

    for name, fn, notional in strategies:
        all_rts: list[float] = []
        perk: list[float] = []
        for bars in books.values():
            rts = fn(bars)
            all_rts.extend(rts)
            k = kappa3_proxy(rts)
            if k is not None:
                perk.append(k)
        s = summarize(name, all_rts, len(books), notional)
        s.books_k3 = len(perk)
        s.med_k3 = st.median(perk) if perk else 0.0
        rows.append((s, all_rts))
        per_book_k[name] = perk

    hdr = (
        f"{'Strategy':<22} {'RTs':>6} {'Win%':>6} {'Mean':>8} {'Med':>8} "
        f"{'Worst':>8} {'<-200bp':>7} {'medK3':>7} {'RT/book':>7} {'~Vol':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s, _ in rows:
        print(
            f"{s.name:<22} {s.rts:>6} {s.win_pct:>5.1f}% "
            f"{s.mean_bps:>+7.1f} {s.med_bps:>+7.1f} "
            f"{s.worst_bps:>7.0f} {s.losses_lt_200bps:>7} "
            f"{s.med_k3:>+6.3f} {s.rts_per_book:>7.1f} "
            f"{s.est_quote_vol:>10,.0f}"
        )

    print("\nNotes:")
    print("  • Mid-fill replay on 1s trade bars (not full matching / fees / maker queue).")
    print("  • Momentum uses per-second trend+flow from trades; LOB imb ≈ 5s signed trade imbalance.")
    print("  • ~Vol = round_trips × 2 × quote_notional (entry+exit quote, rough).")
    print("  • medK3 = median per-book Kappa-3 proxy (books with ≥3 RTs).")
    print("  • Top miners on this validator: ~35–45k quote vol / day, ~7–8k RTs, Activity≈1.")


if __name__ == "__main__":
    main()
