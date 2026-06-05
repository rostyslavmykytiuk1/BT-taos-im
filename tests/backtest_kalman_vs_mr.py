#!/usr/bin/env python3
"""Offline mid-fill backtest: KalmanMomentumAgent vs MeanReversionAgent (v2).

Replays validator trades.csv on 3-second bars (matches candle_s=3 live agents).
Assumes fills at bar mid; no fees, maker queue, or activity ping.
"""

from __future__ import annotations

import csv
import math
import statistics as st
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "dashboard"))

from chart_data import sim_seconds_from_duration
from agents.KalmanMomentumAgent import _KalmanFair
from taos.im.telemetry.paths import data_root

CANDLE_S = 3
UID = "189"

# Default tapes (newest sim with full book coverage first)
DEFAULT_TAPES = [
    ("5DaBmjuw8WNi8kb1MU8Fz6cguX9GjM8DWHiu1d9btfMW4r5h", "20260604_1512"),
    ("5GKj3UR5WchME5QKgzWY3j383qaYz3WuvHrhxS1t9MzF8s79", "20260604_0751"),
    ("5C7TLbDb1BG9MXedF9wephtmw8E5vt9rAcyuhZ5sDkERsn79", "20260605_1210"),
]


def is_buy(side_raw: str) -> bool:
    return str(side_raw).strip() in ("0", "BUY", "buy")


@dataclass
class Bar:
    t: int          # sim seconds (end of 3s bucket)
    mid: float
    flow: float     # signed base qty in bucket (imbalance proxy)


def load_books_3s(path: str, candle_s: int = CANDLE_S) -> dict[int, list[Bar]]:
    """Per-book 3s candles from trade tape (last price + signed flow per bucket)."""
    per_bucket: dict[int, dict[int, list[tuple[float, float, bool]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                b = int(row["bookId"])
                p = float(row["price"])
                q = float(row["quantity"])
                sec = sim_seconds_from_duration(row["timestamp"])
                bucket = sec // candle_s
            except (KeyError, TypeError, ValueError):
                continue
            per_bucket[b][bucket].append((p, q, is_buy(row["side"])))

    books: dict[int, list[Bar]] = {}
    for b, buckets in per_bucket.items():
        if not buckets:
            continue
        lo, hi = min(buckets), max(buckets)
        bars: list[Bar] = []
        last_mid = buckets[lo][0][0]
        for k in range(lo, hi + 1):
            t = k * candle_s
            if k not in buckets:
                bars.append(Bar(t, last_mid, 0.0))
                continue
            pts = buckets[k]
            prices = [x[0] for x in pts]
            last_mid = prices[-1]
            flow = sum(q if buy else -q for _, q, buy in pts)
            bars.append(Bar(t, last_mid, flow))
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


def run_mean_reversion_v2(bars: list[Bar]) -> list[float]:
    """MeanReversionAgent v2 defaults (deployed miner-1 params)."""
    if len(bars) < 40:
        return []

    MEAN_W, K = 120, 1.4
    MINBAND, MAXBAND = 8.0, 120.0
    TP, SL, HOLD = 12.0, 16.0, 150.0
    CRASH_BPS, CRASH_W, REC_W = 35.0, 20, 300
    REC_TP, REC_HOLD = 1.8, 2.0
    KNIFE_BPS, KNIFE_S = 8.0, 8
    TREND_W, TREND_GATE = 600.0, 25.0
    SHORT_BLOCK_S = 1800
    COOL_S = 30

    prices: deque = deque()
    mids: deque = deque()
    ema = 0.0
    pos = 0
    avg = 0.0
    ent = 0
    postc = False
    crash_until = 0
    knife_until = 0
    short_block_until = 0
    cool = 0
    rts: list[float] = []

    for bar in bars:
        t, m = bar.t, bar.mid
        prices.append((t, m))
        while prices and prices[0][0] < t - MEAN_W:
            prices.popleft()
        mids.append((t, m))
        while mids and mids[0][0] < t - CRASH_W:
            mids.popleft()
        ema = m if ema <= 0 else ema + (1 - math.exp(-1 / TREND_W)) * (m - ema)
        if len(prices) < 8:
            continue

        ps = [p for _, p in prices]
        ref = sum(ps) / len(ps)
        var = sum((p - ref) ** 2 for p in ps) / len(ps)
        disp = (math.sqrt(var) / ref) * 1e4 if ref > 0 else 0.0
        band = max(MINBAND, min(MAXBAND, K * disp))

        hi = max(p for _, p in mids)
        drop = max(0.0, (hi - m) / hi * 1e4) if hi > 0 else 0.0
        step = (m - mids[-2][1]) / mids[-2][1] * 1e4 if len(mids) >= 2 else 0.0
        if drop >= CRASH_BPS:
            crash_until = t + REC_W
            short_block_until = t + SHORT_BLOCK_S
        if step <= -KNIFE_BPS:
            knife_until = t + KNIFE_S

        in_rec = t < crash_until
        knife = t < knife_until
        short_blocked = t < short_block_until
        trend = (ref - ema) / ema * 1e4 if ema > 0 else 0.0
        up = trend > TREND_GATE
        dn = trend < -TREND_GATE
        dev = (m - ref) / ref * 1e4 if ref > 0 else 0.0

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
        elif dev >= band and not in_rec and not short_blocked and not up:
            pos = -1
            avg = m
            ent = t
            postc = False

    return rts


def run_kalman_momentum(bars: list[Bar]) -> list[float]:
    """KalmanMomentumAgent: slope entry, exit on slope sign flip."""
    if len(bars) < 20:
        return []

    SLOPE_GATE = 10.0
    WARMUP = 8

    kf = _KalmanFair(dt=CANDLE_S, process_var=1e-5, meas_var=1e-3)
    steps = 0
    pos = 0
    avg = 0.0
    rts: list[float] = []

    for bar in bars:
        m = bar.mid
        fair = m
        level, slope, _ = kf.update(fair)
        steps += 1
        slope_bps = slope / level * 1e4 if level > 0 else 0.0

        if pos != 0:
            exit_ = (pos > 0 and slope_bps < 0) or (pos < 0 and slope_bps > 0)
            if exit_:
                pnl = ((m - avg) if pos > 0 else (avg - m)) / avg * 1e4
                rts.append(pnl / 1e4)
                pos = 0
                avg = 0.0
            continue

        if steps < WARMUP:
            continue

        gate = SLOPE_GATE
        if slope_bps >= gate and fair >= level:
            pos = 1
            avg = m
        elif slope_bps <= -gate and fair <= level:
            pos = -1
            avg = m

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


def summarize(name: str, all_rts: list[float], n_books: int) -> Summary:
    n = len(all_rts)
    wins = sum(1 for r in all_rts if r > 0)
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
    )


def run_tape(validator: str, sim_id: str) -> None:
    path = data_root() / UID / validator / sim_id / "trades.csv"
    if not path.is_file():
        print(f"SKIP missing: {path}")
        return

    books = load_books_3s(str(path))
    if not books:
        print(f"SKIP empty: {path}")
        return

    durations = [b[-1].t - b[0].t for b in books.values() if len(b) > 1]
    print(f"\n{'=' * 72}")
    print(f"Tape: {sim_id}  validator: {validator[:12]}...")
    print(f"Path: {path}")
    print(f"Books: {len(books)}  |  median duration: {st.median(durations) / 60:.1f} min")

    strategies = [
        ("MeanReversionAgent v2", run_mean_reversion_v2),
        ("KalmanMomentumAgent", run_kalman_momentum),
    ]

    hdr = (
        f"{'Strategy':<24} {'RTs':>6} {'Win%':>6} {'Mean':>8} {'Med':>8} "
        f"{'Worst':>8} {'<-200bp':>7} {'medK3':>7} {'RT/book':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    for name, fn in strategies:
        all_rts: list[float] = []
        perk: list[float] = []
        for bars in books.values():
            rts = fn(bars)
            all_rts.extend(rts)
            k = kappa3_proxy(rts)
            if k is not None:
                perk.append(k)
        s = summarize(name, all_rts, len(books))
        s.books_k3 = len(perk)
        s.med_k3 = st.median(perk) if perk else 0.0
        print(
            f"{s.name:<24} {s.rts:>6} {s.win_pct:>5.1f}% "
            f"{s.mean_bps:>+7.1f} {s.med_bps:>+7.1f} "
            f"{s.worst_bps:>7.0f} {s.losses_lt_200bps:>7} "
            f"{s.med_k3:>+6.3f} {s.rts_per_book:>7.1f}"
        )


def main() -> None:
    print("Backtest: KalmanMomentum vs MeanReversion v2")
    print(f"Bar size: {CANDLE_S}s  |  UID tape: {UID}  |  mid-fill, no ping/fees")
    for validator, sim_id in DEFAULT_TAPES:
        run_tape(validator, sim_id)
    print("\nNotes:")
    print("  • 3s bars from validator trades.csv (all agents' prints, not miner-only).")
    print("  • Kalman exits when slope_bps crosses sign (long while <0, short while >0).")
    print("  • medK3 = median per-book Kappa-3 proxy (books with ≥3 RTs).")


if __name__ == "__main__":
    main()
