#!/usr/bin/env python3
"""
Hourly market regime monitor for SN-79.
Reads miner-1 pm2 logs, infers price trend, recommends agent.

Run once:   python3 tools/market_monitor.py --once
Run hourly: python3 tools/market_monitor.py        (loops, use pm2 to manage)
"""
import argparse
import re
import subprocess
import sys
import time
from datetime import datetime

TOP_AGENTS = {
    "136": "fee-paying taker + net short (downtrend specialist)",
    "126": "rebate scalp taker (sideways / high-rebate books)",
    "149": "tight maker (low-fee maker books)",
}

DOWNTREND_THRESHOLD_BPS = -3.0
UPTREND_THRESHOLD_BPS   =  3.0


def fetch_logs(miner: str = "miner-1", lines: int = 3000) -> str:
    result = subprocess.run(
        ["pm2", "logs", miner, "--lines", str(lines), "--nostream"],
        capture_output=True, text=True
    )
    return result.stdout + result.stderr


def extract_prices(logs: str) -> list[float]:
    """Pull prices from 'BUY/SELL qty@price ON BOOK n' lines."""
    prices = []
    for m in re.finditer(r'(?:BUY|SELL)\s+[\d.]+@([\d.]+)\s+ON BOOK', logs):
        try:
            prices.append(float(m.group(1)))
        except ValueError:
            pass
    return prices


def extract_routing_events(logs: str) -> dict:
    events = {
        "adverse_sel": 0,
        "pnl_backoff": 0,
        "fallback_maker": 0,
        "idle_to_maker": 0,
    }
    for line in logs.splitlines():
        if "[adverse-sel]" in line:
            events["adverse_sel"] += 1
        if "PNL-BACKOFF" in line:
            events["pnl_backoff"] += 1
        if "[fallback-maker]" in line:
            events["fallback_maker"] += 1
        if "idle->maker" in line:
            events["idle_to_maker"] += 1
    return events


def classify_regime(prices: list[float]) -> tuple[str, float]:
    if len(prices) < 30:
        return "UNKNOWN", 0.0
    # Compare first-third avg vs last-third avg
    n = len(prices) // 3
    old_avg = sum(prices[:n]) / n
    new_avg = sum(prices[-n:]) / n
    drift_bps = (new_avg - old_avg) / old_avg * 1e4 if old_avg > 0 else 0.0
    if drift_bps < DOWNTREND_THRESHOLD_BPS:
        regime = "DOWNTREND"
    elif drift_bps > UPTREND_THRESHOLD_BPS:
        regime = "UPTREND"
    else:
        regime = "SIDEWAYS"
    return regime, drift_bps


def recommend(regime: str) -> str:
    if regime == "DOWNTREND":
        return (
            "ApexTakerAgent  — fee-paying taker, directional SHORT lean, positive-skew exit.\n"
            "  (136 wins this regime: kappa from MAD via volume+breadth, not maker edge)"
        )
    elif regime == "UPTREND":
        return (
            "ApexTakerAgent  — same structure but EMA drift flips lean to LONG.\n"
            "  ApexTaker handles uptrend via CH_DRIFT_DIR_BPS lean; no agent swap needed."
        )
    else:
        return (
            "AdaptiveRouterAgent / PureMakerAgent  — maker and rebate-scalp books are profitable.\n"
            "  126-style rebate scalp wins on rebate books; tight maker wins on low-fee books.\n"
            "  ApexTaker REBATE lane still active — no urgent swap but maker adds edge."
        )


def run_once(miner: str = "miner-1") -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs = fetch_logs(miner)
    prices = extract_prices(logs)
    events = extract_routing_events(logs)
    regime, drift_bps = classify_regime(prices)

    print(f"\n{'='*60}")
    print(f"[{ts}]  MARKET REGIME: {regime}  ({drift_bps:+.1f} bps drift)")
    print(f"{'='*60}")
    print(f"  Price samples : {len(prices)}")
    if prices:
        print(f"  Price range   : {min(prices):.2f} – {max(prices):.2f}")
        print(f"  Current       : {prices[-1]:.2f}")
    print(f"\n  Routing events (last {len(logs.splitlines())} log lines):")
    print(f"    [adverse-sel]   : {events['adverse_sel']:3d}  (slow-bleed → taker)")
    print(f"    PNL-BACKOFF     : {events['pnl_backoff']:3d}  (fast-bleed → idle)")
    print(f"    [fallback-maker]: {events['fallback_maker']:3d}")
    print(f"    idle→maker      : {events['idle_to_maker']:3d}  (recovery)")
    print(f"\n  RECOMMENDATION:")
    for line in recommend(regime).splitlines():
        print(f"    {line}")
    print()
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--miner", default="miner-1", help="pm2 process name")
    parser.add_argument("--interval", type=int, default=3600, help="Interval in seconds")
    args = parser.parse_args()

    if args.once:
        run_once(args.miner)
        return

    print(f"Market monitor started — checking every {args.interval}s. Ctrl-C to stop.")
    while True:
        run_once(args.miner)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
