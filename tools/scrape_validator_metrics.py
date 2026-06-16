#!/usr/bin/env python3
"""
Scrape a SN-79 validator's public Prometheus endpoint (port 9001) to accumulate
an unbounded per-agent trade history from the rolling 25-slot `trades` buffer.

The validator only retains ~25 trades per book in the metric at any instant, but
each trade carries a unique `trade_id`, so polling over time + deduping by
trade_id reconstructs a long history.

Usage:
    python scrape_validator_metrics.py --host 84.32.70.8 --port 9001 \
        --agent-id 114 --netuid 79 --interval 5 --out-dir "other agents data"

No auth / cookies / Cloudflare involved: /metrics/* is plain Prometheus text.
"""
import argparse
import csv
import os
import re
import sys
import time
import urllib.request

# Matches:  trades{label="v",...} 1.23e4
_LINE = re.compile(r'^(\w+)\{([^}]*)\}\s+([-+0-9.eEnaN]+)\s*$')
_LABEL = re.compile(r'(\w+)="([^"]*)"')

# trade_gauge_name -> output column
FIELDS = (
    "trade_id", "timestamp", "side", "price", "volume",
    "maker_fee", "taker_fee", "maker_agent_id", "taker_agent_id",
    "aggressing_order_id", "resting_order_id",
)


def fetch(url: str, timeout: float = 8.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def parse_trades(text: str):
    """Return {(book_id, slot): {gauge_name: value_str}} for the `trades` metric."""
    grouped: dict[tuple[str, str], dict[str, str]] = {}
    for line in text.splitlines():
        if not line.startswith("trades{"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, labelstr, value = m.groups()
        if name != "trades":
            continue
        labels = dict(_LABEL.findall(labelstr))
        key = (labels.get("book_id", "?"), labels.get("slot", "?"))
        grouped.setdefault(key, {})[labels.get("trade_gauge_name", "")] = value
    return grouped


def num(v: str):
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=9001)
    ap.add_argument("--agent-id", required=True)
    ap.add_argument("--netuid", default="79")
    ap.add_argument("--interval", type=float, default=5.0, help="poll seconds")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--once", action="store_true", help="single pass then exit")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/metrics/trades"
    target = str(args.agent_id)
    os.makedirs(args.out_dir, exist_ok=True)

    # book_id -> {trade_id -> record}
    seen: dict[str, dict[int, dict]] = {}

    def flush(book_id: str):
        path = os.path.join(args.out_dir, f"{target}_trades_book_{book_id}.csv")
        rows = sorted(seen[book_id].values(), key=lambda r: r.get("timestamp", 0))
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trade_id", "time", "side", "price", "volume", "role", "fee"])
            for r in rows:
                role = "MAKER" if str(r.get("maker_agent_id")) == target else "TAKER"
                ts = r.get("timestamp")
                tstr = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts)) if isinstance(ts, (int, float)) else ts
                fee = r.get("maker_fee") if role == "MAKER" else r.get("taker_fee")
                side = "BUY" if r.get("side") == 0 else "SELL"
                w.writerow([r.get("trade_id"), tstr, side, r.get("price"), r.get("volume"), role, fee])
        return path

    print(f"Polling {url} for agent {target} every {args.interval}s (Ctrl-C to stop)")
    try:
        while True:
            try:
                grouped = parse_trades(fetch(url))
            except Exception as e:  # noqa: BLE001
                print(f"  fetch error: {e}", file=sys.stderr)
                if args.once:
                    return 1
                time.sleep(args.interval)
                continue

            added = 0
            for (book_id, _slot), gauges in grouped.items():
                rec = {fld: num(gauges[fld]) for fld in FIELDS if fld in gauges}
                if str(rec.get("maker_agent_id")) != target and str(rec.get("taker_agent_id")) != target:
                    continue
                tid = rec.get("trade_id")
                if tid is None:
                    continue
                book = seen.setdefault(book_id, {})
                if tid not in book:
                    book[tid] = rec
                    added += 1

            for book_id in seen:
                flush(book_id)
            total = sum(len(b) for b in seen.values())
            print(f"  total={total} (+{added})  books={sorted(seen)}")

            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
