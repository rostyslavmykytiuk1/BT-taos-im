#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Read-only API + static UI for miner telemetry."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from taos.im.telemetry.paths import data_root, slug_validator, telemetry_root
from taos.im.utils import duration_from_timestamp, timestamp_from_duration

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="τaos Miner Telemetry", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _connect(db: Path) -> sqlite3.Connection:
    if not db.is_file():
        raise HTTPException(404, f"Database not found: {db}")
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def _db_path(uid: int, validator_slug: str, sim_id: str) -> Path:
    return telemetry_root() / str(uid) / validator_slug / sim_id / "telemetry.sqlite"


def _row(r: sqlite3.Row | None) -> dict[str, Any]:
    return dict(r) if r else {}


def _fnum(val: Any) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _is_buy_side(side_raw: Any) -> bool:
    return str(side_raw) in ("0", "BUY", "buy")


def _chart_time(duration: str) -> int:
    return timestamp_from_duration(duration.strip()) // 1_000_000_000


def _chart_time_label(sim_sec: int) -> str:
    """HH:MM:SS label aligned with the chart time axis."""
    if sim_sec <= 0:
        return ""
    h, rem = divmod(int(sim_sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _discover_miners() -> list[dict[str, Any]]:
    root = telemetry_root()
    miners: list[dict[str, Any]] = []
    if not root.is_dir():
        return miners
    for uid_dir in sorted(root.iterdir()):
        if not uid_dir.is_dir() or not uid_dir.name.isdigit():
            continue
        uid = int(uid_dir.name)
        for val_dir in uid_dir.iterdir():
            if not val_dir.is_dir():
                continue
            for sim_dir in val_dir.iterdir():
                if not sim_dir.is_dir():
                    continue
                db = sim_dir / "telemetry.sqlite"
                if not db.is_file():
                    continue
                meta: dict[str, Any] = {}
                meta_file = sim_dir / "meta.json"
                if meta_file.is_file():
                    try:
                        meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        pass
                miners.append(
                    {
                        "uid": uid,
                        "validator_slug": val_dir.name,
                        "simulation_id": sim_dir.name,
                        "agent_class": meta.get("agent_class"),
                    }
                )
    return miners


def _quote_volume_from_trades_csv(path: Path, uid: int, book: int) -> float:
    """Sum price*quantity for fills where this uid is taker or maker on the book."""
    uid_s = str(uid)
    total = 0.0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row.get("bookId", -1)) != book:
                continue
            if str(row.get("takerAgentId", "")) != uid_s and str(row.get("makerAgentId", "")) != uid_s:
                continue
            total += _fnum(row.get("quantity")) * _fnum(row.get("price"))
    return total


def _find_trades_csv(uid: int, validator_slug: str, sim_id: str) -> Path | None:
    root = data_root() / str(uid)
    if not root.is_dir():
        return None
    for val_dir in root.iterdir():
        if not val_dir.is_dir():
            continue
        if val_dir.name != validator_slug and slug_validator(val_dir.name) != validator_slug:
            continue
        candidate = val_dir / sim_id / "trades.csv"
        if candidate.is_file():
            return candidate
    for val_dir in root.iterdir():
        candidate = val_dir / sim_id / "trades.csv"
        if candidate.is_file():
            return candidate
    return None


def _trade_sort_key(row: dict[str, Any]) -> tuple:
    ts = row.get("timestamp") or ""
    try:
        trade_id = int(row.get("tradeId") or 0)
    except (TypeError, ValueError):
        trade_id = 0
    return (ts, trade_id)


def _ohlcv_add_price(buckets: dict[int, dict[str, float]], bucket: int, price: float) -> None:
    b = buckets[bucket]
    if not b["open"] and not b["high"]:
        b["open"] = b["high"] = b["low"] = b["close"] = price
    else:
        b["high"] = max(b["high"], price)
        b["low"] = min(b["low"], price)
        b["close"] = price


def _ohlcv_from_trades(path: Path, book: int, resolution: int) -> dict[int, dict[str, float]]:
    buckets: dict[int, dict[str, float]] = defaultdict(
        lambda: {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0}
    )
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row.get("bookId", -1)) != book:
                continue
            price = _fnum(row.get("price"))
            if price <= 0:
                continue
            try:
                bucket = _chart_time(row.get("timestamp", "")) // resolution
            except ValueError:
                continue
            _ohlcv_add_price(buckets, bucket, price)
    return buckets


def _ohlcv_from_snapshots(
    conn: sqlite3.Connection, book: int, resolution: int
) -> dict[int, dict[str, float]]:
    buckets: dict[int, dict[str, float]] = defaultdict(
        lambda: {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0}
    )
    rows = conn.execute(
        """
        SELECT ts_ns, mid FROM snapshots
        WHERE book_id = ? AND mid IS NOT NULL
        ORDER BY ts_ns ASC
        """,
        (book,),
    ).fetchall()
    for row in rows:
        bucket = (int(row["ts_ns"]) // 1_000_000_000) // resolution
        _ohlcv_add_price(buckets, bucket, float(row["mid"]))
    return buckets


def _aggregate_taker_orders(rows: list[dict[str, Any]], uid: int) -> list[dict[str, Any]]:
    uid_s = str(uid)
    groups: dict[str, list[dict[str, Any]]] = {}
    order_keys: list[str] = []

    for row in rows:
        if str(row.get("takerAgentId", "")) != uid_s:
            continue
        oid = (row.get("takerOrderId") or "").strip()
        if not oid:
            oid = f"{row.get('timestamp')}|{row.get('bookId')}|{row.get('side')}"
        if oid not in groups:
            groups[oid] = []
            order_keys.append(oid)
        groups[oid].append(row)

    position = 0.0
    orders: list[dict[str, Any]] = []
    for oid in order_keys:
        fills = groups[oid]
        first = fills[0]
        is_buy = _is_buy_side(first.get("side"))
        qty = sum(_fnum(f.get("quantity")) for f in fills)
        notional = sum(_fnum(f.get("quantity")) * _fnum(f.get("price")) for f in fills)
        vwap = notional / qty if qty > 0 else 0.0

        if is_buy:
            if position < -1e-9:
                action, position = "close_short", position + qty
            else:
                action, position = "open_long", position + qty
        else:
            if position > 1e-9:
                action, position = "close_long", position - qty
            else:
                action, position = "open_short", position - qty

        try:
            t = _chart_time(first.get("timestamp", ""))
        except ValueError:
            t = 0

        orders.append(
            {
                "orderId": oid,
                "timestamp": first.get("timestamp", ""),
                "time": t,
                "time_label": _chart_time_label(t),
                "bookId": first.get("bookId", ""),
                "action": action,
                "side": "buy" if is_buy else "sell",
                "price": round(vwap, 6),
                "quantity": round(qty, 6),
                "fills": len(fills),
            }
        )
    return orders


def _format_hold_s(seconds: float | None) -> str:
    if seconds is None:
        return ""
    s = float(seconds)
    if s >= 3600:
        h, rem = divmod(int(s), 3600)
        m, sec = divmod(rem, 60)
        return f"{h}h {m}m {sec:.1f}s"
    if s >= 60:
        m, sec = divmod(int(s), 60)
        return f"{m}m {sec:.1f}s"
    return f"{s:.1f}s"


def _format_round_trip(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    ts_ns = int(out.pop("ts_close_ns", 0) or 0)
    out.pop("id", None)
    if ts_ns > 0:
        out["closed_at"] = duration_from_timestamp(ts_ns)
    else:
        out["closed_at"] = ""
    hold = out.pop("hold_s", None)
    out["hold"] = _format_hold_s(hold) if hold is not None else ""
    for key in ("qty", "entry_avg", "exit_avg", "realized_pnl"):
        if out.get(key) is not None:
            out[key] = round(float(out[key]), 4)
    return out


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "telemetry_root": str(telemetry_root())}


@app.get("/api/miners")
def list_miners() -> list[dict[str, Any]]:
    return _discover_miners()


@app.get("/api/{uid}/{validator_slug}/{sim_id}/summary")
def get_summary(uid: int, validator_slug: str, sim_id: str, book: int = 0) -> dict[str, Any]:
    conn = _connect(_db_path(uid, validator_slug, sim_id))
    try:
        snap = conn.execute(
            "SELECT * FROM snapshots WHERE book_id = ? ORDER BY ts_ns DESC LIMIT 1",
            (book,),
        ).fetchone()
        summary = conn.execute(
            "SELECT * FROM agent_summary ORDER BY ts_ns DESC LIMIT 1"
        ).fetchone()
        if book < 0:
            rt = conn.execute(
                """
                SELECT COUNT(*) AS n, COALESCE(SUM(realized_pnl), 0) AS total_pnl
                FROM round_trips
                """
            ).fetchone()
        else:
            rt = conn.execute(
                """
                SELECT COUNT(*) AS n, COALESCE(SUM(realized_pnl), 0) AS total_pnl
                FROM round_trips WHERE book_id = ?
                """,
                (book,),
            ).fetchone()
    finally:
        conn.close()

    rt_d = _row(rt)
    snap_d = _row(snap)
    traded = snap_d.get("traded_volume")
    vol_cap = snap_d.get("volume_cap")
    if (traded is None or traded == 0) and vol_cap:
        trades_path = _find_trades_csv(uid, validator_slug, sim_id)
        if trades_path:
            csv_vol = _quote_volume_from_trades_csv(trades_path, uid, book)
            if csv_vol > 0:
                snap_d["traded_volume"] = csv_vol
                snap_d["volume_remaining"] = max(0.0, float(vol_cap) - csv_vol)
    elif traded is not None and vol_cap is not None and snap_d.get("volume_remaining") is None:
        snap_d["volume_remaining"] = max(0.0, float(vol_cap) - float(traded))

    return {
        "book_id": book,
        "latest_snapshot": snap_d,
        "latest_summary": _row(summary),
        "round_trips": rt_d,
        "pnl_per_rt": (rt_d["total_pnl"] / rt_d["n"]) if rt_d.get("n") else None,
    }


@app.get("/api/{uid}/{validator_slug}/{sim_id}/ohlcv")
def get_ohlcv(
    uid: int,
    validator_slug: str,
    sim_id: str,
    book: int = 0,
    resolution: Annotated[int, Query(ge=1, le=3600)] = 1,
    limit: Annotated[int, Query(ge=10, le=10000)] = 5000,
) -> list[dict[str, Any]]:
    buckets: dict[int, dict[str, float]] = defaultdict(
        lambda: {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0}
    )

    trades_path = _find_trades_csv(uid, validator_slug, sim_id)
    if trades_path:
        for bucket, b in _ohlcv_from_trades(trades_path, book, resolution).items():
            buckets[bucket] = b

    db = _db_path(uid, validator_slug, sim_id)
    if db.is_file():
        conn = _connect(db)
        try:
            for bucket, b in _ohlcv_from_snapshots(conn, book, resolution).items():
                _ohlcv_add_price(buckets, bucket, b["close"])
        finally:
            conn.close()

    if not buckets:
        return []

    return [
        {
            "time": bucket * resolution,
            "open": buckets[bucket]["open"],
            "high": buckets[bucket]["high"],
            "low": buckets[bucket]["low"],
            "close": buckets[bucket]["close"],
        }
        for bucket in sorted(buckets.keys())[-limit:]
    ]


@app.get("/api/{uid}/{validator_slug}/{sim_id}/snapshots")
def get_snapshots(
    uid: int,
    validator_slug: str,
    sim_id: str,
    book: int = 0,
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
) -> list[dict[str, Any]]:
    conn = _connect(_db_path(uid, validator_slug, sim_id))
    try:
        rows = conn.execute(
            """
            SELECT ts_ns, mid, signal_trend_bps, signal_flow, signal_imb,
                   action, pos_qty, spread_bps
            FROM snapshots
            WHERE book_id = ?
            ORDER BY ts_ns DESC LIMIT ?
            """,
            (book, limit),
        ).fetchall()
    finally:
        conn.close()
    out = []
    for r in reversed(rows):
        d = dict(r)
        ts_ns = int(d.pop("ts_ns", 0) or 0)
        d["closed_at"] = duration_from_timestamp(ts_ns) if ts_ns else ""
        out.append(d)
    return out


@app.get("/api/{uid}/{validator_slug}/{sim_id}/round_trips")
def get_round_trips(
    uid: int,
    validator_slug: str,
    sim_id: str,
    book: Annotated[int, Query()] = -1,
    limit: Annotated[int, Query(ge=1, le=2000)] = 200,
) -> list[dict[str, Any]]:
    conn = _connect(_db_path(uid, validator_slug, sim_id))
    try:
        if book < 0:
            rows = conn.execute(
                "SELECT * FROM round_trips ORDER BY ts_close_ns DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM round_trips
                WHERE book_id = ? ORDER BY ts_close_ns DESC LIMIT ?
                """,
                (book, limit),
            ).fetchall()
    finally:
        conn.close()
    formatted = [_format_round_trip(dict(r)) for r in rows]
    formatted.reverse()
    for i, row in enumerate(formatted, start=1):
        row["seq"] = i
    return formatted


@app.get("/api/{uid}/{validator_slug}/{sim_id}/trades")
def get_trades(
    uid: int,
    validator_slug: str,
    sim_id: str,
    book: Annotated[int, Query()] = -1,
    limit: Annotated[int, Query(ge=1, le=1000)] = 500,
) -> dict[str, Any]:
    path = _find_trades_csv(uid, validator_slug, sim_id)
    if path is None:
        return {"orders": []}
    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if book >= 0 and int(row.get("bookId", -1)) != book:
                continue
            rows.append(dict(row))
    rows.sort(key=_trade_sort_key)
    orders = _aggregate_taker_orders(rows, uid)
    window = list(reversed(orders[-limit:]))
    for i, order in enumerate(window, start=1):
        order["seq"] = i
    return {"orders": window}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.is_dir():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


def main() -> None:
    import uvicorn

    host = os.environ.get("TAOS_DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("TAOS_DASHBOARD_PORT", "8787"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
