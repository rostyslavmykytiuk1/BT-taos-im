#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Read-only FastAPI server + static UI for miner telemetry."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from chart_data import (
    build_mid_series,
    find_trades_csv,
    format_round_trip,
    load_trades_for_api,
    quote_volume_from_trades_csv,
    sim_time_label,
    sim_seconds_from_ts_ns,
    telemetry_db,
    visible_sim_range,
)
from taos.im.telemetry.paths import telemetry_root

WEB_DIR = Path(__file__).resolve().parent / "web"

# /api/validators/{validator_id}/agents/{uid}/simulations/{simulation_id}/books/{book_id}/...
BOOK_SCOPE = (
    "/api/validators/{validator_id}/agents/{uid}/simulations/{simulation_id}/books/{book_id}"
)

app = FastAPI(title="τaos Miner Telemetry", version="0.3.0")
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


def _row(r: sqlite3.Row | None) -> dict[str, Any]:
    return dict(r) if r else {}


def _discover_catalog() -> list[dict[str, Any]]:
    root = telemetry_root()
    if not root.is_dir():
        return []
    sessions: list[dict[str, Any]] = []
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
                if not (sim_dir / "telemetry.sqlite").is_file():
                    continue
                meta: dict[str, Any] = {}
                meta_path = sim_dir / "meta.json"
                if meta_path.is_file():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        pass
                sessions.append(
                    {
                        "uid": uid,
                        "validator_id": val_dir.name,
                        "simulation_id": sim_dir.name,
                        "agent_class": meta.get("agent_class"),
                    }
                )
    return sessions


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "telemetry_root": str(telemetry_root())}


@app.get("/api/catalog")
def list_catalog() -> list[dict[str, Any]]:
    return _discover_catalog()


@app.get("/api/miners")
def list_miners() -> list[dict[str, Any]]:
    """Deprecated alias for /api/catalog."""
    return _discover_catalog()


@app.get(f"{BOOK_SCOPE}/summary")
def get_summary(
    validator_id: str,
    uid: int,
    simulation_id: str,
    book_id: int,
) -> dict[str, Any]:
    conn = _connect(telemetry_db(uid, validator_id, simulation_id))
    try:
        snap = conn.execute(
            "SELECT * FROM snapshots WHERE book_id = ? ORDER BY ts_ns DESC LIMIT 1",
            (book_id,),
        ).fetchone()
        summary = conn.execute(
            "SELECT * FROM agent_summary ORDER BY ts_ns DESC LIMIT 1"
        ).fetchone()
        rt = conn.execute(
            """
            SELECT COUNT(*) AS n, COALESCE(SUM(realized_pnl), 0) AS total_pnl
            FROM round_trips WHERE book_id = ?
            """,
            (book_id,),
        ).fetchone()
    finally:
        conn.close()

    rt_d = _row(rt)
    snap_d = _row(snap)
    traded = snap_d.get("traded_volume")
    vol_cap = snap_d.get("volume_cap")
    if (traded is None or traded == 0) and vol_cap:
        trades_path = find_trades_csv(uid, validator_id, simulation_id)
        if trades_path:
            csv_vol = quote_volume_from_trades_csv(trades_path, uid, book_id)
            if csv_vol > 0:
                snap_d["traded_volume"] = csv_vol
                snap_d["volume_remaining"] = max(0.0, float(vol_cap) - csv_vol)
    elif traded is not None and vol_cap is not None and snap_d.get("volume_remaining") is None:
        snap_d["volume_remaining"] = max(0.0, float(vol_cap) - float(traded))

    return {
        "validator_id": validator_id,
        "uid": uid,
        "simulation_id": simulation_id,
        "book_id": book_id,
        "latest_snapshot": snap_d,
        "latest_summary": _row(summary),
        "round_trips": rt_d,
        "pnl_per_rt": (rt_d["total_pnl"] / rt_d["n"]) if rt_d.get("n") else None,
    }


@app.get(f"{BOOK_SCOPE}/mid")
def get_mid(
    validator_id: str,
    uid: int,
    simulation_id: str,
    book_id: int,
    resolution: Annotated[int, Query(ge=1, le=3600)] = 1,
    limit: Annotated[int, Query(ge=10, le=10000)] = 5000,
) -> dict[str, Any]:
    return build_mid_series(
        uid, validator_id, simulation_id, book_id, resolution, limit, _connect
    )


@app.get(f"{BOOK_SCOPE}/snapshots")
def get_snapshots(
    validator_id: str,
    uid: int,
    simulation_id: str,
    book_id: int,
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
) -> list[dict[str, Any]]:
    conn = _connect(telemetry_db(uid, validator_id, simulation_id))
    try:
        rows = conn.execute(
            """
            SELECT ts_ns, mid, signal_trend_bps, signal_flow, signal_imb,
                   action, pos_qty, spread_bps
            FROM snapshots
            WHERE book_id = ?
            ORDER BY ts_ns DESC LIMIT ?
            """,
            (book_id, limit),
        ).fetchall()
    finally:
        conn.close()

    out: list[dict[str, Any]] = []
    for r in reversed(rows):
        d = dict(r)
        ts_ns = int(d.pop("ts_ns", 0) or 0)
        time_sec = sim_seconds_from_ts_ns(ts_ns) if ts_ns else 0
        d["time_sec"] = time_sec
        d["closed_at"] = sim_time_label(time_sec) if time_sec else ""
        out.append(d)
    return out


@app.get(f"{BOOK_SCOPE}/round_trips")
def get_round_trips(
    validator_id: str,
    uid: int,
    simulation_id: str,
    book_id: int,
    limit: Annotated[int, Query(ge=1, le=2000)] = 200,
) -> list[dict[str, Any]]:
    conn = _connect(telemetry_db(uid, validator_id, simulation_id))
    try:
        rows = conn.execute(
            """
            SELECT * FROM round_trips
            WHERE book_id = ? ORDER BY ts_close_ns DESC LIMIT ?
            """,
            (book_id, limit),
        ).fetchall()
    finally:
        conn.close()

    t_min, t_max = visible_sim_range(uid, validator_id, simulation_id, book_id, 1, _connect)
    formatted = [format_round_trip(dict(r)) for r in rows]
    if t_max > t_min:
        formatted = [
            row for row in formatted
            if t_min <= int(row.get("time_sec") or 0) <= t_max
        ]
    formatted.reverse()
    for i, row in enumerate(formatted, start=1):
        row["seq"] = i
    return formatted


@app.get(f"{BOOK_SCOPE}/trades")
def get_trades(
    validator_id: str,
    uid: int,
    simulation_id: str,
    book_id: int,
    limit: Annotated[int, Query(ge=1, le=1000)] = 500,
) -> dict[str, Any]:
    return load_trades_for_api(uid, validator_id, simulation_id, book_id, limit, _connect)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get(
    "/validators/{validator_id}/agents/{uid}/simulations/{simulation_id}/books/{book_id}"
)
def dashboard_spa(
    validator_id: str,
    uid: int,
    simulation_id: str,
    book_id: int,
) -> FileResponse:
    """Serve the SPA shell for bookmarkable dashboard URLs."""
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
