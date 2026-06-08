# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Write per-step miner telemetry to SQLite (optional, env-gated)."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from taos.im.telemetry.paths import (
    db_path,
    meta_path,
    parse_sample_books,
    simulation_dir,
    telemetry_enabled,
)
from taos.im.telemetry.schema import init_db

_log = logging.getLogger("taos.telemetry")
_last_error_log: float = 0.0


def _log_error_once(msg: str, exc: BaseException | None = None) -> None:
    global _last_error_log
    now = time.time()
    if now - _last_error_log < 60.0:
        return
    _last_error_log = now
    if exc:
        _log.warning("%s: %s", msg, exc)
    else:
        _log.warning(msg)


@dataclass
class _StepBuffer:
    ts_ns: int = 0
    snapshots: list[tuple] = field(default_factory=list)
    open_positions: int = 0
    instructions: int = 0
    loop_start: float = 0.0


class _NoOpTelemetry:
    """Drop-in stub when telemetry is disabled."""

    def begin_step(self, state: Any) -> None:
        pass

    def snapshot(
        self,
        *,
        book_id: int,
        mid: float | None = None,
        bid: float | None = None,
        ask: float | None = None,
        pos_qty: float = 0.0,
        pos_avg: float = 0.0,
        base_bal: float | None = None,
        quote_bal: float | None = None,
        traded_volume: float | None = None,
        volume_cap: float | None = None,
        volume_remaining: float | None = None,
        signals: dict[str, float] | None = None,
        action: str = "hold",
    ) -> None:
        pass

    def record_round_trip(
        self,
        *,
        book_id: int,
        ts_close_ns: int,
        side: str,
        qty: float,
        entry_avg: float,
        exit_avg: float,
        realized_pnl: float,
        hold_s: float | None = None,
        reason: str = "",
    ) -> None:
        pass

    def end_step(self, state: Any, instructions: int = 0) -> None:
        pass

    def flush(self) -> None:
        pass


class MinerTelemetry:
    """Batch writer for one (uid, validator, simulation_id) SQLite file."""

    @classmethod
    def from_agent(
        cls,
        agent: Any,
        *,
        agent_class: str | None = None,
    ) -> MinerTelemetry | _NoOpTelemetry:
        if not telemetry_enabled():
            return _NoOpTelemetry()
        name = agent_class or type(agent).__name__
        params = _agent_params_dict(agent)
        return cls(
            uid=int(agent.uid),
            agent_class=name,
            params=params,
            wallet=getattr(agent.config, "wallet", None),
            hotkey=getattr(agent.config, "hotkey", None),
        )

    def __init__(
        self,
        uid: int,
        agent_class: str,
        params: dict[str, Any] | None = None,
        wallet: str | None = None,
        hotkey: str | None = None,
    ) -> None:
        self.uid = uid
        self.agent_class = agent_class
        self.params = params or {}
        self.wallet = wallet
        self.hotkey = hotkey
        self._sample_books = parse_sample_books()
        self._conn: sqlite3.Connection | None = None
        self._key: tuple[str, str] | None = None
        self._buf = _StepBuffer()

    def _should_sample(self, book_id: int) -> bool:
        if self._sample_books is None:
            return True
        return book_id in self._sample_books

    def _ensure_session(self, validator: str, simulation_id: str) -> None:
        key = (validator, simulation_id)
        if self._key == key and self._conn is not None:
            return
        if self._conn is not None:
            try:
                self._conn.commit()
                self._conn.close()
            except Exception:
                pass
        sim_dir = simulation_dir(self.uid, validator, simulation_id)
        sim_dir.mkdir(parents=True, exist_ok=True)
        db = db_path(self.uid, validator, simulation_id)
        conn = sqlite3.connect(str(db), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        init_db(conn)
        self._conn = conn
        self._key = key
        self._write_meta(validator, simulation_id)

    def _write_meta(self, validator: str, simulation_id: str) -> None:
        path = meta_path(self.uid, validator, simulation_id)
        if path.exists():
            return
        payload = {
            "uid": self.uid,
            "agent_class": self.agent_class,
            "params": self.params,
            "wallet": self.wallet,
            "hotkey": self.hotkey,
            "validator": validator,
            "simulation_id": simulation_id,
            "started_wall": datetime.now(timezone.utc).isoformat(),
        }
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            _log_error_once("telemetry meta write failed", exc)

    def begin_step(self, state: Any) -> None:
        try:
            validator = state.dendrite.hotkey
            sim_id = state.config.simulation_id
            self._ensure_session(validator, sim_id)
            self._buf = _StepBuffer(
                ts_ns=int(state.timestamp),
                loop_start=time.perf_counter(),
            )
        except Exception as exc:
            _log_error_once("telemetry begin_step failed", exc)

    def snapshot(
        self,
        *,
        book_id: int,
        mid: float | None = None,
        bid: float | None = None,
        ask: float | None = None,
        pos_qty: float = 0.0,
        pos_avg: float = 0.0,
        base_bal: float | None = None,
        quote_bal: float | None = None,
        traded_volume: float | None = None,
        volume_cap: float | None = None,
        volume_remaining: float | None = None,
        signals: dict[str, float] | None = None,
        action: str = "hold",
    ) -> None:
        if self._conn is None or not self._should_sample(book_id):
            return
        sig = signals or {}
        spread_bps = None
        if mid and bid is not None and ask is not None and mid > 0:
            spread_bps = (ask - bid) / mid * 1e4
        unrealized = None
        if mid and pos_avg > 0 and abs(pos_qty) > 1e-12:
            if pos_qty > 0:
                unrealized = (mid - pos_avg) * abs(pos_qty)
            else:
                unrealized = (pos_avg - mid) * abs(pos_qty)
        row = (
            self._buf.ts_ns,
            book_id,
            mid,
            bid,
            ask,
            spread_bps,
            pos_qty,
            pos_avg,
            unrealized,
            base_bal,
            quote_bal,
            sig.get("taker_bps", sig.get("trend_bps")),
            sig.get("kappa3", sig.get("flow")),
            sig.get("est_pnl", sig.get("gap_s", sig.get("imb"))),
            action,
            traded_volume,
            volume_cap,
            volume_remaining,
        )
        self._buf.snapshots.append(row)
        if abs(pos_qty) >= 1e-12:
            self._buf.open_positions += 1

    def record_round_trip(
        self,
        *,
        book_id: int,
        ts_close_ns: int,
        side: str,
        qty: float,
        entry_avg: float,
        exit_avg: float,
        realized_pnl: float,
        hold_s: float | None = None,
        reason: str = "",
    ) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute(
                """
                INSERT INTO round_trips (
                    ts_close_ns, book_id, side, qty, entry_avg, exit_avg,
                    realized_pnl, hold_s, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_close_ns,
                    book_id,
                    side,
                    qty,
                    entry_avg,
                    exit_avg,
                    realized_pnl,
                    hold_s,
                    reason,
                ),
            )
        except Exception as exc:
            _log_error_once("telemetry round_trip failed", exc)

    def end_step(self, state: Any, instructions: int = 0) -> None:
        if self._conn is None:
            return
        try:
            loop_ms = (time.perf_counter() - self._buf.loop_start) * 1000.0
            self._conn.execute(
                """
                INSERT OR REPLACE INTO agent_summary (ts_ns, open_positions, instructions, loop_ms)
                VALUES (?, ?, ?, ?)
                """,
                (
                    self._buf.ts_ns,
                    self._buf.open_positions,
                    instructions,
                    loop_ms,
                ),
            )
            if self._buf.snapshots:
                self._conn.executemany(
                    """
                    INSERT OR REPLACE INTO snapshots (
                        ts_ns, book_id, mid, bid, ask, spread_bps,
                        pos_qty, pos_avg, unrealized_pnl, base_bal, quote_bal,
                        signal_trend_bps, signal_flow, signal_imb, action,
                        traded_volume, volume_cap, volume_remaining
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._buf.snapshots,
                )
            self._conn.commit()
            self._buf.snapshots.clear()
        except Exception as exc:
            _log_error_once("telemetry end_step failed", exc)

    def flush(self) -> None:
        if self._conn is not None:
            try:
                self._conn.commit()
            except Exception as exc:
                _log_error_once("telemetry flush failed", exc)


def _agent_params_dict(agent: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "quote_notional",
        "tp_bps",
        "sl_bps",
        "max_hold_s",
        "signal_bps",
        "imbalance_depth",
        "min_imbalance",
        "volume_safety",
    ):
        if hasattr(agent, key):
            out[key] = getattr(agent, key)
    return out


def seed_demo_db(
    uid: int = 999,
    validator: str = "demo_validator",
    simulation_id: str = "demo_sim",
    n_points: int = 60,
) -> Path:
    """Phase-0 spike: write synthetic mid series for dashboard smoke test."""
    tel = MinerTelemetry(
        uid=uid,
        agent_class="DemoAgent",
        params={"demo": True},
    )
    tel._ensure_session(validator, simulation_id)
    base_mid = 100.0
    base_ts = 1_700_000_000_000_000_000  # fits SQLite INTEGER (ns)
    for i in range(n_points):
        ts_ns = base_ts + i * 1_000_000_000
        mid = base_mid + 0.05 * i + 0.02 * (i % 7)
        tel._buf.ts_ns = ts_ns
        tel.snapshot(
            book_id=0,
            mid=mid,
            bid=mid - 0.01,
            ask=mid + 0.01,
            pos_qty=0.0 if i % 15 else 2.5,
            pos_avg=mid - 0.03 if i % 15 else 0.0,
            base_bal=10.0,
            quote_bal=5000.0,
            traded_volume=float(i * 1200),
            volume_cap=300_000.0,
            volume_remaining=max(0.0, 300_000.0 - i * 1200),
            signals={"trend_bps": 2.0, "flow": 0.5, "imb": 0.12},
            action="hold" if i % 5 else "enter_long",
        )
        class _State:
            timestamp = ts_ns
            config = type("C", (), {"simulation_id": simulation_id})()
            dendrite = type("D", (), {"hotkey": validator})()

        tel.end_step(_State(), instructions=i % 3)
    if tel._conn:
        tel._conn.execute(
            """
            INSERT INTO round_trips (
                ts_close_ns, book_id, side, qty, entry_avg, exit_avg,
                realized_pnl, hold_s, reason
            ) VALUES (?, 0, 'long', 2.5, 100.0, 100.12, 0.30, 45.0, 'tp')
            """,
            (base_ts + 30 * 1_000_000_000,),
        )
        tel._conn.commit()
    return db_path(uid, validator, simulation_id)
