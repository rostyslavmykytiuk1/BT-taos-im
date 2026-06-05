# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT

import os
import sqlite3
import tempfile
from pathlib import Path

from taos.im.telemetry import MinerTelemetry, seed_demo_db
from taos.im.telemetry.schema import init_db


def test_seed_demo_db_has_snapshots():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["TAOS_TELEMETRY_ROOT"] = tmp
        path = seed_demo_db(uid=42, n_points=10)
        assert path.is_file()
        conn = sqlite3.connect(path)
        init_db(conn)
        n = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
        conn.close()
        assert n == 10


def test_miner_telemetry_batch_insert():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["TAOS_TELEMETRY_ROOT"] = tmp
        tel = MinerTelemetry(uid=7, agent_class="TestAgent", params={"x": 1})
        tel._ensure_session("val_test", "sim_test")

        class _Cfg:
            simulation_id = "sim_test"

        class _State:
            timestamp = 1000
            config = _Cfg()
            dendrite = type("D", (), {"hotkey": "val_test"})()

        tel.begin_step(_State())
        tel.snapshot(
            book_id=1,
            mid=50.0,
            bid=49.9,
            ask=50.1,
            traded_volume=12_000.0,
            volume_cap=300_000.0,
            volume_remaining=288_000.0,
            signals={"trend_bps": 4.0, "flow": 1.0, "imb": 0.0, "level": 49.8},
            action="hold",
        )
        tel.end_step(_State(), instructions=2)

        row = tel._conn.execute(
            """
            SELECT traded_volume, volume_cap, volume_remaining, signal_level
            FROM snapshots WHERE book_id=1
            """
        ).fetchone()
        assert row == (12_000.0, 300_000.0, 288_000.0, 49.8)
