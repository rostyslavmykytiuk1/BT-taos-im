# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""SQLite schema for miner telemetry."""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 3

DDL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
    ts_ns INTEGER NOT NULL,
    book_id INTEGER NOT NULL,
    mid REAL,
    bid REAL,
    ask REAL,
    spread_bps REAL,
    pos_qty REAL,
    pos_avg REAL,
    unrealized_pnl REAL,
    base_bal REAL,
    quote_bal REAL,
    signal_trend_bps REAL,
    signal_flow REAL,
    signal_imb REAL,
    signal_level REAL,
    action TEXT,
    traded_volume REAL,
    volume_cap REAL,
    volume_remaining REAL,
    PRIMARY KEY (book_id, ts_ns)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON snapshots (ts_ns);

CREATE TABLE IF NOT EXISTS round_trips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_close_ns INTEGER NOT NULL,
    book_id INTEGER NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    entry_avg REAL NOT NULL,
    exit_avg REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    hold_s REAL,
    reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_round_trips_book ON round_trips (book_id, ts_close_ns);

CREATE TABLE IF NOT EXISTS agent_summary (
    ts_ns INTEGER PRIMARY KEY,
    open_positions INTEGER,
    instructions INTEGER,
    loop_ms REAL
);
"""


_MIGRATIONS: dict[int, list[str]] = {
    2: [
        "ALTER TABLE snapshots ADD COLUMN traded_volume REAL",
        "ALTER TABLE snapshots ADD COLUMN volume_cap REAL",
        "ALTER TABLE snapshots ADD COLUMN volume_remaining REAL",
    ],
    3: [
        "ALTER TABLE snapshots ADD COLUMN signal_level REAL",
    ],
}


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    version = int(row[0]) if row else 0
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        version = SCHEMA_VERSION
    for target in range(version + 1, SCHEMA_VERSION + 1):
        for sql in _MIGRATIONS.get(target, []):
            try:
                conn.execute(sql)
            except sqlite3.OperationalError:
                pass
        conn.execute("UPDATE schema_version SET version = ?", (target,))
    conn.commit()
