# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Filesystem layout for miner telemetry."""

from __future__ import annotations

import os
import re
from pathlib import Path


def telemetry_enabled() -> bool:
    return os.environ.get("TAOS_TELEMETRY_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def telemetry_root() -> Path:
    raw = os.environ.get("TAOS_TELEMETRY_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".taos" / "telemetry"


def data_root() -> Path:
    """Agent event CSV root (orders/trades)."""
    raw = os.environ.get("TAOS_DATA_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser()
    # Default matches SimulationAgent.data_dir convention in-repo.
    return Path(__file__).resolve().parents[3] / "agents" / "data"


def slug_validator(hotkey: str) -> str:
    """Filesystem-safe directory name for a validator hotkey."""
    s = re.sub(r"[^A-Za-z0-9_-]", "_", hotkey)
    return s[:48] if len(s) > 48 else s


def simulation_dir(uid: int, validator: str, simulation_id: str) -> Path:
    return telemetry_root() / str(uid) / slug_validator(validator) / simulation_id


def db_path(uid: int, validator: str, simulation_id: str) -> Path:
    return simulation_dir(uid, validator, simulation_id) / "telemetry.sqlite"


def meta_path(uid: int, validator: str, simulation_id: str) -> Path:
    return simulation_dir(uid, validator, simulation_id) / "meta.json"


def event_csv_dir(uid: int, validator: str, simulation_id: str) -> Path:
    return data_root() / str(uid) / validator / simulation_id


def parse_sample_books() -> set[int] | None:
    """None = all books; empty set after parse = all books."""
    raw = os.environ.get("TAOS_TELEMETRY_SAMPLE_BOOKS", "").strip()
    if not raw:
        return None
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out if out else None
