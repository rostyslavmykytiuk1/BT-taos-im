#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Phase-0 spike: seed demo telemetry DB for dashboard smoke test."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from taos.im.telemetry import seed_demo_db, telemetry_root


def main() -> None:
    os.environ.setdefault("TAOS_TELEMETRY_ROOT", os.path.expanduser("~/.taos/telemetry"))
    path = seed_demo_db()
    print(f"Seeded demo telemetry: {path}")
    print(f"Telemetry root: {telemetry_root()}")
    print("Start dashboard: ./dashboard/start.sh")
    print("Open http://127.0.0.1:8787/ and select UID 999, validator demo_validator, sim demo_sim")


if __name__ == "__main__":
    main()
