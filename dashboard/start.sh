#!/bin/bash
# Start local miner telemetry dashboard (API + UI).
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Load repo .env if present (same file as run_miner.sh)
[ -f "$REPO_ROOT/.env" ] && { set -a; . "$REPO_ROOT/.env"; set +a; }

export TAOS_TELEMETRY_ROOT="${TAOS_TELEMETRY_ROOT:-$HOME/.taos/telemetry}"
export TAOS_DATA_ROOT="${TAOS_DATA_ROOT:-$REPO_ROOT/agents/data}"
export TAOS_DASHBOARD_HOST="${TAOS_DASHBOARD_HOST:-127.0.0.1}"
export TAOS_DASHBOARD_PORT="${TAOS_DASHBOARD_PORT:-8787}"

echo "Telemetry root: $TAOS_TELEMETRY_ROOT"
echo "Data root:      $TAOS_DATA_ROOT"
echo "Dashboard:      http://${TAOS_DASHBOARD_HOST}:${TAOS_DASHBOARD_PORT}/"

exec python "$REPO_ROOT/dashboard/telemetry_server.py"
