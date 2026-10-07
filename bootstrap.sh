#!/usr/bin/env bash
# One-command setup for Market Lab in any fresh container or machine.
#   bash bootstrap.sh            # install + health check
#   bash bootstrap.sh --quiet    # install only, minimal output
# The venv lives outside the workspace (fast local disk, not the shared mount).
# Activate afterwards with:  source "$MLAB_VENV/bin/activate"   (or just call  ./mlab ...)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLAB_VENV="${MLAB_VENV:-$HOME/.venvs/market-lab}"
QUIET="${1:-}"

log() { [ "$QUIET" = "--quiet" ] || echo "[mlab] $*"; }

if command -v uv >/dev/null 2>&1; then
  log "creating venv with uv at $MLAB_VENV"
  [ -x "$MLAB_VENV/bin/python" ] || uv venv -q "$MLAB_VENV"
  VIRTUAL_ENV="$MLAB_VENV" uv pip install -q -e "$HERE[stream,dev]"
else
  log "creating venv with python -m venv at $MLAB_VENV"
  [ -x "$MLAB_VENV/bin/python" ] || python3 -m venv "$MLAB_VENV"
  "$MLAB_VENV/bin/pip" install -q --upgrade pip
  "$MLAB_VENV/bin/pip" install -q -e "$HERE[stream,dev]"
fi

log "installed. Running health check (network + credentials)..."
[ "$QUIET" = "--quiet" ] || "$MLAB_VENV/bin/mlab" doctor || true
log "done. Use:  $HERE/mlab <command>   e.g.  ./mlab price AAPL --period 1y"
