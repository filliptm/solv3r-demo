#!/usr/bin/env bash
# Starts the Solv3R demo on http://127.0.0.1:7860 (Linux / macOS / Git Bash). PORT and HOST override the defaults.
set -euo pipefail
demo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$demo")"
py="$root/.venv/bin/python"
[ -x "$py" ] || py="$root/.venv/Scripts/python.exe"
[ -x "$py" ] || { echo "No Python environment at $root/.venv (see demo/README.md)"; exit 1; }
if [ ! -f "$demo/web/dist/index.html" ] || [ "${1:-}" = "--rebuild" ]; then
  (cd "$demo/web" && { [ -d node_modules ] || npm install; } && npm run build)
fi
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
cd "$demo"
echo "Solv3R demo: http://${HOST:-127.0.0.1}:${PORT:-7860}"
exec "$py" -m uvicorn server.app:app --host "${HOST:-127.0.0.1}" --port "${PORT:-7860}"
