#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! python3 -c "import shooklink, PySide6, serial, cryptography, pyte" >/dev/null 2>&1; then
  echo "Installing ShookLink dependencies from requirements.txt"
  python3 -m pip install -r requirements.txt
fi

exec python3 -m shooklink "$@"
