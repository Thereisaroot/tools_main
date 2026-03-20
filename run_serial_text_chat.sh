#!/usr/bin/env bash
set -euo pipefail

if ! python3 -c "import serial" >/dev/null 2>&1; then
  echo "Installing dependency: pyserial"
  python3 -m pip install -r requirements.txt
fi

exec python3 serial_text_chat.py "$@"
