#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

choose_python() {
  if command -v python3.12 >/dev/null 2>&1; then
    echo "python3.12"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi
  echo ""
}

PY_BIN="$(choose_python)"
if [[ -z "$PY_BIN" ]]; then
  echo "[build] python not found"
  exit 1
fi

echo "[build] using interpreter: $PY_BIN"
$PY_BIN -m venv .venv-build
source .venv-build/bin/activate

python -m pip install --upgrade pip
if ! python -m pip install -r requirements.txt pyinstaller; then
  echo "[build] base dependency install failed."
  echo "[build] install Python 3.12 and retry if using an unsupported interpreter."
  exit 1
fi

if python - <<'PY'
import sys
raise SystemExit(0 if sys.version_info < (3, 13) else 1)
PY
then
  python -m pip install "rapidocr-onnxruntime==1.2.3" || true
fi
python -m pip install av || true

pyinstaller \
  --noconfirm \
  --clean \
  --windowed \
  --onedir \
  --name isac_labelr_portable \
  isac_labelr/__main__.py

if [[ -d models ]]; then
  mkdir -p dist/isac_labelr_portable/models
  cp -R models/. dist/isac_labelr_portable/models/
fi

echo "Build complete: dist/isac_labelr_portable"
