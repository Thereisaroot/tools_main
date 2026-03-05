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
  echo "[setup] python not found"
  exit 1
fi

echo "[setup] using interpreter: $PY_BIN"
$PY_BIN -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
if ! python -m pip install -r requirements.txt; then
  echo "[setup] base dependency install failed."
  echo "[setup] install Python 3.12 and retry if using an unsupported interpreter."
  exit 1
fi

if python - <<'PY'
import sys
raise SystemExit(0 if sys.version_info < (3, 13) else 1)
PY
then
  if ! python -m pip install "rapidocr-onnxruntime==1.2.3"; then
    echo "[setup] rapidocr install failed. OCR fallback will use pytesseract."
  fi
else
  echo "[setup] python>=3.13 detected. rapidocr skipped, pytesseract fallback mode."
fi

if ! python -m pip install av; then
  echo "[setup] av install failed. video decoding fallback will use OpenCV."
fi

if ! command -v tesseract >/dev/null 2>&1; then
  echo "[setup] warning: tesseract binary not found. OCR fallback may fail."
fi

DOWNLOAD_ARGS=()
if [[ "${MODEL_DOWNLOAD_INSECURE:-0}" == "1" ]]; then
  DOWNLOAD_ARGS+=(--insecure)
  echo "[setup] MODEL_DOWNLOAD_INSECURE=1 -> TLS verification disabled for curl fallback."
fi

if [[ ! -f models/person_detector.onnx ]]; then
  echo "[setup] model not found. trying auto-download..."
  download_status=0
  if [[ ${#DOWNLOAD_ARGS[@]} -gt 0 ]]; then
    python scripts/download_default_model.py --output models/person_detector.onnx "${DOWNLOAD_ARGS[@]}" || download_status=$?
  else
    python scripts/download_default_model.py --output models/person_detector.onnx || download_status=$?
  fi
  if [[ $download_status -ne 0 ]]; then
    echo "[setup] auto-download failed. Place ONNX model at models/person_detector.onnx"
    echo "[setup] tip: MODEL_DOWNLOAD_INSECURE=1 ./scripts/setup_run_ubuntu.sh"
  fi
fi

echo "[run] starting app"
python -m isac_labelr
