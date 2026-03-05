$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

& py -3.12 -V *> $null
if ($LASTEXITCODE -eq 0) {
  $PySelector = "py -3.12"
} else {
  $PySelector = "py -3"
}

Write-Host "[setup] using interpreter: $PySelector"
Invoke-Expression "$PySelector -m venv .venv"
& .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
  Write-Host "[setup] base dependency install failed."
  Write-Host "[setup] install Python 3.12 and retry if using an unsupported interpreter."
  exit 1
}

$PyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([version]$PyVersion -lt [version]"3.13") {
  python -m pip install "rapidocr-onnxruntime==1.2.3"
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[setup] rapidocr install failed. OCR fallback will use pytesseract."
  }
} else {
  Write-Host "[setup] python>=3.13 detected. rapidocr skipped, pytesseract fallback mode."
}

python -m pip install av
if ($LASTEXITCODE -ne 0) {
  Write-Host "[setup] av install failed. video decoding fallback will use OpenCV."
}

if (-not (Get-Command tesseract -ErrorAction SilentlyContinue)) {
  Write-Host "[setup] warning: tesseract binary not found. OCR fallback may fail."
}

$DownloadArgs = @("--output", ".\\models\\person_detector.onnx")
if ($env:MODEL_DOWNLOAD_INSECURE -eq "1") {
  $DownloadArgs += "--insecure"
  Write-Host "[setup] MODEL_DOWNLOAD_INSECURE=1 -> TLS verification disabled for curl fallback."
}

if (-not (Test-Path .\models\person_detector.onnx)) {
  Write-Host "[setup] model not found. trying auto-download..."
  python .\scripts\download_default_model.py @DownloadArgs
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[setup] auto-download failed. Place ONNX model at models/person_detector.onnx"
    Write-Host "[setup] tip: `$env:MODEL_DOWNLOAD_INSECURE='1'; .\\scripts\\setup_run_windows.ps1"
  }
}

Write-Host "[run] starting app"
python -m isac_labelr
