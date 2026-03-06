$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

try {
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force | Out-Null
} catch {
}

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Get-PythonCandidate {
  $candidates = @(
    @{ exe = "py"; args = @("-3.12"); label = "py -3.12" },
    @{ exe = "py"; args = @("-3.13"); label = "py -3.13" },
    @{ exe = "py"; args = @("-3.14"); label = "py -3.14" },
    @{ exe = "py"; args = @("-3"); label = "py -3" },
    @{ exe = "python3.12"; args = @(); label = "python3.12" },
    @{ exe = "python3.13"; args = @(); label = "python3.13" },
    @{ exe = "python3.14"; args = @(); label = "python3.14" },
    @{ exe = "python3"; args = @(); label = "python3" },
    @{ exe = "python"; args = @(); label = "python" }
  )
  foreach ($cand in $candidates) {
    if (-not (Get-Command $cand.exe -ErrorAction SilentlyContinue)) {
      continue
    }
    try {
      $out = & $cand.exe @($cand.args) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
      if ($LASTEXITCODE -ne 0) {
        continue
      }
      $ver = "$out".Trim()
      if (-not $ver) {
        continue
      }
      $v = [version]$ver
      if ($v -lt [version]"3.11") {
        continue
      }
      return @{
        exe = $cand.exe
        args = $cand.args
        label = $cand.label
        version = $v
      }
    } catch {
      continue
    }
  }
  return $null
}

$py = Get-PythonCandidate
if ($null -eq $py) {
  Write-Host "[setup] No suitable Python runtime found (need >= 3.11)."
  Write-Host "[setup] Install Python and ensure one of these works: py, python, or python3."
  Write-Host "[setup] Example: winget install Python.Python.3.12"
  exit 1
}

Write-Host "[setup] using interpreter: $($py.label) (version $($py.version))"
if ($py.version -ge [version]"3.14") {
  Write-Host "[setup] warning: Python 3.14 detected. Some wheels may be unavailable; 3.12/3.13 is recommended."
}

& $py.exe @($py.args) -m venv .venv
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
  Write-Host "[setup] venv python not found: $VenvPython"
  exit 1
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
  Write-Host "[setup] base dependency install failed."
  Write-Host "[setup] If this PC only has Python 3.14, install Python 3.12/3.13 and rerun."
  exit 1
}

$PyVersion = (& $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
$RapidOCRAvailable = $false
if ([version]$PyVersion -lt [version]"3.13") {
  & $VenvPython -m pip install "rapidocr-onnxruntime==1.2.3"
  if ($LASTEXITCODE -eq 0) {
    $RapidOCRAvailable = $true
  } else {
    Write-Host "[setup] rapidocr install failed. OCR fallback will use pytesseract."
  }
} else {
  Write-Host "[setup] python>=3.13 detected. installing rapidocr (new package)."
  & $VenvPython -m pip install "rapidocr>=3.7.0"
  if ($LASTEXITCODE -eq 0) {
    $RapidOCRAvailable = $true
  } else {
    Write-Host "[setup] rapidocr install failed. OCR fallback will use pytesseract."
  }
}

& $VenvPython -m pip install av
if ($LASTEXITCODE -ne 0) {
  Write-Host "[setup] av install failed. video decoding fallback will use OpenCV."
}

if (-not (Get-Command tesseract -ErrorAction SilentlyContinue)) {
  if ($RapidOCRAvailable) {
    Write-Host "[setup] tesseract not found, but rapidocr is installed. OCR can still run."
  } else {
    Write-Host "[setup] warning: tesseract binary not found and rapidocr is unavailable."
    Write-Host "[setup] install tesseract (example): winget install UB-Mannheim.TesseractOCR"
    Write-Host "[setup] then restart terminal so PATH updates are applied."
  }
}

$DownloadArgs = @("--output", ".\\models\\person_detector.onnx")
if ($env:MODEL_DOWNLOAD_INSECURE -eq "1") {
  $DownloadArgs += "--insecure"
  Write-Host "[setup] MODEL_DOWNLOAD_INSECURE=1 -> TLS verification disabled for curl fallback."
}

if (-not (Test-Path .\models\person_detector.onnx)) {
  Write-Host "[setup] model not found. trying auto-download..."
  & $VenvPython .\scripts\download_default_model.py @DownloadArgs
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[setup] auto-download failed. Place ONNX model at models/person_detector.onnx"
    Write-Host "[setup] tip: `$env:MODEL_DOWNLOAD_INSECURE='1'; .\\scripts\\setup_run_windows.cmd"
  }
}

Write-Host "[run] starting app"
& $VenvPython -m isac_labelr
