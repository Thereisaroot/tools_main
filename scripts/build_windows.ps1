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
  Write-Host "[build] No suitable Python runtime found (need >= 3.11)."
  Write-Host "[build] Install Python and ensure one of these works: py, python, or python3."
  Write-Host "[build] Example: winget install Python.Python.3.12"
  exit 1
}

Write-Host "[build] using interpreter: $($py.label) (version $($py.version))"
if ($py.version -ge [version]"3.14") {
  Write-Host "[build] warning: Python 3.14 detected. Some wheels may be unavailable; 3.12/3.13 is recommended."
}

& $py.exe @($py.args) -m venv .venv-build
$VenvPython = Join-Path $Root ".venv-build\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
  Write-Host "[build] venv python not found: $VenvPython"
  exit 1
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt pyinstaller
if ($LASTEXITCODE -ne 0) {
  Write-Host "[build] base dependency install failed."
  Write-Host "[build] If this PC only has Python 3.14, install Python 3.12/3.13 and rerun."
  exit 1
}

$PyVersion = (& $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ([version]$PyVersion -lt [version]"3.13") {
  & $VenvPython -m pip install "rapidocr-onnxruntime==1.2.3"
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[build] rapidocr install failed. continuing without rapidocr."
  }
}
& $VenvPython -m pip install av
if ($LASTEXITCODE -ne 0) {
  Write-Host "[build] av install failed. continuing with OpenCV decoding fallback."
}

& $VenvPython -m PyInstaller `
  --noconfirm `
  --clean `
  --windowed `
  --onedir `
  --name isac_labelr_portable `
  isac_labelr/__main__.py

if (Test-Path .\models) {
  New-Item -ItemType Directory -Force -Path .\dist\isac_labelr_portable\models | Out-Null
  Copy-Item .\models\* .\dist\isac_labelr_portable\models -Recurse -Force
}

Write-Host "Build complete: dist/isac_labelr_portable"
