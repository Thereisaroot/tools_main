$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

& py -3.12 -V *> $null
if ($LASTEXITCODE -eq 0) {
  $PySelector = "py -3.12"
} else {
  $PySelector = "py -3"
}

Write-Host "[build] using interpreter: $PySelector"
Invoke-Expression "$PySelector -m venv .venv-build"
& .\.venv-build\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller
if ($LASTEXITCODE -ne 0) {
  Write-Host "[build] base dependency install failed."
  Write-Host "[build] install Python 3.12 and retry if using an unsupported interpreter."
  exit 1
}

$PyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([version]$PyVersion -lt [version]"3.13") {
  python -m pip install "rapidocr-onnxruntime==1.2.3"
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[build] rapidocr install failed. continuing without rapidocr."
  }
}
python -m pip install av
if ($LASTEXITCODE -ne 0) {
  Write-Host "[build] av install failed. continuing with OpenCV decoding fallback."
}

pyinstaller `
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
