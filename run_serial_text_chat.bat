@echo off
setlocal

pushd "%~dp0"
if errorlevel 1 exit /b 1

echo Updating ShookLink
git pull --ff-only
if errorlevel 1 (
    echo Failed to update ShookLink.
    popd
    exit /b 1
)

py -3 -c "import shooklink, PySide6, serial, cryptography, pyte" >nul 2>&1
if errorlevel 1 (
    echo Installing ShookLink dependencies from requirements.txt
    py -3 -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install dependencies.
        popd
        exit /b 1
    )
)

py -3 -m shooklink %*
set "SHOOKLINK_EXIT_CODE=%errorlevel%"
popd
exit /b %SHOOKLINK_EXIT_CODE%
