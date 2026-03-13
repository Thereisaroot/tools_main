@echo off
setlocal

py -3 -c "import serial" >nul 2>&1
if errorlevel 1 (
    echo Installing dependency: pyserial
    py -3 -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install dependencies.
        exit /b 1
    )
)

py -3 serial_text_chat.py
