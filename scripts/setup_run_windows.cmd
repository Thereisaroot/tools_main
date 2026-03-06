@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_run_windows.ps1" %*
exit /b %errorlevel%
