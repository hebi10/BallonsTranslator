@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_local.ps1" %*
if errorlevel 1 (
    echo.
    echo Local launch failed. Read the error above and retry after correcting it.
    pause
    exit /b 1
)
endlocal
