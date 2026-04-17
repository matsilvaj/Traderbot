@echo off
setlocal

cd /d "%~dp0"

where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw -m traderbot.launcher_entry
    exit /b 0
)

where pyw >nul 2>nul
if %errorlevel%==0 (
    start "" pyw -m traderbot.launcher_entry
    exit /b 0
)

start "" python -m traderbot.launcher_entry
