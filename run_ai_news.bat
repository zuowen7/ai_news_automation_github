@echo off
chcp 65001 > nul
echo ========================================
echo AI News Automation System
echo ========================================
echo.

cd /d "%~dp0"

echo Current directory: %CD%
echo Config file: config.json
echo.

python run.py

echo.
echo ========================================
echo Press any key to exit...
pause > nul
