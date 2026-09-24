@echo off
setlocal
title Video Processing UI
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo Python environment not found: venv\Scripts\python.exe
  pause
  exit /b 1
)
echo Starting Video Processing UI. Keep this window open.
"venv\Scripts\python.exe" -m beginner_webui
set "app_exit_code=%errorlevel%"
if not "%app_exit_code%"=="0" (
  echo.
  echo Startup failed. Review the error above.
  pause
)
exit /b %app_exit_code%
