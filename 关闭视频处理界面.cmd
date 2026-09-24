@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo Python environment not found: venv\Scripts\python.exe
  pause
  exit /b 1
)
"venv\Scripts\python.exe" -m beginner_webui.stop
set "app_exit_code=%errorlevel%"
if not "%app_exit_code%"=="0" pause
exit /b %app_exit_code%
