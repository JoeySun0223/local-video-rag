@echo off
cd /d "%~dp0"
call venv\Scripts\python.exe rag.py start
if errorlevel 1 pause
