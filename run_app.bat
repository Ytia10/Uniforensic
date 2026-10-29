@echo off
REM UniForensic Workbench - start WITHOUT Docker. Opens http://localhost:8000
cd /d "%~dp0"
start "" http://localhost:8000
.venv\Scripts\python -m uvicorn ufapp.main:app --port 8000
