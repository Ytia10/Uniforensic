@echo off
REM UniForensic Workbench - one-time setup WITHOUT Docker (Windows 10/11, Python 3.12).
REM Re-uses .venv if you already created it for N1-N5; otherwise creates it.
cd /d "%~dp0"
if not exist .venv (
  py -3.12 -m venv .venv || (echo Python 3.12 not found - install it from python.org & exit /b 1)
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
)
.venv\Scripts\python -m pip install -r requirements_app.txt || exit /b 1
.venv\Scripts\python -m pip check
.venv\Scripts\python app_selftest.py --quick
echo.
echo Setup finished. Start the app with run_app.bat
