@echo off
REM UniForensic Workbench - one-time setup of the automated browser test (ui_test.py). Not needed to run the app.
cd /d "%~dp0"
if not exist .venv (echo Run setup_app.bat first. & exit /b 1)
.venv\Scripts\python -m pip install -r requirements_test.txt || exit /b 1
.venv\Scripts\python -m pip check
.venv\Scripts\python -m playwright install chromium || exit /b 1
echo.
echo Test setup finished. Run: .venv\Scripts\python ui_test.py
