@echo off
REM scripts\setup.bat
REM ─────────────────────────────────────────────────────────────────────────────
REM Bootstrap script for Windows developer environments.
REM
REM Usage (from repo root):
REM   scripts\setup.bat
REM ─────────────────────────────────────────────────────────────────────────────

setlocal EnableDelayedExpansion

set VENV_DIR=.venv
set ENV_FILE=.env
set ENV_EXAMPLE=.env.example

echo ════════════════════════════════════════════════════════════
echo   Meridian Financial — Development Environment Setup
echo ════════════════════════════════════════════════════════════

REM ── 1. Check Python ──────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found on PATH. Install Python 3.11+ and retry.
    exit /b 1
)
python --version

REM ── 2. Create virtual environment ────────────────────────────────────────────
if not exist "%VENV_DIR%\" (
    echo Creating virtual environment in %VENV_DIR% ...
    python -m venv %VENV_DIR%
) else (
    echo Virtual environment already exists at %VENV_DIR%
)

REM ── 3. Activate venv ─────────────────────────────────────────────────────────
call %VENV_DIR%\Scripts\activate.bat
echo Activated: %VIRTUAL_ENV%

REM ── 4. Upgrade pip ───────────────────────────────────────────────────────────
python -m pip install --quiet --upgrade pip

REM ── 5. Install dependencies ───────────────────────────────────────────────────
echo Installing dependencies from requirements.txt ...
pip install --quiet -r requirements.txt
echo Dependencies installed.

REM ── 6. Copy .env.example → .env (if not already present) ────────────────────
if not exist "%ENV_FILE%" (
    copy "%ENV_EXAMPLE%" "%ENV_FILE%" >nul
    echo Created %ENV_FILE% from %ENV_EXAMPLE% — update it with your real values.
) else (
    echo %ENV_FILE% already exists — skipping copy.
)

REM ── 7. Create required directories ───────────────────────────────────────────
for %%d in (
    data\raw
    data\samples
    artifacts\features
    artifacts\models
    chroma_store
    mlruns
    monitoring\reports
) do (
    if not exist "%%d\" mkdir "%%d"
)
echo Directory structure verified.

REM ── 8. Smoke-test imports ─────────────────────────────────────────────────────
echo Running smoke tests ...
python -c "from src.common.config import settings; print('  config OK:', settings)"
if errorlevel 1 (
    echo ERROR: config smoke test failed.
    exit /b 1
)
python -c "from src.common.logger import get_logger; l=get_logger('setup'); l.info('logger OK')"
if errorlevel 1 (
    echo ERROR: logger smoke test failed.
    exit /b 1
)
echo Smoke tests passed.

echo.
echo ════════════════════════════════════════════════════════════
echo   Setup complete. Activate with: %VENV_DIR%\Scripts\activate
echo ════════════════════════════════════════════════════════════

endlocal
