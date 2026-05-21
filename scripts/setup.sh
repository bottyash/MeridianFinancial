#!/usr/bin/env bash
# scripts/setup.sh
# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap script for Linux / macOS developer environments.
#
# Usage:
#   chmod +x scripts/setup.sh
#   ./scripts/setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON_MIN="3.11"
VENV_DIR=".venv"
ENV_FILE=".env"
ENV_EXAMPLE=".env.example"

echo "════════════════════════════════════════════════════════════"
echo "  Meridian Financial — Development Environment Setup"
echo "════════════════════════════════════════════════════════════"

# ── 1. Check Python version ──────────────────────────────────────────────────
PYTHON_BIN=$(command -v python3.11 2>/dev/null || command -v python3 2>/dev/null || echo "")
if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: Python 3.11+ not found. Install it and retry." >&2
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_BIN" --version 2>&1 | awk '{print $2}')
echo "Python: $PYTHON_VERSION ($PYTHON_BIN)"

# ── 2. Create virtual environment ────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# ── 3. Activate venv ─────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "Activated: $VIRTUAL_ENV"

# ── 4. Upgrade pip ───────────────────────────────────────────────────────────
pip install --quiet --upgrade pip

# ── 5. Install dependencies ───────────────────────────────────────────────────
echo "Installing dependencies from requirements.txt ..."
pip install --quiet -r requirements.txt
echo "Dependencies installed."

# ── 6. Copy .env.example → .env (if not already present) ────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    echo "Created $ENV_FILE from $ENV_EXAMPLE — update it with your real values."
else
    echo "$ENV_FILE already exists — skipping copy."
fi

# ── 7. Create required directories ───────────────────────────────────────────
for dir in data/raw data/samples artifacts/features artifacts/models \
            chroma_store mlruns monitoring/reports; do
    mkdir -p "$dir"
done
echo "Directory structure verified."

# ── 8. Smoke-test imports ─────────────────────────────────────────────────────
echo "Running smoke tests ..."
python -c "from src.common.config import settings; print('  config OK:', settings)"
python -c "from src.common.logger import get_logger; l=get_logger('setup'); l.info('logger OK')"
echo "Smoke tests passed."

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Setup complete. Activate with: source $VENV_DIR/bin/activate"
echo "════════════════════════════════════════════════════════════"
