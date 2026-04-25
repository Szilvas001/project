#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# Solar Forecast Pro — Installation Script
# ─────────────────────────────────────────────────────────────────────────
set -e

PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║         Solar Forecast Pro — Installer           ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Check Python version
PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python 3.10+ required (found $PY_VERSION)"
    exit 1
fi

echo "✓  Python $PY_VERSION found"

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "→  Creating virtual environment in $VENV_DIR …"
    $PYTHON -m venv "$VENV_DIR"
fi

# Activate
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    # Windows Git Bash
    # shellcheck source=/dev/null
    source "$VENV_DIR/Scripts/activate"
fi

echo "→  Installing dependencies …"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# Create data directories
mkdir -p data models .cache

# Copy env template if .env doesn't exist
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓  Created .env from .env.example (demo mode enabled by default)"
fi

# Initialize the SQLite database
$PYTHON -c "
from app.db.sqlite_manager import create_tables, seed_demo_location
create_tables()
seed_demo_location()
print('✓  SQLite database initialized with demo location')
"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║              Installation complete!              ║"
echo "║                                                  ║"
echo "║  Run the dashboard:  ./run.sh                    ║"
echo "║  Run the API:        ./run.sh --api              ║"
echo "║  Run both:           ./run.sh --all              ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
