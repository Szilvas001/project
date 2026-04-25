#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# Solar Forecast Pro — Launch Script
# ─────────────────────────────────────────────────────────────────────────
set -e

VENV_DIR=".venv"
DASHBOARD_PORT=${DASHBOARD_PORT:-8501}
API_PORT=${API_PORT:-8000}

# Activate virtual environment if present
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
fi

# Load .env
if [ -f ".env" ]; then
    set -a
    # shellcheck source=/dev/null
    source ".env"
    set +a
fi

# Parse args
MODE="dashboard"
for arg in "$@"; do
    case $arg in
        --api)    MODE="api" ;;
        --all)    MODE="all" ;;
        --port=*) DASHBOARD_PORT="${arg#*=}" ;;
    esac
done

case $MODE in
    dashboard)
        echo "Starting Solar Forecast Pro dashboard on http://localhost:$DASHBOARD_PORT"
        streamlit run solar_forecast/dashboard/app.py \
            --server.port "$DASHBOARD_PORT" \
            --server.address 0.0.0.0 \
            --server.headless true \
            --theme.base dark \
            --theme.primaryColor "#F4A503" \
            --theme.backgroundColor "#0E1117" \
            --theme.secondaryBackgroundColor "#1E1E2E" \
            --theme.textColor "#FAFAFA"
        ;;
    api)
        echo "Starting Solar Forecast Pro API on http://localhost:$API_PORT"
        echo "  Docs: http://localhost:$API_PORT/docs"
        uvicorn app.api.main:app \
            --host 0.0.0.0 \
            --port "$API_PORT" \
            --reload
        ;;
    all)
        echo "Starting dashboard on :$DASHBOARD_PORT  and  API on :$API_PORT"
        streamlit run solar_forecast/dashboard/app.py \
            --server.port "$DASHBOARD_PORT" \
            --server.address 0.0.0.0 \
            --server.headless true \
            --theme.base dark \
            --theme.primaryColor "#F4A503" \
            --theme.backgroundColor "#0E1117" \
            --theme.secondaryBackgroundColor "#1E1E2E" \
            --theme.textColor "#FAFAFA" &
        DASHBOARD_PID=$!

        uvicorn app.api.main:app \
            --host 0.0.0.0 \
            --port "$API_PORT" &
        API_PID=$!

        trap "kill $DASHBOARD_PID $API_PID 2>/dev/null; exit" INT TERM
        wait
        ;;
esac
