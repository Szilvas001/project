#!/usr/bin/env python
"""
Step 3 — Launch the Streamlit live-forecast dashboard.

Usage:
    python scripts/03_run_dashboard.py
    python scripts/03_run_dashboard.py --port 8502

The dashboard opens in the browser automatically.
No API keys or database are required for the live forecast view
(Open-Meteo is used for weather; the Kt model falls back to physics-only
if no trained model file exists at models/kt_xgb.joblib).
"""

import argparse
import subprocess
import sys
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "solar_forecast" / "dashboard" / "app.py"


def parse_args():
    p = argparse.ArgumentParser(description="Run the Solar Forecast dashboard.")
    p.add_argument("--port", type=int, default=8501, help="Streamlit port (default 8501)")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--no-browser", action="store_true", help="Do not open browser")
    return p.parse_args()


def main():
    args = parse_args()
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(APP_PATH),
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--server.headless", "true" if args.no_browser else "false",
        "--theme.base", "dark",
        "--theme.primaryColor", "#F4A503",
        "--theme.backgroundColor", "#0E1117",
        "--theme.secondaryBackgroundColor", "#1E1E2E",
        "--theme.textColor", "#FAFAFA",
    ]
    print(f"Starting dashboard on http://{args.host}:{args.port}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
