#!/usr/bin/env python
"""
Solar Forecast Pro — default entry point for Docker.

Starts the Streamlit dashboard on port 8501.
Set LAUNCH_API=true to start the FastAPI backend instead.
"""

import os
import subprocess
import sys

def main():
    mode = os.getenv("LAUNCH_MODE", "dashboard").lower()

    if mode == "api":
        port = int(os.getenv("API_PORT", "8000"))
        print(f"Starting API on http://0.0.0.0:{port}")
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "app.api.main:app",
            "--host", "0.0.0.0",
            "--port", str(port),
        ])
    else:
        port = int(os.getenv("DASHBOARD_PORT", "8501"))
        print(f"Starting dashboard on http://0.0.0.0:{port}")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "solar_forecast/dashboard/app.py",
            "--server.port", str(port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--theme.base", "dark",
            "--theme.primaryColor", "#F4A503",
            "--theme.backgroundColor", "#0E1117",
            "--theme.secondaryBackgroundColor", "#1E1E2E",
            "--theme.textColor", "#FAFAFA",
        ])


if __name__ == "__main__":
    main()
