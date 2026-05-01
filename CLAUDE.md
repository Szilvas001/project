# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
# or install as editable package
pip install -e .

# Run tests
pytest

# Run a single test
pytest tests/test_api.py::test_health

# Start Streamlit dashboard (http://localhost:8501)
./run.sh

# Start FastAPI backend (http://localhost:8000, docs at /docs)
./run.sh --api

# Start both services
./run.sh --all

# Docker (recommended for production)
cp .env.example .env
docker compose up -d

# Advanced: download CAMS training data (requires CAMS_API_KEY)
python scripts/01_download_cams.py --start 2021-01-01 --end 2024-12-31

# Advanced: train XGBoost Kt model
python scripts/02_train_kt_model.py
```

## Architecture

This is a physics-based + AI hybrid solar PV production forecasting system. It has three entry points: a Streamlit dashboard, a FastAPI REST API, and direct Python usage via `run_demo_forecast()`.

### Forecast Pipeline

The single entry point for all forecasts is `run_demo_forecast()` in `solar_forecast/demo/pipeline.py`. The pipeline runs these steps in order:

1. **Open-Meteo weather** — fetches live hourly GHI/DNI/DHI/cloud cover/temperature (free, no key)
2. **SPECTRL2 clear-sky** (`solar_forecast/clearsky/spectrl2_model.py`) — pvlib Bird & Riordan spectral model; falls back to `simplified_solis` if it fails
3. **Physics Kt** (`solar_forecast/allsky/physics_kt.py`) — Delta-Eddington two-stream approximation combining cloud optical depth and aerosol excess attenuation
4. **Optional AI Kt** (`solar_forecast/allsky/ai_trainer.py`) — XGBoost model (21 features, trained on CAMS data); blended as `0.4 × Kt_phys + 0.6 × Kt_ai`
5. **Perez transposition** — GHI/DNI/DHI → POA irradiance on the tilted panel surface
6. **IAM correction** (`solar_forecast/production/iam_model.py`) — ASHRAE, Martin-Ruiz, or Fresnel angle-of-incidence models
7. **NOCT cell temperature** — `T_cell = T_air + (NOCT − 20) / 800 × G_POA`
8. **DC → AC power** — temperature coefficient + system losses (inverter × wiring × soiling)

The `AllSkyModel` class in `solar_forecast/allsky/hybrid_model.py` wraps steps 3–4 as an orchestrator; `physics_weight` config key (`α`) controls the blend (default 0.40).

### Service Layer

- **FastAPI** (`app/api/main.py`) — three routers: `health`, `locations`, `forecast`
  - `POST /forecast` — one-off forecast by coordinates
  - `GET /forecast/{location_id}` — forecast for a saved location, results cached in SQLite by date
  - `GET /export/csv` — CSV download of a cached forecast
- **Streamlit** (`solar_forecast/dashboard/app.py`) — three UI tiers: Basic (city + kW), Pro (tilt/azimuth/horizon/CSV), Expert (SR upload, IAM, AI toggle)
- Both the dashboard and the API call `run_demo_forecast()` directly

### Data Storage

- **SQLite** (`data/solar_forecast.db`, WAL mode) — `app/db/sqlite_manager.py` manages two tables:
  - `locations` — saved PV system configurations
  - `forecasts` — daily cached forecast results (upsert keyed by `location_id + forecast_date`)
- **PostgreSQL** — optional, only used when downloading CAMS historical training data (`solar_forecast/data_ingestion/db_manager.py`)
- **Models** — trained XGBoost model serialised with `joblib` to `models/kt_xgb.joblib`

### Configuration

- `config.yaml` — all non-secret settings (location defaults, PV system parameters, model paths, CAMS date ranges)
- `.env` — secrets and overrides (`CAMS_API_KEY`, `PGPASSWORD`, `DEMO_MODE`); loaded automatically by `python-dotenv`
- Key defaults: `tilt=None` auto-computes as `abs(lat) × 0.76`; `azimuth=None` defaults to 180° south

### Testing

`tests/conftest.py` monkeypatches `sqlite_manager.DB_PATH` to a temporary file for every test so tests never touch the real database. API tests mock `run_demo_forecast()` to avoid live network calls. Physics tests patch `_fetch_openmeteo` to return an empty DataFrame and verify the pipeline degrades gracefully to clear-sky fallback.
