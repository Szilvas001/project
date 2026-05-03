# Installation

**AI Solar Production Forecast SaaS — Physics + CAMS + Open-Meteo**

---

## Docker Quickstart (Recommended)

Requires Docker Engine 24+ and Docker Compose v2.

```bash
git clone <repo-url> solar-forecast && cd solar-forecast
cp .env.example .env          # edit secrets — see Environment Variables below
docker compose up -d
```

| Service | URL |
|---|---|
| Streamlit dashboard | http://localhost:8501 |
| FastAPI backend | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |

Both services share named Docker volumes (`solar_data`, `solar_models`, `solar_cache`), so data persists across restarts.

### Enable PostgreSQL (CAMS training data only)

```bash
docker compose --profile training up -d   # adds postgres:16 on port 5432
```

PostgreSQL is not needed for normal operation. It is only required when downloading CAMS historical atmospheric data and training the XGBoost Kt model.

---

## Manual Install

**Requirements:** Python 3.10+, pip 23+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or editable install for development:
pip install -e .
```

### Optional: GRIB ingestion (pygrib)

```bash
sudo apt install libeccodes-dev       # Ubuntu / Debian
pip install pygrib>=2.1.5
```

Only needed if you configure the CAMS fetcher for GRIB output format. Default is NetCDF, which works without `pygrib`.

---

## Environment Variables

Copy `.env.example` to `.env`. All variables are optional in demo mode.

| Variable | Default | Description |
|---|---|---|
| `DEMO_MODE` | `true` | Skip CAMS/PostgreSQL; use climatology fallbacks |
| `CAMS_API_KEY` | — | Copernicus ADS key for CAMS data |
| `PGHOST` | `localhost` | PostgreSQL host |
| `PGDATABASE` | `solar_forecast` | PostgreSQL database |
| `PGUSER` | `solar` | PostgreSQL user |
| `PGPASSWORD` | `changeme` | PostgreSQL password |
| `DEFAULT_LAT` | `47.4979` | Dashboard startup latitude |
| `DEFAULT_LON` | `19.0402` | Dashboard startup longitude |
| `API_PORT` | `8000` | FastAPI listen port |

---

## Run Services Manually

```bash
./run.sh                  # dashboard only (port 8501)
./run.sh --api            # API only (port 8000)
./run.sh --all            # both services

# Or directly:
streamlit run solar_forecast/dashboard/app.py --server.port 8501
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Verify Installation

```bash
pytest                                         # full test suite (~30 s)
pytest tests/test_api.py::test_health          # single health check test
curl http://localhost:8000/health              # API liveness
curl http://localhost:8000/locations           # list saved locations
```

## System Requirements

| Item | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11+ |
| RAM | 1 GB | 2 GB |
| Disk | 500 MB | 2 GB (with CAMS cache) |
| Internet | Required (Open-Meteo) | Required |
