# Installation

Solar Forecast Pro can be installed in three ways. **Docker is recommended** for production.

## Option 1 — Docker (recommended)

Requirements: Docker + Docker Compose v2.

```bash
git clone <your-repo-url> solar-forecast-pro
cd solar-forecast-pro
cp .env.example .env
docker compose up -d
```

That's it. Open the dashboard at:

- **Dashboard** → http://localhost:8501
- **API** → http://localhost:8000/docs

To shut down:
```bash
docker compose down
```

To enable PostgreSQL for CAMS training:
```bash
docker compose --profile training up -d
```

## Option 2 — One-command install (Linux/macOS)

Requirements: Python 3.10+ and `bash`.

```bash
git clone <your-repo-url> solar-forecast-pro
cd solar-forecast-pro
./install.sh
./run.sh
```

The installer creates a virtual environment in `.venv/`, installs all
dependencies, and initializes the SQLite database with a Budapest demo
location.

## Option 3 — Manual Python install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# Initialize the SQLite DB
python -c "from app.db.sqlite_manager import create_tables, seed_demo_location; create_tables(); seed_demo_location()"

# Run the dashboard
streamlit run solar_forecast/dashboard/app.py

# OR run the API
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

## Verifying the install

```bash
# Health check
curl http://localhost:8000/health

# Run pytest
pytest

# Sample forecast
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"lat": 47.5, "lon": 19.0, "capacity_kw": 5.0, "horizon_days": 1}'
```

## System requirements

| Item | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11+ |
| RAM | 1 GB | 2 GB |
| Disk | 500 MB | 2 GB (with cache) |
| Internet | Required for Open-Meteo | Required |

## Troubleshooting

**"Module not found" errors** — Make sure you activated the virtual
environment (`source .venv/bin/activate`).

**Streamlit won't start on port 8501** — Use `./run.sh --port=8502`.

**Docker build fails on `netcdf`** — Make sure you have at least 2 GB RAM
allocated to Docker.

See [faq.md](faq.md) for more.
