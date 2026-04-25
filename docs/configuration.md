# Configuration

All configuration is via two files: **`.env`** (secrets) and
**`config.yaml`** (non-secret defaults for the CLI scripts).

## `.env` (secrets and runtime flags)

Copy `.env.example` to `.env` and edit:

```bash
# Demo mode — set to 'false' to require CAMS + PostgreSQL for full training
DEMO_MODE=true

# Copernicus ADS / CAMS — only needed for AI training
CAMS_API_KEY=00000:xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# PostgreSQL — only needed when training the AI model from CAMS history
PGHOST=localhost
PGPORT=5432
PGDATABASE=solar_forecast
PGUSER=solar
PGPASSWORD=changeme

# Dashboard defaults
DEFAULT_LAT=47.4979
DEFAULT_LON=19.0402
DEFAULT_LOCATION_NAME=Budapest

# API
API_HOST=0.0.0.0
API_PORT=8000
```

## `config.yaml` (CLI defaults)

Used by `scripts/01_download_cams.py` and `scripts/02_train_kt_model.py`.

```yaml
location:
  lat: 47.498
  lon: 19.040
  altitude: 120
  name: Budapest
  timezone: Europe/Budapest

system:
  capacity_kw: 5.0
  tilt: null        # null = lat × 0.76
  azimuth: null     # null = 180 (N hemi) or 0 (S hemi)
  technology: mono_si

cams:
  training_start: 2022-01-01
  training_end:   2023-12-31
  fetch_spectral_aod: true
  fetch_species_aod:  true
  fetch_optional:     true

model:
  kt_model_path: models/kt_xgb.joblib
  n_cv_folds: 5

database:
  host: localhost
  port: 5432
  name: solar_forecast
  user: solar
  password: ${PGPASSWORD}
```

## Cell technology codes

| Code | Description | Default temperature coefficient |
|---|---|---|
| `mono_si` | Mono-crystalline silicon | -0.45 %/°C |
| `poly_si` | Poly-crystalline silicon | -0.45 %/°C |
| `cdte`    | Cadmium telluride (thin-film) | -0.25 %/°C |
| `cigs`    | CIGS / CIS (thin-film) | -0.36 %/°C |
| `hit`     | HIT / heterojunction | -0.25 %/°C |

## Tilt and azimuth

If you leave **tilt** and **azimuth** blank when creating a location:
- **tilt** defaults to `|latitude| × 0.76` (annual-yield optimum)
- **azimuth** defaults to `180°` for the northern hemisphere, `0°` (north)
  for the southern hemisphere

## Storage paths

| Path | Purpose |
|---|---|
| `data/solar_forecast.db` | SQLite — locations + cached forecasts |
| `models/kt_xgb.joblib`   | (Optional) Trained XGBoost Kt model |
| `.cache/openmeteo`       | Open-Meteo response cache (30-min TTL) |

## Timezones

All forecasts are computed and stored in **UTC** internally. The dashboard
converts to your selected IANA timezone for display only. CAMS and
Open-Meteo are always requested in UTC.
