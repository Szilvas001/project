# CAMS Atmospheric Data Ingestion

CAMS (Copernicus Atmosphere Monitoring Service) provides per-timestep aerosol optical depth (AOD), ozone, water vapour, boundary layer height (BLH), and speciated aerosols. These replace the continental-Europe climatology constants used in demo mode and significantly improve forecast accuracy.

---

## API Key Setup

1. Register at https://ads.atmosphere.copernicus.eu/profile (free)
2. Copy your UID and API key from the profile page
3. Set credentials in **either** `.env` or `~/.cdsapirc`:

**.env (recommended for Docker):**
```bash
CAMS_API_KEY=<uid>:<api-key>
```

**~/.cdsapirc (alternative for manual installs):**
```ini
url: https://ads.atmosphere.copernicus.eu/api
key: <uid>:<api-key>
```

The client reads `CADS_URL` / `CADS_KEY` env vars first, then falls back to `~/.cdsapirc`.

---

## Backfill CLI

Downloads historical CAMS data for a location and stores it in the `cams_atmospheric_forecast` table.

```bash
# Backfill 365 days for location ID 1
python -m solar_forecast.ingestion.cams.backfill \
  --location-id 1 \
  --days 365

# Backfill with custom config
python -m solar_forecast.ingestion.cams.backfill \
  --location-id 1 \
  --days 90 \
  --config path/to/config.yaml

# Dry run (no writes)
python -m solar_forecast.ingestion.cams.backfill \
  --location-id 1 \
  --days 30 \
  --dry-run
```

Backfill processes up to 50 missing forecast runs per execution (configurable via `backfill.max_per_run` in `config_default.yaml`).

---

## Live Fetch CLI

Downloads the latest CAMS forecast (0–48 h lead time) for a location.

```bash
# Fetch next 12 hours for location ID 1
python -m solar_forecast.ingestion.cams.live \
  --location-id 1 \
  --hours 12

# Fetch 48-hour forecast
python -m solar_forecast.ingestion.cams.live \
  --location-id 1 \
  --hours 48
```

CAMS issues two forecasts per day: initialised at 00:00 UTC (available ~10:15 UTC) and 12:00 UTC (available ~22:15 UTC).

---

## Automated Scheduler

Install a cron job to run the live fetch automatically:

```bash
python -c "from solar_forecast.cams_fetcher.scheduler import setup_cron; setup_cron()"
```

This installs two crontab entries:
- `15 10 * * * python -m solar_forecast.ingestion.cams.live` — 10:15 UTC daily
- `15 22 * * * python -m solar_forecast.ingestion.cams.live` — 22:15 UTC daily

The `CamsScheduler` class is also available as a daemon thread if you prefer in-process scheduling:

```python
from solar_forecast.cams_fetcher.scheduler import CamsScheduler
scheduler = CamsScheduler()
scheduler.start()   # daemon thread — stops with the main process
```

---

## Variables Collected

### Dataset: `cams-global-atmospheric-composition-forecasts`

**Surface composition** (`cams_surface` table):

| Variable | Unit | Used for |
|---|---|---|
| `total_column_ozone` | kg/m² → DU | SPECTRL2 ozone absorption |
| `total_column_water_vapour` | kg/m² → cm | SPECTRL2 water vapour band |
| `total_aerosol_optical_depth_469nm` | — | Ångström exponent α₁ |
| `total_aerosol_optical_depth_550nm` | — | Primary AOD input |
| `total_aerosol_optical_depth_670nm` | — | Ångström exponent α₁/α₂ |
| `total_aerosol_optical_depth_865nm` | — | Ångström exponent α₂ |
| `total_aerosol_optical_depth_1240nm` | — | Ångström exponent α₂ |
| `boundary_layer_height` | m | Aerosol mixing correction |
| `2m_temperature` | K → °C | Cell temperature supplement |
| `surface_pressure` | Pa → hPa | SPECTRL2 Rayleigh scattering |

**Aerosol species** (`cams_species` table):

| Variable | Used for |
|---|---|
| `dust_aerosol_optical_depth_550nm` | Mixed SSA / asymmetry |
| `black_carbon_aerosol_optical_depth_550nm` | Mixed SSA / asymmetry |
| `organic_matter_aerosol_optical_depth_550nm` | Mixed SSA / asymmetry |
| `sea_salt_aerosol_optical_depth_550nm` | Mixed SSA / asymmetry |
| `sulphate_aerosol_optical_depth_550nm` | Mixed SSA / asymmetry |

Ångström exponents α₁ (340–500 nm) and α₂ (500–1064 nm) are derived from the multi-wavelength AOD pairs by `solar_forecast/data_ingestion/cams_query.py`. Mixed SSA and asymmetry are computed from the speciated AOD fractions.

---

## Spatial Interpolation

CAMS data is delivered on a global grid. The fetcher applies bilinear interpolation (via `grib_processor.py` or `xarray`) to the target lat/lon ± `area_margin` (default 0.5°) and stores the interpolated point value in the database.

---

## Fallback Behaviour

When CAMS data is unavailable the pipeline uses continental-Europe climatology:

| Parameter | Climatology default |
|---|---|
| AOD 550 nm | 0.12 |
| Ångström α₁ | 1.30 |
| Ångström α₂ | 1.10 |
| SSA | 0.92 |
| Asymmetry g | 0.65 |
| Ozone | 310 DU |
| Precipitable water | 1.5 cm |
| Surface pressure | 1013.25 hPa |
