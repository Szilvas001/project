# Advanced — Train the XGBoost Kt Model

The default forecast pipeline is **physics-only** (pvlib spectrl2 +
climatological aerosol fallbacks) and works without any API key.

For maximum accuracy you can train an **XGBoost regressor** that learns a
clearness-index correction from full CAMS atmospheric features.

## What you get

- ~10–20% lower RMSE on cloudy/aerosol-heavy days
- Better handling of regional aerosol composition (dust, BC, sulfate, …)
- Physics-aware features: SSA, asymmetry parameter, Ångström exponents,
  PM2.5/PM10, BLH, low-cloud fraction, …

## Prerequisites

1. **Free CAMS API key** — register at
   https://ads.atmosphere.copernicus.eu/profile
2. **PostgreSQL** — for caching ~1 GB of CAMS history per year per location
3. **Disk space** — ~2 GB
4. **Time** — initial download takes 30 minutes to a few hours

## Step 1 — Configure

Edit `.env`:
```bash
CAMS_API_KEY=00000:xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
PGHOST=localhost
PGDATABASE=solar_forecast
PGUSER=solar
PGPASSWORD=changeme
DEMO_MODE=false
```

Edit `config.yaml`:
```yaml
location:
  lat: 47.498
  lon: 19.040
  altitude: 120

cams:
  training_start: 2022-01-01
  training_end:   2023-12-31
  fetch_spectral_aod: true
  fetch_species_aod:  true
```

Start PostgreSQL via Docker Compose:
```bash
docker compose --profile training up -d postgres
```

## Step 2 — Download CAMS history

```bash
python scripts/01_download_cams.py
```

This pulls:
- **CAMS EAC4** reanalysis (3-hourly, full atmospheric state):
  AOD at 469/550/670/865/1240 nm, speciated AOD (dust/BC/OM/SS/SO₄),
  ozone, water vapour, pressure, cloud cover, T2m, BLH, surface albedo,
  PM2.5, PM10, CO, NO₂
- **CAMS solar radiation** service (hourly all-sky + clear-sky GHI/DHI/DNI)

Both datasets land in PostgreSQL (`cams_atmo` and `cams_radiation` tables).

## Step 3 — Train

```bash
python scripts/02_train_kt_model.py --cv 5
```

This:
1. Loads CAMS atmo + radiation from PostgreSQL
2. Computes spectrl2 clear-sky reference (with full physics: SSA, g,
   ALPHA1/ALPHA2, Hänel-corrected AOD, …)
3. Builds a 21-feature training set
4. Fits XGBoost (RMSE objective, early stopping)
5. Runs 5-fold cross-validation
6. Saves to `models/kt_xgb.joblib`

Typical metrics on a clean dataset:

| Metric | Value |
|---|---|
| MAE | 0.05–0.07 |
| RMSE | 0.07–0.10 |
| R² | 0.88–0.94 |

## Step 4 — Use

The dashboard and API automatically pick up the model file at
`models/kt_xgb.joblib`. No restart needed; the next forecast uses it.

## Feature set (21 features)

**Core (13)**: solar zenith/azimuth, ETR (Spencer 1971), AOD550, total
ozone, precipitable water, surface pressure, temperature, RH, wind, hour
of day (sin/cos), day of year (sin/cos).

**Extended (8)**: SSA (normalized), asymmetry factor (normalized),
Ångström α1 (340–500 nm), Ångström α2 (500–1064 nm), PM2.5 (log),
boundary layer height, total cloud composite, low-cloud fraction.

## CLI arguments

```bash
python scripts/02_train_kt_model.py \
  --start 2022-01-01 \
  --end   2023-12-31 \
  --cv    5 \
  --output models/kt_custom.joblib
```

## Re-training

After collecting a new month of CAMS data, re-run step 2 + 3. The model
file is overwritten in place.
