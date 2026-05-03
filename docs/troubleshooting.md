# Troubleshooting

---

## CAMS / Copernicus

**"No CAMS key found"** — System falls back to climatology automatically. To enable CAMS:

```bash
# Option 1: .env
CAMS_API_KEY=<uid>:<key>

# Option 2: ~/.cdsapirc
url: https://ads.atmosphere.copernicus.eu/api
key: <uid>:<key>
```

Register free at https://ads.atmosphere.copernicus.eu/profile. Verify with:
```bash
python -c "import cdsapi; c = cdsapi.Client(); print('OK')"
```

**"CAMS request queued / timeout"** — The Copernicus ADS queue can take 30–120 minutes for large requests. Use `--days 30` for initial testing. Monitor at https://ads.atmosphere.copernicus.eu/requests.

**"CAMS data empty after download"** — Increase `area_margin` in `solar_forecast/cams_fetcher/config_default.yaml` from `0.5` to `1.0` for coastal or island locations.

---

## GRIB / pygrib

**"pygrib not found" / "eccodes library not found"** — Expected without the system dependency. Either install it:

```bash
sudo apt install libeccodes-dev && pip install pygrib>=2.1.5   # Ubuntu
brew install eccodes && pip install pygrib                      # macOS
```

Or keep the default NetCDF output format — `pygrib` is only needed for GRIB-format CAMS output.

---

## PostgreSQL / psycopg2

**"psycopg2 not installed"**
```bash
pip install psycopg2-binary>=2.9.0
```

**"could not connect to server"** — Verify PostgreSQL is running and `.env` credentials are correct:
```bash
docker compose --profile training up -d
psql -h $PGHOST -U $PGUSER -d $PGDATABASE -c "SELECT 1"
```

**"relation does not exist"** — Create tables:
```bash
python -c "from solar_forecast.data_ingestion.db_manager import create_tables; create_tables()"
```

---

## Forecast Returns All Zeros

**Night-only horizon** — `horizon_days=1` after sunset returns zeros. Use `horizon_days=2`.

**Open-Meteo unreachable** — Pipeline falls back to demo CSV for a different location. Check connectivity:
```bash
curl "https://api.open-meteo.com/v1/forecast?latitude=47.5&longitude=19.0&hourly=shortwave_radiation&forecast_days=1"
```

**Swapped coordinates** — `lat` must be -90 to 90, `lon` must be -180 to 180. Swapping them (e.g. lat=19, lon=47) produces zero irradiance.

**Polar night** — Locations above ~65°N in winter have zero solar irradiance. This is correct.

---

## SPECTRL2 / Physics Errors

**"SPECTRL2 failed, using simplified_solis"** — Warning, not a fatal error. Check atmospheric input ranges:
- `ozone_du`: 200–400 DU
- `precipitable_water`: 0.1–10 cm
- `aod_550nm`: 0.01–2.0

**"SolarPosition: sun below horizon"** — Normal. Night rows are automatically zeroed.

---

## XGBoost / AI Model

**"kt_xgb.joblib not found"** — Physics-only mode is used. To train:
```bash
python scripts/01_download_cams.py --start 2022-01-01 --end 2023-12-31
python scripts/02_train_kt_model.py
```
Requires CAMS key and PostgreSQL.

**"HistoricalGHITrainer: R² below 0.75"** — Extend the training period (365 days minimum recommended) or verify Open-Meteo historical data has no gaps > 48 h.

**XGBoost not installed** — Falls back to `sklearn.ensemble.HistGradientBoostingRegressor` automatically. Install with `pip install xgboost>=2.0.0`.

---

## Dashboard

**Port in use:**
```bash
./run.sh --port 8502
```

**Permission denied on run.sh / install.sh:**
```bash
chmod +x install.sh run.sh
```

**Real-Time tab not auto-refreshing** — Verify `streamlit-autorefresh>=1.0.1` is installed. Some browsers throttle background tabs; keep the tab in the foreground.

---

## Docker

**Build fails on netCDF4** — Increase Docker Desktop memory to 2 GB: Settings → Resources → Memory.

**Services restart immediately** — Check logs:
```bash
docker compose logs api
docker compose logs dashboard
```
Most common cause: missing `.env` file. Run `cp .env.example .env` first.

**Data lost after restart** — Named volumes (`solar_data`, `solar_models`) survive `docker compose down`. Data is only wiped by `docker compose down -v`. Never use `-v` unless intentionally resetting state.
