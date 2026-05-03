# Data Pipeline — Feature Frame & Storage

The forecast engine consumes a unified feature frame built by `solar_forecast/features/builder.py` from two independent ingestion streams: CAMS atmospheric data and Open-Meteo weather. The pipeline degrades gracefully when either source is unavailable.

---

## Feature Frame Priority

The builder resolves atmospheric inputs in this order:

| Priority | Condition | Atmospheric source | Weather source |
|---|---|---|---|
| 1 (best) | CAMS in DB + Open-Meteo live | CAMS per-timestep AOD/ozone/SSA/BLH | Open-Meteo GHI/DNI/DHI/T |
| 2 | Open-Meteo live + climatology | Continental-Europe climatology constants | Open-Meteo GHI/DNI/DHI/T |
| 3 | Open-Meteo live only | Climatology + Hänel hygroscopic AOD | Open-Meteo GHI/DNI/DHI/T |
| 4 (demo) | No live data | Climatology constants | Demo CSV (`demo-data/`) |

Priority 1 gives the most accurate physics because CAMS provides per-timestep AOD at 469/550/670/865/1240 nm, which drives both the SPECTRL2 clear-sky model and the Delta-Eddington Kt cloud model.

The `_resolve_atmosphere()` function in `solar_forecast/demo/pipeline.py` implements this logic. The returned `source` key (`"cams"` or `"climatology"`) is exposed in the API `atmosphere` field and the confidence model.

---

## SQLite Database Schema

Default path: `data/solar_forecast.db` (WAL mode). Managed by `app/db/sqlite_manager.py`.

### `locations`

PV system registry.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `name` | TEXT | 1–120 chars |
| `lat` | REAL | -90 to 90 |
| `lon` | REAL | -180 to 180 |
| `altitude` | REAL | metres, default 0.0 |
| `capacity_kw` | REAL | DC installed capacity |
| `tilt` | REAL | NULL → auto (`|lat| × 0.76`) |
| `azimuth` | REAL | NULL → auto (180° north hemi) |
| `technology` | TEXT | `mono_si` / `poly_si` / `cdte` / `cigs` / `hit` |
| `timezone` | TEXT | IANA timezone name |
| `created_at` | TEXT | UTC ISO 8601 |
| `updated_at` | TEXT | UTC ISO 8601 |

### `forecasts`

Cached daily forecast results. Unique index on `(location_id, forecast_date)`.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `location_id` | INTEGER FK | References `locations.id` CASCADE DELETE |
| `forecast_date` | TEXT | `YYYY-MM-DD` |
| `generated_at` | TEXT | UTC ISO 8601 |
| `horizon_days` | INTEGER | 1–14 |
| `payload_json` | TEXT | JSON array of `HourlyPoint` objects |
| `summary_json` | TEXT | JSON object: today/tomorrow/7d kWh, peak, CF, cloud loss |

### `cams_atmospheric_forecast`

CAMS surface composition per reference time and lead time hour.

| Column | Type | Notes |
|---|---|---|
| `location_id` | INTEGER | |
| `reference_time` | TEXT | CAMS run init time (UTC) |
| `forecast_hours` | INTEGER | Lead time in hours (0–48) |
| `valid_time` | TEXT | `reference_time + forecast_hours` |
| `aod_469nm` … `aod_1240nm` | REAL | AOD at each wavelength |
| `ozone_kg_m2` | REAL | Total column ozone |
| `water_vapour_kg_m2` | REAL | Total column water vapour |
| `surface_pressure_pa` | REAL | |
| `blh_m` | REAL | Boundary layer height |

### `openmeteo_forecast`

Live Open-Meteo weather per location and timestamp.

| Column | Type | Notes |
|---|---|---|
| `location_id` | INTEGER | |
| `timestamp_utc` | TEXT | UTC ISO 8601 |
| `ghi_wm2` | REAL | Shortwave radiation |
| `dni_wm2` | REAL | Direct normal irradiance |
| `dhi_wm2` | REAL | Diffuse radiation |
| `temp_c` | REAL | 2 m air temperature |
| `cloud_cover_frac` | REAL | 0–1 |
| `rh_pct` | REAL | Relative humidity % |
| `pressure_hpa` | REAL | Surface pressure |

### `model_feature_frame`

Merged CAMS + Open-Meteo features used for AI model training and inference.

| Column | Type | Notes |
|---|---|---|
| `location_id` | INTEGER | |
| `timestamp_utc` | TEXT | |
| `aod_550nm` | REAL | Merged from CAMS (or climatology) |
| `angstrom_alpha1` | REAL | Derived from 469/550/670 nm AODs |
| `angstrom_alpha2` | REAL | Derived from 670/865/1240 nm AODs |
| `ssa_mix` | REAL | Species-weighted SSA |
| `asym_mix` | REAL | Species-weighted asymmetry |
| `ozone_du` | REAL | Converted from kg/m² |
| `precipitable_water` | REAL | cm |
| `ghi_wm2` | REAL | Open-Meteo |
| `cloud_cover_frac` | REAL | Open-Meteo |
| `temp_c` | REAL | Open-Meteo |

### `ingestion_runs`

Audit log of CAMS and Open-Meteo fetch operations.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `source` | TEXT | `cams` or `openmeteo` |
| `location_id` | INTEGER | |
| `started_at` | TEXT | UTC |
| `finished_at` | TEXT | UTC |
| `rows_written` | INTEGER | |
| `status` | TEXT | `ok` / `error` |
| `error_msg` | TEXT | NULL on success |

### `forecast_runs`

Audit log of forecast pipeline executions.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `location_id` | INTEGER | |
| `run_at` | TEXT | UTC |
| `horizon_days` | INTEGER | |
| `atm_source` | TEXT | `cams` or `climatology` |
| `ai_used` | INTEGER | 0 or 1 |
| `confidence_pct` | INTEGER | 0–100 |
| `runtime_ms` | INTEGER | Wall time |
