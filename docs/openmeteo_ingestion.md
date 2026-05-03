# Open-Meteo Live Weather Ingestion

Open-Meteo provides live and historical weather data for free with no API key. It is the primary weather source for all forecasts. All timestamps are stored and returned in UTC.

---

## Live Fetch CLI

```bash
# Fetch 72-hour forecast for location ID 1
python -m solar_forecast.ingestion.openmeteo_live \
  --location-id 1 \
  --hours 72

# Fetch 24-hour forecast
python -m solar_forecast.ingestion.openmeteo_live \
  --location-id 1 \
  --hours 24

# Fetch historical backfill (last 30 days)
python -m solar_forecast.ingestion.openmeteo_live \
  --location-id 1 \
  --mode historical \
  --days 30
```

Fetched data is written to the `openmeteo_forecast` table (upserted on `location_id + timestamp_utc`). Duplicate rows are silently skipped.

The module is also callable directly in Python:

```python
from solar_forecast.data_ingestion.openmeteo_live import fetch_openmeteo

df = fetch_openmeteo(lat=47.498, lon=19.040, hours=72)
# Returns a UTC-indexed DataFrame with all variables below
```

---

## Variables Collected

### Forecast endpoint (0–16 days ahead)

| Variable | Column name | Unit | Used for |
|---|---|---|---|
| `shortwave_radiation` | `ghi_wm2` | W/m² | Primary GHI input |
| `direct_normal_irradiance` | `dni_wm2` | W/m² | Perez transposition |
| `diffuse_radiation` | `dhi_wm2` | W/m² | Perez transposition |
| `direct_radiation` | `beam_horiz_wm2` | W/m² | Auxiliary |
| `terrestrial_radiation` | `thermal_wm2` | W/m² | Diagnostic |
| `cloud_cover` | `cloud_cover_frac` | % → 0–1 | Kt cloud model |
| `cloud_cover_low` | `cloud_cover_low` | % → 0–1 | Kt diagnostics |
| `cloud_cover_mid` | `cloud_cover_mid` | % → 0–1 | Kt diagnostics |
| `cloud_cover_high` | `cloud_cover_high` | % → 0–1 | Kt diagnostics |
| `temperature_2m` | `temp_c` | °C | NOCT cell temperature |
| `relative_humidity_2m` | `rh_pct` | % | Hygroscopic AOD correction |
| `surface_pressure` | `pressure_hpa` | hPa | SPECTRL2 fallback |
| `precipitation` | `precip_mm` | mm | Confidence model |
| `precipitation_probability` | `precip_prob` | % | Confidence model |
| `wind_speed_10m` | `wind_speed_ms` | m/s | Soiling / cooling |
| `wind_direction_10m` | `wind_dir_deg` | ° | Diagnostic |
| `visibility` | `visibility_m` | m | Fog / dust flag |
| `cape` | `cape_jkg` | J/kg | Convective instability flag |
| `vapour_pressure_deficit` | `vpd_hpa` | hPa | Auxiliary |

### Historical endpoint (archive)

Subset of the above — irradiance, cloud cover, temperature, humidity, precipitation. Used by `HistoricalGHITrainer` to train the lightweight GHI correction model.

---

## AOD Note

Open-Meteo does not provide aerosol optical depth. When CAMS AOD is unavailable, the module computes a monthly climatological AOD profile (from MERRA-2/CAMS statistics) and scales it by relative humidity using the Hänel hygroscopic factor:

```
AOD_eff = AOD_ref × (1 + γ × (RH − RH_ref) / (1 − RH))
```

where γ = 0.40, RH_ref = 0.50. This is implemented in `solar_forecast/physics/aerosol.py`.

---

## Deduplication

The `openmeteo_forecast` table has a unique index on `(location_id, timestamp_utc)`. All writes use `INSERT OR REPLACE` (SQLite upsert), so re-fetching the same window is safe and idempotent.

The response cache (`requests-cache`, `.cache/openmeteo/`) has a 30-minute TTL. Calls within the same TTL window return the cached response without hitting the Open-Meteo API.

---

## Timestamp Handling

Open-Meteo always returns timestamps in the timezone specified in the request. This module requests `timezone=UTC` — all returned and stored timestamps are UTC-aware ISO 8601 strings. Never convert away from UTC within this module; display conversion is done at the dashboard layer only.

---

## API Endpoints Used

| Purpose | Endpoint |
|---|---|
| Forecast (0–16 d) | `https://api.open-meteo.com/v1/forecast` |
| Historical archive | `https://archive-api.open-meteo.com/v1/archive` |

Both endpoints are free, require no registration, and support global lat/lon queries.
