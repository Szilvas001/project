# Quickstart

Three user tiers are available: **Basic**, **Pro**, and **Expert**. Each tier unlocks additional physics controls. All tiers work without a CAMS API key.

---

## Basic — location + kW → forecast

The minimum required inputs are coordinates and system capacity.

### Dashboard

1. Open http://localhost:8501
2. In the sidebar select **Basic** tier
3. Enter city coordinates and installed kW
4. Click **Run Forecast**

The Dashboard tab shows today's kWh, tomorrow's kWh, 7-day total, peak power, capacity factor, and cloud loss %.

### API

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 47.498,
    "lon": 19.040,
    "capacity_kw": 5.0,
    "horizon_days": 7
  }'
```

`tilt` defaults to `|lat| × 0.76`. `azimuth` defaults to 180° (south, northern hemisphere).

---

## Pro — tilt / azimuth / panel technology / 15-min resolution

### Dashboard

1. Select **Pro** tier in the sidebar
2. Set **Tilt** (0–90°), **Azimuth** (0–360°), **Technology** (`mono_si` / `poly_si` / `cdte` / `cigs` / `hit`)
3. Enable **15-minute sub-hourly** output via the Real-Time tab

The Real-Time tab auto-refreshes every 60 s, shows a smooth production curve, a NOW marker, Kt sub-chart, and cell temperature.

### API — real-time sub-hourly curve

```bash
curl -X POST http://localhost:8000/forecast/realtime \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 47.498,
    "lon": 19.040,
    "capacity_kw": 10.0,
    "tilt": 35.0,
    "azimuth": 180.0,
    "technology": "mono_si",
    "resolution_minutes": 15,
    "horizon_hours": 24
  }'
```

Response includes `now_power_kw`, `now_utc`, and a `curve` array with `ghi_wm2`, `poa_wm2`, `power_kw`, `kt`, `t_cell_c` per step.

### Saved locations (Pro+)

```bash
# Create a location
curl -X POST http://localhost:8000/locations \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Office Roof",
    "lat": 47.498,
    "lon": 19.040,
    "altitude": 120.0,
    "capacity_kw": 10.0,
    "tilt": 35.0,
    "azimuth": 180.0,
    "technology": "mono_si",
    "timezone": "Europe/Budapest"
  }'

# Get forecast for location ID 1 (result cached in SQLite)
curl "http://localhost:8000/forecast/1?horizon_days=7"

# Download as CSV
curl "http://localhost:8000/export/csv?location_id=1&date=2026-05-03" \
  -o forecast.csv
```

---

## Expert — CAMS / SR / IAM / denorm / AI blend

Expert tier exposes full physics controls.

### Dashboard

1. Select **Expert** tier in the sidebar
2. Configure:
   - **IAM model**: `ashrae` (default) / `martin_ruiz` / `fresnel`
   - **SR upload**: custom spectral response CSV (`wavelength_nm`, `sr_value`)
   - **Denorm factor**: override automatic spectral denormalisation
   - **AI toggle**: enable XGBoost Kt blend (requires trained model at `models/kt_xgb.joblib`)
   - **CAMS diagnostics**: view per-timestep AOD, ozone, water vapour, BLH

### API — expert realtime with AI and custom IAM

```bash
curl -X POST http://localhost:8000/forecast/realtime \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 47.498,
    "lon": 19.040,
    "capacity_kw": 10.0,
    "tilt": 35.0,
    "azimuth": 180.0,
    "technology": "mono_si",
    "iam_model": "martin_ruiz",
    "resolution_minutes": 15,
    "horizon_hours": 48,
    "use_ai_ghi": true,
    "ghi_model_path": "models/ghi_historical.joblib"
  }'
```

When `use_ai_ghi` is true the AI-corrected GHI replaces the Open-Meteo raw value before the physics stack runs. The blend weight is configurable via `physics_weight` in `config.yaml` (default 0.40 physics / 0.60 AI).

---

## Confidence Output

Every forecast includes a confidence block:

```json
{
  "confidence_pct": 82,
  "confidence_label": "High",
  "confidence_reasons": [
    "CAMS atmospheric data available",
    "Cloud cover < 40%",
    "Horizon < 48 h"
  ]
}
```

Confidence degrades when CAMS is unavailable (falls back to climatology), cloud cover is high, or the horizon exceeds 72 h.
