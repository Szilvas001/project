# REST API Reference

**Base URL:** `http://localhost:8000`  
**Swagger UI:** `http://localhost:8000/docs`  
**ReDoc:** `http://localhost:8000/redoc`

No authentication by default. Add FastAPI `Depends` middleware to secure public deployments.

---

## `GET /health`

```json
{ "status": "ok", "version": "2.0.0", "timestamp": "2026-05-03T10:00:00+00:00" }
```

---

## Locations

`GET /locations` — list all. `GET /locations/{id}` — get one. Both return `LocationOut`.

### `POST /locations` — create

```bash
curl -X POST http://localhost:8000/locations \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Office Roof", "lat": 47.498, "lon": 19.040,
    "altitude": 120.0, "capacity_kw": 10.0,
    "tilt": 35.0, "azimuth": 180.0,
    "technology": "mono_si", "timezone": "Europe/Budapest"
  }'
```

| Field | Type | Default | Constraints |
|---|---|---|---|
| `name` | string | required | 1–120 chars |
| `lat` / `lon` | float | required | ±90 / ±180 |
| `altitude` | float | 0.0 | 0–8848 m |
| `capacity_kw` | float | 5.0 | > 0 |
| `tilt` | float | auto | 0–90° |
| `azimuth` | float | auto | 0–360° |
| `technology` | string | `mono_si` | `mono_si` \| `poly_si` \| `cdte` \| `cigs` \| `hit` |

**Response 201** — `LocationOut` with `id`, `created_at`, `updated_at`.

`PATCH /locations/{id}` accepts any subset of `LocationCreate` fields. `DELETE /locations/{id}` removes the location and all cached forecasts; returns 204.

---

## `POST /forecast` — one-off forecast

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 47.498, "lon": 19.040, "altitude": 120.0,
    "capacity_kw": 10.0, "tilt": 35.0, "azimuth": 180.0,
    "technology": "mono_si", "horizon_days": 7, "use_ai": false
  }'
```

**Response 200** — `ForecastOut`:

```json
{
  "location_id": null,
  "summary": { "today_kwh": 32.4, "tomorrow_kwh": 28.1, "total_7d_kwh": 198.7,
    "peak_power_kw": 9.2, "peak_hour_utc": "2026-05-03 11:00:00+00:00",
    "capacity_factor_pct": 11.8, "cloud_loss_pct": 14.2 },
  "hourly": [{ "timestamp_utc": "2026-05-03 06:00:00+00:00",
    "ghi_wm2": 312.5, "power_kw": 2.84, "energy_kwh": 2.84, "kt": 0.71, "t_cell_c": 28.3 }],
  "generated_at": "2026-05-03T10:00:00+00:00"
}
```

---

## `GET /forecast/{location_id}?horizon_days=7` — cached location forecast

Result cached in SQLite per day. Subsequent calls within the same UTC day return immediately.

```bash
curl "http://localhost:8000/forecast/1?horizon_days=7"
```

**Response 200** — same `ForecastOut` structure, `location_id` populated. **404** if location missing.

---

## `POST /forecast/realtime` — sub-hourly curve

```bash
curl -X POST http://localhost:8000/forecast/realtime \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 47.498, "lon": 19.040, "capacity_kw": 10.0,
    "tilt": 35.0, "azimuth": 180.0, "technology": "mono_si",
    "iam_model": "ashrae", "resolution_minutes": 15,
    "horizon_hours": 24, "use_ai_ghi": false
  }'
```

| Field | Default | Constraints |
|---|---|---|
| `resolution_minutes` | 15 | 5–60 |
| `horizon_hours` | 24 | 1–72 |
| `iam_model` | `ashrae` | `ashrae` \| `martin_ruiz` \| `fresnel` |
| `use_ai_ghi` | `false` | Requires `models/ghi_historical.joblib` |

**Response 200:**

```json
{
  "now_power_kw": 5.21, "now_utc": "2026-05-03T10:07:00+00:00",
  "curve": [{
    "timestamp_utc": "2026-05-03T10:00:00+00:00",
    "ghi_wm2": 520.0, "ghi_clear_wm2": 732.1, "poa_wm2": 610.4,
    "power_kw": 5.18, "kt": 0.71, "t_cell_c": 31.2, "cloud_cover_frac": 0.22
  }],
  "atmosphere": { "source": "cams", "aod_550nm": 0.14, "ozone_du": 315.0 },
  "generated_at": "2026-05-03T10:07:00+00:00"
}
```

---

## `GET /export/csv?location_id=1&date=2026-05-03`

Downloads a cached forecast as CSV. Call `GET /forecast/{id}` first to populate cache.

```bash
curl "http://localhost:8000/export/csv?location_id=1&date=2026-05-03" -o forecast.csv
```

**Response 200** — `text/csv` attachment. **404** if not cached.

---

## Error Format

```json
{ "detail": "Location not found" }
```

| Code | Meaning |
|---|---|
| 404 | Location or cached forecast not found |
| 422 | Validation error (field-level detail in body) |
| 500 | Physics failure or upstream API unreachable |
