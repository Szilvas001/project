# REST API Reference

Base URL: `http://localhost:8000`

Interactive Swagger UI: `http://localhost:8000/docs`
ReDoc: `http://localhost:8000/redoc`

## Authentication

The API has no authentication by default — it's intended for internal
deployments. Add your own auth middleware (FastAPI's `Depends`) if you
expose it publicly.

---

## Endpoints

### `GET /health`

Health check.

```json
{
  "status": "ok",
  "version": "2.0.0",
  "timestamp": "2026-04-24T10:30:00+00:00"
}
```

---

### `GET /locations`

List all saved locations.

**Response 200** → `LocationOut[]`

---

### `POST /locations`

Create a new location.

**Body** → `LocationCreate`

```json
{
  "name": "Office Roof Array",
  "lat": 47.498,
  "lon": 19.040,
  "altitude": 120.0,
  "capacity_kw": 10.0,
  "tilt": 35.0,
  "azimuth": 180.0,
  "technology": "mono_si",
  "timezone": "Europe/Budapest"
}
```

**Response 201** → `LocationOut`

| Field | Type | Required | Constraints |
|---|---|---|---|
| name | string | yes | 1–120 chars |
| lat | float | yes | -90 to 90 |
| lon | float | yes | -180 to 180 |
| capacity_kw | float | no (default 5.0) | > 0 |
| tilt | float | no | 0–90 |
| azimuth | float | no | 0–360 |
| technology | enum | no | `mono_si` \| `poly_si` \| `cdte` \| `cigs` \| `hit` |

---

### `GET /locations/{id}`

Get a single location by ID.

**Response 200** → `LocationOut`
**Response 404** → if not found

---

### `PATCH /locations/{id}`

Partial update.

**Body** → `LocationUpdate` (any subset of `LocationCreate` fields)

---

### `DELETE /locations/{id}`

Delete a location and all its cached forecasts.

**Response 204**

---

### `POST /forecast`

Run a one-off forecast for arbitrary coordinates (not saved).

**Body** → `ForecastRequest`

```json
{
  "lat": 47.498,
  "lon": 19.040,
  "altitude": 120.0,
  "capacity_kw": 10.0,
  "tilt": 35.0,
  "azimuth": 180.0,
  "technology": "mono_si",
  "timezone": "Europe/Budapest",
  "horizon_days": 7,
  "use_ai": false
}
```

**Response 200** → `ForecastOut`

```json
{
  "location_id": null,
  "summary": {
    "today_kwh": 32.4,
    "tomorrow_kwh": 28.1,
    "total_7d_kwh": 198.7,
    "peak_power_kw": 9.2,
    "peak_hour_utc": "2026-04-24 11:00:00+00:00",
    "capacity_factor_pct": 11.8,
    "cloud_loss_pct": 14.2
  },
  "hourly": [
    {
      "timestamp_utc": "2026-04-24 00:00:00+00:00",
      "ghi_wm2": 0.0,
      "power_kw": 0.0,
      "energy_kwh": 0.0,
      "kt": null,
      "t_cell_c": 12.5
    }
  ],
  "generated_at": "2026-04-24T10:30:00+00:00"
}
```

---

### `GET /forecast/{location_id}?horizon_days=7`

Forecast for a saved location. Cached per day per location.

**Query params**:
- `horizon_days` (int, 1–14, default 7)

**Response 200** → `ForecastOut`
**Response 404** → if location not found

---

### `GET /export/csv?location_id=1&date=2026-04-24`

Download a previously cached forecast as CSV.

**Query params**:
- `location_id` (int, required)
- `date` (string `YYYY-MM-DD`, optional — defaults to today UTC)

**Response 200** → `text/csv` attachment

---

## Error format

All errors follow the FastAPI default:

```json
{ "detail": "Location not found" }
```

Validation errors use HTTP 422 with field-level detail.
