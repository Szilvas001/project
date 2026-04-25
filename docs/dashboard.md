# Dashboard Guide

The Streamlit dashboard is a buyer-friendly interface designed so anyone
(not just scientists) can run forecasts and export results.

## Launch

```bash
./run.sh                      # localhost:8501
./run.sh --port=8502          # custom port
```

Or via Docker:
```bash
docker compose up dashboard
```

## Sidebar

The sidebar contains the **active location selector** and the
**forecast horizon slider** (1–14 days). Use the **🔄 Refresh data**
button to bypass the 30-minute cache.

## Tabs

### 📊 Dashboard

The home view. Shows:
- **Today / Tomorrow / 7-day** kWh totals
- **Peak power** in kW and **peak hour** (in your local timezone)
- **Capacity factor %** and **cloud loss %**
- **Today's hourly production curve**
- **7-day daily energy bar chart**

### 📍 Locations

CRUD interface for your installed PV systems. Each location stores:

| Field | Description |
|---|---|
| Name | Human-readable identifier |
| Latitude / Longitude | GPS coordinates |
| Altitude | Meters above sea level |
| System size | Installed DC kW |
| Tilt | Panel angle (0=flat, 90=vertical) |
| Azimuth | Compass direction (180=South, 90=East) |
| Technology | Cell type (`mono_si`, `poly_si`, `cdte`, `cigs`, `hit`) |
| Timezone | IANA timezone for display |

### ☀️ Forecast

Detailed forecast view:
- **Hourly power curve** with GHI overlay
- **Full hourly table** (power, energy, GHI, Kt, cell temperature)

### 📁 Reports

Export options:
- **Hourly CSV** — every hour, every variable
- **Daily summary CSV** — daily kWh, peak kW, average GHI

### ⚙️ Settings

Application info, demo mode status, and instructions to enable CAMS for
higher accuracy.

### 🧠 Model Training (Advanced)

Status of the optional XGBoost Kt model. Includes step-by-step CLI
instructions to train it from CAMS history. **Only needed for users who
want maximum accuracy and have a free CAMS API key.**

## Theming

The dashboard uses a custom dark theme defined in `run.sh` and `Dockerfile`:

- Primary color: `#F4A503` (sunny gold)
- Background: `#0E1117`
- Secondary background: `#1E1E2E`

To change the theme, edit the `--theme.*` flags in `run.sh`.

## Caching

Forecasts are cached for 30 minutes per (location, date, horizon).
Click **🔄 Refresh data** to invalidate.

## Performance

A 7-day forecast for a single location typically completes in
**under 5 seconds** on a modest machine, including:
- Open-Meteo API call (~500 ms)
- pvlib spectrl2 clear-sky calculation (~1.5 s)
- Perez transposition + IAM + power conversion (~200 ms)
