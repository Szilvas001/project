# Quickstart — your first forecast in 60 seconds

After installing (see [installation.md](installation.md)), you're ready to go.
**No CAMS API key, no PostgreSQL, no AI model required.**

## 1. Open the dashboard

```bash
./run.sh
```

Browse to **http://localhost:8501**.

## 2. Pick the demo location

The app ships with a Budapest demo location pre-loaded. You'll see:

- **Today's expected production** in kWh
- **Tomorrow's forecast**
- **7-day energy total**
- **Peak power and peak hour**
- **Capacity factor** and **cloud loss %**

## 3. Add your own location

Click the **📍 Locations** tab → **➕ Add new location** form.

Required fields:
- **Name** — anything you like (e.g. "Office Roof")
- **Latitude / Longitude** — decimal degrees
- **System size** — installed kW (DC)

Optional but recommended:
- **Tilt** — panel angle in degrees (0=flat, 90=vertical)
- **Azimuth** — 180°=South, 90°=East, 270°=West
- **Technology** — mono-Si is the default
- **Timezone** — IANA name (e.g. `Europe/Budapest`)

Click **✓ Save location**, then return to the **📊 Dashboard** tab and
select your new site from the sidebar.

## 4. Export the forecast

Open the **📁 Reports** tab and click:
- **⬇ Download hourly CSV** — every hour, every variable
- **⬇ Download daily summary CSV** — daily kWh + peak

## 5. Use the REST API

Start the API:
```bash
./run.sh --api
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

```bash
# List all locations
curl http://localhost:8000/locations

# Run a one-off forecast
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{
       "lat": 47.498,
       "lon": 19.040,
       "capacity_kw": 10.0,
       "tilt": 35.0,
       "azimuth": 180.0,
       "horizon_days": 7
     }'
```

## What's next?

- For maximum accuracy, **enable CAMS** and **train the AI model** —
  see [training.md](training.md).
- Read [configuration.md](configuration.md) for tuning options.
- Read [api.md](api.md) for full REST endpoint reference.
