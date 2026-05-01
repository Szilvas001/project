# FAQ — Frequently Asked Questions

## General

**Q: Does this require a CAMS API key to work?**  
No. Demo mode uses Open-Meteo (free, no key) and physics-only forecasts. CAMS is optional for training the AI model.

**Q: Does this require PostgreSQL?**  
No. SQLite is the default database. PostgreSQL is only needed if you download CAMS historical data for model training.

**Q: How accurate are the forecasts?**  
Physics-only mode (default): typical RMSE ~0.10–0.15 on clearness index Kt. With trained XGBoost model on CAMS data: ~0.07–0.10 RMSE, R² ~0.88–0.94.

**Q: What weather data source is used?**  
Open-Meteo (https://open-meteo.com) — free, no API key, global coverage, hourly resolution.

**Q: How far ahead can it forecast?**  
Up to 14 days (limited by Open-Meteo's forecast horizon).

---

## Installation

**Q: The install script fails with "Python not found".**  
Make sure `python3` is on your PATH. On some systems use `PYTHON=python3.11 ./install.sh`.

**Q: Streamlit throws a port-in-use error.**  
Use `./run.sh --port=8502` to use a different port.

**Q: Docker build fails on netCDF4.**  
Increase Docker memory to at least 2 GB in Docker Desktop → Settings → Resources.

**Q: Permission denied on install.sh / run.sh.**  
Run `chmod +x install.sh run.sh`.

---

## Dashboard

**Q: The forecast shows zero for all hours.**  
Check that your location coordinates are correct and that Open-Meteo is reachable (internet required).

**Q: The peak time shows UTC but I want local time.**  
Select your timezone in the **⚙️ Settings** tab or the sidebar timezone selector.

**Q: How do I add multiple locations?**  
Go to the **📍 Locations** tab and use the "Add new location" form. All saved locations appear in the sidebar.

**Q: Can I upload a custom spectral response curve?**  
Yes — select "Expert" mode in the sidebar and use the SR CSV upload. CSV format: two columns, `wavelength_nm` and `sr_value` (normalised 0–1).

---

## API

**Q: How do I start the REST API?**  
```bash
./run.sh --api
```
Then open http://localhost:8000/docs for the Swagger UI.

**Q: The /forecast endpoint is slow the first time.**  
The first call fetches live weather from Open-Meteo and computes the clear-sky model. Subsequent calls use the 30-minute cache and are <200ms.

**Q: Can I use this with my own front-end?**  
Yes. The FastAPI backend has CORS enabled (`allow_origins=["*"]` by default). See [api.md](api.md) for all endpoints.

---

## Model Training (Advanced)

**Q: Do I need to train the model to get forecasts?**  
No. Physics-only mode works without any trained model.

**Q: How long does CAMS download take?**  
Typically 30 minutes to 2 hours for 2 years of data at one location, depending on the Copernicus queue.

**Q: The training script fails with "Insufficient training samples".**  
You need at least 90 days of CAMS data. Extend `--start`/`--end` range.

**Q: How often should I retrain?**  
Retraining monthly or quarterly is sufficient. The model is stable across seasons once trained on ≥1 year.

---

## Accuracy & limitations

**Q: What is the expected forecast error?**  
Physics-only Kt RMSE is typically 0.10 – 0.15 over a 24-hour horizon at
mid-latitudes; with a CAMS-trained XGBoost correction, RMSE drops to
~0.07 – 0.10. Expressed as kWh, the day-ahead error band on a 5 kW
residential array is roughly ±10 % in clear-to-partly-cloudy conditions
and widens to ±20 – 30 % during convective storms or rapidly evolving
overcast.

**Q: How is the confidence indicator computed?**  
It is a rule-based score (0 – 100) over five inputs: live-weather
availability, cloud-cover data, AI Kt model, custom SR curve, and CAMS
training. The `confidence_reasons` array on every API response lists
exactly which inputs contributed. It is **not** a calibrated probability.

**Q: Is the 15-minute mode physically resolved at 15-min steps?**  
No. Open-Meteo source data is hourly. The 15-minute mode upsamples via
time-aware interpolation. Use it for charting and energy management
display, **not** for grid-frequency-balancing decisions.

**Q: Are forecasts beyond 7 days reliable?**  
Numerical-weather-prediction skill drops sharply past D+7. We expose up
to 14 days because Open-Meteo serves it, but the confidence_pct should
be read accordingly.

---

## Disclaimer

> **This software is provided for forecasting and modelling purposes
> only. The output is an estimate of solar electricity production and
> must not be used as the sole basis for financial, legal, contractual,
> grid-balancing, safety, or insurance decisions. Actual energy output
> depends on factors outside the model — module degradation, soiling
> events, partial shading, inverter clipping, grid curtailment,
> weather-forecast error, and instrumentation accuracy. The authors
> make no warranty of fitness for any particular purpose. See
> LICENSE.txt for the full legal terms.**
