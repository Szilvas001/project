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
