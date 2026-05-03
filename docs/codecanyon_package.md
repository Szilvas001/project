# CodeCanyon Package — What's Included

**Item name:** AI Solar Production Forecast SaaS — Physics + CAMS + Open-Meteo  
**Category:** PHP Scripts → Widgets (listed under Python Applications)  
**License:** Regular / Extended (see below)

---

## Package Contents

```
solar-forecast-pro/
├── app/
│   ├── api/
│   │   ├── main.py                  FastAPI application entry point
│   │   ├── models.py                Pydantic request/response schemas
│   │   └── routes/
│   │       ├── forecast.py          /forecast, /forecast/realtime, /export/csv
│   │       ├── health.py            /health
│   │       └── locations.py         /locations CRUD
│   └── db/
│       └── sqlite_manager.py        SQLite ORM layer (WAL mode)
├── solar_forecast/
│   ├── allsky/
│   │   ├── ai_trainer.py            XGBoost Kt model (21 CAMS features)
│   │   ├── historical_trainer.py    HistoricalGHITrainer (Open-Meteo only)
│   │   ├── hybrid_model.py          AllSkyModel — physics/AI blend orchestrator
│   │   └── physics_kt.py            Delta-Eddington two-stream Kt model
│   ├── cams_fetcher/
│   │   ├── client.py                cdsapi wrapper (lazy import)
│   │   ├── config_default.yaml      Variable list and schedule config
│   │   ├── db.py                    PostgreSQL upsert layer
│   │   ├── grib_processor.py        GRIB decode + bilinear interpolation
│   │   ├── runner.py                run_once() orchestrator
│   │   └── scheduler.py             Daemon thread + cron installer
│   ├── clearsky/
│   │   └── spectrl2_model.py        pvlib Bird & Riordan SPECTRL2 clear-sky
│   ├── dashboard/
│   │   └── app.py                   Streamlit 7-tab dashboard
│   ├── data_ingestion/
│   │   ├── cams_loader.py           CAMS → DataFrame loader
│   │   ├── cams_query.py            AOD → Ångström exponents + SSA/asym
│   │   ├── db_manager.py            PostgreSQL training DB manager
│   │   └── openmeteo_live.py        Open-Meteo live + historical fetcher
│   ├── demo/
│   │   └── pipeline.py              run_demo_forecast(), run_realtime_forecast()
│   ├── physics/
│   │   └── aerosol.py               Hänel hygroscopic AOD correction
│   └── production/
│       ├── iam_model.py             ASHRAE / Martin-Ruiz / Fresnel IAM
│       ├── pv_output.py             DC → AC power conversion
│       └── spectral_response.py     SR curves + MM factor (5 technologies)
├── scripts/
│   ├── 01_download_cams.py          CLI: download CAMS training data
│   ├── 02_train_kt_model.py         CLI: train XGBoost Kt model
│   ├── 03_run_dashboard.py          Convenience launcher
│   └── 04_generate_demo_model.py    Generate demo model for offline use
├── tests/                           Pytest test suite (30+ tests)
├── demo-data/
│   └── demo_forecast_budapest.csv   Offline demo data
├── models/
│   └── README.md                    Model storage instructions
├── docs/                            This documentation
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.py
├── config.yaml
├── .env.example
├── run.sh                           Convenience launcher script
└── install.sh                       One-command install (Linux/macOS)
```

---

## What Is Included

- Full Python source code — no obfuscation, no compiled blobs
- Streamlit dashboard with 7 tabs: Dashboard, Real-Time, Forecast, Locations, Reports, Settings, Model Training
- FastAPI REST backend with Swagger UI
- Complete physics engine: SPECTRL2 → Kt → Perez → IAM → SR → denorm → DC → AC
- Three user tiers: Basic, Pro, Expert
- CAMS atmospheric data ingestion (CLI + scheduler)
- Open-Meteo live weather ingestion
- XGBoost Kt model training pipeline (requires CAMS key)
- HistoricalGHITrainer (works with Open-Meteo data alone)
- SQLite database layer (no PostgreSQL required for normal operation)
- Docker Compose configuration for production deployment
- Full pytest test suite
- This documentation set

## What Is Not Included

- Pre-trained model files (`models/kt_xgb.joblib`) — must be trained using your own CAMS data at your location; training instructions are in `docs/` and `scripts/`
- CAMS API key — free registration at https://ads.atmosphere.copernicus.eu/profile
- Hosting infrastructure — you run this on your own server or cloud VM
- Custom branding / white-label work — available as paid support (contact author)

---

## License

### Regular License

- Use in one end product for your own use or a single client
- The end product may not be sold or redistributed
- Source code modification is permitted for your own installation

### Extended License

- Use in one end product that is sold to end users (SaaS, subscription, etc.)
- Multiple deployments of the same product are permitted under one Extended License
- Source code modification is permitted

Both licenses prohibit redistribution of the source code itself on any marketplace, including CodeCanyon.

Full license text: https://codecanyon.net/licenses/standard

---

## Support

**Included support period:** 6 months from purchase date.

Support covers:
- Installation issues and environment setup
- Bug reports with reproducible steps
- Clarification of documented behaviour

Support does not cover:
- Custom feature development
- Third-party API issues (Copernicus ADS queue, Open-Meteo outages)
- Hosting, server configuration, or cloud infrastructure

**Support channel:** CodeCanyon item comments or the author's profile contact form.

**Response time:** 1–3 business days.

---

## Version History

See `CHANGELOG.md` in the package root for the full version history.

Current version: **2.0.0**  
Minimum Python: **3.10**  
Tested on: Ubuntu 22.04, macOS 13, Docker Engine 24
