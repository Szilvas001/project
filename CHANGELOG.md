# Changelog

All notable changes to **AI Solar Production Forecast SaaS** are documented here.

---

## [2.0.0] — 2026-04-25

### Added
- **Full productization** — buyer-friendly SaaS dashboard with 3-tier UX (Basic / Pro / Expert)
- **Demo mode** — works with zero configuration (no CAMS key, no PostgreSQL, no trained model)
- **FastAPI REST backend** — GET /health, POST /forecast, GET/POST /locations, GET /export/csv
- **SQLite multi-location manager** — save unlimited PV sites, automatic schema migration
- **Docker Compose** — `docker compose up` → working dashboard + API in one command
- **install.sh / run.sh** — one-command install and launch for Linux/macOS
- **3-tier user levels** (Basic → Pro → Expert) — complexity hidden until needed
- **Productized dashboard** — tabs: Dashboard / Locations / Forecast / Reports / Settings / Training
- **pytest test suite** — locations CRUD, API schema, pipeline smoke tests, import tests
- **Full /docs** — installation, quickstart, configuration, API reference, training guide, FAQ

### Changed
- Default database changed from PostgreSQL to **SQLite** (PostgreSQL remains optional)
- Dashboard completely rewritten — modern dark SaaS theme, KPI cards, Plotly charts
- All zip archive files removed from repository

### Physics & AI (carried over from 1.x + enhanced)
- spectrl2 clear-sky with full Ångström / Hänel / SSA / asymmetry parameter support
- XGBoost Kt model with 21 atmospheric features (CAMS-backed, optional)
- Perez transposition + ASHRAE/Martin-Ruiz/Fresnel IAM models
- NOCT cell temperature model
- 5 built-in spectral response curves (mono-Si, poly-Si, CdTe, CIGS, HIT)

---

## [1.0.0] — 2024-12-01

### Added
- Initial release: pvlib spectrl2 clear-sky + CAMS atmospheric physics
- XGBoost Kt model training from CAMS EAC4 + solar radiation service
- Basic Streamlit dashboard (Data / Train / Live Forecast tabs)
- PostgreSQL storage for CAMS data
