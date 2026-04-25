# ☀️ AI Solar Production Forecast SaaS

**Predict your solar energy output in seconds — no physics knowledge required.**

> Physics-based + AI hybrid PV production forecasting system.  
> Works out of the box. No API keys. No configuration needed.

---

## 🚀 One-command startup

```bash
docker compose up
```

Open **http://localhost:8501** → enter your city + system size → instant forecast.

---

## ✨ What you get

| Feature | Basic | Pro | Expert |
|---|:---:|:---:|:---:|
| City / GPS location | ✓ | ✓ | ✓ |
| Today & tomorrow kWh | ✓ | ✓ | ✓ |
| Hourly production curve | ✓ | ✓ | ✓ |
| Peak production time | ✓ | ✓ | ✓ |
| Cloud loss % | ✓ | ✓ | ✓ |
| 7-day forecast | | ✓ | ✓ |
| Custom tilt / azimuth | | ✓ | ✓ |
| Panel technology selector | | ✓ | ✓ |
| Multi-location support | | ✓ | ✓ |
| CSV export | | ✓ | ✓ |
| REST API | | ✓ | ✓ |
| Custom spectral response | | | ✓ |
| IAM model selection | | | ✓ |
| XGBoost AI model | | | ✓ |

---

## 📦 What's inside

```
solar-forecast-pro/
├── solar_forecast/          # Core physics + AI engine
│   ├── physics/             # Ångström, Hänel, SSA, aerosol optics
│   ├── clearsky/            # pvlib spectrl2 clear-sky irradiance
│   ├── allsky/              # Kt physics model + XGBoost trainer
│   ├── production/          # SR curves, IAM, PV power conversion
│   ├── data_ingestion/      # CAMS loader, Open-Meteo client, PostgreSQL
│   ├── demo/                # Demo pipeline (no keys needed)
│   └── dashboard/           # Streamlit SaaS UI
├── app/
│   ├── api/                 # FastAPI REST backend
│   └── db/                  # SQLite location manager
├── tests/                   # pytest suite
├── docs/                    # Full documentation
├── demo-data/               # Sample CSV for offline demo
├── scripts/                 # CLI: download CAMS, train model
├── Dockerfile
├── docker-compose.yml
├── install.sh               # One-command install (Linux/macOS)
├── run.sh                   # Launch dashboard or API
└── config.yaml
```

---

## 🛠 Installation

### Docker (recommended)
```bash
git clone <repo-url> solar-forecast-pro
cd solar-forecast-pro
cp .env.example .env
docker compose up -d
```

### Manual
```bash
./install.sh
./run.sh
```

See [docs/installation.md](docs/installation.md) for full details.

---

## 🌐 REST API

```bash
./run.sh --api
# → http://localhost:8000/docs
```

```bash
# One-off forecast
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"lat": 47.5, "lon": 19.0, "capacity_kw": 5.0, "horizon_days": 7}'
```

---

## 🧪 Tests

```bash
pytest
```

---

## 📖 Documentation

| Doc | Link |
|---|---|
| Installation | [docs/installation.md](docs/installation.md) |
| Quickstart | [docs/quickstart.md](docs/quickstart.md) |
| Configuration | [docs/configuration.md](docs/configuration.md) |
| API Reference | [docs/api.md](docs/api.md) |
| Dashboard Guide | [docs/dashboard.md](docs/dashboard.md) |
| Training (Advanced) | [docs/training.md](docs/training.md) |
| FAQ | [docs/faq.md](docs/faq.md) |

---

## 🔬 Physics engine

- **pvlib spectrl2** (Bird & Riordan 1986) clear-sky spectral irradiance
- **Ångström turbidity** formula for spectral AOD
- **Hänel hygroscopic** growth factor for aerosol correction
- **SSA / asymmetry parameter** mixing from CAMS species AOD
- **Perez transposition** model (GHI+DNI+DHI → POA)
- **NOCT cell temperature** model
- **ASHRAE / Martin-Ruiz / Fresnel IAM** models
- **XGBoost Kt regressor** with 21 atmospheric features (optional)

---

## 💰 Pricing

| License | Price | Use |
|---|---|---|
| Regular | $79 | Single site, personal/client project |
| Extended | $349 | SaaS / multi-tenant deployment |

---

## 📄 License

CodeCanyon Regular / Extended License. See [LICENSE.txt](LICENSE.txt).

Third-party: pvlib (BSD), XGBoost (Apache 2.0), FastAPI (MIT), Streamlit (Apache 2.0).
