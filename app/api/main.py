"""
Solar Forecast Pro — FastAPI backend v2.1.0

Start:
    uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

Docs:
    http://localhost:8000/docs
    http://localhost:8000/redoc
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, locations, forecast, ingestion, features
from app.api.routes import model as model_routes
from app.db.sqlite_manager import create_tables, seed_demo_location

app = FastAPI(
    title="Solar Forecast Pro",
    description=(
        "Physics-accurate + AI hybrid PV production forecasting platform. "
        "SPECTRL2 clear-sky · Perez transposition · CAMS atmospheric data · "
        "XGBoost Kt correction · SR/IAM/denorm spectral integration."
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={"name": "Solar Forecast Pro", "url": "https://github.com/Szilvas001/project"},
    license_info={"name": "Proprietary"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(locations.router)
app.include_router(forecast.router)
app.include_router(ingestion.router)
app.include_router(features.router)
app.include_router(model_routes.router)


@app.on_event("startup")
def startup():
    create_tables()
    seed_demo_location()
    # Ensure new SF DB tables are created too
    try:
        from solar_forecast.db.manager import create_tables as sf_create
        sf_create()
    except Exception:
        pass
