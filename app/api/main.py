"""
Solar Forecast Pro — FastAPI backend.

Start:
    uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

Docs:
    http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, locations, forecast
from app.db.sqlite_manager import create_tables, seed_demo_location

app = FastAPI(
    title="Solar Forecast Pro",
    description="Physics-based + AI hybrid PV production forecasting API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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


@app.on_event("startup")
def startup():
    create_tables()
    seed_demo_location()
