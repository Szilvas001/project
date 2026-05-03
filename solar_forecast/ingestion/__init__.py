"""
solar_forecast.ingestion — data collection layer.

Subpackages
-----------
    cams          CAMS atmospheric forecast ingestion (AOD, ozone, BLH, …)
    openmeteo_live  Open-Meteo live weather ingestion

Both write to the shared SQLite database (solar_forecast/db/manager.py).
PostgreSQL is optional (set DATABASE_URL env var).
"""
