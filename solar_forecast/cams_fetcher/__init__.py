"""
CAMS Fetcher — automated atmospheric data collection from ECMWF CAMS.

This subpackage was integrated from the standalone `cams-fetcher` tool. It
downloads CAMS forecast data (AOD, ozone, water vapour, BLH, …) for a target
location and stores it in PostgreSQL so the forecast pipeline can consume it.

Heavy dependencies (cdsapi, pygrib, psycopg2) are imported lazily inside
their respective modules so the rest of the project still works in
test / demo environments where those native libraries are not installed.

Public entry points
-------------------
    from solar_forecast.cams_fetcher.runner import run_once
    from solar_forecast.cams_fetcher.scheduler import CamsScheduler
"""

from __future__ import annotations

__all__ = ["runner", "scheduler", "client", "db", "grib_processor", "backfill"]
