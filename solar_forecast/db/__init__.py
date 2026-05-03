"""Unified database layer for Solar Forecast Pro.

Primary storage: SQLite (WAL mode, zero-config).
PostgreSQL optional: set DATABASE_URL env var.

Public API
----------
    from solar_forecast.db.manager import (
        get_connection, create_tables,
        get_location, list_locations, upsert_location,
        upsert_cams, query_cams,
        upsert_openmeteo, query_openmeteo,
        upsert_feature_frame,
        log_ingestion_run, log_forecast_run,
    )
"""
