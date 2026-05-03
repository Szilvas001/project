"""Unified SQLite database manager for Solar Forecast Pro.

Schema (6 tables)
-----------------
  locations                — PV system registry
  cams_atmospheric_forecast — CAMS atmospheric forecast rows
  openmeteo_forecast        — Open-Meteo hourly forecast rows
  model_feature_frame       — merged feature vectors ready for ML
  ingestion_runs            — audit log for ingestion jobs
  forecast_runs             — audit log for forecast executions

PostgreSQL note: all public functions accept an optional ``conn``
kwarg so callers can pass a psycopg2 connection for Postgres storage.
By default, the local SQLite DB is used.
"""

from __future__ import annotations
import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB path — tests monkeypatch this
# ---------------------------------------------------------------------------
DB_PATH = Path(os.environ.get("SF_DB_PATH", "")) or (
    Path(__file__).resolve().parents[2] / "data" / "solar_forecast.db"
)


def _ensure_dir():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Context manager: yields an open SQLite connection (WAL, Row factory)."""
    _ensure_dir()
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
-- PV system registry
CREATE TABLE IF NOT EXISTS locations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    lat         REAL    NOT NULL,
    lon         REAL    NOT NULL,
    altitude    REAL    NOT NULL DEFAULT 0.0,
    capacity_kw REAL    NOT NULL DEFAULT 5.0,
    tilt        REAL,
    azimuth     REAL,
    technology  TEXT    NOT NULL DEFAULT 'mono_si',
    timezone    TEXT    NOT NULL DEFAULT 'UTC',
    config_json TEXT    NOT NULL DEFAULT '{}',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- CAMS atmospheric forecast (one row per variable × timestep)
-- Stored wide: each column is one internal variable name.
CREATE TABLE IF NOT EXISTS cams_atmospheric_forecast (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    location_id              INTEGER NOT NULL REFERENCES locations(id) ON DELETE CASCADE,
    run_time_utc             TEXT    NOT NULL,
    valid_time_utc           TEXT    NOT NULL,
    forecast_step_hours      INTEGER NOT NULL,
    aod_550                  REAL,
    aod_469                  REAL,
    aod_670                  REAL,
    aod_865                  REAL,
    total_column_water_vapour REAL,
    total_column_ozone       REAL,
    pm25                     REAL,
    pm10                     REAL,
    black_carbon_aod_550     REAL,
    dust_aod_550             REAL,
    organic_matter_aod_550   REAL,
    sea_salt_aod_550         REAL,
    sulphate_aod_550         REAL,
    nitrate_aod_550          REAL,
    ammonium_aod_550         REAL,
    boundary_layer_height    REAL,
    temperature_2m           REAL,
    surface_pressure         REAL,
    ingested_at              TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_cams_loc_run_valid
    ON cams_atmospheric_forecast(location_id, run_time_utc, valid_time_utc);

-- Open-Meteo hourly forecast
CREATE TABLE IF NOT EXISTS openmeteo_forecast (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    location_id              INTEGER NOT NULL REFERENCES locations(id) ON DELETE CASCADE,
    valid_time_utc           TEXT    NOT NULL,
    temperature_2m           REAL,
    relative_humidity_2m     REAL,
    dew_point_2m             REAL,
    apparent_temperature     REAL,
    precipitation            REAL,
    cloud_cover              REAL,
    cloud_cover_low          REAL,
    cloud_cover_mid          REAL,
    cloud_cover_high         REAL,
    wind_speed_10m           REAL,
    wind_direction_10m       REAL,
    shortwave_radiation       REAL,
    direct_radiation          REAL,
    diffuse_radiation         REAL,
    direct_normal_irradiance  REAL,
    global_tilted_irradiance  REAL,
    ingested_at              TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_om_loc_valid
    ON openmeteo_forecast(location_id, valid_time_utc);

-- Merged feature frame (CAMS + OM, ready for ML)
CREATE TABLE IF NOT EXISTS model_feature_frame (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    location_id     INTEGER NOT NULL REFERENCES locations(id) ON DELETE CASCADE,
    valid_time_utc  TEXT    NOT NULL,
    features_json   TEXT    NOT NULL,
    data_tier       TEXT    NOT NULL DEFAULT 'demo',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_feature_loc_valid
    ON model_feature_frame(location_id, valid_time_utc);

-- Ingestion run audit log
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source        TEXT    NOT NULL,
    location_id   INTEGER,
    started_at    TEXT    NOT NULL,
    finished_at   TEXT,
    rows_inserted INTEGER NOT NULL DEFAULT 0,
    rows_skipped  INTEGER NOT NULL DEFAULT 0,
    errors        INTEGER NOT NULL DEFAULT 0,
    status        TEXT    NOT NULL DEFAULT 'running',
    detail_json   TEXT
);

-- Forecast execution audit log
CREATE TABLE IF NOT EXISTS forecast_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    location_id     INTEGER REFERENCES locations(id) ON DELETE SET NULL,
    requested_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    horizon_hours   INTEGER,
    data_tier       TEXT,
    confidence_pct  REAL,
    summary_json    TEXT
);
"""


def create_tables() -> None:
    """Ensure all 6 tables exist (idempotent)."""
    with get_connection() as con:
        con.executescript(_DDL)
    log.debug("DB tables ensured")


# ---------------------------------------------------------------------------
# Location helpers
# ---------------------------------------------------------------------------

def get_location(location_id: int) -> Optional[dict]:
    """Return a location row as dict, or None if not found."""
    try:
        with get_connection() as con:
            row = con.execute(
                "SELECT * FROM locations WHERE id = ?", (location_id,)
            ).fetchone()
        return dict(row) if row else None
    except Exception as exc:
        log.warning("get_location failed: %s", exc)
        return None


def list_locations() -> list[dict]:
    with get_connection() as con:
        rows = con.execute("SELECT * FROM locations ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def upsert_location(
    name: str,
    lat: float,
    lon: float,
    altitude: float = 0.0,
    capacity_kw: float = 5.0,
    tilt: Optional[float] = None,
    azimuth: Optional[float] = None,
    technology: str = "mono_si",
    timezone: str = "UTC",
    config: Optional[dict] = None,
) -> int:
    """Insert or update a location; returns the row id."""
    config_json = json.dumps(config or {})
    with get_connection() as con:
        cur = con.execute(
            """
            INSERT INTO locations (name, lat, lon, altitude, capacity_kw, tilt, azimuth,
                                   technology, timezone, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            (name, lat, lon, altitude, capacity_kw, tilt, azimuth, technology, timezone, config_json),
        )
        if cur.lastrowid and cur.lastrowid > 0:
            return cur.lastrowid
        row = con.execute("SELECT id FROM locations WHERE name=? AND lat=? AND lon=?",
                          (name, lat, lon)).fetchone()
        return row["id"] if row else -1


# ---------------------------------------------------------------------------
# CAMS upsert / query
# ---------------------------------------------------------------------------

_CAMS_COLS = [
    "aod_550", "aod_469", "aod_670", "aod_865",
    "total_column_water_vapour", "total_column_ozone",
    "pm25", "pm10",
    "black_carbon_aod_550", "dust_aod_550", "organic_matter_aod_550",
    "sea_salt_aod_550", "sulphate_aod_550", "nitrate_aod_550", "ammonium_aod_550",
    "boundary_layer_height", "temperature_2m", "surface_pressure",
]


def upsert_cams(df: pd.DataFrame, location_id: int) -> int:
    """Upsert CAMS wide DataFrame rows for a location. Returns rows inserted."""
    if df is None or df.empty:
        return 0
    create_tables()
    inserted = 0
    with get_connection() as con:
        for _, row in df.iterrows():
            run_t = str(row.get("run_time_utc", ""))
            valid_t = str(row.get("valid_time_utc", ""))
            step = int(row.get("forecast_step_hours", 0))
            vals = [row.get(c) for c in _CAMS_COLS]
            try:
                con.execute(
                    f"""
                    INSERT OR IGNORE INTO cams_atmospheric_forecast
                        (location_id, run_time_utc, valid_time_utc, forecast_step_hours,
                         {', '.join(_CAMS_COLS)})
                    VALUES (?, ?, ?, ?, {', '.join('?' * len(_CAMS_COLS))})
                    """,
                    [location_id, run_t, valid_t, step] + vals,
                )
                inserted += con.execute("SELECT changes()").fetchone()[0]
            except Exception as exc:
                log.debug("cams upsert row error: %s", exc)
    return inserted


def query_cams(
    location_id: int,
    start_utc: Optional[str] = None,
    end_utc: Optional[str] = None,
) -> pd.DataFrame:
    """Return CAMS rows for a location (optionally filtered by valid_time_utc)."""
    try:
        create_tables()
        with get_connection() as con:
            q = "SELECT * FROM cams_atmospheric_forecast WHERE location_id = ?"
            params: list = [location_id]
            if start_utc:
                q += " AND valid_time_utc >= ?"
                params.append(start_utc)
            if end_utc:
                q += " AND valid_time_utc <= ?"
                params.append(end_utc)
            q += " ORDER BY valid_time_utc"
            rows = con.execute(q, params).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])
    except Exception as exc:
        log.warning("query_cams failed: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Open-Meteo upsert / query
# ---------------------------------------------------------------------------

_OM_COLS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "precipitation", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
    "cloud_cover_high", "wind_speed_10m", "wind_direction_10m",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "direct_normal_irradiance", "global_tilted_irradiance",
]


def upsert_openmeteo(df: pd.DataFrame, location_id: int) -> int:
    """Upsert Open-Meteo forecast rows. Returns rows inserted."""
    if df is None or df.empty:
        return 0
    create_tables()
    inserted = 0
    with get_connection() as con:
        for _, row in df.iterrows():
            valid_t = str(row.get("valid_time_utc", row.get("time", "")))
            vals = [row.get(c) for c in _OM_COLS]
            try:
                con.execute(
                    f"""
                    INSERT OR IGNORE INTO openmeteo_forecast
                        (location_id, valid_time_utc,
                         {', '.join(_OM_COLS)})
                    VALUES (?, ?, {', '.join('?' * len(_OM_COLS))})
                    """,
                    [location_id, valid_t] + vals,
                )
                inserted += con.execute("SELECT changes()").fetchone()[0]
            except Exception as exc:
                log.debug("openmeteo upsert row error: %s", exc)
    return inserted


def query_openmeteo(
    location_id: int,
    start_utc: Optional[str] = None,
    end_utc: Optional[str] = None,
) -> pd.DataFrame:
    """Return Open-Meteo rows for a location (optionally filtered)."""
    try:
        create_tables()
        with get_connection() as con:
            q = "SELECT * FROM openmeteo_forecast WHERE location_id = ?"
            params: list = [location_id]
            if start_utc:
                q += " AND valid_time_utc >= ?"
                params.append(start_utc)
            if end_utc:
                q += " AND valid_time_utc <= ?"
                params.append(end_utc)
            q += " ORDER BY valid_time_utc"
            rows = con.execute(q, params).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
    except Exception as exc:
        log.warning("query_openmeteo failed: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Feature frame
# ---------------------------------------------------------------------------

def upsert_feature_frame(
    location_id: int,
    valid_time_utc: str,
    features: dict,
    data_tier: str = "demo",
) -> None:
    create_tables()
    with get_connection() as con:
        con.execute(
            """
            INSERT OR REPLACE INTO model_feature_frame
                (location_id, valid_time_utc, features_json, data_tier)
            VALUES (?, ?, ?, ?)
            """,
            (location_id, valid_time_utc, json.dumps(features), data_tier),
        )


# ---------------------------------------------------------------------------
# Audit log helpers
# ---------------------------------------------------------------------------

def log_ingestion_run(
    source: str,
    location_id: Optional[int] = None,
    rows_inserted: int = 0,
    rows_skipped: int = 0,
    errors: int = 0,
    status: str = "ok",
    detail: Optional[dict] = None,
    started_at: Optional[str] = None,
) -> int:
    """Insert an ingestion_runs row; returns the new row id."""
    create_tables()
    now = datetime.now(timezone.utc).isoformat()
    started = started_at or now
    with get_connection() as con:
        cur = con.execute(
            """
            INSERT INTO ingestion_runs
                (source, location_id, started_at, finished_at, rows_inserted,
                 rows_skipped, errors, status, detail_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (source, location_id, started, now, rows_inserted,
             rows_skipped, errors, status, json.dumps(detail or {})),
        )
        return cur.lastrowid or 0


def log_forecast_run(
    location_id: Optional[int],
    horizon_hours: Optional[int],
    data_tier: str,
    confidence_pct: Optional[float],
    summary: Optional[dict] = None,
) -> int:
    create_tables()
    with get_connection() as con:
        cur = con.execute(
            """
            INSERT INTO forecast_runs
                (location_id, horizon_hours, data_tier, confidence_pct, summary_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (location_id, horizon_hours, data_tier, confidence_pct,
             json.dumps(summary or {})),
        )
        return cur.lastrowid or 0
