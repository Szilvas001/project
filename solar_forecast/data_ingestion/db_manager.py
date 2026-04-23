"""
PostgreSQL persistence layer.

Tables
------
cams_atmo        — Hourly atmospheric data from CAMS EAC4 reanalysis
cams_radiation   — Hourly all-sky / clear-sky GHI from CAMS radiation service
kt_features      — Pre-computed feature table used for Kt model training
forecasts        — Stored hourly production forecasts
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import (
    Column, DateTime, Float, Integer, String, UniqueConstraint,
    create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS cams_atmo (
    id                  BIGSERIAL PRIMARY KEY,
    timestamp           TIMESTAMPTZ NOT NULL,
    lat                 DOUBLE PRECISION NOT NULL,
    lon                 DOUBLE PRECISION NOT NULL,
    aod_550nm           DOUBLE PRECISION,
    angstrom_exponent   DOUBLE PRECISION,
    total_ozone         DOUBLE PRECISION,   -- Dobson units
    precipitable_water  DOUBLE PRECISION,   -- cm
    surface_pressure    DOUBLE PRECISION,   -- hPa
    cloud_cover         DOUBLE PRECISION,   -- fraction [0-1]
    cloud_optical_depth DOUBLE PRECISION,
    UNIQUE (timestamp, lat, lon)
);
CREATE INDEX IF NOT EXISTS idx_cams_atmo_ts  ON cams_atmo (timestamp);
CREATE INDEX IF NOT EXISTS idx_cams_atmo_loc ON cams_atmo (lat, lon);

CREATE TABLE IF NOT EXISTS cams_radiation (
    id          BIGSERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL,
    lat         DOUBLE PRECISION NOT NULL,
    lon         DOUBLE PRECISION NOT NULL,
    ghi         DOUBLE PRECISION,      -- W/m²
    dhi         DOUBLE PRECISION,
    dni         DOUBLE PRECISION,
    ghi_clear   DOUBLE PRECISION,
    dhi_clear   DOUBLE PRECISION,
    dni_clear   DOUBLE PRECISION,
    UNIQUE (timestamp, lat, lon)
);
CREATE INDEX IF NOT EXISTS idx_cams_rad_ts ON cams_radiation (timestamp);

CREATE TABLE IF NOT EXISTS forecasts (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT now(),
    timestamp   TIMESTAMPTZ NOT NULL,
    lat         DOUBLE PRECISION NOT NULL,
    lon         DOUBLE PRECISION NOT NULL,
    capacity_kw DOUBLE PRECISION,
    power_kw    DOUBLE PRECISION,
    ghi         DOUBLE PRECISION,
    kt          DOUBLE PRECISION,
    UNIQUE (timestamp, lat, lon, capacity_kw)
);
CREATE INDEX IF NOT EXISTS idx_fcst_ts ON forecasts (timestamp);
"""


class DBManager:
    """Thin wrapper around SQLAlchemy engine for the solar forecast schema."""

    def __init__(self, cfg: dict):
        db = cfg["database"]
        url = (
            f"postgresql+psycopg2://{db['user']}:{db['password']}"
            f"@{db['host']}:{db['port']}/{db['name']}"
        )
        self.engine = create_engine(url, echo=False, pool_pre_ping=True)

    def create_tables(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(_DDL))
        logger.info("Database schema ensured.")

    # ------------------------------------------------------------------
    # CAMS atmospheric
    # ------------------------------------------------------------------

    def upsert_cams_atmo(self, df: pd.DataFrame) -> int:
        """Bulk-insert CAMS atmospheric rows; skip existing timestamps."""
        if df.empty:
            return 0
        rows = df.reset_index().rename(columns={"index": "timestamp"}).to_dict("records")
        inserted = 0
        with self.engine.begin() as conn:
            for row in rows:
                r = conn.execute(text("""
                    INSERT INTO cams_atmo
                        (timestamp, lat, lon, aod_550nm, angstrom_exponent,
                         total_ozone, precipitable_water, surface_pressure,
                         cloud_cover, cloud_optical_depth)
                    VALUES
                        (:timestamp, :lat, :lon, :aod_550nm, :angstrom_exponent,
                         :total_ozone, :precipitable_water, :surface_pressure,
                         :cloud_cover, :cloud_optical_depth)
                    ON CONFLICT (timestamp, lat, lon) DO NOTHING
                """), row)
                inserted += r.rowcount
        return inserted

    def load_cams_atmo(
        self, lat: float, lon: float, start: datetime, end: datetime
    ) -> pd.DataFrame:
        q = text("""
            SELECT timestamp, aod_550nm, angstrom_exponent, total_ozone,
                   precipitable_water, surface_pressure, cloud_cover, cloud_optical_depth
            FROM cams_atmo
            WHERE lat = :lat AND lon = :lon
              AND timestamp BETWEEN :start AND :end
            ORDER BY timestamp
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(q, conn, params={"lat": lat, "lon": lon,
                                               "start": start, "end": end})
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        return df

    # ------------------------------------------------------------------
    # CAMS radiation
    # ------------------------------------------------------------------

    def upsert_cams_radiation(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        rows = df.reset_index().rename(columns={"index": "timestamp"}).to_dict("records")
        inserted = 0
        with self.engine.begin() as conn:
            for row in rows:
                r = conn.execute(text("""
                    INSERT INTO cams_radiation
                        (timestamp, lat, lon, ghi, dhi, dni, ghi_clear, dhi_clear, dni_clear)
                    VALUES
                        (:timestamp, :lat, :lon, :ghi, :dhi, :dni, :ghi_clear, :dhi_clear, :dni_clear)
                    ON CONFLICT (timestamp, lat, lon) DO NOTHING
                """), row)
                inserted += r.rowcount
        return inserted

    def load_cams_radiation(
        self, lat: float, lon: float, start: datetime, end: datetime
    ) -> pd.DataFrame:
        q = text("""
            SELECT timestamp, ghi, dhi, dni, ghi_clear, dhi_clear, dni_clear
            FROM cams_radiation
            WHERE lat = :lat AND lon = :lon
              AND timestamp BETWEEN :start AND :end
            ORDER BY timestamp
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(q, conn, params={"lat": lat, "lon": lon,
                                               "start": start, "end": end})
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        return df

    # ------------------------------------------------------------------
    # Forecasts
    # ------------------------------------------------------------------

    def store_forecast(self, df: pd.DataFrame, capacity_kw: float,
                       lat: float, lon: float) -> None:
        if df.empty:
            return
        df = df.copy()
        df["lat"] = lat
        df["lon"] = lon
        df["capacity_kw"] = capacity_kw
        rows = df.reset_index().rename(columns={"index": "timestamp"}).to_dict("records")
        with self.engine.begin() as conn:
            for row in rows:
                conn.execute(text("""
                    INSERT INTO forecasts
                        (timestamp, lat, lon, capacity_kw, power_kw, ghi, kt)
                    VALUES
                        (:timestamp, :lat, :lon, :capacity_kw, :power_kw, :ghi, :kt)
                    ON CONFLICT (timestamp, lat, lon, capacity_kw) DO UPDATE
                        SET power_kw = EXCLUDED.power_kw,
                            ghi      = EXCLUDED.ghi,
                            kt       = EXCLUDED.kt,
                            created_at = now()
                """), row)
