"""
PostgreSQL persistence layer — extended schema.

Tables
------
cams_atmo        — 3-hourly EAC4 atmospheric data (full variable set)
cams_radiation   — Hourly all-sky / clear-sky solar radiation (CAMS)
forecasts        — Stored hourly production forecasts

All timestamps are stored tz-aware UTC (TIMESTAMPTZ).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS cams_atmo (
    id                    BIGSERIAL PRIMARY KEY,
    timestamp             TIMESTAMPTZ NOT NULL,
    lat                   DOUBLE PRECISION NOT NULL,
    lon                   DOUBLE PRECISION NOT NULL,
    -- Multi-wavelength AOD
    aod_550nm             DOUBLE PRECISION,
    aod_469nm             DOUBLE PRECISION,
    aod_670nm             DOUBLE PRECISION,
    aod_865nm             DOUBLE PRECISION,
    aod_1240nm            DOUBLE PRECISION,
    -- Speciated AOD at 550 nm
    aod_dust_550nm        DOUBLE PRECISION,
    aod_bc_550nm          DOUBLE PRECISION,
    aod_om_550nm          DOUBLE PRECISION,
    aod_ss_550nm          DOUBLE PRECISION,
    aod_su_550nm          DOUBLE PRECISION,
    -- Ångström exponents (derived)
    angstrom_alpha1       DOUBLE PRECISION,   -- 340–500 nm
    angstrom_alpha2       DOUBLE PRECISION,   -- 500–1064 nm
    angstrom_exponent     DOUBLE PRECISION,   -- legacy single-α
    -- SSA and asymmetry (derived from speciated AOD)
    ssa_550nm             DOUBLE PRECISION,
    asymmetry_factor      DOUBLE PRECISION,
    -- Atmospheric composition
    total_ozone           DOUBLE PRECISION,   -- Dobson units
    precipitable_water    DOUBLE PRECISION,   -- cm
    surface_pressure      DOUBLE PRECISION,   -- hPa
    cloud_cover           DOUBLE PRECISION,   -- fraction [0–1]
    cloud_optical_depth   DOUBLE PRECISION,
    temperature_2m        DOUBLE PRECISION,   -- °C
    boundary_layer_height DOUBLE PRECISION,   -- m
    forecast_albedo       DOUBLE PRECISION,
    snow_albedo           DOUBLE PRECISION,
    -- PM and gas
    pm25                  DOUBLE PRECISION,   -- μg/m³
    pm10                  DOUBLE PRECISION,   -- μg/m³
    total_column_co       DOUBLE PRECISION,   -- mol/m²
    total_column_no2      DOUBLE PRECISION,   -- mol/m²
    UNIQUE (timestamp, lat, lon)
);
CREATE INDEX IF NOT EXISTS idx_cams_atmo_ts  ON cams_atmo (timestamp);
CREATE INDEX IF NOT EXISTS idx_cams_atmo_loc ON cams_atmo (lat, lon);

CREATE TABLE IF NOT EXISTS cams_radiation (
    id          BIGSERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL,
    lat         DOUBLE PRECISION NOT NULL,
    lon         DOUBLE PRECISION NOT NULL,
    ghi         DOUBLE PRECISION,
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
    power_dc_kw DOUBLE PRECISION,
    ghi         DOUBLE PRECISION,
    kt          DOUBLE PRECISION,
    t_cell      DOUBLE PRECISION,
    g_eff       DOUBLE PRECISION,
    mm          DOUBLE PRECISION,
    UNIQUE (timestamp, lat, lon, capacity_kw)
);
CREATE INDEX IF NOT EXISTS idx_fcst_ts ON forecasts (timestamp);
"""

# Column list for cams_atmo upsert
_ATMO_COLS = [
    "timestamp", "lat", "lon",
    "aod_550nm", "aod_469nm", "aod_670nm", "aod_865nm", "aod_1240nm",
    "aod_dust_550nm", "aod_bc_550nm", "aod_om_550nm", "aod_ss_550nm", "aod_su_550nm",
    "angstrom_alpha1", "angstrom_alpha2", "angstrom_exponent",
    "ssa_550nm", "asymmetry_factor",
    "total_ozone", "precipitable_water", "surface_pressure",
    "cloud_cover", "cloud_optical_depth",
    "temperature_2m", "boundary_layer_height",
    "forecast_albedo", "snow_albedo",
    "pm25", "pm10", "total_column_co", "total_column_no2",
]


class DBManager:
    """Thin SQLAlchemy wrapper for the solar forecast schema."""

    def __init__(self, cfg: dict):
        db  = cfg["database"]
        url = (
            f"postgresql+psycopg2://{db['user']}:{db['password']}"
            f"@{db['host']}:{db['port']}/{db['name']}"
        )
        self.engine = create_engine(url, echo=False, pool_pre_ping=True)

    def create_tables(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(_DDL))
        logger.info("Database schema ensured.")

    # ──────────────────────────────────────────────────────────────────────
    # CAMS atmospheric
    # ──────────────────────────────────────────────────────────────────────

    def upsert_cams_atmo(self, df: pd.DataFrame) -> int:
        """Bulk-insert CAMS atmospheric rows (skip existing timestamps)."""
        if df.empty:
            return 0

        rows = df.reset_index().rename(columns={"index": "timestamp"})
        # Ensure 'timestamp' column exists (may be index)
        if "timestamp" not in rows.columns and rows.index.name == "timestamp":
            rows = rows.reset_index()

        inserted = 0
        with self.engine.begin() as conn:
            for _, row in rows.iterrows():
                params = {col: _safe(row.get(col)) for col in _ATMO_COLS}
                params.setdefault("angstrom_exponent",
                                  params.get("angstrom_alpha2"))

                cols_sql   = ", ".join(_ATMO_COLS)
                values_sql = ", ".join(f":{c}" for c in _ATMO_COLS)
                r = conn.execute(text(f"""
                    INSERT INTO cams_atmo ({cols_sql})
                    VALUES ({values_sql})
                    ON CONFLICT (timestamp, lat, lon) DO NOTHING
                """), params)
                inserted += r.rowcount
        return inserted

    def load_cams_atmo(
        self, lat: float, lon: float, start: datetime, end: datetime
    ) -> pd.DataFrame:
        q = text("""
            SELECT * FROM cams_atmo
            WHERE lat = :lat AND lon = :lon
              AND timestamp BETWEEN :start AND :end
            ORDER BY timestamp
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(q, conn, params={"lat": lat, "lon": lon,
                                               "start": start, "end": end})
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp").drop(columns=["id"], errors="ignore")
        return df

    # ──────────────────────────────────────────────────────────────────────
    # CAMS radiation
    # ──────────────────────────────────────────────────────────────────────

    def upsert_cams_radiation(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        rows = df.reset_index().rename(columns={"index": "timestamp"})
        if "timestamp" not in rows.columns and rows.index.name == "timestamp":
            rows = rows.reset_index()

        inserted = 0
        with self.engine.begin() as conn:
            for _, row in rows.iterrows():
                r = conn.execute(text("""
                    INSERT INTO cams_radiation
                        (timestamp, lat, lon, ghi, dhi, dni,
                         ghi_clear, dhi_clear, dni_clear)
                    VALUES
                        (:timestamp, :lat, :lon, :ghi, :dhi, :dni,
                         :ghi_clear, :dhi_clear, :dni_clear)
                    ON CONFLICT (timestamp, lat, lon) DO NOTHING
                """), {
                    "timestamp": row.get("timestamp"),
                    "lat": row.get("lat"), "lon": row.get("lon"),
                    "ghi": _safe(row.get("ghi")),
                    "dhi": _safe(row.get("dhi")),
                    "dni": _safe(row.get("dni")),
                    "ghi_clear": _safe(row.get("ghi_clear")),
                    "dhi_clear": _safe(row.get("dhi_clear")),
                    "dni_clear": _safe(row.get("dni_clear")),
                })
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

    # ──────────────────────────────────────────────────────────────────────
    # Forecasts
    # ──────────────────────────────────────────────────────────────────────

    def store_forecast(
        self,
        df: pd.DataFrame,
        capacity_kw: float,
        lat: float,
        lon: float,
    ) -> None:
        if df.empty:
            return
        df = df.copy()
        df["lat"] = lat
        df["lon"] = lon
        df["capacity_kw"] = capacity_kw
        rows = df.reset_index().rename(columns={"index": "timestamp"})
        with self.engine.begin() as conn:
            for _, row in rows.iterrows():
                conn.execute(text("""
                    INSERT INTO forecasts
                        (timestamp, lat, lon, capacity_kw, power_kw,
                         power_dc_kw, ghi, kt, t_cell, g_eff, mm)
                    VALUES
                        (:timestamp, :lat, :lon, :capacity_kw, :power_kw,
                         :power_dc_kw, :ghi, :kt, :t_cell, :g_eff, :mm)
                    ON CONFLICT (timestamp, lat, lon, capacity_kw) DO UPDATE
                        SET power_kw    = EXCLUDED.power_kw,
                            power_dc_kw = EXCLUDED.power_dc_kw,
                            ghi         = EXCLUDED.ghi,
                            kt          = EXCLUDED.kt,
                            t_cell      = EXCLUDED.t_cell,
                            g_eff       = EXCLUDED.g_eff,
                            mm          = EXCLUDED.mm,
                            created_at  = now()
                """), {
                    "timestamp":   row.get("timestamp"),
                    "lat":         lat,
                    "lon":         lon,
                    "capacity_kw": capacity_kw,
                    "power_kw":    _safe(row.get("power_kw")),
                    "power_dc_kw": _safe(row.get("power_dc_kw")),
                    "ghi":         _safe(row.get("ghi")),
                    "kt":          _safe(row.get("kt")),
                    "t_cell":      _safe(row.get("t_cell")),
                    "g_eff":       _safe(row.get("g_eff")),
                    "mm":          _safe(row.get("mm")),
                })


def _safe(v):
    """Convert NaN/None to None (SQL NULL) and floats to Python float."""
    if v is None:
        return None
    try:
        import math
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return str(v) if v else None
