"""
SQLite-based multi-location manager.

This is the default database for Solar Forecast Pro.
PostgreSQL remains optional (for CAMS training data only).

Schema:
  locations   — PV system registry (name, coords, capacity, config)
  forecasts   — Cached forecast results (keyed by location + date)
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "solar_forecast.db"


def _ensure_dir():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    _ensure_dir()
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def create_tables() -> None:
    with _conn() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS locations (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            name                 TEXT NOT NULL,
            lat                  REAL NOT NULL,
            lon                  REAL NOT NULL,
            altitude             REAL NOT NULL DEFAULT 0.0,
            capacity_kw          REAL NOT NULL DEFAULT 5.0,
            tilt                 REAL,
            azimuth              REAL,
            technology           TEXT NOT NULL DEFAULT 'mono_si',
            timezone             TEXT NOT NULL DEFAULT 'UTC',
            electricity_price    REAL NOT NULL DEFAULT 0.12,
            feedin_tariff        REAL NOT NULL DEFAULT 0.06,
            system_cost_eur      REAL,
            self_consumption_pct REAL NOT NULL DEFAULT 30.0,
            iam_model            TEXT NOT NULL DEFAULT 'ashrae',
            notes                TEXT,
            created_at           TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at           TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS forecasts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            location_id INTEGER NOT NULL REFERENCES locations(id) ON DELETE CASCADE,
            forecast_date TEXT NOT NULL,
            generated_at  TEXT NOT NULL DEFAULT (datetime('now')),
            horizon_days  INTEGER NOT NULL DEFAULT 7,
            payload_json  TEXT NOT NULL,
            summary_json  TEXT NOT NULL DEFAULT '{}'
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_forecast_loc_date
            ON forecasts(location_id, forecast_date);
        """)


# ── Location CRUD ──────────────────────────────────────────────────────────

def list_locations() -> list[dict[str, Any]]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM locations ORDER BY name").fetchall()
        return [dict(r) for r in rows]


def get_location(location_id: int) -> Optional[dict[str, Any]]:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM locations WHERE id = ?", (location_id,)
        ).fetchone()
        return dict(row) if row else None


def create_location(data: dict[str, Any]) -> dict[str, Any]:
    required = {"name", "lat", "lon"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    cols = ["name", "lat", "lon", "altitude", "capacity_kw",
            "tilt", "azimuth", "technology", "timezone",
            "electricity_price", "feedin_tariff", "system_cost_eur",
            "self_consumption_pct", "iam_model", "notes"]
    vals = [
        str(data["name"]),
        float(data["lat"]),
        float(data["lon"]),
        float(data.get("altitude", 0.0)),
        float(data.get("capacity_kw", 5.0)),
        float(data["tilt"]) if data.get("tilt") is not None else None,
        float(data["azimuth"]) if data.get("azimuth") is not None else None,
        str(data.get("technology", "mono_si")),
        str(data.get("timezone", "UTC")),
        float(data.get("electricity_price", 0.12)),
        float(data.get("feedin_tariff", 0.06)),
        float(data["system_cost_eur"]) if data.get("system_cost_eur") is not None else None,
        float(data.get("self_consumption_pct", 30.0)),
        str(data.get("iam_model", "ashrae")),
        str(data["notes"]) if data.get("notes") else None,
    ]
    with _conn() as con:
        cur = con.execute(
            f"INSERT INTO locations ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            vals,
        )
        return get_location(cur.lastrowid)


def update_location(location_id: int, data: dict[str, Any]) -> Optional[dict[str, Any]]:
    allowed = {"name", "lat", "lon", "altitude", "capacity_kw",
               "tilt", "azimuth", "technology", "timezone",
               "electricity_price", "feedin_tariff", "system_cost_eur",
               "self_consumption_pct", "iam_model", "notes"}
    updates = {k: v for k, v in data.items() if k in allowed}
    if not updates:
        return get_location(location_id)

    updates["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    vals = list(updates.values()) + [location_id]
    with _conn() as con:
        con.execute(
            f"UPDATE locations SET {set_clause} WHERE id = ?", vals
        )
    return get_location(location_id)


def delete_location(location_id: int) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM locations WHERE id = ?", (location_id,))
        return cur.rowcount > 0


# ── Forecast cache ─────────────────────────────────────────────────────────

def save_forecast(location_id: int, forecast_date: str,
                  payload: list[dict], summary: dict) -> None:
    with _conn() as con:
        con.execute("""
        INSERT INTO forecasts (location_id, forecast_date, payload_json, summary_json)
             VALUES (?, ?, ?, ?)
             ON CONFLICT(location_id, forecast_date)
             DO UPDATE SET
                 payload_json = excluded.payload_json,
                 summary_json = excluded.summary_json,
                 generated_at = datetime('now')
        """, (location_id, forecast_date, json.dumps(payload), json.dumps(summary)))


def load_forecast(location_id: int, forecast_date: str) -> Optional[dict]:
    with _conn() as con:
        row = con.execute("""
            SELECT payload_json, summary_json, generated_at
              FROM forecasts
             WHERE location_id = ? AND forecast_date = ?
        """, (location_id, forecast_date)).fetchone()
        if not row:
            return None
        return {
            "payload": json.loads(row["payload_json"]),
            "summary": json.loads(row["summary_json"]),
            "generated_at": row["generated_at"],
        }


def migrate_schema() -> None:
    """Add new columns to existing locations table (safe to run repeatedly)."""
    new_cols = [
        ("electricity_price",    "REAL NOT NULL DEFAULT 0.12"),
        ("feedin_tariff",        "REAL NOT NULL DEFAULT 0.06"),
        ("system_cost_eur",      "REAL"),
        ("self_consumption_pct", "REAL NOT NULL DEFAULT 30.0"),
        ("iam_model",            "TEXT NOT NULL DEFAULT 'ashrae'"),
        ("notes",                "TEXT"),
    ]
    with _conn() as con:
        existing = {row[1] for row in con.execute("PRAGMA table_info(locations)").fetchall()}
        for col, defn in new_cols:
            if col not in existing:
                con.execute(f"ALTER TABLE locations ADD COLUMN {col} {defn}")


def seed_demo_location() -> None:
    """Insert Budapest + Vienna demo locations if the table is empty."""
    with _conn() as con:
        count = con.execute("SELECT COUNT(*) FROM locations").fetchone()[0]
        if count == 0:
            con.execute("""
            INSERT INTO locations (name, lat, lon, altitude, capacity_kw,
                                   tilt, azimuth, technology, timezone,
                                   electricity_price, feedin_tariff,
                                   system_cost_eur, self_consumption_pct)
            VALUES ('Budapest — Residential', 47.4979, 19.0402, 120.0, 5.0,
                    36.0, 180.0, 'mono_si', 'Europe/Budapest',
                    0.13, 0.06, 4500.0, 35.0)
            """)
            con.execute("""
            INSERT INTO locations (name, lat, lon, altitude, capacity_kw,
                                   tilt, azimuth, technology, timezone,
                                   electricity_price, feedin_tariff,
                                   system_cost_eur, self_consumption_pct)
            VALUES ('Vienna — Office Park', 48.2082, 16.3738, 180.0, 12.0,
                    30.0, 180.0, 'poly_si', 'Europe/Vienna',
                    0.15, 0.07, 10800.0, 25.0)
            """)
