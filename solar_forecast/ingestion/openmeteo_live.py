"""Open-Meteo live ingestion — fetch the next N hours of weather for a location.

Fetches 15 specific variables with deduplication by location_id + valid_time_utc.

CLI
---
    python -m solar_forecast.ingestion.openmeteo_live --location-id 1 --hours 72
    python -m solar_forecast.ingestion.openmeteo_live --location-id 1 --hours 72 --dry-run
"""

from __future__ import annotations
import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger(__name__)

_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# 15 required variables (mix of weather + irradiance)
_OM_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
]


def fetch_openmeteo(
    lat: float,
    lon: float,
    hours: int = 72,
    timezone_str: str = "UTC",
) -> Optional[pd.DataFrame]:
    """Download Open-Meteo forecast and return a wide DataFrame.

    Returns columns: valid_time_utc + one column per variable.
    Returns None on network/parse failure.
    """
    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly":    ",".join(_OM_VARIABLES),
        "timezone":  timezone_str,
        "forecast_days": max(1, (hours + 23) // 24),
    }
    try:
        resp = requests.get(_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.error("Open-Meteo request failed: %s", exc)
        return None

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        log.warning("Open-Meteo returned no hourly data")
        return None

    rows = []
    for i, t in enumerate(times[:hours]):
        row = {"valid_time_utc": t}
        for var in _OM_VARIABLES:
            vals = hourly.get(var, [])
            row[var] = vals[i] if i < len(vals) else None
        rows.append(row)

    df = pd.DataFrame(rows)
    log.info("Open-Meteo: fetched %d hourly rows (lat=%.3f lon=%.3f)", len(df), lat, lon)
    return df


def run_openmeteo_live(
    location_id: int,
    hours: int = 72,
    dry_run: bool = False,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    timezone_str: str = "UTC",
) -> dict:
    """Fetch and store Open-Meteo forecast for a location.

    Returns a status dict: {rows_fetched, rows_inserted, error}.
    """
    if lat is None or lon is None:
        try:
            from solar_forecast.db.manager import get_location
            loc = get_location(location_id)
            if loc is None:
                raise ValueError(f"Location {location_id} not found in DB")
            lat, lon = float(loc["lat"]), float(loc["lon"])
            timezone_str = loc.get("timezone", "UTC")
        except Exception as exc:
            raise RuntimeError(f"Cannot resolve location {location_id}: {exc}") from exc

    status: dict = {"rows_fetched": 0, "rows_inserted": 0, "error": None}

    df = fetch_openmeteo(lat=lat, lon=lon, hours=hours, timezone_str=timezone_str)
    if df is None:
        status["error"] = "fetch returned None"
        return status

    status["rows_fetched"] = len(df)

    if dry_run:
        log.info("[DRY-RUN] would store %d Open-Meteo rows for location %d", len(df), location_id)
        return status

    try:
        from solar_forecast.db.manager import upsert_openmeteo
        n = upsert_openmeteo(df, location_id)
        status["rows_inserted"] = n
        log.info("stored %d new Open-Meteo rows for location %d", n, location_id)
    except Exception as exc:
        status["error"] = str(exc)
        log.error("Open-Meteo DB insert failed: %s", exc)

    return status


def _cli():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser(description="Open-Meteo live forecast ingestion")
    p.add_argument("--location-id", type=int, required=True, help="Location ID from DB")
    p.add_argument("--hours", type=int, default=72, help="Forecast horizon hours (default 72)")
    p.add_argument("--lat", type=float, default=None, help="Override latitude")
    p.add_argument("--lon", type=float, default=None, help="Override longitude")
    p.add_argument("--timezone", type=str, default="UTC", help="Timezone string (default UTC)")
    p.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    args = p.parse_args()

    try:
        status = run_openmeteo_live(
            location_id=args.location_id,
            hours=args.hours,
            dry_run=args.dry_run,
            lat=args.lat,
            lon=args.lon,
            timezone_str=args.timezone,
        )
        print(f"Done: {status}")
        sys.exit(0 if status.get("error") is None else 1)
    except Exception as exc:
        log.error("openmeteo_live failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    _cli()
