"""Live CAMS ingestion — fetch the next N hours of CAMS forecast for a location.

CLI
---
    python -m solar_forecast.ingestion.cams.live --location-id 1 --hours 12
    python -m solar_forecast.ingestion.cams.live --location-id 1 --hours 12 --dry-run
"""

from __future__ import annotations
import argparse
import logging
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd

from .fetcher import fetch_cams_window
from .client import is_cams_configured

log = logging.getLogger(__name__)

# CAMS runs at 00Z and 12Z; each run covers up to 120h
_CAMS_RUNS = ["00:00", "12:00"]
# How old (minutes) a stored forecast may be before we re-fetch
_FRESHNESS_MINUTES = 90


def _latest_run_for_hour(utc_hour: int) -> tuple[str, str]:
    """Return (date_str, time_str) of the most recent CAMS run relative to utc_hour."""
    now_utc = datetime.now(timezone.utc)
    if utc_hour >= 12:
        run_time = "12:00"
    else:
        # Use previous day 12Z run if today's 00Z isn't out yet (available ~3h after run)
        run_time = "00:00"
    return now_utc.strftime("%Y-%m-%d"), run_time


def _is_fresh(location_id: int, date_str: str, time_str: str) -> bool:
    """Return True if we already have recent CAMS data for this run."""
    try:
        from solar_forecast.db.manager import get_connection
        with get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM cams_atmospheric_forecast "
                "WHERE location_id = ? AND date(run_time_utc) = ? "
                "AND time(run_time_utc) LIKE ?",
                (location_id, date_str, f"{time_str}%"),
            ).fetchone()
        return (row[0] if row else 0) > 0
    except Exception as exc:
        log.debug("freshness check failed: %s", exc)
        return False


def _store(df: pd.DataFrame, location_id: int) -> int:
    if df is None or df.empty:
        return 0
    try:
        from solar_forecast.db.manager import upsert_cams
        return upsert_cams(df, location_id)
    except Exception as exc:
        log.error("DB insert failed: %s", exc)
        return 0


def run_live(
    location_id: int,
    hours: int = 12,
    dry_run: bool = False,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    force: bool = False,
) -> dict:
    """Fetch the latest CAMS forecast window for a location.

    Returns a status dict: {run_date, run_time, rows_stored, skipped, error}.
    """
    _started_at = datetime.now(timezone.utc).isoformat()

    if not is_cams_configured() and not dry_run:
        raise RuntimeError(
            "No CAMS credentials found. Set CADS_KEY or CAMS_API_KEY env var, "
            "or create ~/.cdsapirc. See docs/cams_ingestion.md."
        )

    if lat is None or lon is None:
        try:
            from solar_forecast.db.manager import get_location
            loc = get_location(location_id)
            if loc is None:
                raise ValueError(f"Location {location_id} not found in DB")
            lat, lon = float(loc["lat"]), float(loc["lon"])
        except Exception as exc:
            raise RuntimeError(f"Cannot resolve location {location_id}: {exc}") from exc

    now_utc = datetime.now(timezone.utc)
    date_str, time_str = _latest_run_for_hour(now_utc.hour)

    status = {"run_date": date_str, "run_time": time_str, "rows_stored": 0, "skipped": False, "error": None}

    if not force and _is_fresh(location_id, date_str, time_str):
        log.info("CAMS data is fresh for %s %s — skipping", date_str, time_str)
        status["skipped"] = True
        return status

    log.info("fetching live CAMS %s %s (horizon=%dh lat=%.3f lon=%.3f)",
             date_str, time_str, hours, lat, lon)

    df = fetch_cams_window(
        lat=lat, lon=lon,
        date_str=date_str, time_str=time_str,
        horizon_hours=hours,
        dry_run=dry_run,
    )

    if df is not None:
        n = _store(df, location_id)
        status["rows_stored"] = n
        log.info("stored %d CAMS rows for %s %s", n, date_str, time_str)
    elif dry_run:
        log.info("[DRY-RUN] would have stored CAMS data")
    else:
        status["error"] = "fetch returned None after retries"
        log.warning("live CAMS fetch returned None for %s %s", date_str, time_str)

    if not dry_run:
        try:
            from solar_forecast.db.manager import log_ingestion_run
            log_ingestion_run(
                source="cams_live",
                location_id=location_id,
                rows_inserted=status.get("rows_stored", 0),
                rows_skipped=1 if status.get("skipped") else 0,
                errors=1 if status.get("error") else 0,
                status="skipped" if status.get("skipped") else ("error" if status.get("error") else "ok"),
                detail={"run_date": date_str, "run_time": time_str, "hours": hours},
                started_at=_started_at,
            )
        except Exception as exc:
            log.warning("audit log failed: %s", exc)

    return status


def _cli():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser(description="CAMS live forecast ingestion")
    p.add_argument("--location-id", type=int, required=True, help="Location ID from DB")
    p.add_argument("--hours", type=int, default=12, help="Forecast horizon hours (default 12)")
    p.add_argument("--lat", type=float, default=None, help="Override latitude")
    p.add_argument("--lon", type=float, default=None, help="Override longitude")
    p.add_argument("--force", action="store_true", help="Re-fetch even if data is fresh")
    p.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    args = p.parse_args()

    try:
        status = run_live(
            location_id=args.location_id,
            hours=args.hours,
            dry_run=args.dry_run,
            lat=args.lat,
            lon=args.lon,
            force=args.force,
        )
        print(f"Done: {status}")
        sys.exit(0 if status.get("error") is None else 1)
    except Exception as exc:
        log.error("live fetch failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    _cli()
