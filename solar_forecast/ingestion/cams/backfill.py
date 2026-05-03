"""Historical CAMS backfill — fetch last N days of CAMS data.

CLI
---
    python -m solar_forecast.ingestion.cams.backfill --location-id 1 --days 365
    python -m solar_forecast.ingestion.cams.backfill --location-id 1 --days 30 --dry-run
"""

from __future__ import annotations
import argparse
import logging
import sys
import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from .fetcher import fetch_cams_window
from .client import is_cams_configured

log = logging.getLogger(__name__)

# CAMS delivers two forecast runs per day: 00Z and 12Z
_RUNS = [("00:00", 0), ("12:00", 12)]   # (time_str, UTC hour offset)
_CHUNK_DAYS = 1          # one day per CAMS request (safe for quota)
_RETRY_SLEEP = [30, 60, 120]   # seconds before each retry attempt


def _existing_dates(location_id: int) -> set[tuple[str, str]]:
    """Return set of (date_str, time_str) already in DB for this location."""
    try:
        from solar_forecast.db.manager import get_connection
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT date(run_time_utc), "
                "time(run_time_utc) FROM cams_atmospheric_forecast "
                "WHERE location_id = ?", (location_id,)
            ).fetchall()
        return {(r[0], r[1][:5]) for r in rows}
    except Exception as exc:
        log.warning("could not query existing CAMS dates: %s", exc)
        return set()


def _store(df: pd.DataFrame, location_id: int) -> int:
    """Insert CAMS rows; return number of rows inserted."""
    if df is None or df.empty:
        return 0
    try:
        from solar_forecast.db.manager import upsert_cams
        return upsert_cams(df, location_id)
    except Exception as exc:
        log.error("DB insert failed: %s", exc)
        return 0


def run_backfill(
    location_id: int,
    days: int = 365,
    dry_run: bool = False,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> dict:
    """Run historical CAMS backfill for a location.

    Returns a status dict: {total_days, fetched, skipped, errors}
    """
    if not is_cams_configured() and not dry_run:
        raise RuntimeError(
            "No CAMS credentials found. Set CADS_KEY or CAMS_API_KEY env var, "
            "or create ~/.cdsapirc. See docs/cams_ingestion.md."
        )

    # Resolve location coordinates if not provided
    if lat is None or lon is None:
        try:
            from solar_forecast.db.manager import get_location
            loc = get_location(location_id)
            if loc is None:
                raise ValueError(f"Location {location_id} not found in DB")
            lat, lon = float(loc["lat"]), float(loc["lon"])
        except Exception as exc:
            raise RuntimeError(f"Cannot resolve location {location_id}: {exc}") from exc

    today = date.today()
    start = today - timedelta(days=days - 1)
    existing = _existing_dates(location_id)

    stats = {"total_days": days * len(_RUNS), "fetched": 0, "skipped": 0, "errors": 0}
    current = start

    while current <= today:
        date_str = current.isoformat()
        for time_str, _ in _RUNS:
            key = (date_str, time_str)
            if key in existing:
                log.debug("skip existing: %s %s", date_str, time_str)
                stats["skipped"] += 1
                current += timedelta(days=1) if time_str == _RUNS[-1][0] else timedelta(0)
                continue

            log.info("backfill: %s %s (lat=%.3f lon=%.3f)", date_str, time_str, lat, lon)

            df = None
            for attempt, sleep_s in enumerate([0] + _RETRY_SLEEP):
                if sleep_s:
                    log.info("retry %d/%d after %ds", attempt, len(_RETRY_SLEEP), sleep_s)
                    time.sleep(sleep_s)
                df = fetch_cams_window(
                    lat=lat, lon=lon,
                    date_str=date_str, time_str=time_str,
                    horizon_hours=12, dry_run=dry_run,
                )
                if df is not None or dry_run:
                    break

            if df is not None:
                n = _store(df, location_id)
                stats["fetched"] += 1
                log.info("stored %d rows for %s %s", n, date_str, time_str)
            elif not dry_run:
                stats["errors"] += 1
                log.warning("failed to fetch %s %s after retries", date_str, time_str)
            else:
                stats["fetched"] += 1

        current += timedelta(days=1)

    log.info("backfill complete: %s", stats)
    return stats


def _cli():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser(description="CAMS historical backfill")
    p.add_argument("--location-id", type=int, required=True, help="Location ID from DB")
    p.add_argument("--days", type=int, default=365, help="Days to backfill (default 365)")
    p.add_argument("--lat", type=float, default=None, help="Override latitude")
    p.add_argument("--lon", type=float, default=None, help="Override longitude")
    p.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    args = p.parse_args()

    try:
        stats = run_backfill(
            location_id=args.location_id,
            days=args.days,
            dry_run=args.dry_run,
            lat=args.lat,
            lon=args.lon,
        )
        print(f"Done: {stats}")
        sys.exit(0 if stats["errors"] == 0 else 1)
    except Exception as exc:
        log.error("backfill failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    _cli()
