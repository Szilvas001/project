"""Backfill — detect and re-pull missing forecast runs."""

from __future__ import annotations

import logging
from datetime import date, timedelta

log = logging.getLogger(__name__)


def _expected_dates(start_date: date, schedule: dict) -> list[tuple[date, int]]:
    """All (date, UTC-hour) pairs from `start_date` up to yesterday."""
    hours = sorted(int(t.split(":")[0]) for t in schedule)
    today = date.today()
    out: list[tuple[date, int]] = []
    d = start_date
    while d < today:
        for h in hours:
            out.append((d, h))
        d += timedelta(days=1)
    return out


def find_missing_forecasts(
    cur,
    dataset_config: dict,
    schedule: dict,
) -> list[tuple[date, int]]:
    """Return (date, UTC-hour) pairs that are absent from the target table."""
    backfill_cfg = dataset_config.get("backfill", {})
    if not backfill_cfg:
        return []

    start_str = backfill_cfg.get("start_date", "")
    if not start_str:
        return []

    try:
        start = date.fromisoformat(start_str)
    except ValueError:
        log.error("invalid backfill.start_date: %s", start_str)
        return []

    max_per_run = int(backfill_cfg.get("max_per_run", 50))
    table = dataset_config["target_table"]

    try:
        cur.execute(
            f"SELECT DISTINCT DATE(reference_time AT TIME ZONE 'UTC'), "
            f"EXTRACT(HOUR FROM reference_time AT TIME ZONE 'UTC')::int "
            f"FROM {table}"
        )
        existing = {(row[0], row[1]) for row in cur.fetchall()}
    except Exception as exc:
        log.warning("could not read %s for backfill: %s", table, exc)
        existing = set()

    all_expected = _expected_dates(start, schedule)
    missing = [(d, h) for d, h in all_expected if (d, h) not in existing]

    if missing:
        log.info("[%s] %d missing run(s); processing up to %d",
                 dataset_config["name"], len(missing), max_per_run)

    return missing[:max_per_run]
