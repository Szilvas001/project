"""Core CAMS fetch → parse → store pipeline."""

from __future__ import annotations
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .client import get_cams_client
from .parser import parse_grib, pivot_to_wide
from .variables import CAMS_LONG_NAMES

log = logging.getLogger(__name__)

DATASET = "cams-global-atmospheric-composition-forecasts"
# Only request variables the product actually needs (speed optimisation)
_VARIABLES = CAMS_LONG_NAMES


def _leadtime_range(hours: int) -> list[str]:
    return [str(h) for h in range(0, hours + 1)]


def fetch_cams_window(
    lat: float,
    lon: float,
    date_str: str,          # YYYY-MM-DD
    time_str: str = "00:00",
    horizon_hours: int = 12,
    area_margin: float = 0.5,
    dry_run: bool = False,
) -> Optional[pd.DataFrame]:
    """Download one CAMS forecast window and return wide DataFrame.

    Parameters
    ----------
    lat, lon        : target coordinates
    date_str        : forecast reference date  "YYYY-MM-DD"
    time_str        : forecast reference time  "00:00" or "12:00"
    horizon_hours   : number of lead-time hours to retrieve
    area_margin     : bounding box half-width in degrees
    dry_run         : if True, skip actual download and return None

    Returns
    -------
    pd.DataFrame with columns: run_time_utc, valid_time_utc,
    forecast_step_hours, + all CAMS internal variable names.
    Returns None on failure.
    """
    if dry_run:
        log.info("[DRY-RUN] would fetch CAMS %s %s horizon=%dh", date_str, time_str, horizon_hours)
        return None

    client = get_cams_client()
    request = {
        "variable":     _VARIABLES,
        "date":         [f"{date_str}/{date_str}"],
        "time":         [time_str],
        "leadtime_hour": _leadtime_range(horizon_hours),
        "type":         ["forecast"],
        "data_format":  "grib",
        "area":         [lat + area_margin, lon - area_margin,
                         lat - area_margin, lon + area_margin],
    }

    tmp = tempfile.NamedTemporaryFile(suffix=".grib", delete=False)
    tmp.close()
    try:
        log.info("fetching CAMS %s %s (%dh) …", date_str, time_str, horizon_hours)
        result = client.retrieve(DATASET, request)
        result.download(tmp.name)
        log.info("GRIB downloaded: %s", tmp.name)

        df_long = parse_grib(tmp.name, lat, lon)
        df_wide = pivot_to_wide(df_long)
        return df_wide

    except Exception as exc:
        log.error("CAMS fetch failed: %s", exc)
        return None
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
