"""CAMS GRIB / netCDF parser with bilinear interpolation to target point.

pygrib is imported lazily — the rest of the package works without it.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .variables import map_row, CAMS_VARIABLES

log = logging.getLogger(__name__)


# ── Bilinear interpolation ────────────────────────────────────────────────

def bilinear_interp(lats_2d, lons_2d, values_2d, target_lat, target_lon) -> float:
    lats_1d = np.sort(np.unique(lats_2d))
    lons_1d = np.sort(np.unique(lons_2d))
    j = max(0, min(int(np.searchsorted(lats_1d, target_lat)) - 1, len(lats_1d) - 2))
    i = max(0, min(int(np.searchsorted(lons_1d, target_lon)) - 1, len(lons_1d) - 2))
    lat0, lat1 = lats_1d[j], lats_1d[j + 1]
    lon0, lon1 = lons_1d[i], lons_1d[i + 1]

    vals = (np.ma.filled(values_2d, np.nan) if isinstance(values_2d, np.ma.MaskedArray)
            else np.asarray(values_2d, float))
    fl, flo, fv = lats_2d.ravel(), lons_2d.ravel(), vals.ravel()
    nearest = lambda la, lo: float(fv[int(np.argmin((fl - la) ** 2 + (flo - lo) ** 2))])
    q11, q21 = nearest(lat0, lon0), nearest(lat0, lon1)
    q12, q22 = nearest(lat1, lon0), nearest(lat1, lon1)
    t = (target_lat - lat0) / (lat1 - lat0 or 1.0)
    u = (target_lon - lon0) / (lon1 - lon0 or 1.0)
    return (1 - t) * (1 - u) * q11 + (1 - t) * u * q21 + t * (1 - u) * q12 + t * u * q22


# ── GRIB parser ───────────────────────────────────────────────────────────

def parse_grib(path: str | Path, lat: float, lon: float) -> pd.DataFrame:
    """Parse GRIB file → long-format DataFrame.

    Returns columns: run_time_utc, valid_time_utc, forecast_step_hours,
    variable_cams (short name), value.
    """
    try:
        import pygrib
    except ImportError as exc:
        raise ImportError(
            "pygrib needed for GRIB: apt install libeccodes-dev && pip install pygrib"
        ) from exc

    rows = []
    with pygrib.open(str(path)) as grbs:
        for grb in grbs:
            try:
                lats, lons = grb.latlons()
                value = bilinear_interp(lats, lons, grb.values, lat, lon)
                run_time = pd.Timestamp(
                    year=grb.year, month=grb.month, day=grb.day,
                    hour=grb.hour, minute=grb.minute, tz="UTC",
                )
                step = int(getattr(grb, "endStep", 0) or 0)
                valid_time = run_time + pd.Timedelta(hours=step)
                rows.append({
                    "run_time_utc":         run_time,
                    "valid_time_utc":       valid_time,
                    "forecast_step_hours":  step,
                    "variable_cams":        grb.shortName,
                    "value":                float(value),
                })
            except Exception as exc:
                log.debug("skipping GRIB msg %s: %s", getattr(grb, "shortName", "?"), exc)

    if not rows:
        raise ValueError(f"No GRIB messages decoded from {path}")
    return pd.DataFrame(rows)


def pivot_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format GRIB DataFrame → wide, then map to internal names."""
    idx = ["run_time_utc", "valid_time_utc", "forecast_step_hours"]
    df_wide = (df_long
               .pivot_table(index=idx, columns="variable_cams", values="value", aggfunc="mean")
               .reset_index())
    df_wide.columns.name = None

    # Map CAMS short names → internal names
    raw_dict = df_wide.to_dict(orient="list")
    mapped_rows = []
    for i in range(len(df_wide)):
        raw_row = {col: df_wide[col].iloc[i] for col in df_wide.columns}
        mapped = map_row(raw_row)
        mapped["run_time_utc"]        = raw_row["run_time_utc"]
        mapped["valid_time_utc"]      = raw_row["valid_time_utc"]
        mapped["forecast_step_hours"] = raw_row["forecast_step_hours"]
        mapped_rows.append(mapped)
    return pd.DataFrame(mapped_rows)
