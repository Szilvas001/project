"""GRIB file processor — read, bilinear-interpolate, pivot to wide format.

`pygrib` is imported lazily so the rest of the package works in environments
that don't ship libeccodes (e.g. CI, Docker images for the dashboard).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Unit corrections ──────────────────────────────────────────────────────

UNIT_ADJUSTMENTS = {
    "divide_by_10":  lambda x: x / 10.0,
    "divide_by_100": lambda x: x / 100.0,
}


def apply_unit_adjustments(df: pd.DataFrame, adjustments: dict) -> pd.DataFrame:
    """Apply per-column unit corrections from config."""
    for col, op in adjustments.items():
        if col in df.columns:
            fn = UNIT_ADJUSTMENTS.get(op)
            if fn:
                df[col] = fn(df[col])
                log.debug("unit-adjust %s → %s", col, op)
            else:
                log.warning("unknown unit adjustment %s (skipped)", op)
    return df


# ── Bilinear interpolation ────────────────────────────────────────────────

def bilinear_interpolate(
    lats_2d: np.ndarray,
    lons_2d: np.ndarray,
    values_2d: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> float:
    """Bilinear interpolation of a 2-D grid at a single target point.

    Handles descending-latitude grids (common in GRIB) and numpy masked
    arrays.
    """
    lats_1d = np.sort(np.unique(lats_2d))
    lons_1d = np.sort(np.unique(lons_2d))

    j = int(np.searchsorted(lats_1d, target_lat)) - 1
    i = int(np.searchsorted(lons_1d, target_lon)) - 1
    j = max(0, min(j, len(lats_1d) - 2))
    i = max(0, min(i, len(lons_1d) - 2))

    lat0, lat1 = lats_1d[j], lats_1d[j + 1]
    lon0, lon1 = lons_1d[i], lons_1d[i + 1]

    vals = (
        np.ma.filled(values_2d, fill_value=np.nan)
        if isinstance(values_2d, np.ma.MaskedArray)
        else np.asarray(values_2d, dtype=float)
    )
    flat_lats = lats_2d.ravel()
    flat_lons = lons_2d.ravel()
    flat_vals = vals.ravel()

    def nearest(lat: float, lon: float) -> float:
        idx = int(np.argmin((flat_lats - lat) ** 2 + (flat_lons - lon) ** 2))
        return float(flat_vals[idx])

    q11, q21 = nearest(lat0, lon0), nearest(lat0, lon1)
    q12, q22 = nearest(lat1, lon0), nearest(lat1, lon1)

    dlat = lat1 - lat0 or 1.0
    dlon = lon1 - lon0 or 1.0
    t = (target_lat - lat0) / dlat
    u = (target_lon - lon0) / dlon

    return (1 - t) * (1 - u) * q11 + (1 - t) * u * q21 + t * (1 - u) * q12 + t * u * q22


# ── GRIB I/O ──────────────────────────────────────────────────────────────

def parse_grib_file(
    path: str,
    target_lat: float,
    target_lon: float,
) -> pd.DataFrame:
    """Read a GRIB file and return a long-format DataFrame.

    Columns: reference_time, forecast_hours, variable, model_level, value
    """
    try:
        import pygrib
    except ImportError as exc:
        raise ImportError(
            "pygrib is required to read GRIB files. "
            "Install libeccodes-dev and `pip install pygrib`."
        ) from exc

    rows: list[dict[str, Any]] = []
    with pygrib.open(path) as grbs:
        for grb in grbs:
            try:
                lats, lons = grb.latlons()
                value = bilinear_interpolate(lats, lons, grb.values, target_lat, target_lon)

                ref_time = pd.Timestamp(
                    year=grb.year, month=grb.month, day=grb.day,
                    hour=grb.hour, minute=grb.minute, tz="UTC",
                )
                forecast_hours = int(getattr(grb, "stepRange", grb.endStep or 0))
                model_level = int(getattr(grb, "level", 0))

                rows.append({
                    "reference_time": ref_time,
                    "forecast_hours": forecast_hours,
                    "variable": grb.shortName,
                    "model_level": model_level,
                    "value": value,
                })
            except Exception as exc:
                log.warning("GRIB message skipped (%s): %s", grb.shortName, exc)

    if not rows:
        raise ValueError(f"No GRIB messages decoded from {path}")

    return pd.DataFrame(rows)


# ── Pivot + clean ─────────────────────────────────────────────────────────

def pivot_and_clean(
    df_long: pd.DataFrame,
    dataset_config: dict,
) -> pd.DataFrame:
    """Long → wide pivot + unit corrections."""
    pk = dataset_config["primary_key"]
    has_model_level = "model_level" in pk

    index_cols = ["reference_time", "forecast_hours"]
    if has_model_level:
        index_cols.append("model_level")

    df_wide = df_long.pivot_table(
        index=index_cols,
        columns="variable",
        values="value",
        aggfunc="mean",
    ).reset_index()
    df_wide.columns.name = None

    adjustments = dataset_config.get("unit_adjustments", {})
    if adjustments:
        df_wide = apply_unit_adjustments(df_wide, adjustments)

    return df_wide
