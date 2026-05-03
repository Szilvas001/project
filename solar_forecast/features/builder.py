"""Feature frame builder — merges CAMS + Open-Meteo into ML-ready feature vectors.

Data tier priority (4-tier fallback)
-------------------------------------
  Tier 1 — CAMS + Open-Meteo   (full physics)
  Tier 2 — Open-Meteo + CAMS climatology  (CAMS missing for this timestep)
  Tier 3 — Open-Meteo only     (no CAMS at all)
  Tier 4 — Demo constants      (neither CAMS nor OM available)
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Ångström exponent derived from multi-wavelength AOD (550/870 pair)
def _angstrom_exponent(aod_550: float, aod_865: float) -> float:
    if aod_550 <= 0 or aod_865 <= 0:
        return 1.3
    return -np.log(aod_550 / aod_865) / np.log(550 / 865)


# Demo / climatology constants (continental Europe)
_DEMO = {
    "aod_550": 0.12,
    "aod_469": 0.16,
    "aod_670": 0.09,
    "aod_865": 0.06,
    "angstrom_exponent": 1.3,
    "total_column_water_vapour": 15.0,   # kg/m²
    "total_column_ozone": 0.006642,      # kg/m² (~310 DU)
    "surface_pressure": 101325.0,         # Pa
    "boundary_layer_height": 1000.0,      # m
    "temperature_2m": 293.15,             # K
    "cloud_cover": 50.0,
    "cloud_cover_low": 20.0,
    "cloud_cover_mid": 20.0,
    "cloud_cover_high": 20.0,
    "wind_speed_10m": 4.0,
    "shortwave_radiation": None,
    "direct_normal_irradiance": None,
    "diffuse_radiation": None,
}


def build_feature_frame(
    location_id: int,
    start_utc: Optional[str] = None,
    end_utc: Optional[str] = None,
    horizon_hours: int = 72,
) -> tuple[pd.DataFrame, str]:
    """Build merged feature frame for a location.

    Returns (DataFrame, data_tier_str) where data_tier is one of:
    'cams_om', 'om_climatology', 'om_only', 'demo'.

    The DataFrame has one row per hour with columns from both CAMS and OM.
    """
    if start_utc is None:
        start_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00:00")
    if end_utc is None:
        end_utc = (
            datetime.fromisoformat(start_utc.replace("Z", "+00:00")) +
            timedelta(hours=horizon_hours)
        ).strftime("%Y-%m-%dT%H:00:00")

    cams_df = _load_cams(location_id, start_utc, end_utc)
    om_df = _load_om(location_id, start_utc, end_utc)

    if om_df.empty and cams_df.empty:
        return _demo_frame(start_utc, end_utc), "demo"

    if om_df.empty:
        return _cams_only_frame(cams_df), "cams_only"

    if cams_df.empty:
        return _om_plus_climatology(om_df), "om_climatology"

    merged, tier = _merge_cams_om(cams_df, om_df)
    return merged, tier


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------

def _load_cams(location_id: int, start_utc: str, end_utc: str) -> pd.DataFrame:
    try:
        from solar_forecast.db.manager import query_cams
        return query_cams(location_id, start_utc, end_utc)
    except Exception as exc:
        log.debug("CAMS query failed: %s", exc)
        return pd.DataFrame()


def _load_om(location_id: int, start_utc: str, end_utc: str) -> pd.DataFrame:
    try:
        from solar_forecast.db.manager import query_openmeteo
        return query_openmeteo(location_id, start_utc, end_utc)
    except Exception as exc:
        log.debug("OpenMeteo query failed: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Frame builders
# ---------------------------------------------------------------------------

def _add_derived_cams(df: pd.DataFrame) -> pd.DataFrame:
    """Add Ångström exponent and other derived CAMS columns."""
    df = df.copy()
    if "aod_550" in df.columns and "aod_865" in df.columns:
        df["angstrom_exponent"] = df.apply(
            lambda r: _angstrom_exponent(
                r.get("aod_550") or _DEMO["aod_550"],
                r.get("aod_865") or _DEMO["aod_865"],
            ),
            axis=1,
        )
    else:
        df["angstrom_exponent"] = _DEMO["angstrom_exponent"]
    return df


def _merge_cams_om(cams: pd.DataFrame, om: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Merge CAMS + OM on the hour; fill gaps with climatology."""
    cams = cams.copy()
    om = om.copy()

    # Normalise timestamps to hour precision
    cams["hour"] = pd.to_datetime(cams["valid_time_utc"]).dt.floor("h")
    om["hour"] = pd.to_datetime(om["valid_time_utc"]).dt.floor("h")

    merged = om.merge(cams, on="hour", how="left", suffixes=("_om", "_cams"))

    # Fill CAMS cols with climatology where NaN
    cams_numeric_cols = [
        "aod_550", "aod_469", "aod_670", "aod_865",
        "total_column_water_vapour", "total_column_ozone",
        "surface_pressure", "boundary_layer_height",
    ]
    tier = "cams_om"
    for col in cams_numeric_cols:
        if col in merged.columns:
            null_frac = merged[col].isna().mean()
            if null_frac > 0:
                merged[col] = merged[col].fillna(_DEMO.get(col, np.nan))
            if null_frac > 0.5:
                tier = "om_climatology"
        else:
            merged[col] = _DEMO.get(col, np.nan)
            tier = "om_climatology"

    merged = _add_derived_cams(merged)
    return merged, tier


def _om_plus_climatology(om: pd.DataFrame) -> pd.DataFrame:
    """Return OM frame augmented with climatology CAMS values."""
    df = om.copy()
    for col, val in _DEMO.items():
        if col not in df.columns and val is not None:
            df[col] = val
    df = _add_derived_cams(df)
    return df


def _cams_only_frame(cams: pd.DataFrame) -> pd.DataFrame:
    cams = _add_derived_cams(cams.copy())
    return cams


def _demo_frame(start_utc: str, end_utc: str) -> pd.DataFrame:
    """Return a demo frame filled with climatology constants."""
    start = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
    end = datetime.fromisoformat(end_utc.replace("Z", "+00:00"))
    hours = int((end - start).total_seconds() / 3600) + 1
    rows = []
    for h in range(hours):
        t = start + timedelta(hours=h)
        row = {"valid_time_utc": t.isoformat(), "hour": t}
        row.update({k: v for k, v in _DEMO.items() if v is not None})
        rows.append(row)
    return pd.DataFrame(rows)
