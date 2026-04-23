"""Shared utilities: config loading, tilt/azimuth defaults, interpolation."""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load YAML config and overlay environment variables for secrets."""
    load_dotenv()
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Overlay env vars for secrets
    if api_key := os.getenv("CAMS_API_KEY"):
        cfg.setdefault("cams", {})["api_key"] = api_key
    if pw := os.getenv("PGPASSWORD"):
        cfg.setdefault("database", {})["password"] = pw
    if host := os.getenv("PGHOST"):
        cfg["database"]["host"] = host
    if db := os.getenv("PGDATABASE"):
        cfg["database"]["name"] = db
    if user := os.getenv("PGUSER"):
        cfg["database"]["user"] = user

    return cfg


def resolve_tilt_azimuth(cfg: dict) -> tuple[float, float]:
    """
    Return (tilt, azimuth) from config, applying defaults when null.

    Default logic:
      tilt    = latitude × 0.76  (optimal annual yield heuristic)
      azimuth = 180° (south) for northern hemisphere, 0° (north) for southern
    """
    lat = cfg["location"]["lat"]
    tilt = cfg["system"].get("tilt")
    azimuth = cfg["system"].get("azimuth")

    if tilt is None:
        tilt = abs(lat) * 0.76
        tilt = round(tilt, 1)

    if azimuth is None:
        azimuth = 180.0 if lat >= 0 else 0.0

    return float(tilt), float(azimuth)


def resample_to_1min(df: pd.DataFrame, method: str = "cubic") -> pd.DataFrame:
    """
    Resample hourly or 3-hourly DataFrame to 1-minute frequency.

    Uses cubic spline for smooth atmospheric variables and linear for
    bounded variables (cloud cover, fractions) to avoid overshoots.
    """
    if df.empty:
        return df

    idx_1min = pd.date_range(df.index[0], df.index[-1], freq="1min")

    # Bounded columns: clamp after interpolation
    bounded = {"cloud_cover", "aod_550nm", "total_ozone",
               "precipitable_water", "surface_pressure"}

    df_out = df.reindex(df.index.union(idx_1min))

    for col in df_out.columns:
        if col in bounded:
            df_out[col] = df_out[col].interpolate("linear")
        else:
            try:
                df_out[col] = df_out[col].interpolate(method)
            except Exception:
                df_out[col] = df_out[col].interpolate("linear")

    df_out = df_out.reindex(idx_1min)

    # Clamp bounded variables
    if "cloud_cover" in df_out:
        df_out["cloud_cover"] = df_out["cloud_cover"].clip(0.0, 1.0)
    if "aod_550nm" in df_out:
        df_out["aod_550nm"] = df_out["aod_550nm"].clip(0.0, 5.0)
    if "precipitable_water" in df_out:
        df_out["precipitable_water"] = df_out["precipitable_water"].clip(0.0, 10.0)

    return df_out


def cyclic_encode(series: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    """Return (sin, cos) cyclic encoding of a periodic variable."""
    angle = 2 * np.pi * series / period
    return np.sin(angle), np.cos(angle)


def geocode_city(city: str) -> tuple[float, float, str]:
    """
    Convert city name to (lat, lon, display_name) using Open-Meteo geocoding.
    Returns None on failure.
    """
    import requests
    url = "https://geocoding-api.open-meteo.com/v1/search"
    resp = requests.get(url, params={"name": city, "count": 1, "language": "en"}, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        raise ValueError(f"City not found: {city!r}")
    r = results[0]
    return float(r["latitude"]), float(r["longitude"]), r.get("name", city)
