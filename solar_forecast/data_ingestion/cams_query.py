"""Read CAMS atmospheric state from PostgreSQL into the forecast pipeline.

This bridges the CAMS Fetcher (`solar_forecast.cams_fetcher`) to the
all-sky / clear-sky models. The forecast pipeline calls
`load_cams_atmospheric_state()` for a target time range; if the Postgres
store is reachable and populated, the returned DataFrame contains
hourly-resampled atmospheric variables. Otherwise an empty frame is returned
and the caller falls back to climatology.

Returned columns (where available)
----------------------------------
    aod_550nm                  total AOD at 550 nm
    aod_469nm, aod_670nm       multi-wavelength AOD (Ångström α)
    aod_865nm, aod_1240nm
    ozone_du                   total column ozone (Dobson units)
    precipitable_water         cm
    surface_pressure_hpa
    boundary_layer_height_m
    temp_2m_c                  near-surface temperature
    aod_dust_550nm             speciated AODs (SSA / asymmetry mixing)
    aod_bc_550nm
    aod_om_550nm
    aod_ss_550nm
    aod_so4_550nm
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Map raw CAMS variable names → pipeline-friendly column names
_RENAME = {
    "total_column_ozone":                       "ozone_atm_cm",
    "total_column_water_vapour":                "water_vapour_kg_m2",
    "total_aerosol_optical_depth_550nm":        "aod_550nm",
    "total_aerosol_optical_depth_469nm":        "aod_469nm",
    "total_aerosol_optical_depth_670nm":        "aod_670nm",
    "total_aerosol_optical_depth_865nm":        "aod_865nm",
    "total_aerosol_optical_depth_1240nm":       "aod_1240nm",
    "boundary_layer_height":                    "boundary_layer_height_m",
    "2m_temperature":                           "temp_2m_k",
    "surface_pressure":                         "surface_pressure_pa",
    "dust_aerosol_optical_depth_550nm":         "aod_dust_550nm",
    "black_carbon_aerosol_optical_depth_550nm": "aod_bc_550nm",
    "organic_matter_aerosol_optical_depth_550nm": "aod_om_550nm",
    "sea_salt_aerosol_optical_depth_550nm":     "aod_ss_550nm",
    "sulphate_aerosol_optical_depth_550nm":     "aod_so4_550nm",
}


def _to_pipeline_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert CAMS native units → forecast-pipeline units."""
    if df.empty:
        return df
    out = df.copy()
    if "ozone_atm_cm" in out:
        # CAMS reports kg/m² for total column ozone; 1 atm-cm ≈ 21.4 g/m²
        # but the forecast pipeline already uses Dobson Units (1 DU = 0.001 atm-cm)
        # The reanalysis variable in some CAMS products is already Dobson — handle both
        # by treating large values as DU and small ones as kg/m².
        v = out["ozone_atm_cm"].astype(float)
        is_kg_m2 = v.median() < 1.0
        out["ozone_du"] = (v * 46_697.0) if is_kg_m2 else v
    if "water_vapour_kg_m2" in out:
        # 1 cm precipitable water ≈ 10 kg/m²
        out["precipitable_water"] = out["water_vapour_kg_m2"].astype(float) / 10.0
    if "temp_2m_k" in out:
        out["temp_2m_c"] = out["temp_2m_k"].astype(float) - 273.15
    if "surface_pressure_pa" in out:
        out["surface_pressure_hpa"] = out["surface_pressure_pa"].astype(float) / 100.0
    return out


def load_cams_atmospheric_state(
    times: pd.DatetimeIndex,
    target_lat: float,
    target_lon: float,
    tables: tuple[str, ...] = ("cams_surface", "cams_species"),
) -> pd.DataFrame:
    """Return per-timestep CAMS atmospheric state aligned to `times`.

    Returns an **empty DataFrame** if PostgreSQL is unreachable or the
    tables haven't been populated yet — the caller should treat that as
    "fall back to climatology".
    """
    try:
        from solar_forecast.cams_fetcher import db as cams_db
    except Exception as exc:
        log.debug("cams_fetcher unavailable: %s", exc)
        return pd.DataFrame(index=times)

    if not isinstance(times, pd.DatetimeIndex):
        times = pd.DatetimeIndex(times)
    if times.tz is None:
        times = times.tz_localize("UTC")

    target = times[len(times) // 2]

    try:
        conn = cams_db.get_connection()
    except Exception as exc:
        log.info("CAMS DB unreachable (%s) — using climatology", exc)
        return pd.DataFrame(index=times)

    frames: list[pd.DataFrame] = []
    try:
        cur = conn.cursor()
        for tbl in tables:
            try:
                df = cams_db.read_latest_forecast(cur, tbl, target, horizon_hours=len(times))
            except Exception as exc:
                log.debug("read %s failed: %s", tbl, exc)
                continue
            if df is None or df.empty:
                continue
            df = df.rename(columns=_RENAME)
            if "reference_time" in df and "forecast_hours" in df:
                df["valid_time"] = pd.to_datetime(df["reference_time"], utc=True) \
                    + pd.to_timedelta(df["forecast_hours"], unit="h")
                df = df.set_index("valid_time")
            frames.append(df)
        cur.close()
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not frames:
        return pd.DataFrame(index=times)

    merged = pd.concat(frames, axis=1)
    merged = merged.loc[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()
    merged = _to_pipeline_units(merged)

    # Resample / interpolate to the requested hourly grid
    out = merged.reindex(times.union(merged.index)).interpolate(method="time").reindex(times)
    return out


def angstrom_alpha(aod_short: pd.Series, aod_long: pd.Series,
                   wl_short: float, wl_long: float) -> pd.Series:
    """Ångström exponent α between two AOD wavelengths."""
    eps = 1e-9
    s = aod_short.clip(lower=eps).astype(float)
    l = aod_long.clip(lower=eps).astype(float)
    return -np.log(s / l) / np.log(wl_short / wl_long)


def derive_extras(df: pd.DataFrame) -> pd.DataFrame:
    """Attach Ångström exponents and SSA-mixing diagnostics if AOD species
    are present.

    Adds where available:
        angstrom_alpha1   from 469 / 870 nm
        angstrom_alpha2   from 670 / 1240 nm
        ssa_mix           speciated SSA via fixed per-species values
        asym_mix          speciated asymmetry parameter
    """
    if df.empty:
        return df
    out = df.copy()

    if {"aod_469nm", "aod_865nm"}.issubset(out.columns):
        out["angstrom_alpha1"] = angstrom_alpha(out["aod_469nm"], out["aod_865nm"], 469, 865)
    if {"aod_670nm", "aod_1240nm"}.issubset(out.columns):
        out["angstrom_alpha2"] = angstrom_alpha(out["aod_670nm"], out["aod_1240nm"], 670, 1240)

    species_cols = ["aod_dust_550nm", "aod_bc_550nm", "aod_om_550nm",
                    "aod_ss_550nm", "aod_so4_550nm"]
    if all(c in out.columns for c in species_cols):
        # Tabulated SSA (550 nm) and asymmetry parameters per species
        ssa  = {"dust": 0.92, "bc": 0.20, "om": 0.95, "ss": 0.99, "so4": 0.98}
        asym = {"dust": 0.72, "bc": 0.55, "om": 0.66, "ss": 0.78, "so4": 0.70}
        species_map = {"dust": "aod_dust_550nm", "bc": "aod_bc_550nm",
                       "om": "aod_om_550nm",   "ss": "aod_ss_550nm",
                       "so4": "aod_so4_550nm"}
        total = sum(out[col] for col in species_map.values()).replace(0, np.nan)
        out["ssa_mix"]  = sum(out[col] * ssa[k] for k, col in species_map.items()) / total
        out["asym_mix"] = sum(out[col] * asym[k] for k, col in species_map.items()) / total
    return out
