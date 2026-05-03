"""
CAMS variable mapping — canonical internal names → CAMS API names → units.

Rules
-----
* All internal names are stable snake_case identifiers.
* Source CAMS names change occasionally; only this file needs updating.
* Missing variables are stored as NULL, never crash ingestion.
* Every missing variable emits a WARNING log at parse time.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ── Canonical variable registry ───────────────────────────────────────────
#
# Each entry:
#   internal_name  → (cams_short_name, cams_long_name, unit, default)
#
# default is the climatological fallback when CAMS data is unavailable.

CAMS_VARIABLES: dict[str, dict[str, Any]] = {
    # ── AOD (Aerosol Optical Depth) ──────────────────────────────────────
    "aod_550": {
        "cams_short": "aod550",
        "cams_long":  "total_aerosol_optical_depth_550nm",
        "unit":       "dimensionless",
        "default":    0.12,
    },
    "aod_469": {
        "cams_short": "aod469",
        "cams_long":  "total_aerosol_optical_depth_469nm",
        "unit":       "dimensionless",
        "default":    0.16,
    },
    "aod_670": {
        "cams_short": "aod670",
        "cams_long":  "total_aerosol_optical_depth_670nm",
        "unit":       "dimensionless",
        "default":    0.09,
    },
    "aod_865": {
        "cams_short": "aod865",
        "cams_long":  "total_aerosol_optical_depth_865nm",
        "unit":       "dimensionless",
        "default":    0.06,
    },
    # ── Column gases ─────────────────────────────────────────────────────
    "total_column_water_vapour": {
        "cams_short": "tcwv",
        "cams_long":  "total_column_water_vapour",
        "unit":       "kg m-2",
        "default":    15.0,   # ~1.5 cm precipitable water
    },
    "total_column_ozone": {
        "cams_short": "gtco3",
        "cams_long":  "total_column_ozone",
        "unit":       "kg m-2",    # convert to DU in pipeline: 1 DU ≈ 2.14e-5 kg/m²
        "default":    0.006642,    # ~310 DU
    },
    # ── Particulate matter ────────────────────────────────────────────────
    "pm25": {
        "cams_short": "pm2p5",
        "cams_long":  "particulate_matter_d_less_than_2_5_um_surface",
        "unit":       "kg m-3",
        "default":    None,
    },
    "pm10": {
        "cams_short": "pm10",
        "cams_long":  "particulate_matter_d_less_than_10_um_surface",
        "unit":       "kg m-3",
        "default":    None,
    },
    # ── Speciated aerosol AOD (for SSA/asymmetry mixing) ─────────────────
    "black_carbon_aod_550": {
        "cams_short": "bcaod550",
        "cams_long":  "black_carbon_aerosol_optical_depth_550nm",
        "unit":       "dimensionless",
        "default":    None,
    },
    "dust_aod_550": {
        "cams_short": "duaod550",
        "cams_long":  "dust_aerosol_optical_depth_550nm",
        "unit":       "dimensionless",
        "default":    None,
    },
    "organic_matter_aod_550": {
        "cams_short": "omaod550",
        "cams_long":  "organic_matter_aerosol_optical_depth_550nm",
        "unit":       "dimensionless",
        "default":    None,
    },
    "sea_salt_aod_550": {
        "cams_short": "ssaod550",
        "cams_long":  "sea_salt_aerosol_optical_depth_550nm",
        "unit":       "dimensionless",
        "default":    None,
    },
    "sulphate_aod_550": {
        "cams_short": "suaod550",
        "cams_long":  "sulphate_aerosol_optical_depth_550nm",
        "unit":       "dimensionless",
        "default":    None,
    },
    "nitrate_aod_550": {
        "cams_short": "nitaod550",
        "cams_long":  "nitrate_aerosol_optical_depth_550nm",
        "unit":       "dimensionless",
        "default":    None,
    },
    "ammonium_aod_550": {
        "cams_short": "nhaod550",
        "cams_long":  "ammonium_aerosol_optical_depth_550nm",
        "unit":       "dimensionless",
        "default":    None,
    },
    # ── Boundary layer / surface ──────────────────────────────────────────
    "boundary_layer_height": {
        "cams_short": "blh",
        "cams_long":  "boundary_layer_height",
        "unit":       "m",
        "default":    1000.0,
    },
    "temperature_2m": {
        "cams_short": "2t",
        "cams_long":  "2m_temperature",
        "unit":       "K",
        "default":    None,
    },
    "surface_pressure": {
        "cams_short": "sp",
        "cams_long":  "surface_pressure",
        "unit":       "Pa",
        "default":    101325.0,
    },
}

# Long names requested from CAMS API (for surface_composition dataset)
CAMS_LONG_NAMES: list[str] = [v["cams_long"] for v in CAMS_VARIABLES.values()]

# Reverse map: CAMS long name → internal name
_LONG_TO_INTERNAL: dict[str, str] = {
    v["cams_long"]: k for k, v in CAMS_VARIABLES.items()
}
# Reverse map: CAMS short name → internal name
_SHORT_TO_INTERNAL: dict[str, str] = {
    v["cams_short"]: k for k, v in CAMS_VARIABLES.items()
}


def map_row(raw: dict[str, Any]) -> dict[str, Any]:
    """Map a raw CAMS record (using short or long names) to internal names.

    Missing variables are stored as None (NULL in DB).  A WARNING is emitted
    for each expected variable that is absent from ``raw``.

    Parameters
    ----------
    raw : dict  keyed by CAMS short or long variable names

    Returns
    -------
    dict  keyed by internal snake_case names
    """
    out: dict[str, Any] = {}
    for internal, spec in CAMS_VARIABLES.items():
        # Try long name first, then short name
        value = raw.get(spec["cams_long"], raw.get(spec["cams_short"]))
        if value is None:
            default = spec["default"]
            if default is not None:
                log.debug("CAMS var %s missing → using default %.4g", internal, default)
                value = default
            else:
                log.warning("CAMS var %s missing → stored as NULL", internal)
        elif np.isnan(float(value)) if isinstance(value, (int, float)) else False:
            log.warning("CAMS var %s is NaN → stored as NULL", internal)
            value = None
        out[internal] = value
    return out


def get_climatology_defaults() -> dict[str, float]:
    """Return continental-Europe climatology defaults for all CAMS variables."""
    return {k: v["default"] for k, v in CAMS_VARIABLES.items() if v["default"] is not None}
