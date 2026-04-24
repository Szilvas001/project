"""
CAMS historical data downloader — complete atmospheric dataset.

Downloads two complementary datasets from the Copernicus ADS:

1. cams-global-reanalysis-eac4  (3-hourly, 0.75° grid)
   Full atmospheric state: multi-wavelength AOD, ozone, water vapour,
   pressure, cloud cover, PM2.5, PM10, speciated AOD (dust/BC/OM/SS/SO4),
   boundary layer height, surface albedo, temperature.

2. cams-solar-radiation-timeseries  (hourly, point location)
   All-sky and clear-sky GHI/DNI/DHI for model training (observed Kt).

Post-processing computes derived quantities:
   - Ångström exponents ALPHA1 (340–500 nm) and ALPHA2 (500–1064 nm)
   - SSA (single-scattering albedo) from speciated AOD mixing
   - Asymmetry parameter g from speciated AOD
   - Cloud optical depth from total cloud cover

UTC timestamps throughout — CAMS radiation service uses UTC; all timestamps
are stored tz-aware UTC and never silently converted.

Usage
-----
    python scripts/01_download_cams.py --start 2021-01-01 --end 2024-12-31
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from solar_forecast.physics.aerosol import (
    compute_alpha1_alpha2,
    estimate_ssa_g_from_species,
)

logger = logging.getLogger(__name__)

# ── EAC4 single-level variables ──────────────────────────────────────────
# Core: always downloaded
_EAC4_VARS_CORE = [
    "total_aerosol_optical_depth_at_550nm",       # aod550
    "total_column_ozone",                          # gtco3 — DU
    "total_column_water_vapour",                   # tcwv — kg/m²
    "surface_pressure",                            # sp — Pa
    "total_cloud_cover",                           # tcc — fraction
    "2m_temperature",                              # t2m — K
    "boundary_layer_height",                       # blh — m
    "forecast_albedo",                             # fal
]

# Extended: spectral AOD wavelengths (for Ångström exponent)
_EAC4_VARS_SPECTRAL_AOD = [
    "total_aerosol_optical_depth_at_469nm",
    "total_aerosol_optical_depth_at_670nm",
    "total_aerosol_optical_depth_at_865nm",
    "total_aerosol_optical_depth_at_1240nm",
]

# Extended: speciated AOD (for SSA/g estimation)
_EAC4_VARS_SPECIES = [
    "dust_aerosol_optical_depth_at_550nm",
    "black_carbon_aerosol_optical_depth_at_550nm",
    "organic_matter_aerosol_optical_depth_at_550nm",
    "sea_salt_aerosol_optical_depth_at_550nm",
    "sulphate_aerosol_optical_depth_at_550nm",
]

# Optional: PM and additional composition
_EAC4_VARS_OPTIONAL = [
    "particulate_matter_d_less_than_25_um_surface",   # PM2.5 — μg/m³
    "particulate_matter_d_less_than_10_um_surface",   # PM10  — μg/m³
    "total_column_carbon_monoxide",                    # tcco
    "total_column_nitrogen_dioxide",                   # tc_no2
    "snow_albedo",                                     # asn
]

# 3-hourly time steps in EAC4
_TIMES_3H = [f"{h:02d}:00" for h in range(0, 24, 3)]

# Internal column name mapping from raw CDS/xarray names
_RENAME_MAP = {
    # 550 nm total AOD
    "aod550":                                          "aod_550nm",
    "total_aerosol_optical_depth_at_550nm":            "aod_550nm",
    "aod469":                                          "aod_469nm",
    "total_aerosol_optical_depth_at_469nm":            "aod_469nm",
    "aod670":                                          "aod_670nm",
    "total_aerosol_optical_depth_at_670nm":            "aod_670nm",
    "aod865":                                          "aod_865nm",
    "total_aerosol_optical_depth_at_865nm":            "aod_865nm",
    "aod1240":                                         "aod_1240nm",
    "total_aerosol_optical_depth_at_1240nm":           "aod_1240nm",
    # Speciated
    "duaod550":                                        "aod_dust_550nm",
    "dust_aerosol_optical_depth_at_550nm":             "aod_dust_550nm",
    "bcaod550":                                        "aod_bc_550nm",
    "black_carbon_aerosol_optical_depth_at_550nm":     "aod_bc_550nm",
    "omaod550":                                        "aod_om_550nm",
    "organic_matter_aerosol_optical_depth_at_550nm":   "aod_om_550nm",
    "ssaod550":                                        "aod_ss_550nm",
    "sea_salt_aerosol_optical_depth_at_550nm":         "aod_ss_550nm",
    "suaod550":                                        "aod_su_550nm",
    "sulphate_aerosol_optical_depth_at_550nm":         "aod_su_550nm",
    # Atmospheric composition
    "gtco3":                                           "total_ozone",
    "tco3":                                            "total_ozone",
    "total_column_ozone":                              "total_ozone",
    "tcwv":                                            "precipitable_water",
    "total_column_water_vapour":                       "precipitable_water",
    "sp":                                              "surface_pressure",
    "surface_pressure":                                "surface_pressure",
    "tcc":                                             "cloud_cover",
    "total_cloud_cover":                               "cloud_cover",
    "t2m":                                             "temperature_2m",
    "2m_temperature":                                  "temperature_2m",
    "blh":                                             "boundary_layer_height",
    "boundary_layer_height":                           "boundary_layer_height",
    "fal":                                             "forecast_albedo",
    "forecast_albedo":                                 "forecast_albedo",
    "asn":                                             "snow_albedo",
    "snow_albedo":                                     "snow_albedo",
    # PM
    "pm2p5":                                           "pm25",
    "particulate_matter_d_less_than_25_um_surface":    "pm25",
    "pm10":                                            "pm10",
    "particulate_matter_d_less_than_10_um_surface":    "pm10",
    # Gas columns
    "tcco":                                            "total_column_co",
    "total_column_carbon_monoxide":                    "total_column_co",
    "tc_no2":                                          "total_column_no2",
    "total_column_nitrogen_dioxide":                   "total_column_no2",
    "no2":                                             "total_column_no2",
}


class CamsLoader:
    """
    Downloads and processes CAMS EAC4 + CAMS solar radiation.

    Stores results in PostgreSQL via DBManager.  All timestamps are UTC.
    """

    def __init__(self, cfg: dict, db):
        self.cfg = cfg
        self.db = db
        self.lat = cfg["location"]["lat"]
        self.lon = cfg["location"]["lon"]
        self.altitude = cfg["location"].get("altitude", 0)

        cams_cfg = cfg.get("cams", {})
        api_key = cams_cfg.get("api_key") or os.environ.get("CAMS_API_KEY", "")

        try:
            import cdsapi
            self._client = cdsapi.Client(
                url=cams_cfg.get("api_url", "https://ads.atmosphere.copernicus.eu/api/v2"),
                key=api_key,
                quiet=True,
                verify=True,
            )
        except Exception as exc:
            logger.error("Could not initialise cdsapi client: %s", exc)
            self._client = None

        # Whether to attempt optional/extended variable groups
        self._fetch_spectral_aod = cams_cfg.get("fetch_spectral_aod", True)
        self._fetch_species_aod = cams_cfg.get("fetch_species_aod", True)
        self._fetch_optional = cams_cfg.get("fetch_optional", True)

    # ──────────────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────────────

    def run_backfill(self, start: str, end: str) -> dict:
        """
        Download all CAMS data for [start, end] and store to PostgreSQL.

        Processes one calendar month at a time.

        Returns dict with row counts per table.
        """
        if self._client is None:
            raise RuntimeError("cdsapi not configured — set CAMS_API_KEY in .env")

        self.db.create_tables()
        start_d = date.fromisoformat(start)
        end_d   = date.fromisoformat(end)

        months = []
        cur = start_d.replace(day=1)
        while cur <= end_d:
            months.append((cur.year, cur.month))
            cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)

        atmo_total = 0
        rad_total  = 0

        for year, month in tqdm(months, desc="CAMS months"):
            try:
                df_atmo = self._download_eac4_month(year, month)
                n = self.db.upsert_cams_atmo(df_atmo)
                atmo_total += n
                logger.info("EAC4  %d-%02d → %d rows  (cols: %s)",
                            year, month, n, list(df_atmo.columns))
            except Exception as exc:
                logger.error("EAC4  %d-%02d failed: %s", year, month, exc)

            try:
                df_rad = self._download_radiation_month(year, month)
                n = self.db.upsert_cams_radiation(df_rad)
                rad_total += n
                logger.info("Rad   %d-%02d → %d rows", year, month, n)
            except Exception as exc:
                logger.error("Rad   %d-%02d failed: %s", year, month, exc)

        return {"cams_atmo": atmo_total, "cams_radiation": rad_total}

    # ──────────────────────────────────────────────────────────────────────
    # EAC4 atmospheric reanalysis
    # ──────────────────────────────────────────────────────────────────────

    def _download_eac4_month(self, year: int, month: int) -> pd.DataFrame:
        import xarray as xr

        last_day = _last_day_of_month(year, month)
        area = [self.lat + 0.75, self.lon - 0.75,
                self.lat - 0.75, self.lon + 0.75]
        dates = [f"{year}-{month:02d}-{d:02d}" for d in range(1, last_day + 1)]

        # Build the combined variable list
        variables = list(_EAC4_VARS_CORE)
        if self._fetch_spectral_aod:
            variables += _EAC4_VARS_SPECTRAL_AOD
        if self._fetch_species_aod:
            variables += _EAC4_VARS_SPECIES
        if self._fetch_optional:
            variables += _EAC4_VARS_OPTIONAL

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "eac4.nc"
            try:
                self._client.retrieve(
                    "cams-global-reanalysis-eac4",
                    {
                        "variable": variables,
                        "date": dates,
                        "time": _TIMES_3H,
                        "area": area,
                        "format": "netcdf",
                    },
                    str(out),
                )
            except Exception as exc:
                # Retry with core-only if extended set fails
                logger.warning(
                    "EAC4 full variable set failed (%s), retrying core only.", exc
                )
                self._client.retrieve(
                    "cams-global-reanalysis-eac4",
                    {
                        "variable": _EAC4_VARS_CORE,
                        "date": dates,
                        "time": _TIMES_3H,
                        "area": area,
                        "format": "netcdf",
                    },
                    str(out),
                )

            ds = xr.open_dataset(out)
            df = self._xr_to_df(ds)
            ds.close()

        df = self._compute_derived(df)
        return df

    def _xr_to_df(self, ds) -> pd.DataFrame:
        """Extract nearest grid point, rename, and unit-convert."""
        pt = ds.sel(
            latitude=self.lat,
            longitude=self.lon,
            method="nearest",
        )

        cols: dict[str, np.ndarray] = {}
        for var in pt.data_vars:
            key = _RENAME_MAP.get(str(var).lower(), _RENAME_MAP.get(str(var), str(var)))
            vals = pt[var].values.ravel().astype(float)
            cols[key] = vals

        times = pd.to_datetime(pt["time"].values).tz_localize("UTC")
        df = pd.DataFrame(cols, index=times)
        df.index.name = "timestamp"

        # ── Unit conversions ──────────────────────────────────────────────
        if "total_ozone" in df:
            # kg/m² → Dobson units (1 DU = 2.1415e-5 kg/m²)
            df["total_ozone"] = (df["total_ozone"] / 2.1415e-5).clip(100, 600)

        if "precipitable_water" in df:
            # kg/m² (= mm) → cm
            df["precipitable_water"] = (df["precipitable_water"] / 10.0).clip(0, 10)

        if "surface_pressure" in df:
            # Pa → hPa
            df["surface_pressure"] = (df["surface_pressure"] / 100.0).clip(800, 1100)

        if "cloud_cover" in df:
            df["cloud_cover"] = df["cloud_cover"].clip(0.0, 1.0)

        if "temperature_2m" in df:
            # K → °C
            df["temperature_2m"] = (df["temperature_2m"] - 273.15).clip(-80, 60)

        if "boundary_layer_height" in df:
            df["boundary_layer_height"] = df["boundary_layer_height"].clip(10, 5000)

        if "forecast_albedo" in df:
            df["forecast_albedo"] = df["forecast_albedo"].clip(0.0, 1.0)

        if "snow_albedo" in df:
            df["snow_albedo"] = df["snow_albedo"].clip(0.0, 1.0)

        # AOD values: clip to physical range
        aod_cols = [c for c in df.columns if c.startswith("aod")]
        for c in aod_cols:
            df[c] = df[c].clip(0.0, 5.0)

        # PM: kg/m³ → μg/m³
        for pm_col in ["pm25", "pm10"]:
            if pm_col in df:
                df[pm_col] = (df[pm_col] * 1e9).clip(0, 2000)   # kg → μg

        # Gas columns: kg/m² → mol/m² for CO (MW=28), NO2 (MW=46)
        if "total_column_co" in df:
            df["total_column_co"] = (df["total_column_co"] / 28e-3).clip(0, 1e4)
        if "total_column_no2" in df:
            df["total_column_no2"] = (df["total_column_no2"] / 46e-3).clip(0, 1e4)

        df["lat"] = self.lat
        df["lon"] = self.lon

        return df

    def _compute_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived atmospheric quantities:
          - ALPHA1, ALPHA2 Ångström exponents
          - TAU550 best estimate
          - SSA, GG (single-scattering albedo, asymmetry parameter)
          - Cloud optical depth from cloud cover
        """
        # ── Ångström exponents ────────────────────────────────────────────
        alpha1, alpha2, tau550 = compute_alpha1_alpha2(
            tau340=None,
            tau500=None,
            tau550=df.get("aod_550nm"),
            tau670=df.get("aod_670nm"),
            tau865=df.get("aod_865nm"),
            tau1020=df.get("aod_1240nm"),   # closest available to 1064
        )
        df["angstrom_alpha1"] = alpha1   # 340–500 nm (SMARTS ALPHA1)
        df["angstrom_alpha2"] = alpha2   # 500–1064 nm (SMARTS ALPHA2)
        df["aod_550nm_best"]  = tau550   # best estimate of AOD at 550 nm
        # Back-compat alias
        df["aod_550nm"]     = df.get("aod_550nm", tau550)
        df["angstrom_exponent"] = alpha2  # single-α for spectrl2

        # ── SSA and asymmetry parameter ───────────────────────────────────
        has_species = all(c in df.columns for c in
                          ["aod_dust_550nm", "aod_bc_550nm", "aod_om_550nm",
                           "aod_ss_550nm", "aod_su_550nm"])
        if has_species:
            ssa_vals = []
            g_vals   = []
            for _, row in df.iterrows():
                ssa, g = estimate_ssa_g_from_species(
                    aod_dust=row.get("aod_dust_550nm", 0),
                    aod_bc=row.get("aod_bc_550nm", 0),
                    aod_oc=row.get("aod_om_550nm", 0),
                    aod_sea_salt=row.get("aod_ss_550nm", 0),
                    aod_sulphate=row.get("aod_su_550nm", 0),
                )
                ssa_vals.append(ssa)
                g_vals.append(g)
            df["ssa_550nm"] = ssa_vals
            df["asymmetry_factor"] = g_vals
        else:
            df["ssa_550nm"] = 0.92        # continental background
            df["asymmetry_factor"] = 0.65

        # ── Cloud optical depth from cover ────────────────────────────────
        fc = df["cloud_cover"].clip(1e-4, 0.9999) if "cloud_cover" in df \
             else pd.Series(0.0, index=df.index)
        df["cloud_optical_depth"] = (-np.log(1.0 - fc) * 14.0).clip(0, 200)

        return df

    # ──────────────────────────────────────────────────────────────────────
    # CAMS solar radiation service
    # ──────────────────────────────────────────────────────────────────────

    def _download_radiation_month(self, year: int, month: int) -> pd.DataFrame:
        """Download hourly all-sky + clear-sky solar radiation for one month."""
        last_day = _last_day_of_month(year, month)
        start_str = f"{year}-{month:02d}-01"
        end_str   = f"{year}-{month:02d}-{last_day:02d}"

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rad.csv"
            self._client.retrieve(
                "cams-solar-radiation-timeseries",
                {
                    "sky_type": ["observed_cloud", "clear_sky"],
                    "location": {"latitude": self.lat, "longitude": self.lon},
                    "altitude": str(self.altitude),
                    "date": f"{start_str}/{end_str}",
                    "time_step": "1hour",
                    "time_reference": "universal_time",   # UTC — always
                    "format": "csv",
                },
                str(out),
            )
            df = _parse_radiation_csv(out, self.lat, self.lon)

        return df

    # ──────────────────────────────────────────────────────────────────────
    # Convenience: load from DB for training
    # ──────────────────────────────────────────────────────────────────────

    def load_training_data(
        self,
        start: str,
        end: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load CAMS atmospheric data and radiation from DB for a date range.

        Returns (df_atmo, df_radiation) both indexed by UTC timestamp.
        """
        from datetime import datetime, timezone
        s = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        e = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
        df_atmo = self.db.load_cams_atmo(self.lat, self.lon, s, e)
        df_rad  = self.db.load_cams_radiation(self.lat, self.lon, s, e)
        return df_atmo, df_rad


# ── Module-level helpers ──────────────────────────────────────────────────

def _last_day_of_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - timedelta(days=1)).day


def _parse_radiation_csv(path: Path, lat: float, lon: float) -> pd.DataFrame:
    """
    Parse the CAMS solar radiation service CSV.

    The file has a metadata/comment header block before the data rows.
    Timestamps are in UTC ("Observation period" column).
    """
    with open(path) as f:
        lines = f.readlines()

    # Find the first non-comment, non-blank header line
    header_idx = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if not line.startswith("#") and line:
            header_idx = i
            break

    df = pd.read_csv(
        path,
        skiprows=header_idx,
        sep=";",
        engine="python",
    )
    df.columns = [c.strip() for c in df.columns]

    # Flexible column name matching
    col_map = {
        # All-sky
        "Observation period":  "timestamp",
        "GHI":       "ghi",
        "DHI":       "dhi",
        "BNI":       "dni",
        "BHI":       "bhi",            # beam horizontal
        "GTI":       "gti",            # global tilted (if present)
        # Clear-sky
        "Clear sky GHI":  "ghi_clear",
        "Clear sky DHI":  "dhi_clear",
        "Clear sky BNI":  "dni_clear",
        "Clear sky BHI":  "bhi_clear",
        # Reliability
        "Reliability":    "reliability",
        "sza":            "sza_cams",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "timestamp" not in df.columns:
        raise ValueError("CAMS radiation CSV: 'Observation period' column not found.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # Columns we always want — fill missing with NaN
    irr_cols = ["ghi", "dhi", "dni", "ghi_clear", "dhi_clear", "dni_clear"]
    for c in irr_cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[irr_cols + [c for c in df.columns if c not in irr_cols]]

    # Replace CAMS sentinel values (−999, −9999) with NaN; clip to ≥ 0
    df = df.where(df > -900, other=float("nan"))
    df = df.clip(lower=0)

    df["lat"] = lat
    df["lon"] = lon

    return df


def merge_eac4_with_radiation(
    df_atmo: pd.DataFrame,
    df_radiation: pd.DataFrame,
    resample_freq: str = "1h",
) -> pd.DataFrame:
    """
    Merge 3-hourly EAC4 atmospheric data with 1-hourly radiation.

    EAC4 is linearly interpolated to the hourly radiation timestamps.

    Returns
    -------
    Merged DataFrame indexed by UTC timestamps.
    """
    if df_atmo.empty or df_radiation.empty:
        return pd.DataFrame()

    # Resample EAC4 (3h) → 1h via linear interpolation
    idx_1h = df_radiation.index
    atmo_1h = (
        df_atmo
        .reindex(df_atmo.index.union(idx_1h))
        .interpolate("linear")
        .reindex(idx_1h)
    )

    merged = pd.concat([df_radiation, atmo_1h], axis=1)
    merged = merged.loc[~merged.index.duplicated(keep="first")]
    return merged
