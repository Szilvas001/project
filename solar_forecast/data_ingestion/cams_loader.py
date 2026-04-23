"""
CAMS historical data downloader.

Downloads two complementary datasets from the Copernicus ADS:

1. cams-global-reanalysis-eac4
   3-hourly global reanalysis (0.75°): AOD, ozone, water vapour, pressure,
   cloud cover.  Used as feature inputs to the Kt model.

2. cams-solar-radiation-timeseries
   Hourly all-sky and clear-sky solar radiation at a given point.
   Used to compute the observed Kt = GHI / GHI_clear for training.

Usage:
    python scripts/01_download_cams.py --start 2021-01-01 --end 2024-12-31
"""

import logging
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# EAC4 variable names on ADS
_EAC4_VARS = [
    "aerosol_optical_depth_550nm",
    "total_column_ozone",
    "total_column_water_vapour",
    "surface_pressure",
    "fraction_of_cloud_cover",
]

# 3-hourly time steps in EAC4
_TIMES_3H = [f"{h:02d}:00" for h in range(0, 24, 3)]


class CamsLoader:
    """
    Downloads CAMS EAC4 atmospheric reanalysis and CAMS solar radiation
    for a fixed lat/lon and stores both in PostgreSQL.
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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_backfill(self, start: str, end: str) -> dict:
        """
        Download all available CAMS data for [start, end] and store to DB.

        Processes one calendar month at a time to keep request sizes
        manageable (ADS has a queue system).

        Returns dict with row counts inserted per table.
        """
        if self._client is None:
            raise RuntimeError("cdsapi not configured — set CAMS_API_KEY in .env")

        self.db.create_tables()
        start_d = date.fromisoformat(start)
        end_d = date.fromisoformat(end)

        months = []
        cur = start_d.replace(day=1)
        while cur <= end_d:
            months.append((cur.year, cur.month))
            cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)

        atmo_total = 0
        rad_total = 0

        for year, month in tqdm(months, desc="CAMS months"):
            try:
                # --- atmospheric reanalysis ---
                df_atmo = self._download_eac4_month(year, month)
                n = self.db.upsert_cams_atmo(df_atmo)
                atmo_total += n
                logger.info("EAC4  %d-%02d → %d rows", year, month, n)
            except Exception as exc:
                logger.error("EAC4  %d-%02d failed: %s", year, month, exc)

            try:
                # --- solar radiation timeseries ---
                df_rad = self._download_radiation_month(year, month)
                n = self.db.upsert_cams_radiation(df_rad)
                rad_total += n
                logger.info("Rad   %d-%02d → %d rows", year, month, n)
            except Exception as exc:
                logger.error("Rad   %d-%02d failed: %s", year, month, exc)

        return {"cams_atmo": atmo_total, "cams_radiation": rad_total}

    # ------------------------------------------------------------------
    # EAC4 atmospheric reanalysis
    # ------------------------------------------------------------------

    def _download_eac4_month(self, year: int, month: int) -> pd.DataFrame:
        import xarray as xr

        last_day = (date(year, month % 12 + 1, 1) - timedelta(days=1)).day \
                   if month < 12 else 31
        area = [self.lat + 0.75, self.lon - 0.75,
                self.lat - 0.75, self.lon + 0.75]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "eac4.nc"
            self._client.retrieve(
                "cams-global-reanalysis-eac4",
                {
                    "variable": _EAC4_VARS,
                    "date": [f"{year}-{month:02d}-{d:02d}"
                             for d in range(1, last_day + 1)],
                    "time": _TIMES_3H,
                    "area": area,
                    "format": "netcdf",
                },
                str(out),
            )
            ds = xr.open_dataset(out)
            df = self._xr_to_df(ds)
            ds.close()

        return df

    def _xr_to_df(self, ds) -> pd.DataFrame:
        """Extract nearest grid point and convert units."""
        pt = ds.sel(latitude=self.lat, longitude=self.lon, method="nearest")

        rename = {
            "aerosol_optical_depth_550nm": "aod_550nm",
            "aod550":                      "aod_550nm",
            "total_column_ozone":          "total_ozone",
            "tco3":                        "total_ozone",
            "total_column_water_vapour":   "precipitable_water",
            "tcwv":                        "precipitable_water",
            "surface_pressure":            "surface_pressure",
            "sp":                          "surface_pressure",
            "fraction_of_cloud_cover":     "cloud_cover",
            "cc":                          "cloud_cover",
        }

        cols = {}
        for var in pt.data_vars:
            key = rename.get(str(var), str(var))
            vals = pt[var].values.ravel()
            cols[key] = vals

        times = pd.to_datetime(pt["time"].values).tz_localize("UTC")
        df = pd.DataFrame(cols, index=times)
        df.index.name = "timestamp"

        # Unit conversions
        if "total_ozone" in df:
            # kg/m² → Dobson units (1 DU = 2.1415e-5 kg/m²)
            df["total_ozone"] = (df["total_ozone"] / 2.1415e-5).clip(100, 600)
        if "precipitable_water" in df:
            # kg/m² (= mm water) → cm
            df["precipitable_water"] = (df["precipitable_water"] / 10.0).clip(0, 10)
        if "surface_pressure" in df:
            # Pa → hPa
            df["surface_pressure"] = (df["surface_pressure"] / 100.0).clip(800, 1100)
        if "cloud_cover" in df:
            df["cloud_cover"] = df["cloud_cover"].clip(0.0, 1.0)
        if "aod_550nm" in df:
            df["aod_550nm"] = df["aod_550nm"].clip(0.0, 5.0)

        # Angstrom exponent: climatological default, can be refined with multi-wavelength AOD
        df["angstrom_exponent"] = 1.30

        # Cloud optical depth from cover fraction (Stephens 1978 empirical)
        fc = df["cloud_cover"].clip(1e-4, 0.9999) if "cloud_cover" in df \
             else pd.Series(0.0, index=df.index)
        df["cloud_optical_depth"] = (-np.log(1 - fc) * 14.0).clip(0, 100)

        df["lat"] = self.lat
        df["lon"] = self.lon

        return df

    # ------------------------------------------------------------------
    # CAMS solar radiation timeseries
    # ------------------------------------------------------------------

    def _download_radiation_month(self, year: int, month: int) -> pd.DataFrame:
        """
        Download hourly all-sky + clear-sky solar radiation from CAMS
        radiation service for one month.
        """
        last_day = (date(year, month % 12 + 1, 1) - timedelta(days=1)).day \
                   if month < 12 else 31
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
                    "time_reference": "universal_time",
                    "format": "csv",
                },
                str(out),
            )
            df = self._parse_radiation_csv(out)

        return df

    def _parse_radiation_csv(self, path: Path) -> pd.DataFrame:
        """Parse CAMS radiation service CSV output."""
        # The CSV has a comment block before data; find the header line
        with open(path) as f:
            lines = f.readlines()

        header_idx = next(i for i, l in enumerate(lines) if l.startswith("#") is False)
        df = pd.read_csv(path, skiprows=header_idx, sep=";",
                         parse_dates=["Observation period"])

        df = df.rename(columns={
            "Observation period": "timestamp",
            "GHI": "ghi",
            "DHI": "dhi",
            "BNI": "dni",
            "Clear sky GHI": "ghi_clear",
            "Clear sky DHI": "dhi_clear",
            "Clear sky BNI": "dni_clear",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        df = df[["ghi", "dhi", "dni", "ghi_clear", "dhi_clear", "dni_clear"]].copy()

        # Replace sentinel values (-999) with NaN
        df = df.where(df > -900, other=float("nan"))
        df = df.clip(lower=0)

        df["lat"] = self.lat
        df["lon"] = self.lon

        return df
