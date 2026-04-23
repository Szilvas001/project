"""
Open-Meteo live and historical weather fetcher.

Provides hourly weather data needed for real-time forecasting:
  cloud_cover, relative_humidity, temperature, surface_pressure,
  shortwave_radiation, direct_radiation, diffuse_radiation.

Note on AOD: Open-Meteo does not supply aerosol optical depth.
We substitute with a monthly climatological AOD profile derived
from a multi-year MERRA-2/CAMS median, scaled by the atmospheric
turbidity implied by the available relative humidity and season.
Users with an active CAMS account can optionally enable the NRT
aerosol endpoint to replace this estimate.

API docs: https://open-meteo.com/en/docs
Historical: https://archive-api.open-meteo.com/v1/archive
"""

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import requests_cache
from retry_requests import retry

logger = logging.getLogger(__name__)

# Hourly variables to request
_HOURLY_VARS = [
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "relative_humidity_2m",
    "temperature_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
]

# Monthly AOD climatology for mid-latitude Europe (MERRA-2 derived, 550 nm)
# Index 0 = January, 11 = December
_AOD_CLIM_EU = np.array([
    0.08, 0.09, 0.11, 0.14, 0.16, 0.17,
    0.16, 0.15, 0.14, 0.12, 0.09, 0.08,
])

# Angstrom exponent climatology (dimensionless)
_ALPHA_CLIM_EU = np.array([
    1.35, 1.30, 1.28, 1.25, 1.20, 1.15,
    1.18, 1.20, 1.25, 1.30, 1.35, 1.38,
])


def _make_session(cache_expire_hours: int = 1):
    session = requests_cache.CachedSession(
        ".openmeteo_cache",
        expire_after=timedelta(hours=cache_expire_hours),
    )
    return retry(session, retries=4, backoff_factor=0.5)


class OpenMeteoClient:
    """Fetches hourly weather for any lat/lon, forecast or historical."""

    def __init__(self, cfg: dict):
        om = cfg.get("openmeteo", {})
        self._forecast_url = om.get("forecast_url",
                                    "https://api.open-meteo.com/v1/forecast")
        self._historical_url = om.get("historical_url",
                                      "https://archive-api.open-meteo.com/v1/archive")
        self._session = _make_session(om.get("cache_expire_hours", 1))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_forecast(self, lat: float, lon: float, days: int = 7) -> pd.DataFrame:
        """Fetch hourly forecast for the next `days` days."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(_HOURLY_VARS),
            "forecast_days": days,
            "timezone": "UTC",
        }
        raw = self._session.get(self._forecast_url, params=params, timeout=30)
        raw.raise_for_status()
        df = self._parse_response(raw.json())
        df = self._enrich_aod(df, lat)
        return df

    def get_historical(self, lat: float, lon: float,
                       start: str, end: str) -> pd.DataFrame:
        """Fetch hourly historical data for the given date range."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(_HOURLY_VARS),
            "start_date": start,
            "end_date": end,
            "timezone": "UTC",
        }
        raw = self._session.get(self._historical_url, params=params, timeout=60)
        raw.raise_for_status()
        df = self._parse_response(raw.json())
        df = self._enrich_aod(df, lat)
        return df

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_response(self, data: dict) -> pd.DataFrame:
        hourly = data.get("hourly", {})
        times = pd.to_datetime(hourly.pop("time"), utc=True)
        df = pd.DataFrame(hourly, index=times)
        df.index.name = "timestamp"

        # Rename to internal names
        df = df.rename(columns={
            "shortwave_radiation": "ghi_measured",
            "direct_radiation":    "dni_horiz",
            "diffuse_radiation":   "dhi_measured",
            "temperature_2m":      "temperature",
            "relative_humidity_2m": "relative_humidity",
            "wind_speed_10m":      "wind_speed",
        })

        # Composite cloud cover from multi-level (weighted by typical transmittance impact)
        if "cloud_cover_low" in df and "cloud_cover_mid" in df and "cloud_cover_high" in df:
            df["cloud_cover_composite"] = (
                0.5 * df["cloud_cover_low"].fillna(0)
                + 0.3 * df["cloud_cover_mid"].fillna(0)
                + 0.2 * df["cloud_cover_high"].fillna(0)
            ).clip(0, 100) / 100.0

        # Convert cloud_cover from % to fraction
        if "cloud_cover" in df:
            df["cloud_cover"] = df["cloud_cover"].clip(0, 100) / 100.0

        # Cloud optical depth from cover fraction (used when CAMS not available)
        fc = df["cloud_cover"].clip(1e-4, 0.9999)
        df["cloud_optical_depth"] = (-np.log(1 - fc) * 14.0).clip(0, 100)

        return df

    def _enrich_aod(self, df: pd.DataFrame, lat: float) -> pd.DataFrame:
        """
        Inject climatological AOD and Angstrom exponent when no NRT source
        is available.  A small humidity-based scaling is applied: higher
        relative humidity swells aerosol particles, increasing AOD.

        Scale: AOD_eff = AOD_clim × (1 + k_rh × (RH - 0.5))
        where k_rh ≈ 0.6 for typical continental aerosols (Hänel 1976).
        """
        months = df.index.month.values - 1  # 0-based
        aod_base = _AOD_CLIM_EU[months]
        alpha_base = _ALPHA_CLIM_EU[months]

        # Humidity correction
        rh = df.get("relative_humidity", pd.Series(50.0, index=df.index)) / 100.0
        k_rh = 0.6
        rh_factor = (1.0 + k_rh * (rh.values - 0.5)).clip(0.7, 2.5)

        df["aod_550nm"] = (aod_base * rh_factor).clip(0.01, 3.0)
        df["angstrom_exponent"] = alpha_base
        df["precipitable_water"] = self._estimate_pw(df)

        return df

    @staticmethod
    def _estimate_pw(df: pd.DataFrame) -> pd.Series:
        """
        Estimate precipitable water (cm) from surface temperature and
        relative humidity using the Leckner / Prata approximation:

            e_s = 6.1078 × exp(17.27 × T / (T + 237.3))  [hPa]
            e   = RH × e_s
            W   ≈ 0.493 × e / (T + 273.15)               [cm]
        """
        T = df.get("temperature", pd.Series(15.0, index=df.index))
        rh = df.get("relative_humidity", pd.Series(60.0, index=df.index)) / 100.0

        e_s = 6.1078 * np.exp(17.27 * T / (T + 237.3))
        e = (rh * e_s).clip(0.01)
        W = 0.493 * e / (T + 273.15)
        return W.clip(0.1, 10.0)
