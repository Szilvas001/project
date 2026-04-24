"""
Open-Meteo live and historical weather fetcher — complete variable set.

Fetches all variables relevant for PV production forecasting:
  - Irradiance: shortwave (GHI), direct normal (DNI), diffuse (DHI),
                global tilted (POA if tilt/azimuth provided)
  - Cloud cover: total, low, mid, high
  - Temperature, humidity, pressure, wind
  - Visibility (for fog/dust detection)
  - CAPE (convective instability indicator)

NOTE ON TIMESTAMPS (CRITICAL):
  Open-Meteo always returns timestamps in the timezone specified in the
  request.  This module requests "timezone=UTC" so all returned timestamps
  are UTC-aware.  The caller is responsible for converting to local display
  time.  Never silently convert away from UTC in this module.

NOTE ON AOD:
  Open-Meteo does not provide aerosol optical depth.  We substitute a
  monthly climatological AOD profile (MERRA-2/CAMS) scaled by relative
  humidity using the Hänel hygroscopic factor (γ=0.40, RH_ref=0.50).

API docs: https://open-meteo.com/en/docs
Historical: https://archive-api.open-meteo.com/v1/archive
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import requests_cache
from retry_requests import retry

from solar_forecast.physics.aerosol import hanel_corrected_aod

logger = logging.getLogger(__name__)

# ── Hourly variables to request from forecast endpoint ───────────────────
_HOURLY_FORECAST = [
    # Irradiance
    "shortwave_radiation",           # GHI W/m²
    "direct_radiation",              # DNI × cos(sza) — beam horizontal
    "diffuse_radiation",             # DHI W/m²
    "direct_normal_irradiance",      # DNI W/m²
    "terrestrial_radiation",         # downwelling thermal IR
    # Cloud cover
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    # Meteorology
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "precipitation",
    "precipitation_probability",
    "wind_speed_10m",
    "wind_direction_10m",
    "visibility",
    "cape",                          # convective available potential energy
    "vapour_pressure_deficit",
]

# Historical endpoint supports a subset
_HOURLY_HISTORICAL = [
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
]

# ── Monthly AOD climatology (MERRA-2/CAMS 550 nm, EU mid-latitudes) ───────
# Jan..Dec  (0-based)
_AOD_CLIM_EU = np.array([
    0.08, 0.09, 0.11, 0.14, 0.16, 0.17,
    0.16, 0.15, 0.14, 0.12, 0.09, 0.08,
])

# Ångström exponent climatology
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
    """
    Fetches hourly weather for any lat/lon, forecast or historical.

    All returned timestamps are UTC-aware (tz='UTC').
    """

    def __init__(self, cfg: dict):
        om = cfg.get("openmeteo", {})
        self._forecast_url   = om.get("forecast_url",
                                      "https://api.open-meteo.com/v1/forecast")
        self._historical_url = om.get("historical_url",
                                      "https://archive-api.open-meteo.com/v1/archive")
        self._elev_url       = "https://api.open-meteo.com/v1/elevation"
        self._geo_url        = "https://geocoding-api.open-meteo.com/v1/search"
        self._session = _make_session(om.get("cache_expire_hours", 1))
        self._cfg = cfg

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def get_forecast(
        self,
        lat: float,
        lon: float,
        days: int = 7,
        tilt: float = 0.0,
        azimuth: float = 180.0,
    ) -> pd.DataFrame:
        """
        Fetch hourly forecast for the next `days` days.

        Parameters
        ----------
        lat, lon  : Location (decimal degrees)
        days      : Forecast horizon (1–16)
        tilt      : Panel tilt for POA calculation (°)
        azimuth   : Panel azimuth (° clockwise from north, 180=south)

        Returns
        -------
        DataFrame indexed by UTC timestamps.
        """
        vars_list = list(_HOURLY_FORECAST)
        if tilt > 0:
            vars_list.append(f"global_tilted_irradiance")

        params: dict = {
            "latitude":      lat,
            "longitude":     lon,
            "hourly":        ",".join(vars_list),
            "forecast_days": int(days),
            "timezone":      "UTC",            # always UTC
        }
        if tilt > 0:
            params["tilt"]    = tilt
            params["azimuth"] = azimuth - 180  # Open-Meteo: -180..180, 0=south

        raw = self._session.get(self._forecast_url, params=params, timeout=30)
        raw.raise_for_status()
        df = self._parse_response(raw.json())
        df = self._enrich_aod(df)
        df = self._enrich_precipitable_water(df)
        return df

    def get_historical(
        self,
        lat: float,
        lon: float,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Fetch hourly historical data.

        Parameters
        ----------
        start, end : ISO date strings (YYYY-MM-DD)

        Returns
        -------
        DataFrame indexed by UTC timestamps.
        """
        params = {
            "latitude":   lat,
            "longitude":  lon,
            "hourly":     ",".join(_HOURLY_HISTORICAL),
            "start_date": start,
            "end_date":   end,
            "timezone":   "UTC",
        }
        raw = self._session.get(self._historical_url, params=params, timeout=60)
        raw.raise_for_status()
        df = self._parse_response(raw.json())
        df = self._enrich_aod(df)
        df = self._enrich_precipitable_water(df)
        return df

    def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        """Query Open-Meteo elevation API for ground altitude (m)."""
        try:
            resp = self._session.get(
                self._elev_url,
                params={"latitude": lat, "longitude": lon},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return float(data["elevation"][0])
        except Exception as exc:
            logger.warning("Elevation API failed: %s", exc)
            return None

    def geocode(self, city: str) -> tuple[float, float, str, Optional[float]]:
        """
        Convert city name → (lat, lon, display_name, elevation_m).

        Returns elevation=None if the elevation API fails.
        """
        resp = self._session.get(
            self._geo_url,
            params={"name": city, "count": 1, "language": "en"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            raise ValueError(f"City not found: {city!r}")
        r = results[0]
        lat = float(r["latitude"])
        lon = float(r["longitude"])
        name = r.get("name", city)
        elev = r.get("elevation") or self.get_elevation(lat, lon)
        return lat, lon, name, elev

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _parse_response(self, data: dict) -> pd.DataFrame:
        hourly = dict(data.get("hourly", {}))
        times = pd.to_datetime(hourly.pop("time"), utc=True)
        df = pd.DataFrame(hourly, index=times)
        df.index.name = "timestamp"

        # Rename to internal standard names
        rename = {
            "shortwave_radiation":          "ghi_om",        # Open-Meteo GHI
            "direct_radiation":             "bhi_om",        # beam horizontal
            "diffuse_radiation":            "dhi_om",
            "direct_normal_irradiance":     "dni_om",
            "terrestrial_radiation":        "lw_down",
            "global_tilted_irradiance":     "poa_om",
            "temperature_2m":               "temperature",
            "relative_humidity_2m":         "relative_humidity",
            "wind_speed_10m":               "wind_speed",
            "wind_direction_10m":           "wind_direction",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        # Use best available GHI source (prefer measured, use composite)
        if "ghi_om" in df:
            df["ghi_measured"] = df["ghi_om"].clip(lower=0)
        if "dhi_om" in df:
            df["dhi_measured"] = df["dhi_om"].clip(lower=0)
        if "dni_om" in df:
            df["dni_measured"] = df["dni_om"].clip(lower=0)

        # Composite cloud cover (3-level weighted)
        if all(c in df for c in ["cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"]):
            df["cloud_cover_composite"] = (
                0.50 * df["cloud_cover_low"].fillna(0)
                + 0.30 * df["cloud_cover_mid"].fillna(0)
                + 0.20 * df["cloud_cover_high"].fillna(0)
            ).clip(0, 100) / 100.0

        # Normalise cloud_cover to fraction
        if "cloud_cover" in df:
            df["cloud_cover"] = df["cloud_cover"].clip(0, 100) / 100.0

        # Cloud optical depth from cover fraction
        fc = df.get("cloud_cover", pd.Series(0.0, index=df.index))
        fc_safe = fc.clip(1e-4, 0.9999)
        df["cloud_optical_depth"] = (-np.log(1.0 - fc_safe) * 14.0).clip(0, 200)

        return df

    def _enrich_aod(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inject climatological AOD with Hänel humidity correction.

        AOD_wet = AOD_clim(month) × f_Hänel(RH)
        where f_Hänel uses γ=0.40 (continental default, Hänel 1976).
        """
        months = df.index.month.values - 1   # 0-based index
        aod_base  = _AOD_CLIM_EU[months]
        alpha_base = _ALPHA_CLIM_EU[months]

        rh = df.get("relative_humidity", pd.Series(50.0, index=df.index))
        # Convert % to fraction if needed
        rh_frac = np.where(rh.values > 1.5, rh.values / 100.0, rh.values)
        rh_frac = np.clip(rh_frac, 0.01, 0.98)

        aod_wet = hanel_corrected_aod(aod_base, rh_frac, aerosol_type="continental")

        df["aod_550nm"]          = pd.Series(np.clip(aod_wet, 0.01, 3.0), index=df.index)
        df["angstrom_exponent"]  = pd.Series(alpha_base, index=df.index)
        df["angstrom_alpha1"]    = df["angstrom_exponent"]
        df["angstrom_alpha2"]    = df["angstrom_exponent"]
        df["ssa_550nm"]          = 0.92
        df["asymmetry_factor"]   = 0.65

        return df

    def _enrich_precipitable_water(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate precipitable water (cm) from surface T and RH.

        Uses Leckner / Gueymard approximation:
            e_s = 6.1078 × exp(17.27 T / (T + 237.3))   [hPa]
            e   = RH × e_s
            W   ≈ 0.493 × e / (T + 273.15)               [cm]
        """
        T  = df.get("temperature",       pd.Series(15.0, index=df.index))
        rh = df.get("relative_humidity", pd.Series(60.0, index=df.index))
        rh_frac = np.where(rh.values > 1.5, rh.values / 100.0, rh.values)

        e_s = 6.1078 * np.exp(17.27 * T.values / (T.values + 237.3))
        e   = np.clip(rh_frac * e_s, 0.01, None)
        W   = 0.493 * e / (T.values + 273.15)
        df["precipitable_water"] = pd.Series(np.clip(W, 0.1, 10.0), index=df.index)
        return df


# ── Module-level convenience ──────────────────────────────────────────────

def local_timezone_from_lon(lon: float) -> str:
    """
    Approximate local timezone string from longitude.

    This is a rough heuristic (±30 min accuracy) used only as a last resort
    when an explicit timezone is not provided.  Prefer passing an explicit
    timezone name (e.g. 'Europe/Budapest') when available.

    Returns a UTC offset string like 'Etc/GMT+2'.
    """
    offset_h = int(round(lon / 15.0))
    offset_h = max(-12, min(14, offset_h))
    sign = "-" if offset_h >= 0 else "+"   # Etc/GMT sign convention is reversed
    return f"Etc/GMT{sign}{abs(offset_h)}"


def utc_to_local(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """
    Convert a UTC-indexed DataFrame's index to local time.

    The returned DataFrame has a tz-aware index in `tz`.

    Parameters
    ----------
    df : DataFrame with UTC DatetimeIndex
    tz : IANA timezone string (e.g. 'Europe/Budapest', 'US/Eastern')

    Returns
    -------
    DataFrame with local-time index (tz-aware).
    """
    if df.empty:
        return df
    df = df.copy()
    df.index = df.index.tz_convert(tz)
    return df
