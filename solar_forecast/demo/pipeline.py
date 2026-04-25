"""
Demo-mode forecast pipeline.

Works with NO external dependencies:
  - No CAMS API key required
  - No PostgreSQL required
  - No trained XGBoost model required
  - Only Open-Meteo (free, no key) + pvlib physics

Returns a dict with:
  {
    "hourly": pd.DataFrame (timestamp index, ghi_wm2, power_kw, energy_kwh, kt, t_cell_c),
    "summary": dict (today_kwh, tomorrow_kwh, total_7d_kwh, peak_power_kw, ...),
    "clearsky_hourly": pd.DataFrame,  # clear-sky reference
  }
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default climatological aerosol fallbacks (continental Europe)
_DEFAULT_AOD_550 = 0.12
_DEFAULT_ALPHA = 1.3
_DEFAULT_SSA = 0.92
_DEFAULT_ASYM = 0.65
_DEFAULT_OZONE_ATM_CM = 0.32
_DEFAULT_PRECIP_WATER = 1.5


def _resolve_tilt_azimuth(lat: float, tilt: Optional[float], azimuth: Optional[float]):
    t = tilt if tilt is not None else round(abs(lat) * 0.76, 1)
    a = azimuth if azimuth is not None else (180.0 if lat >= 0 else 0.0)
    return float(t), float(a)


def _fetch_openmeteo(lat: float, lon: float, horizon_days: int) -> pd.DataFrame:
    """Fetch Open-Meteo hourly forecast. Returns UTC-indexed DataFrame."""
    try:
        import openmeteo_requests
        import requests_cache
        from retry_requests import retry

        session = requests_cache.CachedSession(".cache/openmeteo", expire_after=1800)
        session = retry(session, retries=3, backoff_factor=0.5)
        om = openmeteo_requests.Client(session=session)

        resp = om.weather_api(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": [
                    "shortwave_radiation",
                    "direct_normal_irradiance",
                    "diffuse_radiation",
                    "cloud_cover",
                    "cloud_cover_low",
                    "temperature_2m",
                    "relative_humidity_2m",
                    "surface_pressure",
                    "wind_speed_10m",
                ],
                "forecast_days": horizon_days,
                "timezone": "UTC",
            },
        )[0]

        h = resp.Hourly()
        times = pd.date_range(
            start=pd.Timestamp(h.Time(), unit="s", tz="UTC"),
            end=pd.Timestamp(h.TimeEnd(), unit="s", tz="UTC"),
            freq=pd.Timedelta(seconds=h.Interval()),
            inclusive="left",
        )

        vars_ = [
            "shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation",
            "cloud_cover", "cloud_cover_low",
            "temperature_2m", "relative_humidity_2m",
            "surface_pressure", "wind_speed_10m",
        ]
        data = {v: h.Variables(i).ValuesAsNumpy() for i, v in enumerate(vars_)}
        df = pd.DataFrame(data, index=times)
        df.rename(columns={
            "shortwave_radiation": "ghi",
            "direct_normal_irradiance": "dni",
            "diffuse_radiation": "dhi",
            "cloud_cover": "cloud_cover_frac",
            "cloud_cover_low": "cloud_low_frac",
            "temperature_2m": "temp_c",
            "relative_humidity_2m": "rh",
            "surface_pressure": "pressure_hpa",
            "wind_speed_10m": "wind_speed",
        }, inplace=True)
        df["cloud_cover_frac"] /= 100.0
        df["cloud_low_frac"] /= 100.0
        df.index.name = "timestamp"
        return df.clip(lower={"ghi": 0, "dni": 0, "dhi": 0})

    except Exception as exc:
        logger.warning("Open-Meteo fetch failed (%s). Using synthetic clear-sky.", exc)
        return pd.DataFrame()


def _compute_clearsky(lat: float, lon: float, altitude: float,
                      tilt: float, azimuth: float,
                      times: pd.DatetimeIndex) -> pd.DataFrame:
    """pvlib spectrl2 clear-sky, returning ghi_cs, dhi_cs, dni_cs, poa_cs."""
    try:
        import pvlib

        loc = pvlib.location.Location(lat, lon, altitude=altitude)
        solpos = loc.get_solarposition(times)
        et_rad = pvlib.irradiance.get_extra_radiation(times)

        aod_500 = _DEFAULT_AOD_550 * (500 / 550) ** _DEFAULT_ALPHA

        sp = pvlib.irradiance.spectrl2(
            apparent_zenith=solpos["apparent_zenith"],
            aoi=pvlib.irradiance.aoi(tilt, azimuth,
                                      solpos["apparent_zenith"],
                                      solpos["azimuth"]),
            surface_tilt=tilt,
            ground_albedo=0.2,
            surface_pressure=101325.0,
            ozone=_DEFAULT_OZONE_ATM_CM,
            precipitable_water=_DEFAULT_PRECIP_WATER,
            aerosol_turbidity_500nm=aod_500,
            alpha=_DEFAULT_ALPHA,
            dayofyear=times.dayofyear,
        )

        wl = sp["wavelength"].values
        pv_mask = (wl >= 300) & (wl <= 1200)

        poa_cs = np.trapz(sp["poa_global"].values[:, pv_mask], wl[pv_mask], axis=1)
        ghi_cs = np.trapz(sp["poa_global"].values[:, pv_mask], wl[pv_mask], axis=1)

        cs = loc.get_clearsky(times, model="simplified_solis")
        cs_poa = pvlib.irradiance.get_total_irradiance(
            tilt, azimuth,
            solpos["apparent_zenith"], solpos["azimuth"],
            cs["dni"], cs["ghi"], cs["dhi"],
        )

        df = pd.DataFrame({
            "ghi_cs": cs["ghi"].clip(lower=0),
            "dhi_cs": cs["dhi"].clip(lower=0),
            "dni_cs": cs["dni"].clip(lower=0),
            "poa_cs": cs_poa["poa_global"].clip(lower=0),
        }, index=times)
        return df

    except Exception as exc:
        logger.warning("spectrl2 failed (%s), using simplified_solis.", exc)
        try:
            import pvlib
            loc = pvlib.location.Location(lat, lon, altitude=altitude)
            cs = loc.get_clearsky(times, model="simplified_solis")
            solpos = loc.get_solarposition(times)
            cs_poa = pvlib.irradiance.get_total_irradiance(
                tilt, azimuth,
                solpos["apparent_zenith"], solpos["azimuth"],
                cs["dni"], cs["ghi"], cs["dhi"],
            )
            return pd.DataFrame({
                "ghi_cs": cs["ghi"].clip(lower=0),
                "dhi_cs": cs["dhi"].clip(lower=0),
                "dni_cs": cs["dni"].clip(lower=0),
                "poa_cs": cs_poa["poa_global"].clip(lower=0),
            }, index=times)
        except Exception:
            return pd.DataFrame()


def _poa_from_components(ghi: pd.Series, dhi: pd.Series, dni: pd.Series,
                          lat: float, lon: float, tilt: float, azimuth: float,
                          times: pd.DatetimeIndex) -> pd.Series:
    """Perez transposition: GHI+DNI+DHI → POA."""
    try:
        import pvlib
        loc = pvlib.location.Location(lat, lon)
        solpos = loc.get_solarposition(times)
        poa = pvlib.irradiance.get_total_irradiance(
            tilt, azimuth,
            solpos["apparent_zenith"], solpos["azimuth"],
            dni, ghi, dhi, model="perez",
        )
        return poa["poa_global"].clip(lower=0)
    except Exception:
        return (ghi * 0.9).clip(lower=0)


def _iam_factor(tilt: float) -> float:
    """ASHRAE IAM for fixed tilt (simplified, mean over day)."""
    import math
    b0 = 0.05
    aoi = abs(tilt - 15)
    return max(0.0, 1 - b0 * (1 / max(math.cos(math.radians(aoi)), 0.01) - 1))


def _cell_temperature(poa: pd.Series, temp_c: pd.Series,
                      wind: Optional[pd.Series] = None) -> pd.Series:
    """NOCT model: T_cell = T_air + (NOCT-20)/800 * G_POA."""
    noct = 45.0
    return temp_c + (noct - 20.0) / 800.0 * poa


def _dc_power(poa_eff: pd.Series, t_cell: pd.Series,
              capacity_kw: float, technology: str) -> pd.Series:
    """STC-referenced DC power with temperature coefficient."""
    temp_coeff = {
        "mono_si": -0.0045,
        "poly_si": -0.0045,
        "cdte": -0.0025,
        "cigs": -0.0036,
        "hit": -0.0025,
    }.get(technology, -0.0045)

    stc_irr = 1000.0
    p_dc = capacity_kw * (poa_eff / stc_irr) * (1 + temp_coeff * (t_cell - 25.0))
    return p_dc.clip(lower=0)


def _build_summary(hourly_df: pd.DataFrame, capacity_kw: float) -> dict:
    """Compute summary KPIs from the hourly output DataFrame."""
    now_utc = pd.Timestamp.now(tz="UTC").normalize()
    today = now_utc
    tomorrow = today + pd.Timedelta(days=1)

    def daily_kwh(day: pd.Timestamp) -> float:
        mask = (hourly_df.index >= day) & (hourly_df.index < day + pd.Timedelta(days=1))
        return float(hourly_df.loc[mask, "energy_kwh"].sum())

    total_kwh = float(hourly_df["energy_kwh"].sum())
    today_kwh = daily_kwh(today)
    tomorrow_kwh = daily_kwh(tomorrow)

    peak_idx = hourly_df["power_kw"].idxmax()
    peak_power = float(hourly_df["power_kw"].max())
    peak_hour = str(peak_idx) if peak_idx is not pd.NaT else ""

    hours = max(len(hourly_df), 1)
    cap_factor = (total_kwh / (capacity_kw * hours)) * 100.0

    clear_kwh = float(hourly_df.get("energy_kwh_cs", pd.Series([total_kwh])).sum())
    cloud_loss = max(0.0, (clear_kwh - total_kwh) / max(clear_kwh, 1e-6) * 100.0)

    return {
        "today_kwh": round(today_kwh, 3),
        "tomorrow_kwh": round(tomorrow_kwh, 3),
        "total_7d_kwh": round(total_kwh, 3),
        "peak_power_kw": round(peak_power, 3),
        "peak_hour_utc": peak_hour,
        "capacity_factor_pct": round(min(cap_factor, 100.0), 2),
        "cloud_loss_pct": round(min(cloud_loss, 100.0), 2),
    }


def run_demo_forecast(
    lat: float,
    lon: float,
    altitude: float = 0.0,
    capacity_kw: float = 5.0,
    tilt: Optional[float] = None,
    azimuth: Optional[float] = None,
    technology: str = "mono_si",
    horizon_days: int = 7,
) -> dict:
    """
    Run a complete physics-based forecast. No keys or DB required.

    Returns dict with 'hourly' DataFrame and 'summary' dict.
    """
    tilt, azimuth = _resolve_tilt_azimuth(lat, tilt, azimuth)

    # 1. Fetch live weather (Open-Meteo)
    weather = _fetch_openmeteo(lat, lon, horizon_days)

    now_utc = pd.Timestamp.now(tz="UTC").floor("h")
    if weather.empty:
        # Synthetic fallback: pure clear-sky
        times = pd.date_range(now_utc, periods=horizon_days * 24, freq="h", tz="UTC")
        weather = pd.DataFrame(index=times)

    times = weather.index if not weather.empty else pd.date_range(
        now_utc, periods=horizon_days * 24, freq="h", tz="UTC"
    )

    # 2. Clear-sky
    cs = _compute_clearsky(lat, lon, altitude, tilt, azimuth, times)

    # 3. POA irradiance
    if not weather.empty and "ghi" in weather.columns:
        ghi = weather["ghi"]
        dhi = weather.get("dhi", weather["ghi"] * 0.15)
        dni = weather.get("dni", (weather["ghi"] - dhi) / np.maximum(
            np.cos(np.radians(45)), 0.1))
        poa = _poa_from_components(ghi, dhi, dni, lat, lon, tilt, azimuth, times)
        temp_c = weather.get("temp_c", pd.Series(20.0, index=times))
        wind = weather.get("wind_speed", pd.Series(2.0, index=times))
        cloud_frac = weather.get("cloud_cover_frac", pd.Series(0.3, index=times))
    else:
        poa = cs.get("poa_cs", pd.Series(0.0, index=times))
        ghi = cs.get("ghi_cs", pd.Series(0.0, index=times))
        temp_c = pd.Series(20.0, index=times)
        wind = pd.Series(2.0, index=times)
        cloud_frac = pd.Series(0.0, index=times)

    # Align indices
    poa = poa.reindex(times).fillna(0.0)
    ghi = ghi.reindex(times).fillna(0.0)
    temp_c = temp_c.reindex(times).fillna(20.0)
    wind = wind.reindex(times).fillna(2.0)

    # 4. IAM correction
    iam = _iam_factor(tilt)
    poa_eff = poa * iam

    # 5. Cell temperature
    t_cell = _cell_temperature(poa_eff, temp_c, wind)

    # 6. DC power
    p_dc = _dc_power(poa_eff, t_cell, capacity_kw, technology)

    # 7. AC losses (inverter η=0.96, wiring=0.99, soiling=0.98, mismatch=0.99)
    system_loss = 0.96 * 0.99 * 0.98 * 0.99
    p_ac = (p_dc * system_loss).clip(lower=0)

    # 8. Energy per hour (Wh → kWh, 1 h intervals)
    energy_kwh = p_ac  # already in kW × 1h = kWh

    # 9. Clearness index
    ghi_cs = cs.get("ghi_cs", pd.Series(np.nan, index=times)).reindex(times)
    kt = (ghi / ghi_cs.replace(0, np.nan)).clip(0, 1.5)

    # 10. Clear-sky energy (for cloud-loss metric)
    poa_cs = cs.get("poa_cs", pd.Series(0.0, index=times)).reindex(times).fillna(0.0)
    p_ac_cs = (_dc_power(poa_cs * iam, t_cell, capacity_kw, technology) * system_loss).clip(0)

    out = pd.DataFrame({
        "ghi_wm2": ghi.values,
        "poa_wm2": poa_eff.values,
        "power_kw": p_ac.values,
        "energy_kwh": energy_kwh.values,
        "kt": kt.values,
        "t_cell_c": t_cell.values,
        "cloud_cover_frac": cloud_frac.reindex(times).fillna(0.0).values,
        "energy_kwh_cs": p_ac_cs.values,
    }, index=times)

    out.index.name = "timestamp_utc"

    summary = _build_summary(out, capacity_kw)

    return {
        "hourly": out,
        "summary": summary,
        "clearsky_hourly": cs.reindex(times),
        "location": {
            "lat": lat, "lon": lon, "altitude": altitude,
            "tilt": tilt, "azimuth": azimuth,
            "capacity_kw": capacity_kw, "technology": technology,
        },
    }
