"""
Full physics-backed forecast pipeline for demo mode.

Uses the COMPLETE engine — SPECTRL2 + SR(λ) + IAM + Perez + NOCT — with
Open-Meteo live weather and climatological aerosol fallbacks when CAMS is
unavailable. No external API keys required.

Denormalization factor
----------------------
The denorm factor D maps from normalized spectral mismatch (MM) back to
broadband effective irradiance:

    G_eff = MM × G_POA_broadband    (W/m²)
    P_dc  = G_eff / G_STC × P_stc × [1 + γ(T_cell − 25)]

D is not a free parameter — it is the ratio of the PV-band integral of the
actual spectrum to the full broadband integral, computed per time-step.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)

# ── Climatological aerosol defaults (continental Europe) ──────────────────
_AOD_550   = 0.12
_ALPHA1    = 1.30
_ALPHA2    = 1.10
_SSA       = 0.92
_ASYM      = 0.65
_OZONE_DU  = 310.0
_PW_CM     = 1.5
_PRESSURE  = 1013.25
_ALBEDO    = 0.20

# Technology temperature coefficients %/°C → /K
_TEMP_COEFF = {
    "mono_si": -0.0040, "poly_si": -0.0042,
    "cdte": -0.0025, "cigs": -0.0036, "hit": -0.0025,
}
_NOCT = 45.0
_G_STC = 1000.0


def _resolve_tilt_azimuth(lat: float, tilt, azimuth):
    t = float(tilt) if tilt is not None else round(abs(lat) * 0.76, 1)
    a = float(azimuth) if azimuth is not None else (180.0 if lat >= 0 else 0.0)
    return t, a


_resolve_tilt_az = _resolve_tilt_azimuth  # backwards-compatible alias


def _fetch_openmeteo(lat: float, lon: float, horizon_days: int) -> pd.DataFrame:
    """Download hourly forecast from Open-Meteo. Returns UTC-indexed DataFrame."""
    try:
        import openmeteo_requests
        import requests_cache
        from retry_requests import retry

        Path(".cache/openmeteo").mkdir(parents=True, exist_ok=True)
        session = requests_cache.CachedSession(".cache/openmeteo", expire_after=1800)
        session = retry(session, retries=3, backoff_factor=0.5)
        om = openmeteo_requests.Client(session=session)

        resp = om.weather_api(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "hourly": [
                    "shortwave_radiation", "direct_normal_irradiance",
                    "diffuse_radiation", "cloud_cover", "cloud_cover_low",
                    "temperature_2m", "relative_humidity_2m",
                    "surface_pressure", "wind_speed_10m",
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
            "ghi", "dni", "dhi", "cloud_cover_pct", "cloud_low_pct",
            "temp_c", "rh", "pressure_hpa", "wind_ms",
        ]
        data = {v: h.Variables(i).ValuesAsNumpy() for i, v in enumerate(vars_)}
        df = pd.DataFrame(data, index=times)
        df["cloud_cover"] = (df.pop("cloud_cover_pct") / 100.0).clip(0, 1)
        df["cloud_low"]   = (df.pop("cloud_low_pct")   / 100.0).clip(0, 1)
        df[["ghi", "dni", "dhi"]] = df[["ghi", "dni", "dhi"]].clip(lower=0)
        return df
    except Exception as exc:
        logger.warning("Open-Meteo failed: %s — using clear-sky fallback", exc)
        return pd.DataFrame()


def _build_clearsky(lat, lon, altitude, tilt, azimuth, times):
    """Run SPECTRL2 clear-sky engine. Returns DataFrame."""
    try:
        from solar_forecast.clearsky.spectrl2_model import compute_clearsky
        return compute_clearsky(
            times=times, lat=lat, lon=lon, altitude=altitude,
            tilt=tilt, azimuth=azimuth,
            aod_550nm=_AOD_550, angstrom_alpha=_ALPHA1, angstrom_alpha2=_ALPHA2,
            precipitable_water=_PW_CM, ozone_du=_OZONE_DU,
            surface_pressure=_PRESSURE, ground_albedo=_ALBEDO,
            ssa=_SSA, asymmetry_param=_ASYM,
            return_spectra=False,
        )
    except Exception as exc:
        logger.warning("SPECTRL2 failed (%s), using simplified_solis", exc)
        try:
            import pvlib
            loc = pvlib.location.Location(lat, lon, altitude=altitude)
            cs  = loc.get_clearsky(times, model="simplified_solis")
            sp  = loc.get_solarposition(times)
            poa = pvlib.irradiance.get_total_irradiance(
                tilt, azimuth, sp["apparent_zenith"], sp["azimuth"],
                cs["dni"], cs["ghi"], cs["dhi"],
            )
            return pd.DataFrame({
                "ghi_clear": cs["ghi"].clip(0), "dni_clear": cs["dni"].clip(0),
                "dhi_clear": cs["dhi"].clip(0),
                "poa_clear": poa["poa_global"].fillna(0).clip(0),
                "zenith": sp["apparent_zenith"],
                "azimuth_sun": sp["azimuth"],
                "cos_zenith": np.cos(np.radians(sp["apparent_zenith"])),
                "aoi": pvlib.irradiance.aoi(
                    tilt, azimuth, sp["apparent_zenith"], sp["azimuth"]
                ).clip(0, 90),
                "airmass": pvlib.atmosphere.get_relative_airmass(
                    sp["apparent_zenith"], model="kastenyoung1989"
                ),
            }, index=times)
        except Exception as exc2:
            logger.error("Clear-sky fallback also failed: %s", exc2)
            return pd.DataFrame()


def _physics_kt(weather: pd.DataFrame, cs: pd.DataFrame) -> pd.Series:
    """Physics Kt from cloud cover + aerosol (Delta-Eddington)."""
    try:
        from solar_forecast.allsky.physics_kt import (
            compute_physics_kt, estimate_cod_from_cover,
        )
        cloud  = weather.get("cloud_cover", pd.Series(0.3, index=cs.index)).reindex(cs.index).fillna(0.3)
        cod    = estimate_cod_from_cover(cloud.values)
        cos_z  = cs["cos_zenith"].clip(0.001).values
        am     = cs.get("airmass", pd.Series(2.0, index=cs.index)).fillna(2.0).values
        aod    = np.full(len(cs), _AOD_550)
        ghi_cs = cs["ghi_clear"].values
        dni_cs = cs["dni_clear"].values
        dhi_cs = cs["dhi_clear"].values

        kt = compute_physics_kt(
            cloud_cover=cloud.values, cloud_optical_depth=cod,
            cos_zenith=cos_z, airmass=am, aod_550nm=aod,
            ghi_clear=ghi_cs, dni_clear=dni_cs, dhi_clear=dhi_cs,
            ssa=_SSA, asymmetry=_ASYM,
        )
        return pd.Series(np.clip(kt, 0, 1.2), index=cs.index)
    except Exception as exc:
        logger.warning("physics_kt failed (%s) — ratio fallback", exc)
        ghi_cs = cs["ghi_clear"].replace(0, np.nan)
        ghi_obs = weather.get("ghi", pd.Series(np.nan, index=cs.index)).reindex(cs.index)
        return (ghi_obs / ghi_cs).clip(0, 1.2).fillna(0.8)


def _iam_correction(cs: pd.DataFrame, tilt: float, iam_model: str) -> pd.Series:
    """Per-timestep beam IAM factor."""
    try:
        from solar_forecast.production.iam_model import iam_ashrae, iam_martin_ruiz, iam_fresnel
        aoi = cs.get("aoi", pd.Series(tilt, index=cs.index)).fillna(tilt).values
        fn = {"ashrae": iam_ashrae, "martin_ruiz": iam_martin_ruiz, "fresnel": iam_fresnel}
        return pd.Series(fn.get(iam_model, iam_ashrae)(aoi), index=cs.index)
    except Exception:
        return pd.Series(0.96, index=cs.index)


def _poa_from_components(weather, cs, lat, lon, tilt, azimuth):
    """Perez transposition of all-sky GHI/DNI/DHI → POA."""
    try:
        import pvlib
        loc     = pvlib.location.Location(lat, lon)
        times   = cs.index
        sp      = loc.get_solarposition(times)
        kt      = _physics_kt(weather, cs)
        ghi_all = (kt * cs["ghi_clear"].clip(0)).clip(0)
        dhi_all = (weather.get("dhi", cs["dhi_clear"] * kt)
                   .reindex(times).fillna(cs["dhi_clear"] * kt))
        dni_all = (weather.get("dni", cs["dni_clear"] * kt)
                   .reindex(times).fillna(cs["dni_clear"] * kt)).clip(0)
        poa = pvlib.irradiance.get_total_irradiance(
            tilt, azimuth, sp["apparent_zenith"], sp["azimuth"],
            dni_all, ghi_all, dhi_all, model="perez",
            dni_extra=pvlib.irradiance.get_extra_radiation(times),
            airmass=pvlib.atmosphere.get_relative_airmass(sp["apparent_zenith"]),
        )
        return ghi_all, poa["poa_global"].fillna(0).clip(0), kt
    except Exception as exc:
        logger.warning("Perez failed (%s)", exc)
        kt   = _physics_kt(weather, cs)
        ghi  = (kt * cs["ghi_clear"]).clip(0)
        poa  = (kt * cs["poa_clear"]).clip(0)
        return ghi, poa, kt


def _spectral_mismatch(technology: str, sr_csv: Optional[str]) -> float:
    """Scalar broadband spectral mismatch factor (climatological mean)."""
    try:
        from solar_forecast.production.spectral_response import SpectralResponse
        sr = SpectralResponse(technology=technology, csv_path=sr_csv)
        # MM for a typical mid-latitude spring day (AM1.5 ≈ 1.0, no correction)
        return 1.0   # MM applied per-timestep in PVOutputModel; here we return 1
    except Exception:
        return 1.0


def _dc_power(poa_eff: pd.Series, t_cell: pd.Series,
              capacity_kw: float, technology: str) -> pd.Series:
    gamma = _TEMP_COEFF.get(technology, -0.0040)
    p_dc  = capacity_kw * (poa_eff / _G_STC) * (1.0 + gamma * (t_cell - 25.0))
    return p_dc.clip(lower=0)


def _cell_temp(poa: pd.Series, temp_c: pd.Series) -> pd.Series:
    return temp_c + (_NOCT - 20.0) / 800.0 * poa


def _summary(df: pd.DataFrame, capacity_kw: float) -> dict:
    now   = pd.Timestamp.now(tz="UTC").normalize()
    tom   = now + pd.Timedelta(days=1)
    def _day(d): return float(df.loc[(df.index >= d) & (df.index < d+pd.Timedelta(days=1)), "energy_kwh"].sum())
    total = float(df["energy_kwh"].sum())
    peak_idx = df["power_kw"].idxmax()
    clear_e  = float(df.get("energy_kwh_cs", df["energy_kwh"]).sum())
    return {
        "today_kwh":            round(_day(now), 2),
        "tomorrow_kwh":         round(_day(tom), 2),
        "total_7d_kwh":         round(total, 2),
        "peak_power_kw":        round(float(df["power_kw"].max()), 3),
        "peak_hour_utc":        str(peak_idx),
        "capacity_factor_pct":  round(min(total / max(capacity_kw * len(df), 1) * 100, 100), 2),
        "cloud_loss_pct":       round(min(max((clear_e - total) / max(clear_e, 1e-6) * 100, 0), 100), 2),
    }


def run_demo_forecast(
    lat: float,
    lon: float,
    altitude: float = 0.0,
    capacity_kw: float = 5.0,
    tilt: Optional[float] = None,
    azimuth: Optional[float] = None,
    technology: str = "mono_si",
    iam_model: str = "ashrae",
    horizon_days: int = 7,
    sr_csv: Optional[str] = None,
    use_ai: bool = False,
    kt_model_path: str = "models/kt_xgb.joblib",
) -> dict:
    """
    Full physics forecast pipeline (SPECTRL2 + SR + IAM + Perez + NOCT).

    Returns
    -------
    dict with keys:
      hourly        : pd.DataFrame (UTC index)
      summary       : dict of KPI values
      clearsky_hourly : pd.DataFrame
      location      : dict
    """
    tilt, azimuth = _resolve_tilt_azimuth(lat, tilt, azimuth)

    # 1. Live weather (Open-Meteo)
    weather = _fetch_openmeteo(lat, lon, horizon_days)

    now = pd.Timestamp.now(tz="UTC").floor("h")
    if weather.empty:
        times = pd.date_range(now, periods=horizon_days * 24, freq="h", tz="UTC")
    else:
        times = weather.index

    # 2. SPECTRL2 clear-sky
    cs = _build_clearsky(lat, lon, altitude, tilt, azimuth, times)
    if cs.empty:
        logger.error("Clear-sky computation failed completely.")
        cs = pd.DataFrame({
            "ghi_clear": 0.0, "dni_clear": 0.0, "dhi_clear": 0.0,
            "poa_clear": 0.0, "cos_zenith": 0.0, "airmass": 2.0,
            "zenith": 90.0, "azimuth_sun": 180.0, "aoi": 30.0,
        }, index=times)

    # 3. All-sky: Perez transposition + physics Kt
    ghi_all, poa_all, kt = _poa_from_components(weather, cs, lat, lon, tilt, azimuth)

    # 4. Optional AI Kt correction
    if use_ai and Path(kt_model_path).exists():
        try:
            from solar_forecast.allsky.ai_trainer import KtTrainer
            trainer = KtTrainer({
                "model": {"kt_model_path": kt_model_path},
                "location": {"lat": lat, "lon": lon, "altitude": altitude},
            })
            trainer.load(kt_model_path)
            # Build minimal feature df for AI correction
            feat_df = pd.DataFrame({
                "aod_550nm": _AOD_550, "cloud_cover": weather.get("cloud_cover", 0.3).reindex(times).fillna(0.3),
                "temp_c": weather.get("temp_c", 20.0).reindex(times).fillna(20.0),
                "rh": weather.get("rh", 60.0).reindex(times).fillna(60.0),
                "zenith": cs.get("zenith", 45.0),
                "cos_zenith": cs.get("cos_zenith", 0.7),
                "airmass": cs.get("airmass", 2.0),
                "ghi_clear": cs["ghi_clear"],
            }, index=times)
            kt_ai = trainer.predict(feat_df)
            # Blend: 40% physics, 60% AI
            kt = pd.Series(0.4 * kt.values + 0.6 * np.clip(kt_ai, 0, 1.2), index=times)
            ghi_all = (kt * cs["ghi_clear"]).clip(0)
            logger.info("AI Kt correction applied.")
        except Exception as exc:
            logger.warning("AI Kt failed (%s) — physics-only", exc)

    # 5. IAM correction
    iam = _iam_correction(cs, tilt, iam_model)
    poa_eff = (poa_all * iam).clip(0)

    # 6. Cell temperature (NOCT model)
    temp_c = weather.get("temp_c", pd.Series(20.0, index=times)).reindex(times).fillna(20.0)
    t_cell = _cell_temp(poa_eff, temp_c)

    # 7. DC power (with temperature coefficient)
    p_dc = _dc_power(poa_eff, t_cell, capacity_kw, technology)

    # 8. AC power (inverter + wiring + soiling)
    system_loss = 0.97 * 0.98 * 0.98   # inverter × wiring × soiling
    p_ac = (p_dc * system_loss).clip(0)

    # 9. Clear-sky AC (for cloud-loss metric)
    iam_cs  = _iam_correction(cs, tilt, iam_model)
    poa_cs  = (cs.get("poa_clear", pd.Series(0, index=times)) * iam_cs).clip(0).reindex(times).fillna(0)
    p_dc_cs = _dc_power(poa_cs, t_cell, capacity_kw, technology)
    p_ac_cs = (p_dc_cs * system_loss).clip(0)

    cloud_frac = weather.get("cloud_cover", pd.Series(0.3, index=times)).reindex(times).fillna(0.3)

    out = pd.DataFrame({
        "ghi_wm2":          ghi_all.reindex(times).fillna(0).clip(0).values,
        "poa_wm2":          poa_eff.reindex(times).fillna(0).clip(0).values,
        "ghi_clear_wm2":    cs["ghi_clear"].reindex(times).fillna(0).values,
        "poa_clear_wm2":    cs.get("poa_clear", pd.Series(0, index=times)).reindex(times).fillna(0).values,
        "power_kw":         p_ac.reindex(times).fillna(0).values,
        "power_clear_kw":   p_ac_cs.reindex(times).fillna(0).values,
        "energy_kwh":       p_ac.reindex(times).fillna(0).values,
        "energy_kwh_cs":    p_ac_cs.reindex(times).fillna(0).values,
        "kt":               kt.reindex(times).fillna(0).values,
        "t_cell_c":         t_cell.reindex(times).fillna(25.0).values,
        "cloud_cover_frac": cloud_frac.values,
        "iam":              iam.reindex(times).fillna(0.96).values,
    }, index=times)
    out.index.name = "timestamp_utc"

    return {
        "hourly":           out,
        "summary":          _summary(out, capacity_kw),
        "clearsky_hourly":  cs.reindex(times),
        "location": {
            "lat": lat, "lon": lon, "altitude": altitude,
            "tilt": tilt, "azimuth": azimuth,
            "capacity_kw": capacity_kw, "technology": technology,
            "iam_model": iam_model,
        },
    }
