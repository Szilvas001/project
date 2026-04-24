"""
Clear-sky spectral irradiance model — full physics implementation.

Uses pvlib.spectrum.spectrl2 (Bird & Riordan 1986) to compute the solar
spectrum across 300–4000 nm, then integrates to broadband GHI/DNI/DHI/POA.

Physical inputs per time step
------------------------------
  apparent_zenith      Solar apparent zenith (°) — from pvlib ephemeris
  aoi                  Angle of incidence on tilted surface (°)
  surface_pressure     hPa
  relative_airmass     Kasten-Young 1989
  precipitable_water   cm  (total column water vapour)
  ozone                atm-cm  (= DU / 1000)
  aerosol_turbidity    AOD at 500 nm
  angstrom_alpha1      Ångström exponent (340–500 nm band, SMARTS ALPHA1)
  angstrom_alpha2      Ångström exponent (500–1064 nm band, SMARTS ALPHA2)
  ssa                  Aerosol single-scattering albedo ω₀
  asymmetry_param      Aerosol asymmetry parameter g (Henyey-Greenstein)
  ground_albedo        Surface/terrain albedo

Spectral integration
--------------------
  Standard: broadband integral over 300–4000 nm
  PV-window: integral over 300–1200 nm (Si-relevant)

UTC note: `times` must be tz-aware UTC DatetimeIndex.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.spectrum import spectrl2

logger = logging.getLogger(__name__)

_WL_MIN    = 300.0
_WL_MAX    = 4000.0
_PV_WL_MIN = 300.0
_PV_WL_MAX = 1200.0   # Si cell cut-off


def compute_clearsky(
    times: pd.DatetimeIndex,
    lat: float,
    lon: float,
    altitude: float,
    tilt: float,
    azimuth: float,
    aod_550nm:          float | np.ndarray | pd.Series = 0.10,
    angstrom_alpha:     float | np.ndarray | pd.Series = 1.30,
    angstrom_alpha2:    float | np.ndarray | pd.Series | None = None,
    precipitable_water: float | np.ndarray | pd.Series = 1.50,
    ozone_du:           float | np.ndarray | pd.Series = 310.0,
    surface_pressure:   float | np.ndarray | pd.Series = 1013.25,
    ground_albedo:      float | np.ndarray | pd.Series = 0.20,
    ssa:                float | np.ndarray | pd.Series = 0.92,
    asymmetry_param:    float | np.ndarray | pd.Series = 0.65,
    return_spectra: bool = False,
) -> pd.DataFrame:
    """
    Compute clear-sky broadband irradiance for UTC timestamps.

    Returns DataFrame with:
        ghi_clear, dni_clear, dhi_clear, poa_clear
        ghi_clear_pv, dni_clear_pv, dhi_clear_pv, poa_clear_pv  (PV window)
        zenith, azimuth_sun, airmass, cos_zenith, aoi, e0, kt_clear
    """
    if not isinstance(times, pd.DatetimeIndex):
        times = pd.DatetimeIndex(times)
    if times.tz is None:
        times = times.tz_localize("UTC")

    loc = Location(lat, lon, "UTC", altitude)
    solar_pos    = loc.get_solarposition(times)
    airmass_rel  = pvlib.atmosphere.get_relative_airmass(
        solar_pos["apparent_zenith"], model="kastenyoung1989"
    )

    n = len(times)
    pres_arr   = _broadcast(surface_pressure, n)
    aod_arr    = _broadcast(aod_550nm, n)
    alpha_arr  = _broadcast(angstrom_alpha, n)
    alpha2_arr = _broadcast(
        angstrom_alpha2 if angstrom_alpha2 is not None else angstrom_alpha, n
    )
    pw_arr     = _broadcast(precipitable_water, n)
    oz_arr     = _broadcast(ozone_du, n) / 1000.0   # DU → atm-cm
    alb_arr    = _broadcast(ground_albedo, n)
    ssa_arr    = _broadcast(ssa, n)
    g_arr      = _broadcast(asymmetry_param, n)

    aoi_series = pvlib.irradiance.aoi(
        tilt, azimuth,
        solar_pos["apparent_zenith"],
        solar_pos["azimuth"],
    )
    e0_arr       = _extraterrestrial(times.dayofyear.values)
    zenith_arr   = solar_pos["apparent_zenith"].values
    azimuth_sun  = solar_pos["azimuth"].values
    am_rel_arr   = airmass_rel.values
    aoi_arr      = aoi_series.values

    rows = []
    spectra_list = [] if return_spectra else None

    for i in range(n):
        sza    = float(zenith_arr[i])
        am_rel = float(am_rel_arr[i]) if np.isfinite(am_rel_arr[i]) else np.nan

        if sza >= 89.9 or not np.isfinite(am_rel):
            rows.append(_zero_row(sza, azimuth_sun[i], e0_arr[i]))
            if return_spectra:
                spectra_list.append(None)
            continue

        # Convert AOD_550 → AOD_500 using Ångström α1 (UV-visible)
        a1 = float(np.clip(alpha_arr[i], 0.0, 3.0))
        a2 = float(np.clip(alpha2_arr[i], 0.0, 3.0))
        aod_500 = float(aod_arr[i]) * (500.0 / 550.0) ** a1
        aod_500 = float(np.clip(aod_500, 0.001, 5.0))

        # spectrl2 uses a single Ångström exponent — use the mean of α1/α2
        alpha_eff = 0.5 * (a1 + a2)

        try:
            sp = spectrl2(
                apparent_zenith=sza,
                aoi=float(np.clip(aoi_arr[i], 0.0, 90.0)),
                surface_tilt=float(tilt),
                ground_albedo=float(alb_arr[i]),
                surface_pressure=float(pres_arr[i]),
                relative_airmass=am_rel,
                precipitable_water=float(pw_arr[i]),
                ozone=float(oz_arr[i]),
                aerosol_turbidity=aod_500,
                dayofyear=int(times[i].dayofyear),
                scattering_albedo=float(np.clip(ssa_arr[i], 0.5, 1.0)),
                asymmetry_parameter=float(np.clip(g_arr[i], 0.3, 0.9)),
                aerosol_angstrom_exponent=alpha_eff,
            )
        except Exception as exc:
            logger.debug("spectrl2 step %d (sza=%.1f°): %s", i, sza, exc)
            rows.append(_zero_row(sza, azimuth_sun[i], e0_arr[i]))
            if return_spectra:
                spectra_list.append(None)
            continue

        wl       = np.asarray(sp["wavelength"])
        dni_spec = np.asarray(sp["dni"])
        dhi_spec = np.asarray(sp["dhi"])
        poa_spec = np.asarray(sp["poa_global"])

        mask_full = (wl >= _WL_MIN) & (wl <= _WL_MAX)
        mask_pv   = (wl >= _PV_WL_MIN) & (wl <= _PV_WL_MAX)

        def _integrate(arr, mask):
            return max(0.0, float(np.trapz(arr[mask], wl[mask])))

        dni_clear = _integrate(dni_spec, mask_full)
        dhi_clear = _integrate(dhi_spec, mask_full)
        poa_clear = _integrate(poa_spec, mask_full)
        cos_z     = float(np.cos(np.radians(sza)))
        ghi_clear = max(0.0, dni_clear * cos_z + dhi_clear)

        dni_pv = _integrate(dni_spec, mask_pv)
        dhi_pv = _integrate(dhi_spec, mask_pv)
        poa_pv = _integrate(poa_spec, mask_pv)
        ghi_pv = max(0.0, dni_pv * cos_z + dhi_pv)

        rows.append({
            "ghi_clear":     ghi_clear,
            "dni_clear":     dni_clear,
            "dhi_clear":     dhi_clear,
            "poa_clear":     poa_clear,
            "ghi_clear_pv":  ghi_pv,
            "dni_clear_pv":  dni_pv,
            "dhi_clear_pv":  dhi_pv,
            "poa_clear_pv":  poa_pv,
            "zenith":        sza,
            "azimuth_sun":   float(azimuth_sun[i]),
            "airmass":       am_rel,
            "cos_zenith":    cos_z,
            "aoi":           float(np.clip(aoi_arr[i], 0, 90)),
            "e0":            float(e0_arr[i]),
            "kt_clear":      1.0,
        })

        if return_spectra:
            spectra_list.append({
                "wavelength":  wl,
                "dni":         dni_spec,
                "dhi":         dhi_spec,
                "poa_global":  poa_spec,
            })

    df = pd.DataFrame(rows, index=times)
    df = df.fillna(0.0)
    if return_spectra:
        df["spectra"] = spectra_list

    return df


def compute_clearsky_from_weather(
    df_weather: pd.DataFrame,
    lat: float,
    lon: float,
    altitude: float,
    tilt: float,
    azimuth: float,
    return_spectra: bool = False,
) -> pd.DataFrame:
    """
    Compute clear-sky using atmospheric columns present in a weather DataFrame.

    Automatically uses aod_550nm, angstrom_alpha1/2, precipitable_water,
    total_ozone, surface_pressure, forecast_albedo, ssa_550nm, asymmetry_factor.
    """
    def _col(name, default):
        if name in df_weather.columns:
            return df_weather[name].values
        return default

    return compute_clearsky(
        times=df_weather.index,
        lat=lat, lon=lon, altitude=altitude,
        tilt=tilt, azimuth=azimuth,
        aod_550nm          =_col("aod_550nm",         0.10),
        angstrom_alpha     =_col("angstrom_alpha1",
                             _col("angstrom_exponent", 1.30)),
        angstrom_alpha2    =_col("angstrom_alpha2",   None),
        precipitable_water =_col("precipitable_water", 1.50),
        ozone_du           =_col("total_ozone",        310.0),
        surface_pressure   =_col("surface_pressure",   1013.25),
        ground_albedo      =_col("forecast_albedo",
                             _col("snow_albedo",        0.20)),
        ssa                =_col("ssa_550nm",            0.92),
        asymmetry_param    =_col("asymmetry_factor",    0.65),
        return_spectra=return_spectra,
    )


# ── Internal helpers ──────────────────────────────────────────────────────

def _zero_row(sza: float, azimuth_sun: float, e0: float) -> dict:
    cos_z = max(0.0, float(np.cos(np.radians(sza))))
    return {
        "ghi_clear": 0.0, "dni_clear": 0.0, "dhi_clear": 0.0, "poa_clear": 0.0,
        "ghi_clear_pv": 0.0, "dni_clear_pv": 0.0, "dhi_clear_pv": 0.0, "poa_clear_pv": 0.0,
        "zenith": sza, "azimuth_sun": azimuth_sun,
        "airmass": np.nan, "cos_zenith": cos_z,
        "aoi": 90.0, "e0": e0, "kt_clear": 1.0,
    }


def _broadcast(value, n: int) -> np.ndarray:
    if isinstance(value, pd.Series):
        value = value.values
    if value is None:
        return np.full(n, np.nan)
    v = np.asarray(value, dtype=float)
    if v.ndim == 0 or len(v) == 1:
        return np.full(n, float(v.flat[0]))
    if len(v) != n:
        raise ValueError(f"_broadcast: expected {n}, got {len(v)}")
    return v


def _extraterrestrial(dayofyear: np.ndarray) -> np.ndarray:
    E_sc  = 1361.0
    gamma = 2.0 * np.pi * (dayofyear - 1.0) / 365.0
    return E_sc * (
        1.00011
        + 0.034221 * np.cos(gamma)
        + 0.001280 * np.sin(gamma)
        + 0.000719 * np.cos(2.0 * gamma)
        + 0.000077 * np.sin(2.0 * gamma)
    )
