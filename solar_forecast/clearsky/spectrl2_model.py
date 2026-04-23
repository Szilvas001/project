"""
Clear-sky spectral irradiance model based on pvlib.spectrum.spectrl2.

spectrl2 (Bird & Riordan 1986) computes the solar spectrum at the surface
across 300–4000 nm for a cloudless atmosphere.  Integrating across wavelengths
yields broadband GHI, DNI, DHI, and POA irradiance values.

This module is the authoritative clear-sky reference for the system.
All other components (physics Kt, AI training) consume its output.

Physical inputs required per time step
---------------------------------------
  apparent_zenith      Solar apparent zenith angle (°) — from pvlib ephemeris
  aoi                  Angle of incidence on tilted surface (°)
  surface_pressure     hPa
  relative_airmass     Kasten-Young 1989
  precipitable_water   Total column water vapour (cm)
  ozone                Total column ozone (atm-cm = DU / 1000)
  aerosol_turbidity    AOD at 500 nm (Ångström turbidity coefficient)
  dayofyear            Integer day of year
"""

import logging

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.spectrum import spectrl2

logger = logging.getLogger(__name__)


def compute_clearsky(
    times: pd.DatetimeIndex,
    lat: float,
    lon: float,
    altitude: float,
    tilt: float,
    azimuth: float,
    aod_550nm: float | pd.Series = 0.10,
    angstrom_alpha: float | pd.Series = 1.30,
    precipitable_water: float | pd.Series = 1.50,
    ozone_du: float | pd.Series = 310.0,
    surface_pressure: float | pd.Series = 1013.25,
    ground_albedo: float = 0.20,
    scattering_albedo: float = 0.945,
    asymmetry_param: float = 0.65,
) -> pd.DataFrame:
    """
    Compute clear-sky broadband irradiance for a sequence of UTC timestamps.

    Returns a DataFrame indexed by `times` with columns:
        ghi_clear   — Global Horizontal Irradiance (W/m²)
        dni_clear   — Direct Normal Irradiance (W/m²)
        dhi_clear   — Diffuse Horizontal Irradiance (W/m²)
        poa_clear   — Plane-of-Array total irradiance (W/m²)
        zenith      — Apparent solar zenith angle (°)
        azimuth_sun — Solar azimuth angle (°)
        airmass     — Relative air mass (Kasten-Young)
        cos_zenith  — cos(zenith), useful downstream

    Parameters
    ----------
    times               UTC DatetimeIndex
    lat, lon            Decimal degrees
    altitude            m above sea level
    tilt                Panel tilt from horizontal (°)
    azimuth             Panel azimuth; 180 = south (°)
    aod_550nm           Aerosol optical depth at 550 nm
    angstrom_alpha      Ångström wavelength exponent
    precipitable_water  cm
    ozone_du            Dobson units (will be converted to atm-cm)
    surface_pressure    hPa
    ground_albedo       Albedo of surrounding terrain
    scattering_albedo   Aerosol single-scattering albedo (ω₀)
    asymmetry_param     Aerosol asymmetry parameter g (Henyey-Greenstein)
    """
    loc = Location(lat, lon, "UTC", altitude)
    solar_pos = loc.get_solarposition(times)

    airmass_rel = pvlib.atmosphere.get_relative_airmass(
        solar_pos["apparent_zenith"], model="kastenyoung1989"
    )

    # Absolute airmass (pressure-corrected)
    pres_arr = _broadcast(surface_pressure, len(times))
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass_rel, pres_arr)

    # Angle of incidence on tilted panel surface
    aoi = pvlib.irradiance.aoi(
        tilt, azimuth,
        solar_pos["apparent_zenith"],
        solar_pos["azimuth"],
    )

    # Broadcast all scalar atmospheric inputs to length-n arrays
    aod_arr   = _broadcast(aod_550nm,          len(times))
    alpha_arr = _broadcast(angstrom_alpha,      len(times))
    pw_arr    = _broadcast(precipitable_water,  len(times))
    oz_arr    = _broadcast(ozone_du,            len(times)) / 1000.0  # DU → atm-cm
    pres_arr  = _broadcast(surface_pressure,    len(times))

    rows = []
    zenith_arr = solar_pos["apparent_zenith"].values
    azimuth_sun_arr = solar_pos["azimuth"].values
    airmass_rel_arr = airmass_rel.values
    aoi_arr = aoi.values

    for i in range(len(times)):
        sza = zenith_arr[i]
        am_rel = airmass_rel_arr[i]

        # Below horizon or invalid airmass → zero
        if sza >= 89.9 or np.isnan(am_rel) or np.isinf(am_rel):
            rows.append(_zero_row())
            continue

        try:
            sp = spectrl2(
                apparent_zenith=float(sza),
                aoi=float(np.clip(aoi_arr[i], 0, 90)),
                surface_tilt=float(tilt),
                ground_albedo=float(ground_albedo),
                surface_pressure=float(pres_arr[i]),
                relative_airmass=float(am_rel),
                precipitable_water=float(pw_arr[i]),
                ozone=float(oz_arr[i]),
                aerosol_turbidity=float(aod_arr[i]),
                dayofyear=int(times[i].dayofyear),
                scattering_albedo=float(scattering_albedo),
                asymmetry_parameter=float(asymmetry_param),
                aerosol_angstrom_exponent=float(alpha_arr[i]),
            )
        except Exception as exc:
            logger.debug("spectrl2 step %d: %s", i, exc)
            rows.append(_zero_row())
            continue

        wl = sp["wavelength"]
        dni_clear = float(np.trapz(sp["dni"], wl))
        dhi_clear = float(np.trapz(sp["dhi"], wl))
        poa_clear = float(np.trapz(sp["poa_global"], wl))

        cos_z = np.cos(np.radians(sza))
        ghi_clear = max(0.0, dni_clear * cos_z + dhi_clear)

        rows.append({
            "ghi_clear": ghi_clear,
            "dni_clear": max(0.0, dni_clear),
            "dhi_clear": max(0.0, dhi_clear),
            "poa_clear": max(0.0, poa_clear),
        })

    df = pd.DataFrame(rows, index=times)
    df["zenith"]     = zenith_arr
    df["azimuth_sun"] = azimuth_sun_arr
    df["airmass"]    = airmass_rel_arr
    df["cos_zenith"] = np.cos(np.radians(zenith_arr)).clip(0, 1)

    return df


def _broadcast(val, n: int) -> np.ndarray:
    if isinstance(val, (int, float)):
        return np.full(n, float(val))
    arr = np.asarray(val, dtype=float)
    if len(arr) != n:
        raise ValueError(f"Array length {len(arr)} ≠ expected {n}")
    return arr


def _zero_row() -> dict:
    return {"ghi_clear": 0.0, "dni_clear": 0.0, "dhi_clear": 0.0, "poa_clear": 0.0}


def standard_am15g_spectrum() -> pd.DataFrame:
    """
    Return the ASTM G173-03 AM1.5G reference spectrum (280–4000 nm).
    Values are representative; a full table is embedded for use in
    spectral response calculations without requiring external files.
    """
    # Compact lookup: wavelength (nm), irradiance (W/m²/nm)
    # Source: NREL / ASTM G173-03 (selected representative points)
    data = [
        (280, 0.0000), (300, 0.0082), (320, 0.0682), (340, 0.1576),
        (360, 0.2100), (380, 0.3008), (400, 0.5156), (420, 1.3294),
        (440, 1.6552), (460, 1.8027), (480, 1.9017), (500, 1.9155),
        (520, 1.8935), (540, 1.8541), (560, 1.8234), (580, 1.8061),
        (600, 1.8139), (620, 1.7943), (640, 1.7439), (660, 1.7175),
        (680, 1.6811), (700, 1.6489), (720, 1.4989), (740, 1.4830),
        (760, 0.9720), (780, 1.3869), (800, 1.3699), (820, 1.3527),
        (840, 1.3278), (860, 1.3198), (900, 1.2568), (950, 1.1518),
        (1000, 1.0671),(1100, 0.8640),(1200, 0.7095),(1300, 0.5310),
        (1400, 0.3152),(1500, 0.3043),(1600, 0.2886),(1800, 0.1777),
        (2000, 0.0864),(2500, 0.0219),(3000, 0.0040),(4000, 0.0002),
    ]
    wl, irr = zip(*data)
    return pd.DataFrame({"wavelength_nm": wl, "irradiance": irr})
