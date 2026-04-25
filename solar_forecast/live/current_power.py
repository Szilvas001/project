"""
Real-time power estimation from Open-Meteo current conditions.
Returns current power (kW), today's cumulative energy (kWh),
expected end-of-day energy (kWh), and time-since-update (minutes).

Uses the Open-Meteo free forecast API (no API key required).
Caches responses for 300 seconds via requests_cache to avoid hammering the
upstream endpoint on every UI refresh.

Typical usage
-------------
>>> from solar_forecast.live.current_power import estimate_current_power
>>> result = estimate_current_power(
...     lat=47.5, lon=19.0, altitude=120,
...     capacity_kw=10.0, tilt=30, azimuth=180,
...     technology="monocrystalline", iam_model="physical"
... )
>>> print(result["current_kw"], result["utilization_pct"])
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------
try:
    import requests_cache

    _SESSION = requests_cache.CachedSession(
        cache_name="open_meteo_cache",
        expire_after=300,  # 5-minute TTL
        backend="memory",
    )
except ImportError:  # pragma: no cover
    import requests as _requests_plain

    class _PlainSession:  # minimal shim
        def get(self, url: str, **kwargs):
            return _requests_plain.get(url, **kwargs)

    _SESSION = _PlainSession()  # type: ignore[assignment]
    log.warning("requests_cache not installed; responses will not be cached.")

try:
    import pvlib
    from pvlib import irradiance as pvlib_irradiance
    from pvlib import atmosphere as pvlib_atmosphere

    _PVLIB_AVAILABLE = True
except ImportError:
    _PVLIB_AVAILABLE = False
    log.warning("pvlib not installed; simplified irradiance model will be used.")

# ---------------------------------------------------------------------------
# Open-Meteo endpoint
# ---------------------------------------------------------------------------
_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Temperature coefficient for power (fraction per °C above 25 °C STC)
_TEMP_COEFF = {
    "monocrystalline": -0.004,
    "polycrystalline": -0.0045,
    "thin_film": -0.002,
}
_DEFAULT_TEMP_COEFF = -0.004

# Combined balance-of-system / wiring / inverter losses
_SYSTEM_LOSS = 0.14  # 14 %

# IAM correction lookup (simplified; full calculation requires pvlib)
_IAM_MODELS = {"physical", "ashrae", "martin_ruiz"}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_current_conditions(lat: float, lon: float) -> dict[str, Any]:
    """
    Fetch current weather conditions from the Open-Meteo forecast API.

    Retrieves both the ``current_weather`` snapshot and today's full hourly
    timeseries so callers can access sub-hourly radiation values.

    Parameters
    ----------
    lat:
        Latitude of the site in decimal degrees.
    lon:
        Longitude of the site in decimal degrees.

    Returns
    -------
    dict
        Keys: ``temp_c``, ``cloud_cover``, ``shortwave_radiation``,
        ``wind_speed``, ``time_utc``, ``hourly_time``,
        ``hourly_shortwave_radiation``, ``hourly_temperature_2m``,
        ``hourly_cloudcover``.
        Returns an **empty dict** on any network or parse error so callers
        never need to handle exceptions from this function.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": "true",
        "hourly": (
            "shortwave_radiation,"
            "diffuse_radiation,"
            "direct_radiation,"
            "cloudcover,"
            "temperature_2m,"
            "windspeed_10m"
        ),
        "forecast_days": 1,
        "timezone": "UTC",
    }
    try:
        response = _SESSION.get(_OPEN_METEO_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        log.error("Open-Meteo request failed: %s", exc)
        return {}

    try:
        cw = data.get("current_weather", {})
        hourly = data.get("hourly", {})
        return {
            # --- current snapshot -----------------------------------------
            "temp_c": cw.get("temperature", 20.0),
            "wind_speed": cw.get("windspeed", 0.0),
            "time_utc": cw.get("time", ""),
            # Open-Meteo does not expose cloud_cover in current_weather;
            # we derive it from the closest hourly value below.
            "cloud_cover": _current_hour_value(hourly.get("cloudcover", []), 0.0),
            "shortwave_radiation": _current_hour_value(
                hourly.get("shortwave_radiation", []), 0.0
            ),
            # --- full hourly arrays ---------------------------------------
            "hourly_time": hourly.get("time", []),
            "hourly_shortwave_radiation": hourly.get("shortwave_radiation", []),
            "hourly_diffuse_radiation": hourly.get("diffuse_radiation", []),
            "hourly_direct_radiation": hourly.get("direct_radiation", []),
            "hourly_temperature_2m": hourly.get("temperature_2m", []),
            "hourly_cloudcover": hourly.get("cloudcover", []),
            "hourly_windspeed": hourly.get("windspeed_10m", []),
        }
    except Exception as exc:
        log.error("Failed to parse Open-Meteo response: %s", exc)
        return {}


def estimate_current_power(
    lat: float,
    lon: float,
    altitude: float,
    capacity_kw: float,
    tilt: float,
    azimuth: float,
    technology: str = "monocrystalline",
    iam_model: str = "physical",
) -> dict[str, Any]:
    """
    Estimate the instantaneous AC power output of a PV system right now.

    Steps
    -----
    1. Fetch current conditions from Open-Meteo (cached 5 min).
    2. Decompose GHI into plane-of-array (POA) irradiance using either pvlib
       or a simplified cosine-tilt model as fallback.
    3. Apply incidence-angle modifier (IAM), temperature derating, and
       combined system losses.

    Parameters
    ----------
    lat, lon:
        Site coordinates (decimal degrees).
    altitude:
        Site elevation above sea level (metres).
    capacity_kw:
        DC nameplate capacity of the PV array (kW-peak).
    tilt:
        Panel tilt from horizontal (degrees, 0 = flat).
    azimuth:
        Panel azimuth measured clockwise from North (degrees).
        180 = south-facing (Northern Hemisphere optimum).
    technology:
        One of ``"monocrystalline"``, ``"polycrystalline"``, ``"thin_film"``.
    iam_model:
        One of ``"physical"``, ``"ashrae"``, ``"martin_ruiz"``.

    Returns
    -------
    dict
        ``current_kw``, ``capacity_kw``, ``utilization_pct``, ``temp_c``,
        ``cloud_pct``, ``ghi_wm2``, ``poa_wm2``, ``updated_at_utc``.
        On failure returns a zeroed dict so the UI can still render.
    """
    _zero = {
        "current_kw": 0.0,
        "capacity_kw": capacity_kw,
        "utilization_pct": 0.0,
        "temp_c": None,
        "cloud_pct": None,
        "ghi_wm2": 0.0,
        "poa_wm2": 0.0,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        cond = get_current_conditions(lat, lon)
        if not cond:
            return _zero

        ghi = float(cond.get("shortwave_radiation", 0.0) or 0.0)
        temp_c = float(cond.get("temp_c", 20.0))
        cloud_pct = float(cond.get("cloud_cover", 0.0))
        updated_at = cond.get("time_utc", datetime.now(timezone.utc).isoformat())

        # --- Plane-of-array irradiance ------------------------------------
        if _PVLIB_AVAILABLE and ghi > 0:
            poa_wm2 = _pvlib_poa(lat, lon, altitude, tilt, azimuth, ghi, cond)
        else:
            poa_wm2 = _simple_poa(ghi, tilt)

        # --- IAM correction -----------------------------------------------
        iam_factor = _compute_iam(tilt, iam_model)

        # --- Temperature derating -----------------------------------------
        tc = temp_c + 0.035 * poa_wm2  # NOCT-based cell temp estimate
        temp_coeff = _TEMP_COEFF.get(technology, _DEFAULT_TEMP_COEFF)
        temp_factor = 1.0 + temp_coeff * max(0.0, tc - 25.0)

        # --- DC → AC power ------------------------------------------------
        dc_power_kw = capacity_kw * (poa_wm2 / 1000.0) * iam_factor * temp_factor
        ac_power_kw = max(0.0, dc_power_kw * (1.0 - _SYSTEM_LOSS))
        ac_power_kw = min(ac_power_kw, capacity_kw)

        utilization = (ac_power_kw / capacity_kw * 100.0) if capacity_kw > 0 else 0.0

        return {
            "current_kw": round(ac_power_kw, 3),
            "capacity_kw": capacity_kw,
            "utilization_pct": round(utilization, 1),
            "temp_c": round(temp_c, 1),
            "cloud_pct": round(cloud_pct, 1),
            "ghi_wm2": round(ghi, 1),
            "poa_wm2": round(poa_wm2, 1),
            "updated_at_utc": updated_at,
        }
    except Exception as exc:
        log.error("estimate_current_power failed: %s", exc)
        return _zero


def get_today_cumulative(
    lat: float,
    lon: float,
    altitude: float,
    capacity_kw: float,
    tilt: float,
    azimuth: float,
    technology: str = "monocrystalline",
) -> dict[str, Any]:
    """
    Compute cumulative energy generated today and the expected end-of-day total.

    Fetches today's full hourly forecast from Open-Meteo, converts each hour's
    GHI to POA power, integrates elapsed hours, and extrapolates the remaining
    daytime hours to predict total daily yield.

    Parameters
    ----------
    lat, lon:
        Site coordinates (decimal degrees).
    altitude:
        Site elevation above sea level (metres).
    capacity_kw:
        DC nameplate capacity (kW-peak).
    tilt:
        Panel tilt from horizontal (degrees).
    azimuth:
        Panel azimuth clockwise from North (degrees).
    technology:
        PV cell technology string (controls temperature coefficient).

    Returns
    -------
    dict
        ``cumulative_kwh`` – energy produced so far today,
        ``expected_today_kwh`` – projected full-day yield,
        ``hours_elapsed`` – how many hours since midnight UTC,
        ``sunrise_hour`` – first hour with GHI > 10 W/m²,
        ``sunset_hour`` – last hour with GHI > 10 W/m².
    """
    _zero = {
        "cumulative_kwh": 0.0,
        "expected_today_kwh": 0.0,
        "hours_elapsed": 0,
        "sunrise_hour": None,
        "sunset_hour": None,
    }
    try:
        cond = get_current_conditions(lat, lon)
        if not cond:
            return _zero

        hourly_ghi: list = cond.get("hourly_shortwave_radiation", [])
        hourly_temp: list = cond.get("hourly_temperature_2m", [])

        if not hourly_ghi:
            return _zero

        now_utc = datetime.now(timezone.utc)
        current_hour = now_utc.hour  # 0–23

        temp_coeff = _TEMP_COEFF.get(technology, _DEFAULT_TEMP_COEFF)

        hourly_power_kw: list[float] = []
        for h, ghi_val in enumerate(hourly_ghi):
            ghi_val = float(ghi_val or 0.0)
            temp_val = float((hourly_temp[h] if h < len(hourly_temp) else 20.0) or 20.0)

            if _PVLIB_AVAILABLE and ghi_val > 0:
                poa = _pvlib_poa(lat, lon, altitude, tilt, azimuth, ghi_val, cond)
            else:
                poa = _simple_poa(ghi_val, tilt)

            tc = temp_val + 0.035 * poa
            temp_factor = 1.0 + temp_coeff * max(0.0, tc - 25.0)
            dc_kw = capacity_kw * (poa / 1000.0) * temp_factor
            ac_kw = max(0.0, min(dc_kw * (1.0 - _SYSTEM_LOSS), capacity_kw))
            hourly_power_kw.append(ac_kw)

        # Cumulative energy = integrate completed hours (trapezoidal, 1-h steps)
        cumulative_kwh = sum(hourly_power_kw[:current_hour])

        # Expected today = sum of all 24 hours
        expected_today_kwh = sum(hourly_power_kw)

        # Sunrise / sunset detection
        sunrise_hour = next(
            (h for h, g in enumerate(hourly_ghi) if float(g or 0.0) > 10.0), None
        )
        sunset_hour = next(
            (
                h
                for h in range(len(hourly_ghi) - 1, -1, -1)
                if float(hourly_ghi[h] or 0.0) > 10.0
            ),
            None,
        )

        return {
            "cumulative_kwh": round(cumulative_kwh, 3),
            "expected_today_kwh": round(expected_today_kwh, 3),
            "hours_elapsed": current_hour,
            "sunrise_hour": sunrise_hour,
            "sunset_hour": sunset_hour,
        }
    except Exception as exc:
        log.error("get_today_cumulative failed: %s", exc)
        return _zero


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _current_hour_value(series: list, default: float = 0.0) -> float:
    """Return the value for the current UTC hour from an hourly list."""
    if not series:
        return default
    hour = datetime.now(timezone.utc).hour
    idx = min(hour, len(series) - 1)
    val = series[idx]
    return float(val) if val is not None else default


def _simple_poa(ghi: float, tilt: float) -> float:
    """
    Simplified GHI → POA conversion using a cosine-tilt correction.

    Assumes diffuse isotropic sky and ignores ground albedo.  Accurate to
    within ~10 % for south-facing systems with moderate tilt.

    Parameters
    ----------
    ghi:
        Global Horizontal Irradiance (W/m²).
    tilt:
        Panel tilt from horizontal (degrees).

    Returns
    -------
    float
        Plane-of-array irradiance (W/m²).
    """
    if ghi <= 0:
        return 0.0
    tilt_rad = math.radians(tilt)
    # Beam fraction on a tilted surface (simplified Erbs decomposition)
    beam_fraction = 0.75  # assumed
    diffuse_fraction = 1.0 - beam_fraction
    poa_beam = ghi * beam_fraction * math.cos(tilt_rad)
    poa_diffuse = ghi * diffuse_fraction * (1.0 + math.cos(tilt_rad)) / 2.0
    return max(0.0, poa_beam + poa_diffuse)


def _pvlib_poa(
    lat: float,
    lon: float,
    altitude: float,
    tilt: float,
    azimuth: float,
    ghi: float,
    cond: dict,
) -> float:
    """
    Compute POA irradiance using pvlib's full Perez transposition model.

    Falls back to :func:`_simple_poa` on any pvlib error.

    Parameters
    ----------
    lat, lon:
        Site coordinates.
    altitude:
        Site elevation (m).
    tilt:
        Panel tilt (degrees).
    azimuth:
        Panel azimuth clockwise from North (degrees).
    ghi:
        Global Horizontal Irradiance (W/m²).
    cond:
        Conditions dict from :func:`get_current_conditions`.

    Returns
    -------
    float
        POA irradiance (W/m²).
    """
    try:
        now = datetime.now(timezone.utc)
        import pandas as pd

        times = pd.DatetimeIndex([now], tz="UTC")
        location = pvlib.location.Location(
            latitude=lat, longitude=lon, altitude=altitude, tz="UTC"
        )
        solar_pos = location.get_solarposition(times)
        zenith = float(solar_pos["apparent_zenith"].iloc[0])
        azimuth_sun = float(solar_pos["azimuth"].iloc[0])

        if zenith >= 90.0:
            return 0.0

        # Decompose GHI into DNI + DHI via Erbs model
        dhi_raw = cond.get("hourly_diffuse_radiation", [])
        hour = now.hour
        if dhi_raw and hour < len(dhi_raw) and dhi_raw[hour] is not None:
            dhi = float(dhi_raw[hour])
            cos_z = math.cos(math.radians(zenith))
            dni = max(0.0, (ghi - dhi) / cos_z) if cos_z > 0.01 else 0.0
        else:
            erbs = pvlib_irradiance.erbs(ghi, zenith, datetime_or_doy=now.timetuple().tm_yday)
            dni = float(erbs["dni"])
            dhi = float(erbs["dhi"])

        poa_components = pvlib_irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            solar_zenith=zenith,
            solar_azimuth=azimuth_sun,
            dni=dni,
            ghi=ghi,
            dhi=dhi,
            model="perez",
        )
        return max(0.0, float(poa_components["poa_global"]))
    except Exception as exc:
        log.debug("pvlib POA calculation failed (%s); using simple model.", exc)
        return _simple_poa(ghi, tilt)


def _compute_iam(tilt: float, iam_model: str) -> float:
    """
    Compute the incidence-angle modifier (IAM) for the panel tilt angle.

    Uses simplified analytical expressions matching common pvlib models.

    Parameters
    ----------
    tilt:
        Panel tilt from horizontal (degrees).
    iam_model:
        One of ``"physical"``, ``"ashrae"``, ``"martin_ruiz"``.

    Returns
    -------
    float
        IAM correction factor in [0, 1].
    """
    aoi = tilt  # Simplified: treat tilt as AOI for direct beam at solar noon
    aoi_rad = math.radians(min(aoi, 89.9))

    model = (iam_model or "physical").lower()
    try:
        if model == "ashrae":
            b = 0.05
            return max(0.0, 1.0 - b * (1.0 / math.cos(aoi_rad) - 1.0))
        elif model == "martin_ruiz":
            ar = 0.16
            return 1.0 - math.exp(-math.cos(aoi_rad) / ar)
        else:  # physical / default
            n1, n2 = 1.0, 1.526  # air / glass
            sin_r = (n1 / n2) * math.sin(aoi_rad)
            if abs(sin_r) >= 1.0:
                return 0.0
            r_rad = math.asin(sin_r)
            if math.sin(aoi_rad + r_rad) == 0 or math.sin(aoi_rad - r_rad) == 0:
                return 1.0
            tau_r = 1.0 - 0.5 * (
                (math.sin(aoi_rad - r_rad) / math.sin(aoi_rad + r_rad)) ** 2
                + (math.tan(aoi_rad - r_rad) / math.tan(aoi_rad + r_rad)) ** 2
            )
            return max(0.0, min(1.0, tau_r))
    except (ValueError, ZeroDivisionError):
        return 1.0
