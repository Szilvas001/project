"""
PV power output model.

Converts all-sky plane-of-array (POA) irradiance → AC power (kW).

Pipeline
--------
    POA_incident                   (from transposition model)
        → spectral correction MM   (SpectralResponse)
        → IAM correction           (iam_model.iam_ashrae / martin_ruiz)
        → G_eff                    effective irradiance (W/m²)
        → cell temperature T_cell  (NOCT model, IEC 61215)
        → P_dc                     DC power with temperature derating
        → system losses            (wiring, soiling)
        → P_ac                     AC power via inverter efficiency

Temperature model (NOCT)
------------------------
    T_cell = T_air + (NOCT − 20) / 800 × G_POA

    Valid for free-standing roof/field systems.  BIPV or insulated mounts
    should use a higher effective NOCT (50–60 °C).

Transposition (GHI + DNI + DHI → POA)
---------------------------------------
    Uses pvlib.irradiance.get_total_irradiance with the Perez anisotropic
    sky model for diffuse, as it performs best under partly cloudy skies.
"""

import logging

import numpy as np
import pandas as pd
import pvlib

from .iam_model import iam_ashrae, iam_martin_ruiz, iam_diffuse
from .spectral_response import SpectralResponse

logger = logging.getLogger(__name__)

_G_STC = 1000.0    # W/m²  standard test condition irradiance
_T_STC = 25.0      # °C    STC temperature


class PVOutputModel:
    """
    Converts all-sky irradiance DataFrame to AC power timeseries.

    Parameters
    ----------
    cfg         : System config dict (see config.yaml)
    sr_csv      : Optional path to custom spectral response CSV
    iam_model   : 'ashrae' | 'martin_ruiz' | 'fresnel'
    """

    def __init__(
        self,
        cfg: dict,
        sr_csv: str | None = None,
        iam_model: str = "ashrae",
    ):
        sys = cfg["system"]
        self.capacity_kw  = float(sys["capacity_kw"])
        self.efficiency   = float(sys.get("module_efficiency", 0.205))
        self.gamma        = float(sys.get("temperature_coefficient", -0.0040))
        self.noct         = float(sys.get("noct", 44.0))
        self.albedo       = float(sys.get("ground_albedo", 0.20))
        self.inv_eff      = float(sys.get("inverter_efficiency", 0.97))
        self.wiring_loss  = float(sys.get("wiring_loss", 0.02))
        self.soiling_loss = float(sys.get("soiling_loss", 0.02))
        self.iam_type     = iam_model

        from solar_forecast.utils import resolve_tilt_azimuth
        self.tilt, self.azimuth = resolve_tilt_azimuth(cfg)

        self.sr = SpectralResponse(sr_csv)
        self._iam_diffuse = iam_diffuse(self.tilt, model=iam_model)

    def run(
        self,
        allsky_df: pd.DataFrame,
        temperature: pd.Series,
        wind_speed: pd.Series | None = None,
        solar_pos_df: pd.DataFrame | None = None,
        lat: float | None = None,
        lon: float | None = None,
        altitude: float = 0.0,
    ) -> pd.DataFrame:
        """
        Compute hourly AC power from all-sky irradiance.

        Parameters
        ----------
        allsky_df    : DataFrame with ghi, dni, dhi, zenith, cos_zenith, poa_clear
        temperature  : Air temperature at 2 m (°C), aligned to allsky_df.index
        wind_speed   : Wind speed at 10 m (m/s); optional, used for advanced NOCT
        solar_pos_df : Precomputed solar position; computed internally if None
        lat, lon     : Required if solar_pos_df is None
        altitude     : m above sea level

        Returns
        -------
        DataFrame with power_kw, g_eff, t_cell, kt, ghi, poa, iam_beam, iam_diff
        """
        times = allsky_df.index

        # Solar position (for transposition)
        if solar_pos_df is None:
            if lat is None:
                raise ValueError("Provide lat/lon or solar_pos_df.")
            loc = pvlib.location.Location(lat, lon, "UTC", altitude)
            solar_pos_df = loc.get_solarposition(times)

        # --- Transposition: GHI + DNI + DHI → POA (Perez model) ---
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=self.tilt,
            surface_azimuth=self.azimuth,
            solar_zenith=solar_pos_df["apparent_zenith"],
            solar_azimuth=solar_pos_df["azimuth"],
            dni=allsky_df["dni"].fillna(0),
            ghi=allsky_df["ghi"].fillna(0),
            dhi=allsky_df["dhi"].fillna(0),
            dni_extra=pvlib.irradiance.get_extra_radiation(times),
            airmass=pvlib.atmosphere.get_relative_airmass(
                solar_pos_df["apparent_zenith"], model="kastenyoung1989"
            ),
            model="perez",
            albedo=self.albedo,
        )

        poa_beam  = poa["poa_direct"].fillna(0).clip(0)
        poa_diff  = poa["poa_diffuse"].fillna(0).clip(0)
        poa_total = poa["poa_global"].fillna(0).clip(0)

        # --- Incidence Angle Modifier ---
        aoi = pvlib.irradiance.aoi(
            self.tilt, self.azimuth,
            solar_pos_df["apparent_zenith"],
            solar_pos_df["azimuth"],
        ).fillna(90.0)

        if self.iam_type == "martin_ruiz":
            iam_b = iam_martin_ruiz(aoi.values)
        else:
            iam_b = iam_ashrae(aoi.values)

        iam_d = self._iam_diffuse   # scalar for diffuse

        poa_beam_eff = poa_beam.values * iam_b
        poa_diff_eff = poa_diff.values * iam_d
        poa_eff      = poa_beam_eff + poa_diff_eff

        # --- Cell Temperature (NOCT model) ---
        t_air = temperature.reindex(times, method="nearest", tolerance="31min").values
        ws    = (wind_speed.reindex(times, method="nearest", tolerance="31min").values
                 if wind_speed is not None else np.full(len(times), 1.0))

        # NOCT adjusted for wind (wind reduces cell temperature)
        noct_adj = self.noct - 0.5 * np.clip(ws - 1.0, 0, 10)
        t_cell = t_air + (noct_adj - 20.0) / 800.0 * poa_total.values

        # --- Temperature derating ---
        eta_T = 1.0 + self.gamma * (t_cell - _T_STC)
        eta_T = np.clip(eta_T, 0.60, 1.10)

        # --- DC power ---
        # P_dc = P_stc × (G_eff / G_stc) × η_T
        p_dc = self.capacity_kw * (poa_eff / _G_STC) * eta_T

        # System losses
        p_dc *= (1.0 - self.wiring_loss) * (1.0 - self.soiling_loss)

        # --- AC power ---
        p_ac = np.maximum(0.0, p_dc * self.inv_eff)

        result = pd.DataFrame({
            "power_kw":  p_ac,
            "poa":       poa_total.values,
            "g_eff":     poa_eff,
            "t_cell":    t_cell,
            "iam_beam":  iam_b,
            "ghi":       allsky_df["ghi"].values,
            "kt":        allsky_df.get("kt", pd.Series(np.nan, index=times)).values,
        }, index=times)

        return result

    def run_from_live(
        self,
        allsky_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        lat: float,
        lon: float,
        altitude: float = 0.0,
    ) -> pd.DataFrame:
        """
        Convenience wrapper for real-time forecasting from Open-Meteo weather.

        Extracts temperature and wind from weather_df and delegates to run().
        """
        temperature = weather_df.get(
            "temperature",
            pd.Series(15.0, index=weather_df.index),
        )
        wind_speed = weather_df.get(
            "wind_speed",
            pd.Series(2.0, index=weather_df.index),
        )
        return self.run(
            allsky_df=allsky_df,
            temperature=temperature,
            wind_speed=wind_speed,
            lat=lat,
            lon=lon,
            altitude=altitude,
        )
