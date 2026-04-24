"""
PV power output model — full pipeline with spectral correction.

Converts all-sky POA irradiance → AC power (kW).

Pipeline
--------
    POA_beam + POA_diffuse + POA_ground    (from Perez transposition)
        → IAM correction                   (ASHRAE / Martin-Ruiz / Fresnel)
        → Spectral mismatch MM             (per time step, if spectra available)
        → G_eff                            effective irradiance (W/m²)
        → T_cell                           cell temperature (NOCT model)
        → P_dc                             DC power (temperature + irradiance)
        → system losses                    (wiring, soiling)
        → P_ac                             AC power via inverter efficiency

Temperature model (NOCT / IEC 61215)
-------------------------------------
    T_cell = T_air + (NOCT − 20) / 800 × G_POA

Cell temperature has a strong effect on efficiency via the temperature
coefficient γ (typically −0.0035 to −0.005 %/K for c-Si).

Spectral mismatch
------------------
When pvlib spectrl2 spectra are available (return_spectra=True in clearsky
model), the effective irradiance is:

    E_eff = MM × G_POA_broadband

where MM = ∫SR(λ)I(λ)dλ / (∫SR(λ)G_AM15(λ)dλ) normalised by broadband ratio.

For MM computation, the POA spectral irradiance from spectrl2 is used.

UTC / timezone
--------------
All internal timestamps remain UTC.  The `run_from_live` method returns
a DataFrame with UTC-indexed timestamps.  The caller (dashboard) is
responsible for converting to local display time.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pvlib

from .iam_model import iam_ashrae, iam_martin_ruiz, iam_fresnel, iam_diffuse
from .spectral_response import SpectralResponse, TECHNOLOGY_LABELS

logger = logging.getLogger(__name__)

_G_STC   = 1000.0   # W/m²  standard test condition irradiance
_T_STC   = 25.0     # °C


class PVOutputModel:
    """
    Converts all-sky irradiance to AC power timeseries.

    Parameters
    ----------
    cfg        : System config dict (see config.yaml)
    technology : PV cell technology for spectral response
                 ('mono_si', 'poly_si', 'cdte', 'cigs', 'hit')
    sr_csv     : Path to custom SR CSV (overrides `technology`)
    iam_model  : 'ashrae' | 'martin_ruiz' | 'fresnel'
    """

    def __init__(
        self,
        cfg: dict,
        technology: str = "mono_si",
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

        # Spectral response
        self.sr = SpectralResponse(
            technology=technology,
            csv_path=sr_csv,
        )
        logger.info("SR curve: %s", TECHNOLOGY_LABELS.get(self.sr.name, self.sr.name))

        # Diffuse IAM (constant for isotropic sky dome)
        self._iam_diffuse = iam_diffuse(self.tilt, model=iam_model)

    # ──────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────

    def run(
        self,
        allsky_df:    pd.DataFrame,
        temperature:  pd.Series,
        wind_speed:   pd.Series | None = None,
        solar_pos_df: pd.DataFrame | None = None,
        lat: float | None = None,
        lon: float | None = None,
        altitude: float = 0.0,
        spectra_list: list | None = None,
    ) -> pd.DataFrame:
        """
        Compute hourly AC power from all-sky irradiance.

        Parameters
        ----------
        allsky_df    : DataFrame (ghi, dni, dhi, zenith, cos_zenith, poa_clear)
        temperature  : Air temperature (°C), same index as allsky_df
        wind_speed   : Wind speed (m/s); optional
        solar_pos_df : Pre-computed solar positions; computed if None
        lat, lon     : Required if solar_pos_df is None
        altitude     : m above sea level
        spectra_list : List of spectral dicts from spectrl2 (for MM correction)
                       Length must equal len(allsky_df).  None entries → MM=1.

        Returns
        -------
        DataFrame with power_kw, g_eff, t_cell, kt, ghi, poa, mm,
                         iam_beam, iam_diff
        """
        times = allsky_df.index

        # Solar position for transposition
        if solar_pos_df is None:
            if lat is None:
                raise ValueError("Provide lat/lon or solar_pos_df.")
            loc = pvlib.location.Location(lat, lon, "UTC", altitude)
            solar_pos_df = loc.get_solarposition(times)

        # ── Perez transposition: GHI + DNI + DHI → POA ───────────────────
        dni_extra = pvlib.irradiance.get_extra_radiation(times)
        airmass_rel = pvlib.atmosphere.get_relative_airmass(
            solar_pos_df["apparent_zenith"], model="kastenyoung1989"
        )

        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt  =self.tilt,
            surface_azimuth=self.azimuth,
            solar_zenith  =solar_pos_df["apparent_zenith"],
            solar_azimuth =solar_pos_df["azimuth"],
            dni=allsky_df["dni"].fillna(0),
            ghi=allsky_df["ghi"].fillna(0),
            dhi=allsky_df["dhi"].fillna(0),
            dni_extra=dni_extra,
            airmass=airmass_rel,
            model="perez",
            albedo=self.albedo,
        )

        poa_beam  = poa["poa_direct"].fillna(0).clip(lower=0)
        poa_diff  = poa["poa_diffuse"].fillna(0).clip(lower=0)
        poa_total = poa["poa_global"].fillna(0).clip(lower=0)

        # ── Incidence Angle Modifier ──────────────────────────────────────
        aoi = pvlib.irradiance.aoi(
            self.tilt, self.azimuth,
            solar_pos_df["apparent_zenith"],
            solar_pos_df["azimuth"],
        ).fillna(90.0)

        if self.iam_type == "martin_ruiz":
            iam_b = iam_martin_ruiz(aoi.values)
        elif self.iam_type == "fresnel":
            iam_b = iam_fresnel(aoi.values)
        else:
            iam_b = iam_ashrae(aoi.values)

        iam_d = self._iam_diffuse

        poa_beam_eff = poa_beam.values * iam_b
        poa_diff_eff = poa_diff.values * iam_d

        # ── Spectral mismatch correction ──────────────────────────────────
        if spectra_list is not None and len(spectra_list) == len(times):
            mm_arr = self.sr.mismatch_series(spectra_list)
        else:
            mm_arr = np.ones(len(times))

        # Effective irradiance
        g_eff = np.clip((poa_beam_eff + poa_diff_eff) * mm_arr, 0, None)

        # ── Cell temperature (NOCT model) ─────────────────────────────────
        T_air = temperature.reindex(times, method="nearest").fillna(15.0).values
        t_cell = T_air + (self.noct - 20.0) / 800.0 * poa_total.values

        # ── DC power ─────────────────────────────────────────────────────
        # P_dc = capacity × (G_eff / G_STC) × [1 + γ × (T_cell − T_STC)]
        temp_factor = 1.0 + self.gamma * (t_cell - _T_STC)
        temp_factor = np.clip(temp_factor, 0.5, 1.2)

        p_dc = self.capacity_kw * (g_eff / _G_STC) * temp_factor
        p_dc = np.clip(p_dc, 0.0, self.capacity_kw * 1.05)

        # ── System losses ─────────────────────────────────────────────────
        loss_factor = (1.0 - self.wiring_loss) * (1.0 - self.soiling_loss)
        p_ac = p_dc * loss_factor * self.inv_eff
        p_ac = np.clip(p_ac, 0.0, self.capacity_kw * 1.1)

        return pd.DataFrame({
            "power_kw": p_ac,
            "power_dc_kw": p_dc,
            "g_eff": g_eff,
            "t_cell": t_cell,
            "poa": poa_total.values,
            "mm": mm_arr,
            "iam_beam": iam_b,
            "kt": allsky_df.get("kt", pd.Series(np.nan, index=times)).values,
            "ghi": allsky_df["ghi"].values,
        }, index=times)

    def run_from_live(
        self,
        allsky_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        lat: float,
        lon: float,
        altitude: float = 0.0,
    ) -> pd.DataFrame:
        """
        Convenience wrapper for live forecast mode.

        Extracts temperature and wind speed from weather_df and calls run().
        """
        T = weather_df.get("temperature", pd.Series(15.0, index=weather_df.index))
        ws = weather_df.get("wind_speed", None)

        # Check for stored spectra (from clearsky model with return_spectra=True)
        spectra = allsky_df.get("spectra", None)
        if spectra is not None:
            spectra_list = spectra.tolist()
        else:
            spectra_list = None

        return self.run(
            allsky_df=allsky_df,
            temperature=T,
            wind_speed=ws,
            lat=lat, lon=lon, altitude=altitude,
            spectra_list=spectra_list,
        )
