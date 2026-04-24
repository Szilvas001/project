"""
Hybrid all-sky irradiance model.

Combines the physics-based Kt with the AI Kt using a blending weight α:

    Kt_hybrid = α × Kt_phys + (1 − α) × Kt_ai

Physics component (α) ensures:
  - Physical bounds (0 ≤ Kt ≤ 1.05)
  - Correct clear/overcast limiting behaviour
  - Proper aerosol attenuation via SSA/GG from CAMS

AI component (1−α) corrects for:
  - Sub-grid cloud inhomogeneity
  - Aerosol absorption not in the background clear-sky reference
  - Sensor-specific or seasonal biases

When the AI model is not available (no training data), the system falls back
to physics-only mode (α = 1.0).

CAMS vs Open-Meteo live mode
----------------------------
CAMS historical data (SSA, GG, multi-wavelength AOD) feeds the physics model
with full accuracy.  In live mode (Open-Meteo), the physics model falls back
to climatological SSA=0.92 and G=0.65.  The AI model absorbs residual error.

UTC note: all timestamps and indices must be UTC-aware.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .physics_kt import (
    compute_physics_kt,
    estimate_cod_from_cover,
    kt_to_allsky_ghi,
    decompose_allsky,
)
from .ai_trainer import KtTrainer

logger = logging.getLogger(__name__)


class AllSkyModel:
    """
    Orchestrates the physics + AI hybrid Kt model.

    Returns complete all-sky irradiance DataFrames for a sequence of UTC
    timestamps.
    """

    def __init__(self, cfg: dict):
        self.cfg     = cfg
        self.alpha   = float(cfg["model"].get("physics_weight", 0.40))
        self._trainer = KtTrainer(cfg)
        self._ai_ready = False

    def load_kt_model(self, path: str | Path | None = None) -> None:
        """Load the pre-trained XGBoost Kt model from disk."""
        try:
            self._trainer.load(path)
            self._ai_ready = True
            logger.info("AI Kt model loaded (α=%.2f phys, %.2f AI).",
                        self.alpha, 1.0 - self.alpha)
        except FileNotFoundError:
            logger.warning("No trained Kt model found — physics-only mode.")
            self._ai_ready = False

    def forecast(
        self,
        times: pd.DatetimeIndex,
        atmo_df: pd.DataFrame,
        clearsky_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Produce all-sky irradiance forecast.

        Parameters
        ----------
        times       : UTC timestamps to forecast (hourly)
        atmo_df     : Atmospheric features (UTC index)
                      Columns: cloud_cover, aod_550nm, precipitable_water,
                      ssa_550nm, asymmetry_factor, angstrom_alpha1/2, pm25, …
        clearsky_df : Output of clearsky.spectrl2_model.compute_clearsky
                      Columns: ghi_clear, dni_clear, dhi_clear, cos_zenith,
                      airmass, poa_clear, zenith, …

        Returns
        -------
        DataFrame indexed by `times` with columns:
            kt, ghi, dni, dhi, ghi_clear, poa_clear, zenith, cos_zenith
        """
        # Align to requested timestamps
        atmo = atmo_df.reindex(times, method="nearest", tolerance="31min")
        cs   = clearsky_df.reindex(times, method="nearest", tolerance="31min")

        # Cloud optical depth
        if "cloud_optical_depth" in atmo.columns:
            cod = atmo["cloud_optical_depth"].fillna(0).values
        else:
            cod = estimate_cod_from_cover(atmo["cloud_cover"].fillna(0).values)

        # SSA and asymmetry from CAMS (or defaults for live mode)
        ssa       = atmo.get("ssa_550nm",      pd.Series(0.92, index=atmo.index)).fillna(0.92).values
        asymmetry = atmo.get("asymmetry_factor", pd.Series(0.65, index=atmo.index)).fillna(0.65).values

        # ── Physics Kt ────────────────────────────────────────────────────
        kt_phys = compute_physics_kt(
            cloud_cover        =atmo["cloud_cover"].fillna(0).values,
            cloud_optical_depth=cod,
            cos_zenith         =cs["cos_zenith"].values,
            airmass            =cs["airmass"].values,
            aod_550nm          =atmo.get("aod_550nm", pd.Series(0.1, index=atmo.index)).fillna(0.1).values,
            ghi_clear          =cs["ghi_clear"].values,
            dni_clear          =cs["dni_clear"].values,
            dhi_clear          =cs["dhi_clear"].values,
            ssa                =ssa,
            asymmetry          =asymmetry,
        )

        # ── AI Kt ─────────────────────────────────────────────────────────
        if self._ai_ready:
            feature_df = atmo.copy()
            feature_df["cos_zenith"] = cs["cos_zenith"].values
            feature_df["airmass"]    = cs["airmass"].values
            feature_df["ghi_clear"]  = cs["ghi_clear"].values
            feature_df["dni_clear"]  = cs["dni_clear"].values
            feature_df["dhi_clear"]  = cs["dhi_clear"].values
            feature_df["Kt_phys"]    = kt_phys
            kt_ai = self._trainer.predict(feature_df)
        else:
            kt_ai = kt_phys   # physics-only fallback

        # ── Blend ─────────────────────────────────────────────────────────
        alpha   = self.alpha
        kt_raw  = np.where(
            np.isnan(kt_phys),
            kt_ai,
            alpha * kt_phys + (1.0 - alpha) * kt_ai,
        )
        kt = np.clip(kt_raw, 0.0, 1.05)

        # ── All-sky GHI ───────────────────────────────────────────────────
        ghi = kt_to_allsky_ghi(kt, cs["ghi_clear"].values)

        # ── Decompose → DNI, DHI ──────────────────────────────────────────
        dni, dhi = decompose_allsky(
            ghi_all   =ghi,
            ghi_clear =cs["ghi_clear"].values,
            dni_clear =cs["dni_clear"].values,
            dhi_clear =cs["dhi_clear"].values,
            cos_zenith=cs["cos_zenith"].values,
        )

        result = pd.DataFrame({
            "kt":         kt,
            "ghi":        ghi,
            "dni":        dni,
            "dhi":        dhi,
            "ghi_clear":  cs["ghi_clear"].values,
            "poa_clear":  cs["poa_clear"].values,
            "zenith":     cs["zenith"].values,
            "cos_zenith": cs["cos_zenith"].values,
        }, index=times)

        return result
