"""
Hybrid all-sky irradiance model.

Combines the deterministic physics Kt with the AI-trained Kt using a
configurable blending weight α:

    Kt_hybrid = α × Kt_phys + (1 − α) × Kt_ai

The physics component (α) provides structural guarantees (physical bounds,
correct limiting behaviour at clear/overcast extremes).  The AI component
(1−α) captures residual patterns not explained by the analytical model:
sub-grid cloud inhomogeneity, aerosol absorption not in climatological AOD,
and sensor-specific biases absorbed during training.

After training, it is good practice to evaluate MAE vs a held-out year and
tune α accordingly (default 0.40 tends to outperform either component alone).

Entry point
-----------
    from solar_forecast.allsky.hybrid_model import AllSkyModel
    model = AllSkyModel(cfg)
    model.load_kt_model()
    forecast = model.forecast(times, atmo_df, clearsky_df)
"""

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
    Orchestrates the physics + AI hybrid Kt model and returns complete
    all-sky irradiance DataFrames.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.alpha = float(cfg["model"].get("physics_weight", 0.40))
        self._trainer = KtTrainer(cfg)
        self._ai_ready = False

    def load_kt_model(self, path: str | Path | None = None) -> None:
        """Load the pre-trained XGBoost Kt model from disk."""
        try:
            self._trainer.load(path)
            self._ai_ready = True
            logger.info("AI Kt model loaded (α=%.2f physics, %.2f AI).",
                        self.alpha, 1 - self.alpha)
        except FileNotFoundError:
            logger.warning("No trained Kt model found — using physics-only mode.")
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
        times       : UTC timestamps to forecast (1-min or hourly)
        atmo_df     : Atmospheric features indexed by timestamp
                      (cloud_cover, aod_550nm, precipitable_water, …)
        clearsky_df : Output of clearsky.spectrl2_model.compute_clearsky
                      (ghi_clear, dni_clear, dhi_clear, cos_zenith, airmass)

        Returns
        -------
        DataFrame with columns:
            kt, ghi, dni, dhi, poa_clear (as reference)
        All indexed by `times`.
        """
        # Align inputs on requested timestamps
        atmo = atmo_df.reindex(times, method="nearest", tolerance="31min")
        cs   = clearsky_df.reindex(times, method="nearest", tolerance="31min")

        # Cloud optical depth
        if "cloud_optical_depth" in atmo.columns:
            cod = atmo["cloud_optical_depth"].values
        else:
            cod = estimate_cod_from_cover(atmo["cloud_cover"].fillna(0).values)

        # --- Physics Kt ---
        kt_phys = compute_physics_kt(
            cloud_cover=atmo["cloud_cover"].fillna(0).values,
            cloud_optical_depth=cod,
            cos_zenith=cs["cos_zenith"].values,
            airmass=cs["airmass"].values,
            aod_550nm=atmo["aod_550nm"].fillna(0.10).values,
            ghi_clear=cs["ghi_clear"].values,
            dni_clear=cs["dni_clear"].values,
            dhi_clear=cs["dhi_clear"].values,
        )

        # --- AI Kt ---
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
            kt_ai = kt_phys   # fallback: physics only

        # --- Blend ---
        alpha = self.alpha
        kt_raw = np.where(
            np.isnan(kt_phys),
            kt_ai,
            alpha * kt_phys + (1.0 - alpha) * kt_ai,
        )
        kt = np.clip(kt_raw, 0.0, 1.05)

        # --- All-sky GHI ---
        ghi = kt_to_allsky_ghi(kt, cs["ghi_clear"].values)

        # --- Decompose to DNI and DHI ---
        dni, dhi = decompose_allsky(
            ghi_all=ghi,
            ghi_clear=cs["ghi_clear"].values,
            dni_clear=cs["dni_clear"].values,
            dhi_clear=cs["dhi_clear"].values,
            cos_zenith=cs["cos_zenith"].values,
        )

        result = pd.DataFrame({
            "kt":        kt,
            "ghi":       ghi,
            "dni":       dni,
            "dhi":       dhi,
            "ghi_clear": cs["ghi_clear"].values,
            "poa_clear": cs["poa_clear"].values,
            "zenith":    cs["zenith"].values,
            "cos_zenith": cs["cos_zenith"].values,
        }, index=times)

        return result
