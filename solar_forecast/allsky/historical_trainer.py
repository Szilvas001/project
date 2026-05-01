"""
Historical-data AI module for all-sky GHI prediction.

Goal
----
Learn the mapping
    GHI_obs = f(GHI_clear, cloud_cover, …)
from historical (CAMS / Open-Meteo) records. The model is intentionally
small and CPU-friendly: gradient-boosted trees over a handful of physically
motivated features.

It is independent of the larger XGBoost Kt trainer in `ai_trainer.py`:
that one targets the clearness index Kt and uses 21 atmospheric features;
this one targets GHI directly and works with as little as
`(GHI_clear, cloud_cover)` so it can be trained from Open-Meteo history
alone.

Performance contract
--------------------
On the validation split, the trained model must satisfy:

    R²   ≥ 0.75
    RMSE ≤ 10 %  (relative to clear-sky GHI peak)

These thresholds are enforced by `train_and_validate()`; failure raises
`AccuracyTargetNotMet`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


CORE_FEATURES = [
    "ghi_clear",
    "cloud_cover",
    "cloud_cover_low",
    "cos_zenith",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]


# ── Errors ────────────────────────────────────────────────────────────────

class AccuracyTargetNotMet(RuntimeError):
    """Raised when validation R² / RMSE fails the contract."""


# ── Result containers ────────────────────────────────────────────────────

@dataclass
class TrainingResult:
    r2: float
    rmse: float
    rmse_relative: float
    mae: float
    n_train: int
    n_val: int
    feature_importance: dict[str, float] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)

    def meets_contract(self, r2_min: float = 0.75, rmse_rel_max: float = 0.10) -> bool:
        return self.r2 >= r2_min and self.rmse_relative <= rmse_rel_max


# ── Feature engineering ──────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive cyclic time encodings + ensure required columns exist.

    Expects an hourly UTC-indexed frame with at minimum
    `ghi_clear` and `cloud_cover`.
    """
    if "ghi_clear" not in df or "cloud_cover" not in df:
        raise ValueError("input frame must contain 'ghi_clear' and 'cloud_cover'")

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)

    hour = out.index.hour + out.index.minute / 60.0
    doy  = out.index.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["doy_sin"]  = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"]  = np.cos(2 * np.pi * doy / 365.25)

    if "cos_zenith" not in out.columns:
        # Approx daylight envelope as fallback when solar geometry is missing.
        out["cos_zenith"] = np.clip(out["ghi_clear"] / out["ghi_clear"].max(), 0, 1).fillna(0)
    if "cloud_cover_low" not in out.columns:
        out["cloud_cover_low"] = out["cloud_cover"]

    return out


# ── Trainer ──────────────────────────────────────────────────────────────

class HistoricalGHITrainer:
    """Gradient-boosted regressor mapping (GHI_clear, cloud, …) → GHI_obs.

    Tries XGBoost first, falls back to sklearn's `HistGradientBoostingRegressor`
    so that test environments without xgboost still work.
    """

    def __init__(
        self,
        features: Optional[list[str]] = None,
        target: str = "ghi_obs",
        n_estimators: int = 400,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.features = list(features) if features else list(CORE_FEATURES)
        self.target = target
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.model: object | None = None
        self._impl: str = "uninitialized"

    # ---- core fit/predict -------------------------------------------------

    def _new_model(self):
        try:
            from xgboost import XGBRegressor
            self._impl = "xgboost"
            return XGBRegressor(
                objective="reg:squarederror",
                n_estimators=self.params["n_estimators"],
                max_depth=self.params["max_depth"],
                learning_rate=self.params["learning_rate"],
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=self.params["random_state"],
                tree_method="hist",
                verbosity=0,
            )
        except Exception:
            from sklearn.ensemble import HistGradientBoostingRegressor
            self._impl = "sklearn"
            return HistGradientBoostingRegressor(
                max_iter=self.params["n_estimators"],
                max_depth=self.params["max_depth"],
                learning_rate=self.params["learning_rate"],
                random_state=self.params["random_state"],
            )

    def fit(self, df: pd.DataFrame) -> "HistoricalGHITrainer":
        df = build_features(df)
        if self.target not in df.columns:
            raise ValueError(f"target column '{self.target}' missing")

        feats = [c for c in self.features if c in df.columns]
        if not feats:
            raise ValueError("no usable feature columns found")

        X = df[feats].astype(float).values
        y = df[self.target].astype(float).values

        self.model = self._new_model()
        self.model.fit(X, y)
        self._fitted_features = feats
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model not fitted")
        df = build_features(df)
        feats = getattr(self, "_fitted_features", self.features)
        X = df[feats].astype(float).values
        return np.clip(self.model.predict(X), 0, None)

    # ---- training entry --------------------------------------------------

    def train_and_validate(
        self,
        df: pd.DataFrame,
        val_fraction: float = 0.25,
        r2_min: float = 0.75,
        rmse_rel_max: float = 0.10,
        enforce: bool = True,
    ) -> TrainingResult:
        """Fit on a random training split and validate on a holdout.

        Raises `AccuracyTargetNotMet` if the holdout metrics fail the
        contract (R² ≥ r2_min and RMSE/peak ≤ rmse_rel_max), unless
        `enforce=False`.
        """
        df = build_features(df)
        rng = np.random.default_rng(self.params["random_state"])
        idx = rng.permutation(len(df))
        n_val = max(1, int(len(df) * val_fraction))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]

        self.fit(train_df)
        y_pred = self.predict(val_df)
        y_true = val_df[self.target].astype(float).values

        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) or 1.0
        r2 = 1.0 - ss_res / ss_tot
        rmse = float(np.sqrt(ss_res / max(len(y_true), 1)))
        peak = float(np.max(np.abs(y_true))) or 1.0
        rmse_rel = rmse / peak
        mae = float(np.mean(np.abs(y_true - y_pred)))

        importance: dict[str, float] = {}
        feats = getattr(self, "_fitted_features", self.features)
        try:
            if self._impl == "xgboost":
                importance = dict(zip(feats, self.model.feature_importances_.tolist()))
            elif hasattr(self.model, "feature_importances_"):
                importance = dict(zip(feats, self.model.feature_importances_.tolist()))
        except Exception:
            pass

        result = TrainingResult(
            r2=r2, rmse=rmse, rmse_relative=rmse_rel, mae=mae,
            n_train=len(train_df), n_val=len(val_df),
            feature_importance=importance,
            feature_columns=feats,
        )

        log.info(
            "historical-trainer (%s) — R²=%.3f RMSE=%.1f W/m² (%.1f%% of peak)",
            self._impl, r2, rmse, 100 * rmse_rel,
        )

        if enforce and not result.meets_contract(r2_min, rmse_rel_max):
            raise AccuracyTargetNotMet(
                f"R²={r2:.3f} (≥{r2_min}) RMSE_rel={rmse_rel:.3f} (≤{rmse_rel_max})"
            )
        return result

    # ---- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("nothing to save — call fit() first")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "impl": self._impl,
            "features": getattr(self, "_fitted_features", self.features),
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "HistoricalGHITrainer":
        bundle = joblib.load(path)
        obj = cls(features=bundle["features"])
        obj.model = bundle["model"]
        obj._impl = bundle["impl"]
        obj._fitted_features = bundle["features"]
        return obj


# ── Synthetic data generation (used by tests + offline training) ─────────

def synthesize_training_data(
    n_days: int = 60,
    lat: float = 47.5,
    seed: int = 42,
    noise_std_rel: float = 0.05,
) -> pd.DataFrame:
    """Build a self-consistent synthetic training set of hourly records.

    The clear-sky GHI follows a simple Hottel-style daily curve (sufficient
    to exercise the regressor); cloud cover is sampled from a Beta
    distribution and applied via Kt = 1 − 0.75·cloud (with Gaussian noise).

    Use this for unit tests or for warming up the model before any CAMS data
    is available. Real training should use Open-Meteo history.
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=n_days * 24, freq="h", tz="UTC")
    hour = times.hour + times.minute / 60.0
    doy  = times.dayofyear

    decl = 23.45 * np.sin(np.deg2rad(360 / 365 * (doy - 81)))
    hra  = 15.0 * (hour - 12.0)
    sin_alt = (
        np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(decl))
        + np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(decl)) * np.cos(np.deg2rad(hra))
    )
    cos_zenith = np.clip(sin_alt, 0, 1)
    ghi_clear = 1100.0 * cos_zenith ** 1.15  # rough clear-sky envelope

    cloud = rng.beta(1.5, 3.0, size=len(times))
    cloud_low = np.clip(cloud + rng.normal(0, 0.05, len(times)), 0, 1)
    kt = 1.0 - 0.75 * cloud + rng.normal(0, noise_std_rel, len(times))
    kt = np.clip(kt, 0.05, 1.05)
    ghi_obs = ghi_clear * kt

    return pd.DataFrame({
        "ghi_clear":       ghi_clear,
        "cloud_cover":     cloud,
        "cloud_cover_low": cloud_low,
        "cos_zenith":      cos_zenith,
        "ghi_obs":         np.clip(ghi_obs, 0, None),
    }, index=times)


__all__ = [
    "CORE_FEATURES",
    "HistoricalGHITrainer",
    "TrainingResult",
    "AccuracyTargetNotMet",
    "build_features",
    "synthesize_training_data",
]
