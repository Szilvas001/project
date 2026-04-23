"""
XGBoost-based Kt (clearness index) trainer.

The model learns the mapping:
    Kt = f(cloud_cover, cloud_optical_depth, aod_550nm, precipitable_water,
            total_ozone, surface_pressure, cos_zenith, airmass,
            hour_sin, hour_cos, doy_sin, doy_cos, Kt_phys)

`Kt_phys` (from physics_kt.py) is included as a feature so the AI learns
residuals on top of the physics model, preserving physical constraints while
correcting systematic biases from real atmospheric observations.

Training data: merged CAMS atmospheric data (features) and CAMS radiation
service (target Kt) for the configured historical window.

Usage
-----
    from solar_forecast.allsky.ai_trainer import KtTrainer
    trainer = KtTrainer(cfg)
    trainer.train(df_features, df_radiation)
    trainer.save("models/kt_xgb.joblib")
    kt_pred = trainer.predict(df_features)
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

_FEATURE_COLS = [
    "cloud_cover",
    "cloud_optical_depth",
    "log_aod",            # log(aod + 0.01)
    "precipitable_water",
    "total_ozone_norm",   # normalised
    "surface_pressure_norm",
    "cos_zenith",
    "log_airmass",        # log(airmass)
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "Kt_phys",
]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer feature matrix from merged atmospheric + solar position data.

    All transformations are monotone or bounded, preventing leakage of
    target information through feature construction.
    """
    out = pd.DataFrame(index=df.index)

    out["cloud_cover"] = df["cloud_cover"].clip(0, 1)
    out["cloud_optical_depth"] = np.log1p(df["cloud_optical_depth"].clip(0, 100))
    out["log_aod"] = np.log(df["aod_550nm"].clip(0.005, 5.0) + 0.01)
    out["precipitable_water"] = df["precipitable_water"].clip(0, 10)
    out["total_ozone_norm"] = (df.get("total_ozone", 310.0) - 300.0) / 100.0
    out["surface_pressure_norm"] = (df.get("surface_pressure", 1013.25) - 1013.25) / 50.0
    out["cos_zenith"] = df["cos_zenith"].clip(0, 1)
    out["log_airmass"] = np.log(df["airmass"].clip(1, 40))

    hour_frac = df.index.hour + df.index.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0)
    doy = df.index.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    if "Kt_phys" in df:
        out["Kt_phys"] = df["Kt_phys"].clip(0, 1.05)
    else:
        out["Kt_phys"] = 0.5   # neutral prior if physics Kt not available

    return out


class KtTrainer:
    """
    Trains an XGBoost regressor that predicts Kt from atmospheric features.

    The pipeline is: features → RobustScaler → XGBRegressor.
    RobustScaler handles outliers in AOD and precipitable_water.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_path = Path(cfg["model"]["kt_model_path"])
        self.min_samples = cfg["model"].get("min_train_samples", 500)
        self.pipeline: Pipeline | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def build_training_set(
        self,
        df_atmo: pd.DataFrame,
        df_radiation: pd.DataFrame,
        df_clearsky: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge atmospheric features, clear-sky, and CAMS radiation into one
        training-ready DataFrame with target column `Kt_target`.

        Kt_target = GHI_all_sky / GHI_clear  (from CAMS radiation service)
        """
        # Align on common timestamps
        idx = df_atmo.index.intersection(df_radiation.index).intersection(df_clearsky.index)
        if len(idx) == 0:
            raise ValueError("No overlapping timestamps in training data.")

        df = df_atmo.loc[idx].copy()
        df["ghi_clear"]  = df_clearsky.loc[idx, "ghi_clear"].values
        df["dni_clear"]  = df_clearsky.loc[idx, "dni_clear"].values
        df["dhi_clear"]  = df_clearsky.loc[idx, "dhi_clear"].values
        df["cos_zenith"] = df_clearsky.loc[idx, "cos_zenith"].values
        df["airmass"]    = df_clearsky.loc[idx, "airmass"].values

        df["ghi_obs"] = df_radiation.loc[idx, "ghi"].values
        df["ghi_cs"]  = df_radiation.loc[idx, "ghi_clear"].values   # CAMS McClear

        # Target: Kt from CAMS radiation service (ground truth)
        with np.errstate(invalid="ignore", divide="ignore"):
            df["Kt_target"] = np.where(
                df["ghi_cs"] > 5.0,
                (df["ghi_obs"] / df["ghi_cs"]).clip(0.0, 1.1),
                np.nan,
            )

        # Physics Kt as a feature
        from .physics_kt import compute_physics_kt, estimate_cod_from_cover
        cod = df.get("cloud_optical_depth",
                     pd.Series(estimate_cod_from_cover(df["cloud_cover"].values),
                                index=df.index))
        df["Kt_phys"] = compute_physics_kt(
            cloud_cover=df["cloud_cover"].values,
            cloud_optical_depth=cod.values,
            cos_zenith=df["cos_zenith"].values,
            airmass=df["airmass"].values,
            aod_550nm=df["aod_550nm"].values,
            ghi_clear=df["ghi_clear"].values,
            dni_clear=df["dni_clear"].values,
            dhi_clear=df["dhi_clear"].values,
        )

        # Drop night time and bad rows
        df = df[df["ghi_clear"] > 10.0]
        df = df.dropna(subset=["Kt_target", "Kt_phys"])

        return df

    def train(self, df_train: pd.DataFrame) -> dict:
        """
        Fit the XGBoost pipeline on training data.

        `df_train` must be the output of `build_training_set`.

        Returns dict with training metrics.
        """
        import xgboost as xgb

        if len(df_train) < self.min_samples:
            raise ValueError(
                f"Only {len(df_train)} training samples; need ≥ {self.min_samples}. "
                "Download more CAMS data or reduce min_train_samples."
            )

        X = _build_features(df_train)[_FEATURE_COLS].values
        y = df_train["Kt_target"].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42, shuffle=True
        )

        xgb_model = xgb.XGBRegressor(
            n_estimators=600,
            learning_rate=0.04,
            max_depth=6,
            subsample=0.80,
            colsample_bytree=0.80,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            eval_metric="mae",
            early_stopping_rounds=40,
        )

        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)

        xgb_model.fit(
            X_train_s, y_train,
            eval_set=[(X_val_s, y_val)],
            verbose=False,
        )

        self.pipeline = {"scaler": scaler, "model": xgb_model}

        y_pred = xgb_model.predict(X_val_s)
        metrics = {
            "n_train": len(X_train),
            "n_val":   len(X_val),
            "mae":     float(mean_absolute_error(y_val, y_pred)),
            "r2":      float(r2_score(y_val, y_pred)),
            "best_iteration": int(xgb_model.best_iteration),
        }
        logger.info("Kt model trained: MAE=%.4f  R²=%.4f  iters=%d",
                    metrics["mae"], metrics["r2"], metrics["best_iteration"])
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted Kt array for rows in df."""
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        X = _build_features(df)[_FEATURE_COLS].values
        X_s = self.pipeline["scaler"].transform(X)
        return self.pipeline["model"].predict(X_s).clip(0.0, 1.05)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> None:
        path = Path(path or self.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info("Kt model saved to %s", path)

    def load(self, path: str | Path | None = None) -> None:
        path = Path(path or self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"No model at {path}. Run training first.")
        self.pipeline = joblib.load(path)
        logger.info("Kt model loaded from %s", path)
