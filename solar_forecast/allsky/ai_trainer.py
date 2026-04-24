"""
XGBoost-based Kt (clearness index) trainer — enhanced feature set.

The model learns the mapping:
    Kt = f(cloud_cover, cloud_optical_depth, aod_550nm, precipitable_water,
            total_ozone, surface_pressure, cos_zenith, airmass,
            hour_sin, hour_cos, doy_sin, doy_cos, Kt_phys,
            ssa, asymmetry, angstrom_alpha1, angstrom_alpha2,
            pm25_log, blh_norm, spectral_mismatch, cloud_composite)

`Kt_phys` (from physics_kt.py) is included as a feature so the AI learns
residuals on top of the physics model, preserving physical constraints.

New features vs. original:
  - ssa, asymmetry_factor    : aerosol absorption/scattering character
  - angstrom_alpha1/2        : spectral AOD slope
  - pm25_log                 : log(PM2.5) — surface pollution proxy
  - blh_norm                 : boundary layer height (normalised)
  - cloud_cover_composite    : 3-level weighted cloud cover
  - cloud_cover_low_frac     : low cloud fraction (most impactful for GHI)

Training target:
  Kt_target = GHI_obs / GHI_clear     (from CAMS radiation service)

De-normalisation:
  GHI_pred = Kt_pred × GHI_clear_live

RMSE minimisation is the default objective (reg:squarederror).

Usage
-----
    trainer = KtTrainer(cfg)
    df_train = trainer.build_training_set(df_atmo, df_radiation, df_clearsky)
    metrics  = trainer.train(df_train)
    trainer.save()
    kt_pred  = trainer.predict(df_features)
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

# ── Feature column definitions ────────────────────────────────────────────

# Core features (always computed)
_FEATURES_CORE = [
    "cloud_cover",
    "cloud_optical_depth_log",    # log1p(COD)
    "log_aod",                    # log(AOD + 0.01)
    "precipitable_water",
    "total_ozone_norm",
    "surface_pressure_norm",
    "cos_zenith",
    "log_airmass",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "Kt_phys",
]

# Extended features from new CAMS variables (used if available)
_FEATURES_EXTENDED = [
    "ssa_norm",                   # (SSA - 0.92) / 0.10
    "asymmetry_norm",             # (g - 0.65) / 0.10
    "angstrom_alpha1_norm",
    "angstrom_alpha2_norm",
    "pm25_log",                   # log(PM2.5 + 1)
    "blh_norm",                   # BLH / 2000
    "cloud_composite",            # 3-level weighted cloud cover
    "cloud_low_frac",             # low cloud cover
]

# All features used during training (subset available at inference)
_FEATURE_COLS = _FEATURES_CORE + _FEATURES_EXTENDED


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer the full feature matrix.

    Missing extended features are filled with neutral values so the model
    remains usable even without CAMS SSA/GG data (Open-Meteo live mode).
    """
    out = pd.DataFrame(index=df.index)

    # ── Core ─────────────────────────────────────────────────────────────
    out["cloud_cover"]           = df.get("cloud_cover", pd.Series(0.0, index=df.index)).clip(0, 1)
    out["cloud_optical_depth_log"] = np.log1p(
        df.get("cloud_optical_depth", pd.Series(0.0, index=df.index)).clip(0, 200)
    )
    aod_raw = df.get("aod_550nm", pd.Series(0.10, index=df.index)).clip(0.005, 5.0)
    out["log_aod"]               = np.log(aod_raw + 0.01)
    out["precipitable_water"]    = df.get("precipitable_water", pd.Series(1.5, index=df.index)).clip(0, 10)
    out["total_ozone_norm"]      = (df.get("total_ozone", pd.Series(310.0, index=df.index)) - 300.0) / 100.0
    out["surface_pressure_norm"] = (
        df.get("surface_pressure", pd.Series(1013.25, index=df.index)) - 1013.25
    ) / 50.0
    out["cos_zenith"]            = df.get("cos_zenith", pd.Series(0.5, index=df.index)).clip(0, 1)
    out["log_airmass"]           = np.log(df.get("airmass", pd.Series(2.0, index=df.index)).clip(1, 40))

    hour_frac = df.index.hour + df.index.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0)
    doy = df.index.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    out["Kt_phys"] = df.get("Kt_phys", pd.Series(0.5, index=df.index)).clip(0, 1.05)

    # ── Extended (CAMS-enriched) ──────────────────────────────────────────
    out["ssa_norm"] = (
        df.get("ssa_550nm", pd.Series(0.92, index=df.index)).clip(0.5, 1.0) - 0.92
    ) / 0.10
    out["asymmetry_norm"] = (
        df.get("asymmetry_factor", pd.Series(0.65, index=df.index)).clip(0.3, 0.9) - 0.65
    ) / 0.10
    out["angstrom_alpha1_norm"] = (
        df.get("angstrom_alpha1", df.get("angstrom_exponent", pd.Series(1.30, index=df.index))).clip(0, 3) - 1.30
    ) / 0.50
    out["angstrom_alpha2_norm"] = (
        df.get("angstrom_alpha2", df.get("angstrom_exponent", pd.Series(1.30, index=df.index))).clip(0, 3) - 1.30
    ) / 0.50

    pm25 = df.get("pm25", pd.Series(10.0, index=df.index)).clip(0, 500)
    out["pm25_log"] = np.log1p(pm25)

    blh = df.get("boundary_layer_height", pd.Series(1000.0, index=df.index)).clip(10, 5000)
    out["blh_norm"] = blh / 2000.0

    # Composite cloud cover (best available)
    if "cloud_cover_composite" in df.columns:
        out["cloud_composite"] = df["cloud_cover_composite"].clip(0, 1)
    else:
        out["cloud_composite"] = out["cloud_cover"]

    if "cloud_cover_low" in df.columns:
        out["cloud_low_frac"] = (df["cloud_cover_low"] / 100.0).clip(0, 1)
    else:
        out["cloud_low_frac"] = out["cloud_cover"] * 0.5   # rough estimate

    return out[_FEATURE_COLS]


class KtTrainer:
    """
    XGBoost Kt model trainer.

    Training pipeline: features → RobustScaler → XGBRegressor.
    """

    def __init__(self, cfg: dict):
        self.cfg        = cfg
        self.model_path = Path(cfg["model"]["kt_model_path"])
        self.min_samples = cfg["model"].get("min_train_samples", 500)
        self.pipeline: dict | None = None

    # ──────────────────────────────────────────────────────────────────────
    # Training data assembly
    # ──────────────────────────────────────────────────────────────────────

    def build_training_set(
        self,
        df_atmo:      pd.DataFrame,
        df_radiation: pd.DataFrame,
        df_clearsky:  pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge atmospheric features, clear-sky reference, and CAMS radiation.

        Target: Kt_target = GHI_all_sky / GHI_clear  (CAMS radiation service).

        Returns a training-ready DataFrame with `Kt_target` column.
        """
        idx = (df_atmo.index
               .intersection(df_radiation.index)
               .intersection(df_clearsky.index))
        if len(idx) == 0:
            raise ValueError("No overlapping timestamps in training data.")

        df = df_atmo.loc[idx].copy()
        for col in ["ghi_clear", "dni_clear", "dhi_clear", "cos_zenith", "airmass"]:
            if col in df_clearsky.columns:
                df[col] = df_clearsky.loc[idx, col].values

        df["ghi_obs"] = df_radiation.loc[idx, "ghi"].values
        df["ghi_cs"]  = df_radiation.loc[idx, "ghi_clear"].values   # CAMS McClear

        # Target Kt (from CAMS radiation service — high quality clear-sky ref)
        with np.errstate(invalid="ignore", divide="ignore"):
            df["Kt_target"] = np.where(
                df["ghi_cs"] > 5.0,
                np.clip(df["ghi_obs"] / df["ghi_cs"], 0.0, 1.1),
                np.nan,
            )

        # Physics Kt as feature
        from .physics_kt import compute_physics_kt, estimate_cod_from_cover
        cod = df.get("cloud_optical_depth",
                     pd.Series(estimate_cod_from_cover(df["cloud_cover"].fillna(0).values),
                               index=df.index))
        df["Kt_phys"] = compute_physics_kt(
            cloud_cover        =df["cloud_cover"].fillna(0).values,
            cloud_optical_depth=cod.values,
            cos_zenith         =df["cos_zenith"].values,
            airmass            =df["airmass"].values,
            aod_550nm          =df.get("aod_550nm", pd.Series(0.1, index=df.index)).values,
            ghi_clear          =df["ghi_clear"].values,
            dni_clear          =df["dni_clear"].values,
            dhi_clear          =df["dhi_clear"].values,
            ssa                =df.get("ssa_550nm", pd.Series(0.92, index=df.index)).values,
            asymmetry          =df.get("asymmetry_factor", pd.Series(0.65, index=df.index)).values,
        )

        # Drop night and NaN rows
        df = df[df["ghi_clear"] > 10.0]
        df = df.dropna(subset=["Kt_target", "Kt_phys"])

        return df

    # ──────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────

    def train(
        self,
        df_train: pd.DataFrame,
        n_cv_folds: int = 0,
    ) -> dict:
        """
        Fit the XGBoost pipeline.

        Parameters
        ----------
        df_train   : Output of build_training_set()
        n_cv_folds : If > 1, run k-fold cross-validation and report CV metrics.

        Returns
        -------
        dict with training metrics (mae, rmse, r2, n_train, n_val, best_iter, cv_*)
        """
        import xgboost as xgb

        if len(df_train) < self.min_samples:
            raise ValueError(
                f"Only {len(df_train)} samples; need ≥ {self.min_samples}. "
                "Download more CAMS data or reduce min_train_samples."
            )

        X_all = _build_features(df_train)[_FEATURE_COLS].values
        y_all = df_train["Kt_target"].values

        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.15, random_state=42, shuffle=True
        )

        xgb_model = xgb.XGBRegressor(
            n_estimators         =800,
            learning_rate        =0.03,
            max_depth            =6,
            subsample            =0.80,
            colsample_bytree     =0.80,
            colsample_bylevel    =0.80,
            min_child_weight     =5,
            reg_alpha            =0.1,
            reg_lambda           =1.5,
            gamma                =0.05,
            objective            ="reg:squarederror",   # RMSE minimisation
            tree_method          ="hist",
            random_state         =42,
            n_jobs               =-1,
            eval_metric          ="rmse",
            early_stopping_rounds=50,
        )

        scaler     = RobustScaler()
        X_train_s  = scaler.fit_transform(X_train)
        X_val_s    = scaler.transform(X_val)

        xgb_model.fit(
            X_train_s, y_train,
            eval_set=[(X_val_s, y_val)],
            verbose=False,
        )

        self.pipeline = {"scaler": scaler, "model": xgb_model,
                         "feature_cols": _FEATURE_COLS}

        y_pred = xgb_model.predict(X_val_s)
        metrics = {
            "n_train":      len(X_train),
            "n_val":        len(X_val),
            "mae":          float(mean_absolute_error(y_val, y_pred)),
            "rmse":         float(np.sqrt(mean_squared_error(y_val, y_pred))),
            "r2":           float(r2_score(y_val, y_pred)),
            "best_iteration": int(xgb_model.best_iteration),
        }

        # Optional k-fold CV
        if n_cv_folds > 1:
            cv_metrics = self._cross_validate(X_all, y_all, scaler, n_cv_folds)
            metrics.update(cv_metrics)

        logger.info(
            "Kt model: MAE=%.4f  RMSE=%.4f  R²=%.4f  iters=%d",
            metrics["mae"], metrics["rmse"], metrics["r2"], metrics["best_iteration"]
        )
        return metrics

    def _cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scaler: RobustScaler,
        k: int,
    ) -> dict:
        """Run k-fold CV and return mean RMSE/R2."""
        import xgboost as xgb

        kf = KFold(n_splits=k, shuffle=True, random_state=0)
        rmse_list, r2_list = [], []

        for tr_idx, val_idx in kf.split(X):
            s = RobustScaler()
            X_tr_s = s.fit_transform(X[tr_idx])
            X_val_s = s.transform(X[val_idx])

            m = xgb.XGBRegressor(
                n_estimators=400, learning_rate=0.05, max_depth=5,
                objective="reg:squarederror", tree_method="hist",
                n_jobs=-1, random_state=42,
            )
            m.fit(X_tr_s, y[tr_idx], verbose=False)
            y_pred = m.predict(X_val_s)
            rmse_list.append(float(np.sqrt(mean_squared_error(y[val_idx], y_pred))))
            r2_list.append(float(r2_score(y[val_idx], y_pred)))

        return {
            f"cv_{k}fold_rmse_mean": float(np.mean(rmse_list)),
            f"cv_{k}fold_rmse_std":  float(np.std(rmse_list)),
            f"cv_{k}fold_r2_mean":   float(np.mean(r2_list)),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return predicted Kt array.

        Handles missing extended features gracefully (filled with defaults
        in _build_features).
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        X = _build_features(df)[_FEATURE_COLS].values
        X_s = self.pipeline["scaler"].transform(X)
        return self.pipeline["model"].predict(X_s).clip(0.0, 1.05)

    def predict_ghi(
        self,
        df: pd.DataFrame,
        ghi_clear: np.ndarray,
    ) -> np.ndarray:
        """
        Predict all-sky GHI by de-normalising the Kt prediction.

            GHI_pred = Kt_pred × GHI_clear

        Parameters
        ----------
        df        : Feature DataFrame (same structure as training)
        ghi_clear : Clear-sky GHI at forecast times (W/m²)

        Returns
        -------
        ghi_pred : Predicted GHI (W/m²), clipped ≥ 0
        """
        kt = self.predict(df)
        return np.clip(kt * ghi_clear, 0.0, None)

    # ──────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────

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
