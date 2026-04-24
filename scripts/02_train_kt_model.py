#!/usr/bin/env python
"""
Step 2 — Train the XGBoost Kt model.

Loads CAMS atmospheric data + CAMS radiation from PostgreSQL, computes the
clear-sky reference (spectrl2), builds the training set with full physics
features (SSA, GG, Ångström exponents, PM, BLH, …), trains the model, and
optionally runs k-fold cross-validation.

Usage:
    python scripts/02_train_kt_model.py
    python scripts/02_train_kt_model.py --cv 5
    python scripts/02_train_kt_model.py --start 2022-01-01 --end 2023-12-31
    python scripts/02_train_kt_model.py --output models/kt_custom.joblib
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solar_forecast.utils import load_config, resolve_tilt_azimuth
from solar_forecast.data_ingestion.db_manager import DBManager
from solar_forecast.clearsky.spectrl2_model import compute_clearsky_from_weather
from solar_forecast.allsky.ai_trainer import KtTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_kt")


def parse_args():
    p = argparse.ArgumentParser(description="Train the XGBoost Kt model.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--start",  default=None, help="Training start YYYY-MM-DD")
    p.add_argument("--end",    default=None, help="Training end YYYY-MM-DD")
    p.add_argument("--output", default=None, help="Model output path (.joblib)")
    p.add_argument("--cv", type=int, default=0,
                   help="Number of cross-validation folds (0 = skip CV)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    start_str = args.start or cfg["cams"].get("training_start", "2021-01-01")
    end_str   = args.end   or cfg["cams"].get("training_end",   "2024-12-31")
    start = datetime.fromisoformat(start_str).replace(tzinfo=timezone.utc)
    end   = datetime.fromisoformat(end_str).replace(tzinfo=timezone.utc)

    lat      = cfg["location"]["lat"]
    lon      = cfg["location"]["lon"]
    altitude = cfg["location"].get("altitude", 0.0)
    tilt, azimuth = resolve_tilt_azimuth(cfg)

    cv_folds = args.cv or cfg["model"].get("n_cv_folds", 0)

    logger.info("═" * 64)
    logger.info("Training Kt model — enhanced feature set")
    logger.info("  Period   : %s → %s  (UTC)", start_str, end_str)
    logger.info("  Location : %.4f°, %.4f°  alt=%.0f m", lat, lon, altitude)
    logger.info("  Tilt/Az  : %.1f° / %.1f°", tilt, azimuth)
    logger.info("  CV folds : %d", cv_folds)
    logger.info("═" * 64)

    db = DBManager(cfg)

    # ── Load atmospheric data ─────────────────────────────────────────
    logger.info("Loading CAMS atmospheric data…")
    df_atmo = db.load_cams_atmo(lat, lon, start, end)
    if df_atmo.empty:
        logger.error("No CAMS atmospheric data in DB. Run 01_download_cams.py first.")
        sys.exit(1)
    logger.info("  %d records loaded (%.0f days, %d columns)",
                len(df_atmo),
                (df_atmo.index[-1] - df_atmo.index[0]).days,
                df_atmo.shape[1])

    # ── Load radiation ────────────────────────────────────────────────
    logger.info("Loading CAMS radiation data…")
    df_rad = db.load_cams_radiation(lat, lon, start, end)
    if df_rad.empty:
        logger.error("No CAMS radiation in DB. Cannot train Kt model.")
        sys.exit(1)
    logger.info("  %d records loaded", len(df_rad))

    # ── Compute clear-sky (spectrl2) with full physics ────────────────
    logger.info("Computing spectrl2 clear-sky irradiance (with SSA, GG)…")
    df_cs = compute_clearsky_from_weather(
        df_atmo, lat, lon, altitude, tilt, azimuth,
        return_spectra=False,
    )
    logger.info("  Clear-sky computed for %d time steps.", len(df_cs))

    # ── Build training set ────────────────────────────────────────────
    trainer = KtTrainer(cfg)
    logger.info("Building training set (physics + CAMS features)…")
    df_train = trainer.build_training_set(df_atmo, df_rad, df_cs)
    logger.info("  %d daytime samples available for training.", len(df_train))

    if len(df_train) < trainer.min_samples:
        logger.error("Insufficient training samples (%d < %d).",
                     len(df_train), trainer.min_samples)
        sys.exit(1)

    # ── Train ──────────────────────────────────────────────────────────
    logger.info("Training XGBoost model (RMSE objective)…")
    metrics = trainer.train(df_train, n_cv_folds=cv_folds)
    logger.info("  MAE   = %.4f", metrics["mae"])
    logger.info("  RMSE  = %.4f", metrics["rmse"])
    logger.info("  R²    = %.4f", metrics["r2"])
    logger.info("  Best iteration : %d", metrics["best_iteration"])

    if cv_folds > 1:
        key_rmse = f"cv_{cv_folds}fold_rmse_mean"
        key_std  = f"cv_{cv_folds}fold_rmse_std"
        key_r2   = f"cv_{cv_folds}fold_r2_mean"
        logger.info("  CV RMSE = %.4f ± %.4f", metrics[key_rmse], metrics[key_std])
        logger.info("  CV R²   = %.4f", metrics[key_r2])

    # ── Save ──────────────────────────────────────────────────────────
    out_path = args.output or cfg["model"]["kt_model_path"]
    trainer.save(out_path)
    logger.info("Model saved to %s", out_path)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
