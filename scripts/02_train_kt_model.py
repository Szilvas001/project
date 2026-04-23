#!/usr/bin/env python
"""
Step 2 — Train the XGBoost Kt model.

Loads CAMS atmospheric data and CAMS radiation from PostgreSQL, computes
clear-sky irradiance with spectrl2, builds the training set, trains the
model, and saves it to disk.

Usage:
    python scripts/02_train_kt_model.py
    python scripts/02_train_kt_model.py --config my_config.yaml
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
from solar_forecast.clearsky.spectrl2_model import compute_clearsky
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

    logger.info("═" * 60)
    logger.info("Training Kt model")
    logger.info("  Period   : %s → %s", start_str, end_str)
    logger.info("  Location : %.4f°, %.4f°  alt=%.0f m", lat, lon, altitude)
    logger.info("  Tilt/Az  : %.1f° / %.1f°", tilt, azimuth)
    logger.info("═" * 60)

    db = DBManager(cfg)

    # Load atmospheric data
    logger.info("Loading CAMS atmospheric data…")
    df_atmo = db.load_cams_atmo(lat, lon, start, end)
    if df_atmo.empty:
        logger.error("No CAMS atmospheric data in DB. Run 01_download_cams.py first.")
        sys.exit(1)
    logger.info("  %d records loaded (%.0f days)", len(df_atmo),
                (df_atmo.index[-1] - df_atmo.index[0]).days)

    # Load radiation (CAMS McClear / observed)
    logger.info("Loading CAMS radiation data…")
    df_rad = db.load_cams_radiation(lat, lon, start, end)
    if df_rad.empty:
        logger.warning("No CAMS radiation in DB — Kt target will be estimated from physics.")

    # Compute clear-sky for training timestamps
    logger.info("Computing spectrl2 clear-sky irradiance…")
    times = df_atmo.index

    aod_s  = df_atmo["aod_550nm"].fillna(0.10)
    pw_s   = df_atmo["precipitable_water"].fillna(1.50)
    pres_s = df_atmo.get("surface_pressure",
             __import__("pandas").Series(1013.25, index=times)).fillna(1013.25)
    oz_s   = df_atmo.get("total_ozone",
             __import__("pandas").Series(310.0, index=times)).fillna(310.0)

    df_cs = compute_clearsky(
        times=times,
        lat=lat, lon=lon, altitude=altitude,
        tilt=tilt, azimuth=azimuth,
        aod_550nm=aod_s.values,
        precipitable_water=pw_s.values,
        surface_pressure=pres_s.values,
        ozone_du=oz_s.values,
    )
    logger.info("  Clear-sky computed for %d time steps.", len(df_cs))

    # Build training set and train
    trainer = KtTrainer(cfg)
    logger.info("Building training set…")
    df_train = trainer.build_training_set(df_atmo, df_rad, df_cs)
    logger.info("  %d daytime samples available for training.", len(df_train))

    logger.info("Training XGBoost model…")
    metrics = trainer.train(df_train)
    logger.info("  MAE  = %.4f  (mean absolute error on Kt)", metrics["mae"])
    logger.info("  R²   = %.4f", metrics["r2"])
    logger.info("  Best iteration: %d", metrics["best_iteration"])

    out_path = args.output or cfg["model"]["kt_model_path"]
    trainer.save(out_path)
    logger.info("Model saved to %s", out_path)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
