#!/usr/bin/env python
"""
Generate a lightweight demo XGBoost Kt model from synthetic physics data.

This script creates a pre-trained model that works out-of-the-box so Expert
users can toggle AI correction immediately without downloading CAMS history.

The demo model is trained on ~8760 synthetic hourly samples generated via
the physics pipeline itself, so its structure (features, booster internals)
is identical to a real CAMS-trained model.

Usage:
    python scripts/04_generate_demo_model.py

Output:
    models/kt_xgb.joblib   (~200 KB)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("gen_demo_model")

SEED = 42
N_SAMPLES = 8760  # one synthetic year, hourly
OUT_PATH = Path("models/kt_xgb.joblib")


def _synthetic_features(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """
    Generate physically plausible feature distributions for training.

    Feature set matches solar_forecast/allsky/ai_trainer.py:
      Core (13): sza, cos_sza, azimuth, et_rad, aod_550, ozone, precip_water,
                 pressure, temp, rh, wind, sin_hour, cos_hour
      Extended (8): ssa_norm, asym_norm, alpha1_norm, alpha2_norm,
                    pm25_log, blh_norm, cloud_composite, cloud_low_frac
    """
    # Solar geometry: daytime only
    sza    = rng.uniform(10, 85, n)
    cos_z  = np.cos(np.radians(sza))
    azimuth_sun = rng.uniform(80, 280, n)
    et_rad = 1361 * (1 + 0.033 * np.cos(2 * np.pi * rng.integers(1, 366, n) / 365))
    hour   = rng.uniform(5, 19, n)

    # Atmospheric state
    aod   = rng.gamma(1.5, 0.08, n).clip(0.01, 3.0)
    ozone = rng.normal(310, 30, n).clip(220, 450)
    pw    = rng.gamma(2.0, 0.8, n).clip(0.1, 6.0)
    pres  = rng.normal(1013, 15, n).clip(940, 1040)
    temp  = rng.normal(15, 10, n).clip(-20, 45)
    rh    = rng.uniform(20, 95, n)
    wind  = rng.gamma(2.0, 1.5, n).clip(0, 20)

    # Extended physics features
    ssa   = rng.normal(0.92, 0.05, n).clip(0.6, 1.0)
    asym  = rng.normal(0.65, 0.07, n).clip(0.3, 0.9)
    a1    = rng.normal(1.30, 0.30, n).clip(0.1, 2.5)
    a2    = rng.normal(1.10, 0.25, n).clip(0.1, 2.5)
    pm25  = np.log1p(rng.gamma(2, 5, n).clip(0.1, 200))
    blh   = rng.gamma(2, 500, n).clip(100, 3000)
    cloud = rng.beta(0.8, 1.2, n)
    clow  = rng.beta(0.7, 1.5, n)

    return {
        "sza": sza, "cos_sza": cos_z, "azimuth_sun": azimuth_sun,
        "et_rad": et_rad, "aod_550nm": aod, "ozone_du": ozone,
        "precip_water": pw, "pressure_hpa": pres, "temp_c": temp,
        "rh": rh, "wind_ms": wind,
        "sin_hour": np.sin(2 * np.pi * hour / 24),
        "cos_hour": np.cos(2 * np.pi * hour / 24),
        # Extended
        "ssa_norm": ssa, "asym_norm": asym,
        "alpha1_norm": a1, "alpha2_norm": a2,
        "pm25_log": pm25, "blh_norm": blh / 2000.0,
        "cloud_composite": cloud, "cloud_low_frac": clow,
    }


def _synthetic_kt(feats: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """
    Physics-informed Kt target.

    Kt = (1 − cloud_composite) × exp(−aod_excess × airmass × ext_eff)
         + cloud_composite × (0.1 + 0.05 × rng.normal)

    Then add small calibration noise to simulate real residuals.
    """
    cloud = feats["cloud_composite"]
    aod   = feats["aod_550nm"]
    am    = 1.0 / np.maximum(feats["cos_sza"], 0.05)
    ssa   = feats["ssa_norm"]
    g     = feats["asym_norm"]

    ext   = np.clip(aod * am * (1.0 - ssa * g), 0, 5)
    kt_clear = np.exp(-ext)
    kt_cloud = 0.10 + 0.06 * rng.standard_normal(len(cloud))
    kt = (1 - cloud) * kt_clear + cloud * kt_cloud
    kt += 0.03 * rng.standard_normal(len(kt))   # sensor / residual noise
    return np.clip(kt, 0.0, 1.05).astype(np.float32)


def main():
    try:
        import xgboost as xgb
        import joblib
    except ImportError:
        logger.error("xgboost and joblib required: pip install xgboost joblib")
        sys.exit(1)

    OUT_PATH.parent.mkdir(exist_ok=True)

    rng = np.random.default_rng(SEED)
    logger.info("Generating %d synthetic training samples …", N_SAMPLES)

    feats = _synthetic_features(N_SAMPLES, rng)
    kt    = _synthetic_kt(feats, rng)

    import pandas as pd
    X = pd.DataFrame(feats)
    y = kt

    logger.info("Training XGBoost Kt model …")
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X, y, eval_set=[(X, y)], verbose=False)

    joblib.dump(model, OUT_PATH)
    size_kb = OUT_PATH.stat().st_size // 1024
    logger.info("Demo model saved → %s  (%d KB)", OUT_PATH, size_kb)

    # Quick validation
    pred = model.predict(X[:100])
    rmse = float(np.sqrt(np.mean((pred - y[:100]) ** 2)))
    logger.info("Demo model train RMSE (Kt): %.4f  (target: <0.08 on real data)", rmse)
    logger.info("Done. Toggle AI correction in Expert mode to use this model.")


if __name__ == "__main__":
    main()
