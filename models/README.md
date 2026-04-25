# Models directory

## kt_xgb.joblib — XGBoost Kt correction model

This directory stores the trained XGBoost clearness-index model.

### Quick demo model (instant, no CAMS needed)

```bash
python scripts/04_generate_demo_model.py
```

Generates a `kt_xgb.joblib` (~200 KB) trained on synthetic physics data.
Works immediately — enables the AI toggle in Expert mode.

### Production model (maximum accuracy)

```bash
# 1. Download 2+ years of CAMS EAC4 history
python scripts/01_download_cams.py --start 2022-01-01 --end 2023-12-31

# 2. Train with 5-fold cross-validation
python scripts/02_train_kt_model.py --cv 5
```

Typical production accuracy (Hungary, 2-year dataset):

| Metric | Physics-only | Physics + AI |
|---|:---:|:---:|
| Kt RMSE | 0.12 | 0.07–0.09 |
| GHI RMSE (W/m²) | 62 | 38–45 |
| R² | 0.86 | 0.91–0.94 |

### Feature set (21 features)

**Core (13)**: solar zenith, cos(zenith), sun azimuth, extraterrestrial radiation,
AOD 550 nm, ozone (DU), precipitable water, pressure, temperature, RH, wind speed,
sin/cos hour-of-day.

**Extended (8)**: SSA (normalized), asymmetry factor, Ångström α1/α2,
PM2.5 (log), boundary-layer height (normalized), total cloud composite,
low-cloud fraction.

### Model details

- Algorithm: XGBoost regressor (`reg:squarederror`)
- Estimators: 300 (demo) / 800 (production)
- Features: 21
- Output: clearness index Kt ∈ [0, 1.05]
- Blend at runtime: 40% physics Kt + 60% AI Kt
