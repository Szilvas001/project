# Forecast Engine

Physics chain implemented in `solar_forecast/engine/`, orchestrated by `run_demo_forecast()` / `run_realtime_forecast()` in `solar_forecast/demo/pipeline.py`. Each step has an independent fallback.

---

## Step 1 — Open-Meteo Weather

Live hourly GHI, DNI, DHI, cloud cover, temperature, humidity, pressure. Free, no key, global.

```
inputs : lat, lon, horizon_days
outputs: ghi_wm2, dni_wm2, dhi_wm2, cloud_cover_frac, temp_c, pressure_hpa
```

## Step 2 — CAMS Atmosphere (`_resolve_atmosphere`)

Per-timestep AOD, ozone, water vapour, SSA, asymmetry, BLH from PostgreSQL. Falls back to continental-Europe climatology if unavailable.

```
outputs: aod_550nm, angstrom_alpha1/2, ssa, asymmetry,
         ozone_du, precipitable_water_cm, surface_pressure_hpa
source : "cams" | "climatology"
```

## Step 3 — SPECTRL2 Clear-Sky (`solar_forecast/clearsky/spectrl2_model.py`)

pvlib Bird & Riordan (1986). Computes full 300–4000 nm spectrum; integrates to broadband GHI/DNI/DHI/POA. Fallback: `simplified_solis`.

## Step 4 — Physics Kt (`solar_forecast/allsky/physics_kt.py`)

Delta-Eddington two-stream approximation. Clearness index Kt = GHI_all / GHI_clear.

**Cloud component:**
```
T_eff    = ω_c + (1 − ω_c) × exp(−τ_c / μ₀)      ω_c = 0.9997
Kt_cloud = (1 − f_c) + f_c × [R_d + R_n × T_eff]
```
τ_c from Open-Meteo COD or Stephens (1978): `τ_c = −ln(1 − f_c) × 14`.

**Aerosol excess:**
```
Kt_aer  = exp(−ΔAOD × airmass × (1 − ω₀ × g))
Kt_phys = Kt_cloud × Kt_aer
```

## Step 5 — Optional AI Kt (`solar_forecast/allsky/hybrid_model.py`)

XGBoost model (21 CAMS features). Blended with physics:

```
Kt_final = α × Kt_phys + (1 − α) × Kt_ai      α = 0.40 (physics_weight)
GHI_all  = Kt_final × GHI_clear
```

Skipped if `models/kt_xgb.joblib` absent.

## Step 6 — Perez Transposition

GHI/DNI/DHI → plane-of-array (POA) irradiance on the tilted surface. Perez anisotropic diffuse model (pvlib).

```
inputs : ghi, dni, dhi, solar_zenith, solar_azimuth, surface_tilt, surface_azimuth
outputs: poa_global, poa_direct, poa_diffuse
```

## Step 7 — Spectral Response & Denormalisation

Spectral mismatch factor MM from SPECTRL2 spectrum vs AM1.5G reference:

```
MM    = [∫ SR(λ)·I(λ) dλ / ∫ SR(λ)·G_AM15(λ) dλ] / [∫ I(λ) dλ / ∫ G_AM15(λ) dλ]
D     = ∫₃₀₀¹²⁰⁰ I(λ) dλ / ∫₃₀₀⁴⁰⁰⁰ I(λ) dλ     (PV band fraction, per time step)
G_eff = MM × D × G_POA_broadband
```

## Step 8 — IAM Correction (`solar_forecast/production/iam_model.py`)

Angle-of-incidence modifier on beam component. Three models: `ashrae` (default), `martin_ruiz`, `fresnel`. See [sr_iam_denorm.md](sr_iam_denorm.md).

```
G_POA_eff = IAM(AOI) × G_POA_direct + G_POA_diffuse × IAM_diffuse
```

## Step 9 — NOCT Cell Temperature

```
T_cell = T_air + (NOCT − 20) / 800 × G_POA_eff     NOCT = 45°C default
```

## Step 10 — DC Power

```
P_dc = G_eff / 1000 × P_stc × [1 + γ × (T_cell − 25)]
```

γ = −0.0045 /°C (mono-Si). Technology-specific values in `config.yaml`.

## Step 11 — AC Power

```
P_ac = P_dc × η_inverter × η_wiring × η_soiling
```

Defaults: η_inverter = 0.97, η_wiring = 0.98, η_soiling = 0.97.

---

## Sub-Hourly Real-Time Mode

`run_realtime_forecast()` runs the identical chain at 5–60 min resolution over 1–72 h. Open-Meteo hourly weather is interpolated to the target timestep; solar position and all physics are recomputed at each sub-step. Produces a smooth continuous power curve without hour-boundary aliasing.

---

## Fallback Hierarchy

| Failure | Fallback |
|---|---|
| Open-Meteo unreachable | Demo CSV (`demo-data/`) |
| CAMS unavailable | Continental-Europe climatology |
| SPECTRL2 exception | pvlib `simplified_solis` |
| XGBoost model missing | Physics-only (α = 1.0) |
| pygrib import fails | NetCDF / xarray |
