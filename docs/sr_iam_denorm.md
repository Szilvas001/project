# Spectral Response, IAM Models, and Denormalisation

Expert-tier controls in `solar_forecast/production/` that affect the conversion from plane-of-array irradiance to effective cell irradiance.

---

## Spectral Response (SR)

### What it does

SR(λ) [A/W] describes current generated per unit spectral irradiance. The spectral mismatch factor MM corrects for sky spectrum deviating from the AM1.5G reference:

```
MM = [∫ SR(λ)·I(λ) dλ / ∫ SR(λ)·G_AM15(λ) dλ]
   / [∫ I(λ) dλ / ∫ G_AM15(λ) dλ]
```

Computed per time step from the SPECTRL2 spectrum against the embedded ASTM G173-03 AM1.5G reference (122 points, 280–4000 nm).

### Built-in SR curves

| Code | Technology | Response peak |
|---|---|---|
| `mono_si` | Monocrystalline silicon (default) | ~780 nm |
| `poly_si` | Polycrystalline silicon | ~790 nm |
| `cdte` | Cadmium telluride thin-film | ~500–900 nm |
| `cigs` | CIGS / CIS thin-film | ~800–1100 nm |
| `hit` | HIT / heterojunction | ~780 nm, wider |

### Custom SR CSV upload

Dashboard: Expert tier → SR Upload. API: `sr_path` parameter.

**Required CSV format:**

```csv
wavelength_nm,sr_value
280,0.000
300,0.010
400,0.440
780,1.000
1100,0.240
1200,0.000
```

- `wavelength_nm`: monotonically increasing, 280–1200 nm minimum range
- `sr_value`: normalised 0–1 (peak = 1.0); absolute A/W values are auto-normalised
- Minimum 20 points; linear interpolation maps to SPECTRL2 wavelength grid

---

## IAM Models

The incidence angle modifier reduces beam transmission as the sun moves away from normal incidence. All models are in `solar_forecast/production/iam_model.py`.

### ASHRAE (default)

Souka & Safwat (1966). One parameter, fastest:

```
IAM = 1 − b₀ × (1/cos(AOI) − 1)
```

b₀ = 0.05 (antireflection-coated glass). Returns 0 for AOI ≥ 85°.

### Martin-Ruiz

Martin & Ruiz (2001). Exponential, more physical at low-sun angles:

```
IAM = exp(−c₁ × (1/cos(AOI) − 1)^c₂)
```

c₁ = 0.16, c₂ = 1.0 (c-Si with standard AR coating). Returns 0 for AOI ≥ 85°.

### Physical Fresnel

Exact Fresnel equations, S and P polarisations averaged:

```
sin(θₜ) = sin(θᵢ) / n_glass           Snell's law, n_glass = 1.526
r_s = (cos θᵢ − n·cos θₜ) / (cos θᵢ + n·cos θₜ)
r_p = (n·cos θᵢ − cos θₜ) / (n·cos θᵢ + cos θₜ)
R   = (r_s² + r_p²) / 2
R₀  = ((1 − n)/(1 + n))²
IAM = (1 − R) / (1 − R₀)
```

Returns 0 for AOI ≥ 89.5°. Most accurate for detailed loss analysis.

### Diffuse IAM

All models apply the De Soto (2006) equivalent angle for isotropic sky:

```
AOI_eq = 59.68 − 0.1388 × tilt + 0.001497 × tilt²
IAM_diffuse = IAM(AOI_eq)
```

### Selecting a model

Dashboard: Expert tier → IAM Model dropdown.  
API: set `"iam_model": "ashrae"` | `"martin_ruiz"` | `"fresnel"` in `RealtimeRequest`.

---

## Denormalisation Factor

### Definition

D maps the spectral mismatch factor (dimensionless) back to effective broadband irradiance:

```
D     = ∫₃₀₀¹²⁰⁰ I(λ) dλ / ∫₃₀₀⁴⁰⁰⁰ I(λ) dλ    (PV-band fraction)
G_eff = MM × D × G_POA_broadband                   W/m²
```

D is not a free parameter — it is computed per time step from SPECTRL2 output. It is not the same for every hour because the spectral composition of sunlight changes with air mass, aerosol load, and cloud conditions.

### Typical values

| Condition | D |
|---|---|
| Clear, low air mass (SZA < 30°) | 0.52–0.56 |
| Overcast / high air mass | 0.48–0.52 |
| Sunrise / sunset | 0.44–0.50 |

### When to override

Override (Expert tier slider or API `denorm_factor`) only when:
- Using a custom SR with response beyond 1200 nm (e.g. InGaAs)
- Calibrating against measured production data without SPECTRL2

Set to `null` to restore automatic per-step computation. Valid override range: 0.5–1.0.
