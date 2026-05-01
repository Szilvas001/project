# User Flows

The dashboard exposes three progressively richer levels — Basic, Pro, and
Expert. The underlying physics engine is **always the same**: SPECTRL2
clear-sky → Perez transposition → IAM × SR(λ) → NOCT cell temp →
temperature-derated DC → AC. Only the **visible knobs** change.

---

## Basic — "city + kW, give me the answer"

**Audience:** homeowners, sales teams, end customers.

1. Open the dashboard at `http://localhost:8501`.
2. Sidebar → **Location**: pick a **Preset** (Budapest, Madrid, Berlin, Cairo,
   Sydney, Phoenix) **or** type a city name and press *Find*. GPS mode is
   also available for power users.
3. Set **System size (kW)** — that is the rated DC capacity of the array.
4. The Dashboard tab updates immediately with:
   - **Today / Tomorrow / N-Day total** (kWh)
   - **Peak power** (kW) and the time it occurs
   - **Capacity factor** and **cloud loss** (%)
   - **Forecast confidence** with a "Why?" panel listing reasons

No tilt, azimuth, panel-tech, or physics settings are exposed. The system
infers a sane tilt (latitude × 0.76) and a default south-facing azimuth.

---

## Pro — "I know my array geometry"

**Audience:** installers, asset managers, technical sales.

Adds, on top of Basic:

- **Tilt** and **Azimuth** sliders (180° = south, 90° = east, 270° = west)
- **Panel type** selector — mono-Si, poly-Si, CdTe, CIGS, HIT/HJT
- **Forecast days** slider (1 → 14)
- **Forecast resolution** — `hourly` (default) or `15min`
- **Timezone** — display only, the engine works in UTC
- **Multi-location**: save sites in the Locations tab, run forecasts per ID,
  download CSV reports per site

The physics is exactly the same — the panel-type selector swaps the SR(λ)
curve, which feeds the per-timestep spectral mismatch factor. CdTe vs
mono-Si typically differs by 3 – 6 % in summer due to spectral and
temperature-coefficient differences.

---

## Expert — "let me tune everything"

**Audience:** PV engineers, researchers, calibration specialists.

Adds, on top of Pro, an **Advanced physics settings** expander with:

- **IAM model** — `ashrae` (default), `martin_ruiz`, `fresnel`
- **AI Kt correction toggle** — uses the trained XGBoost model at
  `models/kt_xgb.joblib`. Falls back to physics-only if no model is present
  or training has not been run.
- **Custom SR curve upload** — drop a CSV with two columns
  (`wavelength_nm`, `sr_value`). See [sr-csv-format.md](sr-csv-format.md).
- **Calibration / denormalization** is exposed via `config.yaml` and
  `scripts/02_train_kt_model.py`.

Every Expert toggle is **opt-in** — the Basic and Pro flows continue to work
unchanged when no Expert option is touched.

---

## API equivalents

| UI control | API field |
|---|---|
| Preset / City / GPS | `lat`, `lon` |
| System size | `capacity_kw` |
| Tilt / Azimuth | `tilt`, `azimuth` |
| Panel type | `technology` |
| Forecast days | `horizon_days` |
| Resolution | `resolution` (`"hourly"` \| `"15min"`) |
| IAM model | `iam_model` |
| AI toggle | `use_ai` |
| Custom SR | `sr_csv_path` (server-side path) |

See [api.md](api.md) for the full request/response schema.

---

## What's *not* a feature

- **Financial yield projection.** This product forecasts kWh, not income.
  Tariff structures, curtailment, and storage round-trip losses are out of
  scope. See the disclaimer in [faq.md](faq.md).
- **Beyond 14 days.** Open-Meteo numerical-weather-prediction skill drops
  rapidly past D+10. Longer horizons fall back to climatology.
- **Sub-15-minute resolution.** Open-Meteo source data is hourly; sub-hourly
  outputs are interpolated and should not be used for grid-balancing
  decisions.
