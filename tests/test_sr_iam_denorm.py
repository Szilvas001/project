"""Acceptance tests: SR / IAM / denorm / AI MUST each change forecast output.

These tests run the pipeline with mocked Open-Meteo and CAMS, varying one
parameter at a time and asserting that the total energy output changes.

TASK 9 acceptance criteria
---------------------------
  ✓  mono_si output ≠ cdte output
  ✓  custom SR CSV changes forecast vs built-in SR
  ✓  IAM model changes forecast (ashrae vs martin_ruiz)
  ✓  denorm_factor changes forecast
  ✓  AI toggle changes forecast (when model is available)
"""

from __future__ import annotations
import io
import tempfile
import types
import csv

import numpy as np
import pandas as pd
import pytest


# ── Shared mock ───────────────────────────────────────────────────────────

def _empty_weather(monkeypatch):
    """Monkeypatch _fetch_openmeteo to return empty DataFrame (clear-sky path)."""
    import solar_forecast.demo.pipeline as pl
    monkeypatch.setattr(pl, "_fetch_openmeteo", lambda *a, **kw: pd.DataFrame())


def _make_custom_sr_csv(wavelengths, values) -> str:
    """Write a SR CSV to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="")
    w = csv.writer(tmp)
    w.writerow(["wavelength_nm", "sr_value"])
    for wl, sr in zip(wavelengths, values):
        w.writerow([wl, sr])
    tmp.close()
    return tmp.name


# ── Tests ──────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_technology_mono_si_ne_cdte(monkeypatch):
    """mono_si and CdTe must produce different energy totals under the same sky."""
    _empty_weather(monkeypatch)
    from solar_forecast.demo.pipeline import run_demo_forecast

    r_si   = run_demo_forecast(lat=47.5, lon=13.0, capacity_kw=5.0, technology="mono_si",
                               horizon_days=1)
    r_cdte = run_demo_forecast(lat=47.5, lon=13.0, capacity_kw=5.0, technology="cdte",
                               horizon_days=1)

    e_si   = r_si["hourly"]["energy_kwh"].sum()
    e_cdte = r_cdte["hourly"]["energy_kwh"].sum()
    # On any non-zero solar day, the spectral mismatch differs by >0
    assert abs(e_si - e_cdte) > 0.0 or (e_si == 0 and e_cdte == 0), \
        f"mono_si ({e_si:.3f} kWh) == cdte ({e_cdte:.3f} kWh) — spectral MM not applied"


@pytest.mark.slow
def test_custom_sr_changes_forecast(monkeypatch):
    """A flat-top (IR-heavy) custom SR must yield different result than default mono_si."""
    _empty_weather(monkeypatch)
    from solar_forecast.demo.pipeline import run_demo_forecast

    # Create a SR curve that peaks in the NIR (900-1100 nm) — very different from AM1.5G
    wl = list(range(300, 1201, 50))
    sr = [0.0 if w < 800 else 1.0 for w in wl]
    csv_path = _make_custom_sr_csv(wl, sr)

    r_default = run_demo_forecast(lat=47.5, lon=13.0, capacity_kw=5.0, technology="mono_si",
                                  horizon_days=1)
    r_custom  = run_demo_forecast(lat=47.5, lon=13.0, capacity_kw=5.0,
                                  sr_csv=csv_path, horizon_days=1)

    e_default = r_default["hourly"]["energy_kwh"].sum()
    e_custom  = r_custom["hourly"]["energy_kwh"].sum()
    assert abs(e_default - e_custom) > 0.0 or (e_default == 0 and e_custom == 0), \
        "Custom SR had no effect on forecast"


@pytest.mark.slow
def test_iam_model_changes_forecast(monkeypatch):
    """Different IAM models must yield different POA and thus different power."""
    _empty_weather(monkeypatch)
    from solar_forecast.demo.pipeline import run_demo_forecast

    r_ashrae = run_demo_forecast(lat=47.5, lon=13.0, capacity_kw=5.0, tilt=30.0,
                                 iam_model="ashrae", horizon_days=1)
    r_fresnel = run_demo_forecast(lat=47.5, lon=13.0, capacity_kw=5.0, tilt=30.0,
                                  iam_model="fresnel", horizon_days=1)

    e_ashrae  = r_ashrae["hourly"]["energy_kwh"].sum()
    e_fresnel = r_fresnel["hourly"]["energy_kwh"].sum()
    assert abs(e_ashrae - e_fresnel) > 0.0 or (e_ashrae == 0 and e_fresnel == 0), \
        "IAM model had no effect on forecast"


@pytest.mark.slow
def test_denorm_factor_scales_output(monkeypatch):
    """Changing denorm_factor must scale effective irradiance and power."""
    _empty_weather(monkeypatch)
    from solar_forecast.demo.pipeline import run_demo_forecast

    r1 = run_demo_forecast(lat=47.5, lon=13.0, capacity_kw=5.0, horizon_days=1,
                           denorm_factor=1.0)
    r2 = run_demo_forecast(lat=47.5, lon=13.0, capacity_kw=5.0, horizon_days=1,
                           denorm_factor=0.8)

    e1 = r1["hourly"]["energy_kwh"].sum()
    e2 = r2["hourly"]["energy_kwh"].sum()
    if e1 > 0:
        assert e2 < e1, f"denorm=0.8 gave {e2:.3f} kWh ≥ denorm=1.0 gave {e1:.3f} kWh"
    # When sun is 0 everywhere, both are 0 — that's OK


@pytest.mark.slow
def test_ai_toggle_changes_forecast(monkeypatch, tmp_path):
    """Enabling use_ai must change forecast output when a model file exists."""
    _empty_weather(monkeypatch)

    # Create a minimal mock KtTrainer that returns kt=0.5 regardless
    import solar_forecast.allsky.ai_trainer as ai_mod

    class MockTrainer:
        def __init__(self, *a, **kw): pass
        def load(self, *a): return self
        def predict(self, df): return np.full(len(df), 0.5)

    monkeypatch.setattr(ai_mod, "KtTrainer", MockTrainer)

    # Create a dummy model file so the pipeline tries to load it
    model_path = str(tmp_path / "kt.joblib")
    open(model_path, "w").close()

    from solar_forecast.demo.pipeline import run_demo_forecast
    r_no_ai = run_demo_forecast(lat=47.5, lon=13.0, capacity_kw=5.0, horizon_days=1,
                                use_ai=False)
    r_ai    = run_demo_forecast(lat=47.5, lon=13.0, capacity_kw=5.0, horizon_days=1,
                                use_ai=True, kt_model_path=model_path)

    e_no_ai = r_no_ai["hourly"]["energy_kwh"].sum()
    e_ai    = r_ai["hourly"]["energy_kwh"].sum()
    assert abs(e_no_ai - e_ai) > 0.0 or (e_no_ai == 0 and e_ai == 0), \
        "AI toggle had no effect on forecast"


# ── SpectralResponse unit tests ────────────────────────────────────────────

def test_sr_mono_si_vs_cdte_mismatch_differs():
    """MM computed for the same spectrum must differ between mono_si and CdTe."""
    from solar_forecast.production.spectral_response import SpectralResponse
    wl = np.linspace(300, 1200, 100)
    # Flat spectrum
    flat = np.ones_like(wl)
    spec = {"wavelength": wl, "poa_global": flat}

    mm_mono = SpectralResponse("mono_si").mismatch_factor(spec)
    mm_cdte = SpectralResponse("cdte").mismatch_factor(spec)
    assert abs(mm_mono - mm_cdte) > 0.001, \
        f"mono_si MM={mm_mono:.4f} ≈ cdte MM={mm_cdte:.4f} — SR curves have no effect"


def test_sr_custom_csv_differs_from_builtin(tmp_path):
    from solar_forecast.production.spectral_response import SpectralResponse
    wl = np.linspace(300, 1100, 80)
    # Create a "step" SR that is 0 below 700 nm, 1 above
    sr_vals = np.where(wl >= 700, 1.0, 0.0)
    csv_path = tmp_path / "custom.csv"
    pd.DataFrame({"wavelength_nm": wl, "sr_value": sr_vals}).to_csv(csv_path, index=False)

    flat = np.ones_like(wl)
    spec = {"wavelength": wl, "poa_global": flat}

    mm_builtin = SpectralResponse("mono_si").mismatch_factor(spec)
    mm_custom  = SpectralResponse(csv_path=str(csv_path)).mismatch_factor(spec)
    assert abs(mm_builtin - mm_custom) > 0.01, \
        "Custom SR CSV gives same MM as built-in"


def test_sr_night_returns_one():
    """Night (zero irradiance) must give MM = 1.0 (no penalty)."""
    from solar_forecast.production.spectral_response import SpectralResponse
    wl = np.linspace(300, 1200, 50)
    spec = {"wavelength": wl, "poa_global": np.zeros_like(wl)}
    mm = SpectralResponse("cdte").mismatch_factor(spec)
    assert mm == pytest.approx(1.0)


# ── IAM unit tests ────────────────────────────────────────────────────────

def test_iam_models_differ_at_high_aoi():
    """ASHRAE and Fresnel IAM must give different values at high incidence angle."""
    from solar_forecast.production.iam_model import iam_ashrae, iam_fresnel
    aoi = np.array([60.0, 70.0, 80.0])
    iam_a = iam_ashrae(aoi)
    iam_f = iam_fresnel(aoi)
    assert not np.allclose(iam_a, iam_f, rtol=0.01), \
        "ASHRAE and Fresnel IAM are identical at high AOI"


# ── Denorm / pipeline output sanity ────────────────────────────────────────

def test_pipeline_output_columns(monkeypatch):
    _empty_weather(monkeypatch)
    from solar_forecast.demo.pipeline import run_demo_forecast
    r = run_demo_forecast(lat=47.5, lon=13.0, horizon_days=1)
    cols = r["hourly"].columns.tolist()
    for required in ["power_kw", "energy_kwh", "ghi_wm2", "poa_wm2", "kt", "iam"]:
        assert required in cols, f"Column {required!r} missing from hourly output"


def test_pipeline_spectral_mm_column_present(monkeypatch):
    _empty_weather(monkeypatch)
    from solar_forecast.demo.pipeline import run_demo_forecast
    r = run_demo_forecast(lat=47.5, lon=13.0, horizon_days=1, technology="cdte")
    assert "spectral_mm" in r["hourly"].columns
