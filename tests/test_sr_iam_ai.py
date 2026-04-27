"""
Verifies that SR(λ), IAM, and the AI toggle actually move the forecast —
not silent no-ops. These are the contract tests CodeCanyon reviewers will
run to confirm the physics surface area is wired through.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def empty_weather():
    """Force the climatological/no-CAMS code path so tests are deterministic."""
    return pd.DataFrame()


def _run(**overrides):
    from solar_forecast.demo.pipeline import run_demo_forecast
    base = dict(
        lat=47.5, lon=19.0, altitude=120.0,
        capacity_kw=5.0, tilt=30.0, azimuth=180.0,
        technology="mono_si", iam_model="ashrae",
        horizon_days=2, resolution="hourly",
    )
    base.update(overrides)
    return run_demo_forecast(**base)


# ─── SR(λ) ────────────────────────────────────────────────────────────────

def test_panel_technology_changes_output(empty_weather):
    """Mono-Si vs CdTe must produce measurably different production."""
    with patch("solar_forecast.demo.pipeline._fetch_openmeteo", return_value=empty_weather):
        mono = _run(technology="mono_si")["hourly"]["energy_kwh"].sum()
        cdte = _run(technology="cdte")["hourly"]["energy_kwh"].sum()

    assert mono > 0 and cdte > 0
    # Different SR cut-offs + temperature coefficients → ≥ 1% delta in 48h energy
    rel_delta = abs(mono - cdte) / mono
    assert rel_delta > 0.01, (
        f"Tech swap changed output by only {rel_delta*100:.3f}% — SR may be inert"
    )


def test_custom_sr_csv_changes_output(tmp_path, empty_weather):
    """A custom SR CSV with a different cut-off must shift output."""
    # Synthetic narrow-band SR (peaked at 600nm, cut at 800nm)
    csv_path = tmp_path / "narrow_sr.csv"
    wl = np.arange(280, 1200, 5)
    sr = np.exp(-((wl - 600) ** 2) / (2 * 80 ** 2))
    pd.DataFrame({"wavelength_nm": wl, "sr_value": sr}).to_csv(csv_path, index=False)

    with patch("solar_forecast.demo.pipeline._fetch_openmeteo", return_value=empty_weather):
        builtin = _run(technology="mono_si")["hourly"]["energy_kwh"].sum()
        custom  = _run(technology="mono_si", sr_csv=str(csv_path))["hourly"]["energy_kwh"].sum()

    rel_delta = abs(builtin - custom) / max(builtin, 1e-6)
    assert rel_delta > 0.005, (
        f"Custom SR CSV changed output by only {rel_delta*100:.3f}% — upload is inert"
    )


# ─── IAM ──────────────────────────────────────────────────────────────────

def test_iam_model_changes_output(empty_weather):
    """ASHRAE vs Fresnel IAM should produce measurably different energy."""
    with patch("solar_forecast.demo.pipeline._fetch_openmeteo", return_value=empty_weather):
        ashrae  = _run(iam_model="ashrae")["hourly"]["energy_kwh"].sum()
        fresnel = _run(iam_model="fresnel")["hourly"]["energy_kwh"].sum()

    assert ashrae > 0 and fresnel > 0
    # IAM curves diverge most at high AOI; require a small but real delta
    rel_delta = abs(ashrae - fresnel) / ashrae
    assert rel_delta > 0.001


# ─── AI toggle pass-through ───────────────────────────────────────────────

def test_use_ai_is_passed_to_pipeline_via_api():
    """ForecastRequest.use_ai must reach run_demo_forecast.

    Regression for the silent drop: the request had use_ai=True but the
    route never forwarded it.
    """
    from fastapi.testclient import TestClient
    from app.api.main import app

    captured = {}

    def fake_run(*, lat, lon, altitude, capacity_kw, tilt, azimuth,
                 technology, iam_model, horizon_days, sr_csv, use_ai,
                 resolution, **kwargs):
        captured.update(use_ai=use_ai, sr_csv=sr_csv,
                        iam_model=iam_model, resolution=resolution)
        idx = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
        df = pd.DataFrame({
            "ghi_wm2": [0.0, 100.0, 200.0, 0.0],
            "power_kw": [0.0, 1.0, 2.0, 0.0],
            "energy_kwh": [0.0, 1.0, 2.0, 0.0],
            "kt": [None] * 4, "t_cell_c": [25.0] * 4,
            "cloud_cover_frac": [0.2] * 4,
        }, index=idx)
        return {
            "hourly": df,
            "summary": {
                "today_kwh": 3.0, "tomorrow_kwh": 0.0, "total_7d_kwh": 3.0,
                "peak_power_kw": 2.0, "peak_hour_utc": str(idx[2]),
                "capacity_factor_pct": 10.0, "cloud_loss_pct": 0.0,
                "confidence_pct": 70.0, "confidence_label": "Medium",
                "confidence_reasons": [], "resolution": resolution,
            },
            "clearsky_hourly": df, "location": {},
        }

    client = TestClient(app)
    with patch("app.api.routes.forecast.run_demo_forecast", side_effect=fake_run):
        r = client.post("/forecast", json={
            "lat": 47.5, "lon": 19.0, "capacity_kw": 5.0, "horizon_days": 1,
            "use_ai": True, "iam_model": "fresnel",
            "resolution": "15min", "sr_csv_path": "/tmp/foo.csv",
        })

    assert r.status_code == 200, r.text
    assert captured["use_ai"] is True
    assert captured["sr_csv"] == "/tmp/foo.csv"
    assert captured["iam_model"] == "fresnel"
    assert captured["resolution"] == "15min"


# ─── Output schema ────────────────────────────────────────────────────────

def test_spectral_factor_in_hourly_columns(empty_weather):
    """The hourly frame must expose the per-step spectral mismatch factor."""
    with patch("solar_forecast.demo.pipeline._fetch_openmeteo", return_value=empty_weather):
        out = _run()["hourly"]
    assert "spectral_factor" in out.columns
    assert (out["spectral_factor"] > 0.5).all()
    assert (out["spectral_factor"] < 1.5).all()
