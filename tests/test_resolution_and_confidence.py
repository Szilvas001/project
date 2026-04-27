"""
Tests for forecast resolution (hourly vs 15-minute) and the rule-based
confidence indicator that replaced the cloud-loss-derived placeholder.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from solar_forecast.demo.pipeline import (
    _confidence_score,
    run_demo_forecast,
)


@pytest.fixture
def empty_weather():
    return pd.DataFrame()


# ─── Resolution ───────────────────────────────────────────────────────────

def test_15min_mode_produces_4x_steps(empty_weather):
    with patch("solar_forecast.demo.pipeline._fetch_openmeteo", return_value=empty_weather):
        h = run_demo_forecast(lat=47.5, lon=19.0, capacity_kw=5.0,
                              horizon_days=2, resolution="hourly")
        q = run_demo_forecast(lat=47.5, lon=19.0, capacity_kw=5.0,
                              horizon_days=2, resolution="15min")

    assert len(h["hourly"]) == 48
    assert len(q["hourly"]) == 192
    assert h["summary"]["resolution"] == "hourly"
    assert q["summary"]["resolution"] == "15min"
    assert h["summary"]["timestep_hours"] == pytest.approx(1.0)
    assert q["summary"]["timestep_hours"] == pytest.approx(0.25)


def test_15min_energy_sums_match_hourly_within_tolerance(empty_weather):
    """Total energy at 15-min resolution should match hourly within ~5 %."""
    with patch("solar_forecast.demo.pipeline._fetch_openmeteo", return_value=empty_weather):
        h = run_demo_forecast(lat=47.5, lon=19.0, capacity_kw=5.0,
                              horizon_days=2, resolution="hourly")
        q = run_demo_forecast(lat=47.5, lon=19.0, capacity_kw=5.0,
                              horizon_days=2, resolution="15min")

    e_hourly = h["hourly"]["energy_kwh"].sum()
    e_15min  = q["hourly"]["energy_kwh"].sum()
    assert e_hourly > 0
    rel = abs(e_15min - e_hourly) / e_hourly
    assert rel < 0.05, f"15-min total deviates {rel*100:.2f}% from hourly"


def test_invalid_resolution_rejected(empty_weather):
    with patch("solar_forecast.demo.pipeline._fetch_openmeteo", return_value=empty_weather):
        with pytest.raises(ValueError):
            run_demo_forecast(lat=47.5, lon=19.0, capacity_kw=5.0,
                              horizon_days=1, resolution="seconds")


# ─── Energy = power × dt ───────────────────────────────────────────────────

def test_energy_equals_power_times_timestep(empty_weather):
    """energy_kwh must be power_kw × timestep_hours, not raw power."""
    import numpy as np
    with patch("solar_forecast.demo.pipeline._fetch_openmeteo", return_value=empty_weather):
        q = run_demo_forecast(lat=47.5, lon=19.0, capacity_kw=5.0,
                              horizon_days=1, resolution="15min")
    h = q["hourly"]
    expected = h["power_kw"].values * 0.25
    assert np.allclose(h["energy_kwh"].values, expected, atol=1e-9)


# ─── Confidence indicator ─────────────────────────────────────────────────

def test_confidence_high_when_everything_present():
    pct, label, reasons = _confidence_score(
        weather_available=True, cloud_data_available=True,
        ai_used=True, sr_custom=True, cams_used=True,
    )
    assert label == "High"
    assert pct >= 75


def test_confidence_low_in_pure_demo_mode():
    pct, label, reasons = _confidence_score(
        weather_available=False, cloud_data_available=False,
        ai_used=False, sr_custom=False, cams_used=False,
    )
    assert label == "Low"
    assert pct < 50
    assert any("climatological fallback" in r for r in reasons)


def test_confidence_in_summary(empty_weather):
    with patch("solar_forecast.demo.pipeline._fetch_openmeteo", return_value=empty_weather):
        s = run_demo_forecast(lat=47.5, lon=19.0, capacity_kw=5.0,
                              horizon_days=1)["summary"]
    assert "confidence_pct" in s
    assert "confidence_label" in s
    assert "confidence_reasons" in s
    assert s["confidence_label"] in ("Low", "Medium", "High")


def test_forecast_api_response_includes_confidence():
    """API response schema must surface the confidence triple."""
    from fastapi.testclient import TestClient
    from app.api.main import app

    def fake_run(**_):
        idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        df = pd.DataFrame({
            "ghi_wm2": [0.0, 100.0], "power_kw": [0.0, 2.0],
            "energy_kwh": [0.0, 2.0], "kt": [None, None],
            "t_cell_c": [25.0, 25.0], "cloud_cover_frac": [0.1, 0.1],
        }, index=idx)
        return {
            "hourly": df,
            "summary": {
                "today_kwh": 2.0, "tomorrow_kwh": 0.0, "total_7d_kwh": 2.0,
                "peak_power_kw": 2.0, "peak_hour_utc": str(idx[1]),
                "capacity_factor_pct": 20.0, "cloud_loss_pct": 0.0,
                "confidence_pct": 85.0, "confidence_label": "High",
                "confidence_reasons": ["Live weather forecast available"],
                "resolution": "hourly",
            },
            "clearsky_hourly": df, "location": {},
        }

    client = TestClient(app)
    with patch("app.api.routes.forecast.run_demo_forecast", side_effect=fake_run):
        r = client.post("/forecast", json={"lat": 47.5, "lon": 19.0, "capacity_kw": 5.0})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["confidence_pct"] == 85.0
    assert body["summary"]["confidence_label"] == "High"
    assert body["summary"]["confidence_reasons"]
    assert body["summary"]["resolution"] == "hourly"


def test_demo_runs_without_cams_or_keys(empty_weather):
    """Acceptance: 100% offline demo path must succeed end-to-end."""
    with patch("solar_forecast.demo.pipeline._fetch_openmeteo", return_value=empty_weather):
        out = run_demo_forecast(lat=47.5, lon=19.0, capacity_kw=5.0, horizon_days=2)
    assert not out["hourly"].empty
    assert out["summary"]["confidence_label"] in ("Low", "Medium", "High")
