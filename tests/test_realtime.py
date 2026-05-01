"""Tests for the real-time forecast endpoint and run_realtime_forecast()."""

from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def _empty_weather(*args, **kwargs):
    return pd.DataFrame()


# ── run_realtime_forecast (unit) ──────────────────────────────────────────

@patch("solar_forecast.demo.pipeline._fetch_openmeteo", side_effect=_empty_weather)
def test_realtime_pipeline_returns_curve(mock_fetch):
    from solar_forecast.demo.pipeline import run_realtime_forecast
    result = run_realtime_forecast(
        lat=47.5, lon=19.0, altitude=120.0,
        capacity_kw=5.0, tilt=30.0, azimuth=180.0,
        technology="mono_si", horizon_hours=6, resolution_minutes=15,
    )
    assert "curve" in result
    assert "now_power_kw" in result
    assert "now_utc" in result
    curve = result["curve"]
    assert not curve.empty
    assert "power_kw" in curve.columns
    assert len(curve) == 6 * 4   # 6h × 4 per hour


@patch("solar_forecast.demo.pipeline._fetch_openmeteo", side_effect=_empty_weather)
def test_realtime_power_non_negative(mock_fetch):
    from solar_forecast.demo.pipeline import run_realtime_forecast
    result = run_realtime_forecast(lat=47.5, lon=19.0, capacity_kw=5.0, horizon_hours=4)
    assert (result["curve"]["power_kw"] >= 0).all()
    assert result["now_power_kw"] >= 0


@patch("solar_forecast.demo.pipeline._fetch_openmeteo", side_effect=_empty_weather)
def test_realtime_atmosphere_key(mock_fetch):
    from solar_forecast.demo.pipeline import run_realtime_forecast
    result = run_realtime_forecast(lat=47.5, lon=19.0, capacity_kw=5.0, horizon_hours=2)
    assert "atmosphere" in result
    assert "source" in result["atmosphere"]


@patch("solar_forecast.demo.pipeline._fetch_openmeteo", side_effect=_empty_weather)
def test_realtime_30min_resolution(mock_fetch):
    from solar_forecast.demo.pipeline import run_realtime_forecast
    result = run_realtime_forecast(
        lat=47.5, lon=19.0, capacity_kw=5.0,
        horizon_hours=4, resolution_minutes=30,
    )
    assert len(result["curve"]) == 4 * 2  # 4h × 2 per hour


# ── POST /forecast/realtime (API) ─────────────────────────────────────────

def _mock_realtime_result(*args, **kwargs):
    import numpy as np
    times = pd.date_range("2025-06-01", periods=4, freq="15min", tz="UTC")
    curve = pd.DataFrame({
        "ghi_wm2":          [0, 100, 300, 200],
        "ghi_clear_wm2":    [0, 120, 350, 250],
        "poa_wm2":          [0, 110, 320, 210],
        "power_kw":         [0, 1.5, 4.2, 2.8],
        "energy_kwh":       [0, 0.375, 1.05, 0.7],
        "kt":               [0, 0.83, 0.86, 0.80],
        "t_cell_c":         [20, 30, 40, 35],
        "cloud_cover_frac": [0.1, 0.2, 0.15, 0.25],
    }, index=times)
    curve.index.name = "timestamp_utc"
    return {
        "curve": curve,
        "now_power_kw": 2.1,
        "now_utc": "2025-06-01T12:00:00+00:00",
        "atmosphere": {"source": "climatology"},
        "location": {"lat": 47.5, "lon": 19.0},
    }


@patch("app.api.routes.forecast.run_realtime_forecast", side_effect=_mock_realtime_result)
def test_realtime_api_200(mock_fn):
    r = client.post("/forecast/realtime", json={
        "lat": 47.5, "lon": 19.0, "capacity_kw": 5.0,
        "horizon_hours": 1, "resolution_minutes": 15,
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert "now_power_kw" in data
    assert "curve" in data
    assert len(data["curve"]) == 4
    assert data["curve"][0]["power_kw"] >= 0


@patch("app.api.routes.forecast.run_realtime_forecast", side_effect=_mock_realtime_result)
def test_realtime_api_curve_fields(mock_fn):
    r = client.post("/forecast/realtime", json={
        "lat": 47.5, "lon": 19.0, "capacity_kw": 5.0,
        "horizon_hours": 1, "resolution_minutes": 15,
    })
    point = r.json()["curve"][0]
    assert "timestamp_utc" in point
    assert "ghi_wm2" in point
    assert "power_kw" in point


def test_realtime_api_invalid_lat():
    r = client.post("/forecast/realtime", json={
        "lat": 999, "lon": 19.0, "capacity_kw": 5.0,
    })
    assert r.status_code == 422
