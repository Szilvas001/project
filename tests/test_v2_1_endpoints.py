"""Tests for v2.1.0 API features: model status, pagination, confidence, energy_kwh."""

from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def _mock_forecast(*args, **kwargs):
    times = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({
        "ghi_wm2":       [0.0] * 6 + [400.0] * 12 + [0.0] * 6,
        "poa_wm2":       [0.0] * 6 + [450.0] * 12 + [0.0] * 6,
        "power_kw":      [0.0] * 6 + [3.0]   * 12 + [0.0] * 6,
        "energy_kwh":    [0.0] * 6 + [3.0]   * 12 + [0.0] * 6,
        "kt":            [None] * 24,
        "t_cell_c":      [25.0] * 24,
        "cloud_cover_frac": [0.2] * 24,
        "energy_kwh_cs": [3.5] * 24,
        "spectral_mm":   [1.02] * 24,
        "iam":           [0.98] * 24,
    }, index=times)
    return {
        "hourly": df,
        "summary": {
            "today_kwh": 36.0, "tomorrow_kwh": 36.0,
            "total_7d_kwh": 252.0,
            "peak_power_kw": 3.0, "peak_hour_utc": str(times[12]),
            "capacity_factor_pct": 25.0, "cloud_loss_pct": 8.0,
        },
        "clearsky_hourly": df,
        "location": {},
    }


# ── /model/status ──────────────────────────────────────────────────────────

def test_model_status_returns_ok():
    r = client.get("/model/status")
    assert r.status_code == 200
    body = r.json()
    assert "kt_model_available" in body
    assert "ghi_model_available" in body
    assert "registered_versions" in body
    assert isinstance(body["registered_versions"], int)


def test_model_versions_empty_list():
    r = client.get("/model/versions")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_model_versions_filter_by_type():
    r = client.get("/model/versions?model_type=kt_xgb")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# ── /locations pagination ──────────────────────────────────────────────────

def test_locations_paginated_response_shape():
    r = client.get("/locations")
    assert r.status_code == 200
    body = r.json()
    assert "items" in body
    assert "total" in body
    assert "page" in body
    assert "per_page" in body


def test_locations_pagination_params():
    # Create two locations
    for name in ("Alpha", "Beta"):
        client.post("/locations", json={
            "name": name, "lat": 47.5, "lon": 19.0, "capacity_kw": 5.0,
        })

    r = client.get("/locations?page=1&per_page=1")
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) == 1
    assert body["per_page"] == 1


def test_locations_search_filter():
    client.post("/locations", json={"name": "Szeged Solar", "lat": 46.2, "lon": 20.1, "capacity_kw": 3.0})
    client.post("/locations", json={"name": "Budapest Grid", "lat": 47.5, "lon": 19.0, "capacity_kw": 5.0})

    r = client.get("/locations?search=Szeged")
    assert r.status_code == 200
    body = r.json()
    assert all("Szeged" in item["name"] for item in body["items"])


# ── /forecast with iam_model + denorm_factor ───────────────────────────────

@patch("app.api.routes.forecast.run_demo_forecast", side_effect=_mock_forecast)
def test_forecast_with_iam_and_denorm(mock_fn):
    r = client.post("/forecast", json={
        "lat": 47.5, "lon": 19.0, "capacity_kw": 5.0,
        "iam_model": "martin_ruiz", "denorm_factor": 0.9,
    })
    assert r.status_code == 200
    body = r.json()
    assert "summary" in body
    assert "hourly" in body
    # Verify iam_model and denorm_factor were passed through
    call_kwargs = mock_fn.call_args[1]
    assert call_kwargs.get("iam_model") == "martin_ruiz"
    assert call_kwargs.get("denorm_factor") == 0.9


@patch("app.api.routes.forecast.run_demo_forecast", side_effect=_mock_forecast)
def test_forecast_invalid_iam_model_rejected(mock_fn):
    r = client.post("/forecast", json={
        "lat": 47.5, "lon": 19.0, "capacity_kw": 5.0,
        "iam_model": "BOGUS_MODEL",
    })
    assert r.status_code == 422


@patch("app.api.routes.forecast.run_demo_forecast", side_effect=_mock_forecast)
def test_forecast_confidence_in_response(mock_fn):
    r = client.post("/forecast", json={"lat": 47.5, "lon": 19.0, "capacity_kw": 5.0})
    assert r.status_code == 200
    body = r.json()
    if body.get("confidence"):
        c = body["confidence"]
        assert "confidence_pct" in c
        assert "confidence_label" in c
        assert c["confidence_label"] in ("High", "Medium", "Low")
        assert isinstance(c["confidence_reasons"], list)


# ── energy_kwh formula validation ─────────────────────────────────────────

@patch("app.api.routes.forecast.run_demo_forecast", side_effect=_mock_forecast)
def test_energy_kwh_equals_power_times_one_hour(mock_fn):
    r = client.post("/forecast", json={"lat": 47.5, "lon": 19.0, "capacity_kw": 5.0})
    assert r.status_code == 200
    for pt in r.json()["hourly"]:
        # energy_kwh must equal power_kw × 1h (within floating point tolerance)
        assert abs(pt["energy_kwh"] - pt["power_kw"]) < 1e-6, (
            f"energy_kwh={pt['energy_kwh']} != power_kw={pt['power_kw']}"
        )


# ── /health v2.1.0 ────────────────────────────────────────────────────────

def test_health_version_and_engine():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == "2.1.0"
    assert "engine" in body
    assert "SPECTRL2" in body["engine"]
