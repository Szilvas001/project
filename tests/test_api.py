"""FastAPI route tests using TestClient (no live network)."""

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
        "ghi_wm2":     [0.0] * 6 + [400.0] * 12 + [0.0] * 6,
        "poa_wm2":     [0.0] * 6 + [450.0] * 12 + [0.0] * 6,
        "power_kw":    [0.0] * 6 + [3.0]   * 12 + [0.0] * 6,
        "energy_kwh":  [0.0] * 6 + [3.0]   * 12 + [0.0] * 6,
        "kt":          [None] * 24,
        "t_cell_c":    [25.0] * 24,
        "cloud_cover_frac": [0.2] * 24,
        "energy_kwh_cs":    [3.5] * 24,
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


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_locations_crud():
    # Create
    r = client.post("/locations", json={
        "name": "API Test", "lat": 47.5, "lon": 19.0,
        "capacity_kw": 5.0, "technology": "mono_si",
    })
    assert r.status_code == 201, r.text
    loc = r.json()
    assert loc["name"] == "API Test"

    # List
    r = client.get("/locations")
    assert r.status_code == 200
    assert any(l["name"] == "API Test" for l in r.json())

    # Get
    r = client.get(f"/locations/{loc['id']}")
    assert r.status_code == 200

    # Patch
    r = client.patch(f"/locations/{loc['id']}", json={"capacity_kw": 7.5})
    assert r.status_code == 200
    assert r.json()["capacity_kw"] == 7.5

    # Delete
    r = client.delete(f"/locations/{loc['id']}")
    assert r.status_code == 204


def test_invalid_technology_rejected():
    r = client.post("/locations", json={
        "name": "Bad", "lat": 0.0, "lon": 0.0, "technology": "FOOBAR",
    })
    assert r.status_code == 422


@patch("app.api.routes.forecast.run_demo_forecast", side_effect=_mock_forecast)
def test_forecast_endpoint(mock_fn):
    r = client.post("/forecast", json={
        "lat": 47.5, "lon": 19.0, "capacity_kw": 5.0, "horizon_days": 1,
    })
    assert r.status_code == 200
    body = r.json()
    assert "summary" in body
    assert "hourly" in body
    assert body["summary"]["today_kwh"] == 36.0
    assert len(body["hourly"]) == 24
    # Power must be non-negative
    assert all(p["power_kw"] >= 0 for p in body["hourly"])
