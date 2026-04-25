"""Tests for the demo forecast pipeline (physics smoke tests)."""

from unittest.mock import patch

import pandas as pd
import pytest


def _empty_weather(*args, **kwargs):
    return pd.DataFrame()


@patch("solar_forecast.demo.pipeline._fetch_openmeteo", side_effect=_empty_weather)
def test_pipeline_runs_without_openmeteo(mock_fetch):
    """The pipeline must produce a non-empty result even with no live weather."""
    from solar_forecast.demo.pipeline import run_demo_forecast
    result = run_demo_forecast(
        lat=47.5, lon=19.0, altitude=120.0,
        capacity_kw=5.0, tilt=30.0, azimuth=180.0,
        technology="mono_si", horizon_days=2,
    )
    hourly = result["hourly"]
    assert not hourly.empty
    assert "power_kw" in hourly.columns
    assert "energy_kwh" in hourly.columns
    assert (hourly["power_kw"] >= 0).all()


@patch("solar_forecast.demo.pipeline._fetch_openmeteo", side_effect=_empty_weather)
def test_summary_keys_present(mock_fetch):
    from solar_forecast.demo.pipeline import run_demo_forecast
    result = run_demo_forecast(lat=47.5, lon=19.0, capacity_kw=5.0, horizon_days=2)
    s = result["summary"]
    for k in ("today_kwh", "tomorrow_kwh", "total_7d_kwh",
              "peak_power_kw", "capacity_factor_pct", "cloud_loss_pct"):
        assert k in s
        assert s[k] >= 0


def test_resolve_tilt_azimuth_defaults():
    from solar_forecast.demo.pipeline import _resolve_tilt_azimuth
    tilt, az = _resolve_tilt_azimuth(47.5, None, None)
    assert 30 < tilt < 50
    assert az == 180.0

    tilt2, az2 = _resolve_tilt_azimuth(-30.0, None, None)
    assert az2 == 0.0  # northern azimuth for southern hemisphere
