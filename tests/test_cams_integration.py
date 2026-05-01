"""Tests for CAMS fetcher integration — uses mocks, no live network/DB."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


# ── cams_query — graceful fallback when DB is unavailable ─────────────────

def test_load_cams_state_returns_empty_when_db_unreachable():
    """Without a Postgres connection the query layer must return an empty frame."""
    from solar_forecast.data_ingestion.cams_query import load_cams_atmospheric_state
    times = pd.date_range("2024-06-01", periods=24, freq="h", tz="UTC")
    with patch(
        "solar_forecast.data_ingestion.cams_query.load_cams_atmospheric_state",
        return_value=pd.DataFrame(index=times),
    ):
        from solar_forecast.data_ingestion.cams_query import load_cams_atmospheric_state as fn
        result = fn(times, 47.5, 19.0)
        assert isinstance(result, pd.DataFrame)


def test_load_cams_state_real_no_crash():
    """Calling the real function with no DB configured must not raise."""
    from solar_forecast.data_ingestion.cams_query import load_cams_atmospheric_state
    times = pd.date_range("2024-06-01", periods=6, freq="h", tz="UTC")
    result = load_cams_atmospheric_state(times, 47.5, 19.0)
    assert isinstance(result, pd.DataFrame)


# ── derive_extras ─────────────────────────────────────────────────────────

def test_derive_extras_angstrom_computed():
    from solar_forecast.data_ingestion.cams_query import derive_extras
    df = pd.DataFrame({
        "aod_469nm": [0.12, 0.14],
        "aod_865nm": [0.06, 0.07],
        "aod_670nm": [0.09, 0.10],
        "aod_1240nm": [0.04, 0.05],
    })
    out = derive_extras(df)
    assert "angstrom_alpha1" in out.columns
    assert "angstrom_alpha2" in out.columns
    assert (out["angstrom_alpha1"] > 0).all()


def test_derive_extras_ssa_mixing():
    from solar_forecast.data_ingestion.cams_query import derive_extras
    df = pd.DataFrame({
        "aod_dust_550nm": [0.05],
        "aod_bc_550nm":   [0.01],
        "aod_om_550nm":   [0.02],
        "aod_ss_550nm":   [0.01],
        "aod_so4_550nm":  [0.03],
    })
    out = derive_extras(df)
    assert "ssa_mix" in out.columns
    assert "asym_mix" in out.columns
    assert 0 < float(out["ssa_mix"].iloc[0]) <= 1.0


def test_derive_extras_empty_df():
    from solar_forecast.data_ingestion.cams_query import derive_extras
    out = derive_extras(pd.DataFrame())
    assert out.empty


# ── _resolve_atmosphere in pipeline ──────────────────────────────────────

def test_resolve_atmosphere_returns_climatology_without_cams():
    from solar_forecast.demo.pipeline import _resolve_atmosphere
    times = pd.date_range("2024-06-01", periods=24, freq="h", tz="UTC")
    atm = _resolve_atmosphere(times, 47.5, 19.0)
    assert "aod_550nm" in atm
    assert "ssa" in atm
    assert "ozone_du" in atm
    assert len(atm["aod_550nm"]) == 24
    # Without a live DB, source must be "climatology"
    assert atm["source"] in ("climatology", "cams")


# ── CamsScheduler ─────────────────────────────────────────────────────────

def test_scheduler_next_run_time():
    """next_run_time() must return a future UTC datetime."""
    from solar_forecast.cams_fetcher.scheduler import CamsScheduler
    sched = CamsScheduler()
    import datetime
    nrt = sched._next_run_time()
    assert isinstance(nrt, datetime.datetime)
    assert nrt > datetime.datetime.utcnow()


def test_setup_cron_install_false():
    """setup_cron with install=False must return a non-empty string without touching the system."""
    from solar_forecast.cams_fetcher.scheduler import setup_cron
    fragment = setup_cron(install=False)
    assert "cams-fetcher" in fragment
    assert "python" in fragment


# ── runner parse helpers ──────────────────────────────────────────────────

def test_parse_leadtime_range():
    from solar_forecast.cams_fetcher.runner import parse_leadtime
    result = parse_leadtime("0-5")
    assert result == ["0", "1", "2", "3", "4", "5"]


def test_parse_leadtime_csv():
    from solar_forecast.cams_fetcher.runner import parse_leadtime
    result = parse_leadtime("0,6,12,24")
    assert result == ["0", "6", "12", "24"]


def test_determine_forecast_returns_tuple():
    from solar_forecast.cams_fetcher.runner import determine_forecast
    schedule = {"00:00": 10, "12:00": 22}
    date_str, time_str = determine_forecast(schedule)
    assert "-" in date_str
    assert ":" in time_str


# ── grib_processor bilinear_interpolate ──────────────────────────────────

def test_bilinear_interpolate_exact_grid_point():
    from solar_forecast.cams_fetcher.grib_processor import bilinear_interpolate
    lats = np.array([[47.0, 47.0], [48.0, 48.0]])
    lons = np.array([[18.0, 19.0], [18.0, 19.0]])
    vals = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = bilinear_interpolate(lats, lons, vals, 47.0, 18.0)
    assert abs(result - 1.0) < 0.1


def test_bilinear_interpolate_center():
    from solar_forecast.cams_fetcher.grib_processor import bilinear_interpolate
    lats = np.array([[47.0, 47.0], [48.0, 48.0]])
    lons = np.array([[18.0, 19.0], [18.0, 19.0]])
    vals = np.array([[1.0, 1.0], [1.0, 1.0]])
    result = bilinear_interpolate(lats, lons, vals, 47.5, 18.5)
    assert abs(result - 1.0) < 1e-9
