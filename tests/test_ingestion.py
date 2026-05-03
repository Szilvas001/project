"""Tests for the new ingestion package and DB layer.

All external I/O is mocked — no CAMS API calls, no Open-Meteo calls,
no real DB connections beyond the temp SQLite set up in conftest.py.
"""

from __future__ import annotations
import types
import pytest
import pandas as pd
import numpy as np


# ── CAMS variable mapping ──────────────────────────────────────────────────

def test_cams_variable_map_basic():
    from solar_forecast.ingestion.cams.variables import map_row, CAMS_VARIABLES
    raw = {spec["cams_short"]: 0.1 for spec in CAMS_VARIABLES.values()}
    result = map_row(raw)
    assert "aod_550" in result
    assert "total_column_ozone" in result
    assert result["aod_550"] == pytest.approx(0.1)


def test_cams_variable_map_missing_uses_default():
    from solar_forecast.ingestion.cams.variables import map_row
    result = map_row({})
    # aod_550 has default 0.12
    assert result["aod_550"] == pytest.approx(0.12)


def test_cams_variable_map_missing_no_default_is_none():
    from solar_forecast.ingestion.cams.variables import map_row
    result = map_row({})
    # pm25 has no default
    assert result["pm25"] is None


def test_cams_variable_map_nan_stored_as_none():
    from solar_forecast.ingestion.cams.variables import map_row, CAMS_VARIABLES
    raw = {"aod550": float("nan")}
    result = map_row(raw)
    assert result["aod_550"] is None or result["aod_550"] != result["aod_550"]  # nan or None


def test_cams_long_names_count():
    from solar_forecast.ingestion.cams.variables import CAMS_LONG_NAMES, CAMS_VARIABLES
    assert len(CAMS_LONG_NAMES) == len(CAMS_VARIABLES)
    assert "total_aerosol_optical_depth_550nm" in CAMS_LONG_NAMES


def test_cams_climatology_defaults_not_empty():
    from solar_forecast.ingestion.cams.variables import get_climatology_defaults
    defaults = get_climatology_defaults()
    assert len(defaults) > 0
    assert "aod_550" in defaults
    assert defaults["aod_550"] == pytest.approx(0.12)


# ── CAMS parser ────────────────────────────────────────────────────────────

def test_pivot_to_wide_basic():
    from solar_forecast.ingestion.cams.parser import pivot_to_wide
    df_long = pd.DataFrame({
        "run_time_utc":        [pd.Timestamp("2024-01-01 00:00", tz="UTC")] * 2,
        "valid_time_utc":      [pd.Timestamp("2024-01-01 00:00", tz="UTC"),
                                pd.Timestamp("2024-01-01 01:00", tz="UTC")],
        "forecast_step_hours": [0, 1],
        "variable_cams":       ["aod550", "aod550"],
        "value":               [0.10, 0.11],
    })
    wide = pivot_to_wide(df_long)
    assert "aod_550" in wide.columns or "aod550" in wide.columns


def test_bilinear_interp_returns_float():
    from solar_forecast.ingestion.cams.parser import bilinear_interp
    lats = np.array([[47.0, 47.0], [48.0, 48.0]])
    lons = np.array([[13.0, 14.0], [13.0, 14.0]])
    vals = np.array([[0.1, 0.2], [0.15, 0.25]])
    result = bilinear_interp(lats, lons, vals, 47.5, 13.5)
    assert isinstance(result, float)
    assert 0.0 < result < 1.0


# ── DB manager ────────────────────────────────────────────────────────────

def test_db_create_tables(tmp_path, monkeypatch):
    import solar_forecast.db.manager as mgr
    monkeypatch.setattr(mgr, "DB_PATH", tmp_path / "test.db")
    mgr.create_tables()
    with mgr.get_connection() as con:
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    required = {"locations", "cams_atmospheric_forecast", "openmeteo_forecast",
                "model_feature_frame", "ingestion_runs", "forecast_runs"}
    assert required.issubset(tables)


def test_upsert_cams_dedup(tmp_path, monkeypatch):
    import solar_forecast.db.manager as mgr
    monkeypatch.setattr(mgr, "DB_PATH", tmp_path / "test.db")
    mgr.create_tables()
    # Add a location first
    mgr.upsert_location("TestLoc", 47.0, 13.0)

    df = pd.DataFrame({
        "run_time_utc":        ["2024-01-01T00:00:00+00:00"],
        "valid_time_utc":      ["2024-01-01T01:00:00+00:00"],
        "forecast_step_hours": [1],
        "aod_550":             [0.10],
    })
    n1 = mgr.upsert_cams(df, location_id=1)
    n2 = mgr.upsert_cams(df, location_id=1)   # duplicate — must be ignored
    assert n1 == 1
    assert n2 == 0                              # dedup


def test_upsert_openmeteo_dedup(tmp_path, monkeypatch):
    import solar_forecast.db.manager as mgr
    monkeypatch.setattr(mgr, "DB_PATH", tmp_path / "test.db")
    mgr.create_tables()
    mgr.upsert_location("TestLoc", 47.0, 13.0)
    df = pd.DataFrame({"valid_time_utc": ["2024-01-01T00:00:00"], "cloud_cover": [50.0]})
    n1 = mgr.upsert_openmeteo(df, location_id=1)
    n2 = mgr.upsert_openmeteo(df, location_id=1)
    assert n1 == 1
    assert n2 == 0


# ── Open-Meteo live fetcher ────────────────────────────────────────────────

def test_openmeteo_fetch_mocked(monkeypatch):
    from solar_forecast.ingestion import openmeteo_live

    fake_data = {
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
            "temperature_2m": [10.0, 11.0],
            "relative_humidity_2m": [80.0, 79.0],
            "dew_point_2m": [7.0, 8.0],
            "apparent_temperature": [9.0, 10.0],
            "precipitation": [0.0, 0.0],
            "cloud_cover": [30.0, 25.0],
            "cloud_cover_low": [10.0, 8.0],
            "cloud_cover_mid": [10.0, 9.0],
            "cloud_cover_high": [10.0, 8.0],
            "wind_speed_10m": [3.0, 4.0],
            "wind_direction_10m": [270.0, 271.0],
            "shortwave_radiation": [200.0, 250.0],
            "direct_radiation": [150.0, 190.0],
            "diffuse_radiation": [50.0, 60.0],
            "direct_normal_irradiance": [300.0, 350.0],
        }
    }

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return fake_data

    monkeypatch.setattr("solar_forecast.ingestion.openmeteo_live.requests.get",
                        lambda *a, **kw: FakeResp())

    df = openmeteo_live.fetch_openmeteo(47.0, 13.0, hours=2)
    assert df is not None
    assert len(df) == 2
    assert "cloud_cover" in df.columns
    assert "valid_time_utc" in df.columns


# ── Feature frame builder ──────────────────────────────────────────────────

def test_feature_frame_demo_tier(tmp_path, monkeypatch):
    import solar_forecast.db.manager as mgr
    monkeypatch.setattr(mgr, "DB_PATH", tmp_path / "test.db")
    mgr.create_tables()

    from solar_forecast.features.builder import build_feature_frame
    df, tier = build_feature_frame(
        location_id=999,
        start_utc="2024-01-01T06:00:00",
        end_utc="2024-01-01T12:00:00",
    )
    assert tier == "demo"
    assert len(df) > 0
    assert "aod_550" in df.columns


def test_feature_frame_has_angstrom(tmp_path, monkeypatch):
    import solar_forecast.db.manager as mgr
    monkeypatch.setattr(mgr, "DB_PATH", tmp_path / "test.db")
    mgr.create_tables()

    from solar_forecast.features.builder import build_feature_frame
    df, tier = build_feature_frame(
        location_id=999,
        start_utc="2024-01-01T06:00:00",
        end_utc="2024-01-01T10:00:00",
    )
    assert "angstrom_exponent" in df.columns


# ── Confidence model ───────────────────────────────────────────────────────

def test_confidence_cams_increases_score():
    from solar_forecast.engine.confidence import compute_confidence
    demo = compute_confidence(atmosphere_source="climatology")
    cams = compute_confidence(atmosphere_source="cams")
    assert cams["confidence_pct"] > demo["confidence_pct"]


def test_confidence_labels():
    from solar_forecast.engine.confidence import compute_confidence
    c = compute_confidence(atmosphere_source="cams", has_openmeteo=True, use_ai=True,
                           has_historical_model=True, horizon_days=7)
    assert c["confidence_label"] in ("High", "Very High")
    assert len(c["confidence_reasons"]) > 0


def test_confidence_demo_is_low():
    from solar_forecast.engine.confidence import compute_confidence
    c = compute_confidence(atmosphere_source="climatology", has_openmeteo=False)
    assert c["confidence_pct"] < 55
