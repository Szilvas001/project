"""Smoke test: dashboard module imports without errors."""

import importlib

import pytest


def test_dashboard_imports():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    mod = importlib.import_module("solar_forecast.dashboard.app")
    assert hasattr(mod, "main")


def test_demo_pipeline_imports():
    mod = importlib.import_module("solar_forecast.demo.pipeline")
    assert hasattr(mod, "run_demo_forecast")


def test_sqlite_manager_imports():
    mod = importlib.import_module("app.db.sqlite_manager")
    for fn in ("create_tables", "list_locations", "create_location",
               "get_location", "update_location", "delete_location",
               "save_forecast", "load_forecast", "seed_demo_location"):
        assert hasattr(mod, fn)
