"""Pytest fixtures for Solar Forecast Pro tests."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    """Use a per-test SQLite file so tests don't pollute the real DB."""
    db_path = tmp_path / "test_solar.db"
    from app.db import sqlite_manager
    monkeypatch.setattr(sqlite_manager, "DB_PATH", db_path)
    sqlite_manager.create_tables()
    yield db_path
