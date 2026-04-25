"""Tests for the SQLite multi-location manager."""

import pytest

from app.db import sqlite_manager as db


def test_create_and_list_location():
    new = db.create_location({
        "name": "Test Site",
        "lat": 47.5, "lon": 19.0,
        "capacity_kw": 10.0,
        "tilt": 30.0, "azimuth": 180.0,
        "technology": "mono_si",
        "timezone": "Europe/Budapest",
    })
    assert new["id"] > 0
    assert new["name"] == "Test Site"

    rows = db.list_locations()
    names = [r["name"] for r in rows]
    assert "Test Site" in names


def test_get_location():
    created = db.create_location({"name": "Foo", "lat": 0.0, "lon": 0.0})
    fetched = db.get_location(created["id"])
    assert fetched is not None
    assert fetched["name"] == "Foo"


def test_update_location():
    created = db.create_location({"name": "Bar", "lat": 1.0, "lon": 2.0})
    updated = db.update_location(created["id"], {"capacity_kw": 25.0})
    assert updated["capacity_kw"] == 25.0


def test_delete_location():
    created = db.create_location({"name": "Tmp", "lat": 0.0, "lon": 0.0})
    assert db.delete_location(created["id"]) is True
    assert db.get_location(created["id"]) is None


def test_create_requires_name_lat_lon():
    with pytest.raises(ValueError):
        db.create_location({"lat": 0.0, "lon": 0.0})


def test_seed_demo_location():
    # No locations to start; seed should add Budapest demo
    db.seed_demo_location()
    rows = db.list_locations()
    assert any("Budapest" in r["name"] for r in rows)
