"""Forecast API routes — physics-first, AI optional."""

from __future__ import annotations

import io
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.api.models import ForecastOut, ForecastRequest, ForecastSummary, HourlyPoint
from app.db import sqlite_manager as db
from solar_forecast.demo.pipeline import run_demo_forecast

router = APIRouter(tags=["forecast"])


def _build_response(result: dict, location_id: Optional[int] = None) -> ForecastOut:
    hourly_df: pd.DataFrame = result["hourly"]

    hourly = [
        HourlyPoint(
            timestamp_utc=str(row.Index),
            ghi_wm2=float(row.ghi_wm2) if pd.notna(row.ghi_wm2) else 0.0,
            power_kw=float(row.power_kw) if pd.notna(row.power_kw) else 0.0,
            energy_kwh=float(row.energy_kwh) if pd.notna(row.energy_kwh) else 0.0,
            kt=float(row.kt) if hasattr(row, "kt") and pd.notna(row.kt) else None,
            t_cell_c=float(row.t_cell_c) if hasattr(row, "t_cell_c") and pd.notna(row.t_cell_c) else None,
        )
        for row in hourly_df.itertuples()
    ]

    s = result["summary"]
    summary = ForecastSummary(
        today_kwh=round(float(s.get("today_kwh", 0)), 3),
        tomorrow_kwh=round(float(s.get("tomorrow_kwh", 0)), 3),
        total_7d_kwh=round(float(s.get("total_7d_kwh", 0)), 3),
        peak_power_kw=round(float(s.get("peak_power_kw", 0)), 3),
        peak_hour_utc=str(s.get("peak_hour_utc", "")),
        capacity_factor_pct=round(float(s.get("capacity_factor_pct", 0)), 2),
        cloud_loss_pct=round(float(s.get("cloud_loss_pct", 0)), 2),
    )

    return ForecastOut(
        location_id=location_id,
        summary=summary,
        hourly=hourly,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@router.post("/forecast", response_model=ForecastOut)
def run_forecast(req: ForecastRequest):
    """Run a one-off forecast for any coordinates."""
    try:
        result = run_demo_forecast(
            lat=req.lat,
            lon=req.lon,
            altitude=req.altitude,
            capacity_kw=req.capacity_kw,
            tilt=req.tilt,
            azimuth=req.azimuth,
            technology=req.technology,
            horizon_days=req.horizon_days,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return _build_response(result)


@router.get("/forecast/{location_id}", response_model=ForecastOut)
def get_location_forecast(location_id: int, horizon_days: int = Query(7, ge=1, le=14)):
    """Run or retrieve cached forecast for a saved location."""
    loc = db.get_location(location_id)
    if not loc:
        raise HTTPException(status_code=404, detail="Location not found")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cached = db.load_forecast(location_id, today)
    if cached:
        hourly_df = pd.DataFrame(cached["payload"])
        if not hourly_df.empty:
            hourly_df.index = pd.to_datetime(hourly_df.get("timestamp_utc", hourly_df.index))
        return ForecastOut(
            location_id=location_id,
            summary=ForecastSummary(**cached["summary"]),
            hourly=[HourlyPoint(**p) for p in cached["payload"]],
            generated_at=cached["generated_at"],
        )

    try:
        result = run_demo_forecast(
            lat=loc["lat"],
            lon=loc["lon"],
            altitude=loc.get("altitude", 0.0),
            capacity_kw=loc.get("capacity_kw", 5.0),
            tilt=loc.get("tilt"),
            azimuth=loc.get("azimuth"),
            technology=loc.get("technology", "mono_si"),
            horizon_days=horizon_days,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    out = _build_response(result, location_id=location_id)

    payload_list = [h.dict() for h in out.hourly]
    db.save_forecast(location_id, today, payload_list, out.summary.dict())

    return out


@router.get("/export/csv")
def export_csv(location_id: int, date: Optional[str] = None):
    """Download forecast as CSV."""
    loc = db.get_location(location_id)
    if not loc:
        raise HTTPException(status_code=404, detail="Location not found")

    target_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cached = db.load_forecast(location_id, target_date)

    if not cached:
        raise HTTPException(
            status_code=404,
            detail="No forecast cached for this location/date. Call GET /forecast/{id} first."
        )

    df = pd.DataFrame(cached["payload"])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    filename = f"forecast_{loc['name'].replace(' ', '_')}_{target_date}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
