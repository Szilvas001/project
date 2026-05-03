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

from app.api.models import (
    ConfidenceOut, ForecastOut, ForecastRequest, ForecastSummary,
    HourlyPoint, RealtimeOut, RealtimePoint, RealtimeRequest,
)
from app.db import sqlite_manager as db
from solar_forecast.demo.pipeline import run_demo_forecast, run_realtime_forecast

router = APIRouter(tags=["forecast"])


def _confidence(result: dict, req_tech: str, use_ai: bool) -> ConfidenceOut:
    try:
        from solar_forecast.engine.confidence import compute_confidence
        atm_src = result.get("atmosphere", {}).get("source", "climatology")
        c = compute_confidence(
            atmosphere_source=atm_src,
            has_openmeteo=not result["hourly"].empty,
            use_ai=use_ai,
            has_historical_model=Path("models/ghi_historical.joblib").exists(),
            technology=req_tech,
        )
        return ConfidenceOut(**c)
    except Exception:
        return ConfidenceOut(confidence_pct=65, confidence_label="Medium", confidence_reasons=[])


def _build_hourly(hourly_df: pd.DataFrame) -> list[HourlyPoint]:
    points = []
    for row in hourly_df.itertuples():
        # energy_kwh = power_kw × 1h for hourly data (already correctly stored)
        points.append(HourlyPoint(
            timestamp_utc=str(row.Index),
            ghi_wm2=float(getattr(row, "ghi_wm2", 0) or 0),
            power_kw=float(getattr(row, "power_kw", 0) or 0),
            energy_kwh=float(getattr(row, "energy_kwh", 0) or 0),
            kt=float(row.kt) if hasattr(row, "kt") and pd.notna(row.kt) else None,
            t_cell_c=float(row.t_cell_c) if hasattr(row, "t_cell_c") and pd.notna(row.t_cell_c) else None,
            spectral_mm=float(row.spectral_mm) if hasattr(row, "spectral_mm") and pd.notna(row.spectral_mm) else None,
            iam=float(row.iam) if hasattr(row, "iam") and pd.notna(row.iam) else None,
        ))
    return points


def _build_response(
    result: dict,
    location_id: Optional[int] = None,
    technology: str = "mono_si",
    use_ai: bool = False,
) -> ForecastOut:
    hourly_df: pd.DataFrame = result["hourly"]
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
        hourly=_build_hourly(hourly_df),
        confidence=_confidence(result, technology, use_ai),
        atmosphere=result.get("atmosphere"),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@router.post("/forecast", response_model=ForecastOut)
def run_forecast(req: ForecastRequest):
    """One-off forecast for any coordinates.

    All physics parameters (SR, IAM, denorm) are applied server-side.
    `energy_kwh` per row = `power_kw × 1h` for hourly resolution.
    """
    try:
        result = run_demo_forecast(
            lat=req.lat,
            lon=req.lon,
            altitude=req.altitude,
            capacity_kw=req.capacity_kw,
            tilt=req.tilt,
            azimuth=req.azimuth,
            technology=req.technology,
            iam_model=req.iam_model,
            horizon_days=req.horizon_days,
            use_ai=req.use_ai,
            denorm_factor=req.denorm_factor,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    out = _build_response(result, technology=req.technology, use_ai=req.use_ai)

    # Audit log (best-effort)
    try:
        from solar_forecast.db.manager import log_forecast_run
        log_forecast_run(
            location_id=None,
            horizon_hours=req.horizon_days * 24,
            data_tier=result.get("atmosphere", {}).get("source", "demo"),
            confidence_pct=out.confidence.confidence_pct if out.confidence else None,
        )
    except Exception:
        pass

    return out


@router.get("/forecast/{location_id}", response_model=ForecastOut)
def get_location_forecast(location_id: int, horizon_days: int = Query(7, ge=1, le=14)):
    """Run or retrieve cached forecast for a saved location."""
    loc = db.get_location(location_id)
    if not loc:
        raise HTTPException(status_code=404, detail="Location not found")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cached = db.load_forecast(location_id, today)
    if cached:
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
            iam_model="ashrae",
            horizon_days=horizon_days,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    out = _build_response(result, location_id=location_id, technology=loc.get("technology", "mono_si"))
    payload_list = [h.model_dump() for h in out.hourly]
    db.save_forecast(location_id, today, payload_list, out.summary.model_dump())
    return out


@router.get("/export/csv")
def export_csv(location_id: int, date: Optional[str] = None):
    """Download a cached forecast as CSV."""
    loc = db.get_location(location_id)
    if not loc:
        raise HTTPException(status_code=404, detail="Location not found")

    target_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cached = db.load_forecast(location_id, target_date)
    if not cached:
        raise HTTPException(
            status_code=404,
            detail="No forecast cached for this location/date. Call GET /forecast/{id} first.",
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


@router.post("/forecast/realtime", response_model=RealtimeOut)
def get_realtime_forecast(req: RealtimeRequest):
    """Sub-hourly real-time production estimate.

    Returns `now_power_kw` (interpolated to current UTC) and a fine-resolution
    production curve. `energy_kwh` per point = `power_kw × (resolution_minutes / 60)`.
    """
    try:
        result = run_realtime_forecast(
            lat=req.lat,
            lon=req.lon,
            altitude=req.altitude,
            capacity_kw=req.capacity_kw,
            tilt=req.tilt,
            azimuth=req.azimuth,
            technology=req.technology,
            iam_model=req.iam_model,
            resolution_minutes=req.resolution_minutes,
            horizon_hours=req.horizon_hours,
            use_ai_ghi=req.use_ai_ghi,
            ghi_model_path=req.ghi_model_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    curve_df: pd.DataFrame = result["curve"]
    res_h = req.resolution_minutes / 60.0

    curve_pts = [
        RealtimePoint(
            timestamp_utc=str(row.Index),
            ghi_wm2=float(getattr(row, "ghi_wm2", 0) or 0),
            ghi_clear_wm2=float(getattr(row, "ghi_clear_wm2", 0) or 0),
            poa_wm2=float(getattr(row, "poa_wm2", 0) or 0),
            power_kw=float(getattr(row, "power_kw", 0) or 0),
            energy_kwh=round(float(getattr(row, "power_kw", 0) or 0) * res_h, 6),
            kt=float(row.kt) if hasattr(row, "kt") and pd.notna(row.kt) else None,
            t_cell_c=float(row.t_cell_c) if hasattr(row, "t_cell_c") and pd.notna(row.t_cell_c) else None,
            cloud_cover_frac=float(row.cloud_cover_frac) if hasattr(row, "cloud_cover_frac") and pd.notna(row.cloud_cover_frac) else None,
        )
        for row in curve_df.itertuples()
    ]

    return RealtimeOut(
        now_power_kw=result["now_power_kw"],
        now_utc=result["now_utc"],
        curve=curve_pts,
        atmosphere=result.get("atmosphere"),
        location=result.get("location"),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
