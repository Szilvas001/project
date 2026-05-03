"""Ingestion API routes — trigger CAMS/OM data collection and status checks."""

from __future__ import annotations
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter(prefix="/ingestion", tags=["ingestion"])


# ── Request / response models ──────────────────────────────────────────────

class BackfillRequest(BaseModel):
    location_id: int
    days: int = 365
    dry_run: bool = False
    lat: Optional[float] = None
    lon: Optional[float] = None


class LiveFetchRequest(BaseModel):
    location_id: int
    hours: int = 12
    dry_run: bool = False
    force: bool = False
    lat: Optional[float] = None
    lon: Optional[float] = None


class OpenMeteoRequest(BaseModel):
    location_id: int
    hours: int = 72
    dry_run: bool = False
    lat: Optional[float] = None
    lon: Optional[float] = None


class IngestionStatus(BaseModel):
    cams_configured: bool
    last_cams_run: Optional[str]
    last_om_run: Optional[str]
    cams_rows: int
    om_rows: int


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/cams/backfill")
def trigger_cams_backfill(req: BackfillRequest, background: BackgroundTasks):
    """Trigger historical CAMS backfill for a location (runs in background)."""
    try:
        from solar_forecast.ingestion.cams.client import is_cams_configured
        if not is_cams_configured() and not req.dry_run:
            raise HTTPException(
                status_code=422,
                detail="CAMS credentials not configured. Set CADS_KEY or CAMS_API_KEY.",
            )
    except ImportError:
        pass

    def _run():
        try:
            from solar_forecast.ingestion.cams.backfill import run_backfill
            stats = run_backfill(
                location_id=req.location_id,
                days=req.days,
                dry_run=req.dry_run,
                lat=req.lat,
                lon=req.lon,
            )
            log.info("backfill complete for location %d: %s", req.location_id, stats)
        except Exception as exc:
            log.error("backfill error: %s", exc)

    background.add_task(_run)
    return {
        "status": "accepted",
        "message": f"CAMS backfill started for location {req.location_id} ({req.days} days)",
        "dry_run": req.dry_run,
    }


@router.post("/cams/live")
def trigger_cams_live(req: LiveFetchRequest):
    """Fetch the latest CAMS forecast for a location (synchronous)."""
    try:
        from solar_forecast.ingestion.cams.live import run_live
        status = run_live(
            location_id=req.location_id,
            hours=req.hours,
            dry_run=req.dry_run,
            lat=req.lat,
            lon=req.lon,
            force=req.force,
        )
        return {"status": "ok", "result": status}
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/openmeteo/live")
def trigger_openmeteo_live(req: OpenMeteoRequest):
    """Fetch and store Open-Meteo forecast for a location (synchronous)."""
    try:
        from solar_forecast.ingestion.openmeteo_live import run_openmeteo_live
        status = run_openmeteo_live(
            location_id=req.location_id,
            hours=req.hours,
            dry_run=req.dry_run,
            lat=req.lat,
            lon=req.lon,
        )
        if status.get("error"):
            raise HTTPException(status_code=502, detail=status["error"])
        return {"status": "ok", "result": status}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/status")
def ingestion_status(location_id: Optional[int] = Query(None)):
    """Return ingestion status: CAMS credentials, last run times, row counts."""
    try:
        from solar_forecast.ingestion.cams.client import is_cams_configured
        cams_ok = is_cams_configured()
    except Exception:
        cams_ok = False

    last_cams = None
    last_om = None
    cams_rows = 0
    om_rows = 0

    try:
        from solar_forecast.db.manager import get_connection, create_tables
        create_tables()
        with get_connection() as conn:
            q_cams = "SELECT MAX(ingested_at), COUNT(*) FROM cams_atmospheric_forecast"
            q_om   = "SELECT MAX(ingested_at), COUNT(*) FROM openmeteo_forecast"
            if location_id is not None:
                q_cams += " WHERE location_id = ?"
                q_om   += " WHERE location_id = ?"
                r1 = conn.execute(q_cams, (location_id,)).fetchone()
                r2 = conn.execute(q_om, (location_id,)).fetchone()
            else:
                r1 = conn.execute(q_cams).fetchone()
                r2 = conn.execute(q_om).fetchone()
            if r1:
                last_cams = r1[0]
                cams_rows = r1[1] or 0
            if r2:
                last_om = r2[0]
                om_rows = r2[1] or 0
    except Exception as exc:
        log.debug("status DB query failed: %s", exc)

    return IngestionStatus(
        cams_configured=cams_ok,
        last_cams_run=last_cams,
        last_om_run=last_om,
        cams_rows=cams_rows,
        om_rows=om_rows,
    )
