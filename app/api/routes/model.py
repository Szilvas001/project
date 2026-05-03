"""Model management API — version registry, status, retraining triggers."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter(prefix="/model", tags=["model"])


class ModelVersionOut(BaseModel):
    id: int
    model_type: str
    version: str
    path: str
    r2: Optional[float]
    rmse: Optional[float]
    n_features: Optional[int]
    trained_at: str


class ModelStatusOut(BaseModel):
    kt_model_available: bool
    kt_model_path: Optional[str]
    kt_model_size_kb: Optional[int]
    ghi_model_available: bool
    ghi_model_path: Optional[str]
    ghi_model_size_kb: Optional[int]
    registered_versions: int
    latest_kt: Optional[ModelVersionOut]
    latest_ghi: Optional[ModelVersionOut]


@router.get("/status", response_model=ModelStatusOut)
def get_model_status():
    """Return availability and version info for all ML models."""
    kt_path  = Path("models/kt_xgb.joblib")
    ghi_path = Path("models/ghi_historical.joblib")

    latest_kt  = None
    latest_ghi = None
    total_versions = 0

    try:
        from solar_forecast.db.manager import get_model_versions
        all_versions = get_model_versions()
        total_versions = len(all_versions)
        kt_vers  = [v for v in all_versions if v["model_type"] == "kt_xgb"]
        ghi_vers = [v for v in all_versions if v["model_type"] == "ghi_historical"]
        if kt_vers:
            latest_kt = ModelVersionOut(**kt_vers[0])
        if ghi_vers:
            latest_ghi = ModelVersionOut(**ghi_vers[0])
    except Exception as exc:
        log.debug("model version query failed: %s", exc)

    return ModelStatusOut(
        kt_model_available=kt_path.exists(),
        kt_model_path=str(kt_path) if kt_path.exists() else None,
        kt_model_size_kb=int(kt_path.stat().st_size / 1024) if kt_path.exists() else None,
        ghi_model_available=ghi_path.exists(),
        ghi_model_path=str(ghi_path) if ghi_path.exists() else None,
        ghi_model_size_kb=int(ghi_path.stat().st_size / 1024) if ghi_path.exists() else None,
        registered_versions=total_versions,
        latest_kt=latest_kt,
        latest_ghi=latest_ghi,
    )


@router.get("/versions", response_model=list[ModelVersionOut])
def list_model_versions(
    model_type: Optional[str] = Query(None, description="Filter by model_type (kt_xgb, ghi_historical)"),
    limit: int = Query(20, ge=1, le=100),
):
    """Return registered model versions, newest first."""
    try:
        from solar_forecast.db.manager import get_model_versions
        versions = get_model_versions(model_type)[:limit]
        return [ModelVersionOut(**v) for v in versions]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
