"""Health check and system info endpoint."""

from datetime import datetime, timezone
from fastapi import APIRouter

router = APIRouter()

_VERSION = "2.1.0"


@router.get("/health", tags=["system"])
def health_check():
    """Returns API health status, version, and timestamp."""
    return {
        "status": "ok",
        "version": _VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine": "SPECTRL2 + CAMS + Perez + XGBoost",
    }
