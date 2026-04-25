from fastapi import APIRouter
from datetime import datetime, timezone

router = APIRouter()


@router.get("/health", tags=["system"])
def health_check():
    return {
        "status": "ok",
        "version": "2.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
