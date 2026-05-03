"""Features API — serve merged feature frames for a location."""

from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/features", tags=["features"])


@router.get("/{location_id}")
def get_features(
    location_id: int,
    start_utc: Optional[str] = Query(None, description="ISO start time UTC"),
    end_utc: Optional[str] = Query(None, description="ISO end time UTC"),
    horizon_hours: int = Query(72, description="Horizon hours if no explicit range"),
):
    """Return the merged feature frame for a location.

    Falls back through 4 tiers: CAMS+OM → OM+climatology → OM-only → demo.
    """
    try:
        from solar_forecast.features.builder import build_feature_frame
        df, tier = build_feature_frame(
            location_id=location_id,
            start_utc=start_utc,
            end_utc=end_utc,
            horizon_hours=horizon_hours,
        )
        if df.empty:
            raise HTTPException(status_code=404, detail="No feature data available")
        records = df.fillna(0).to_dict(orient="records")
        return {
            "location_id": location_id,
            "data_tier":   tier,
            "rows":        len(records),
            "features":    records,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
