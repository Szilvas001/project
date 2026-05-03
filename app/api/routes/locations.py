"""Location CRUD with pagination."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from app.api.models import LocationCreate, LocationOut, LocationUpdate, PaginatedLocations
from app.db import sqlite_manager as db

router = APIRouter(prefix="/locations", tags=["locations"])


@router.get("", response_model=PaginatedLocations)
def list_locations(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=200, description="Items per page"),
    search: Optional[str] = Query(None, description="Filter by name substring"),
):
    """List all saved locations with optional pagination and name search."""
    all_locs: list = db.list_locations()

    if search:
        s = search.lower()
        all_locs = [l for l in all_locs if s in l.get("name", "").lower()]

    total = len(all_locs)
    start = (page - 1) * per_page
    page_items = all_locs[start: start + per_page]

    return PaginatedLocations(
        total=total,
        page=page,
        per_page=per_page,
        items=[LocationOut(**loc) for loc in page_items],
    )


@router.post("", response_model=LocationOut, status_code=status.HTTP_201_CREATED)
def create_location(payload: LocationCreate):
    try:
        return db.create_location(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{location_id}", response_model=LocationOut)
def get_location(location_id: int):
    loc = db.get_location(location_id)
    if not loc:
        raise HTTPException(status_code=404, detail="Location not found")
    return loc


@router.patch("/{location_id}", response_model=LocationOut)
def update_location(location_id: int, payload: LocationUpdate):
    if not db.get_location(location_id):
        raise HTTPException(status_code=404, detail="Location not found")
    return db.update_location(location_id, payload.model_dump(exclude_none=True))


@router.delete("/{location_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_location(location_id: int):
    if not db.delete_location(location_id):
        raise HTTPException(status_code=404, detail="Location not found")
