from fastapi import APIRouter, HTTPException, status
from typing import List

from app.api.models import LocationCreate, LocationOut, LocationUpdate
from app.db import sqlite_manager as db

router = APIRouter(prefix="/locations", tags=["locations"])


@router.get("", response_model=List[LocationOut])
def list_locations():
    return db.list_locations()


@router.post("", response_model=LocationOut, status_code=status.HTTP_201_CREATED)
def create_location(payload: LocationCreate):
    try:
        return db.create_location(payload.dict())
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
    existing = db.get_location(location_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Location not found")
    return db.update_location(location_id, payload.dict(exclude_none=True))


@router.delete("/{location_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_location(location_id: int):
    if not db.delete_location(location_id):
        raise HTTPException(status_code=404, detail="Location not found")
