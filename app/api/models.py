"""Pydantic models for the Solar Forecast Pro API."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, validator


class LocationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    altitude: float = Field(0.0, ge=0.0, le=8848.0)
    capacity_kw: float = Field(5.0, gt=0.0, le=10_000.0)
    tilt: Optional[float] = Field(None, ge=0.0, le=90.0)
    azimuth: Optional[float] = Field(None, ge=0.0, le=360.0)
    technology: str = Field("mono_si")
    timezone: str = Field("UTC")

    @validator("technology")
    def validate_tech(cls, v):
        valid = {"mono_si", "poly_si", "cdte", "cigs", "hit"}
        if v not in valid:
            raise ValueError(f"technology must be one of {valid}")
        return v


class LocationUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    lon: Optional[float] = Field(None, ge=-180.0, le=180.0)
    altitude: Optional[float] = Field(None, ge=0.0)
    capacity_kw: Optional[float] = Field(None, gt=0.0)
    tilt: Optional[float] = Field(None, ge=0.0, le=90.0)
    azimuth: Optional[float] = Field(None, ge=0.0, le=360.0)
    technology: Optional[str] = None
    timezone: Optional[str] = None


class LocationOut(BaseModel):
    id: int
    name: str
    lat: float
    lon: float
    altitude: float
    capacity_kw: float
    tilt: Optional[float]
    azimuth: Optional[float]
    technology: str
    timezone: str
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class ForecastRequest(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    altitude: float = Field(0.0, ge=0.0)
    capacity_kw: float = Field(5.0, gt=0.0)
    tilt: Optional[float] = Field(None, ge=0.0, le=90.0)
    azimuth: Optional[float] = Field(None, ge=0.0, le=360.0)
    technology: str = Field("mono_si")
    timezone: str = Field("UTC")
    horizon_days: int = Field(7, ge=1, le=14)
    use_ai: bool = Field(False)


class HourlyPoint(BaseModel):
    timestamp_utc: str
    ghi_wm2: float
    power_kw: float
    energy_kwh: float
    kt: Optional[float]
    t_cell_c: Optional[float]


class ForecastSummary(BaseModel):
    today_kwh: float
    tomorrow_kwh: float
    total_7d_kwh: float
    peak_power_kw: float
    peak_hour_utc: str
    capacity_factor_pct: float
    cloud_loss_pct: float
    location: Optional[dict[str, Any]] = None


class ForecastOut(BaseModel):
    location_id: Optional[int] = None
    summary: ForecastSummary
    hourly: list[HourlyPoint]
    generated_at: str
