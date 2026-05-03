"""Pydantic models for the Solar Forecast Pro API."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, validator

_VALID_TECH = {"mono_si", "poly_si", "cdte", "cigs", "hit"}
_VALID_IAM  = {"ashrae", "martin_ruiz", "fresnel"}


class LocationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    altitude: float = Field(0.0, ge=0.0, le=8848.0)
    capacity_kw: float = Field(5.0, gt=0.0, le=500_000.0)
    tilt: Optional[float] = Field(None, ge=0.0, le=90.0)
    azimuth: Optional[float] = Field(None, ge=0.0, le=360.0)
    technology: str = Field("mono_si")
    timezone: str = Field("UTC")

    @validator("technology")
    def validate_tech(cls, v):
        if v not in _VALID_TECH:
            raise ValueError(f"technology must be one of {_VALID_TECH}")
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


class PaginatedLocations(BaseModel):
    total: int
    page: int
    per_page: int
    items: list[LocationOut]


class ForecastRequest(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude (°)")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude (°)")
    altitude: float = Field(0.0, ge=0.0, le=8848.0, description="Altitude above sea level (m)")
    capacity_kw: float = Field(5.0, gt=0.0, le=500_000.0, description="Installed DC capacity (kW)")
    tilt: Optional[float] = Field(None, ge=0.0, le=90.0, description="Panel tilt (°); auto-computed if null")
    azimuth: Optional[float] = Field(None, ge=0.0, le=360.0, description="Panel azimuth (°); 180=South")
    technology: str = Field("mono_si", description="PV technology: mono_si, poly_si, cdte, cigs, hit")
    iam_model: str = Field("ashrae", description="IAM model: ashrae, martin_ruiz, fresnel")
    timezone: str = Field("UTC")
    horizon_days: int = Field(7, ge=1, le=14, description="Forecast horizon (days)")
    use_ai: bool = Field(False, description="Enable XGBoost Kt correction (requires trained model)")
    denorm_factor: float = Field(1.0, ge=0.5, le=2.0, description="Effective irradiance scale factor")

    @validator("technology")
    def validate_tech(cls, v):
        if v not in _VALID_TECH:
            raise ValueError(f"technology must be one of {_VALID_TECH}")
        return v

    @validator("iam_model")
    def validate_iam(cls, v):
        if v not in _VALID_IAM:
            raise ValueError(f"iam_model must be one of {_VALID_IAM}")
        return v


class HourlyPoint(BaseModel):
    timestamp_utc: str
    ghi_wm2: float
    power_kw: float
    energy_kwh: float
    kt: Optional[float] = None
    t_cell_c: Optional[float] = None
    spectral_mm: Optional[float] = None
    iam: Optional[float] = None


class ConfidenceOut(BaseModel):
    confidence_pct: int
    confidence_label: str
    confidence_reasons: list[str]


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
    confidence: Optional[ConfidenceOut] = None
    atmosphere: Optional[dict[str, Any]] = None
    generated_at: str


class RealtimePoint(BaseModel):
    timestamp_utc: str
    ghi_wm2: float
    ghi_clear_wm2: float
    poa_wm2: float
    power_kw: float
    energy_kwh: float
    kt: Optional[float] = None
    t_cell_c: Optional[float] = None
    cloud_cover_frac: Optional[float] = None


class RealtimeRequest(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    altitude: float = Field(0.0, ge=0.0)
    capacity_kw: float = Field(5.0, gt=0.0)
    tilt: Optional[float] = Field(None, ge=0.0, le=90.0)
    azimuth: Optional[float] = Field(None, ge=0.0, le=360.0)
    technology: str = Field("mono_si")
    iam_model: str = Field("ashrae")
    resolution_minutes: int = Field(15, ge=5, le=60)
    horizon_hours: int = Field(24, ge=1, le=72)
    use_ai_ghi: bool = Field(False)
    ghi_model_path: Optional[str] = Field(None)


class RealtimeOut(BaseModel):
    now_power_kw: float
    now_utc: str
    curve: list[RealtimePoint]
    atmosphere: Optional[dict[str, Any]] = None
    location: Optional[dict[str, Any]] = None
    generated_at: str
