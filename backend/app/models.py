"""
ICE MIND - core data models.

These describe every object that flows through the pipeline:
satellite/environmental input -> iceberg state -> trajectory prediction ->
risk assessment -> route optimisation -> copilot explanation.
"""
from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class LatLon(BaseModel):
    lat: float
    lon: float


class Environment(BaseModel):
    """Environmental drivers used by the physics-informed drift model."""
    current_speed_kn: float = Field(..., description="Ocean current speed, knots")
    current_dir_deg: float = Field(..., description="Ocean current bearing, degrees true")
    wind_speed_kn: float
    wind_dir_deg: float
    wave_height_m: float
    sea_ice_concentration_pct: float = Field(..., ge=0, le=100)
    sea_temp_c: float


class Iceberg(BaseModel):
    id: str
    position: LatLon
    size_km: List[float] = Field(..., description="[length_km, width_km]")
    velocity_kn: float
    heading_deg: float
    detection_confidence: float = Field(..., ge=0, le=1)
    environment: Environment


class TrajectoryPoint(BaseModel):
    hours_ahead: float
    position: LatLon
    confidence: float = Field(..., ge=0, le=1)
    uncertainty_radius_km: float


class TrajectoryPrediction(BaseModel):
    iceberg_id: str
    generated_at_hours: float = 0.0
    points: List[TrajectoryPoint]
    method: str = "physics-guided-drift + persistence correction"


class Vessel(BaseModel):
    name: str = "M/V Endeavor"
    position: LatLon
    heading_deg: float
    speed_kn: float
    ice_class: str = "PC5"


class RiskFactor(BaseModel):
    label: str
    impact: Literal["Low", "Moderate", "High"]
    weight: float
    contribution: float


class RiskAssessment(BaseModel):
    iceberg_id: Optional[str] = None
    score: float = Field(..., ge=0, le=100)
    band: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    time_to_encounter_hours: Optional[float] = None
    factors: List[RiskFactor]
    narrative: str


class RiskTimelinePoint(BaseModel):
    hours_ahead: float
    score: float
    band: str


class RouteOption(BaseModel):
    id: str
    label: str
    waypoints: List[LatLon]
    distance_km: float
    eta_hours: float
    risk_score: float
    recommended: bool = False


class RouteRequest(BaseModel):
    start: LatLon
    destination: LatLon
    max_speed_kn: float = 14.0
    vessel_ice_class: str = "PC5"


class WhatIfRequest(BaseModel):
    speed_kn: float
    heading_deg: float
    hours_ahead: float = 0.0


class CopilotQuery(BaseModel):
    question: str


class CopilotResponse(BaseModel):
    answer: str
    risk_before: Optional[float] = None
    risk_after: Optional[float] = None
    recommended_action: Optional[str] = None
