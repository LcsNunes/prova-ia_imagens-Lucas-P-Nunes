"""Stable response contracts exposed by the FastAPI application."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Minimal operational status for local use and Docker health checks."""

    status: str = "ok"
    model_loaded: bool


class AnalysisResponse(BaseModel):
    """Analysis metadata and display-ready JPEG overlays."""

    vehicle_count: int = Field(ge=0)
    model_name: str
    sahi_enabled: bool
    confidence: float = Field(ge=0, le=1)
    inference_ms: float = Field(ge=0)
    detections_image: str
    roads_image: str
