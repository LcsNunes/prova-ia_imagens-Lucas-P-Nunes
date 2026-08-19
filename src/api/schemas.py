"""Contratos estaveis de resposta expostos pela aplicacao FastAPI."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Status operacional minimo para uso local e health checks do Docker."""

    status: str = "ok"
    model_loaded: bool


class AnalysisResponse(BaseModel):
    """Metadados da analise e overlays JPEG prontos para exibicao."""

    vehicle_count: int = Field(ge=0)
    model_name: str
    sahi_enabled: bool
    confidence: float = Field(ge=0, le=1)
    inference_ms: float = Field(ge=0)
    vehicle_inference_ms: float = Field(ge=0)
    road_inference_ms: float = Field(ge=0)
    road_method: str
    detections_image: str
    combined_image: str
    roads_image: str
