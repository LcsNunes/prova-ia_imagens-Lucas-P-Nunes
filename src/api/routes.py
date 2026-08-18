"""Synchronous image-analysis endpoints."""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Annotated

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from PIL import Image

from src.api.image_validation import ImageValidationError, decode_rgb_image
from src.api.schemas import AnalysisResponse, HealthResponse
from src.pipeline import VehicleAnalysisPipeline

LOGGER = logging.getLogger(__name__)
router = APIRouter()
ImageUpload = Annotated[UploadFile, File(...)]


@router.get("/api/health", response_model=HealthResponse)
def read_health(request: Request) -> HealthResponse:
    """Report whether the configured pipeline has completed its local startup."""
    return HealthResponse(model_loaded=request.app.state.pipeline is not None)


@router.post("/api/analyses", response_model=AnalysisResponse, status_code=status.HTTP_200_OK)
async def create_analysis(request: Request, image: ImageUpload) -> AnalysisResponse:
    """Validate one upload and return vehicle and road visualisations synchronously."""
    content = await image.read(request.app.state.max_upload_bytes + 1)
    try:
        rgb_image = decode_rgb_image(
            content,
            max_upload_bytes=request.app.state.max_upload_bytes,
            max_pixels=request.app.state.max_pixels,
        )
    except ImageValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": "invalid_image", "message": str(error)},
        ) from error

    pipeline: VehicleAnalysisPipeline | None = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "model_unavailable", "message": "O modelo ainda nao esta disponivel."},
        )

    LOGGER.info("Image received for analysis: pixels=%s", rgb_image.shape[0] * rgb_image.shape[1])
    try:
        result = pipeline.analyze(rgb_image)
    except Exception as error:
        LOGGER.exception("Image analysis failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "analysis_failed", "message": "Nao foi possivel analisar a imagem."},
        ) from error

    return AnalysisResponse(
        vehicle_count=result.vehicle_count,
        model_name=result.model_name,
        sahi_enabled=result.sahi_enabled,
        confidence=result.confidence,
        inference_ms=result.inference_ms,
        vehicle_inference_ms=result.vehicle_inference_ms,
        road_inference_ms=result.road_inference_ms,
        road_method=result.road_method,
        detections_image=_to_jpeg_data_url(result.annotated_image),
        combined_image=_to_jpeg_data_url(result.combined_image),
        roads_image=_to_jpeg_data_url(result.roads.overlay),
    )


def _to_jpeg_data_url(image: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
