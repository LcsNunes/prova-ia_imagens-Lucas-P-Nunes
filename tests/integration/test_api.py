from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import create_app
from src.detection.base import Detection
from src.pipeline import AnalysisResult
from src.roads.opencv_road import RoadHighlight


class FakePipeline:
    def analyze(self, image: np.ndarray) -> AnalysisResult:
        return AnalysisResult(
            detections=(Detection((1, 1, 5, 5), 0.9, "small vehicle"),),
            annotated_image=image,
            combined_image=image,
            roads=RoadHighlight(mask=np.zeros(image.shape[:2], dtype=np.uint8), overlay=image),
            inference_ms=12.5,
            vehicle_inference_ms=10.0,
            road_inference_ms=2.0,
            model_name="fake-obb.pt",
            confidence=0.25,
            sahi_enabled=True,
            road_method="mask2former",
        )


def image_payload() -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.full((10, 10, 3), 127, dtype=np.uint8)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_api_analyzes_image_and_serializes_overlays() -> None:
    app = create_app(
        pipeline=FakePipeline(),
        configuration={"api": {"max_upload_mb": 1, "max_pixels": 1_000}},
    )
    with TestClient(app) as client:
        home = client.get("/")
        health = client.get("/api/health")
        response = client.post(
            "/api/analyses",
            files={"image": ("scene.jpg", image_payload(), "image/jpeg")},
        )

    assert home.status_code == 200
    assert "Vista" in home.text
    assert "combined-image" in home.text
    assert "Mask2Former" in home.text
    assert health.json() == {"status": "ok", "model_loaded": True}
    assert response.status_code == 200
    assert response.json()["vehicle_count"] == 1
    assert response.json()["detections_image"].startswith("data:image/jpeg;base64,")
    assert response.json()["combined_image"].startswith("data:image/jpeg;base64,")
    assert response.json()["roads_image"].startswith("data:image/jpeg;base64,")


def test_api_rejects_invalid_image() -> None:
    app = create_app(
        pipeline=FakePipeline(),
        configuration={"api": {"max_upload_mb": 1, "max_pixels": 1_000}},
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/analyses",
            files={"image": ("not-image.txt", b"invalid", "text/plain")},
        )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_image"
