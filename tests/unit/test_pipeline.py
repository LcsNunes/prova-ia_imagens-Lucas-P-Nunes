from pathlib import Path

import numpy as np
import pytest

from src.detection.base import Detection
from src.pipeline import VehicleAnalysisPipeline


class FakeDetector:
    def predict(self, image: np.ndarray) -> list[Detection]:
        assert image.shape == (32, 32, 3)
        return [Detection((4, 4, 12, 12), 0.9, "small vehicle")]


def test_pipeline_returns_visual_outputs_and_metadata() -> None:
    pipeline = VehicleAnalysisPipeline(
        detector=FakeDetector(),
        model_name="test.pt",
        confidence=0.25,
        sahi_enabled=True,
        road_config={
            "hsv_lower": [0, 0, 50],
            "hsv_upper": [179, 40, 200],
            "kernel_size": 3,
            "min_area": 10,
        },
    )
    image = np.full((32, 32, 3), 120, dtype=np.uint8)

    result = pipeline.analyze(image)

    assert result.vehicle_count == 1
    assert result.model_name == "test.pt"
    assert result.sahi_enabled is True
    assert result.inference_ms >= 0
    assert result.annotated_image.shape == image.shape
    assert result.roads.overlay.shape == image.shape


def test_pipeline_fails_with_helpful_message_when_weight_is_missing(tmp_path: Path) -> None:
    configuration = {
        "model": {"name": "missing.pt", "domain": "aerial", "confidence": 0.25, "imgsz": 640},
        "runtime": {"device": "cpu"},
        "sahi": {"enabled": False},
        "roads": {},
    }

    with pytest.raises(FileNotFoundError, match="download_models"):
        VehicleAnalysisPipeline.from_config(configuration, tmp_path)


def test_pipeline_device_environment_overrides_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "models" / "weight.pt"
    model_path.parent.mkdir()
    model_path.write_bytes(b"weight placeholder")
    received_options: dict[str, object] = {}

    class CapturingDetector:
        def __init__(self, **options: object) -> None:
            received_options.update(options)

    monkeypatch.setattr("src.pipeline.YoloVehicleDetector", CapturingDetector)
    monkeypatch.setenv("DEVICE", "cuda:0")
    configuration = {
        "model": {
            "name": "weight.pt",
            "domain": "aerial",
            "confidence": 0.25,
            "imgsz": 640,
        },
        "runtime": {"device": "cpu"},
        "sahi": {"enabled": False},
        "roads": {},
    }

    VehicleAnalysisPipeline.from_config(configuration, tmp_path)

    assert received_options["device"] == "cuda:0"
