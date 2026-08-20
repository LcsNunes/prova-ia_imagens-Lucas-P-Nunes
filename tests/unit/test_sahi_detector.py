from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.detection import sahi_detector
from src.detection.sahi_detector import SahiVehicleDetector


class FakeAutoDetectionModel:
    calls: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, **kwargs: object) -> object:
        cls.calls.append(kwargs)
        return object()


def prediction(name: str, score: float, bbox: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        category=SimpleNamespace(name=name),
        score=SimpleNamespace(value=score),
        bbox=SimpleNamespace(to_xyxy=lambda: bbox),
    )


def test_sahi_detector_normalizes_and_merges_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAutoDetectionModel.calls = []
    monkeypatch.setattr(sahi_detector, "AutoDetectionModel", FakeAutoDetectionModel)
    captured: dict[str, object] = {}

    def fake_sliced_prediction(*args: object, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            object_prediction_list=[
                prediction("small vehicle", 0.9, [1, 2, 11, 12]),
                prediction("ship", 0.8, [20, 20, 30, 30]),
            ]
        )

    monkeypatch.setattr(sahi_detector, "get_sliced_prediction", fake_sliced_prediction)
    detector = SahiVehicleDetector(
        Path("unused.pt"), "aerial", 0.25, 1024, 512, 0.2, device="cpu", merge_iou=0.5
    )

    detections = detector.predict(np.zeros((32, 32, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].source_class == "small vehicle"
    assert captured["slice_height"] == 512
    assert captured["postprocess_class_agnostic"] is True
    assert FakeAutoDetectionModel.calls[0]["device"] == "cpu"


@pytest.mark.parametrize("overlap, merge_iou", [(-0.1, 0.5), (1.0, 0.5), (0.2, 0.0)])
def test_sahi_detector_rejects_invalid_postprocess_configuration(
    monkeypatch: pytest.MonkeyPatch, overlap: float, merge_iou: float
) -> None:
    monkeypatch.setattr(sahi_detector, "AutoDetectionModel", FakeAutoDetectionModel)

    with pytest.raises(ValueError):
        SahiVehicleDetector(Path("unused.pt"), "aerial", 0.25, 1024, 512, overlap, "cpu", merge_iou)


def test_sahi_detector_validates_rgb_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sahi_detector, "AutoDetectionModel", FakeAutoDetectionModel)
    detector = SahiVehicleDetector(Path("unused.pt"), "aerial", 0.25, 1024, 512, 0.2, "cpu")

    with pytest.raises(ValueError, match="RGB image"):
        detector.predict(np.zeros((32, 32), dtype=np.uint8))
