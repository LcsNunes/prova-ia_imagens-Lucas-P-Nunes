from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.detection import yolo_detector
from src.detection.yolo_detector import YoloVehicleDetector, resolve_device


class FakeYoloModel:
    """Small boundary fake that returns a prepared Ultralytics-like result."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> list[object]:
        self.calls.append(kwargs)
        return [self.result]


def tensor(values: object) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def test_resolve_device_honors_auto_and_explicit_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yolo_detector.torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda:0"
    assert resolve_device("cpu") == "cpu"

    monkeypatch.setattr(yolo_detector.torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="not available"):
        resolve_device("cuda:0")
    with pytest.raises(ValueError, match="device must"):
        resolve_device("metal")


def test_axis_aligned_prediction_filters_non_vehicle_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boxes = SimpleNamespace(
        xyxy=tensor([[1, 2, 11, 12], [20, 20, 30, 30]]),
        conf=tensor([0.9, 0.8]),
        cls=tensor([2, 0]),
    )
    result = SimpleNamespace(boxes=boxes, obb=None, names={0: "person", 2: "car"})
    fake_model = FakeYoloModel(result)
    monkeypatch.setattr(yolo_detector, "YOLO", lambda _: fake_model)
    detector = YoloVehicleDetector(Path("unused.pt"), "coco", 0.25, 640, 0.7, device="cpu")

    detections = detector.predict(np.zeros((32, 32, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].bbox_xyxy == (1.0, 2.0, 11.0, 12.0)
    assert detections[0].source_class == "car"
    assert fake_model.calls[0]["device"] == "cpu"


def test_oriented_prediction_preserves_polygon_points(monkeypatch: pytest.MonkeyPatch) -> None:
    obb = SimpleNamespace(
        xyxy=tensor([[1, 2, 11, 12]]),
        conf=tensor([0.9]),
        cls=tensor([0]),
        xyxyxyxy=tensor([[[1, 2], [11, 2], [11, 12], [1, 12]]]),
    )
    result = SimpleNamespace(boxes=None, obb=obb, names={0: "small-vehicle"})
    fake_model = FakeYoloModel(result)
    monkeypatch.setattr(yolo_detector, "YOLO", lambda _: fake_model)
    detector = YoloVehicleDetector(Path("unused.pt"), "aerial", 0.25, 640, 0.7, device="cpu")

    detections = detector.predict(np.zeros((32, 32, 3), dtype=np.uint8))

    assert detections[0].obb_points == ((1.0, 2.0), (11.0, 2.0), (11.0, 12.0), (1.0, 12.0))


def test_detector_validates_input_and_empty_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    empty_result = SimpleNamespace(boxes=None, obb=None, names={})
    fake_model = FakeYoloModel(empty_result)
    monkeypatch.setattr(yolo_detector, "YOLO", lambda _: fake_model)
    detector = YoloVehicleDetector(Path("unused.pt"), "coco", 0.25, 640, 0.7, device="cpu")

    with pytest.raises(ValueError, match="RGB image"):
        detector.predict(np.zeros((32, 32), dtype=np.uint8))
    assert detector.predict(np.zeros((32, 32, 3), dtype=np.uint8)) == []
