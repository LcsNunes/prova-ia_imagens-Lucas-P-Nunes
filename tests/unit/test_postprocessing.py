import pytest

from src.detection.base import Detection
from src.detection.postprocessing import suppress_vehicle_duplicates


def vehicle(box: tuple[float, float, float, float], score: float) -> Detection:
    return Detection(box, score, "small vehicle")


def test_suppress_vehicle_duplicates_keeps_highest_score_for_contained_box() -> None:
    detections = [vehicle((10, 10, 20, 20), 0.7), vehicle((11, 11, 19, 19), 0.9)]

    kept = suppress_vehicle_duplicates(detections)

    assert kept == [detections[1]]


def test_suppress_vehicle_duplicates_preserves_nearby_distinct_vehicles() -> None:
    detections = [vehicle((0, 0, 10, 10), 0.9), vehicle((12, 0, 22, 10), 0.8)]

    assert suppress_vehicle_duplicates(detections) == detections


@pytest.mark.parametrize("iou, containment", [(0.0, 0.8), (0.5, 0.0), (1.1, 0.8)])
def test_suppress_vehicle_duplicates_validates_thresholds(iou: float, containment: float) -> None:
    with pytest.raises(ValueError):
        suppress_vehicle_duplicates([], iou, containment)
