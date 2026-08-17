import pytest

from src.detection.base import Detection
from src.evaluation.matching import GroundTruthBox, greedy_match, intersection_over_union
from src.evaluation.metrics import calculate_detection_metrics


def vehicle(box: tuple[float, float, float, float], score: float = 0.9) -> Detection:
    return Detection(bbox_xyxy=box, score=score, source_class="car")


def test_intersection_over_union_handles_overlap_and_non_overlap() -> None:
    assert intersection_over_union((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert intersection_over_union((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert intersection_over_union((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)


def test_greedy_match_keeps_only_highest_score_duplicate() -> None:
    ground_truth = [GroundTruthBox("vehicle-001", (0, 0, 10, 10))]
    predictions = [vehicle((0, 0, 10, 10), score=0.8), vehicle((1, 1, 9, 9), score=0.95)]

    matches = greedy_match(predictions, ground_truth, iou_threshold=0.5)

    assert len(matches) == 1
    assert matches[0].prediction_index == 1
    assert matches[0].ground_truth_index == 0
    assert matches[0].iou == pytest.approx(0.64)


def test_calculate_detection_metrics_reports_compensating_count_errors() -> None:
    ground_truth = [
        GroundTruthBox("vehicle-001", (0, 0, 10, 10)),
        GroundTruthBox("vehicle-002", (20, 20, 30, 30)),
        GroundTruthBox("vehicle-003", (40, 40, 50, 50)),
    ]
    predictions = [
        vehicle((0, 0, 10, 10)),
        vehicle((20, 20, 30, 30)),
        vehicle((60, 60, 70, 70)),
    ]

    metrics = calculate_detection_metrics(predictions, ground_truth)

    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.absolute_count_error == 0
