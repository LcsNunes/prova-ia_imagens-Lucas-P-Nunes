from pathlib import Path

import numpy as np

from src.detection.base import Detection
from src.evaluation.matching import GroundTruthBox
from src.visualization.draw import (
    FALSE_NEGATIVE_COLOR,
    FALSE_POSITIVE_COLOR,
    TRUE_POSITIVE_COLOR,
    draw_detection_diagnostics,
    draw_detections,
    draw_ground_truth,
    save_rgb_image,
)


def test_draw_detections_changes_rgb_canvas_without_mutating_input() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    detection = Detection((4, 4, 20, 20), 0.9, "car")

    rendered = draw_detections(image, [detection])

    assert rendered.shape == image.shape
    assert np.array_equal(image, np.zeros((32, 32, 3), dtype=np.uint8))
    assert not np.array_equal(rendered, image)


def test_draw_ground_truth_and_save_rgb_image(tmp_path: Path) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    rendered = draw_ground_truth(image, [GroundTruthBox("vehicle-001", (4, 4, 20, 20))])
    output_path = tmp_path / "comparison.jpg"

    save_rgb_image(output_path, rendered)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_draw_detection_diagnostics_marks_true_false_positive_and_false_negative() -> None:
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    detections = [
        Detection((5, 5, 15, 15), 0.9, "small-vehicle"),
        Detection((35, 5, 45, 15), 0.8, "small-vehicle"),
    ]
    ground_truth = [
        GroundTruthBox("vehicle-001", (5, 5, 15, 15)),
        GroundTruthBox("vehicle-002", (5, 35, 15, 45)),
    ]

    rendered = draw_detection_diagnostics(image, detections, ground_truth, iou_threshold=0.50)

    assert np.array_equal(image, np.zeros((60, 60, 3), dtype=np.uint8))
    assert tuple(rendered[5, 5]) == TRUE_POSITIVE_COLOR
    assert tuple(rendered[5, 35]) == FALSE_POSITIVE_COLOR
    assert tuple(rendered[35, 5]) == FALSE_NEGATIVE_COLOR
