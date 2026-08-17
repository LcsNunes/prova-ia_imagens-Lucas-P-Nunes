from pathlib import Path

import numpy as np

from src.detection.base import Detection
from src.evaluation.matching import GroundTruthBox
from src.visualization.draw import draw_detections, draw_ground_truth, save_rgb_image


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
