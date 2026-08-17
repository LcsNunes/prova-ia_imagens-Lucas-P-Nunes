"""Consistent RGB visualizations used by experiments and the web application."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.detection.base import Detection
from src.evaluation.matching import GroundTruthBox

DETECTION_COLOR = (0, 229, 255)
GROUND_TRUTH_COLOR = (255, 193, 7)


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return an RGB image with vehicle boxes, scores, and OBB polygons when available."""
    canvas = image.copy()
    for detection in detections:
        _draw_box(canvas, detection.bbox_xyxy, DETECTION_COLOR)
        if detection.obb_points is not None:
            points = np.array(detection.obb_points, dtype=np.int32)
            cv2.polylines(canvas, [points], isClosed=True, color=DETECTION_COLOR, thickness=2)
        x1, y1, _, _ = (int(value) for value in detection.bbox_xyxy)
        cv2.putText(
            canvas,
            f"vehicle {detection.score:.2f}",
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            DETECTION_COLOR,
            1,
            cv2.LINE_AA,
        )
    return canvas


def draw_ground_truth(image: np.ndarray, ground_truth: list[GroundTruthBox]) -> np.ndarray:
    """Return an RGB image with manually reviewed vehicle boxes."""
    canvas = image.copy()
    for ground_truth_box in ground_truth:
        _draw_box(canvas, ground_truth_box.bbox_xyxy, GROUND_TRUTH_COLOR)
    return canvas


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    """Persist an RGB visualization, creating its destination directory when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write image to {path}")


def _draw_box(
    image: np.ndarray, bbox_xyxy: tuple[float, float, float, float], color: tuple[int, int, int]
) -> None:
    x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
