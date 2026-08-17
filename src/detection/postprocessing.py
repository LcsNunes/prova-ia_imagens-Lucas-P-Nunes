"""Post-processing rules applied after every detector reaches the vehicle ontology."""

from __future__ import annotations

from src.detection.base import BoundingBox, Detection


def suppress_vehicle_duplicates(
    detections: list[Detection], iou_threshold: float = 0.5, containment_threshold: float = 0.8
) -> list[Detection]:
    """Keep the highest-confidence box when boxes overlap or one substantially contains another."""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in the interval (0, 1].")
    if not 0.0 < containment_threshold <= 1.0:
        raise ValueError("containment_threshold must be in the interval (0, 1].")

    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda detection: detection.score, reverse=True):
        if all(
            _iou(candidate.bbox_xyxy, existing.bbox_xyxy) < iou_threshold
            and _containment(candidate.bbox_xyxy, existing.bbox_xyxy) < containment_threshold
            for existing in kept
        ):
            kept.append(candidate)
    return kept


def _iou(first: BoundingBox, second: BoundingBox) -> float:
    intersection = _intersection_area(first, second)
    union = _area(first) + _area(second) - intersection
    return intersection / union if union > 0.0 else 0.0


def _containment(first: BoundingBox, second: BoundingBox) -> float:
    smaller_area = min(_area(first), _area(second))
    return _intersection_area(first, second) / smaller_area if smaller_area > 0.0 else 0.0


def _intersection_area(first: BoundingBox, second: BoundingBox) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _area(bbox: BoundingBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
