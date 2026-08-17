"""IoU matching between normalized predictions and manual ground truth."""

from __future__ import annotations

from dataclasses import dataclass

from src.detection.base import BoundingBox, Detection


@dataclass(frozen=True)
class GroundTruthBox:
    """One manually reviewed vehicle box used for evaluation."""

    annotation_id: str
    bbox_xyxy: BoundingBox


@dataclass(frozen=True)
class Match:
    """A one-to-one prediction and ground-truth association."""

    prediction_index: int
    ground_truth_index: int
    iou: float


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    """Calculate axis-aligned IoU, returning zero for disjoint or invalid-area boxes."""
    first_width = max(0.0, first[2] - first[0])
    first_height = max(0.0, first[3] - first[1])
    second_width = max(0.0, second[2] - second[0])
    second_height = max(0.0, second[3] - second[1])
    if first_width == 0.0 or first_height == 0.0 or second_width == 0.0 or second_height == 0.0:
        return 0.0

    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    union = first_width * first_height + second_width * second_height - intersection
    return intersection / union if union > 0.0 else 0.0


def greedy_match(
    predictions: list[Detection], ground_truth: list[GroundTruthBox], iou_threshold: float
) -> list[Match]:
    """Match high-confidence predictions to one unmatched ground-truth box each."""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in the interval (0, 1].")

    unmatched_ground_truth = set(range(len(ground_truth)))
    matches: list[Match] = []
    for prediction_index in sorted(
        range(len(predictions)), key=lambda index: predictions[index].score, reverse=True
    ):
        candidates = [
            (
                intersection_over_union(
                    predictions[prediction_index].bbox_xyxy, ground_truth[index].bbox_xyxy
                ),
                index,
            )
            for index in unmatched_ground_truth
        ]
        if not candidates:
            break
        best_iou, ground_truth_index = max(
            candidates, key=lambda candidate: (candidate[0], -candidate[1])
        )
        if best_iou >= iou_threshold:
            matches.append(Match(prediction_index, ground_truth_index, best_iou))
            unmatched_ground_truth.remove(ground_truth_index)
    return matches
