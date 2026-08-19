"""Metricas de avaliacao voltadas ao negocio para existencia de veiculos."""

from __future__ import annotations

from dataclasses import dataclass

from src.detection.base import Detection
from src.evaluation.matching import GroundTruthBox, Match, greedy_match


@dataclass(frozen=True)
class DetectionMetrics:
    """Metricas derivadas do pareamento um-para-um por IoU em limiar fixo."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    absolute_count_error: int
    matches: tuple[Match, ...]


def calculate_detection_metrics(
    predictions: list[Detection], ground_truth: list[GroundTruthBox], iou_threshold: float = 0.5
) -> DetectionMetrics:
    """Calcula a qualidade sem tratar subclasses de veiculos como rotulos distintos."""
    matches = greedy_match(predictions, ground_truth, iou_threshold)
    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1 = 2 * precision * recall / f1_denominator if f1_denominator else 0.0

    return DetectionMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        absolute_count_error=abs(len(predictions) - len(ground_truth)),
        matches=tuple(matches),
    )
