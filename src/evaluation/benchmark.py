"""Medicoes controladas de tempo e qualidade para comparar detectores."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from time import perf_counter

import numpy as np
import torch

from src.detection.base import Detection, Detector
from src.evaluation.matching import GroundTruthBox
from src.evaluation.metrics import DetectionMetrics, calculate_detection_metrics


@dataclass(frozen=True)
class BenchmarkResult:
    """Qualidade e tempo observados para uma configuracao fixa de detector."""

    name: str
    predicted_count: int
    metrics: DetectionMetrics
    median_inference_ms: float
    peak_gpu_memory_mb: float | None
    detections: tuple[Detection, ...]

    def as_dict(self) -> dict[str, float | int | str | None]:
        """Transforma o resultado em uma linha para DataFrame ou relatorio JSON."""
        return {
            "name": self.name,
            "predicted_count": self.predicted_count,
            "true_positives": self.metrics.true_positives,
            "false_positives": self.metrics.false_positives,
            "false_negatives": self.metrics.false_negatives,
            "precision": self.metrics.precision,
            "recall": self.metrics.recall,
            "f1": self.metrics.f1,
            "absolute_count_error": self.metrics.absolute_count_error,
            "median_inference_ms": self.median_inference_ms,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
        }


def benchmark_detector(
    name: str,
    detector: Detector,
    image: np.ndarray,
    ground_truth: list[GroundTruthBox],
    repetitions: int = 3,
    iou_threshold: float = 0.5,
) -> BenchmarkResult:
    """Aquece um detector uma vez e mede inferencias repetidas em condicoes fixas."""
    if repetitions < 1:
        raise ValueError("repetitions must be at least one.")

    detector.predict(image)
    gpu_enabled = torch.cuda.is_available()
    if gpu_enabled:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    timings_ms: list[float] = []
    detections: list[Detection] = []
    for _ in range(repetitions):
        if gpu_enabled:
            torch.cuda.synchronize()
        started_at = perf_counter()
        detections = detector.predict(image)
        if gpu_enabled:
            torch.cuda.synchronize()
        timings_ms.append((perf_counter() - started_at) * 1_000)

    metrics = calculate_detection_metrics(detections, ground_truth, iou_threshold)
    peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if gpu_enabled else None
    return BenchmarkResult(
        name=name,
        predicted_count=len(detections),
        metrics=metrics,
        median_inference_ms=median(timings_ms),
        peak_gpu_memory_mb=peak_gpu_memory_mb,
        detections=tuple(detections),
    )
