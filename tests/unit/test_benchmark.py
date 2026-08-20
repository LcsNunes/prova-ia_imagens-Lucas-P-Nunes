import numpy as np
import pytest

from src.detection.base import Detection
from src.evaluation.benchmark import benchmark_detector
from src.evaluation.matching import GroundTruthBox


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image: np.ndarray) -> list[Detection]:
        self.calls += 1
        return [Detection((0, 0, 10, 10), 0.9, "car")]


def test_benchmark_detector_warms_and_reports_flat_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.evaluation.benchmark.torch.cuda.is_available", lambda: False)
    detector = FakeDetector()

    result = benchmark_detector(
        "fake",
        detector,
        np.zeros((16, 16, 3), dtype=np.uint8),
        [GroundTruthBox("vehicle-001", (0, 0, 10, 10))],
        repetitions=3,
    )

    assert detector.calls == 4
    assert result.predicted_count == 1
    assert result.metrics.f1 == 1.0
    assert result.peak_gpu_memory_mb is None
    assert result.as_dict()["name"] == "fake"


def test_benchmark_detector_rejects_zero_repetitions() -> None:
    with pytest.raises(ValueError, match="at least one"):
        benchmark_detector(
            "fake", FakeDetector(), np.zeros((1, 1, 3), dtype=np.uint8), [], repetitions=0
        )
