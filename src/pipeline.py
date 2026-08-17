"""Final synchronous image-analysis pipeline selected from the experiments."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from src.detection.base import Detection, Detector
from src.detection.sahi_detector import SahiVehicleDetector
from src.detection.yolo_detector import YoloVehicleDetector
from src.roads.opencv_road import RoadHighlight, highlight_roads
from src.visualization.draw import draw_detections

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisResult:
    """Vehicle and road outputs returned after one synchronous image analysis."""

    detections: tuple[Detection, ...]
    annotated_image: np.ndarray
    roads: RoadHighlight
    inference_ms: float
    model_name: str
    confidence: float
    sahi_enabled: bool

    @property
    def vehicle_count(self) -> int:
        """Return the normalized number of vehicle detections."""
        return len(self.detections)


class VehicleAnalysisPipeline:
    """Coordinate the chosen detector, road visualizer, and presentation overlays."""

    def __init__(
        self,
        detector: Detector,
        model_name: str,
        confidence: float,
        sahi_enabled: bool,
        road_config: Mapping[str, Any],
    ) -> None:
        self.detector = detector
        self.model_name = model_name
        self.confidence = confidence
        self.sahi_enabled = sahi_enabled
        self.road_config = road_config

    @classmethod
    def from_config(
        cls, configuration: Mapping[str, Any], project_root: Path
    ) -> VehicleAnalysisPipeline:
        """Create the configured final detector without downloading model weights implicitly."""
        model = _mapping(configuration, "model")
        runtime = _mapping(configuration, "runtime")
        sahi = _mapping(configuration, "sahi")
        road_config = _mapping(configuration, "roads")
        model_name = _required_string(model, "name")
        model_path = project_root / "models" / model_name
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model weight not found at {model_path}. Run scripts/download_models.py first."
            )

        detector_options = {
            "model_path": model_path,
            "model_domain": _required_string(model, "domain"),
            "confidence": float(model["confidence"]),
            "image_size": int(model["imgsz"]),
            "device": str(runtime.get("device", "auto")),
        }
        sahi_enabled = bool(sahi.get("enabled", False))
        detector: Detector
        if sahi_enabled:
            detector = SahiVehicleDetector(
                **detector_options,
                slice_size=int(sahi["slice_size"]),
                overlap=float(sahi["overlap"]),
                merge_iou=float(sahi.get("merge_iou", 0.5)),
            )
        else:
            detector = YoloVehicleDetector(
                **detector_options,
                nms_iou=float(model.get("nms_iou", 0.7)),
            )
        return cls(
            detector=detector,
            model_name=model_name,
            confidence=float(model["confidence"]),
            sahi_enabled=sahi_enabled,
            road_config=road_config,
        )

    def analyze(self, image: np.ndarray) -> AnalysisResult:
        """Detect vehicles, then create road and detection overlays for one RGB image."""
        LOGGER.info(
            "Starting inference with model=%s sahi_enabled=%s", self.model_name, self.sahi_enabled
        )
        started_at = perf_counter()
        detections = self.detector.predict(image)
        inference_ms = (perf_counter() - started_at) * 1_000
        roads = highlight_roads(
            image=image,
            hsv_lower=self.road_config["hsv_lower"],
            hsv_upper=self.road_config["hsv_upper"],
            kernel_size=int(self.road_config["kernel_size"]),
            min_area=int(self.road_config["min_area"]),
        )
        result = AnalysisResult(
            detections=tuple(detections),
            annotated_image=draw_detections(image, detections),
            roads=roads,
            inference_ms=inference_ms,
            model_name=self.model_name,
            confidence=self.confidence,
            sahi_enabled=self.sahi_enabled,
        )
        LOGGER.info(
            "Inference finished with vehicles=%s inference_ms=%.1f",
            result.vehicle_count,
            inference_ms,
        )
        return result


def _mapping(configuration: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = configuration.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration key '{key}' must be a mapping.")
    return value


def _required_string(configuration: Mapping[str, Any], key: str) -> str:
    value = configuration.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Configuration key '{key}' must be a non-empty string.")
    return value
