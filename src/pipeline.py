"""Pipeline final de analise sincrona de imagens escolhido nos experimentos."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from src.detection.base import Detection, Detector
from src.detection.sahi_detector import SahiVehicleDetector
from src.detection.yolo_detector import YoloVehicleDetector
from src.roads.neural_road import Mask2FormerRoadHighlighter
from src.roads.opencv_road import RoadHighlight, highlight_roads
from src.visualization.draw import draw_detections

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisResult:
    """Saidas de veiculos e vias retornadas apos uma analise sincrona de imagem."""

    detections: tuple[Detection, ...]
    annotated_image: np.ndarray
    combined_image: np.ndarray
    roads: RoadHighlight
    inference_ms: float
    vehicle_inference_ms: float
    road_inference_ms: float
    model_name: str
    confidence: float
    sahi_enabled: bool
    road_method: str

    @property
    def vehicle_count(self) -> int:
        """Retorna a quantidade normalizada de deteccoes de veiculos."""
        return len(self.detections)


class VehicleAnalysisPipeline:
    """Coordena o detector escolhido, o visualizador de vias e os overlays de apresentacao."""

    def __init__(
        self,
        detector: Detector,
        model_name: str,
        confidence: float,
        sahi_enabled: bool,
        road_highlighter: Callable[[np.ndarray], RoadHighlight],
        road_method: str,
    ) -> None:
        self.detector = detector
        self.model_name = model_name
        self.confidence = confidence
        self.sahi_enabled = sahi_enabled
        self.road_highlighter = road_highlighter
        self.road_method = road_method

    @classmethod
    def from_config(
        cls, configuration: Mapping[str, Any], project_root: Path
    ) -> VehicleAnalysisPipeline:
        """Cria o detector final configurado sem baixar pesos de forma implicita."""
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
            "device": os.getenv("DEVICE", str(runtime.get("device", "auto"))),
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
        road_highlighter, road_method = _build_road_highlighter(
            road_config,
            project_root,
            os.getenv("DEVICE", str(runtime.get("device", "auto"))),
        )
        return cls(
            detector=detector,
            model_name=model_name,
            confidence=float(model["confidence"]),
            sahi_enabled=sahi_enabled,
            road_highlighter=road_highlighter,
            road_method=road_method,
        )

    def analyze(self, image: np.ndarray) -> AnalysisResult:
        """Detecta veiculos, segmenta vias e compoe os overlays de uma imagem RGB."""
        LOGGER.info(
            "Starting inference with model=%s sahi_enabled=%s", self.model_name, self.sahi_enabled
        )
        total_started_at = perf_counter()
        started_at = perf_counter()
        detections = self.detector.predict(image)
        vehicle_inference_ms = (perf_counter() - started_at) * 1_000
        started_at = perf_counter()
        roads = self.road_highlighter(image)
        road_inference_ms = (perf_counter() - started_at) * 1_000
        inference_ms = (perf_counter() - total_started_at) * 1_000
        result = AnalysisResult(
            detections=tuple(detections),
            annotated_image=draw_detections(image, detections),
            combined_image=draw_detections(roads.overlay, detections),
            roads=roads,
            inference_ms=inference_ms,
            vehicle_inference_ms=vehicle_inference_ms,
            road_inference_ms=road_inference_ms,
            model_name=self.model_name,
            confidence=self.confidence,
            sahi_enabled=self.sahi_enabled,
            road_method=self.road_method,
        )
        LOGGER.info(
            "Inference finished with vehicles=%s vehicle_ms=%.1f road_ms=%.1f total_ms=%.1f",
            result.vehicle_count,
            result.vehicle_inference_ms,
            result.road_inference_ms,
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


def _build_road_highlighter(
    road_config: Mapping[str, Any], project_root: Path, device: str
) -> tuple[Callable[[np.ndarray], RoadHighlight], str]:
    method = str(road_config.get("method", "hsv"))
    if method == "mask2former":
        cache_dir = project_root / _required_string(road_config, "cache_dir")
        highlighter = Mask2FormerRoadHighlighter(
            model_id=_required_string(road_config, "model_id"),
            revision=_required_string(road_config, "revision"),
            road_class_id=int(road_config["road_class_id"]),
            cache_dir=cache_dir,
            device=device,
            overlay_color=tuple(int(value) for value in road_config["overlay_color"]),
            opacity=float(road_config["opacity"]),
        )
        return highlighter.highlight, method
    if method == "hsv":
        return (
            lambda image: highlight_roads(
                image=image,
                hsv_lower=road_config["hsv_lower"],
                hsv_upper=road_config["hsv_upper"],
                kernel_size=int(road_config["kernel_size"]),
                min_area=int(road_config["min_area"]),
            ),
            method,
        )
    raise ValueError(f"Unsupported roads.method: {method}")
