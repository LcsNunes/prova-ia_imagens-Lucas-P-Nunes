"""SAHI adapter for tiled vehicle inference with Ultralytics models."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from src.detection.base import Detection
from src.detection.class_mapping import normalize_vehicle_class
from src.detection.postprocessing import suppress_vehicle_duplicates
from src.detection.yolo_detector import resolve_device


class SahiVehicleDetector:
    """Run sliced inference and return only project-normalized vehicle detections."""

    def __init__(
        self,
        model_path: Path,
        model_domain: str,
        confidence: float,
        image_size: int,
        slice_size: int,
        overlap: float,
        device: str = "auto",
        merge_iou: float = 0.5,
    ) -> None:
        if not 0.0 <= overlap < 1.0:
            raise ValueError("overlap must be in the interval [0, 1).")
        if not 0.0 < merge_iou <= 1.0:
            raise ValueError("merge_iou must be in the interval (0, 1].")

        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[2]))
        self.model_domain = model_domain
        self.slice_size = slice_size
        self.overlap = overlap
        self.merge_iou = merge_iou
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(model_path),
            confidence_threshold=confidence,
            device=resolve_device(device),
            image_size=image_size,
        )

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Slice one RGB image, merge duplicates, and normalize vehicle classes."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape (height, width, 3).")

        result = get_sliced_prediction(
            image,
            self.detection_model,
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=self.overlap,
            overlap_width_ratio=self.overlap,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=self.merge_iou,
            postprocess_class_agnostic=True,
            verbose=0,
        )
        detections: list[Detection] = []
        for prediction in result.object_prediction_list:
            source_class = prediction.category.name
            if normalize_vehicle_class(self.model_domain, source_class):
                detections.append(
                    Detection(
                        bbox_xyxy=tuple(float(value) for value in prediction.bbox.to_xyxy()),
                        score=float(prediction.score.value),
                        source_class=source_class,
                    )
                )
        return suppress_vehicle_duplicates(detections)
