"""Ultralytics YOLO adapter for COCO and oriented aerial vehicle models."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

# Keep Ultralytics settings local to this reproducible project instead of a user profile.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[2]))

from src.detection.base import Detection, OrientedBox
from src.detection.class_mapping import normalize_vehicle_class

YOLO = import_module("ultralytics").YOLO


def resolve_device(requested_device: str) -> str:
    """Resolve the requested runtime device and reject unavailable explicit CUDA requests."""
    if requested_device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch.")
    if requested_device not in {"cpu", "cuda:0"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda:0'.")
    return requested_device


class YoloVehicleDetector:
    """Run one Ultralytics model and expose only normalized vehicle detections."""

    def __init__(
        self,
        model_path: Path,
        model_domain: str,
        confidence: float,
        image_size: int,
        nms_iou: float,
        device: str = "auto",
    ) -> None:
        self.model_domain = model_domain
        self.confidence = confidence
        self.image_size = image_size
        self.nms_iou = nms_iou
        self.device = resolve_device(device)
        self.model = YOLO(str(model_path))

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Run inference on an RGB image and return its vehicle predictions."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape (height, width, 3).")

        bgr_image = image[..., ::-1]
        result = self.model.predict(
            source=bgr_image,
            conf=self.confidence,
            imgsz=self.image_size,
            iou=self.nms_iou,
            device=self.device,
            verbose=False,
        )[0]
        return (
            self._extract_oriented(result)
            if result.obb is not None
            else self._extract_axis_aligned(result)
        )

    def _extract_axis_aligned(self, result: object) -> list[Detection]:
        boxes = result.boxes
        if boxes is None:
            return []
        detections: list[Detection] = []
        for bbox, confidence, class_index in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.conf.cpu().tolist(),
            boxes.cls.cpu().tolist(),
            strict=True,
        ):
            source_class = result.names[int(class_index)]
            if normalize_vehicle_class(self.model_domain, source_class):
                detections.append(
                    Detection(
                        bbox_xyxy=tuple(float(value) for value in bbox),
                        score=float(confidence),
                        source_class=source_class,
                    )
                )
        return detections

    def _extract_oriented(self, result: object) -> list[Detection]:
        obb = result.obb
        if obb is None:
            return []
        detections: list[Detection] = []
        values = zip(
            obb.xyxy.cpu().tolist(),
            obb.conf.cpu().tolist(),
            obb.cls.cpu().tolist(),
            obb.xyxyxyxy.cpu().tolist(),
            strict=True,
        )
        for bbox, confidence, class_index, points in values:
            source_class = result.names[int(class_index)]
            if normalize_vehicle_class(self.model_domain, source_class):
                oriented_box: OrientedBox = tuple(
                    (float(point[0]), float(point[1])) for point in points
                )
                detections.append(
                    Detection(
                        bbox_xyxy=tuple(float(value) for value in bbox),
                        score=float(confidence),
                        source_class=source_class,
                        obb_points=oriented_box,
                    )
                )
        return detections
