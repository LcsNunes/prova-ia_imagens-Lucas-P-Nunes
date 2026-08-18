"""Mask2Former road highlighting used by the synchronous demonstration pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.roads.opencv_road import RoadHighlight


class Mask2FormerRoadHighlighter:
    """Create a road overlay from a locally cached Mask2Former semantic segmenter.

    The model is intentionally loaded from the project cache only. Downloading a large
    artifact during an API request would make the application slow and non-reproducible.
    """

    def __init__(
        self,
        model_id: str,
        revision: str,
        road_class_id: int,
        cache_dir: Path,
        device: str,
        overlay_color: tuple[int, int, int],
        opacity: float,
    ) -> None:
        self.road_class_id = road_class_id
        self.overlay_color = np.asarray(overlay_color, dtype=np.uint8)
        self.opacity = opacity

        try:
            import torch
            from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
        except ImportError as error:
            raise RuntimeError(
                "Road segmentation dependencies are unavailable. Install requirements.txt first."
            ) from error

        self._torch = torch
        self.device = _resolve_device(device, torch)
        try:
            self.processor = Mask2FormerImageProcessor.from_pretrained(
                model_id,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_id,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=True,
            ).to(self.device)
        except OSError as error:
            raise FileNotFoundError(
                "Road segmentation model is unavailable in the local cache. Run "
                "scripts/download_models.py --road-model mask2former_satellite first."
            ) from error
        self.model.eval()

    def highlight(self, image: np.ndarray) -> RoadHighlight:
        """Return the road-class mask and a warm-coloured RGB overlay for one image."""
        _validate_rgb_image(image)
        inputs = self.processor(images=Image.fromarray(image), return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        segmentation = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.shape[:2]]
        )[0]
        mask = (segmentation.detach().cpu().numpy() == self.road_class_id).astype(np.uint8) * 255
        return RoadHighlight(
            mask=mask, overlay=_build_overlay(image, mask, self.overlay_color, self.opacity)
        )


def _resolve_device(requested_device: str, torch_module: Any) -> str:
    if requested_device == "auto":
        return "cuda:0" if torch_module.cuda.is_available() else "cpu"
    return requested_device


def _validate_rgb_image(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3).")


def _build_overlay(
    image: np.ndarray, mask: np.ndarray, color: np.ndarray, opacity: float
) -> np.ndarray:
    overlay = image.copy()
    selected = mask > 0
    overlay[selected] = (image[selected] * (1 - opacity) + color * opacity).astype(np.uint8)
    return overlay
