"""HSV and morphology based road highlighting for the presentation image."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RoadHighlight:
    """Probable-road mask and its semitransparent RGB overlay."""

    mask: np.ndarray
    overlay: np.ndarray


def highlight_roads(
    image: np.ndarray,
    hsv_lower: Sequence[int],
    hsv_upper: Sequence[int],
    kernel_size: int,
    min_area: int,
) -> RoadHighlight:
    """Highlight low-saturation regions that plausibly correspond to roads.

    This intentionally simple visual aid is not a semantic segmentation model. The
    threshold is configurable because lighting, pavement material, shadows, and cameras
    alter the apparent road colour.
    """
    _validate_inputs(image, hsv_lower, hsv_upper, kernel_size, min_area)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_image, np.array(hsv_lower, np.uint8), np.array(hsv_upper, np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    filtered_mask = _keep_large_components(cleaned_mask, min_area)
    return RoadHighlight(mask=filtered_mask, overlay=_build_overlay(image, filtered_mask))


def _validate_inputs(
    image: np.ndarray,
    hsv_lower: Sequence[int],
    hsv_upper: Sequence[int],
    kernel_size: int,
    min_area: int,
) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3).")
    if len(hsv_lower) != 3 or len(hsv_upper) != 3:
        raise ValueError("HSV limits must each contain three values.")
    if any(not 0 <= value <= 255 for value in (*hsv_lower, *hsv_upper)):
        raise ValueError("HSV limits must be integers between 0 and 255.")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    if min_area < 1:
        raise ValueError("min_area must be positive.")


def _keep_large_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    labels_count, labels, statistics, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for label in range(1, labels_count):
        if statistics[label, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == label] = 255
    return filtered


def _build_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    road_colour = np.array([242, 107, 56], dtype=np.uint8)
    selected = mask > 0
    overlay[selected] = (image[selected] * 0.70 + road_colour * 0.30).astype(np.uint8)
    return overlay
