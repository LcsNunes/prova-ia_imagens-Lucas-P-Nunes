"""Shared contracts for interchangeable vehicle detectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np

BoundingBox: TypeAlias = tuple[float, float, float, float]
OrientedBox: TypeAlias = tuple[
    tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
]


@dataclass(frozen=True)
class Detection:
    """A vehicle prediction normalized to the project-wide representation."""

    bbox_xyxy: BoundingBox
    score: float
    source_class: str
    canonical_label: str = "vehicle"
    obb_points: OrientedBox | None = None

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Detection bbox_xyxy must have positive width and height.")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Detection score must be between 0 and 1.")
        if self.canonical_label != "vehicle":
            raise ValueError("Only the canonical 'vehicle' label is supported.")


class Detector(Protocol):
    """Contract implemented by normal and tiled vehicle detectors."""

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Return vehicle predictions for an RGB image."""
