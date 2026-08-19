"""Contratos compartilhados por detectores de veiculos intercambiaveis."""

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
    """Uma predicao de veiculo normalizada para a representacao comum do projeto."""

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
    """Contrato implementado por detectores normais e com inferencia em fatias."""

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Retorna predicoes de veiculos para uma imagem RGB."""
