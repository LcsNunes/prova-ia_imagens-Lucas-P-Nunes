"""Visualizacoes RGB consistentes usadas pelos experimentos e pela aplicacao web."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.detection.base import Detection
from src.evaluation.matching import GroundTruthBox, greedy_match

DETECTION_COLOR = (0, 229, 255)
GROUND_TRUTH_COLOR = (255, 193, 7)
TRUE_POSITIVE_COLOR = (0, 200, 83)
FALSE_POSITIVE_COLOR = (255, 193, 7)
FALSE_NEGATIVE_COLOR = (244, 67, 54)


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Retorna imagem RGB com caixas, scores e poligonos OBB quando disponiveis."""
    canvas = image.copy()
    for detection in detections:
        _draw_box(canvas, detection.bbox_xyxy, DETECTION_COLOR)
        if detection.obb_points is not None:
            points = np.array(detection.obb_points, dtype=np.int32)
            cv2.polylines(canvas, [points], isClosed=True, color=DETECTION_COLOR, thickness=2)
        x1, y1, _, _ = (int(value) for value in detection.bbox_xyxy)
        cv2.putText(
            canvas,
            f"vehicle {detection.score:.2f}",
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            DETECTION_COLOR,
            1,
            cv2.LINE_AA,
        )
    return canvas


def draw_ground_truth(image: np.ndarray, ground_truth: list[GroundTruthBox]) -> np.ndarray:
    """Retorna imagem RGB com caixas de veiculos revisadas manualmente."""
    canvas = image.copy()
    for ground_truth_box in ground_truth:
        _draw_box(canvas, ground_truth_box.bbox_xyxy, GROUND_TRUTH_COLOR)
    return canvas


def draw_detection_diagnostics(
    image: np.ndarray,
    detections: list[Detection],
    ground_truth: list[GroundTruthBox],
    *,
    iou_threshold: float,
) -> np.ndarray:
    """Retorna overlay de erros com caixas TP verdes, FP amarelas e FN vermelhas.

    Aplica o mesmo pareamento IoU um-para-um usado nas metricas; assim, a auditoria visual
    e as contagens TP/FP/FN reportadas descrevem exatamente as mesmas deteccoes.
    """
    canvas = image.copy()
    matches = greedy_match(detections, ground_truth, iou_threshold)
    matched_detection_indices = {match.prediction_index for match in matches}
    matched_ground_truth_indices = {match.ground_truth_index for match in matches}

    for detection_index, detection in enumerate(detections):
        is_true_positive = detection_index in matched_detection_indices
        color = TRUE_POSITIVE_COLOR if is_true_positive else FALSE_POSITIVE_COLOR
        label = "TP" if is_true_positive else "FP"
        _draw_detection(canvas, detection, color, f"{label} {detection.score:.2f}")

    for ground_truth_index, ground_truth_box in enumerate(ground_truth):
        if ground_truth_index not in matched_ground_truth_indices:
            _draw_box(canvas, ground_truth_box.bbox_xyxy, FALSE_NEGATIVE_COLOR)
            x1, y1, _, _ = (int(value) for value in ground_truth_box.bbox_xyxy)
            cv2.putText(
                canvas,
                "FN",
                (x1, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                FALSE_NEGATIVE_COLOR,
                1,
                cv2.LINE_AA,
            )
    return canvas


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    """Salva uma visualizacao RGB e cria o diretorio de destino quando necessario."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write image to {path}")


def _draw_box(
    image: np.ndarray, bbox_xyxy: tuple[float, float, float, float], color: tuple[int, int, int]
) -> None:
    x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)


def _draw_detection(
    image: np.ndarray, detection: Detection, color: tuple[int, int, int], label: str
) -> None:
    _draw_box(image, detection.bbox_xyxy, color)
    if detection.obb_points is not None:
        points = np.array(detection.obb_points, dtype=np.int32)
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
    x1, y1, _, _ = (int(value) for value in detection.bbox_xyxy)
    cv2.putText(
        image,
        label,
        (x1, max(14, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )
