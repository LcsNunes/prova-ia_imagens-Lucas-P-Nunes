"""Persistencia estruturada de ground truth para a avaliacao somente de veiculos."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from src.detection.base import BoundingBox
from src.evaluation.matching import GroundTruthBox


def load_ground_truth(path: Path) -> list[GroundTruthBox]:
    """Carrega anotacoes de veiculos no formato JSON de ground truth do projeto."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations = payload.get("annotations") if isinstance(payload, dict) else None
    if not isinstance(annotations, list):
        raise ValueError("Ground truth must contain an annotations list.")

    ground_truth: list[GroundTruthBox] = []
    for annotation in annotations:
        if not isinstance(annotation, dict) or annotation.get("category") != "vehicle":
            raise ValueError("Every ground-truth annotation must be a vehicle mapping.")
        bbox = annotation.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(
                "Every ground-truth annotation must define bbox_xyxy with four values."
            )
        normalized_bbox = tuple(float(value) for value in bbox)
        if normalized_bbox[2] <= normalized_bbox[0] or normalized_bbox[3] <= normalized_bbox[1]:
            raise ValueError("Ground-truth boxes must have positive width and height.")
        ground_truth.append(GroundTruthBox(str(annotation["id"]), normalized_bbox))
    return ground_truth


def save_ground_truth(
    path: Path,
    image_path: Path,
    boxes: list[BoundingBox],
    annotation_method: str,
    *,
    relative_to: Path | None = None,
) -> None:
    """Salva caixas revisadas, dimensoes e hash da imagem para reprodutibilidade."""
    image_path = image_path.resolve()
    serialized_image_path = image_path
    if relative_to is not None:
        serialized_image_path = image_path.relative_to(relative_to.resolve())

    with Image.open(image_path) as image:
        width, height = image.size
    payload = {
        "schema_version": 1,
        "annotation_method": annotation_method,
        "image": {
            "path": str(serialized_image_path.as_posix()),
            "width": width,
            "height": height,
            "sha256": _sha256(image_path),
        },
        "annotations": [
            {
                "id": f"vehicle-{index:03d}",
                "category": "vehicle",
                "bbox_xyxy": [round(value, 2) for value in bbox],
            }
            for index, bbox in enumerate(boxes, start=1)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
