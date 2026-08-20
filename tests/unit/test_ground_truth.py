import json
from pathlib import Path

import pytest
from PIL import Image

from src.evaluation.ground_truth import load_ground_truth, save_ground_truth


def test_save_and_load_ground_truth(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (20, 10)).save(image_path)
    ground_truth_path = tmp_path / "ground_truth.json"

    save_ground_truth(
        ground_truth_path,
        image_path,
        [(1.111, 2.222, 10.333, 8.444)],
        "manual_review",
    )

    saved = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    assert saved["image"]["width"] == 20
    assert saved["annotations"][0]["bbox_xyxy"] == [1.11, 2.22, 10.33, 8.44]
    assert load_ground_truth(ground_truth_path)[0].annotation_id == "vehicle-001"


def test_save_ground_truth_can_store_a_project_relative_image_path(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    image_path = project_root / "data" / "raw" / "image.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (20, 10)).save(image_path)
    ground_truth_path = project_root / "data" / "annotations" / "ground_truth.json"

    save_ground_truth(
        ground_truth_path,
        image_path,
        [(1, 2, 10, 8)],
        "manual_review",
        relative_to=project_root,
    )

    saved = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    assert saved["image"]["path"] == "data/raw/image.jpg"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"annotations": [{"id": "vehicle-001", "category": "car", "bbox_xyxy": [0, 0, 1, 1]}]},
        {"annotations": [{"id": "vehicle-001", "category": "vehicle", "bbox_xyxy": [0, 0, 0, 1]}]},
    ],
)
def test_load_ground_truth_rejects_invalid_annotations(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_ground_truth(path)
