import hashlib
from pathlib import Path

import pytest

from scripts.download_models import download_model, download_road_model


def test_download_model_validates_an_existing_weight_digest(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"verified model")
    metadata = {"sha256": hashlib.sha256(b"verified model").hexdigest()}

    assert download_model(metadata, model_path) == metadata["sha256"]


def test_download_model_rejects_an_existing_weight_with_wrong_digest(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"unexpected model")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download_model({"sha256": "not-a-real-hash"}, model_path)


def test_download_road_model_uses_pinned_hugging_face_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: dict[str, object] = {}

    def fake_snapshot_download(**options: object) -> str:
        received.update(options)
        return str(tmp_path / "snapshot")

    monkeypatch.setattr("scripts.download_models.snapshot_download", fake_snapshot_download)

    downloaded = download_road_model(
        {"repository": "example/road-model", "revision": "fixed-revision"}, tmp_path / "cache"
    )

    assert downloaded == tmp_path / "snapshot"
    assert received == {
        "repo_id": "example/road-model",
        "revision": "fixed-revision",
        "cache_dir": tmp_path / "cache",
    }
