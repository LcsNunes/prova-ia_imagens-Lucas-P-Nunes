import hashlib
from pathlib import Path

import pytest

from scripts.download_models import download_model


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
