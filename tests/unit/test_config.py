from pathlib import Path

import pytest

from src.config import load_yaml


def test_load_yaml_returns_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  confidence: 0.25\n", encoding="utf-8")

    assert load_yaml(config_path) == {"model": {"confidence": 0.25}}


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- item\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_yaml(config_path)
