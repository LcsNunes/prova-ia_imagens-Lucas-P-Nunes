"""Download the declared model weights without versioning binary artifacts in Git."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

import yaml
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "configs" / "model_registry.yaml"
DEFAULT_MODELS_DIR = ROOT / "models"
DEFAULT_CACHE_DIR = ROOT / "cache" / "huggingface"


def read_registry(path: Path) -> dict[str, dict[str, str]]:
    """Load and validate the minimal model metadata needed for reproducible downloads."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, dict):
        raise ValueError("Model registry must contain a 'models' mapping.")
    return models


def read_road_registry(path: Path) -> dict[str, dict[str, str]]:
    """Load optional Hugging Face road-segmentation artifacts from the same manifest."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    road_models = payload.get("road_models", {}) if isinstance(payload, dict) else {}
    if not isinstance(road_models, dict):
        raise ValueError("Model registry key 'road_models' must be a mapping when present.")
    return road_models


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for a downloaded weight file."""
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(metadata: dict[str, str], destination: Path) -> str:
    """Download one model atomically, validating its declared SHA-256 when available."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        digest = sha256(destination)
        _validate_digest(metadata, digest, destination)
        return digest

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with urlopen(metadata["url"]) as response:
                shutil.copyfileobj(response, temporary_file)
        digest = sha256(temporary_path)
        _validate_digest(metadata, digest, temporary_path)
        temporary_path.replace(destination)
        return digest
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def download_road_model(metadata: dict[str, str], cache_dir: Path) -> Path:
    """Download one pinned Hugging Face model snapshot into the project cache."""
    return Path(
        snapshot_download(
            repo_id=metadata["repository"],
            revision=metadata["revision"],
            cache_dir=cache_dir,
        )
    )


def _validate_digest(metadata: dict[str, str], actual: str, path: Path) -> None:
    expected = metadata.get("sha256")
    if expected and actual.lower() != expected.lower():
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, received {actual}.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", action="append", help="Registry key to download; repeat as needed."
    )
    parser.add_argument(
        "--road-model",
        action="append",
        help="Road-segmentation registry key to download; repeat as needed.",
    )
    parser.add_argument("--all", action="store_true", help="Download every model in the registry.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    registry = read_registry(arguments.registry)
    road_registry = read_road_registry(arguments.registry)
    selected_names = list(registry) if arguments.all else arguments.model or []
    selected_road_names = list(road_registry) if arguments.all else arguments.road_model or []
    if not selected_names and not selected_road_names:
        raise SystemExit("Specify --model <registry-key>, --road-model <registry-key>, or --all.")

    for name in selected_names:
        if name not in registry:
            raise SystemExit(f"Unknown model '{name}'. Available: {', '.join(registry)}")
        metadata = registry[name]
        destination = arguments.models_dir / metadata["filename"]
        digest = download_model(metadata, destination)
        print(f"{name}: {destination} | sha256={digest} | release={metadata['release']}")

    for name in selected_road_names:
        if name not in road_registry:
            raise SystemExit(
                f"Unknown road model '{name}'. Available: {', '.join(road_registry) or 'none'}"
            )
        metadata = road_registry[name]
        snapshot_path = download_road_model(metadata, arguments.cache_dir)
        print(f"{name}: {snapshot_path} | revision={metadata['revision']}")


if __name__ == "__main__":
    main()
