"""Download the declared model weights without versioning binary artifacts in Git."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "configs" / "model_registry.yaml"
DEFAULT_MODELS_DIR = ROOT / "models"


def read_registry(path: Path) -> dict[str, dict[str, str]]:
    """Load and validate the minimal model metadata needed for reproducible downloads."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, dict):
        raise ValueError("Model registry must contain a 'models' mapping.")
    return models


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for a downloaded weight file."""
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(metadata: dict[str, str], destination: Path) -> str:
    """Download one model atomically and return its digest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return sha256(destination)

    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
        with urlopen(metadata["url"]) as response:
            shutil.copyfileobj(response, temporary_file)
    temporary_path.replace(destination)
    return sha256(destination)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", action="append", help="Registry key to download; repeat as needed."
    )
    parser.add_argument("--all", action="store_true", help="Download every model in the registry.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    registry = read_registry(arguments.registry)
    selected_names = list(registry) if arguments.all else arguments.model or []
    if not selected_names:
        raise SystemExit("Specify --model <registry-key> or --all.")

    for name in selected_names:
        if name not in registry:
            raise SystemExit(f"Unknown model '{name}'. Available: {', '.join(registry)}")
        metadata = registry[name]
        destination = arguments.models_dir / metadata["filename"]
        digest = download_model(metadata, destination)
        print(f"{name}: {destination} | sha256={digest} | release={metadata['release']}")


if __name__ == "__main__":
    main()
