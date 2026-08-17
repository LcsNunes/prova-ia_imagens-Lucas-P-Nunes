"""Normalization of dataset-specific vehicle classes to the common ontology."""

from __future__ import annotations

COCO_VEHICLE_CLASSES = {
    "car": "small_vehicle",
    "motorcycle": "small_vehicle",
    "bus": "large_vehicle",
    "truck": "large_vehicle",
}
AERIAL_VEHICLE_CLASSES = {
    "small-vehicle": "small_vehicle",
    "large-vehicle": "large_vehicle",
    "small vehicle": "small_vehicle",
    "large vehicle": "large_vehicle",
}


def normalize_vehicle_class(model_domain: str, source_class: str) -> str | None:
    """Return ``vehicle`` when a source class belongs to the project vehicle ontology."""
    classes = _classes_for_domain(model_domain)
    return "vehicle" if source_class in classes else None


def vehicle_size_group(model_domain: str, source_class: str) -> str | None:
    """Return the optional coarse vehicle-size group used only for diagnostics."""
    return _classes_for_domain(model_domain).get(source_class)


def _classes_for_domain(model_domain: str) -> dict[str, str]:
    if model_domain == "coco":
        return COCO_VEHICLE_CLASSES
    if model_domain == "aerial":
        return AERIAL_VEHICLE_CLASSES
    raise ValueError(f"Unsupported model domain: {model_domain}")
