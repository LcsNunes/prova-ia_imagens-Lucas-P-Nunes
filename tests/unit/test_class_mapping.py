import pytest

from src.detection.class_mapping import normalize_vehicle_class, vehicle_size_group


@pytest.mark.parametrize(
    ("model_domain", "source_class", "expected_group"),
    [
        ("coco", "car", "small_vehicle"),
        ("coco", "truck", "large_vehicle"),
        ("aerial", "small-vehicle", "small_vehicle"),
        ("aerial", "large-vehicle", "large_vehicle"),
        ("aerial", "small vehicle", "small_vehicle"),
        ("aerial", "large vehicle", "large_vehicle"),
    ],
)
def test_vehicle_classes_share_common_ontology(
    model_domain: str, source_class: str, expected_group: str
) -> None:
    assert normalize_vehicle_class(model_domain, source_class) == "vehicle"
    assert vehicle_size_group(model_domain, source_class) == expected_group


def test_non_vehicle_class_is_not_normalized() -> None:
    assert normalize_vehicle_class("coco", "person") is None


def test_unknown_domain_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_vehicle_class("unknown", "car")
