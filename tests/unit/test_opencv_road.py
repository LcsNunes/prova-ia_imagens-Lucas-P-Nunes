import numpy as np
import pytest

from src.roads.opencv_road import highlight_roads


def test_highlight_roads_keeps_large_low_saturation_region() -> None:
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    image[5:25, 5:25] = [120, 120, 120]
    image[0:4, 0:4] = [0, 180, 0]

    result = highlight_roads(
        image,
        hsv_lower=(0, 0, 50),
        hsv_upper=(179, 40, 200),
        kernel_size=3,
        min_area=100,
    )

    assert result.mask[15, 15] == 255
    assert result.mask[1, 1] == 0
    assert not np.array_equal(result.overlay[15, 15], image[15, 15])
    assert np.array_equal(result.overlay[1, 1], image[1, 1])


@pytest.mark.parametrize(
    ("image", "kernel_size", "min_area"),
    [
        (np.zeros((10, 10), dtype=np.uint8), 3, 1),
        (np.zeros((10, 10, 3), dtype=np.uint8), 2, 1),
        (np.zeros((10, 10, 3), dtype=np.uint8), 3, 0),
    ],
)
def test_highlight_roads_rejects_invalid_inputs(
    image: np.ndarray, kernel_size: int, min_area: int
) -> None:
    with pytest.raises(ValueError):
        highlight_roads(image, (0, 0, 0), (179, 255, 255), kernel_size, min_area)
