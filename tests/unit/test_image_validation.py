from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from src.api.image_validation import ImageValidationError, decode_rgb_image


def make_image_bytes(image_format: str = "JPEG") -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.full((8, 12, 3), 100, dtype=np.uint8)).save(buffer, format=image_format)
    return buffer.getvalue()


def test_decode_rgb_image_uses_file_content_and_returns_rgb() -> None:
    result = decode_rgb_image(make_image_bytes(), max_upload_bytes=10_000, max_pixels=1_000)

    assert result.shape == (8, 12, 3)
    assert result.dtype == np.uint8


@pytest.mark.parametrize(
    ("content", "max_upload_bytes", "max_pixels"),
    [
        (b"", 10_000, 1_000),
        (b"not an image", 10_000, 1_000),
        (make_image_bytes("GIF"), 10_000, 1_000),
        (make_image_bytes(), 10, 1_000),
        (make_image_bytes(), 10_000, 10),
    ],
)
def test_decode_rgb_image_rejects_invalid_uploads(
    content: bytes, max_upload_bytes: int, max_pixels: int
) -> None:
    with pytest.raises(ImageValidationError):
        decode_rgb_image(content, max_upload_bytes=max_upload_bytes, max_pixels=max_pixels)
