"""Safe decoding and validation for image uploads."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
# Ultralytics patches PIL.Image.open globally to lazily install HEIF support. Keeping this
# reference before loading a detector prevents invalid user uploads from triggering that path.
PILLOW_IMAGE_OPEN = Image.open


class ImageValidationError(ValueError):
    """Raised when uploaded bytes cannot be accepted as an analysis image."""


def decode_rgb_image(content: bytes, max_upload_bytes: int, max_pixels: int) -> np.ndarray:
    """Validate uploaded bytes and return an isolated RGB pixel array.

    Image format is inspected from its content rather than trusting the HTTP content type.
    """
    if not content:
        raise ImageValidationError("Envie uma imagem nao vazia.")
    if len(content) > max_upload_bytes:
        raise ImageValidationError("A imagem excede o limite de tamanho permitido.")
    try:
        with PILLOW_IMAGE_OPEN(BytesIO(content)) as opened_image:
            if opened_image.format not in ALLOWED_IMAGE_FORMATS:
                raise ImageValidationError("Envie uma imagem JPEG, PNG ou WEBP.")
            width, height = opened_image.size
            if width * height > max_pixels:
                raise ImageValidationError("A imagem excede o limite de pixels permitido.")
            return np.asarray(opened_image.convert("RGB")).copy()
    except UnidentifiedImageError as error:
        raise ImageValidationError("O arquivo enviado nao e uma imagem valida.") from error
