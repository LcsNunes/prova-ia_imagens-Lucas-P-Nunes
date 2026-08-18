"""Safe decoding and validation for image uploads."""

from __future__ import annotations

import cv2
import numpy as np

ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


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
    if _image_format(content) not in ALLOWED_IMAGE_FORMATS:
        raise ImageValidationError("Envie uma imagem JPEG, PNG ou WEBP.")

    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ImageValidationError("O arquivo enviado nao e uma imagem valida.")
    height, width = decoded.shape[:2]
    if width * height > max_pixels:
        raise ImageValidationError("A imagem excede o limite de pixels permitido.")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _image_format(content: bytes) -> str | None:
    """Identify the allowed formats by their signatures before decoding the bytes."""
    if content.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "WEBP"
    return None
