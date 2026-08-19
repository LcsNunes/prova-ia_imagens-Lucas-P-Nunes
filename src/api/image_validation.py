"""Decodificacao e validacao segura de imagens enviadas."""

from __future__ import annotations

import cv2
import numpy as np

ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageValidationError(ValueError):
    """Lancada quando os bytes enviados nao podem ser aceitos como imagem de analise."""


def decode_rgb_image(content: bytes, max_upload_bytes: int, max_pixels: int) -> np.ndarray:
    """Valida os bytes enviados e retorna uma matriz RGB independente.

    O formato e verificado pelo conteudo, em vez de confiar no tipo HTTP informado.
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
    """Identifica formatos permitidos por suas assinaturas antes de decodificar os bytes."""
    if content.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "WEBP"
    return None
