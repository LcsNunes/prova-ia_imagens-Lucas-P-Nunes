"""Logging da aplicacao com formato de console compacto e util."""

from __future__ import annotations

import logging
import os


def configure_logging(log_level: str | None = None) -> None:
    """Configura o logging do processo respeitando LOG_LEVEL quando informado."""
    level_name = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=level_name,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
