"""Application logging with one compact, useful console format."""

from __future__ import annotations

import logging
import os


def configure_logging(log_level: str | None = None) -> None:
    """Configure process logging, honouring LOG_LEVEL when it is supplied."""
    level_name = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=level_name,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
