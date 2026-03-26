"""Application logging setup."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from trace_ml.core.config import Settings


def configure_logger(settings: Settings) -> None:
    log_path = Path(settings.logging.file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.logging.level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )
    logger.add(
        str(log_path),
        level=settings.logging.level.upper(),
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
