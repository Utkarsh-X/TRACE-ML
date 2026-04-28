"""Application logging setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

from trace_aml.core.config import Settings


def configure_logger(settings: Settings) -> None:
    log_path = Path(os.getenv("TRACE_LOG_FILE", settings.logging.file_path))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.logging.level.upper(),
        format="<green>{time:DD-MM-YYYY HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )
    try:
        logger.add(
            str(log_path),
            level=settings.logging.level.upper(),
            rotation=settings.logging.rotation,
            retention=settings.logging.retention,
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )
    except PermissionError:
        logger.warning(
            "Falling back to synchronous file logging because queued logging is unavailable."
        )
        logger.add(
            str(log_path),
            level=settings.logging.level.upper(),
            rotation=settings.logging.rotation,
            retention=settings.logging.retention,
            enqueue=False,
            backtrace=False,
            diagnose=False,
        )
