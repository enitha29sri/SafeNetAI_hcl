"""
utils/logger.py

Centralized logging configuration for SafeNet AI, built on loguru.

Every module should call `get_logger(__name__)` instead of configuring its
own handlers, so log format, level, and destination stay consistent across
the whole application.

Security note: loggers in this project must NEVER receive raw passwords,
OTPs, API keys, or other credentials. Calling code is responsible for
redacting sensitive values before logging anything.
"""

from __future__ import annotations

import sys
from typing import Any

from loguru import logger as _loguru_logger

from config import settings

_configured = False


def _configure_once() -> None:
    """Configure loguru sinks exactly once per process."""
    global _configured
    if _configured:
        return

    # Remove the default stderr handler so we can control format/level ourselves.
    _loguru_logger.remove()

    _loguru_logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[name]}</cyan> - <level>{message}</level>"
        ),
        colorize=True,
    )

    _loguru_logger.add(
        str(settings.log_file_path),
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[name]} - {message}",
        rotation="5 MB",
        retention=5,
        encoding="utf-8",
    )

    _configured = True


def get_logger(name: str) -> Any:
    """
    Return a loguru logger bound to the given module name.

    Args:
        name: Typically `__name__` of the calling module.

    Returns:
        A loguru logger instance with `name` bound for filtering/formatting.
    """
    _configure_once()
    return _loguru_logger.bind(name=name)
