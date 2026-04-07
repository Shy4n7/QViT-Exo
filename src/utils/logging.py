"""Thin logging factory for the exoplanet-detection package.

Returns a consistently formatted ``logging.Logger`` so that every module in
the project uses the same timestamp / name / level layout without repeating
handler-setup boilerplate.

Usage
-----
    from src.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Preprocessing complete.")
"""

from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a StreamHandler at INFO level.

    If *name* already has handlers attached (e.g. from a previous call in the
    same process), no duplicate handler is added.

    Parameters
    ----------
    name:
        Logger name — conventionally ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    return logger
