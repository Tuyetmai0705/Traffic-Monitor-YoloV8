"""Application logging to ``outputs/logs/app.log`` and the console."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT_LOGGER_NAME = "traffic_monitor"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3


def setup_logger(log_path: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Configure the root application logger once. Safe to call again with a new path."""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT)
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger such as ``traffic_monitor.detector``."""
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
