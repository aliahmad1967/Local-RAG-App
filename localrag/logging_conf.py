"""UTF-8 logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import Config
from .paths import ensure_utf8_console

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_LOGGERS: dict[str, logging.Logger] = {}


def setup_logging(cfg: Config) -> logging.Logger:
    """Configure the `localrag` logger once and return it."""
    ensure_utf8_console()
    logger = logging.getLogger("localrag")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    fmt = logging.Formatter(_FORMAT)

    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    log_dir: Path = cfg.path("paths", "logs_dir", default="data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "localrag.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    _LOGGERS["localrag"] = logger
    return logger


def get_logger(name: str = "localrag") -> logging.Logger:
    return logging.getLogger(name if name.startswith("localrag") else f"localrag.{name}")
