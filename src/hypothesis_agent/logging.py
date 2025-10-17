"""Logging utilities for the hypothesis agent services."""
from __future__ import annotations

import logging
from logging.config import dictConfig
from typing import Any, Dict

_DEFAULT_LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        }
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}

_configured = False


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once and adjust log level on subsequent calls."""

    global _configured
    log_level = level.upper()

    if _configured:
        logging.getLogger().setLevel(log_level)
        return

    config = dict(_DEFAULT_LOGGING_CONFIG)
    config_root = dict(config["root"])
    config_root["level"] = log_level
    config["root"] = config_root

    dictConfig(config)
    _configured = True
