"""Logging configuration for the HyperSat pipeline.

Two output formats are supported:

* ``text`` - a compact human-readable line, intended for interactive CLI use;
* ``json`` - one JSON object per line, intended for container/CI log collectors.

Structured detail is attached with the standard ``extra=`` mechanism, e.g.::

    logger.info("stage completed", extra={"stage": "orthorectify", "duration_s": 12.4})

Both formatters render those keys, so a stage never has to build its own message string
out of formatted values.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from enum import StrEnum
from typing import Any

__all__ = ["LogFormat", "configure_logging", "get_logger"]

LOGGER_NAMESPACE = "hypersat"

# Attributes present on every LogRecord; anything else was supplied via `extra=`.
_RESERVED_RECORD_KEYS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class LogFormat(StrEnum):
    """Supported log rendering formats."""

    TEXT = "text"
    JSON = "json"


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    """Return the caller-supplied ``extra`` fields of a log record."""
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _RESERVED_RECORD_KEYS and not key.startswith("_")
    }


class TextFormatter(logging.Formatter):
    """Render a log record as ``timestamp level logger: message key=value``."""

    def __init__(self) -> None:
        super().__init__(fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        self.default_msec_format = "%s.%03d"

    def format(self, record: logging.LogRecord) -> str:
        """Format ``record``, appending any structured extras."""
        base = super().format(record)
        extras = _extras(record)
        if not extras:
            return base
        rendered = " ".join(f"{key}={value!r}" for key, value in sorted(extras.items()))
        return f"{base} | {rendered}"


class JsonFormatter(logging.Formatter):
    """Render a log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        """Format ``record`` as JSON, with extras as top-level keys."""
        payload: dict[str, Any] = {
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_extras(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    level: int | str = logging.INFO,
    log_format: LogFormat = LogFormat.TEXT,
) -> logging.Logger:
    """Configure the ``hypersat`` logger hierarchy and return its root logger.

    The function is idempotent: repeated calls replace the existing handler instead of
    stacking duplicates. Only the ``hypersat`` namespace is touched, so importing this
    package never reconfigures logging for an embedding application.

    Args:
        level: Logging level as an integer or a name such as ``"DEBUG"``.
        log_format: Whether to emit human-readable text or newline-delimited JSON.

    Returns:
        The configured ``hypersat`` logger.
    """
    formatter: logging.Formatter = (
        JsonFormatter() if log_format is LogFormat.JSON else TextFormatter()
    )
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(formatter)

    logger = logging.getLogger(LOGGER_NAMESPACE)
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
        existing.close()
    logger.addHandler(handler)
    logger.setLevel(level)
    # Diagnostics belong on stderr only; the host application decides about the rest.
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger inside the ``hypersat`` namespace.

    Args:
        name: Usually ``__name__``. A fully-qualified ``hypersat.*`` name is used as is;
            anything else is nested under the ``hypersat`` namespace.

    Returns:
        A standard library logger.
    """
    if name == LOGGER_NAMESPACE or name.startswith(f"{LOGGER_NAMESPACE}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAMESPACE}.{name}")
