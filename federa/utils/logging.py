"""Structured (JSON) logging setup shared by the coordinator and clients.

Plain-text logs are hard to grep across thousands of concurrent client
connections. Every log record is emitted as a single JSON line so it can be
shipped straight into a log aggregator (Loki, CloudWatch, ELK, ...).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_RESERVED_KEYS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        extra = {k: v for k, v in record.__dict__.items() if k not in _RESERVED_KEYS}
        payload.update(extra)
        return json.dumps(payload, default=str)


def configure_logging(level: str | int = "INFO", *, json_output: bool = True) -> None:
    """Configure the root logger once. Safe to call multiple times."""
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output else logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
