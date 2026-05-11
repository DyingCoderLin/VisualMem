#!/usr/bin/env python3
"""Shared JSONL evaluation stats logger utilities."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

_LOGGER_LOCK = threading.Lock()
_LOGGERS: Dict[str, logging.Logger] = {}


def get_eval_logger(name: str, path: str) -> logging.Logger:
    """Return a file-only logger for evaluation JSONL events."""
    key = f"{name}:{path}"
    with _LOGGER_LOCK:
        if key in _LOGGERS:
            return _LOGGERS[key]

        logger = logging.getLogger(f"eval.{name}.{Path(path).name}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if not logger.handlers:
            log_path = Path(path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)

        _LOGGERS[key] = logger
        return logger


def emit_jsonl(logger: logging.Logger, event: dict) -> None:
    """Emit one JSON object as one line with UTC timestamp."""
    payload = dict(event)
    payload.setdefault(
        "ts",
        datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
    )
    logger.info(json.dumps(payload, ensure_ascii=False, default=str))
