#!/usr/bin/env python3
"""Periodic backend stability sampler for long-run experiments."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Iterable, Optional

import psutil

from utils.eval_stats import emit_jsonl, get_eval_logger


def _safe_process_io_counters(process: psutil.Process):
    """Return process io counters when supported; otherwise None.

    Notes:
    - psutil.Process.io_counters() is unavailable on some platforms (e.g. macOS),
      which would otherwise crash the sampling thread.
    """
    io_fn = getattr(process, "io_counters", None)
    if io_fn is None:
        return None
    try:
        return io_fn()
    except (AttributeError, NotImplementedError, psutil.Error):
        return None


def _safe_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


def _sampler_loop(
    *,
    interval_s: float,
    log_path: str,
    db_paths: Iterable[str],
    dir_paths: Iterable[str],
    stop_event: threading.Event,
) -> None:
    logger = get_eval_logger("stability", log_path)
    process = psutil.Process()
    process.cpu_percent(interval=0.1)
    last_io = _safe_process_io_counters(process)
    last_ts = time.time()

    db_path_objs = [Path(p) for p in db_paths]
    dir_path_objs = [Path(p) for p in dir_paths]

    while not stop_event.is_set():
        now = time.time()
        cpu_percent = process.cpu_percent(interval=None)
        rss_mb = process.memory_info().rss / (1024 * 1024)
        io_now = _safe_process_io_counters(process)
        elapsed = max(1e-6, now - last_ts)
        read_bps = 0.0
        write_bps = 0.0
        if last_io is not None and io_now is not None:
            read_bps = max(0.0, float(io_now.read_bytes - last_io.read_bytes)) / elapsed
            write_bps = max(0.0, float(io_now.write_bytes - last_io.write_bytes)) / elapsed
        last_io = io_now
        last_ts = now

        db_sizes = {p.name: _safe_size(p) for p in db_path_objs}
        dir_sizes = {p.name: _safe_size(p) for p in dir_path_objs}
        emit_jsonl(
            logger,
            {
                "type": "sample",
                "cpu_percent": cpu_percent,
                "rss_mb": rss_mb,
                "num_threads": process.num_threads(),
                "read_bps": read_bps,
                "write_bps": write_bps,
                "db_sizes": db_sizes,
                "dir_sizes": dir_sizes,
            },
        )
        stop_event.wait(interval_s)


def start_stability_monitor(
    *,
    interval_s: float,
    log_path: str,
    db_paths: Iterable[str],
    dir_paths: Iterable[str],
) -> threading.Event:
    """Start a daemon sampler thread and return its stop event."""
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_sampler_loop,
        kwargs={
            "interval_s": interval_s,
            "log_path": log_path,
            "db_paths": list(db_paths),
            "dir_paths": list(dir_paths),
            "stop_event": stop_event,
        },
        daemon=True,
        name="eval-stability-monitor",
    )
    thread.start()
    return stop_event
