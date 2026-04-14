#!/usr/bin/env python3
"""
Top-level daily report application (外挂 agent).

Runs as a separate process. Implicitly calls the VisualMem data platform
over HTTP (REPORT_DATA_API_BASE, default http://localhost:18080) — not
mounted on the backend as an endpoint.

Usage:
  conda activate mobiagent
  cd <project_root>
  python scripts/daily_report.py --date 2026-03-24
  python scripts/daily_report.py --date 2026-03-24 -v          # DEBUG logs
  python scripts/daily_report.py --date 2026-03-24 --print-json   # also dump full JSON to stdout

By default: writes logs/daily_report_<date>.json only. Stderr prints path + LLM token
totals + pipeline_ms (no report body). Use --print-json to stream full JSON to stdout.

Debug logs: set LOG_LEVEL=DEBUG in .env or use -v (must run before imports).

Requires .env with REPORT_API_KEY (and optional REPORT_API_BASE / models).
The data platform server must be listening on REPORT_DATA_API_BASE.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Before any project imports: DEBUG logging for report.* (dotenv will not override)
if "-v" in sys.argv or "--verbose" in sys.argv:
    os.environ["LOG_LEVEL"] = "DEBUG"


def _write_report_json(result: dict, date: str, *, print_json: bool) -> None:
    """Always write full JSON to REPORT_LOG_DIR (default logs/); optional stdout."""
    from config import config as _cfg

    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    out_dir = Path(_cfg.REPORT_LOG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"daily_report_{date}.json"
    out_path.write_text(text, encoding="utf-8")

    ps = result.get("pipeline_stats") or {}
    tin = ps.get("total_input_tokens")
    tout = ps.get("total_output_tokens")
    tpm = ps.get("total_pipeline_ms")
    print(
        f"[daily_report] wrote {out_path} | LLM tokens in={tin} out={tout} | "
        f"pipeline_ms={tpm}",
        file=sys.stderr,
    )

    if not print_json:
        return

    chunk_size = 16384
    try:
        for i in range(0, len(text), chunk_size):
            sys.stdout.write(text[i : i + chunk_size])
        sys.stdout.flush()
    except (BlockingIOError, BrokenPipeError, OSError) as e:
        print(
            f"[daily_report] stdout failed ({e!r}); JSON already saved at {out_path}",
            file=sys.stderr,
        )


async def _main():
    parser = argparse.ArgumentParser(description="Generate daily report via Map-Reduce pipeline")
    parser.add_argument("--date", required=True, help="ISO date, e.g. 2026-03-24")
    parser.add_argument("--goal", default="", help="Optional daily macro goal")
    parser.add_argument(
        "--language",
        default="auto",
        help="Report language: zh=简体中文, en=English, auto=中文为主则用中文（默认）",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG-level logs for report.* (pipeline, data_fetcher, llm_caller, chunker)",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Also print the full JSON report to stdout (default: file + stderr summary only)",
    )
    args = parser.parse_args()

    from config import config
    from apps.reporting.pipeline import ReportPipeline

    pipeline = ReportPipeline()
    try:
        goal = args.goal.strip() or (config.REPORT_DAILY_GOAL or "").strip()
        result = await pipeline.generate(
            date=args.date,
            daily_goal=goal,
            language=args.language,
        )
        _write_report_json(result, args.date, print_json=args.print_json)
    finally:
        try:
            from apps.reporting.report.llm_caller import drain_litellm_async_logging

            await drain_litellm_async_logging()
        except Exception:
            pass
        await pipeline.fetcher.close()


if __name__ == "__main__":
    asyncio.run(_main())
