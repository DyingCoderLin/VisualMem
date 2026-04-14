"""
Daily report HTTP API: list saved reports, read by date, trigger generation.

Generation matches ``python scripts/daily_report.py --date YYYY-MM-DD``:
:class:`apps.reporting.pipeline.ReportPipeline` + JSON written under ``config.REPORT_LOG_DIR``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import config
from utils.logger import setup_logger

logger = setup_logger("api.daily_report")

router = APIRouter(prefix="/api", tags=["daily-reports"])

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _log_dir() -> Path:
    return Path(config.REPORT_LOG_DIR)


def _path_for_date(date: str) -> Path:
    return _log_dir() / f"daily_report_{date}.json"


def _list_report_dates() -> List[str]:
    base = _log_dir()
    if not base.is_dir():
        return []
    dates: List[str] = []
    for p in base.glob("daily_report_*.json"):
        name = p.name
        # daily_report_YYYY-MM-DD.json
        if not name.startswith("daily_report_") or not name.endswith(".json"):
            continue
        stem = name[len("daily_report_") : -len(".json")]
        if _ISO_DATE.match(stem):
            dates.append(stem)
    return sorted(dates, reverse=True)


def _read_report_file(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("daily report read failed: %s", e)
        raise HTTPException(status_code=500, detail="无法读取日报内容") from e
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("daily report JSON invalid: %s", e)
        raise HTTPException(status_code=500, detail="日报内容格式无效") from e


def _write_report_json(result: Dict[str, Any], date: str) -> None:
    out_dir = _log_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"daily_report_{date}.json"
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    out_path.write_text(text, encoding="utf-8")
    ps = result.get("pipeline_stats") or {}
    logger.info(
        "daily report wrote %s | LLM tokens in=%s out=%s | pipeline_ms=%s",
        out_path,
        ps.get("total_input_tokens"),
        ps.get("total_output_tokens"),
        ps.get("total_pipeline_ms"),
    )


def _validate_date(date: str) -> str:
    if not _ISO_DATE.match(date):
        raise HTTPException(
            status_code=400,
            detail="日期格式应为 YYYY-MM-DD",
        )
    return date


class GenerateDailyReportBody(BaseModel):
    """Body for POST /api/daily-reports/generate."""

    # If omitted, uses **server UTC calendar date** (documented here for operators).
    date: Optional[str] = Field(
        default=None,
        description="ISO date YYYY-MM-DD; default = today in UTC on the server",
    )


@router.get("/daily-reports")
def list_daily_reports() -> Dict[str, List[str]]:
    return {"dates": _list_report_dates()}


@router.get("/daily-reports/{date}")
def get_daily_report(date: str) -> Dict[str, Any]:
    date = _validate_date(date)
    path = _path_for_date(date)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="该日尚无日报")
    return _read_report_file(path)


@router.post("/daily-reports/generate")
async def generate_daily_report(body: GenerateDailyReportBody) -> Dict[str, Any]:
    """
    Run the same pipeline as ``scripts/daily_report.py`` (async).

    When ``date`` is omitted, uses the server's **current UTC** calendar day.
    """
    if body.date is None or not str(body.date).strip():
        date = datetime.now(timezone.utc).date().isoformat()
    else:
        date = _validate_date(str(body.date).strip())

    from apps.reporting.pipeline import ReportPipeline
    from apps.reporting.report.llm_caller import drain_litellm_async_logging

    goal = (config.REPORT_DAILY_GOAL or "").strip()
    pipeline = ReportPipeline()
    try:
        try:
            result = await pipeline.generate(
                date=date,
                daily_goal=goal,
                language="auto",
            )
        except Exception as e:
            logger.exception("daily report pipeline failed: %s", e)
            raise HTTPException(
                status_code=502,
                detail="生成失败，请稍后重试或检查网络与模型配置",
            ) from e

        _write_report_json(result, date)
        return {**result, "ok": True, "date": date}
    finally:
        try:
            await drain_litellm_async_logging()
        except Exception:
            pass
        try:
            await pipeline.fetcher.close()
        except Exception:
            pass
