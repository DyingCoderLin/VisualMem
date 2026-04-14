"""Load prior daily_report_*.json files for Reduce-time context (optional contrast for today_summary)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

from utils.logger import setup_logger

logger = setup_logger("report.history_loader")


def _extract_report_blob(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    if "report" in raw and isinstance(raw["report"], dict):
        return raw["report"]
    return raw


def _compact_one_day(path: str, max_accomplishments: int = 4) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("history_loader: skip %s: %r", path, e)
        return ""

    rep = _extract_report_blob(raw)
    date = rep.get("date") or "?"

    wm = rep.get("work_module") or {}
    lm = rep.get("life_module") or {}

    lines: List[str] = [f"#### {date}"]

    fs = lm.get("focus_score")
    if fs is not None:
        lines.append(f"- focus_score: {fs}")

    fi = (lm.get("focus_interpretation") or "").strip()
    if fi:
        snippet = fi[:280] + ("…" if len(fi) > 280 else "")
        lines.append(f"- focus note: {snippet}")

    acc = wm.get("core_accomplishments") or []
    if acc:
        lines.append("- core_accomplishments (excerpts):")
        for a in acc[:max_accomplishments]:
            s = (a or "").strip()
            if not s:
                continue
            lines.append(f"  - {s[:500]}{'…' if len(s) > 500 else ''}")

    sugg = wm.get("tomorrow_suggestions") or []
    if sugg:
        lines.append("- tomorrow_suggestions (what past-you asked for next):")
        for s in sugg:
            t = (s or "").strip()
            if not t:
                continue
            lines.append(f"  - {t[:500]}{'…' if len(t) > 500 else ''}")

    block = wm.get("blockers_and_unfinished") or []
    if block:
        lines.append("- blockers / unfinished (excerpts):")
        for b in block[:3]:
            u = (b or "").strip()
            if u:
                lines.append(f"  - {u[:400]}{'…' if len(u) > 400 else ''}")

    summ = rep.get("today_summary") or rep.get("cross_day_reflection") or []
    if isinstance(summ, list) and summ:
        lines.append("- today_summary / 小结 (excerpts):")
        for line in summ[:5]:
            u = (line or "").strip()
            if u:
                lines.append(f"  - {u[:400]}{'…' if len(u) > 400 else ''}")

    return "\n".join(lines)


def load_prior_reports_for_reduce(
    current_date_iso: str,
    log_dir: str,
    max_prior_days: int,
    max_total_chars: int = 14000,
) -> str:
    """
    Read daily_report_<YYYY-MM-DD>.json for days strictly before current_date_iso.

    Walks backward day-by-day up to *max_prior_days* calendar steps and includes
    each file that exists. Output is markdown-ish text for the Reduce prompt.
    """
    try:
        current = datetime.strptime(current_date_iso[:10], "%Y-%m-%d").date()
    except ValueError:
        logger.warning("history_loader: bad current_date_iso %r", current_date_iso)
        return ""

    chunks: List[str] = []
    total = 0
    for delta in range(1, max_prior_days + 1):
        d = current - timedelta(days=delta)
        name = f"daily_report_{d.isoformat()}.json"
        path = os.path.join(log_dir, name)
        if not os.path.isfile(path):
            continue
        block = _compact_one_day(path)
        if not block:
            continue
        sep = "\n\n"
        if total + len(block) + len(sep) > max_total_chars:
            chunks.append(
                "\n… [prior report text truncated at max_total_chars budget]\n"
            )
            break
        chunks.append(block)
        total += len(block) + len(sep)

    if not chunks:
        return ""

    header = (
        "The following excerpts are from **previous** daily_report JSON files "
        f"under `{log_dir}` (newest first among listed days). "
        "Use as **optional** context for `today_summary` (contrast, continuity) and "
        "for follow-through on prior `tomorrow_suggestions` / focus_score trends — "
        "today's Map summaries and metrics must stay primary.\n\n"
    )
    return header + "\n\n".join(chunks)
