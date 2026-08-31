"""Generate or refresh soul.md from user goals and assistant personality."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from config import config
from apps.reporting.report.llm_caller import LLMCaller
from apps.reporting.report.preferences import ReportPreferencesManager, report_preferences
from apps.reporting.report.prompts import SOUL_SYSTEM_PROMPT, SOUL_USER_TEMPLATE
from utils.logger import setup_logger

logger = setup_logger("report.soul_builder")

TONE_LABELS = {
    "soft": "柔和 — 温和鼓励、少批评",
    "normal": "正常 — 平衡、客观",
    "push": "Push — 直接鞭策、高标准",
}

PLANNING_LABELS = {
    "detailed-present": "注重细致的当下 — 具体时段、可执行微行动",
    "rough-overall": "粗略的总体规划 — 主题与方向，少细节",
}


async def generate_soul_md(
    prefs: Optional[ReportPreferencesManager] = None,
    llm: Optional[LLMCaller] = None,
    prior_reports_excerpt: str = "",
) -> Dict[str, Any]:
    """LLM-generate soul.md from stored preferences; persist and return metadata."""
    mgr = prefs or report_preferences
    data = mgr.get_all()
    long_term: List[str] = data.get("long_term_goals") or []
    tone = mgr.resolve_tone()
    planning = mgr.resolve_planning_style()
    existing = mgr.read_soul()

    caller = llm or LLMCaller(
        api_key=config.REPORT_API_KEY,
        api_base=config.REPORT_API_BASE,
    )

    user_prompt = SOUL_USER_TEMPLATE.format(
        long_term_goals="\n".join(f"- {g}" for g in long_term) or "(尚未设置长期目标)",
        assistant_tone=TONE_LABELS.get(tone, tone),
        planning_style=PLANNING_LABELS.get(planning, planning),
        existing_soul=existing.strip() or "(尚无 soul.md，请新建)",
        prior_reports=prior_reports_excerpt.strip() or "(无历史日报摘要)",
    )

    result = await caller.call(
        model=config.REPORT_REDUCE_MODEL,
        system_prompt=SOUL_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.4,
        max_tokens=2048,
    )

    content = (result.content or "").strip()
    if not content:
        raise RuntimeError("soul.md 生成结果为空")

    mgr.write_soul(content)
    updated_at = mgr.mark_soul_updated()

    logger.info(
        "soul.md regenerated (%d chars) | LLM in=%d out=%d",
        len(content),
        result.input_tokens,
        result.output_tokens,
    )

    return {
        "content": content,
        "updated_at": updated_at,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }
