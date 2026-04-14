"""
Session-bounded dynamic chunking for the Map phase.

Groups activity sessions into chunks respecting:
  1. Session boundaries (never cuts mid-session)
  2. Token budget per chunk (REPORT_CHUNK_TOKEN_LIMIT)
  3. Time gaps (splits on gaps > REPORT_GAP_MINUTES)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from apps.reporting.report.models import Chunk
from utils.logger import setup_logger

logger = setup_logger("report.chunker")


def estimate_tokens(text: str) -> int:
    """Rough token count: len/3 works for mixed CJK/English."""
    if not text:
        return 0
    return max(1, len(text) // 3)


def _parse_ts(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str)


def _gap_minutes(prev_end: str, next_start: str) -> float:
    """Minutes between previous session end and next session start."""
    try:
        t1 = _parse_ts(prev_end)
        t2 = _parse_ts(next_start)
        return (t2 - t1).total_seconds() / 60.0
    except (ValueError, TypeError):
        return 0.0


def build_chunks(
    sessions: List[Dict[str, Any]],
    token_limit: int = 3000,
    gap_minutes_threshold: int = 10,
) -> List[Chunk]:
    """Split activity sessions into chunks for Map processing.

    Args:
        sessions: list of activity dicts from /api/timeline/activities
                  each has: start, end, app, label, duration_seconds
        token_limit: max estimated tokens per chunk
        gap_minutes_threshold: force split when gap exceeds this

    Returns:
        list of Chunk objects
    """
    if not sessions:
        return []

    chunks: List[Chunk] = []
    current_sessions: List[Dict] = []
    current_tokens = 0

    for session in sessions:
        label = session.get("label", "")
        app = session.get("app", "")
        session_text = f"{app}: {label}"
        session_tokens = estimate_tokens(session_text) + 20

        if current_sessions:
            prev_end = current_sessions[-1].get("end", "")
            next_start = session.get("start", "")
            gap = _gap_minutes(prev_end, next_start)

            should_split = (
                (current_tokens + session_tokens > token_limit)
                or (gap > gap_minutes_threshold)
            )
            if should_split:
                chunks.append(_finalize_chunk(current_sessions, current_tokens))
                current_sessions = []
                current_tokens = 0

        current_sessions.append(session)
        current_tokens += session_tokens

    if current_sessions:
        chunks.append(_finalize_chunk(current_sessions, current_tokens))

    logger.info("Built %d chunks from %d sessions", len(chunks), len(sessions))
    for i, ch in enumerate(chunks):
        apps = sorted({(s.get("app") or "?") for s in ch.sessions})
        logger.debug(
            "chunk[%d] time=[%s .. %s] est_tokens~%d apps=%s sessions=%d",
            i,
            ch.start[:19] if ch.start else "",
            ch.end[:19] if ch.end else "",
            ch.estimated_tokens,
            apps,
            len(ch.sessions),
        )
    return chunks


def _finalize_chunk(sessions: List[Dict], estimated_tokens: int) -> Chunk:
    """Create a Chunk from accumulated sessions."""
    start = sessions[0].get("start", "")
    end = sessions[-1].get("end", "")
    return Chunk(
        sessions=sessions,
        start=start,
        end=end,
        estimated_tokens=estimated_tokens,
    )
