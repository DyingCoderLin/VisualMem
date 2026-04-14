"""
Data Platform API Router — FastAPI endpoints for structured data access.

This is the topmost, external-facing API layer.  It handles:
  - Request parameter parsing & validation
  - Timezone conversion (local ↔ UTC)
  - Boundary clipping for timeline queries
  - Response formatting

All heavy data logic lives in DataService (data_service.py).
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from core.api.data_service import DataService
from core.api.time_utils import parse_utc, to_local_iso, db_ts_to_local
from utils.logger import setup_logger

logger = setup_logger("api.router")

router = APIRouter(prefix="/api", tags=["data-platform"])

# Injected at startup by gui_backend_server
_data_service: Optional[DataService] = None


def init_data_service(db_path: str, activity_db_path: str = None):
    """Called once at server startup to wire the DataService dependency."""
    global _data_service
    _data_service = DataService(db_path, activity_db_path)
    logger.info(f"DataService initialized (db={db_path}, activity_db={activity_db_path})")


def _svc() -> DataService:
    if _data_service is None:
        raise HTTPException(status_code=503, detail="DataService not initialized")
    return _data_service


# ======================================================================
# Timeline
# ======================================================================

@router.get("/timeline/apps")
def timeline_apps(
    start: str = Query(..., description="ISO datetime, query range start"),
    end: str = Query(..., description="ISO datetime, query range end"),
):
    """Activity segments grouped by app, with boundary clipping."""
    start_utc = parse_utc(start)
    end_utc = parse_utc(end)

    sessions = _svc().get_activity_sessions(start_utc, end_utc)

    from collections import defaultdict
    apps_map: dict[str, list] = defaultdict(list)

    for s in sessions:
        s_start = datetime.fromisoformat(s["start_time"])
        s_end = datetime.fromisoformat(s["end_time"])
        if s_start.tzinfo is None:
            s_start = s_start.replace(tzinfo=timezone.utc)
        if s_end.tzinfo is None:
            s_end = s_end.replace(tzinfo=timezone.utc)

        clipped_start = max(s_start, start_utc)
        clipped_end = min(s_end, end_utc)
        if clipped_start > clipped_end:
            continue

        duration_sec = (clipped_end - clipped_start).total_seconds()
        apps_map[s["app_name"]].append({
            "start": to_local_iso(clipped_start),
            "end": to_local_iso(clipped_end),
            "activity_label": s["label"],
            "duration_seconds": round(duration_sec),
        })

    apps_list = [
        {"app_name": app, "segments": segs}
        for app, segs in sorted(apps_map.items())
    ]

    return {
        "query_start": to_local_iso(start_utc),
        "query_end": to_local_iso(end_utc),
        "apps": apps_list,
    }


@router.get("/timeline/activities")
def timeline_activities(
    start: str = Query(..., description="ISO datetime, query range start"),
    end: str = Query(..., description="ISO datetime, query range end"),
):
    """Activity timeline in chronological order (not grouped by app)."""
    start_utc = parse_utc(start)
    end_utc = parse_utc(end)

    sessions = _svc().get_activity_sessions(start_utc, end_utc)

    activities = []
    for s in sessions:
        s_start = datetime.fromisoformat(s["start_time"])
        s_end = datetime.fromisoformat(s["end_time"])
        if s_start.tzinfo is None:
            s_start = s_start.replace(tzinfo=timezone.utc)
        if s_end.tzinfo is None:
            s_end = s_end.replace(tzinfo=timezone.utc)

        clipped_start = max(s_start, start_utc)
        clipped_end = min(s_end, end_utc)
        if clipped_start > clipped_end:
            continue

        duration_sec = (clipped_end - clipped_start).total_seconds()
        activities.append({
            "start": to_local_iso(clipped_start),
            "end": to_local_iso(clipped_end),
            "app": s["app_name"],
            "label": s["label"],
            "duration_seconds": round(duration_sec),
        })

    return {
        "query_start": to_local_iso(start_utc),
        "query_end": to_local_iso(end_utc),
        "activities": activities,
    }


# ======================================================================
# OCR
# ======================================================================

@router.get("/ocr/texts")
def ocr_texts(
    start: str = Query(..., description="ISO datetime"),
    end: str = Query(..., description="ISO datetime"),
    app_name: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    """OCR text records within a time range, with optional FTS5 keyword search."""
    start_utc = parse_utc(start)
    end_utc = parse_utc(end)

    total, items = _svc().get_ocr_texts(
        start_utc, end_utc, app_name, keyword,
        min_confidence, offset, limit,
    )

    for item in items:
        if item.get("timestamp"):
            item["timestamp"] = db_ts_to_local(item["timestamp"])

    return {
        "query_start": to_local_iso(start_utc),
        "query_end": to_local_iso(end_utc),
        "total_count": total,
        "items": items,
    }


@router.get("/ocr/regions")
def ocr_regions(
    sub_frame_id: str = Query(..., description="Target sub_frame ID"),
):
    """Region-level OCR with bounding box coordinates."""
    return _svc().get_ocr_regions(sub_frame_id)


# ======================================================================
# Focus & Analytics
# ======================================================================

@router.get("/focus/score")
def focus_score(
    start: str = Query(..., description="ISO datetime"),
    end: str = Query(..., description="ISO datetime"),
):
    """Focus score and detailed metrics for a time range."""
    start_utc = parse_utc(start)
    end_utc = parse_utc(end)

    data = _svc().get_focus_data(start_utc, end_utc)

    for block in data["metrics"].get("deep_work_blocks", []):
        block["start"] = db_ts_to_local(block["start"])
        block["end"] = db_ts_to_local(block["end"])
    for period in data["metrics"].get("fragmented_periods", []):
        period["start"] = db_ts_to_local(period["start"])
        period["end"] = db_ts_to_local(period["end"])

    return {
        "query_start": to_local_iso(start_utc),
        "query_end": to_local_iso(end_utc),
        **data,
    }


@router.get("/context/keywords")
def context_keywords(
    start: str = Query(..., description="ISO datetime"),
    end: str = Query(..., description="ISO datetime"),
    top_k: int = Query(20, ge=1, le=100),
):
    """High-frequency keywords extracted from OCR text in a time range."""
    start_utc = parse_utc(start)
    end_utc = parse_utc(end)

    data = _svc().get_context_keywords(start_utc, end_utc, top_k)

    return {
        "query_start": to_local_iso(start_utc),
        "query_end": to_local_iso(end_utc),
        **data,
    }


@router.get("/context/keywords-by-app")
def context_keywords_by_app(
    start: str = Query(..., description="ISO datetime"),
    end: str = Query(..., description="ISO datetime"),
    per_app: int = Query(5, ge=1, le=20, description="Top keywords per app from OCR"),
):
    """Per-app keyword frequencies from OCR (join ocr_text + sub_frames), ordered by OCR volume."""
    start_utc = parse_utc(start)
    end_utc = parse_utc(end)

    apps = _svc().get_keywords_by_app(start_utc, end_utc, top_per_app=per_app)

    return {
        "query_start": to_local_iso(start_utc),
        "query_end": to_local_iso(end_utc),
        "apps": apps,
    }


@router.get("/apps/focused-time")
def apps_focused_time(
    start: str = Query(..., description="ISO datetime"),
    end: str = Query(..., description="ISO datetime"),
):
    """Per-app focused time: count of capture intervals where frames.focused_app_name == app × CAPTURE_INTERVAL.

    This is **foreground window** time at each sampled frame, not eye-tracking or “attention to UI text”.
    If another app is focused, this app does not accrue. A window left idle but still foreground still accrues.
    """
    start_utc = parse_utc(start)
    end_utc = parse_utc(end)

    apps = _svc().get_app_focused_minutes(start_utc, end_utc)
    total = round(sum(a["focused_minutes"] for a in apps), 1)

    return {
        "query_start": to_local_iso(start_utc),
        "query_end": to_local_iso(end_utc),
        "total_focused_minutes": total,
        "apps": apps,
    }


@router.get("/recording/span")
def recording_span(
    start: str = Query(..., description="ISO datetime"),
    end: str = Query(..., description="ISO datetime"),
):
    """First and last active timestamps within a time range."""
    start_utc = parse_utc(start)
    end_utc = parse_utc(end)

    first_ts, last_ts = _svc().get_first_last_active(start_utc, end_utc)

    return {
        "query_start": to_local_iso(start_utc),
        "query_end": to_local_iso(end_utc),
        "first_active": db_ts_to_local(first_ts) if first_ts else None,
        "last_active": db_ts_to_local(last_ts) if last_ts else None,
    }


@router.get("/activities/breakdown")
def activities_breakdown(
    start: str = Query(..., description="ISO datetime"),
    end: str = Query(..., description="ISO datetime"),
):
    """Per activity cluster label: focused minutes where that label’s sub_frame matched the focused app.

    Same **foreground** semantics as /apps/focused-time; labels describe clustered OCR/window context, not duration of reading one line of text.
    """
    start_utc = parse_utc(start)
    end_utc = parse_utc(end)

    breakdown = _svc().get_activity_breakdown(start_utc, end_utc)

    return {
        "query_start": to_local_iso(start_utc),
        "query_end": to_local_iso(end_utc),
        "breakdown": breakdown,
    }


@router.get("/windows/titles")
def windows_titles(
    start: str = Query(..., description="ISO datetime"),
    end: str = Query(..., description="ISO datetime"),
    limit: int = Query(50, ge=1, le=200),
):
    """Top window/tab titles with frequency counts."""
    start_utc = parse_utc(start)
    end_utc = parse_utc(end)

    windows = _svc().get_window_names_in_range(start_utc, end_utc, limit)

    return {
        "query_start": to_local_iso(start_utc),
        "query_end": to_local_iso(end_utc),
        "window_titles": [
            {"app": w["app_name"], "window": w["window_name"], "count": w["cnt"]}
            for w in windows
        ],
    }


# ======================================================================
# Context Snapshot
# ======================================================================

@router.get("/context/snapshot")
def context_snapshot(
    timestamp: str = Query(..., description="ISO datetime — nearest frame will be found"),
):
    """Full context at the moment closest to the given timestamp."""
    ts_utc = parse_utc(timestamp)
    data = _svc().get_context_snapshot(ts_utc)

    if data.get("timestamp"):
        data["timestamp"] = db_ts_to_local(data["timestamp"])

    return data
