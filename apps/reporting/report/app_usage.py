"""Per-app foreground minutes (deterministic) + purpose lines from Reduce LLM."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


def _norm_app(name: str) -> str:
    return (name or "").strip()


def build_app_usage_minutes_only(day_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Structured rows: app + focused_minutes + percentage + frame_count (no purpose yet)."""
    apps = (day_data.get("app_focused_time") or {}).get("apps") or []
    out: List[Dict[str, Any]] = []
    for a in apps:
        out.append({
            "app": a.get("app"),
            "focused_minutes": a.get("focused_minutes"),
            "percentage": a.get("percentage"),
            "frame_count": a.get("frame_count"),
        })
    return out


def format_app_usage_hints_for_reduce(day_data: Dict[str, Any]) -> str:
    """Evidence for the Reduce model to infer per-app 用途 (session labels + window titles)."""
    apps_ft = (day_data.get("app_focused_time") or {}).get("apps") or []
    activities = (day_data.get("activities") or {}).get("activities") or []
    window_titles = (day_data.get("window_titles") or {}).get("window_titles") or []

    # Session labels per app (duration-weighted top labels)
    label_sec: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for s in activities:
        app = _norm_app(s.get("app") or "")
        lab = (s.get("label") or "").strip()
        if not app or not lab:
            continue
        label_sec[app][lab] += float(s.get("duration_seconds") or 0)

    lines: List[str] = []
    lines.append(
        "### Foreground minutes (same app names must appear in JSON `app_purpose`)\n"
    )
    for a in apps_ft:
        app = a.get("app")
        lines.append(
            f"- {app}: {a.get('focused_minutes')} min ({a.get('percentage')}%)"
        )

    lines.append("\n### Evidence hints per app (for writing purpose_keywords — not final copy)")
    for a in apps_ft:
        app = _norm_app(a.get("app") or "")
        if not app:
            continue
        sub: List[str] = []

        # Top session labels by duration
        by_lab = label_sec.get(app) or {}
        if not by_lab:
            for k, v in label_sec.items():
                if k.casefold() == app.casefold():
                    by_lab = v
                    break
        if by_lab:
            top_labs = sorted(by_lab.keys(), key=lambda L: by_lab[L], reverse=True)[:5]
            sub.append("活动会话标签: " + "；".join(top_labs))

        # Window titles for this app
        wins = [
            w for w in window_titles
            if _norm_app(w.get("app") or "").casefold() == app.casefold()
        ]
        wins.sort(key=lambda w: int(w.get("count") or 0), reverse=True)
        if wins:
            titles = [f"{w.get('window')} (×{w.get('count')})" for w in wins[:6]]
            sub.append("常见窗口标题: " + " | ".join(titles))

        if sub:
            lines.append(f"\n**{a.get('app')}**\n" + "\n".join(sub))
        else:
            lines.append(f"\n**{a.get('app')}**\n(无会话标签与窗口样本)")

    return "\n".join(lines)


def merge_app_usage_with_llm_purpose(
    base_rows: List[Dict[str, Any]],
    app_purpose: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach purpose_keywords from Reduce `app_purpose` to minute rows (by app name)."""
    pmap: Dict[str, List[str]] = {}
    for p in app_purpose:
        if not isinstance(p, dict):
            continue
        app = _norm_app(p.get("app") or "")
        kws = p.get("purpose_keywords") or p.get("keywords")
        if not isinstance(kws, list):
            kws = []
        phrases = [str(x).strip() for x in kws if str(x).strip()]
        if app:
            pmap[app] = phrases

    def lookup(app: str) -> List[str]:
        if app in pmap:
            return pmap[app]
        for k, v in pmap.items():
            if k.casefold() == app.casefold():
                return v
        return []

    out: List[Dict[str, Any]] = []
    for row in base_rows:
        app = row.get("app")
        out.append({
            **row,
            "purpose_keywords": lookup(_norm_app(app or "")),
        })
    return out
