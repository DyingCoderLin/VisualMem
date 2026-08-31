"""Persistent user preferences for daily report: goals, assistant personality."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from config import config
from utils.logger import setup_logger

logger = setup_logger("report.preferences")

AssistantTone = Literal["soft", "normal", "push"]
PlanningStyle = Literal["detailed-present", "rough-overall"]

DEFAULT_PREFS: Dict[str, Any] = {
    "long_term_goals": [],
    "assistant_tone": "normal",
    "planning_style": "detailed-present",
    "date_goals": {},
    "soul_updated_at": None,
}


class ReportPreferencesManager:
    """JSON-backed store under STORAGE_ROOT/report_preferences.json."""

    def __init__(self, storage_path: str | None = None):
        if storage_path is None:
            storage_root = getattr(config, "STORAGE_ROOT", "./visualmem_storage")
            storage_path = os.path.join(storage_root, "report_preferences.json")
        self.storage_path = storage_path
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.isfile(self.storage_path):
            self._data = dict(DEFAULT_PREFS)
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                self._data = dict(DEFAULT_PREFS)
                return
            merged = dict(DEFAULT_PREFS)
            merged.update(raw)
            if not isinstance(merged.get("long_term_goals"), list):
                merged["long_term_goals"] = []
            if not isinstance(merged.get("date_goals"), dict):
                merged["date_goals"] = {}
            self._data = merged
        except Exception as e:
            logger.error("Failed to load report preferences: %s", e)
            self._data = dict(DEFAULT_PREFS)

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to save report preferences: %s", e)
            raise

    def get_all(self) -> Dict[str, Any]:
        return dict(self._data)

    def update(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        if "long_term_goals" in patch:
            goals = patch["long_term_goals"]
            if isinstance(goals, str):
                goals = [g.strip() for g in goals.split("\n") if g.strip()]
            if isinstance(goals, list):
                self._data["long_term_goals"] = [str(g).strip() for g in goals if str(g).strip()]
        if "assistant_tone" in patch and patch["assistant_tone"] in ("soft", "normal", "push"):
            self._data["assistant_tone"] = patch["assistant_tone"]
        if "planning_style" in patch and patch["planning_style"] in (
            "detailed-present",
            "rough-overall",
        ):
            self._data["planning_style"] = patch["planning_style"]
        if "date_goals" in patch and isinstance(patch["date_goals"], dict):
            self._data["date_goals"] = {
                str(k): str(v).strip()
                for k, v in patch["date_goals"].items()
                if str(v).strip()
            }
        self._save()
        return self.get_all()

    def set_date_goal(self, date: str, goal: str) -> None:
        date_goals: Dict[str, str] = self._data.setdefault("date_goals", {})
        goal = (goal or "").strip()
        if goal:
            date_goals[date] = goal
        elif date in date_goals:
            del date_goals[date]
        self._save()

    def get_date_goal(self, date: str) -> str:
        return (self._data.get("date_goals") or {}).get(date, "")

    def resolve_daily_goal(self, date: str, override: str = "") -> str:
        if (override or "").strip():
            return override.strip()
        stored = self.get_date_goal(date)
        if stored:
            return stored
        env_goal = (getattr(config, "REPORT_DAILY_GOAL", "") or "").strip()
        return env_goal

    def resolve_tone(self, override: Optional[str] = None) -> AssistantTone:
        if override in ("soft", "normal", "push"):
            return override  # type: ignore[return-value]
        tone = self._data.get("assistant_tone", "normal")
        return tone if tone in ("soft", "normal", "push") else "normal"

    def resolve_planning_style(self, override: Optional[str] = None) -> PlanningStyle:
        if override in ("detailed-present", "rough-overall"):
            return override  # type: ignore[return-value]
        style = self._data.get("planning_style", "detailed-present")
        return style if style in ("detailed-present", "rough-overall") else "detailed-present"

    def mark_soul_updated(self) -> str:
        ts = datetime.now(timezone.utc).isoformat()
        self._data["soul_updated_at"] = ts
        self._save()
        return ts

    def soul_path(self) -> str:
        storage_root = getattr(config, "STORAGE_ROOT", "./visualmem_storage")
        return os.path.join(storage_root, "soul.md")

    def read_soul(self) -> str:
        path = self.soul_path()
        if not os.path.isfile(path):
            return ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError as e:
            logger.warning("Failed to read soul.md: %s", e)
            return ""

    def write_soul(self, content: str) -> None:
        path = self.soul_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.rstrip() + "\n")


report_preferences = ReportPreferencesManager()
