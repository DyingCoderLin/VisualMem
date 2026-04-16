"""
Data Service — pure data-access layer for the Data Platform API.

All methods accept UTC datetimes, query SQLite, and return plain dicts/lists.
Timezone conversion (UTC ↔ local) happens in the router layer, not here.

Dual-DB architecture:
  - Main DB (OCR_DB_PATH): frames, sub_frames, ocr_text, ocr_regions, etc.
  - Activity DB (ACTIVITY_DB_PATH): activity_sessions, activity_clusters,
    activity_assignments, pending_groups.
"""

import sqlite3
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from config import config
from utils.logger import setup_logger

logger = setup_logger("api.data_service")


class DataService:
    """Stateless query service backed by the VisualMem SQLite databases."""

    def __init__(self, db_path: str, activity_db_path: str = None):
        self.db_path = db_path
        self.activity_db_path = activity_db_path or db_path

    def _connect(self) -> sqlite3.Connection:
        """Connect to the main DB (frames, sub_frames, ocr_text, etc.)."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _connect_activity(self) -> sqlite3.Connection:
        """Connect to the activity DB (activity_sessions, activity_clusters, etc.)."""
        conn = sqlite3.connect(self.activity_db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def get_activity_sessions(
        self,
        start_utc: datetime,
        end_utc: datetime,
    ) -> List[Dict[str, Any]]:
        """Return committed activity_sessions that overlap [start_utc, end_utc].

        Queries the activity DB (authoritative source).
        """
        conn = self._connect_activity()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT app_name, label, start_time, end_time, frame_count
                FROM activity_sessions
                WHERE end_time >= ? AND start_time <= ?
                  AND session_status = 'committed'
                ORDER BY start_time
                """,
                (start_utc.isoformat(), end_utc.isoformat()),
            )
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            # Table may not exist yet
            return []
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # OCR texts
    # ------------------------------------------------------------------

    def get_ocr_texts(
        self,
        start_utc: datetime,
        end_utc: datetime,
        app_name: Optional[str] = None,
        keyword: Optional[str] = None,
        min_confidence: float = 0.0,
        offset: int = 0,
        limit: int = 50,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """Return OCR text records with sub_frame context.

        Returns (total_count, items).
        """
        conn = self._connect()
        try:
            start_iso = start_utc.isoformat()
            end_iso = end_utc.isoformat()

            if keyword:
                return self._get_ocr_texts_fts(
                    conn, start_iso, end_iso, app_name, keyword,
                    min_confidence, offset, limit,
                )
            return self._get_ocr_texts_plain(
                conn, start_iso, end_iso, app_name,
                min_confidence, offset, limit,
            )
        finally:
            conn.close()

    def _get_ocr_texts_plain(
        self, conn, start_iso, end_iso, app_name,
        min_confidence, offset, limit,
    ) -> Tuple[int, List[Dict]]:
        cursor = conn.cursor()
        where = "sf.timestamp >= ? AND sf.timestamp <= ? AND ot.confidence >= ?"
        params: list = [start_iso, end_iso, min_confidence]
        if app_name:
            where += " AND sf.app_name = ?"
            params.append(app_name)

        cursor.execute(
            f"SELECT COUNT(*) AS cnt FROM ocr_text ot "
            f"JOIN sub_frames sf ON ot.sub_frame_id = sf.sub_frame_id "
            f"WHERE {where}",
            params,
        )
        total = cursor.fetchone()["cnt"]

        cursor.execute(
            f"""
            SELECT ot.sub_frame_id, ot.text, ot.confidence, ot.text_length,
                   sf.timestamp, sf.app_name, sf.window_name, sf.activity_label
            FROM ocr_text ot
            JOIN sub_frames sf ON ot.sub_frame_id = sf.sub_frame_id
            WHERE {where}
            ORDER BY sf.timestamp
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        )
        items = [dict(row) for row in cursor.fetchall()]
        return total, items

    def _get_ocr_texts_fts(
        self, conn, start_iso, end_iso, app_name, keyword,
        min_confidence, offset, limit,
    ) -> Tuple[int, List[Dict]]:
        cursor = conn.cursor()

        extra_where = "AND sf.timestamp >= ? AND sf.timestamp <= ? AND ot.confidence >= ?"
        params: list = [keyword, start_iso, end_iso, min_confidence]
        if app_name:
            extra_where += " AND sf.app_name = ?"
            params.append(app_name)

        cursor.execute(
            f"""
            SELECT COUNT(*) AS cnt
            FROM ocr_text_fts fts
            JOIN ocr_text ot ON fts.rowid = ot.id
            JOIN sub_frames sf ON ot.sub_frame_id = sf.sub_frame_id
            WHERE fts.text MATCH ?
              {extra_where}
            """,
            params,
        )
        total = cursor.fetchone()["cnt"]

        cursor.execute(
            f"""
            SELECT ot.sub_frame_id, ot.text, ot.confidence, ot.text_length,
                   sf.timestamp, sf.app_name, sf.window_name, sf.activity_label
            FROM ocr_text_fts fts
            JOIN ocr_text ot ON fts.rowid = ot.id
            JOIN sub_frames sf ON ot.sub_frame_id = sf.sub_frame_id
            WHERE fts.text MATCH ?
              {extra_where}
            ORDER BY sf.timestamp
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        )
        items = [dict(row) for row in cursor.fetchall()]
        return total, items

    # ------------------------------------------------------------------
    # OCR regions
    # ------------------------------------------------------------------

    def get_ocr_regions(self, sub_frame_id: str) -> Dict[str, Any]:
        """Return region-level OCR for a specific sub_frame."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT region_index, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                       text, ocr_confidence, image_width, image_height
                FROM ocr_regions
                WHERE sub_frame_id = ?
                ORDER BY region_index
                """,
                (sub_frame_id,),
            )
            rows = cursor.fetchall()
            if not rows:
                return {"sub_frame_id": sub_frame_id, "image_width": 0, "image_height": 0, "regions": []}

            regions = []
            img_w = 0
            img_h = 0
            for r in rows:
                img_w = r["image_width"] or img_w
                img_h = r["image_height"] or img_h
                regions.append({
                    "region_index": r["region_index"],
                    "bbox": {
                        "x1": r["bbox_x1"], "y1": r["bbox_y1"],
                        "x2": r["bbox_x2"], "y2": r["bbox_y2"],
                    },
                    "text": r["text"],
                    "confidence": r["ocr_confidence"],
                })
            return {
                "sub_frame_id": sub_frame_id,
                "image_width": img_w,
                "image_height": img_h,
                "regions": regions,
            }
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Focus score
    # ------------------------------------------------------------------

    def get_focus_data(
        self,
        start_utc: datetime,
        end_utc: datetime,
    ) -> Dict[str, Any]:
        """Compute focus metrics from frames.focused_app_name transitions.

        Returns raw metrics dict — the router formats timestamps to local.
        """
        conn = self._connect()
        try:
            cursor = conn.cursor()
            start_iso = start_utc.isoformat()
            end_iso = end_utc.isoformat()

            cursor.execute(
                """
                SELECT timestamp, focused_app_name
                FROM frames
                WHERE timestamp >= ? AND timestamp <= ?
                  AND frame_id LIKE 'frame_%'
                  AND focused_app_name IS NOT NULL AND focused_app_name != ''
                ORDER BY timestamp
                """,
                (start_iso, end_iso),
            )
            rows = cursor.fetchall()
            if not rows:
                return {
                    "focus_score": 0,
                    "metrics": {
                        "longest_focus_streak_minutes": 0,
                        "avg_focus_streak_minutes": 0,
                        "app_switches_per_hour": 0,
                        "top_distraction_apps": [],
                        "deep_work_blocks": [],
                        "fragmented_periods": [],
                    },
                }

            capture_interval = config.CAPTURE_INTERVAL_SECONDS

            # Build streaks: consecutive frames with the same focused app
            streaks: List[Dict] = []
            current_app = rows[0]["focused_app_name"]
            streak_start = rows[0]["timestamp"]
            streak_frames = 1
            switches = 0

            for i in range(1, len(rows)):
                app = rows[i]["focused_app_name"]
                if app != current_app:
                    streaks.append({
                        "app": current_app,
                        "start": streak_start,
                        "end": rows[i - 1]["timestamp"],
                        "frames": streak_frames,
                    })
                    current_app = app
                    streak_start = rows[i]["timestamp"]
                    streak_frames = 1
                    switches += 1
                else:
                    streak_frames += 1
            streaks.append({
                "app": current_app,
                "start": streak_start,
                "end": rows[-1]["timestamp"],
                "frames": streak_frames,
            })

            for s in streaks:
                s["minutes"] = s["frames"] * capture_interval / 60.0

            total_minutes = len(rows) * capture_interval / 60.0
            total_hours = total_minutes / 60.0

            deep_work_blocks = [s for s in streaks if s["minutes"] >= 20]
            deep_work_minutes = sum(s["minutes"] for s in deep_work_blocks)

            streak_minutes = [s["minutes"] for s in streaks]
            longest = max(streak_minutes) if streak_minutes else 0
            avg_streak = sum(streak_minutes) / len(streak_minutes) if streak_minutes else 0

            switches_per_hour = switches / total_hours if total_hours > 0 else 0

            # Fragmented periods: bucket by hour
            hour_switches: Dict[str, int] = defaultdict(int)
            prev_app = rows[0]["focused_app_name"]
            for i in range(1, len(rows)):
                if rows[i]["focused_app_name"] != prev_app:
                    ts = datetime.fromisoformat(rows[i]["timestamp"])
                    hour_key = ts.strftime("%Y-%m-%dT%H:00:00")
                    hour_switches[hour_key] += 1
                    prev_app = rows[i]["focused_app_name"]
                else:
                    prev_app = rows[i]["focused_app_name"]

            fragmented = []
            for hour_start_str, count in sorted(hour_switches.items()):
                if count >= 15:
                    hs = datetime.fromisoformat(hour_start_str)
                    fragmented.append({
                        "start": hour_start_str,
                        "end": (hs + timedelta(hours=1)).isoformat(),
                        "switches": count,
                    })

            # Top distraction apps: apps with many short streaks (< 2 min)
            short_streak_app_count: Dict[str, int] = defaultdict(int)
            for s in streaks:
                if s["minutes"] < 2:
                    short_streak_app_count[s["app"]] += 1
            top_distractions = sorted(
                short_streak_app_count.keys(),
                key=lambda a: short_streak_app_count[a],
                reverse=True,
            )[:5]

            # Focus score: 0-100
            deep_ratio = min(deep_work_minutes / total_minutes, 1.0) if total_minutes > 0 else 0
            switch_score = max(0, 1.0 - switches_per_hour / 30.0)
            longest_score = min(longest / 60.0, 1.0)
            focus_score = int(round(deep_ratio * 40 + switch_score * 30 + longest_score * 30))

            return {
                "focus_score": focus_score,
                "metrics": {
                    "longest_focus_streak_minutes": round(longest, 1),
                    "avg_focus_streak_minutes": round(avg_streak, 1),
                    "app_switches_per_hour": round(switches_per_hour, 1),
                    "top_distraction_apps": top_distractions,
                    "deep_work_blocks": [
                        {"start": s["start"], "end": s["end"], "app": s["app"],
                         "minutes": round(s["minutes"], 1)}
                        for s in deep_work_blocks
                    ],
                    "fragmented_periods": fragmented,
                },
            }
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Context keywords
    # ------------------------------------------------------------------

    def get_context_keywords(
        self,
        start_utc: datetime,
        end_utc: datetime,
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """Extract high-frequency keywords from OCR text in a time range."""
        import re

        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ot.text, sf.app_name
                FROM ocr_text ot
                JOIN sub_frames sf ON ot.sub_frame_id = sf.sub_frame_id
                WHERE sf.timestamp >= ? AND sf.timestamp <= ?
                  AND ot.text_length > 0
                """,
                (start_utc.isoformat(), end_utc.isoformat()),
            )

            word_count: Dict[str, int] = defaultdict(int)
            word_apps: Dict[str, set] = defaultdict(set)

            stop_words = {
                "the", "a", "an", "is", "are", "was", "were", "be", "been",
                "being", "have", "has", "had", "do", "does", "did", "will",
                "would", "could", "should", "may", "might", "can", "shall",
                "to", "of", "in", "for", "on", "with", "at", "by", "from",
                "as", "into", "through", "during", "before", "after", "and",
                "but", "or", "not", "no", "if", "then", "else", "when",
                "up", "out", "so", "than", "too", "very", "just", "about",
                "it", "its", "this", "that", "these", "those", "he", "she",
                "they", "we", "you", "i", "me", "my", "your", "his", "her",
                "our", "their", "what", "which", "who", "whom", "how",
                "all", "each", "every", "both", "few", "more", "most",
                "other", "some", "such", "only", "own", "same", "new",
                "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
                "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
                "你", "会", "着", "没有", "看", "好", "自己", "这",
            }

            for row in cursor.fetchall():
                text = row["text"] or ""
                app = row["app_name"] or ""
                tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)
                cjk_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", text)
                for token in tokens:
                    lower = token.lower()
                    if lower not in stop_words and len(lower) <= 50:
                        word_count[token] += 1
                        word_apps[token].add(app)
                for token in cjk_tokens:
                    if token not in stop_words:
                        word_count[token] += 1
                        word_apps[token].add(app)

            sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:top_k]

            keywords = [
                {"word": word, "count": count, "apps": sorted(word_apps[word])}
                for word, count in sorted_words
            ]
            return {"keywords": keywords}
        finally:
            conn.close()

    def get_keywords_by_app(
        self,
        start_utc: datetime,
        end_utc: datetime,
        top_per_app: int = 5,
    ) -> List[Dict[str, Any]]:
        """Per-app top keywords from OCR text (same tokenization rules as get_context_keywords)."""
        import re

        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ot.text, sf.app_name
                FROM ocr_text ot
                JOIN sub_frames sf ON ot.sub_frame_id = sf.sub_frame_id
                WHERE sf.timestamp >= ? AND sf.timestamp <= ?
                  AND ot.text_length > 0
                """,
                (start_utc.isoformat(), end_utc.isoformat()),
            )

            stop_words = {
                "the", "a", "an", "is", "are", "was", "were", "be", "been",
                "being", "have", "has", "had", "do", "does", "did", "will",
                "would", "could", "should", "may", "might", "can", "shall",
                "to", "of", "in", "for", "on", "with", "at", "by", "from",
                "as", "into", "through", "during", "before", "after", "and",
                "but", "or", "not", "no", "if", "then", "else", "when",
                "up", "out", "so", "than", "too", "very", "just", "about",
                "it", "its", "this", "that", "these", "those", "he", "she",
                "they", "we", "you", "i", "me", "my", "your", "his", "her",
                "our", "their", "what", "which", "who", "whom", "how",
                "all", "each", "every", "both", "few", "more", "most",
                "other", "some", "such", "only", "own", "same", "new",
                "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
                "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
                "你", "会", "着", "没有", "看", "好", "自己", "这",
            }

            per_app: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

            for row in cursor.fetchall():
                text = row["text"] or ""
                app = (row["app_name"] or "").strip() or "_unknown"
                tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)
                cjk_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", text)
                for token in tokens:
                    lower = token.lower()
                    if lower not in stop_words and len(lower) <= 50:
                        per_app[app][token] += 1
                for token in cjk_tokens:
                    if token not in stop_words:
                        per_app[app][token] += 1

            out: List[Dict[str, Any]] = []
            for app in sorted(per_app.keys(), key=lambda a: sum(per_app[a].values()), reverse=True):
                wc = per_app[app]
                top = sorted(wc.items(), key=lambda x: x[1], reverse=True)[:top_per_app]
                out.append({
                    "app": app,
                    "keywords": [{"word": w, "count": c} for w, c in top],
                })
            return out
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # App focused time
    # ------------------------------------------------------------------

    def get_app_focused_minutes(
        self,
        start_utc: datetime,
        end_utc: datetime,
    ) -> List[Dict[str, Any]]:
        """Get per-app focused time (based on frames.focused_app_name count × capture interval)."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            interval = config.CAPTURE_INTERVAL_SECONDS
            cursor.execute(
                """
                SELECT focused_app_name, COUNT(*) AS frame_count
                FROM frames
                WHERE timestamp >= ? AND timestamp <= ?
                  AND frame_id LIKE 'frame_%'
                  AND focused_app_name IS NOT NULL AND focused_app_name != ''
                GROUP BY focused_app_name
                ORDER BY frame_count DESC
                """,
                (start_utc.isoformat(), end_utc.isoformat()),
            )
            total_frames = 0
            results = []
            for row in cursor.fetchall():
                results.append({
                    "app": row["focused_app_name"],
                    "focused_minutes": round(row["frame_count"] * interval / 60.0, 1),
                    "frame_count": row["frame_count"],
                })
                total_frames += row["frame_count"]

            for r in results:
                r["percentage"] = round(r["frame_count"] / total_frames * 100, 1) if total_frames > 0 else 0

            return results
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Recording span
    # ------------------------------------------------------------------

    def get_first_last_active(
        self,
        start_utc: datetime,
        end_utc: datetime,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Return (first_active_ts, last_active_ts) as raw DB strings, or (None, None)."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT MIN(timestamp) AS first_ts, MAX(timestamp) AS last_ts
                FROM frames
                WHERE timestamp >= ? AND timestamp <= ?
                  AND frame_id LIKE 'frame_%'
                """,
                (start_utc.isoformat(), end_utc.isoformat()),
            )
            row = cursor.fetchone()
            if row and row["first_ts"]:
                return row["first_ts"], row["last_ts"]
            return None, None
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Activity breakdown (cross-DB: main DB frames + activity DB labels)
    # ------------------------------------------------------------------

    def get_activity_breakdown(
        self,
        start_utc: datetime,
        end_utc: datetime,
    ) -> List[Dict[str, Any]]:
        """Aggregate focused time per activity label.

        Uses ATTACH to cross-join main DB (frames, frame_subframe_mapping) with
        activity DB (activity_assignments) to find focused frames with labels.
        """
        conn = self._connect_activity()
        try:
            cursor = conn.cursor()
            interval = config.CAPTURE_INTERVAL_SECONDS

            # ATTACH main DB to query frames + mapping
            cursor.execute("ATTACH DATABASE ? AS main_db", (self.db_path,))

            cursor.execute(
                """
                SELECT aa.activity_label, COUNT(DISTINCT f.frame_id) AS focused_frames
                FROM main_db.frames f
                JOIN main_db.frame_subframe_mapping fsm ON f.frame_id = fsm.frame_id
                JOIN activity_assignments aa ON fsm.sub_frame_id = aa.sub_frame_id
                WHERE f.timestamp >= ? AND f.timestamp <= ?
                  AND f.frame_id LIKE 'frame_%'
                  AND f.focused_app_name IS NOT NULL AND f.focused_app_name != ''
                  AND aa.activity_label IS NOT NULL AND aa.activity_label != ''
                  AND f.focused_app_name = aa.app_name
                GROUP BY aa.activity_label
                ORDER BY focused_frames DESC
                """,
                (start_utc.isoformat(), end_utc.isoformat()),
            )
            breakdown = []
            for row in cursor.fetchall():
                minutes = round(row["focused_frames"] * interval / 60.0, 1)
                breakdown.append({
                    "label": row["activity_label"],
                    "focused_minutes": minutes,
                })

            cursor.execute("DETACH DATABASE main_db")
            return breakdown
        except sqlite3.OperationalError as e:
            logger.warning(f"activity_breakdown query failed: {e}")
            return []
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Window titles
    # ------------------------------------------------------------------

    def get_window_names_in_range(
        self,
        start_utc: datetime,
        end_utc: datetime,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get distinct window names with their app and frame count."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT app_name, window_name, COUNT(*) AS cnt
                FROM sub_frames
                WHERE timestamp >= ? AND timestamp <= ?
                  AND window_name IS NOT NULL AND window_name != ''
                GROUP BY app_name, window_name
                ORDER BY cnt DESC
                LIMIT ?
                """,
                (start_utc.isoformat(), end_utc.isoformat(), limit),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Context snapshot
    # ------------------------------------------------------------------

    def get_context_snapshot(
        self,
        target_utc: datetime,
    ) -> Dict[str, Any]:
        """Return the full context at the moment closest to *target_utc*.

        Finds the nearest frame, its sub_frames, and OCR text.
        """
        conn = self._connect()
        try:
            cursor = conn.cursor()
            ts_iso = target_utc.isoformat()

            cursor.execute(
                """
                SELECT frame_id, timestamp, focused_app_name, focused_window_name,
                       image_path
                FROM frames
                WHERE frame_id LIKE 'frame_%'
                ORDER BY ABS(julianday(timestamp) - julianday(?))
                LIMIT 1
                """,
                (ts_iso,),
            )
            frame = cursor.fetchone()
            if not frame:
                return {"timestamp": ts_iso, "found": False}

            frame_id = frame["frame_id"]

            cursor.execute(
                """
                SELECT sf.sub_frame_id, sf.app_name, sf.window_name,
                       sf.activity_label
                FROM sub_frames sf
                JOIN frame_subframe_mapping fsm ON sf.sub_frame_id = fsm.sub_frame_id
                WHERE fsm.frame_id = ?
                """,
                (frame_id,),
            )
            sub_frames = [dict(r) for r in cursor.fetchall()]

            cursor.execute(
                """
                SELECT ot.text, ot.confidence, sf.app_name
                FROM ocr_text ot
                JOIN sub_frames sf ON ot.sub_frame_id = sf.sub_frame_id
                JOIN frame_subframe_mapping fsm ON sf.sub_frame_id = fsm.sub_frame_id
                WHERE fsm.frame_id = ?
                  AND ot.text_length > 0
                ORDER BY ot.confidence DESC
                LIMIT 5
                """,
                (frame_id,),
            )
            ocr_snippets = [dict(r) for r in cursor.fetchall()]

            return {
                "timestamp": frame["timestamp"],
                "found": True,
                "focused_app": frame["focused_app_name"],
                "focused_window": frame["focused_window_name"],
                "image_path": frame["image_path"],
                "visible_windows": [
                    {"app": sf["app_name"], "window": sf["window_name"]}
                    for sf in sub_frames
                ],
                "activity_label": sub_frames[0]["activity_label"] if sub_frames else None,
                "ocr_snippets": [
                    {"app": s["app_name"], "text": (s["text"] or "")[:300]}
                    for s in ocr_snippets
                ],
            }
        finally:
            conn.close()
