"""
Real-time activity clustering with provisional labeling and candidate promotion.

Online flow:
  - Try to assign to committed clusters first.
  - If no committed hit, call VLM immediately (when configured) to get a
    provisional label and attach/create a candidate cluster.
  - Candidate clusters are promoted to committed clusters once they gather
    enough support.
  - Historical timeline stability is preserved by freezing older committed
    assignments instead of routinely reclustering the full app history.

Database split:
  - All activity data (activity_assignments, activity_clusters, activity_sessions,
    pending_groups) lives in a separate activity DB to avoid write lock contention
    with the main capture pipeline.
  - The main DB (frames, sub_frames, ocr_text) is only accessed read-only by
    this module (for OCR text lookup and image extraction during recalculate).
"""

import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config import config
from utils.logger import setup_logger

logger = setup_logger("activity.cluster_manager")


def _utcnow() -> str:
    """Current UTC time as naive ISO string, matching frames/sub_frames format."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v.astype(np.float32)
    return (v / norm).astype(np.float32)


@dataclass
class PendingEntry:
    """A 'leader' frame awaiting VLM label, plus followers that matched it.
    Now backed by a persistent pending_groups row in the activity DB."""
    group_id: int                          # pending_groups.id (DB-persisted)
    leader_frame_id: str
    app_name: str
    embedding: np.ndarray
    label: Optional[str] = None            # filled once VLM returns
    candidate_id: Optional[int] = None     # filled once candidate cluster created
    # In-memory follower tracking for the current session only.
    # DB followers are resolved via pending_group_id in activity_assignments.
    follower_ids: List[str] = field(default_factory=list)
    follower_embeddings: List[np.ndarray] = field(default_factory=list)
    follower_timestamps: List[Optional[str]] = field(default_factory=list)
    follower_window_names: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)


class ClusterManager:
    """Manage committed/candidate activity clusters for online assignment.

    Uses a separate activity DB for all writes to avoid lock contention with
    the capture pipeline writing to the main DB.
    """

    def __init__(self, activity_db_path: str = None, main_db_path: str = None, threshold: float = None):
        self.activity_db_path = activity_db_path or config.ACTIVITY_DB_PATH
        self.main_db_path = main_db_path or config.OCR_DB_PATH
        self.threshold = threshold or config.CLUSTER_SIMILARITY_THRESHOLD
        self.strong_assignment_threshold = config.CLUSTER_STRONG_ASSIGNMENT_THRESHOLD
        self.candidate_threshold = config.CLUSTER_CANDIDATE_SIMILARITY_THRESHOLD
        self.candidate_promotion_count = config.CLUSTER_CANDIDATE_PROMOTION_COUNT
        self.centroid_alpha = config.CLUSTER_CENTROID_EMA_ALPHA
        self.freeze_window = timedelta(minutes=config.CLUSTER_FREEZE_WINDOW_MINUTES)
        self.pending_sim_threshold = getattr(config, "CLUSTER_PENDING_SIMILARITY_THRESHOLD", 0.80)
        self.pending_ttl = getattr(config, "CLUSTER_PENDING_TTL_SECONDS", 1800)
        # {app_name: [(cluster_id, label, centroid_vec), ...]} committed only
        self._cache: Dict[str, List[Tuple[int, str, np.ndarray]]] = {}
        self._lock = threading.Lock()
        self._recalc_lock = threading.Lock()
        self._recalc_running = False
        # Assignment statistics
        self._total_frames: int = 0
        self._vlm_called_frames: int = 0
        # Pending pool: {app_name: [PendingEntry, ...]}
        self._pending_pool: Dict[str, List[PendingEntry]] = {}
        # VLM circuit breaker
        self._vlm_consecutive_failures: int = 0
        self._vlm_circuit_open_until: float = 0.0  # monotonic time
        self._vlm_circuit_max_failures: int = 3
        self._vlm_circuit_cooldown: float = 60.0  # seconds
        # Rate-limit recalculate to avoid hot loops
        self._last_recalc_time: float = 0.0  # monotonic time
        self._recalc_min_interval: float = 10.0  # seconds between recalcs (normal)
        self._recalc_min_interval_circuit_open: float = 60.0  # seconds when VLM circuit is open
        self._vlm_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="activity_vlm",
        )
        self._load_centroids()
        self._rebuild_pending_pool_from_db()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Connect to the activity DB (for all writes)."""
        conn = sqlite3.connect(self.activity_db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _connect_main_readonly(self) -> sqlite3.Connection:
        """Connect to the main DB (read-only, for OCR text / image extraction)."""
        conn = sqlite3.connect(f"file:{self.main_db_path}?mode=ro", uri=True, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=15000")
        return conn

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _load_centroids(self):
        """Load committed cluster centroids from activity DB into memory."""
        cache: Dict[str, List[Tuple[int, str, np.ndarray]]] = {}
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, app_name, label, centroid FROM activity_clusters "
                "WHERE centroid IS NOT NULL AND cluster_status = 'committed'"
            )
            for row in cursor.fetchall():
                vec = _normalize(np.frombuffer(row["centroid"], dtype=np.float32).copy())
                cache.setdefault(row["app_name"], []).append((row["id"], row["label"], vec))
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to load centroids: {e}")

        with self._lock:
            self._cache = cache

        total = sum(len(v) for v in cache.values())
        logger.info(f"Loaded {total} committed centroids for {len(cache)} apps")

    def _load_centroids_for_apps(self, app_names: Set[str]):
        """Reload committed centroids only for specific apps."""
        if not app_names:
            return
        try:
            conn = self._connect()
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in app_names)
            cursor.execute(
                f"SELECT id, app_name, label, centroid FROM activity_clusters "
                f"WHERE centroid IS NOT NULL AND cluster_status = 'committed' "
                f"AND app_name IN ({placeholders})",
                list(app_names),
            )
            new_entries: Dict[str, List[Tuple[int, str, np.ndarray]]] = {}
            for row in cursor.fetchall():
                vec = _normalize(np.frombuffer(row["centroid"], dtype=np.float32).copy())
                new_entries.setdefault(row["app_name"], []).append((row["id"], row["label"], vec))
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to reload centroids for {app_names}: {e}")
            return

        with self._lock:
            for app in app_names:
                if app in new_entries:
                    self._cache[app] = new_entries[app]
                else:
                    self._cache.pop(app, None)

    # ------------------------------------------------------------------
    # Pending pool persistence
    # ------------------------------------------------------------------

    def _rebuild_pending_pool_from_db(self):
        """Rebuild in-memory pending pool from persistent pending_groups table."""
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, app_name, leader_frame_id, centroid, resolved_label, resolved_cluster_id "
                "FROM pending_groups WHERE resolved_label IS NULL"
            )
            pool: Dict[str, List[PendingEntry]] = {}
            count = 0
            for row in cursor.fetchall():
                centroid_blob = row["centroid"]
                if not centroid_blob:
                    continue
                entry = PendingEntry(
                    group_id=row["id"],
                    leader_frame_id=row["leader_frame_id"],
                    app_name=row["app_name"],
                    embedding=_normalize(np.frombuffer(centroid_blob, dtype=np.float32).copy()),
                    label=row["resolved_label"],
                    candidate_id=row["resolved_cluster_id"],
                )
                pool.setdefault(row["app_name"], []).append(entry)
                count += 1
            conn.close()
            with self._lock:
                self._pending_pool = pool
            if count > 0:
                logger.info(f"Rebuilt pending pool from DB: {count} unresolved groups")
        except Exception as e:
            logger.warning(f"Failed to rebuild pending pool: {e}")

    # ------------------------------------------------------------------
    # Label helpers
    # ------------------------------------------------------------------

    def _get_existing_labels(self, conn: sqlite3.Connection, app_name: str) -> List[str]:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT label FROM activity_clusters "
            "WHERE app_name = ? AND label NOT LIKE ? ORDER BY label",
            (app_name, f"{app_name}_candidate_%"),
        )
        return [row["label"] for row in cursor.fetchall() if row["label"]]

    def _is_vlm_circuit_open(self) -> bool:
        """Check if VLM circuit breaker is open (should skip VLM calls)."""
        if self._vlm_consecutive_failures < self._vlm_circuit_max_failures:
            return False
        now = time.monotonic()
        if now >= self._vlm_circuit_open_until:
            # Cooldown expired, allow one retry (half-open state)
            return False
        return True

    def _vlm_call_succeeded(self):
        """Reset circuit breaker on successful VLM call."""
        self._vlm_consecutive_failures = 0

    def _vlm_call_failed(self):
        """Record VLM failure and potentially open circuit breaker."""
        self._vlm_consecutive_failures += 1
        if self._vlm_consecutive_failures >= self._vlm_circuit_max_failures:
            self._vlm_circuit_open_until = time.monotonic() + self._vlm_circuit_cooldown
            logger.warning(
                f"VLM circuit breaker OPEN: {self._vlm_consecutive_failures} consecutive failures, "
                f"pausing VLM calls for {self._vlm_circuit_cooldown}s"
            )

    def _label_unknown_frame(
        self,
        conn: sqlite3.Connection,
        app_name: str,
        image,
        ocr_text: str,
        layout_text: str = "",
        window_name: str = "",
    ) -> Optional[str]:
        from core.activity.vlm_labeler import (
            call_vlm,
            fuzzy_match_label,
            is_cluster_vlm_endpoint_configured,
            log_cluster_labeling_event,
        )

        if not is_cluster_vlm_endpoint_configured():
            return None
        vlm_url = config.CLUSTER_VLM_URL

        # Circuit breaker: skip VLM if too many recent failures
        if self._is_vlm_circuit_open():
            return None

        existing_labels = self._get_existing_labels(conn, app_name)
        raw_label, meta = call_vlm(
            vlm_url,
            app_name,
            image,
            existing_labels,
            ocr_text,
            layout_text=layout_text,
            window_name=window_name,
            return_meta=True,
        )
        if not raw_label:
            log_cluster_labeling_event(
                {
                    "source": "realtime",
                    "app_name": app_name,
                    "window_name": window_name,
                    "label": "",
                    "status": "failed_empty_label",
                    **meta,
                }
            )
            self._vlm_call_failed()
            return None
        self._vlm_call_succeeded()
        matched = fuzzy_match_label(raw_label, existing_labels)
        final_label = matched or raw_label
        log_cluster_labeling_event(
            {
                "source": "realtime",
                "app_name": app_name,
                "window_name": window_name,
                "label": final_label,
                "raw_label": raw_label,
                "status": "merged_existing" if matched else "new_label",
                "merged_into": matched or "",
                **meta,
            }
        )
        return final_label

    def _schedule_async_vlm_leader(
        self,
        leader_entry: PendingEntry,
        app_name: str,
        frame_id: str,
        embedding: np.ndarray,
        image,
        ocr_text: str,
        layout_text: str,
        window_name: str,
        timestamp: Optional[str],
        _ts_display: str,
        _win: str,
        best_sim: float,
    ) -> None:
        """Run VLM + activity DB finalize off the store_frame hot path."""

        def run():
            try:
                self._async_vlm_leader_worker(
                    leader_entry,
                    app_name,
                    frame_id,
                    embedding,
                    image,
                    ocr_text,
                    layout_text,
                    window_name,
                    timestamp,
                    _ts_display,
                    _win,
                    best_sim,
                )
            except Exception as e:
                logger.warning(f"Async VLM finalize failed for {frame_id}: {e}")

        self._vlm_executor.submit(run)

    def _async_vlm_leader_worker(
        self,
        leader_entry: PendingEntry,
        app_name: str,
        frame_id: str,
        embedding: np.ndarray,
        image,
        ocr_text: str,
        layout_text: str,
        window_name: str,
        timestamp: Optional[str],
        _ts_display: str,
        _win: str,
        best_sim: float,
    ) -> None:
        """VLM call (network) then SQLite updates; does not block store_frame."""
        from core.activity.vlm_labeler import (
            call_vlm,
            fuzzy_match_label,
            is_cluster_vlm_endpoint_configured,
            log_cluster_labeling_event,
        )

        if not is_cluster_vlm_endpoint_configured():
            return
        if self._is_vlm_circuit_open():
            return

        conn = self._connect()
        try:
            existing_labels = self._get_existing_labels(conn, app_name)
        finally:
            conn.close()

        if self._is_vlm_circuit_open():
            return

        vlm_url = config.CLUSTER_VLM_URL
        with self._lock:
            self._vlm_called_frames += 1

        logger.info(
            f"VLM labeling triggered (async): app={app_name}{_win} time={_ts_display} sim={best_sim:.3f}"
        )
        raw_label, meta = call_vlm(
            vlm_url,
            app_name,
            image,
            existing_labels,
            ocr_text,
            layout_text=layout_text,
            window_name=window_name,
            return_meta=True,
        )
        if not raw_label:
            log_cluster_labeling_event(
                {
                    "source": "realtime",
                    "app_name": app_name,
                    "window_name": window_name,
                    "label": "",
                    "status": "failed_empty_label",
                    **meta,
                }
            )
            self._vlm_call_failed()
            return
        self._vlm_call_succeeded()
        matched = fuzzy_match_label(raw_label, existing_labels)
        final_label = matched or raw_label
        log_cluster_labeling_event(
            {
                "source": "realtime",
                "app_name": app_name,
                "window_name": window_name,
                "label": final_label,
                "raw_label": raw_label,
                "status": "merged_existing" if matched else "new_label",
                "merged_into": matched or "",
                **meta,
            }
        )
        provisional_label = final_label
        logger.info(f"VLM label result (async): app={app_name}{_win} -> \"{provisional_label}\"")

        conn = self._connect()
        try:
            candidate_id = self._find_matching_candidate_cluster(
                conn, app_name, provisional_label, embedding
            )
            if candidate_id is None:
                candidate_id = self._create_candidate_cluster(
                    conn, app_name, provisional_label, frame_id, embedding, timestamp
                )

            self._assign_sub_frame(
                conn,
                frame_id=frame_id,
                app_name=app_name,
                timestamp=timestamp,
                cluster_id=candidate_id,
                activity_label=provisional_label,
                provisional_label=provisional_label,
                cluster_status="candidate",
                cluster_updated_at=timestamp,
                pending_group_id=None,
            )

            self._resolve_pending_group(
                conn, leader_entry, provisional_label, candidate_id, timestamp
            )

            cursor = conn.cursor()
            cursor.execute(
                "SELECT cluster_status, label FROM activity_clusters WHERE id = ?",
                (candidate_id,),
            )
            cluster_row = cursor.fetchone()
            session_status = cluster_row["cluster_status"] if cluster_row else "candidate"
            session_label = (
                cluster_row["label"] if cluster_row and cluster_row["label"] else provisional_label
            )
            self._upsert_activity_session(
                conn,
                app_name=app_name,
                cluster_id=candidate_id,
                label=session_label,
                session_status=session_status,
                timestamp=timestamp,
            )
            self._freeze_old_committed_frames(conn, app_name)
            conn.commit()
        except Exception as e:
            logger.warning(f"Async VLM DB finalize failed for {frame_id}: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
            return
        finally:
            try:
                conn.close()
            except Exception:
                pass

        self._load_centroids_for_apps({app_name})
        with self._lock:
            self._cleanup_pending_pool()

    def _label_unknown_frame_with_labels(
        self,
        app_name: str,
        image,
        ocr_text: str,
        existing_labels: List[str],
        layout_text: str = "",
        window_name: str = "",
    ) -> Optional[str]:
        """Like _label_unknown_frame but takes pre-fetched labels (no DB needed)."""
        from core.activity.vlm_labeler import (
            call_vlm,
            fuzzy_match_label,
            is_cluster_vlm_endpoint_configured,
            log_cluster_labeling_event,
        )

        if not is_cluster_vlm_endpoint_configured():
            return None
        vlm_url = config.CLUSTER_VLM_URL
        if self._is_vlm_circuit_open():
            return None

        raw_label, meta = call_vlm(
            vlm_url,
            app_name,
            image,
            existing_labels,
            ocr_text,
            layout_text=layout_text,
            window_name=window_name,
            return_meta=True,
        )
        if not raw_label:
            log_cluster_labeling_event(
                {
                    "source": "recalculate",
                    "app_name": app_name,
                    "window_name": window_name,
                    "label": "",
                    "status": "failed_empty_label",
                    **meta,
                }
            )
            self._vlm_call_failed()
            return None
        self._vlm_call_succeeded()
        matched = fuzzy_match_label(raw_label, existing_labels)
        final_label = matched or raw_label
        log_cluster_labeling_event(
            {
                "source": "recalculate",
                "app_name": app_name,
                "window_name": window_name,
                "label": final_label,
                "raw_label": raw_label,
                "status": "merged_existing" if matched else "new_label",
                "merged_into": matched or "",
                **meta,
            }
        )
        return final_label

    # ------------------------------------------------------------------
    # Activity assignment (writes to activity_assignments in activity DB)
    # ------------------------------------------------------------------

    def _assign_sub_frame(
        self,
        conn: sqlite3.Connection,
        frame_id: str,
        app_name: str,
        timestamp: Optional[str],
        cluster_id: Optional[int],
        activity_label: Optional[str],
        provisional_label: Optional[str],
        cluster_status: str,
        cluster_updated_at: Optional[str] = None,
        pending_group_id: Optional[int] = None,
    ):
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO activity_assignments
            (sub_frame_id, app_name, timestamp, activity_cluster_id, activity_label,
             provisional_label, cluster_status, cluster_updated_at, pending_group_id,
             frozen_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            ON CONFLICT(sub_frame_id) DO UPDATE SET
                activity_cluster_id = excluded.activity_cluster_id,
                activity_label = excluded.activity_label,
                provisional_label = excluded.provisional_label,
                cluster_status = excluded.cluster_status,
                cluster_updated_at = excluded.cluster_updated_at,
                pending_group_id = excluded.pending_group_id,
                frozen_at = CASE
                    WHEN excluded.cluster_status = 'committed'
                         AND (activity_assignments.frozen_at IS NULL OR activity_assignments.frozen_at = '')
                    THEN NULL
                    ELSE activity_assignments.frozen_at
                END
            """,
            (
                frame_id,
                app_name,
                timestamp,
                cluster_id,
                activity_label,
                provisional_label,
                cluster_status,
                cluster_updated_at or _utcnow(),
                pending_group_id,
            ),
        )

    def _update_centroid_ema(
        self,
        conn: sqlite3.Connection,
        cluster_id: int,
        centroid: np.ndarray,
        embedding: np.ndarray,
        timestamp: Optional[str],
    ):
        updated = _normalize((1.0 - self.centroid_alpha) * centroid + self.centroid_alpha * embedding)
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE activity_clusters
            SET centroid = ?, updated_at = ?, last_frame_timestamp = ?
            WHERE id = ?
            """,
            (
                updated.astype(np.float32).tobytes(),
                _utcnow(),
                timestamp or _utcnow(),
                cluster_id,
            ),
        )

    def _find_best_committed_cluster(
        self,
        app_name: str,
        embedding: np.ndarray,
    ) -> Tuple[float, Optional[int], Optional[str], Optional[np.ndarray]]:
        with self._lock:
            clusters = list(self._cache.get(app_name, []))

        best_sim = -1.0
        best_cluster_id = None
        best_label = None
        best_centroid = None
        for cluster_id, label, centroid in clusters:
            sim = _cosine_similarity(embedding, centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster_id = cluster_id
                best_label = label
                best_centroid = centroid
        return best_sim, best_cluster_id, best_label, best_centroid

    def _find_matching_candidate_cluster(
        self,
        conn: sqlite3.Connection,
        app_name: str,
        label: str,
        embedding: np.ndarray,
    ) -> Optional[int]:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, centroid
            FROM activity_clusters
            WHERE app_name = ? AND cluster_status = 'candidate' AND label = ?
            """,
            (app_name, label),
        )
        best_id = None
        best_sim = -1.0
        for row in cursor.fetchall():
            centroid_blob = row["centroid"]
            if not centroid_blob:
                continue
            centroid = np.frombuffer(centroid_blob, dtype=np.float32)
            sim = _cosine_similarity(embedding, centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = row["id"]
        if best_sim >= self.candidate_threshold:
            return best_id
        return None

    def _find_any_candidate_by_similarity(
        self,
        conn: sqlite3.Connection,
        app_name: str,
        embedding: np.ndarray,
    ) -> Tuple[float, Optional[int], Optional[str]]:
        """Find the best matching candidate cluster by embedding similarity alone."""
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, label, centroid
            FROM activity_clusters
            WHERE app_name = ? AND cluster_status = 'candidate' AND centroid IS NOT NULL
            """,
            (app_name,),
        )
        best_sim = -1.0
        best_id = None
        best_label = None
        for row in cursor.fetchall():
            centroid = np.frombuffer(row["centroid"], dtype=np.float32)
            sim = _cosine_similarity(embedding, centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = row["id"]
                best_label = row["label"]
        return best_sim, best_id, best_label

    # ------------------------------------------------------------------
    # Pending group management (persistent)
    # ------------------------------------------------------------------

    def _find_or_create_pending_leader(
        self,
        conn: sqlite3.Connection,
        app_name: str,
        frame_id: str,
        embedding: np.ndarray,
    ) -> Tuple[Optional[PendingEntry], bool]:
        """Atomically find a matching pending leader or create a new one.

        Returns (entry, is_new_leader):
            - (existing_entry, False) if a similar leader was found
            - (new_entry, True) if this frame becomes a new leader
        """
        with self._lock:
            now = time.monotonic()
            entries = self._pending_pool.get(app_name, [])
            best_entry = None
            best_sim = -1.0
            for entry in entries:
                if now - entry.created_at > self.pending_ttl:
                    continue
                sim = _cosine_similarity(embedding, entry.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry
            if best_sim >= self.pending_sim_threshold and best_entry is not None:
                return best_entry, False

        # Also check DB pending_groups (for groups from previous sessions)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, leader_frame_id, centroid FROM pending_groups "
                "WHERE app_name = ? AND resolved_label IS NULL",
                (app_name,),
            )
            for row in cursor.fetchall():
                centroid_blob = row["centroid"]
                if not centroid_blob:
                    continue
                centroid = _normalize(np.frombuffer(centroid_blob, dtype=np.float32).copy())
                sim = _cosine_similarity(embedding, centroid)
                if sim >= self.pending_sim_threshold and sim > best_sim:
                    best_sim = sim
                    # Wrap as PendingEntry and add to in-memory pool
                    best_entry = PendingEntry(
                        group_id=row["id"],
                        leader_frame_id=row["leader_frame_id"],
                        app_name=app_name,
                        embedding=centroid,
                    )
                    with self._lock:
                        self._pending_pool.setdefault(app_name, []).append(best_entry)
            if best_entry is not None and best_sim >= self.pending_sim_threshold:
                return best_entry, False
        except Exception as e:
            logger.debug(f"DB pending group lookup failed: {e}")

        # No match — create a new leader with persistent group
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO pending_groups (app_name, leader_frame_id, centroid, member_count, created_at, updated_at)
            VALUES (?, ?, ?, 1, ?, ?)
            """,
            (app_name, frame_id, _normalize(embedding).tobytes(),
             _utcnow(), _utcnow()),
        )
        group_id = cursor.lastrowid

        new_entry = PendingEntry(
            group_id=group_id,
            leader_frame_id=frame_id,
            app_name=app_name,
            embedding=embedding,
        )
        with self._lock:
            self._pending_pool.setdefault(app_name, []).append(new_entry)
        return new_entry, True

    def _add_follower_to_group(self, conn: sqlite3.Connection, group_id: int, embedding: np.ndarray):
        """Update pending_groups: increment member_count and EMA centroid."""
        cursor = conn.cursor()
        cursor.execute("SELECT centroid, member_count FROM pending_groups WHERE id = ?", (group_id,))
        row = cursor.fetchone()
        if not row:
            return
        old_centroid = np.frombuffer(row["centroid"], dtype=np.float32).copy()
        # EMA with small alpha to keep centroid stable
        alpha = 0.02
        new_centroid = _normalize((1.0 - alpha) * old_centroid + alpha * _normalize(embedding))
        cursor.execute(
            "UPDATE pending_groups SET centroid = ?, member_count = member_count + 1, updated_at = ? WHERE id = ?",
            (new_centroid.tobytes(), _utcnow(), group_id),
        )

    def _resolve_pending_group(
        self,
        conn: sqlite3.Connection,
        entry: PendingEntry,
        label: str,
        candidate_id: int,
        timestamp: Optional[str],
    ):
        """Resolve a pending group: assign label to leader and all DB followers."""
        entry.label = label
        entry.candidate_id = candidate_id

        # Mark group as resolved in DB
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE pending_groups SET resolved_label = ?, resolved_cluster_id = ?, updated_at = ? WHERE id = ?",
            (label, candidate_id, _utcnow(), entry.group_id),
        )

        # Resolve all DB followers (frames with pending_group_id pointing to this group)
        cursor.execute(
            "SELECT sub_frame_id, timestamp FROM activity_assignments "
            "WHERE pending_group_id = ? AND cluster_status = 'pending'",
            (entry.group_id,),
        )
        db_followers = cursor.fetchall()
        if db_followers:
            logger.info(
                f"Pending group resolving: leader={entry.leader_frame_id} app={entry.app_name} "
                f"label='{label}' -> {len(db_followers)} DB followers"
            )
        for frow in db_followers:
            fid = frow["sub_frame_id"]
            fts = frow["timestamp"]
            cursor.execute(
                """
                UPDATE activity_assignments
                SET activity_cluster_id = ?, activity_label = ?, provisional_label = ?,
                    cluster_status = 'candidate', cluster_updated_at = ?, pending_group_id = NULL
                WHERE sub_frame_id = ?
                """,
                (candidate_id, label, label, fts or _utcnow(), fid),
            )
            self._upsert_activity_session(
                conn, app_name=entry.app_name, cluster_id=candidate_id,
                label=label, session_status="candidate", timestamp=fts,
            )

        # Also resolve in-memory followers
        if entry.follower_ids:
            logger.info(
                f"Pending group resolving: leader={entry.leader_frame_id} app={entry.app_name} "
                f"label='{label}' -> {len(entry.follower_ids)} in-memory followers"
            )
        for fid, femb, fts, fwin in zip(
            entry.follower_ids, entry.follower_embeddings,
            entry.follower_timestamps, entry.follower_window_names,
        ):
            self._assign_sub_frame(
                conn, frame_id=fid, app_name=entry.app_name, timestamp=fts,
                cluster_id=candidate_id, activity_label=label,
                provisional_label=label, cluster_status="candidate",
                cluster_updated_at=fts,
            )
            self._upsert_activity_session(
                conn, app_name=entry.app_name, cluster_id=candidate_id,
                label=label, session_status="candidate", timestamp=fts,
            )

        self._refresh_cluster_metadata(conn, candidate_id)
        self._maybe_promote_candidate_cluster(conn, candidate_id)

    def _cleanup_pending_pool(self):
        """Remove expired entries from the pending pool."""
        now = time.monotonic()
        for app_name in list(self._pending_pool.keys()):
            entries = self._pending_pool[app_name]
            self._pending_pool[app_name] = [
                e for e in entries if now - e.created_at <= self.pending_ttl
            ]
            if not self._pending_pool[app_name]:
                del self._pending_pool[app_name]

    def _create_candidate_cluster(
        self,
        conn: sqlite3.Connection,
        app_name: str,
        label: str,
        frame_id: str,
        embedding: np.ndarray,
        timestamp: Optional[str],
    ) -> int:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO activity_clusters
            (app_name, label, centroid, frame_count, representative_frame_ids,
             cluster_status, last_frame_timestamp, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'candidate', ?, ?, ?)
            """,
            (
                app_name,
                label,
                _normalize(embedding).tobytes(),
                0,
                json.dumps([frame_id]),
                timestamp or _utcnow(),
                _utcnow(),
                _utcnow(),
            ),
        )
        return cursor.lastrowid

    def _refresh_cluster_metadata(self, conn: sqlite3.Connection, cluster_id: int):
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT sub_frame_id, cluster_updated_at
            FROM activity_assignments
            WHERE activity_cluster_id = ?
            ORDER BY cluster_updated_at DESC, timestamp DESC
            """,
            (cluster_id,),
        )
        rows = cursor.fetchall()
        frame_ids = [row["sub_frame_id"] for row in rows]
        last_ts = rows[0]["cluster_updated_at"] if rows else None
        cursor.execute(
            """
            UPDATE activity_clusters
            SET frame_count = ?, representative_frame_ids = ?, last_frame_timestamp = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                len(frame_ids),
                json.dumps(frame_ids[:3]),
                last_ts,
                _utcnow(),
                cluster_id,
            ),
        )

    def _upsert_activity_session(
        self,
        conn: sqlite3.Connection,
        app_name: str,
        cluster_id: Optional[int],
        label: Optional[str],
        session_status: str,
        timestamp: Optional[str],
    ):
        if not app_name or cluster_id is None or not label:
            return
        timestamp = timestamp or _utcnow()

        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, cluster_id, label, session_status, start_time, end_time, frame_count
            FROM activity_sessions
            WHERE app_name = ?
            ORDER BY end_time DESC, id DESC
            LIMIT 1
            """,
            (app_name,),
        )
        row = cursor.fetchone()

        if row:
            same_cluster = row["cluster_id"] == cluster_id
            same_label = row["label"] == label
            same_status = (row["session_status"] or "committed") == session_status
            if same_cluster and same_label and same_status and timestamp >= row["end_time"]:
                cursor.execute(
                    """
                    UPDATE activity_sessions
                    SET end_time = ?, frame_count = COALESCE(frame_count, 0) + 1
                    WHERE id = ?
                    """,
                    (timestamp, row["id"]),
                )
                return

        cursor.execute(
            """
            INSERT INTO activity_sessions
            (app_name, cluster_id, label, start_time, end_time, frame_count, session_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (app_name, cluster_id, label, timestamp, timestamp, 1, session_status),
        )

    def _sync_sessions_for_promoted_cluster(
        self,
        conn: sqlite3.Connection,
        cluster_id: int,
        label: str,
    ):
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE activity_sessions
            SET label = ?, session_status = 'committed'
            WHERE cluster_id = ?
            """,
            (label, cluster_id),
        )

    def _maybe_promote_candidate_cluster(self, conn: sqlite3.Connection, cluster_id: int):
        cursor = conn.cursor()
        cursor.execute(
            "SELECT app_name, label FROM activity_clusters WHERE id = ? AND cluster_status = 'candidate'",
            (cluster_id,),
        )
        row = cursor.fetchone()
        if not row:
            return
        cursor.execute(
            "SELECT COUNT(*) AS cnt FROM activity_assignments WHERE activity_cluster_id = ?",
            (cluster_id,),
        )
        count = cursor.fetchone()["cnt"]
        if count < self.candidate_promotion_count:
            return

        now = _utcnow()
        cursor.execute(
            """
            UPDATE activity_clusters
            SET cluster_status = 'committed',
                committed_at = COALESCE(committed_at, ?),
                frame_count = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (now, count, now, cluster_id),
        )
        cursor.execute(
            """
            UPDATE activity_assignments
            SET cluster_status = 'committed',
                activity_label = COALESCE(activity_label, provisional_label),
                provisional_label = COALESCE(provisional_label, activity_label),
                cluster_updated_at = ?
            WHERE activity_cluster_id = ?
            """,
            (now, cluster_id),
        )
        self._sync_sessions_for_promoted_cluster(conn, cluster_id, row["label"])
        logger.info(f"Promoted candidate cluster {cluster_id} ({row['app_name']}/{row['label']}) to committed")

    def _freeze_old_committed_frames(self, conn: sqlite3.Connection, app_name: str):
        freeze_before = (datetime.now(timezone.utc).replace(tzinfo=None) - self.freeze_window).isoformat()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE activity_assignments
            SET frozen_at = COALESCE(frozen_at, ?)
            WHERE app_name = ?
              AND cluster_status = 'committed'
              AND timestamp <= ?
              AND (frozen_at IS NULL OR frozen_at = '')
            """,
            (_utcnow(), app_name, freeze_before),
        )

    def _load_embeddings_for_apps(
        self,
        lancedb_table,
        app_names: Set[str],
    ) -> Dict[str, np.ndarray]:
        df = lancedb_table.to_pandas()
        sub_df = df[
            df["frame_id"].str.startswith("subframe_")
            & df["app_name"].isin(app_names)
        ]
        embeddings_map: Dict[str, np.ndarray] = {}
        for _, row in sub_df.iterrows():
            vec = _normalize(np.array(row["vector"], dtype=np.float32))
            if np.linalg.norm(vec) < 1e-6:
                continue
            embeddings_map[row["frame_id"]] = vec
        return embeddings_map

    # ------------------------------------------------------------------
    # Real-time layer
    # ------------------------------------------------------------------

    def assign_frame(
        self,
        app_name: str,
        frame_id: str,
        embedding: np.ndarray,
        image=None,
        ocr_text: str = "",
        layout_text: str = "",
        timestamp: Optional[str] = None,
        window_name: str = "",
    ) -> Optional[str]:
        """Assign a sub_frame using 4-layer dedup: committed → candidate → pending pool → VLM."""
        if not app_name:
            return None

        embedding = _normalize(np.asarray(embedding, dtype=np.float32))
        _ts_display = timestamp[:19].replace("T", " ") if timestamp else "unknown"
        _win = f"/{window_name}" if window_name else ""

        with self._lock:
            self._total_frames += 1

        # --- Layer 1: committed clusters (in-memory cache) ---
        best_sim, best_cluster_id, best_label, best_centroid = self._find_best_committed_cluster(app_name, embedding)

        try:
            conn = self._connect()

            if best_sim >= self.threshold and best_cluster_id is not None and best_label is not None:
                self._assign_sub_frame(
                    conn, frame_id=frame_id, app_name=app_name, timestamp=timestamp,
                    cluster_id=best_cluster_id, activity_label=best_label,
                    provisional_label=best_label, cluster_status="committed",
                    cluster_updated_at=timestamp,
                )
                self._upsert_activity_session(
                    conn, app_name=app_name, cluster_id=best_cluster_id,
                    label=best_label, session_status="committed", timestamp=timestamp,
                )
                if best_centroid is not None and best_sim >= self.strong_assignment_threshold:
                    self._update_centroid_ema(conn, best_cluster_id, best_centroid, embedding, timestamp)
                self._freeze_old_committed_frames(conn, app_name)
                conn.commit()
                conn.close()
                return best_label

            # --- Layer 2: candidate clusters (DB, similarity-only) ---
            cand_sim, cand_id, cand_label = self._find_any_candidate_by_similarity(conn, app_name, embedding)
            if cand_sim >= self.candidate_threshold and cand_id is not None and cand_label is not None:
                logger.info(
                    f"Candidate cluster hit: app={app_name}{_win} "
                    f"cluster={cand_id} sim={cand_sim:.3f} label='{cand_label}'"
                )
                self._assign_sub_frame(
                    conn, frame_id=frame_id, app_name=app_name, timestamp=timestamp,
                    cluster_id=cand_id, activity_label=cand_label,
                    provisional_label=cand_label, cluster_status="candidate",
                    cluster_updated_at=timestamp,
                )
                self._refresh_cluster_metadata(conn, cand_id)
                self._maybe_promote_candidate_cluster(conn, cand_id)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT cluster_status, label FROM activity_clusters WHERE id = ?",
                    (cand_id,),
                )
                cluster_row = cursor.fetchone()
                session_status = cluster_row["cluster_status"] if cluster_row else "candidate"
                session_label = cluster_row["label"] if cluster_row and cluster_row["label"] else cand_label
                self._upsert_activity_session(
                    conn, app_name=app_name, cluster_id=cand_id,
                    label=session_label, session_status=session_status, timestamp=timestamp,
                )
                self._freeze_old_committed_frames(conn, app_name)
                conn.commit()
                conn.close()
                self._load_centroids_for_apps({app_name})
                return cand_label

            # --- Layer 3+4: pending pool (persistent) + VLM ---
            pending_entry, is_new_leader = self._find_or_create_pending_leader(
                conn, app_name, frame_id, embedding,
            )

            if not is_new_leader and pending_entry is not None:
                if pending_entry.label is not None and pending_entry.candidate_id is not None:
                    # Leader already resolved — use its label directly
                    logger.info(
                        f"Pending pool hit (resolved): app={app_name}{_win} "
                        f"frame={frame_id} -> label='{pending_entry.label}'"
                    )
                    self._assign_sub_frame(
                        conn, frame_id=frame_id, app_name=app_name, timestamp=timestamp,
                        cluster_id=pending_entry.candidate_id, activity_label=pending_entry.label,
                        provisional_label=pending_entry.label, cluster_status="candidate",
                        cluster_updated_at=timestamp,
                    )
                    self._refresh_cluster_metadata(conn, pending_entry.candidate_id)
                    self._maybe_promote_candidate_cluster(conn, pending_entry.candidate_id)
                    self._upsert_activity_session(
                        conn, app_name=app_name, cluster_id=pending_entry.candidate_id,
                        label=pending_entry.label, session_status="candidate", timestamp=timestamp,
                    )
                    conn.commit()
                    conn.close()
                    self._load_centroids_for_apps({app_name})
                    return pending_entry.label
                else:
                    # Leader still waiting for VLM — become a follower
                    logger.info(
                        f"Pending pool follower: app={app_name}{_win} "
                        f"frame={frame_id} waiting on leader={pending_entry.leader_frame_id}"
                    )
                    with self._lock:
                        pending_entry.follower_ids.append(frame_id)
                        pending_entry.follower_embeddings.append(embedding)
                        pending_entry.follower_timestamps.append(timestamp)
                        pending_entry.follower_window_names.append(window_name)
                    # Mark as pending in activity DB with group link
                    self._assign_sub_frame(
                        conn, frame_id=frame_id, app_name=app_name, timestamp=timestamp,
                        cluster_id=None, activity_label=None,
                        provisional_label=None, cluster_status="pending",
                        cluster_updated_at=timestamp, pending_group_id=pending_entry.group_id,
                    )
                    self._add_follower_to_group(conn, pending_entry.group_id, embedding)
                    conn.commit()
                    conn.close()
                    return None

            # This frame is a new leader — call VLM (or defer if circuit is open)
            leader_entry = pending_entry
            # Mark leader as pending in DB with group link
            self._assign_sub_frame(
                conn, frame_id=frame_id, app_name=app_name, timestamp=timestamp,
                cluster_id=None, activity_label=None,
                provisional_label=None, cluster_status="pending",
                cluster_updated_at=timestamp, pending_group_id=leader_entry.group_id,
            )

            # If VLM circuit is open, just persist as pending — recalculate will handle it later
            if self._is_vlm_circuit_open():
                conn.commit()
                conn.close()
                return None

            from core.activity.vlm_labeler import is_cluster_vlm_endpoint_configured

            if not is_cluster_vlm_endpoint_configured():
                conn.commit()
                conn.close()
                return None

            conn.commit()
            conn.close()

            img_copy = image.copy() if image is not None else None
            emb_copy = np.copy(embedding)
            logger.info(
                f"VLM labeling scheduled (async): app={app_name}{_win} frame={frame_id} "
                f"time={_ts_display} sim={best_sim:.3f}"
            )
            self._schedule_async_vlm_leader(
                leader_entry=leader_entry,
                app_name=app_name,
                frame_id=frame_id,
                embedding=emb_copy,
                image=img_copy,
                ocr_text=ocr_text or "",
                layout_text=layout_text or "",
                window_name=window_name or "",
                timestamp=timestamp,
                _ts_display=_ts_display,
                _win=_win,
                best_sim=best_sim,
            )
            return None
        except Exception as e:
            logger.warning(f"Cluster assignment failed for {frame_id}: {e}")
            return None

    # ------------------------------------------------------------------
    # Periodic layer
    # ------------------------------------------------------------------

    def _get_non_committed_counts_from_db(self) -> Dict[str, Dict[str, int]]:
        counts: Dict[str, Dict[str, int]] = {}
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT app_name, cluster_status, COUNT(*) AS pending_count
                FROM activity_assignments
                WHERE app_name IS NOT NULL AND app_name != ''
                  AND cluster_status IN ('pending', 'candidate')
                GROUP BY app_name, cluster_status
                """
            )
            for row in cursor.fetchall():
                app_counts = counts.setdefault(row["app_name"], {"pending": 0, "candidate": 0})
                app_counts[row["cluster_status"]] = int(row["pending_count"])
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to load non-committed counts: {e}")
        return counts

    def _get_pending_group_counts(self) -> Dict[str, int]:
        """Return number of unresolved pending groups per app."""
        counts: Dict[str, int] = {}
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT app_name, COUNT(*) AS cnt FROM pending_groups "
                "WHERE resolved_label IS NULL GROUP BY app_name"
            )
            for row in cursor.fetchall():
                counts[row["app_name"]] = row["cnt"]
            conn.close()
        except Exception:
            pass
        return counts

    def _has_ungrouped_pending_frames(self, apps: Set[str]) -> bool:
        """Check if there are pending frames not yet assigned to a group."""
        try:
            conn = self._connect()
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in apps)
            cursor.execute(
                f"SELECT COUNT(*) AS cnt FROM activity_assignments "
                f"WHERE cluster_status = 'pending' AND pending_group_id IS NULL "
                f"AND app_name IN ({placeholders})",
                list(apps),
            )
            count = cursor.fetchone()["cnt"]
            conn.close()
            return count > 0
        except Exception:
            return False

    def get_apps_needing_recalc(self) -> Set[str]:
        trigger = config.CLUSTER_UNCLASSIFIED_TRIGGER
        counts = self._get_non_committed_counts_from_db()
        return {
            app for app, per_status in counts.items()
            if per_status.get("pending", 0) >= trigger or per_status.get("candidate", 0) >= trigger
        }

    def get_debug_status(self, app_name: Optional[str] = None) -> Dict:
        counts = self._get_non_committed_counts_from_db()
        group_counts = self._get_pending_group_counts()
        trigger = config.CLUSTER_UNCLASSIFIED_TRIGGER

        with self._lock:
            committed_cache_counts = {app: len(clusters) for app, clusters in self._cache.items()}

        app_names = set(counts.keys()) | set(committed_cache_counts.keys())
        if app_name:
            app_names = {app_name}

        apps = []
        for name in sorted(app_names):
            per_status = counts.get(name, {"pending": 0, "candidate": 0})
            pending_groups = group_counts.get(name, 0)
            apps.append({
                "app_name": name,
                "pending_unclassified": per_status.get("pending", 0),
                "pending_groups": pending_groups,
                "vlm_calls_needed": pending_groups,
                "candidate_frames": per_status.get("candidate", 0),
                "committed_cluster_count": committed_cache_counts.get(name, 0),
                "threshold": trigger,
                "needs_recalc": (
                    per_status.get("pending", 0) >= trigger or per_status.get("candidate", 0) >= trigger
                ),
            })

        return {
            "threshold": trigger,
            "recalc_running": self._recalc_running,
            "apps_needing_recalc": sorted(app["app_name"] for app in apps if app["needs_recalc"]),
            "apps": apps,
        }

    def get_assignment_stats(self) -> Dict:
        """Return runtime frame assignment statistics."""
        with self._lock:
            total = self._total_frames
            vlm_called = self._vlm_called_frames
        ratio = vlm_called / total if total > 0 else 0.0
        return {
            "total_frames": total,
            "vlm_called_frames": vlm_called,
            "vlm_call_ratio": ratio,
        }

    def reset_assignment_stats(self):
        """Reset counters for a new recording session."""
        with self._lock:
            self._total_frames = 0
            self._vlm_called_frames = 0

    def should_recalculate(self) -> bool:
        # Use lock to prevent concurrent recalculate
        if not self._recalc_lock.acquire(blocking=False):
            return False
        try:
            if self._recalc_running:
                return False
            # Rate-limit: enforce minimum interval between recalculate runs
            now = time.monotonic()
            circuit_open = self._is_vlm_circuit_open()
            min_interval = self._recalc_min_interval_circuit_open if circuit_open else self._recalc_min_interval
            if now - self._last_recalc_time < min_interval:
                return False
            apps = self.get_apps_needing_recalc()
            if not apps:
                return False
            if circuit_open:
                # Only recalc if there are truly ungrouped pending frames to group
                return self._has_ungrouped_pending_frames(apps)
            return True
        finally:
            self._recalc_lock.release()

    def recalculate(self, lancedb_table):
        with self._recalc_lock:
            if self._recalc_running:
                return
            self._recalc_running = True
            self._last_recalc_time = time.monotonic()
        try:
            target_apps = self.get_apps_needing_recalc()
            if not target_apps:
                return
            self._do_recalculate(lancedb_table, target_apps)
        except Exception as e:
            logger.error(f"Cluster recalculation failed: {e}", exc_info=True)
        finally:
            with self._recalc_lock:
                self._recalc_running = False

    def _do_recalculate(self, lancedb_table, target_apps: Set[str]):
        conn = self._connect()
        cursor = conn.cursor()
        affected_apps: Set[str] = set()

        # --- Phase 0: Expire stale pending frames ---
        pending_ttl_iso = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=self.pending_ttl)).isoformat()
        cursor.execute(
            f"""
            UPDATE activity_assignments
            SET cluster_status = 'expired'
            WHERE cluster_status = 'pending'
              AND app_name IN ({",".join("?" for _ in target_apps)})
              AND cluster_updated_at < ?
            """,
            list(target_apps) + [pending_ttl_iso],
        )
        expired_count = cursor.rowcount
        if expired_count > 0:
            conn.commit()
            logger.info(f"Recalculate: expired {expired_count} stale pending frames (older than {self.pending_ttl}s)")

        # --- Phase 0.5: Group ungrouped pending frames ---
        cursor.execute(
            f"""
            SELECT sub_frame_id, app_name
            FROM activity_assignments
            WHERE cluster_status = 'pending' AND pending_group_id IS NULL
              AND app_name IN ({",".join("?" for _ in target_apps)})
            ORDER BY timestamp
            """,
            list(target_apps),
        )
        ungrouped = cursor.fetchall()
        if ungrouped:
            embeddings_map = self._load_embeddings_for_apps(lancedb_table, target_apps)
            grouped_count = 0
            new_groups_count = 0
            for row in ungrouped:
                frame_id = row["sub_frame_id"]
                frame_app = row["app_name"]
                embedding = embeddings_map.get(frame_id)
                if embedding is None:
                    continue

                # Try to match existing unresolved pending group
                matched_group_id = None
                best_sim = -1.0
                cursor.execute(
                    "SELECT id, centroid FROM pending_groups "
                    "WHERE app_name = ? AND resolved_label IS NULL",
                    (frame_app,),
                )
                for grow in cursor.fetchall():
                    centroid = np.frombuffer(grow["centroid"], dtype=np.float32).copy()
                    sim = _cosine_similarity(embedding, _normalize(centroid))
                    if sim >= self.pending_sim_threshold and sim > best_sim:
                        best_sim = sim
                        matched_group_id = grow["id"]

                if matched_group_id is not None:
                    cursor.execute(
                        "UPDATE activity_assignments SET pending_group_id = ? WHERE sub_frame_id = ?",
                        (matched_group_id, frame_id),
                    )
                    self._add_follower_to_group(conn, matched_group_id, embedding)
                    grouped_count += 1
                else:
                    # Create new group
                    cursor.execute(
                        "INSERT INTO pending_groups (app_name, leader_frame_id, centroid, member_count, created_at, updated_at) "
                        "VALUES (?, ?, ?, 1, ?, ?)",
                        (frame_app, frame_id, _normalize(embedding).tobytes(),
                         _utcnow(), _utcnow()),
                    )
                    new_group_id = cursor.lastrowid
                    cursor.execute(
                        "UPDATE activity_assignments SET pending_group_id = ? WHERE sub_frame_id = ?",
                        (new_group_id, frame_id),
                    )
                    new_groups_count += 1

            conn.commit()
            if grouped_count > 0 or new_groups_count > 0:
                logger.info(
                    f"Recalculate: grouped {grouped_count} orphan frames into existing groups, "
                    f"created {new_groups_count} new groups"
                )

        # Early exit if VLM circuit breaker is open — grouping is done, VLM labeling deferred
        if self._is_vlm_circuit_open():
            # Log group-level stats instead of individual frame counts
            cursor.execute(
                f"SELECT COUNT(*) AS cnt FROM pending_groups WHERE resolved_label IS NULL "
                f"AND app_name IN ({','.join('?' for _ in target_apps)})",
                list(target_apps),
            )
            group_count = cursor.fetchone()["cnt"]
            cursor.execute(
                f"SELECT COUNT(*) AS cnt FROM activity_assignments WHERE cluster_status = 'pending' "
                f"AND app_name IN ({','.join('?' for _ in target_apps)})",
                list(target_apps),
            )
            frame_count = cursor.fetchone()["cnt"]
            if frame_count > 0:
                logger.info(
                    f"Recalculate: {frame_count} pending frames in {group_count} groups "
                    f"({group_count} VLM calls needed) — VLM circuit open, deferring"
                )
            conn.close()
            return

        # --- Phase 1: VLM label unresolved groups (not individual frames) ---
        # Close the long-held connection from Phase 0/0.5 before heavy work.
        # Each VLM call takes seconds; holding a write lock blocks assign_frame.
        conn.close()

        embeddings_map = self._load_embeddings_for_apps(lancedb_table, target_apps)

        # Pre-fetch unresolved groups with a short-lived connection
        tmp_conn = self._connect()
        tmp_cursor = tmp_conn.cursor()
        tmp_cursor.execute(
            f"""
            SELECT id, app_name, leader_frame_id
            FROM pending_groups
            WHERE resolved_label IS NULL
              AND app_name IN ({",".join("?" for _ in target_apps)})
            ORDER BY created_at
            """,
            list(target_apps),
        )
        unresolved_groups = [dict(row) for row in tmp_cursor.fetchall()]

        # Pre-fetch existing labels per app (so VLM calls don't need DB)
        app_existing_labels: Dict[str, List[str]] = {}
        for app in target_apps:
            app_existing_labels[app] = self._get_existing_labels(tmp_conn, app)

        if unresolved_groups:
            tmp_cursor.execute(
                f"SELECT COUNT(*) AS cnt FROM activity_assignments WHERE cluster_status = 'pending' "
                f"AND app_name IN ({','.join('?' for _ in target_apps)})",
                list(target_apps),
            )
            total_pending = tmp_cursor.fetchone()["cnt"]
            logger.info(
                f"Recalculate: {total_pending} pending frames in {len(unresolved_groups)} groups "
                f"to process for apps {sorted(target_apps)}"
            )
        tmp_conn.close()

        vlm_failures_in_batch = 0
        max_vlm_failures_in_batch = 5
        processed_count = 0
        sqlite_db = None

        for grow in unresolved_groups:
            if vlm_failures_in_batch >= max_vlm_failures_in_batch or self._is_vlm_circuit_open():
                logger.info(
                    f"Recalculate: stopping early — VLM circuit open or "
                    f"{vlm_failures_in_batch} failures in this batch "
                    f"(processed {processed_count}/{len(unresolved_groups)} groups)"
                )
                break

            group_id = grow["id"]
            group_app = grow["app_name"]
            leader_id = grow["leader_frame_id"]

            embedding = embeddings_map.get(leader_id)
            if embedding is None:
                continue

            # Extract image from main DB (read-only)
            if sqlite_db is None:
                from core.storage.sqlite_storage import SQLiteStorage
                sqlite_db = SQLiteStorage(db_path=self.main_db_path, activity_db_path=self.activity_db_path)
            image = sqlite_db.extract_sub_frame_image(leader_id)

            # Get OCR text + optional OCR region layout from main DB (read-only)
            ocr_text = ""
            layout_text = ""
            window_name = ""
            try:
                from core.activity.vlm_labeler import build_region_layout_text
                main_conn = self._connect_main_readonly()
                oc = main_conn.cursor()
                oc.execute(
                    "SELECT text FROM ocr_text WHERE sub_frame_id = ? ORDER BY text_length DESC LIMIT 1",
                    (leader_id,),
                )
                r = oc.fetchone()
                if r:
                    ocr_text = r["text"]
                oc.execute(
                    "SELECT window_name FROM sub_frames WHERE sub_frame_id = ? LIMIT 1",
                    (leader_id,),
                )
                win_row = oc.fetchone()
                if win_row and win_row["window_name"]:
                    window_name = win_row["window_name"]
                if config.CLUSTER_VLM_INCLUDE_REGION_LAYOUT:
                    oc.execute(
                        """
                        SELECT bbox_x1, bbox_y1, bbox_x2, bbox_y2, text, image_width, image_height
                        FROM ocr_regions
                        WHERE sub_frame_id = ? AND text IS NOT NULL AND text != ''
                        ORDER BY bbox_y1 ASC, bbox_x1 ASC
                        LIMIT 80
                        """,
                        (leader_id,),
                    )
                    region_rows = oc.fetchall()
                    if region_rows:
                        regions = [
                            {
                                "bbox": [rr["bbox_x1"], rr["bbox_y1"], rr["bbox_x2"], rr["bbox_y2"]],
                                "text": rr["text"],
                                "image_width": rr["image_width"],
                                "image_height": rr["image_height"],
                            }
                            for rr in region_rows
                        ]
                        layout_text = build_region_layout_text(regions)
                main_conn.close()
            except Exception:
                pass

            # --- VLM call: NO activity DB connection held here ---
            existing_labels = app_existing_labels.get(group_app, [])
            provisional = self._label_unknown_frame_with_labels(
                group_app,
                image,
                ocr_text,
                existing_labels,
                layout_text=layout_text,
                window_name=window_name,
            )
            if not provisional:
                vlm_failures_in_batch += 1
                continue

            # Reacquire connection for short DB writes, then release
            conn = self._connect()
            try:
                candidate_id = self._find_matching_candidate_cluster(conn, group_app, provisional, embedding)
                if candidate_id is None:
                    candidate_id = self._create_candidate_cluster(
                        conn, group_app, provisional, leader_id, embedding, None,
                    )

                # Assign leader
                self._assign_sub_frame(
                    conn, frame_id=leader_id, app_name=group_app, timestamp=None,
                    cluster_id=candidate_id, activity_label=provisional,
                    provisional_label=provisional, cluster_status="candidate",
                    pending_group_id=None,
                )

                # Resolve all group members
                leader_entry = PendingEntry(
                    group_id=group_id, leader_frame_id=leader_id,
                    app_name=group_app, embedding=embedding,
                )
                self._resolve_pending_group(conn, leader_entry, provisional, candidate_id, None)

                # Get timestamp for session
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT timestamp FROM activity_assignments WHERE sub_frame_id = ?",
                    (leader_id,),
                )
                ts_row = cursor.fetchone()
                self._upsert_activity_session(
                    conn, app_name=group_app, cluster_id=candidate_id,
                    label=provisional, session_status="candidate",
                    timestamp=ts_row["timestamp"] if ts_row else None,
                )
                affected_apps.add(group_app)
                processed_count += 1

                # Update cached labels so next VLM call sees them
                if provisional not in existing_labels:
                    app_existing_labels.setdefault(group_app, []).append(provisional)

                conn.commit()
            finally:
                conn.close()

        # 2. Promote candidate clusters (short transaction)
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, app_name
                FROM activity_clusters
                WHERE cluster_status = 'candidate'
                  AND app_name IN ({",".join("?" for _ in target_apps)})
                """,
                list(target_apps),
            )
            for row in cursor.fetchall():
                self._refresh_cluster_metadata(conn, row["id"])
                self._maybe_promote_candidate_cluster(conn, row["id"])
                self._freeze_old_committed_frames(conn, row["app_name"])
                affected_apps.add(row["app_name"])

            conn.commit()
        finally:
            conn.close()

        self._load_centroids_for_apps(affected_apps or target_apps)
        # Rebuild pending pool after recalculate
        self._rebuild_pending_pool_from_db()
