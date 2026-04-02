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
"""

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config import config
from utils.logger import setup_logger

logger = setup_logger("activity.cluster_manager")


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
    """A 'leader' frame awaiting VLM label, plus followers that matched it."""
    leader_frame_id: str
    app_name: str
    embedding: np.ndarray
    label: Optional[str] = None          # filled once VLM returns
    candidate_id: Optional[int] = None   # filled once candidate cluster created
    follower_ids: List[str] = field(default_factory=list)
    follower_embeddings: List[np.ndarray] = field(default_factory=list)
    follower_timestamps: List[Optional[str]] = field(default_factory=list)
    follower_window_names: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)


class ClusterManager:
    """Manage committed/candidate activity clusters for online assignment."""

    def __init__(self, db_path: str, threshold: float = None):
        self.db_path = db_path
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
        self._load_centroids()

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _load_centroids(self):
        """Load committed cluster centroids from SQLite into memory."""
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
    ) -> Optional[str]:
        from core.activity.vlm_labeler import call_vlm, fuzzy_match_label

        vlm_url = config.CLUSTER_VLM_URL
        if not vlm_url or image is None:
            return None

        # Circuit breaker: skip VLM if too many recent failures
        if self._is_vlm_circuit_open():
            return None

        existing_labels = self._get_existing_labels(conn, app_name)
        label = call_vlm(vlm_url, app_name, image, existing_labels, ocr_text)
        if not label:
            self._vlm_call_failed()
            return None
        self._vlm_call_succeeded()
        return fuzzy_match_label(label, existing_labels) or label

    # ------------------------------------------------------------------
    # Cluster updates
    # ------------------------------------------------------------------

    def _assign_sub_frame(
        self,
        conn: sqlite3.Connection,
        frame_id: str,
        cluster_id: Optional[int],
        activity_label: Optional[str],
        provisional_label: Optional[str],
        cluster_status: str,
        cluster_updated_at: Optional[str] = None,
    ):
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE sub_frames
            SET activity_cluster_id = ?,
                activity_label = ?,
                provisional_label = ?,
                cluster_status = ?,
                cluster_updated_at = ?,
                frozen_at = CASE
                    WHEN cluster_status = 'committed' AND (frozen_at IS NULL OR frozen_at = '') THEN NULL
                    ELSE frozen_at
                END
            WHERE sub_frame_id = ?
            """,
            (
                cluster_id,
                activity_label,
                provisional_label,
                cluster_status,
                cluster_updated_at or datetime.now().isoformat(),
                frame_id,
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
                datetime.now().isoformat(),
                timestamp or datetime.now().isoformat(),
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
        """Find the best matching candidate cluster by embedding similarity alone (no label match required)."""
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

    def _find_or_create_pending_leader(
        self,
        app_name: str,
        frame_id: str,
        embedding: np.ndarray,
    ) -> Tuple[Optional[PendingEntry], bool]:
        """Atomically find a matching pending leader or create a new one.

        Returns (entry, is_new_leader):
            - (existing_entry, False) if a similar leader was found
            - (new_entry, True) if this frame becomes a new leader
            - (None, False) if pending pool is not applicable
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
            # No match — atomically create a new leader
            new_entry = PendingEntry(
                leader_frame_id=frame_id,
                app_name=app_name,
                embedding=embedding,
            )
            self._pending_pool.setdefault(app_name, []).append(new_entry)
            return new_entry, True

    def _resolve_pending_followers(
        self,
        conn: sqlite3.Connection,
        entry: PendingEntry,
        label: str,
        candidate_id: int,
        timestamp: Optional[str],
    ):
        """Assign all followers of a resolved leader to the same candidate cluster."""
        entry.label = label
        entry.candidate_id = candidate_id
        if entry.follower_ids:
            logger.info(
                f"Pending pool resolving: leader={entry.leader_frame_id} app={entry.app_name} "
                f"label='{label}' -> {len(entry.follower_ids)} followers"
            )
        for fid, femb, fts, fwin in zip(
            entry.follower_ids,
            entry.follower_embeddings,
            entry.follower_timestamps,
            entry.follower_window_names,
        ):
            self._assign_sub_frame(
                conn,
                frame_id=fid,
                cluster_id=candidate_id,
                activity_label=label,
                provisional_label=label,
                cluster_status="candidate",
                cluster_updated_at=fts,
            )
            self._upsert_activity_session(
                conn,
                app_name=entry.app_name,
                cluster_id=candidate_id,
                label=label,
                session_status="candidate",
                timestamp=fts,
            )
            logger.debug(f"Pending follower {fid} assigned to cluster {candidate_id} label='{label}'")
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
                timestamp or datetime.now().isoformat(),
                datetime.now().isoformat(),
                datetime.now().isoformat(),
            ),
        )
        return cursor.lastrowid

    def _refresh_cluster_metadata(self, conn: sqlite3.Connection, cluster_id: int):
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT sub_frame_id, cluster_updated_at
            FROM sub_frames
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
                datetime.now().isoformat(),
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
        timestamp = timestamp or datetime.now().isoformat()

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
            "SELECT COUNT(*) AS cnt FROM sub_frames WHERE activity_cluster_id = ?",
            (cluster_id,),
        )
        count = cursor.fetchone()["cnt"]
        if count < self.candidate_promotion_count:
            return

        now = datetime.now().isoformat()
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
            UPDATE sub_frames
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
        freeze_before = (datetime.now() - self.freeze_window).isoformat()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE sub_frames
            SET frozen_at = COALESCE(frozen_at, ?)
            WHERE app_name = ?
              AND cluster_status = 'committed'
              AND timestamp <= ?
              AND (frozen_at IS NULL OR frozen_at = '')
            """,
            (datetime.now().isoformat(), app_name, freeze_before),
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
                    conn,
                    frame_id=frame_id,
                    cluster_id=best_cluster_id,
                    activity_label=best_label,
                    provisional_label=best_label,
                    cluster_status="committed",
                    cluster_updated_at=timestamp,
                )
                self._upsert_activity_session(
                    conn,
                    app_name=app_name,
                    cluster_id=best_cluster_id,
                    label=best_label,
                    session_status="committed",
                    timestamp=timestamp,
                )
                if best_centroid is not None and best_sim >= self.strong_assignment_threshold:
                    self._update_centroid_ema(conn, best_cluster_id, best_centroid, embedding, timestamp)
                self._freeze_old_committed_frames(conn, app_name)
                conn.commit()
                conn.close()
                return best_label

            # --- Layer 2: candidate clusters (DB, similarity-only, no label match required) ---
            cand_sim, cand_id, cand_label = self._find_any_candidate_by_similarity(conn, app_name, embedding)
            if cand_sim >= self.candidate_threshold and cand_id is not None and cand_label is not None:
                logger.info(
                    f"Candidate cluster hit: app={app_name}{_win} "
                    f"cluster={cand_id} sim={cand_sim:.3f} label='{cand_label}'"
                )
                self._assign_sub_frame(
                    conn,
                    frame_id=frame_id,
                    cluster_id=cand_id,
                    activity_label=cand_label,
                    provisional_label=cand_label,
                    cluster_status="candidate",
                    cluster_updated_at=timestamp,
                )
                self._refresh_cluster_metadata(conn, cand_id)
                self._maybe_promote_candidate_cluster(conn, cand_id)
                # Re-read status in case promotion happened
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT cluster_status, label FROM activity_clusters WHERE id = ?",
                    (cand_id,),
                )
                cluster_row = cursor.fetchone()
                session_status = cluster_row["cluster_status"] if cluster_row else "candidate"
                session_label = cluster_row["label"] if cluster_row and cluster_row["label"] else cand_label
                self._upsert_activity_session(
                    conn,
                    app_name=app_name,
                    cluster_id=cand_id,
                    label=session_label,
                    session_status=session_status,
                    timestamp=timestamp,
                )
                self._freeze_old_committed_frames(conn, app_name)
                conn.commit()
                conn.close()
                self._load_centroids_for_apps({app_name})
                return cand_label

            # --- Layer 3+4: pending pool (atomic find-or-create) + VLM ---
            pending_entry, is_new_leader = self._find_or_create_pending_leader(
                app_name, frame_id, embedding,
            )

            if not is_new_leader and pending_entry is not None:
                # Matched an existing leader
                if pending_entry.label is not None and pending_entry.candidate_id is not None:
                    # Leader already resolved — use its label directly
                    logger.info(
                        f"Pending pool hit (resolved): app={app_name}{_win} "
                        f"frame={frame_id} -> label='{pending_entry.label}'"
                    )
                    self._assign_sub_frame(
                        conn,
                        frame_id=frame_id,
                        cluster_id=pending_entry.candidate_id,
                        activity_label=pending_entry.label,
                        provisional_label=pending_entry.label,
                        cluster_status="candidate",
                        cluster_updated_at=timestamp,
                    )
                    self._refresh_cluster_metadata(conn, pending_entry.candidate_id)
                    self._maybe_promote_candidate_cluster(conn, pending_entry.candidate_id)
                    self._upsert_activity_session(
                        conn,
                        app_name=app_name,
                        cluster_id=pending_entry.candidate_id,
                        label=pending_entry.label,
                        session_status="candidate",
                        timestamp=timestamp,
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
                    # Mark as pending in DB so it can be resolved later
                    self._assign_sub_frame(
                        conn,
                        frame_id=frame_id,
                        cluster_id=None,
                        activity_label=None,
                        provisional_label=None,
                        cluster_status="pending",
                        cluster_updated_at=timestamp,
                    )
                    conn.commit()
                    conn.close()
                    return None

            # This frame is a new leader — call VLM
            leader_entry = pending_entry  # was just created by _find_or_create_pending_leader
            with self._lock:
                self._vlm_called_frames += 1

            logger.info(f"VLM labeling triggered: app={app_name}{_win} time={_ts_display} sim={best_sim:.3f}")
            provisional_label = self._label_unknown_frame(conn, app_name, image, ocr_text or "")
            if provisional_label:
                logger.info(f"VLM label result: app={app_name}{_win} -> \"{provisional_label}\"")
            if not provisional_label:
                # VLM failed — mark leader and keep followers pending
                self._assign_sub_frame(
                    conn,
                    frame_id=frame_id,
                    cluster_id=None,
                    activity_label=None,
                    provisional_label=None,
                    cluster_status="pending",
                    cluster_updated_at=timestamp,
                )
                conn.commit()
                conn.close()
                return None

            # VLM returned a label — find or create candidate cluster
            candidate_id = self._find_matching_candidate_cluster(conn, app_name, provisional_label, embedding)
            if candidate_id is None:
                candidate_id = self._create_candidate_cluster(conn, app_name, provisional_label, frame_id, embedding, timestamp)

            # Assign leader frame
            self._assign_sub_frame(
                conn,
                frame_id=frame_id,
                cluster_id=candidate_id,
                activity_label=provisional_label,
                provisional_label=provisional_label,
                cluster_status="candidate",
                cluster_updated_at=timestamp,
            )

            # Resolve all followers waiting on this leader
            leader_entry.label = provisional_label
            leader_entry.candidate_id = candidate_id
            self._resolve_pending_followers(conn, leader_entry, provisional_label, candidate_id, timestamp)

            self._refresh_cluster_metadata(conn, candidate_id)
            self._maybe_promote_candidate_cluster(conn, candidate_id)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT cluster_status, label FROM activity_clusters WHERE id = ?",
                (candidate_id,),
            )
            cluster_row = cursor.fetchone()
            session_status = cluster_row["cluster_status"] if cluster_row else "candidate"
            session_label = cluster_row["label"] if cluster_row and cluster_row["label"] else provisional_label
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
            conn.close()

            self._load_centroids_for_apps({app_name})
            # Periodic cleanup of expired pending entries
            self._cleanup_pending_pool()
            return provisional_label
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
                FROM sub_frames
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

    def get_apps_needing_recalc(self) -> Set[str]:
        trigger = config.CLUSTER_UNCLASSIFIED_TRIGGER
        counts = self._get_non_committed_counts_from_db()
        return {
            app for app, per_status in counts.items()
            if per_status.get("pending", 0) >= trigger or per_status.get("candidate", 0) >= trigger
        }

    def get_debug_status(self, app_name: Optional[str] = None) -> Dict:
        counts = self._get_non_committed_counts_from_db()
        trigger = config.CLUSTER_UNCLASSIFIED_TRIGGER

        with self._lock:
            committed_cache_counts = {app: len(clusters) for app, clusters in self._cache.items()}

        app_names = set(counts.keys()) | set(committed_cache_counts.keys())
        if app_name:
            app_names = {app_name}

        apps = []
        for name in sorted(app_names):
            per_status = counts.get(name, {"pending": 0, "candidate": 0})
            apps.append({
                "app_name": name,
                "pending_unclassified": per_status.get("pending", 0),
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
        if self._recalc_running:
            return False
        return bool(self.get_apps_needing_recalc())

    def recalculate(self, lancedb_table):
        if self._recalc_running:
            return
        self._recalc_running = True
        try:
            target_apps = self.get_apps_needing_recalc()
            if not target_apps:
                return
            self._do_recalculate(lancedb_table, target_apps)
        except Exception as e:
            logger.error(f"Cluster recalculation failed: {e}", exc_info=True)
        finally:
            self._recalc_running = False

    def _do_recalculate(self, lancedb_table, target_apps: Set[str]):
        conn = self._connect()
        cursor = conn.cursor()
        sqlite_db = None
        affected_apps: Set[str] = set()
        embeddings_map = self._load_embeddings_for_apps(lancedb_table, target_apps)

        try:
            # 1. Try to provisionally label pending frames that previously had no image/VLM result.
            cursor.execute(
                f"""
                SELECT sub_frame_id, app_name
                FROM sub_frames
                WHERE cluster_status = 'pending'
                  AND app_name IN ({",".join("?" for _ in target_apps)})
                ORDER BY timestamp
                """,
                list(target_apps),
            )
            pending_rows = cursor.fetchall()
            if pending_rows:
                from core.storage.sqlite_storage import SQLiteStorage
                sqlite_db = SQLiteStorage(db_path=self.db_path)
                logger.info(
                    f"Recalculate: {len(pending_rows)} pending frames to process "
                    f"for apps {sorted(target_apps)}"
                )

            vlm_failures_in_batch = 0
            max_vlm_failures_in_batch = 5  # stop processing if VLM keeps failing

            for row in pending_rows:
                app_name = row["app_name"]
                frame_id = row["sub_frame_id"]
                embedding = embeddings_map.get(frame_id)
                if embedding is None:
                    continue
                image = sqlite_db.extract_sub_frame_image(frame_id) if sqlite_db is not None else None
                ocr_text = ""
                try:
                    oc = conn.cursor()
                    oc.execute(
                        "SELECT text FROM ocr_text WHERE sub_frame_id = ? ORDER BY text_length DESC LIMIT 1",
                        (frame_id,),
                    )
                    r = oc.fetchone()
                    if r:
                        ocr_text = r["text"]
                except Exception:
                    pass
                # Stop batch if VLM keeps failing or circuit breaker is open
                if vlm_failures_in_batch >= max_vlm_failures_in_batch or self._is_vlm_circuit_open():
                    logger.info(
                        f"Recalculate: stopping early — VLM circuit open or "
                        f"{vlm_failures_in_batch} failures in this batch"
                    )
                    break

                provisional = self._label_unknown_frame(conn, app_name, image, ocr_text)
                if not provisional:
                    vlm_failures_in_batch += 1
                    continue
                candidate_id = self._find_matching_candidate_cluster(conn, app_name, provisional, embedding)
                if candidate_id is None:
                    candidate_id = self._create_candidate_cluster(conn, app_name, provisional, frame_id, embedding, None)
                self._assign_sub_frame(
                    conn,
                    frame_id=frame_id,
                    cluster_id=candidate_id,
                    activity_label=provisional,
                    provisional_label=provisional,
                    cluster_status="candidate",
                )
                self._refresh_cluster_metadata(conn, candidate_id)
                cursor.execute(
                    "SELECT timestamp FROM sub_frames WHERE sub_frame_id = ?",
                    (frame_id,),
                )
                ts_row = cursor.fetchone()
                self._upsert_activity_session(
                    conn,
                    app_name=app_name,
                    cluster_id=candidate_id,
                    label=provisional,
                    session_status="candidate",
                    timestamp=ts_row["timestamp"] if ts_row else None,
                )
                affected_apps.add(app_name)
                # Commit after each frame to release the write lock promptly.
                # This prevents the recalculate loop (which may call VLM per frame)
                # from holding the SQLite write lock for minutes at a time.
                conn.commit()

            # 2. Promote candidate clusters that have gathered enough support.
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
