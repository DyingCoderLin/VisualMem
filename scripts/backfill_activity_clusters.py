#!/usr/bin/env python3
"""
Backfill activity clusters for per-app semantic activity labeling.

Three phases:
  A) discover  - Cluster sub_frames by app using cosine similarity on embeddings
  B) label     - Use VLM to generate semantic labels for each cluster
  C) timeline  - Build activity sessions from clustered frames

Usage:
    python scripts/backfill_activity_clusters.py --phase discover
    python scripts/backfill_activity_clusters.py --phase label --vlm-url http://localhost:8088
    python scripts/backfill_activity_clusters.py --phase timeline
    python scripts/backfill_activity_clusters.py --phase all --vlm-url http://localhost:8088
    python scripts/backfill_activity_clusters.py --phase all --mode full --vlm-url http://localhost:8088
"""
import argparse
import base64
import io
import os
import sys
import time
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import re
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config
from core.storage.sqlite_storage import SQLiteStorage
from utils.logger import setup_logger

logger = setup_logger("backfill_activity_clusters")


def _utcnow() -> str:
    """Current UTC time as naive ISO string, matching frames/sub_frames format."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLD = 0.82  # cosine similarity threshold for same cluster
SOLID_COLOR_STD_THRESHOLD = 10.0  # pixel std below this -> solid color / blank image
CLUSTER_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "cluster_output")
REPRESENTATIVE_SAMPLE_COUNT = 3


# ===========================================================================
# Database Schema
# ===========================================================================

def ensure_tables(db_path: str, activity_db_path: str = None):
    """Create activity tables in the activity DB and ensure main DB compatibility."""
    activity_db_path = activity_db_path or config.ACTIVITY_DB_PATH

    # Ensure activity tables in activity DB
    act_conn = sqlite3.connect(activity_db_path)
    act_cursor = act_conn.cursor()

    act_cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_assignments (
            sub_frame_id TEXT PRIMARY KEY,
            app_name TEXT NOT NULL,
            timestamp TEXT,
            activity_cluster_id INTEGER,
            activity_label TEXT,
            provisional_label TEXT,
            cluster_status TEXT,
            frozen_at TEXT,
            cluster_updated_at TEXT,
            pending_group_id INTEGER
        )
    """)

    act_cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            app_name TEXT NOT NULL,
            label TEXT NOT NULL,
            centroid BLOB,
            frame_count INTEGER DEFAULT 0,
            representative_frame_ids TEXT,
            cluster_status TEXT NOT NULL DEFAULT 'committed',
            committed_at TEXT,
            last_frame_timestamp TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    act_cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            app_name TEXT NOT NULL,
            cluster_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            frame_count INTEGER DEFAULT 0,
            session_status TEXT NOT NULL DEFAULT 'committed',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (cluster_id) REFERENCES activity_clusters(id)
        )
    """)

    act_cursor.execute("""
        CREATE TABLE IF NOT EXISTS pending_groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            app_name TEXT NOT NULL,
            leader_frame_id TEXT NOT NULL,
            centroid BLOB NOT NULL,
            member_count INTEGER DEFAULT 1,
            resolved_label TEXT,
            resolved_cluster_id INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    act_cursor.execute("CREATE INDEX IF NOT EXISTS idx_aa_cluster_status ON activity_assignments(app_name, cluster_status)")
    act_cursor.execute("CREATE INDEX IF NOT EXISTS idx_aa_label ON activity_assignments(activity_label)")
    act_cursor.execute("CREATE INDEX IF NOT EXISTS idx_aa_cluster_id ON activity_assignments(activity_cluster_id)")
    act_cursor.execute("CREATE INDEX IF NOT EXISTS idx_activity_clusters_app ON activity_clusters(app_name)")
    act_cursor.execute("CREATE INDEX IF NOT EXISTS idx_activity_clusters_app_status ON activity_clusters(app_name, cluster_status)")
    act_cursor.execute("CREATE INDEX IF NOT EXISTS idx_activity_sessions_app ON activity_sessions(app_name)")
    act_cursor.execute("CREATE INDEX IF NOT EXISTS idx_activity_sessions_time ON activity_sessions(start_time, end_time)")

    act_conn.commit()
    act_conn.close()


def get_activity_connection(activity_db_path: str = None) -> sqlite3.Connection:
    """Get a connection to the activity DB."""
    path = activity_db_path or config.ACTIVITY_DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


# ===========================================================================
# Helpers
# ===========================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def load_sub_frames_from_lancedb() -> "pd.DataFrame":
    """Load all sub_frame rows from LanceDB frames table."""
    import lancedb
    import pandas as pd

    db = lancedb.connect(config.LANCEDB_PATH)
    table = db.open_table("frames")
    df = table.to_pandas()
    sub_df = df[df["frame_id"].str.startswith("subframe_")].copy()
    logger.info(f"Loaded {len(sub_df)} sub_frames from LanceDB ({config.LANCEDB_PATH})")
    return sub_df


def load_json_dict(path: Path) -> Dict:
    """Load a JSON object from disk, returning {} on missing/invalid files."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_json_dict(path: Path, data: Dict):
    """Persist a JSON object atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_pending_unclustered_subframes(main_conn: sqlite3.Connection, act_conn: sqlite3.Connection) -> Dict[str, sqlite3.Row]:
    """Return sub_frame rows that still need clustering.

    Reads from main DB for sub_frame metadata, cross-references activity DB
    to find frames not yet assigned.
    """
    main_conn.row_factory = sqlite3.Row
    act_conn.row_factory = sqlite3.Row
    main_cursor = main_conn.cursor()
    act_cursor = act_conn.cursor()

    # Get all sub_frames from main DB
    main_cursor.execute(
        """
        SELECT sub_frame_id, app_name, timestamp
        FROM sub_frames
        WHERE app_name IS NOT NULL AND app_name != ''
        ORDER BY app_name, timestamp
        """
    )
    all_frames = {row["sub_frame_id"]: row for row in main_cursor.fetchall()}

    # Get already-assigned frame IDs from activity DB
    act_cursor.execute(
        "SELECT sub_frame_id FROM activity_assignments WHERE activity_cluster_id IS NOT NULL"
    )
    assigned_ids = {row["sub_frame_id"] for row in act_cursor.fetchall()}

    # Return unassigned frames
    return {fid: row for fid, row in all_frames.items() if fid not in assigned_ids}


def load_existing_clusters(conn: sqlite3.Connection) -> Dict[str, List[dict]]:
    """Load existing clusters from activity DB grouped by app."""
    """Load existing DB clusters grouped by app."""
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, app_name, label, centroid, frame_count, representative_frame_ids
        FROM activity_clusters
        WHERE centroid IS NOT NULL
        ORDER BY app_name, id
        """
    )
    grouped: Dict[str, List[dict]] = {}
    for row in cursor.fetchall():
        grouped.setdefault(row["app_name"], []).append({
            "id": row["id"],
            "app_name": row["app_name"],
            "label": row["label"],
            "centroid": np.frombuffer(row["centroid"], dtype=np.float32).copy(),
            "frame_count": row["frame_count"] or 0,
            "representative_frame_ids": json.loads(row["representative_frame_ids"] or "[]"),
        })
    return grouped


def recompute_centroids_for_apps(
    conn: sqlite3.Connection,
    embeddings_map: Dict[str, np.ndarray],
    app_names: List[str],
):
    """Recompute exact mean centroids for the given apps."""
    if not app_names:
        return

    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in app_names)
    cursor.execute(
        f"SELECT id FROM activity_clusters WHERE app_name IN ({placeholders})",
        app_names,
    )
    cluster_ids = [row["id"] for row in cursor.fetchall()]
    updated = 0

    for cluster_id in cluster_ids:
        cursor.execute(
            "SELECT sub_frame_id FROM activity_assignments WHERE activity_cluster_id = ?",
            (cluster_id,),
        )
        frame_ids = [row["sub_frame_id"] for row in cursor.fetchall()]
        vectors = [embeddings_map[fid] for fid in frame_ids if fid in embeddings_map]
        if not vectors:
            continue
        centroid = np.mean(vectors, axis=0).astype(np.float32)
        cursor.execute(
            """
            UPDATE activity_clusters
            SET centroid = ?, frame_count = ?, updated_at = ?
            WHERE id = ?
            """,
            (centroid.tobytes(), len(vectors), _utcnow(), cluster_id),
        )
        updated += 1

    logger.info(f"Recomputed {updated} centroids for apps: {app_names}")


def merge_discover_summary(
    summary_path: Path,
    new_summary_by_app: Dict[str, Dict],
):
    """Merge new discover output into the per-app summary JSON."""
    existing = load_json_dict(summary_path)

    for app_name, app_summary in new_summary_by_app.items():
        current = existing.get(app_name, {
            "total_frames": 0,
            "cluster_count": 0,
            "clusters": [],
            "incremental_runs": [],
        })
        current.setdefault("clusters", [])
        current.setdefault("incremental_runs", [])

        current["clusters"].extend(app_summary.get("clusters", []))
        current["cluster_count"] = int(current.get("cluster_count", 0)) + int(app_summary.get("cluster_count", 0))
        current["total_frames"] = int(current.get("total_frames", 0)) + int(app_summary.get("total_frames", 0))
        current["incremental_runs"].append({
            "timestamp": _utcnow(),
            "new_cluster_count": app_summary.get("cluster_count", 0),
            "new_frame_count": app_summary.get("total_frames", 0),
        })
        existing[app_name] = current

    save_json_dict(summary_path, existing)


def extract_and_save_image(
    sqlite_db: SQLiteStorage, sub_frame_id: str, save_path: str
) -> bool:
    """Extract a sub_frame image from MP4 and save as JPEG."""
    try:
        img = sqlite_db.extract_sub_frame_image(sub_frame_id)
        if img is None:
            return False
        img.save(save_path, "JPEG", quality=85)
        return True
    except Exception as e:
        logger.warning(f"Failed to extract image for {sub_frame_id}: {e}")
        return False


def is_solid_color_embedding(embedding: np.ndarray, all_embeddings: List[np.ndarray]) -> bool:
    """
    Heuristic: solid-color images tend to have very low embedding norm
    or near-zero variance across dimensions. Check if embedding norm is
    abnormally low compared to the population.
    """
    norm = np.linalg.norm(embedding)
    return norm < 1e-6


def is_solid_color_image(img) -> bool:
    """Check if a PIL Image is essentially a solid color (black screen, white screen, etc.)."""
    try:
        arr = np.array(img)
        # Check pixel standard deviation - solid/near-solid images have very low std
        return float(arr.std()) < SOLID_COLOR_STD_THRESHOLD
    except Exception:
        return False


def image_to_base64(img) -> str:
    """Convert PIL Image to base64 data URI."""
    buf = io.BytesIO()
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ===========================================================================
# Phase A: Discover Clusters
# ===========================================================================

class ActivityClusterManager:
    """Per-app incremental clustering via cosine similarity with running average centroids."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        # {app_name: [{centroid: np.array, frame_ids: [...], count: int}, ...]}
        self.app_clusters: Dict[str, List[dict]] = {}

    def process_frame(self, app_name: str, frame_id: str, embedding: np.ndarray):
        """Assign a frame to the nearest cluster or create a new one."""
        if app_name not in self.app_clusters:
            self.app_clusters[app_name] = []

        clusters = self.app_clusters[app_name]
        best_sim = -1.0
        best_idx = -1

        for i, cluster in enumerate(clusters):
            sim = cosine_similarity(embedding, cluster["centroid"])
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_sim >= self.threshold and best_idx >= 0:
            # Assign to existing cluster, update running average centroid
            cluster = clusters[best_idx]
            n = cluster["count"]
            cluster["centroid"] = (cluster["centroid"] * n + embedding) / (n + 1)
            cluster["count"] = n + 1
            cluster["frame_ids"].append(frame_id)
        else:
            # New cluster
            clusters.append({
                "centroid": embedding.copy(),
                "count": 1,
                "frame_ids": [frame_id],
            })


def rank_frames_by_centroid(cluster: dict, embeddings_map: Dict[str, np.ndarray]) -> List[str]:
    """Rank all frames in cluster by distance to centroid (closest first)."""
    centroid = cluster["centroid"]
    scored = []
    for fid in cluster["frame_ids"]:
        if fid in embeddings_map:
            sim = cosine_similarity(embeddings_map[fid], centroid)
            scored.append((sim, fid))
    scored.sort(reverse=True)
    return [fid for _, fid in scored]


def phase_discover(args):
    """Phase A: discover clusters from LanceDB embeddings."""
    import pandas as pd

    sqlite_db = SQLiteStorage(db_path=config.OCR_DB_PATH, activity_db_path=config.ACTIVITY_DB_PATH)
    ensure_tables(config.OCR_DB_PATH, config.ACTIVITY_DB_PATH)
    conn = sqlite3.connect(config.OCR_DB_PATH)  # main DB for reading sub_frame metadata
    act_conn = get_activity_connection()  # activity DB for cluster writes

    output_dir = Path(CLUSTER_OUTPUT_DIR)
    preview_dir = output_dir / "preview"

    if args.mode == "full":
        import shutil

        logger.info("Clearing previous cluster data...")
        act_conn.execute("DELETE FROM activity_sessions")
        act_conn.execute("DELETE FROM activity_clusters")
        act_conn.execute("DELETE FROM activity_assignments")
        act_conn.execute("DELETE FROM pending_groups")
        act_conn.commit()

        if output_dir.exists():
            shutil.rmtree(output_dir)
            logger.info(f"Cleared {output_dir}")
    else:
        pending_rows = get_pending_unclustered_subframes(conn, act_conn)
        if not pending_rows:
            print("No pending unclustered sub_frames found. Discover phase skipped.")
            conn.close()
            act_conn.close()
            return
        logger.info(f"Incremental discover: {len(pending_rows)} pending sub_frames")

    # Load sub_frames from LanceDB
    all_sub_df = load_sub_frames_from_lancedb()
    if all_sub_df.empty:
        logger.warning("No sub_frames found in LanceDB.")
        conn.close()
        act_conn.close()
        return

    sub_df = all_sub_df
    if args.mode == "incremental":
        pending_ids = set(pending_rows.keys())
        sub_df = sub_df[sub_df["frame_id"].isin(pending_ids)].copy()
        if sub_df.empty:
            print("No pending unclustered sub_frames were found in LanceDB. Discover phase skipped.")
            conn.close()
            act_conn.close()
            return

    # Build embedding map, filtering out solid-color frames
    embeddings_map: Dict[str, np.ndarray] = {}
    skipped_solid = 0
    for _, row in sub_df.iterrows():
        vec = np.array(row["vector"], dtype=np.float32)
        if is_solid_color_embedding(vec, []):
            skipped_solid += 1
            continue
        embeddings_map[row["frame_id"]] = vec

    if skipped_solid:
        logger.info(f"Skipped {skipped_solid} frames with near-zero embedding norm")

    preview_dir.mkdir(parents=True, exist_ok=True)
    act_conn.row_factory = sqlite3.Row
    cursor = act_conn.cursor()
    existing_clusters = load_existing_clusters(act_conn)
    summary_data = {}  # per-app JSON export for newly created clusters only
    affected_apps = set()

    print("\n=== Clustering Summary ===")
    grouped = sub_df.groupby("app_name")
    for app_name, group in sorted(grouped, key=lambda x: x[0]):
        if not app_name or app_name == "":
            continue

        sorted_group = group.sort_values("timestamp")
        existing = existing_clusters.get(app_name, [])
        new_clusters: List[dict] = []
        assigned_existing = 0

        for _, row in sorted_group.iterrows():
            fid = row["frame_id"]
            if fid not in embeddings_map:
                continue

            emb = embeddings_map[fid]
            best_sim = -1.0
            best_target = None

            for cluster in existing:
                sim = cosine_similarity(emb, cluster["centroid"])
                if sim > best_sim:
                    best_sim = sim
                    best_target = cluster

            for cluster in new_clusters:
                sim = cosine_similarity(emb, cluster["centroid"])
                if sim > best_sim:
                    best_sim = sim
                    best_target = cluster

            if best_sim >= args.threshold and best_target is not None:
                if "id" in best_target:
                    cursor.execute(
                        """INSERT INTO activity_assignments
                        (sub_frame_id, app_name, timestamp, activity_cluster_id, activity_label,
                         cluster_status, cluster_updated_at)
                        VALUES (?, ?, ?, ?, ?, 'committed', ?)
                        ON CONFLICT(sub_frame_id) DO UPDATE SET
                            activity_cluster_id = excluded.activity_cluster_id,
                            activity_label = excluded.activity_label,
                            cluster_status = excluded.cluster_status,
                            cluster_updated_at = excluded.cluster_updated_at""",
                        (fid, app_name, row.get("timestamp"), best_target["id"],
                         best_target["label"], _utcnow()),
                    )
                    assigned_existing += 1
                else:
                    best_target["frame_ids"].append(fid)
                    n = len(best_target["frame_ids"])
                    best_target["count"] = n
                    best_target["centroid"] = (
                        best_target["centroid"] * (n - 1) + emb
                    ) / n
                affected_apps.add(app_name)
            else:
                new_clusters.append({
                    "centroid": emb.copy(),
                    "count": 1,
                    "frame_ids": [fid],
                })
                affected_apps.add(app_name)

        total_frames = assigned_existing + sum(len(c["frame_ids"]) for c in new_clusters)
        new_cluster_frame_count = sum(len(c["frame_ids"]) for c in new_clusters)
        print(f"\nApp: {app_name} ({total_frames} new/pending frames)")
        if assigned_existing:
            print(f"  Assigned to existing clusters: {assigned_existing}")

        app_summary = []
        existing_count = len(existing)
        valid_idx = existing_count
        for cluster in new_clusters:
            # Flush previous writes before preview extraction
            act_conn.commit()
            ranked_fids = rank_frames_by_centroid(cluster, embeddings_map)
            if not ranked_fids:
                continue

            # Check the frame closest to centroid — if it's solid-color, discard entire cluster
            center_img = sqlite_db.extract_sub_frame_image(ranked_fids[0])
            if center_img is None or is_solid_color_image(center_img):
                # Save centroid image for debugging when --save-discarded is set
                if args.save_discarded and center_img is not None:
                    safe_app = app_name.replace("/", "_").replace(" ", "_")
                    discard_dir = output_dir / "discarded"
                    discard_dir.mkdir(parents=True, exist_ok=True)
                    discard_path = discard_dir / f"{safe_app}_discarded_{cluster['count']}frames_{ranked_fids[0]}.jpg"
                    center_img.save(str(discard_path), "JPEG", quality=85)
                    print(f"  [DISCARDED] cluster with {cluster['count']} frames -> {discard_path}")
                else:
                    print(f"  [DISCARDED] cluster with {cluster['count']} frames (centroid is solid-color)")
                continue

            valid_idx += 1
            label = f"{app_name}_activity_{valid_idx}"
            centroid_blob = cluster["centroid"].tobytes()

            # Collect valid preview images
            safe_app = app_name.replace("/", "_").replace(" ", "_")
            saved_samples = []
            valid_rep_ids = []
            for fid in ranked_fids:
                if len(saved_samples) >= REPRESENTATIVE_SAMPLE_COUNT:
                    break
                img = sqlite_db.extract_sub_frame_image(fid)
                if img is None:
                    continue
                if is_solid_color_image(img):
                    continue
                sample_idx = len(saved_samples) + 1
                img_path = preview_dir / f"{safe_app}_cluster_{valid_idx}_sample_{sample_idx}.jpg"
                img.save(str(img_path), "JPEG", quality=85)
                saved_samples.append(str(img_path))
                valid_rep_ids.append(fid)

            cursor.execute("""
                INSERT INTO activity_clusters
                (app_name, label, centroid, frame_count, representative_frame_ids,
                 cluster_status, committed_at, last_frame_timestamp)
                VALUES (?, ?, ?, ?, ?, 'committed', ?, ?)
            """, (
                app_name,
                label,
                centroid_blob,
                cluster["count"],
                json.dumps(valid_rep_ids),
                _utcnow(),
                None,
            ))
            cluster_id = cursor.lastrowid
            now_iso = _utcnow()
            for fid in cluster["frame_ids"]:
                # Get timestamp from LanceDB data
                fid_rows = sub_df[sub_df["frame_id"] == fid]
                fid_ts = fid_rows.iloc[0]["timestamp"] if not fid_rows.empty else None
                cursor.execute(
                    """
                    INSERT INTO activity_assignments
                    (sub_frame_id, app_name, timestamp, activity_cluster_id, activity_label,
                     provisional_label, cluster_status, cluster_updated_at, frozen_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'committed', ?, ?)
                    ON CONFLICT(sub_frame_id) DO UPDATE SET
                        activity_cluster_id = excluded.activity_cluster_id,
                        activity_label = excluded.activity_label,
                        provisional_label = excluded.provisional_label,
                        cluster_status = excluded.cluster_status,
                        cluster_updated_at = excluded.cluster_updated_at,
                        frozen_at = COALESCE(activity_assignments.frozen_at, excluded.frozen_at)
                    """,
                    (fid, app_name, fid_ts, cluster_id, label, label, now_iso, now_iso),
                )
            act_conn.commit()

            print(f"  Cluster {valid_idx}: \"{label}\" ({cluster['count']} frames) [db_id={cluster_id}]")
            for sp in saved_samples:
                print(f"    -> {sp}")

            app_summary.append({
                "cluster_id": cluster_id,
                "label": label,
                "frame_count": cluster["count"],
                "representative_frame_ids": valid_rep_ids,
                "preview_images": saved_samples,
            })

        if app_summary:
            summary_data[app_name] = {
                "total_frames": new_cluster_frame_count,
                "cluster_count": len(app_summary),
                "clusters": app_summary,
            }

    recompute_embeddings_map: Dict[str, np.ndarray] = {}
    if affected_apps:
        recompute_df = all_sub_df[all_sub_df["app_name"].isin(affected_apps)].copy()
        for _, row in recompute_df.iterrows():
            vec = np.array(row["vector"], dtype=np.float32)
            if np.linalg.norm(vec) < 1e-6:
                continue
            recompute_embeddings_map[row["frame_id"]] = vec

    recompute_centroids_for_apps(act_conn, recompute_embeddings_map, sorted(affected_apps))

    act_conn.commit()
    act_conn.close()
    conn.close()

    # Write JSON summary
    summary_path = output_dir / "discover_summary.json"
    if args.mode == "full":
        save_json_dict(summary_path, summary_data)
    else:
        merge_discover_summary(summary_path, summary_data)

    print(f"\nPreview images saved to: {preview_dir}")
    print(f"Summary JSON saved to: {summary_path}")
    print("Review the clusters and preview images, then run --phase label")


# ===========================================================================
# Phase B: VLM Label Generation
# ===========================================================================

_verbose_log_path = os.path.join(CLUSTER_OUTPUT_DIR, "vlm_verbose.json")
_verbose_log_entries = []


def _log_vlm_verbose(app_name: str, raw: str, parsed_label: str):
    """Append a VLM call record to the verbose log file."""
    _verbose_log_entries.append({
        "app_name": app_name,
        "raw_response": raw,
        "parsed_label": parsed_label,
        "timestamp": _utcnow(),
    })
    Path(_verbose_log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(_verbose_log_path, "w", encoding="utf-8") as f:
        json.dump(_verbose_log_entries, f, ensure_ascii=False, indent=2)


def call_vlm(vlm_url: str, app_name: str, images: List, existing_labels: List[str], ocr_texts: List[str], verbose: bool = False) -> str:
    """Call VLM to generate a semantic label. Only sends 1 image + OCR text."""
    import requests

    # Only use the first image — multiple images are usually near-identical
    img = images[0]
    b64 = image_to_base64(img)
    ocr_text = ""
    if ocr_texts and ocr_texts[0] and ocr_texts[0].strip():
        ocr_text = ocr_texts[0].strip()[:500]

    existing_labels_text = ""
    if existing_labels:
        existing_labels_text = (
            f"该应用已有标签供参考：{', '.join(existing_labels)}\n"
            f"如果截图内容与某个已有标签相同，可以复用。但不要强行匹配，内容不同就创建新标签。\n\n"
        )

    user_text = f"应用：{app_name}\n{existing_labels_text}"
    if ocr_text:
        user_text += f"屏幕文字：{ocr_text}\n"
    user_text += (
        "请简要描述这张桌面截图中用户在做什么操作，然后在最后一行输出：\n"
        "标签：（你的标签）\n"
        "标签要求3-8个中文词，描述具体操作。如：和朋友聊天、阅读arXiv论文、编写Python代码、查看邮件通知"
    )

    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": user_text},
        ]},
    ]

    payload = {
        "model": "Qwen/Qwen3.5-9B",
        "messages": messages,
        "temperature": 0.1,
    }

    url = vlm_url.rstrip("/") + "/v1/chat/completions"
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        raw = result["choices"][0]["message"]["content"].strip()
        label = _parse_label(raw)
        if verbose:
            _log_vlm_verbose(app_name, raw, label)
        return label
    except Exception as e:
        logger.error(f"VLM call failed: {e}")
        return ""


def _clean_label(text: str) -> str:
    """Clean a candidate label string."""
    text = text.strip().strip('"').strip("'").strip('*').strip('`').strip()
    # Remove markdown bold
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    # Remove trailing punctuation
    text = re.sub(r'[。.，,！!？?；;：:]+$', '', text).strip()
    # Remove leading markers
    text = re.sub(r'^[\d]+[.、)\]]\s*', '', text).strip()
    text = re.sub(r'^[-•·]\s*', '', text).strip()
    # Remove "或者" prefix (with or without colon/space)
    text = re.sub(r'^或者\s*[：:]?\s*', '', text).strip()
    # If contains "、" (enumeration), take only the first item
    if '、' in text:
        text = text.split('、')[0].strip()
    # If contains "，" or "," (comma), take only the first clause
    for sep in ['，', ',']:
        if sep in text:
            text = text.split(sep)[0].strip()
    # Final cleanup: strip any remaining quotes/parens
    text = text.strip('"').strip("'").strip('"').strip('"').strip('(').strip(')').strip('（').strip('）').strip()
    return text


_GARBAGE_PATTERNS = re.compile(
    r'(xxx|用户|截图|显示|正在|需要|提供|可以|应该|让我|候选|分析|或者|这里截断'
    r'|如果截图|标签要求|中文词|描述具体|请分析|分析完后|输出标签|回复格式'
    r'|属于以上|第一行|看起来像)'
)


def _is_valid_label(text: str) -> bool:
    """Check if a cleaned string is a valid label."""
    if not text or len(text) < 2 or len(text) > 30:
        return False
    if _GARBAGE_PATTERNS.search(text):
        return False
    return True


def _parse_label(raw: str) -> str:
    """Extract a clean short label from VLM response.

    Strategy: scan from LAST to FIRST for '标签[：:]' pattern,
    take the content after it, clean it, validate it.
    """

    # 1. Find ALL "标签：xxx" matches, try from last to first
    matches = re.findall(r'标签[：:]\s*(.+)', raw)
    for match in reversed(matches):
        label = _clean_label(match)
        if _is_valid_label(label):
            return label

    # 2. Try JSON {"label": "..."}
    m = re.search(r'"label"\s*:\s*"([^"]+)"', raw)
    if m:
        label = _clean_label(m.group(1))
        if _is_valid_label(label):
            return label

    # 3. Scan lines from bottom up for a clean short label
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        cleaned = _clean_label(line)
        if 3 <= len(cleaned) <= 20 and _is_valid_label(cleaned):
            return cleaned

    # 4. Last resort: return empty to signal failure (caller will skip)
    return ""


def _fuzzy_match_label(candidate: str, existing_labels: List[str]) -> Optional[str]:
    """Check if candidate label matches any existing label.

    Returns the matched existing label (canonical form), or None if no match.
    Matching rules:
    1. Exact match
    2. Normalized match (strip spaces/punctuation)
    3. One is a substring of the other
    4. High character overlap (Jaccard > 0.6)
    """
    if not existing_labels:
        return None

    import re

    def _normalize(s: str) -> str:
        return re.sub(r'[\s，。、！？""\'\'·\-]', '', s).lower()

    c_norm = _normalize(candidate)

    for existing in existing_labels:
        e_norm = _normalize(existing)
        # Exact or normalized match
        if candidate == existing or c_norm == e_norm:
            return existing
        # Substring match (either direction)
        if len(c_norm) >= 3 and len(e_norm) >= 3:
            if c_norm in e_norm or e_norm in c_norm:
                return existing
        # Jaccard similarity on characters
        c_set = set(c_norm)
        e_set = set(e_norm)
        if c_set and e_set:
            jaccard = len(c_set & e_set) / len(c_set | e_set)
            if jaccard > 0.6:
                return existing

    return None


def get_ocr_for_sub_frame(sqlite_db: SQLiteStorage, sub_frame_id: str) -> str:
    """Get OCR text for a sub_frame from SQLite."""
    try:
        conn = sqlite_db._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT text FROM ocr_text WHERE sub_frame_id = ? ORDER BY text_length DESC LIMIT 1",
            (sub_frame_id,),
        )
        row = cursor.fetchone()
        conn.close()
        return row["text"] if row else ""
    except Exception:
        return ""


def phase_label(args):
    """Phase B: generate VLM labels for pending clusters."""
    if not args.vlm_url:
        print("Error: --vlm-url is required for phase label")
        sys.exit(1)

    sqlite_db = SQLiteStorage(db_path=config.OCR_DB_PATH, activity_db_path=config.ACTIVITY_DB_PATH)
    act_conn = get_activity_connection()
    cursor = act_conn.cursor()

    if args.mode == "full":
        cursor.execute("""
            UPDATE activity_clusters
            SET label = app_name || '_activity_' || id
            WHERE label NOT LIKE '%_activity_%'
        """)
        reset_count = cursor.rowcount
        if reset_count > 0:
            act_conn.commit()
            logger.info(f"Reset {reset_count} previously labeled clusters back to pending")

    # Incremental mode only labels fallback clusters.
    cursor.execute("""
        SELECT id, app_name, label, representative_frame_ids
        FROM activity_clusters
        WHERE label LIKE '%_activity_%'
        ORDER BY app_name, id
    """)
    clusters = [dict(r) for r in cursor.fetchall()]

    if not clusters:
        print("No pending clusters found for labeling.")
        act_conn.close()
        return

    # Group by app
    app_labels: Dict[str, List[str]] = {}  # app_name -> [confirmed labels]
    app_cluster_map: Dict[str, List[dict]] = {}
    for c in clusters:
        app = c["app_name"]
        app_cluster_map.setdefault(app, []).append(c)

    cursor.execute(
        """
        SELECT DISTINCT app_name, label
        FROM activity_clusters
        WHERE label NOT LIKE '%_activity_%'
        ORDER BY app_name, label
        """
    )
    for row in cursor.fetchall():
        app_labels.setdefault(row["app_name"], []).append(row["label"])

    # Label results file — written after each cluster for crash recovery
    output_dir = Path(CLUSTER_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_results_path = output_dir / "label_results.json"
    label_results = load_json_dict(label_results_path)

    def _save_label_results():
        save_json_dict(label_results_path, label_results)

    print("\n=== VLM Label Generation ===")
    for app_name in sorted(app_cluster_map.keys()):
        app_clusters_list = app_cluster_map[app_name]
        existing = app_labels.get(app_name, [])
        print(f"\nApp: {app_name} ({len(app_clusters_list)} clusters to label, {len(existing)} existing labels)")

        for cluster in app_clusters_list:
            rep_ids = json.loads(cluster["representative_frame_ids"])
            if not rep_ids:
                logger.warning(f"Cluster {cluster['id']} has no representative frames, skipping")
                continue

            # Extract images
            images = []
            ocr_texts = []
            for fid in rep_ids[:3]:
                img = sqlite_db.extract_sub_frame_image(fid)
                if img:
                    images.append(img)
                    ocr_texts.append(get_ocr_for_sub_frame(sqlite_db, fid))

            if not images:
                logger.warning(f"Could not extract any images for cluster {cluster['id']}, skipping")
                continue

            # Call VLM
            label = call_vlm(args.vlm_url, app_name, images, existing, ocr_texts, verbose=args.verbose)
            if not label:
                logger.warning(f"VLM returned empty label for cluster {cluster['id']}, keeping fallback")
                continue

            # Check if label matches an existing one (fuzzy: normalize spaces and find substring matches)
            matched_label = _fuzzy_match_label(label, existing)
            if matched_label:
                # Use the canonical existing label, not the VLM's variant
                label = matched_label
                # Merge: find the cluster with this label and merge
                cursor.execute(
                    "SELECT id FROM activity_clusters WHERE app_name = ? AND label = ? AND id != ?",
                    (app_name, matched_label, cluster["id"]),
                )
                target_row = cursor.fetchone()
                if target_row:
                    target_id = target_row["id"]
                    # Update all references from this cluster to target
                    cursor.execute(
                        "UPDATE activity_assignments SET activity_cluster_id = ?, activity_label = ? WHERE activity_cluster_id = ?",
                        (target_id, matched_label, cluster["id"]),
                    )
                    # Merge frame count
                    cursor.execute("SELECT frame_count FROM activity_clusters WHERE id = ?", (target_id,))
                    target_count = cursor.fetchone()["frame_count"]
                    cursor.execute("SELECT frame_count FROM activity_clusters WHERE id = ?", (cluster["id"],))
                    src_count = cursor.fetchone()["frame_count"]
                    cursor.execute(
                        "UPDATE activity_clusters SET frame_count = ?, updated_at = ? WHERE id = ?",
                        (target_count + src_count, _utcnow(), target_id),
                    )
                    # Delete merged cluster
                    cursor.execute("DELETE FROM activity_clusters WHERE id = ?", (cluster["id"],))
                    print(f"  Cluster {cluster['id']}: \"{cluster['label']}\" -> merged into \"{matched_label}\" (cluster {target_id})")
                else:
                    # First cluster with this label — just update
                    cursor.execute(
                        "UPDATE activity_clusters SET label = ?, updated_at = ? WHERE id = ?",
                        (matched_label, _utcnow(), cluster["id"]),
                    )
                    cursor.execute(
                        "UPDATE activity_assignments SET activity_label = ? WHERE activity_cluster_id = ?",
                        (matched_label, cluster["id"]),
                    )
                    print(f"  Cluster {cluster['id']}: \"{cluster['label']}\" -> \"{matched_label}\"")
            else:
                # New label
                cursor.execute(
                    "UPDATE activity_clusters SET label = ?, updated_at = ? WHERE id = ?",
                    (label, _utcnow(), cluster["id"]),
                )
                cursor.execute(
                    "UPDATE activity_assignments SET activity_label = ? WHERE activity_cluster_id = ?",
                    (label, cluster["id"]),
                )
                existing.append(label)
                app_labels[app_name] = existing
                print(f"  Cluster {cluster['id']}: \"{cluster['label']}\" -> \"{label}\" [NEW]")

            act_conn.commit()

            # Persist label result to JSON immediately, appending within the app section.
            label_results.setdefault(app_name, [])
            if not any(item.get("cluster_id") == cluster["id"] for item in label_results[app_name]):
                label_results[app_name].append({
                    "cluster_id": cluster["id"],
                    "old_label": cluster["label"],
                    "new_label": label,
                })
            _save_label_results()

            time.sleep(0.5)  # Rate limit

    act_conn.close()
    print(f"\nLabel results saved to: {label_results_path}")
    print("Labeling complete. Review labels, then run --phase timeline")


# ===========================================================================
# Phase C: Build Timeline Sessions
# ===========================================================================

def phase_timeline(args):
    """Phase C: build activity sessions from clustered frames in activity DB."""
    ensure_tables(config.OCR_DB_PATH, config.ACTIVITY_DB_PATH)

    act_conn = get_activity_connection()
    cursor = act_conn.cursor()

    # Remove stale sessions that point to deleted / non-committed clusters.
    cursor.execute("""
        DELETE FROM activity_sessions
        WHERE cluster_id IN (
            SELECT s.cluster_id
            FROM activity_sessions s
            LEFT JOIN activity_clusters c ON s.cluster_id = c.id
            WHERE c.id IS NULL OR COALESCE(c.cluster_status, 'committed') != 'committed'
        )
    """)
    stale_deleted = cursor.rowcount
    if stale_deleted > 0:
        act_conn.commit()
        logger.info(f"Deleted {stale_deleted} stale activity_sessions referencing missing/non-committed clusters")

    if args.mode == "full":
        cursor.execute("DELETE FROM activity_sessions")
        act_conn.commit()

    print("\n=== Activity Timeline ===")
    total_sessions = 0
    apps_processed = 0

    cursor.execute(
        """
        SELECT DISTINCT app_name
        FROM activity_assignments
        WHERE activity_cluster_id IS NOT NULL
          AND cluster_status = 'committed'
          AND app_name IS NOT NULL
          AND app_name != ''
        ORDER BY app_name
        """
    )
    app_names = [row["app_name"] for row in cursor.fetchall()]

    if not app_names:
        print("No clustered frames found. Run --phase discover first.")
        act_conn.close()
        return

    for app_name in app_names:
        rebuild_from = None

        if args.mode == "incremental":
            cursor.execute(
                """
                SELECT start_time, end_time
                FROM activity_sessions
                WHERE app_name = ?
                ORDER BY end_time DESC, id DESC
                LIMIT 1
                """,
                (app_name,),
            )
            last_session = cursor.fetchone()
            if last_session is None:
                cursor.execute(
                    """
                    SELECT MIN(timestamp) AS min_ts
                    FROM activity_assignments
                    WHERE app_name = ?
                      AND activity_cluster_id IS NOT NULL
                      AND cluster_status = 'committed'
                    """,
                    (app_name,),
                )
                row = cursor.fetchone()
                rebuild_from = row["min_ts"] if row and row["min_ts"] else None
            else:
                cursor.execute(
                    """
                    SELECT COUNT(*) AS cnt
                    FROM activity_assignments
                    WHERE app_name = ?
                      AND activity_cluster_id IS NOT NULL
                      AND cluster_status = 'committed'
                      AND timestamp > ?
                    """,
                    (app_name, last_session["end_time"]),
                )
                has_new = cursor.fetchone()["cnt"] > 0
                if has_new:
                    rebuild_from = last_session["start_time"]
        else:
            cursor.execute(
                """
                SELECT MIN(timestamp) AS min_ts
                FROM activity_assignments
                WHERE app_name = ?
                  AND activity_cluster_id IS NOT NULL
                  AND cluster_status = 'committed'
                """,
                (app_name,),
            )
            row = cursor.fetchone()
            rebuild_from = row["min_ts"] if row and row["min_ts"] else None

        if not rebuild_from:
            continue

        cursor.execute(
            "DELETE FROM activity_sessions WHERE app_name = ? AND start_time >= ?",
            (app_name, rebuild_from),
        )

        cursor.execute(
            """
            SELECT sub_frame_id, timestamp, activity_cluster_id, activity_label
            FROM activity_assignments
            WHERE app_name = ?
              AND activity_cluster_id IS NOT NULL
              AND cluster_status = 'committed'
              AND timestamp >= ?
            ORDER BY timestamp, sub_frame_id
            """,
            (app_name, rebuild_from),
        )
        rows = cursor.fetchall()
        if not rows:
            continue

        current_cluster_id = None
        current_label = None
        session_start = None
        session_frames = 0
        sessions = []
        previous_ts = None

        for row in rows:
            ts = row["timestamp"]
            cluster_id = row["activity_cluster_id"]
            label = row["activity_label"]

            if cluster_id != current_cluster_id:
                # Close previous session
                if current_cluster_id is not None and session_start is not None:
                    sessions.append((app_name, current_cluster_id, current_label, session_start, ts, session_frames))
                # Start new session
                current_cluster_id = cluster_id
                current_label = label
                session_start = ts
                session_frames = 1
            else:
                session_frames += 1
            previous_ts = ts

        # Close last session
        if current_cluster_id is not None and session_start is not None and previous_ts is not None:
            sessions.append((app_name, current_cluster_id, current_label, session_start, previous_ts, session_frames))

        # Write sessions
        for app, cid, clabel, start, end, count in sessions:
            cursor.execute("""
                INSERT INTO activity_sessions
                (app_name, cluster_id, label, start_time, end_time, frame_count, session_status)
                VALUES (?, ?, ?, ?, ?, ?, 'committed')
            """, (app, cid, clabel, start, end, count))

        total_sessions += len(sessions)
        apps_processed += 1

        # Print timeline
        for app, cid, clabel, start, end, count in sessions:
            # Format timestamps for display
            try:
                start_dt = datetime.fromisoformat(start) if isinstance(start, str) else start
                end_dt = datetime.fromisoformat(end) if isinstance(end, str) else end
                start_str = start_dt.strftime("%Y-%m-%d %H:%M") if hasattr(start_dt, "strftime") else str(start)[:16]
                end_str = end_dt.strftime("%H:%M") if hasattr(end_dt, "strftime") else str(end)[11:16]
            except Exception:
                start_str = str(start)[:16]
                end_str = str(end)[11:16]
            print(f"  [{start_str} - {end_str}] {app}: {clabel} ({count} frames)")

    act_conn.commit()
    act_conn.close()
    print(f"\nProcessed apps: {apps_processed}")
    print(f"Total sessions written: {total_sessions}")
    print("Timeline built. Check activity_sessions table in activity DB.")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Backfill activity clusters for per-app semantic labeling")
    parser.add_argument(
        "--phase",
        choices=["discover", "label", "timeline", "all"],
        required=True,
        help="Which phase to run",
    )
    parser.add_argument("--vlm-url", default="", help="VLM server URL (required for label phase)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"Cosine similarity threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument(
        "--mode",
        choices=["incremental", "full"],
        default="incremental",
        help="incremental: only process newly added/unfinished data; full: rebuild clusters/labels/timeline from scratch",
    )
    parser.add_argument("--save-discarded", action="store_true", help="Save centroid images of discarded solid-color clusters to cluster_output/discarded/")
    parser.add_argument("--verbose", action="store_true", help="Print raw VLM response for each cluster during label phase")

    args = parser.parse_args()

    # Ensure tables exist
    ensure_tables(config.OCR_DB_PATH, config.ACTIVITY_DB_PATH)

    if args.phase == "discover" or args.phase == "all":
        phase_discover(args)

    if args.phase == "label" or args.phase == "all":
        phase_label(args)

    if args.phase == "timeline" or args.phase == "all":
        phase_timeline(args)


if __name__ == "__main__":
    main()
