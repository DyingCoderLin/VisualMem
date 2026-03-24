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
"""
import argparse
import base64
import io
import os
import sys
import time
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import re
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config
from core.storage.sqlite_storage import SQLiteStorage
from utils.logger import setup_logger

logger = setup_logger("backfill_activity_clusters")

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

def ensure_tables(db_path: str):
    """Create activity_clusters, activity_sessions tables and add columns to sub_frames."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            app_name TEXT NOT NULL,
            label TEXT NOT NULL,
            centroid BLOB,
            frame_count INTEGER DEFAULT 0,
            representative_frame_ids TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            app_name TEXT NOT NULL,
            cluster_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            frame_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (cluster_id) REFERENCES activity_clusters(id)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_activity_clusters_app
        ON activity_clusters(app_name)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_activity_sessions_app
        ON activity_sessions(app_name)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_activity_sessions_time
        ON activity_sessions(start_time, end_time)
    """)

    # Add columns to sub_frames if missing
    cursor.execute("PRAGMA table_info(sub_frames)")
    existing = {row[1] for row in cursor.fetchall()}
    for col_name, col_type in [
        ("activity_cluster_id", "INTEGER"),
        ("activity_label", "TEXT"),
    ]:
        if col_name not in existing:
            cursor.execute(f"ALTER TABLE sub_frames ADD COLUMN {col_name} {col_type}")
            logger.info(f"Added column sub_frames.{col_name}")

    conn.commit()
    conn.close()


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

    sqlite_db = SQLiteStorage(db_path=config.OCR_DB_PATH)
    ensure_tables(config.OCR_DB_PATH)

    # Always clean previous discover results for a fresh run
    import shutil
    logger.info("Clearing previous cluster data...")
    conn = sqlite3.connect(config.OCR_DB_PATH)
    conn.execute("DELETE FROM activity_sessions")
    conn.execute("DELETE FROM activity_clusters")
    conn.execute("UPDATE sub_frames SET activity_cluster_id = NULL, activity_label = NULL")
    conn.commit()
    conn.close()

    output_dir = Path(CLUSTER_OUTPUT_DIR)
    if output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info(f"Cleared {output_dir}")

    # Load sub_frames from LanceDB
    sub_df = load_sub_frames_from_lancedb()
    if sub_df.empty:
        logger.warning("No sub_frames found in LanceDB.")
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

    # Group by app_name, process in timestamp order
    manager = ActivityClusterManager(threshold=args.threshold)

    grouped = sub_df.groupby("app_name")
    for app_name, group in grouped:
        if not app_name or app_name == "":
            continue
        sorted_group = group.sort_values("timestamp")
        for _, row in sorted_group.iterrows():
            fid = row["frame_id"]
            if fid not in embeddings_map:
                continue  # was filtered as solid-color
            manager.process_frame(app_name, fid, embeddings_map[fid])

    # Save clusters to SQLite and generate preview images
    output_dir = Path(CLUSTER_OUTPUT_DIR)
    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(config.OCR_DB_PATH)
    cursor = conn.cursor()

    summary_data = {}  # for JSON export

    print("\n=== Clustering Summary ===")
    for app_name, clusters in sorted(manager.app_clusters.items()):
        total_frames = sum(c["count"] for c in clusters)
        print(f"\nApp: {app_name} ({total_frames} frames)")
        app_summary = []

        valid_idx = 0
        for cluster in clusters:
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
                INSERT INTO activity_clusters (app_name, label, centroid, frame_count, representative_frame_ids)
                VALUES (?, ?, ?, ?, ?)
            """, (
                app_name,
                label,
                centroid_blob,
                cluster["count"],
                json.dumps(valid_rep_ids),
            ))
            cluster_id = cursor.lastrowid

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

        summary_data[app_name] = {
            "total_frames": total_frames,
            "cluster_count": len(app_summary),
            "clusters": app_summary,
        }

    conn.commit()
    conn.close()

    # Write JSON summary
    summary_path = output_dir / "discover_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

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
        "timestamp": datetime.now().isoformat(),
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

    sqlite_db = SQLiteStorage(db_path=config.OCR_DB_PATH)
    conn = sqlite3.connect(config.OCR_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Reset all labels back to pending so we re-label everything from scratch
    cursor.execute("""
        UPDATE activity_clusters
        SET label = app_name || '_activity_' || id
        WHERE label NOT LIKE '%_activity_%'
    """)
    reset_count = cursor.rowcount
    if reset_count > 0:
        conn.commit()
        logger.info(f"Reset {reset_count} previously labeled clusters back to pending")

    # Load all clusters (now all pending)
    cursor.execute("""
        SELECT id, app_name, label, representative_frame_ids
        FROM activity_clusters
        ORDER BY app_name, id
    """)
    clusters = [dict(r) for r in cursor.fetchall()]

    if not clusters:
        print("No clusters found. Run --phase discover first.")
        return

    # Group by app
    app_labels: Dict[str, List[str]] = {}  # app_name -> [confirmed labels]
    app_cluster_map: Dict[str, List[dict]] = {}
    for c in clusters:
        app = c["app_name"]
        app_cluster_map.setdefault(app, []).append(c)

    # Label results file — written after each cluster for crash recovery
    output_dir = Path(CLUSTER_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_results_path = output_dir / "label_results.json"
    label_results = {}
    if label_results_path.exists():
        try:
            label_results = json.loads(label_results_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    def _save_label_results():
        label_results_path.write_text(json.dumps(label_results, ensure_ascii=False, indent=2), encoding="utf-8")

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
                        "UPDATE sub_frames SET activity_cluster_id = ? WHERE activity_cluster_id = ?",
                        (target_id, cluster["id"]),
                    )
                    # Merge frame count
                    cursor.execute("SELECT frame_count FROM activity_clusters WHERE id = ?", (target_id,))
                    target_count = cursor.fetchone()["frame_count"]
                    cursor.execute("SELECT frame_count FROM activity_clusters WHERE id = ?", (cluster["id"],))
                    src_count = cursor.fetchone()["frame_count"]
                    cursor.execute(
                        "UPDATE activity_clusters SET frame_count = ?, updated_at = ? WHERE id = ?",
                        (target_count + src_count, datetime.now().isoformat(), target_id),
                    )
                    # Delete merged cluster
                    cursor.execute("DELETE FROM activity_clusters WHERE id = ?", (cluster["id"],))
                    print(f"  Cluster {cluster['id']}: \"{cluster['label']}\" -> merged into \"{matched_label}\" (cluster {target_id})")
                else:
                    # First cluster with this label — just update
                    cursor.execute(
                        "UPDATE activity_clusters SET label = ?, updated_at = ? WHERE id = ?",
                        (matched_label, datetime.now().isoformat(), cluster["id"]),
                    )
                    print(f"  Cluster {cluster['id']}: \"{cluster['label']}\" -> \"{matched_label}\"")
            else:
                # New label
                cursor.execute(
                    "UPDATE activity_clusters SET label = ?, updated_at = ? WHERE id = ?",
                    (label, datetime.now().isoformat(), cluster["id"]),
                )
                existing.append(label)
                app_labels[app_name] = existing
                print(f"  Cluster {cluster['id']}: \"{cluster['label']}\" -> \"{label}\" [NEW]")

            conn.commit()

            # Persist label result to JSON immediately
            label_results.setdefault(app_name, [])
            label_results[app_name].append({
                "cluster_id": cluster["id"],
                "old_label": cluster["label"],
                "new_label": label,
            })
            _save_label_results()

            time.sleep(0.5)  # Rate limit

    conn.close()
    print(f"\nLabel results saved to: {label_results_path}")
    print("Labeling complete. Review labels, then run --phase timeline")


# ===========================================================================
# Phase C: Build Timeline Sessions
# ===========================================================================

def phase_timeline(args):
    """Phase C: build activity sessions from clustered sub_frames."""
    import pandas as pd

    sqlite_db = SQLiteStorage(db_path=config.OCR_DB_PATH)
    ensure_tables(config.OCR_DB_PATH)

    # Load sub_frames from LanceDB with embeddings
    sub_df = load_sub_frames_from_lancedb()
    if sub_df.empty:
        logger.warning("No sub_frames found in LanceDB.")
        return

    # Load clusters from SQLite
    conn = sqlite3.connect(config.OCR_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT id, app_name, label, centroid, frame_count FROM activity_clusters ORDER BY app_name, id")
    cluster_rows = cursor.fetchall()

    if not cluster_rows:
        print("No clusters found. Run --phase discover first.")
        conn.close()
        return

    # Build cluster lookup: {app_name: [(cluster_id, label, centroid_vec), ...]}
    app_cluster_info: Dict[str, List[Tuple[int, str, np.ndarray]]] = {}
    for row in cluster_rows:
        app = row["app_name"]
        centroid = np.frombuffer(row["centroid"], dtype=np.float32)
        app_cluster_info.setdefault(app, []).append((row["id"], row["label"], centroid))

    # Clear existing sessions
    cursor.execute("DELETE FROM activity_sessions")

    # Process per app
    print("\n=== Activity Timeline ===")
    total_sessions = 0

    grouped = sub_df.groupby("app_name")
    for app_name, group in sorted(grouped, key=lambda x: x[0]):
        if not app_name or app_name == "" or app_name not in app_cluster_info:
            continue

        clusters_info = app_cluster_info[app_name]
        sorted_group = group.sort_values("timestamp")

        current_cluster_id = None
        current_label = None
        session_start = None
        session_frames = 0
        sessions = []

        for _, row in sorted_group.iterrows():
            vec = np.array(row["vector"], dtype=np.float32)
            ts = row["timestamp"]
            frame_id = row["frame_id"]

            # Find nearest cluster
            best_sim = -1.0
            best_cluster_id = clusters_info[0][0]
            best_label = clusters_info[0][1]
            for cid, clabel, centroid in clusters_info:
                sim = cosine_similarity(vec, centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_cluster_id = cid
                    best_label = clabel

            # Update sub_frame
            cursor.execute(
                "UPDATE sub_frames SET activity_cluster_id = ?, activity_label = ? WHERE sub_frame_id = ?",
                (best_cluster_id, best_label, frame_id),
            )

            # Session tracking
            if best_cluster_id != current_cluster_id:
                # Close previous session
                if current_cluster_id is not None and session_start is not None:
                    sessions.append((app_name, current_cluster_id, current_label, session_start, ts, session_frames))
                # Start new session
                current_cluster_id = best_cluster_id
                current_label = best_label
                session_start = ts
                session_frames = 1
            else:
                session_frames += 1

        # Close last session
        if current_cluster_id is not None and session_start is not None:
            last_ts = sorted_group.iloc[-1]["timestamp"]
            sessions.append((app_name, current_cluster_id, current_label, session_start, last_ts, session_frames))

        # Write sessions
        for app, cid, clabel, start, end, count in sessions:
            cursor.execute("""
                INSERT INTO activity_sessions (app_name, cluster_id, label, start_time, end_time, frame_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (app, cid, clabel, start, end, count))

        total_sessions += len(sessions)

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

    conn.commit()
    conn.close()
    print(f"\nTotal sessions: {total_sessions}")
    print("Timeline built. Check activity_sessions table.")


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
    parser.add_argument("--save-discarded", action="store_true", help="Save centroid images of discarded solid-color clusters to cluster_output/discarded/")
    parser.add_argument("--verbose", action="store_true", help="Print raw VLM response for each cluster during label phase")

    args = parser.parse_args()

    # Ensure tables exist
    ensure_tables(config.OCR_DB_PATH)

    if args.phase == "discover" or args.phase == "all":
        phase_discover(args)

    if args.phase == "label" or args.phase == "all":
        phase_label(args)

    if args.phase == "timeline" or args.phase == "all":
        phase_timeline(args)


if __name__ == "__main__":
    main()
