#!/usr/bin/env python3
"""
Export representative sub_frame images for clusters that **text-only** cluster labeling skips.

When `CLUSTER_VLM_USE_VISION=false`, `phase label` requires at least one of:
- non-empty `ocr_text` for the first representative frame, or
- non-empty region layout text (only if `CLUSTER_VLM_INCLUDE_REGION_LAYOUT=true`).

If both are empty, you see:
  No OCR/layout for cluster N (text-only labeling), skipping

Images are decoded the same way as the rest of VisualMem: MP4 chunks are read via
FFmpeg (`SQLiteStorage.extract_sub_frame_image` → `FFmpegFrameExtractor`), then saved as JPEG.

Usage (conda env `mobiagent`):
  python scripts/dump_no_ocr_cluster_frames.py --output-dir cluster_output/no_ocr_skip_frames
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config
from core.storage.sqlite_storage import SQLiteStorage
from utils.logger import setup_logger

logger = setup_logger("dump_no_ocr_cluster_frames")


def _load_backfill_helpers():
    p = Path(__file__).resolve().parent / "backfill_activity_clusters.py"
    spec = importlib.util.spec_from_file_location("backfill_activity_clusters", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump frames for clusters skipped in text-only labeling (no OCR/layout on first rep)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cluster_output") / "no_ocr_skip_frames",
        help="Directory for extracted JPEGs (created if missing).",
    )
    parser.add_argument(
        "--max-per-cluster",
        type=int,
        default=3,
        help="Max representative frames to save per cluster (same as label phase sampling).",
    )
    args = parser.parse_args()

    bf = _load_backfill_helpers()
    get_ocr = bf.get_ocr_for_sub_frame
    get_win_layout = bf.get_window_and_layout_for_sub_frame
    get_act = bf.get_activity_connection

    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    sqlite_db = SQLiteStorage(db_path=config.OCR_DB_PATH, activity_db_path=config.ACTIVITY_DB_PATH)
    act_conn = get_act()
    cursor = act_conn.cursor()
    cursor.execute(
        """
        SELECT id, app_name, label, representative_frame_ids
        FROM activity_clusters
        WHERE label LIKE '%_activity_%'
        ORDER BY app_name, id
        """
    )
    rows = [dict(r) for r in cursor.fetchall()]
    act_conn.close()

    n_clusters = 0
    n_saved = 0
    for row in rows:
        rep_ids_raw = row["representative_frame_ids"]
        try:
            rep_ids = json.loads(rep_ids_raw) if isinstance(rep_ids_raw, str) else list(rep_ids_raw)
        except Exception:
            logger.warning("Bad representative_frame_ids for cluster %s", row["id"])
            continue
        if not rep_ids:
            continue
        first_id = rep_ids[0]
        ocr = (get_ocr(sqlite_db, first_id) or "").strip()
        _wn, layout = get_win_layout(sqlite_db, first_id)
        layout = (layout or "").strip()
        if ocr or layout:
            continue

        n_clusters += 1
        safe_app = (row["app_name"] or "unknown").replace("/", "_").replace(" ", "_")[:80]
        for fid in rep_ids[: max(1, args.max_per_cluster)]:
            img = sqlite_db.extract_sub_frame_image(fid)
            if img is None:
                logger.warning("Could not extract image cluster=%s sub_frame=%s", row["id"], fid)
                continue
            name = f"c{row['id']}_{safe_app}_{fid}.jpg"
            path = out / name
            img.save(str(path), "JPEG", quality=92)
            n_saved += 1

    print(
        f"Done. Skippable clusters (no OCR/layout on first rep): {n_clusters}, "
        f"images saved: {n_saved} -> {out.resolve()}"
    )


if __name__ == "__main__":
    main()
