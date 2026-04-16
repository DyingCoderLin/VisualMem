#!/usr/bin/env python3
"""
Cleanup Solid-Color Frames

Scans all video chunks (both full-screen and window), extracts frames in bulk
(one FFmpeg process per MP4 chunk), detects solid-color images, and removes
their entries from SQLite and LanceDB.

Video MP4 files are NOT modified — removing frames from the middle of an MP4
would break offset_index references for all subsequent frames.

Usage:
    python scripts/cleanup_solid_color_frames.py [--dry-run] [--std-threshold 5.0]
"""
import sys
import os
import io
import subprocess
import argparse
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from config import config
from core.preprocess.frame_diff import is_solid_color_image
from core.storage.ffmpeg_utils import find_ffmpeg_path

_FFMPEG = find_ffmpeg_path()


def get_connection(db_path):
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def extract_all_frames_from_chunk(video_path: str) -> list:
    """
    Extract ALL frames from a video chunk in a single FFmpeg call.
    Returns a list of PIL Images indexed by frame order (0-based).
    Much faster than one FFmpeg call per frame.
    """
    if not _FFMPEG or not os.path.exists(video_path):
        return []

    try:
        # Decode all frames to raw RGB via stdout
        cmd = [
            _FFMPEG,
            "-i", video_path,
            "-f", "image2pipe",
            "-vcodec", "png",
            "-"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            return []

        # Split the output into individual PNG images
        data = result.stdout
        frames = []
        # PNG magic bytes: \x89PNG\r\n\x1a\n
        PNG_MAGIC = b'\x89PNG\r\n\x1a\n'
        positions = []
        pos = 0
        while True:
            idx = data.find(PNG_MAGIC, pos)
            if idx == -1:
                break
            positions.append(idx)
            pos = idx + 8  # skip past magic

        for i, start in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else len(data)
            try:
                img = Image.open(io.BytesIO(data[start:end]))
                frames.append(img.convert('RGB'))
            except Exception:
                frames.append(None)

        return frames
    except subprocess.TimeoutExpired:
        return []
    except Exception:
        return []


def find_solid_color_frames(db_path, std_threshold=5.0):
    """
    Scan all video/window chunks by bulk-extracting frames per chunk.
    Returns (solid_frame_ids, solid_sub_frame_ids).
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    solid_frame_ids = set()
    solid_sub_frame_ids = set()

    # --- 1. Scan frames (full-screen) ---
    # Group frames by video_chunk so we extract each MP4 only once
    cursor.execute("""
        SELECT f.frame_id, f.video_chunk_id, f.offset_index,
               vc.file_path, vc.fps
        FROM frames f
        JOIN video_chunks vc ON f.video_chunk_id = vc.id
        ORDER BY f.video_chunk_id, f.offset_index
    """)
    frame_rows = cursor.fetchall()

    # Group by chunk
    chunks = defaultdict(list)
    chunk_paths = {}
    for row in frame_rows:
        cid = row["video_chunk_id"]
        chunks[cid].append(row)
        chunk_paths[cid] = row["file_path"]

    total_frames = len(frame_rows)
    total_chunks = len(chunks)
    print(f"Scanning {total_frames} frames across {total_chunks} video chunks...")

    checked = 0
    for chunk_idx, (cid, rows) in enumerate(chunks.items()):
        file_path = chunk_paths[cid]
        if not file_path or not os.path.exists(file_path):
            continue

        all_images = extract_all_frames_from_chunk(file_path)
        if not all_images:
            continue

        for row in rows:
            offset = row["offset_index"]
            if offset is None or offset >= len(all_images):
                continue
            img = all_images[offset]
            if img is None:
                continue

            checked += 1
            if is_solid_color_image(img, std_threshold):
                solid_frame_ids.add(row["frame_id"])
                print(f"  [SOLID] frame: {row['frame_id']} (chunk={cid}, offset={offset})")

        if (chunk_idx + 1) % 20 == 0:
            print(f"  ... {chunk_idx+1}/{total_chunks} chunks done, {checked} frames checked, {len(solid_frame_ids)} solid")

    print(f"Frames scan complete: {checked} checked, {len(solid_frame_ids)} solid-color found\n")

    # --- 2. Scan sub_frames (window captures) ---
    cursor.execute("""
        SELECT sf.sub_frame_id, sf.window_chunk_id, sf.offset_index,
               sf.app_name, sf.window_name,
               wc.file_path, wc.fps
        FROM sub_frames sf
        JOIN window_chunks wc ON sf.window_chunk_id = wc.id
        WHERE sf.window_chunk_id > 0
        ORDER BY sf.window_chunk_id, sf.offset_index
    """)
    sub_rows = cursor.fetchall()

    win_chunks = defaultdict(list)
    win_chunk_paths = {}
    for row in sub_rows:
        cid = row["window_chunk_id"]
        win_chunks[cid].append(row)
        win_chunk_paths[cid] = row["file_path"]

    total_sub = len(sub_rows)
    total_win_chunks = len(win_chunks)
    print(f"Scanning {total_sub} sub_frames across {total_win_chunks} window chunks...")

    checked = 0
    for chunk_idx, (cid, rows) in enumerate(win_chunks.items()):
        file_path = win_chunk_paths[cid]
        if not file_path or not os.path.exists(file_path):
            continue

        all_images = extract_all_frames_from_chunk(file_path)
        if not all_images:
            continue

        for row in rows:
            offset = row["offset_index"]
            if offset is None or offset >= len(all_images):
                continue
            img = all_images[offset]
            if img is None:
                continue

            checked += 1
            if is_solid_color_image(img, std_threshold):
                solid_sub_frame_ids.add(row["sub_frame_id"])
                print(f"  [SOLID] sub_frame: {row['sub_frame_id']} ({row['app_name']}/{row['window_name']})")

        if (chunk_idx + 1) % 50 == 0:
            print(f"  ... {chunk_idx+1}/{total_win_chunks} window chunks done, {checked} sub_frames checked, {len(solid_sub_frame_ids)} solid")

    print(f"Sub-frames scan complete: {checked} checked, {len(solid_sub_frame_ids)} solid-color found")

    conn.close()
    return solid_frame_ids, solid_sub_frame_ids


def delete_from_sqlite(db_path, frame_ids, sub_frame_ids, dry_run=False):
    """Delete solid-color frame entries from all SQLite tables."""
    if not frame_ids and not sub_frame_ids:
        print("Nothing to delete from SQLite.")
        return

    conn = get_connection(db_path)
    cursor = conn.cursor()

    prefix = "[DRY RUN] " if dry_run else ""

    # Delete OCR regions
    if frame_ids:
        placeholders = ",".join(["?"] * len(frame_ids))
        cursor.execute(f"SELECT COUNT(*) FROM ocr_regions WHERE frame_id IN ({placeholders})", list(frame_ids))
        count = cursor.fetchone()[0]
        print(f"{prefix}Deleting {count} ocr_regions rows for {len(frame_ids)} frames")
        if not dry_run:
            cursor.execute(f"DELETE FROM ocr_regions WHERE frame_id IN ({placeholders})", list(frame_ids))

    if sub_frame_ids:
        placeholders = ",".join(["?"] * len(sub_frame_ids))
        cursor.execute(f"SELECT COUNT(*) FROM ocr_regions WHERE sub_frame_id IN ({placeholders})", list(sub_frame_ids))
        count = cursor.fetchone()[0]
        print(f"{prefix}Deleting {count} ocr_regions rows for {len(sub_frame_ids)} sub_frames")
        if not dry_run:
            cursor.execute(f"DELETE FROM ocr_regions WHERE sub_frame_id IN ({placeholders})", list(sub_frame_ids))

    # Delete OCR text
    if frame_ids:
        placeholders = ",".join(["?"] * len(frame_ids))
        cursor.execute(f"SELECT COUNT(*) FROM ocr_text WHERE frame_id IN ({placeholders})", list(frame_ids))
        count = cursor.fetchone()[0]
        print(f"{prefix}Deleting {count} ocr_text rows for frames")
        if not dry_run:
            cursor.execute(f"DELETE FROM ocr_text WHERE frame_id IN ({placeholders})", list(frame_ids))

    if sub_frame_ids:
        placeholders = ",".join(["?"] * len(sub_frame_ids))
        cursor.execute(f"SELECT COUNT(*) FROM ocr_text WHERE sub_frame_id IN ({placeholders})", list(sub_frame_ids))
        count = cursor.fetchone()[0]
        print(f"{prefix}Deleting {count} ocr_text rows for sub_frames")
        if not dry_run:
            cursor.execute(f"DELETE FROM ocr_text WHERE sub_frame_id IN ({placeholders})", list(sub_frame_ids))

    # Delete frame_subframe_mapping
    if frame_ids:
        placeholders = ",".join(["?"] * len(frame_ids))
        cursor.execute(f"SELECT COUNT(*) FROM frame_subframe_mapping WHERE frame_id IN ({placeholders})", list(frame_ids))
        count = cursor.fetchone()[0]
        print(f"{prefix}Deleting {count} mapping rows (by frame_id)")
        if not dry_run:
            cursor.execute(f"DELETE FROM frame_subframe_mapping WHERE frame_id IN ({placeholders})", list(frame_ids))

    if sub_frame_ids:
        placeholders = ",".join(["?"] * len(sub_frame_ids))
        cursor.execute(f"SELECT COUNT(*) FROM frame_subframe_mapping WHERE sub_frame_id IN ({placeholders})", list(sub_frame_ids))
        count = cursor.fetchone()[0]
        print(f"{prefix}Deleting {count} mapping rows (by sub_frame_id)")
        if not dry_run:
            cursor.execute(f"DELETE FROM frame_subframe_mapping WHERE sub_frame_id IN ({placeholders})", list(sub_frame_ids))

    # Delete sub_frames
    if sub_frame_ids:
        placeholders = ",".join(["?"] * len(sub_frame_ids))
        print(f"{prefix}Deleting {len(sub_frame_ids)} sub_frames rows")
        if not dry_run:
            cursor.execute(f"DELETE FROM sub_frames WHERE sub_frame_id IN ({placeholders})", list(sub_frame_ids))

    # Delete frames
    if frame_ids:
        placeholders = ",".join(["?"] * len(frame_ids))
        print(f"{prefix}Deleting {len(frame_ids)} frames rows")
        if not dry_run:
            cursor.execute(f"DELETE FROM frames WHERE frame_id IN ({placeholders})", list(frame_ids))

    if not dry_run:
        conn.commit()
    conn.close()
    print(f"{prefix}SQLite cleanup done.")


def delete_from_lancedb(frame_ids, sub_frame_ids, dry_run=False):
    """Delete solid-color frame entries from LanceDB (frames + ocr_text tables)."""
    all_ids = frame_ids | sub_frame_ids
    if not all_ids:
        print("Nothing to delete from LanceDB.")
        return

    prefix = "[DRY RUN] " if dry_run else ""

    try:
        import lancedb
        db = lancedb.connect(config.LANCEDB_PATH)
    except Exception as e:
        print(f"Cannot connect to LanceDB at {config.LANCEDB_PATH}: {e}")
        return

    for table_name in ["frames", "ocr_text"]:
        try:
            table = db.open_table(table_name)
        except Exception:
            print(f"  LanceDB table '{table_name}' not found, skipping")
            continue

        deleted = 0
        for fid in all_ids:
            try:
                fid_escaped = fid.replace("'", "''")
                if not dry_run:
                    table.delete(f"frame_id = '{fid_escaped}'")
                deleted += 1
            except Exception:
                pass

        print(f"{prefix}Deleted up to {deleted} entries from LanceDB table '{table_name}'")

    # Also delete JPEG images saved by LanceDB
    deleted_images = 0
    if not dry_run:
        import glob as globmod
        for fid in all_ids:
            try:
                pattern = os.path.join(config.IMAGE_STORAGE_PATH, "**", f"{fid}.jpg")
                for img_path in globmod.glob(pattern, recursive=True):
                    os.remove(img_path)
                    deleted_images += 1
            except Exception:
                pass

    if deleted_images > 0:
        print(f"{prefix}Deleted {deleted_images} JPEG image files")

    print(f"{prefix}LanceDB cleanup done.")


def main():
    parser = argparse.ArgumentParser(description="Remove solid-color frames from database")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    parser.add_argument("--std-threshold", type=float, default=5.0,
                        help="Standard deviation threshold for solid-color detection (default: 5.0)")
    parser.add_argument("--db-path", type=str, default=None,
                        help="SQLite database path (default: from config)")
    args = parser.parse_args()

    db_path = args.db_path or config.OCR_DB_PATH

    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        sys.exit(1)

    print(f"Database: {db_path}")
    print(f"LanceDB: {config.LANCEDB_PATH}")
    print(f"Std threshold: {args.std_threshold}")
    if args.dry_run:
        print("=== DRY RUN MODE ===\n")

    # Phase 1: Scan and identify solid-color frames
    print("=" * 60)
    print("Phase 1: Scanning video chunks for solid-color frames...")
    print("=" * 60)
    solid_frames, solid_sub_frames = find_solid_color_frames(db_path, args.std_threshold)

    total = len(solid_frames) + len(solid_sub_frames)
    if total == 0:
        print("\nNo solid-color frames found. Database is clean!")
        return

    print(f"\nFound {len(solid_frames)} solid-color frames + {len(solid_sub_frames)} solid-color sub_frames")

    if not args.dry_run:
        answer = input("\nProceed with deletion? [y/N] ")
        if answer.lower() != 'y':
            print("Aborted.")
            return

    # Phase 2: Delete from SQLite
    print("\n" + "=" * 60)
    print("Phase 2: Cleaning SQLite...")
    print("=" * 60)
    delete_from_sqlite(db_path, solid_frames, solid_sub_frames, dry_run=args.dry_run)

    # Phase 3: Delete from LanceDB
    print("\n" + "=" * 60)
    print("Phase 3: Cleaning LanceDB...")
    print("=" * 60)
    delete_from_lancedb(solid_frames, solid_sub_frames, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("Cleanup complete!")
    print("=" * 60)
    print(f"Removed {len(solid_frames)} frames + {len(solid_sub_frames)} sub_frames")
    print("Note: Video MP4 files were NOT modified (offset indices preserved).")


if __name__ == "__main__":
    main()
