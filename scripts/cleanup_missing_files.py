#!/usr/bin/env python3
"""
Cleanup Missing-File Frames

Scans SQLite for frames / sub_frames whose backing files (MP4 chunks or
direct image paths) no longer exist on disk, then removes their records
from SQLite and LanceDB.

Two kinds of "dead" entries are detected:

  1. video_chunks / window_chunks whose MP4 file is missing.
     → All frames / sub_frames referencing that chunk become unreachable.

  2. frames where video_chunk_id IS NULL and image_path doesn't exist.
     → Legacy direct-image frames with no backing file.

Cascade order (SQLite):
  ocr_regions  → ocr_text  → frame_subframe_mapping
  → sub_frames → frames → window_chunks → video_chunks

LanceDB tables cleaned: frames, ocr_text  (by frame_id / sub_frame_id)

Usage:
    # Dry run (safe, no changes)
    python scripts/cleanup_missing_files.py

    # Actually delete
    python scripts/cleanup_missing_files.py --yes

    # Restrict to a time window (ISO prefix or full ISO string)
    python scripts/cleanup_missing_files.py --since "2026-04-01T18:00" --yes

    # Use a different database
    python scripts/cleanup_missing_files.py --db-path /path/to/ocr.db --yes
"""
import sys
import os
import argparse
import sqlite3
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=OFF")   # we handle cascade manually
    return conn


def table_exists(cursor: sqlite3.Cursor, name: str) -> bool:
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cursor.fetchone() is not None


def _ph(ids) -> tuple:
    """Return (placeholders_str, list) for an IN clause."""
    lst = list(ids)
    return ",".join(["?"] * len(lst)), lst


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

def find_dead_entries(db_path: str, since: str | None = None):
    """
    Return (dead_frame_ids, dead_sub_frame_ids, dead_video_chunk_ids,
            dead_window_chunk_ids).
    All are sets of the appropriate id types.
    """
    conn = get_conn(db_path)
    cur = conn.cursor()

    dead_video_chunk_ids = set()
    dead_window_chunk_ids = set()
    dead_frame_ids = set()
    dead_sub_frame_ids = set()

    since_clause = "AND f.timestamp >= ?" if since else ""
    since_sf_clause = "AND sf.timestamp >= ?" if since else ""
    since_args = [since] if since else []

    # ------------------------------------------------------------------
    # 1. Dead video_chunks (full-screen MP4 missing)
    # ------------------------------------------------------------------
    cur.execute("SELECT id, file_path FROM video_chunks")
    for row in cur.fetchall():
        if not row["file_path"] or not os.path.exists(row["file_path"]):
            dead_video_chunk_ids.add(row["id"])

    # Frames referencing dead video chunks
    if dead_video_chunk_ids:
        ph, vals = _ph(dead_video_chunk_ids)
        cur.execute(
            f"SELECT frame_id, timestamp FROM frames "
            f"WHERE video_chunk_id IN ({ph}) {since_clause}",
            vals + since_args,
        )
        for row in cur.fetchall():
            dead_frame_ids.add(row["frame_id"])

    # ------------------------------------------------------------------
    # 2. Frames with direct image_path (no video chunk) where file missing
    # ------------------------------------------------------------------
    cur.execute(
        f"SELECT frame_id, image_path FROM frames "
        f"WHERE video_chunk_id IS NULL {since_clause}",
        since_args,
    )
    for row in cur.fetchall():
        if row["image_path"] and not os.path.exists(row["image_path"]):
            dead_frame_ids.add(row["frame_id"])

    # ------------------------------------------------------------------
    # 3. Dead window_chunks (window MP4 missing)
    # ------------------------------------------------------------------
    cur.execute("SELECT id, file_path FROM window_chunks")
    for row in cur.fetchall():
        if not row["file_path"] or not os.path.exists(row["file_path"]):
            dead_window_chunk_ids.add(row["id"])

    # sub_frames referencing dead window chunks
    if dead_window_chunk_ids:
        ph, vals = _ph(dead_window_chunk_ids)
        cur.execute(
            f"SELECT sf.sub_frame_id FROM sub_frames sf "
            f"WHERE sf.window_chunk_id IN ({ph}) {since_sf_clause}",
            vals + since_args,
        )
        for row in cur.fetchall():
            dead_sub_frame_ids.add(row["sub_frame_id"])

    # ------------------------------------------------------------------
    # 4. _fullscreen sub_frames whose parent frames are all dead
    #    (window_chunk_id IS NULL → depends on the parent frame's video)
    # ------------------------------------------------------------------
    cur.execute(
        f"SELECT sf.sub_frame_id FROM sub_frames sf "
        f"WHERE (sf.window_chunk_id IS NULL OR sf.window_chunk_id = 0) "
        f"{since_sf_clause}",
        since_args,
    )
    fullscreen_sids = [row["sub_frame_id"] for row in cur.fetchall()]

    if fullscreen_sids:
        # Check each: does it have at least one live parent frame?
        for sid in fullscreen_sids:
            cur.execute(
                "SELECT frame_id FROM frame_subframe_mapping WHERE sub_frame_id = ?",
                (sid,),
            )
            parent_ids = [r["frame_id"] for r in cur.fetchall()]
            live_parents = [p for p in parent_ids if p not in dead_frame_ids]
            if not live_parents:
                dead_sub_frame_ids.add(sid)

    conn.close()
    return dead_frame_ids, dead_sub_frame_ids, dead_video_chunk_ids, dead_window_chunk_ids


# ---------------------------------------------------------------------------
# delete from SQLite
# ---------------------------------------------------------------------------

def delete_sqlite(
    db_path: str,
    frame_ids: set,
    sub_frame_ids: set,
    video_chunk_ids: set,
    window_chunk_ids: set,
    dry_run: bool = True,
):
    if not any([frame_ids, sub_frame_ids, video_chunk_ids, window_chunk_ids]):
        print("SQLite: nothing to delete.")
        return

    conn = get_conn(db_path)
    cur = conn.cursor()
    prefix = "[DRY RUN] " if dry_run else ""

    def _count_and_delete(table, col, ids, label=None):
        if not ids:
            return
        ph, vals = _ph(ids)
        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IN ({ph})", vals)
        n = cur.fetchone()[0]
        tag = label or f"{table}.{col}"
        print(f"  {prefix}{tag}: {n} rows")
        if not dry_run and n:
            cur.execute(f"DELETE FROM {table} WHERE {col} IN ({ph})", vals)

    all_sf_ids = sub_frame_ids  # may grow below; keep as-is for now

    print("\n--- SQLite deletions ---")

    # ocr_regions
    if table_exists(cur, "ocr_regions"):
        _count_and_delete("ocr_regions", "frame_id", frame_ids, "ocr_regions (by frame_id)")
        _count_and_delete("ocr_regions", "sub_frame_id", all_sf_ids, "ocr_regions (by sub_frame_id)")

    # ocr_text
    _count_and_delete("ocr_text", "frame_id", frame_ids, "ocr_text (by frame_id)")
    _count_and_delete("ocr_text", "sub_frame_id", all_sf_ids, "ocr_text (by sub_frame_id)")

    # activity tables (if they exist)
    if table_exists(cur, "activity_sessions"):
        # sessions whose cluster_id has no remaining sub_frames are already
        # handled by the timeline rebuild; just note we won't touch them here.
        pass

    # frame_subframe_mapping
    _count_and_delete("frame_subframe_mapping", "frame_id", frame_ids, "frame_subframe_mapping (by frame_id)")
    _count_and_delete("frame_subframe_mapping", "sub_frame_id", all_sf_ids, "frame_subframe_mapping (by sub_frame_id)")

    # sub_frames
    if all_sf_ids:
        ph, vals = _ph(all_sf_ids)
        cur.execute(f"SELECT COUNT(*) FROM sub_frames WHERE sub_frame_id IN ({ph})", vals)
        n = cur.fetchone()[0]
        print(f"  {prefix}sub_frames: {n} rows")
        if not dry_run and n:
            cur.execute(f"DELETE FROM sub_frames WHERE sub_frame_id IN ({ph})", vals)

    # frames
    if frame_ids:
        ph, vals = _ph(frame_ids)
        cur.execute(f"SELECT COUNT(*) FROM frames WHERE frame_id IN ({ph})", vals)
        n = cur.fetchone()[0]
        print(f"  {prefix}frames: {n} rows")
        if not dry_run and n:
            cur.execute(f"DELETE FROM frames WHERE frame_id IN ({ph})", vals)

    # window_chunks (only those that have no remaining sub_frames)
    if window_chunk_ids and not dry_run:
        for wc_id in window_chunk_ids:
            cur.execute(
                "SELECT COUNT(*) FROM sub_frames WHERE window_chunk_id = ?", (wc_id,)
            )
            remaining = cur.fetchone()[0]
            if remaining == 0:
                cur.execute("DELETE FROM window_chunks WHERE id = ?", (wc_id,))
        print(f"  Cleaned up empty window_chunks entries")
    elif window_chunk_ids:
        print(f"  {prefix}window_chunks: up to {len(window_chunk_ids)} rows (checked after sub_frames)")

    # video_chunks (only those that have no remaining frames)
    if video_chunk_ids and not dry_run:
        for vc_id in video_chunk_ids:
            cur.execute(
                "SELECT COUNT(*) FROM frames WHERE video_chunk_id = ?", (vc_id,)
            )
            remaining = cur.fetchone()[0]
            if remaining == 0:
                cur.execute("DELETE FROM video_chunks WHERE id = ?", (vc_id,))
        print(f"  Cleaned up empty video_chunks entries")
    elif video_chunk_ids:
        print(f"  {prefix}video_chunks: up to {len(video_chunk_ids)} rows (checked after frames)")

    if not dry_run:
        conn.commit()
        print("  SQLite committed.")
    conn.close()


# ---------------------------------------------------------------------------
# delete from LanceDB
# ---------------------------------------------------------------------------

def delete_lancedb(frame_ids: set, sub_frame_ids: set, dry_run: bool = True):
    all_ids = frame_ids | sub_frame_ids
    if not all_ids:
        print("LanceDB: nothing to delete.")
        return

    prefix = "[DRY RUN] " if dry_run else ""
    print("\n--- LanceDB deletions ---")

    try:
        import lancedb
        db = lancedb.connect(config.LANCEDB_PATH)
    except Exception as e:
        print(f"  Cannot connect to LanceDB at {config.LANCEDB_PATH}: {e}")
        return

    for tbl_name in ["frames", "ocr_text"]:
        try:
            tbl = db.open_table(tbl_name)
        except Exception:
            print(f"  Table '{tbl_name}' not found in LanceDB, skipping")
            continue

        count = 0
        for fid in all_ids:
            escaped = fid.replace("'", "''")
            try:
                if not dry_run:
                    tbl.delete(f"frame_id = '{escaped}'")
                count += 1
            except Exception:
                pass
        print(f"  {prefix}LanceDB '{tbl_name}': processed {count} ids")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Remove frames whose backing files are missing from disk"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Actually delete (default is dry-run)",
    )
    parser.add_argument(
        "--since",
        metavar="ISO_TIMESTAMP",
        default=None,
        help='Only consider frames at or after this time, e.g. "2026-04-01T18:30"',
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite database path (default: from config)",
    )
    parser.add_argument(
        "--skip-lancedb",
        action="store_true",
        help="Skip LanceDB cleanup (faster if you don't use vector search)",
    )
    args = parser.parse_args()

    db_path = args.db_path or config.OCR_DB_PATH
    dry_run = not args.yes

    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        sys.exit(1)

    print("=" * 64)
    print("VisualMem — cleanup missing-file frames")
    print("=" * 64)
    print(f"  Database : {db_path}")
    print(f"  LanceDB  : {config.LANCEDB_PATH}")
    print(f"  Since    : {args.since or '(all time)'}")
    print(f"  Mode     : {'DRY RUN' if dry_run else 'LIVE DELETE'}")
    print()

    print("Scanning for dead entries...")
    dead_frames, dead_sub_frames, dead_vchunks, dead_wchunks = find_dead_entries(
        db_path, since=args.since
    )

    print(f"\nFound:")
    print(f"  {len(dead_vchunks):>6} dead video_chunks  (full-screen MP4s missing)")
    print(f"  {len(dead_wchunks):>6} dead window_chunks (window MP4s missing)")
    print(f"  {len(dead_frames):>6} dead frames")
    print(f"  {len(dead_sub_frames):>6} dead sub_frames")

    if not dead_frames and not dead_sub_frames:
        print("\nNothing to clean up — all backing files are present.")
        return

    if dry_run:
        print("\n[DRY RUN] No changes written. Re-run with --yes to delete.")
    else:
        print()
        try:
            answer = input("Proceed with deletion? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return
        if answer != "y":
            print("Aborted.")
            return

    delete_sqlite(
        db_path,
        dead_frames,
        dead_sub_frames,
        dead_vchunks,
        dead_wchunks,
        dry_run=dry_run,
    )

    if not args.skip_lancedb:
        delete_lancedb(dead_frames, dead_sub_frames, dry_run=dry_run)

    print()
    print("=" * 64)
    if dry_run:
        print("Dry run complete. Use --yes to apply.")
    else:
        print("Cleanup complete.")
        print(f"Removed {len(dead_frames)} frames + {len(dead_sub_frames)} sub_frames.")
        if dead_vchunks or dead_wchunks:
            print(
                f"Pruned up to {len(dead_vchunks)} video_chunks "
                f"and {len(dead_wchunks)} window_chunks."
            )
        print("Note: MP4 files on disk were NOT deleted.")
    print("=" * 64)


if __name__ == "__main__":
    main()
