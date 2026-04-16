#!/usr/bin/env python3
"""
Rebuild OCR regions for historical data.

Scans all frames and sub_frames in SQLite, extracts images from MP4 chunks,
runs region-level OCR (UIED + platform OCR), and writes results to
ocr_regions + ocr_text tables.

Usage:
    python scripts/rebuild_ocr_regions.py --limit 5 --yes
    python scripts/rebuild_ocr_regions.py --ocr-engine apple_vision
"""
import argparse
import os
import sys
import shutil
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config
from core.ocr.ocr_engine import create_ocr_engine
from core.ocr.region_detector import UIEDRegionDetector
from core.ocr.region_ocr_engine import RegionOCREngine
from core.storage.sqlite_storage import SQLiteStorage
from utils.logger import setup_logger

logger = setup_logger("rebuild_ocr_regions")


def backup_db(db_path: str):
    """Backup the SQLite database."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.bak.{ts}"
    shutil.copy2(db_path, backup_path)
    logger.info(f"Database backed up to: {backup_path}")
    return backup_path


def get_all_frames(db: SQLiteStorage, limit: int = 0, skip_done: bool = False):
    """Get all frames that have video chunk references."""
    conn = db._get_connection()
    cursor = conn.cursor()
    sql = """
        SELECT f.frame_id, f.video_chunk_id, f.offset_index,
               f.focused_app_name, f.focused_window_name,
               vc.file_path, vc.fps
        FROM frames f
        JOIN video_chunks vc ON f.video_chunk_id = vc.id
        WHERE f.video_chunk_id IS NOT NULL
    """
    if skip_done:
        sql += """
            AND f.frame_id NOT IN (
                SELECT DISTINCT frame_id FROM ocr_regions WHERE frame_id IS NOT NULL
            )
        """
    sql += " ORDER BY f.timestamp ASC"
    if limit > 0:
        sql += f" LIMIT {limit}"
    cursor.execute(sql)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_sub_frames(db: SQLiteStorage, limit: int = 0, skip_done: bool = False):
    """Get all sub_frames that have window chunk references."""
    conn = db._get_connection()
    cursor = conn.cursor()
    sql = """
        SELECT sf.sub_frame_id, sf.window_chunk_id, sf.offset_index,
               sf.app_name, sf.window_name,
               wc.file_path, wc.fps
        FROM sub_frames sf
        JOIN window_chunks wc ON sf.window_chunk_id = wc.id
        WHERE sf.window_chunk_id IS NOT NULL
    """
    if skip_done:
        sql += """
            AND sf.sub_frame_id NOT IN (
                SELECT DISTINCT sub_frame_id FROM ocr_regions WHERE sub_frame_id IS NOT NULL
            )
        """
    sql += " ORDER BY sf.timestamp ASC"
    if limit > 0:
        sql += f" LIMIT {limit}"
    cursor.execute(sql)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def main():
    parser = argparse.ArgumentParser(description="Rebuild OCR regions for historical data")
    parser.add_argument("--db-path", default=config.OCR_DB_PATH, help="SQLite DB path")
    parser.add_argument("--ocr-engine", default=config.OCR_ENGINE_TYPE,
                        help="OCR engine: auto/apple_vision/windows_ocr/pytesseract")
    parser.add_argument("--skip-backup", action="store_true", help="Skip database backup")
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip frames/sub_frames already in ocr_regions (resume from where left off)")
    parser.add_argument("--limit", type=int, default=0, help="Limit frames to process (0=all)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    db_path = args.db_path
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        sys.exit(1)

    # Initialize components
    db = SQLiteStorage(db_path=db_path)
    ocr_engine = create_ocr_engine(args.ocr_engine, lang="chi_sim+eng")
    detector = UIEDRegionDetector()
    region_ocr = RegionOCREngine(ocr_engine=ocr_engine, region_detector=detector)

    from core.storage.ffmpeg_utils import FFmpegFrameExtractor
    extractor = FFmpegFrameExtractor()

    # Get frame counts
    frames = get_all_frames(db, args.limit, skip_done=args.skip_done)
    sub_frames = get_all_sub_frames(db, args.limit, skip_done=args.skip_done)

    print(f"Database: {db_path}")
    print(f"OCR engine: {ocr_engine.engine_name}")
    print(f"Frames to process: {len(frames)}")
    print(f"Sub-frames to process: {len(sub_frames)}")
    print(f"Total: {len(frames) + len(sub_frames)}")

    if not args.yes:
        confirm = input("\nProceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)

    # Backup
    if not args.skip_backup:
        backup_db(db_path)

    start_time = time.perf_counter()
    total_regions = 0
    processed = 0
    errors = 0

    # Process frames
    for i, f in enumerate(frames):
        try:
            file_path = f["file_path"]
            if not file_path or not os.path.exists(file_path):
                logger.warning(f"Video file not found: {file_path}, skipping frame {f['frame_id']}")
                errors += 1
                continue

            image = extractor.extract_frame_by_index(
                file_path, f["offset_index"], f.get("fps", 1.0)
            )
            if image is None:
                logger.warning(f"Failed to extract frame {f['frame_id']}")
                errors += 1
                continue

            regions = region_ocr.recognize_regions(image)
            img_w, img_h = image.size

            db.store_ocr_with_regions(
                frame_id=f["frame_id"],
                regions=regions,
                ocr_engine=ocr_engine.engine_name,
                image_width=img_w,
                image_height=img_h,
                focused_app_name=f.get("focused_app_name"),
                focused_window_name=f.get("focused_window_name"),
            )

            total_regions += len(regions)
            processed += 1

            if (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"  Frames: {i+1}/{len(frames)}, regions={total_regions}, "
                      f"elapsed={elapsed:.1f}s")

        except Exception as e:
            logger.error(f"Error processing frame {f['frame_id']}: {e}")
            errors += 1

    # Process sub_frames
    for i, sf in enumerate(sub_frames):
        try:
            file_path = sf["file_path"]
            if not file_path or not os.path.exists(file_path):
                logger.warning(f"Video file not found: {file_path}, skipping sub_frame {sf['sub_frame_id']}")
                errors += 1
                continue

            image = extractor.extract_frame_by_index(
                file_path, sf["offset_index"], sf.get("fps", 1.0)
            )
            if image is None:
                logger.warning(f"Failed to extract sub_frame {sf['sub_frame_id']}")
                errors += 1
                continue

            regions = region_ocr.recognize_regions(image)
            img_w, img_h = image.size

            db.store_ocr_with_regions(
                sub_frame_id=sf["sub_frame_id"],
                regions=regions,
                ocr_engine=ocr_engine.engine_name,
                image_width=img_w,
                image_height=img_h,
            )

            total_regions += len(regions)
            processed += 1

            if (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"  Sub-frames: {i+1}/{len(sub_frames)}, regions={total_regions}, "
                      f"elapsed={elapsed:.1f}s")

        except Exception as e:
            logger.error(f"Error processing sub_frame {sf['sub_frame_id']}: {e}")
            errors += 1

    elapsed_total = time.perf_counter() - start_time

    print(f"\n{'='*50}")
    print(f"Rebuild complete!")
    print(f"  Processed: {processed}")
    print(f"  Errors: {errors}")
    print(f"  Total regions: {total_regions}")
    print(f"  Elapsed: {elapsed_total:.1f}s")
    if processed > 0:
        print(f"  Avg per frame: {elapsed_total/processed:.2f}s")


if __name__ == "__main__":
    main()
