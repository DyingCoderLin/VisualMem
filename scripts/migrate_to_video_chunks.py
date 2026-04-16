#!/usr/bin/env python3
# scripts/migrate_to_video_chunks.py
"""
Migration Script: Migrate existing data to new video chunk schema

This script:
1. Adds new columns to existing tables if not present
2. Creates new tables (video_chunks, window_chunks, sub_frames, frame_subframe_mapping)
3. Optionally converts existing JPEG frames to video chunks

Usage:
    python scripts/migrate_to_video_chunks.py [--convert-images] [--dry-run]

Options:
    --convert-images    Convert existing JPEG images to MP4 video chunks
    --dry-run           Show what would be done without making changes
    --db-path PATH      Path to the database file
"""
import os
import sys
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__)


class SchemaMigrator:
    """Handles database schema migration"""
    
    def __init__(self, db_path: str, dry_run: bool = False):
        self.db_path = Path(db_path)
        self.dry_run = dry_run
        
        if not self.db_path.exists():
            logger.error(f"Database not found: {db_path}")
            sys.exit(1)
    
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _table_exists(self, cursor, table_name: str) -> bool:
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (table_name,))
        return cursor.fetchone() is not None
    
    def _column_exists(self, cursor, table_name: str, column_name: str) -> bool:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        return column_name in columns
    
    def migrate_schema(self):
        """Run schema migrations"""
        logger.info("Starting schema migration...")
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # 1. Add new columns to frames table
            self._migrate_frames_table(cursor)
            
            # 2. Add new column to ocr_text table
            self._migrate_ocr_text_table(cursor)
            
            # 3. Create new tables
            self._create_video_chunks_table(cursor)
            self._create_window_chunks_table(cursor)
            self._create_sub_frames_table(cursor)
            self._create_frame_subframe_mapping_table(cursor)
            
            # 4. Create new indexes
            self._create_indexes(cursor)
            
            if not self.dry_run:
                conn.commit()
                logger.info("Schema migration completed successfully")
            else:
                logger.info("[DRY RUN] Schema migration would be applied")
                conn.rollback()
                
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _migrate_frames_table(self, cursor):
        """Add new columns to frames table"""
        new_columns = [
            ("video_chunk_id", "INTEGER"),
            ("offset_index", "INTEGER"),
            ("monitor_id", "INTEGER DEFAULT 0"),
            ("image_hash", "INTEGER"),
        ]
        
        for col_name, col_type in new_columns:
            if not self._column_exists(cursor, "frames", col_name):
                logger.info(f"Adding column 'frames.{col_name}'")
                if not self.dry_run:
                    cursor.execute(f"""
                        ALTER TABLE frames ADD COLUMN {col_name} {col_type}
                    """)
    
    def _migrate_ocr_text_table(self, cursor):
        """Add sub_frame_id column to ocr_text table"""
        if not self._column_exists(cursor, "ocr_text", "sub_frame_id"):
            logger.info("Adding column 'ocr_text.sub_frame_id'")
            if not self.dry_run:
                cursor.execute("""
                    ALTER TABLE ocr_text ADD COLUMN sub_frame_id TEXT
                """)
    
    def _create_video_chunks_table(self, cursor):
        """Create video_chunks table"""
        if self._table_exists(cursor, "video_chunks"):
            logger.info("Table 'video_chunks' already exists")
            return
        
        logger.info("Creating table 'video_chunks'")
        if not self.dry_run:
            cursor.execute("""
                CREATE TABLE video_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    monitor_id INTEGER NOT NULL DEFAULT 0,
                    device_name TEXT,
                    fps REAL DEFAULT 1.0,
                    frame_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def _create_window_chunks_table(self, cursor):
        """Create window_chunks table"""
        if self._table_exists(cursor, "window_chunks"):
            logger.info("Table 'window_chunks' already exists")
            return
        
        logger.info("Creating table 'window_chunks'")
        if not self.dry_run:
            cursor.execute("""
                CREATE TABLE window_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    app_name TEXT NOT NULL,
                    window_name TEXT NOT NULL,
                    monitor_id INTEGER NOT NULL DEFAULT 0,
                    fps REAL DEFAULT 1.0,
                    frame_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def _create_sub_frames_table(self, cursor):
        """Create sub_frames table"""
        if self._table_exists(cursor, "sub_frames"):
            logger.info("Table 'sub_frames' already exists")
            return
        
        logger.info("Creating table 'sub_frames'")
        if not self.dry_run:
            cursor.execute("""
                CREATE TABLE sub_frames (
                    sub_frame_id TEXT PRIMARY KEY,
                    window_chunk_id INTEGER,
                    offset_index INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    app_name TEXT NOT NULL,
                    window_name TEXT NOT NULL,
                    process_id INTEGER,
                    is_focused INTEGER DEFAULT 0,
                    image_hash INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (window_chunk_id) REFERENCES window_chunks(id)
                )
            """)
    
    def _create_frame_subframe_mapping_table(self, cursor):
        """Create frame_subframe_mapping table"""
        if self._table_exists(cursor, "frame_subframe_mapping"):
            logger.info("Table 'frame_subframe_mapping' already exists")
            return
        
        logger.info("Creating table 'frame_subframe_mapping'")
        if not self.dry_run:
            cursor.execute("""
                CREATE TABLE frame_subframe_mapping (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame_id TEXT NOT NULL,
                    sub_frame_id TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (frame_id) REFERENCES frames(frame_id),
                    FOREIGN KEY (sub_frame_id) REFERENCES sub_frames(sub_frame_id),
                    UNIQUE(frame_id, sub_frame_id)
                )
            """)
    
    def _create_indexes(self, cursor):
        """Create new indexes"""
        indexes = [
            ("idx_frames_video_chunk", "frames(video_chunk_id)"),
            ("idx_ocr_sub_frame_id", "ocr_text(sub_frame_id)"),
            ("idx_sub_frames_timestamp", "sub_frames(timestamp)"),
            ("idx_sub_frames_window_chunk", "sub_frames(window_chunk_id)"),
            ("idx_sub_frames_app", "sub_frames(app_name)"),
            ("idx_mapping_frame", "frame_subframe_mapping(frame_id)"),
            ("idx_mapping_subframe", "frame_subframe_mapping(sub_frame_id)"),
        ]
        
        for idx_name, idx_def in indexes:
            try:
                logger.info(f"Creating index '{idx_name}'")
                if not self.dry_run:
                    cursor.execute(f"""
                        CREATE INDEX IF NOT EXISTS {idx_name} ON {idx_def}
                    """)
            except sqlite3.OperationalError as e:
                logger.warning(f"Index creation skipped: {e}")


class ImageToVideoConverter:
    """Converts existing JPEG images to video chunks"""
    
    def __init__(
        self,
        db_path: str,
        output_dir: str,
        fps: float = 1.0,
        chunk_duration: int = 60,
        dry_run: bool = False
    ):
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.chunk_duration = chunk_duration
        self.frames_per_chunk = int(fps * chunk_duration)
        self.dry_run = dry_run
    
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def convert_images_to_chunks(self):
        """Convert existing JPEG images to MP4 video chunks"""
        logger.info("Starting image to video conversion...")
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get frames without video_chunk_id
        cursor.execute("""
            SELECT frame_id, timestamp, image_path, device_name
            FROM frames
            WHERE video_chunk_id IS NULL
            ORDER BY timestamp ASC
        """)
        
        frames = cursor.fetchall()
        
        if not frames:
            logger.info("No frames to convert")
            conn.close()
            return
        
        logger.info(f"Found {len(frames)} frames to convert")
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would convert {len(frames)} frames")
            conn.close()
            return
        
        # Group frames by device_name
        frames_by_device: Dict[str, List] = {}
        for frame in frames:
            device = frame["device_name"] or "default"
            if device not in frames_by_device:
                frames_by_device[device] = []
            frames_by_device[device].append(dict(frame))
        
        # Process each device
        for device_name, device_frames in frames_by_device.items():
            self._convert_device_frames(cursor, device_name, device_frames)
        
        conn.commit()
        conn.close()
        
        logger.info("Image to video conversion completed")
    
    def _convert_device_frames(
        self,
        cursor,
        device_name: str,
        frames: List[Dict]
    ):
        """Convert frames for a single device"""
        from core.storage.video_chunk_writer import VideoChunkWriter
        from PIL import Image
        
        logger.info(f"Converting {len(frames)} frames for device '{device_name}'")
        
        # Extract monitor_id from device_name if possible
        try:
            monitor_id = int(device_name.split("_")[1]) if "_" in device_name else 0
        except:
            monitor_id = 0
        
        chunk_dir = self.output_dir / "screens" / "migrated"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        current_chunk_id = None
        frame_count = 0
        
        def on_chunk_created(chunk_path: str):
            nonlocal current_chunk_id
            # Insert chunk record
            cursor.execute("""
                INSERT INTO video_chunks (file_path, monitor_id, device_name, fps)
                VALUES (?, ?, ?, ?)
            """, (chunk_path, monitor_id, device_name, self.fps))
            current_chunk_id = cursor.lastrowid
            logger.info(f"Created chunk {current_chunk_id}: {chunk_path}")
        
        writer = VideoChunkWriter(
            output_dir=str(chunk_dir),
            chunk_type="screen",
            identifier=device_name,
            fps=self.fps,
            chunk_duration=self.chunk_duration,
            on_chunk_created=on_chunk_created
        )
        
        for frame_data in frames:
            image_path = frame_data["image_path"]
            
            # Skip if image doesn't exist
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Load image
            try:
                image = Image.open(image_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
            except Exception as e:
                logger.error(f"Failed to load image {image_path}: {e}")
                continue
            
            # Write to video chunk
            offset_index = writer.write_frame(image)
            
            if offset_index is not None and current_chunk_id:
                # Update frame record
                cursor.execute("""
                    UPDATE frames 
                    SET video_chunk_id = ?, offset_index = ?, monitor_id = ?
                    WHERE frame_id = ?
                """, (current_chunk_id, offset_index, monitor_id, frame_data["frame_id"]))
                frame_count += 1
        
        writer.close()
        
        # Update final chunk frame count
        if current_chunk_id:
            cursor.execute("""
                UPDATE video_chunks SET frame_count = ? WHERE id = ?
            """, (frame_count % self.frames_per_chunk or self.frames_per_chunk, current_chunk_id))
        
        logger.info(f"Converted {frame_count} frames for device '{device_name}'")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate VisualMem database to new video chunk schema"
    )
    parser.add_argument(
        "--convert-images",
        action="store_true",
        help="Convert existing JPEG images to MP4 video chunks"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to the database file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for video chunks"
    )
    
    args = parser.parse_args()
    
    # Determine paths
    db_path = args.db_path or getattr(config, 'OCR_DB_PATH', './data/ocr.db')
    output_dir = args.output_dir or getattr(config, 'DATA_DIR', './data')
    
    logger.info(f"Database path: {db_path}")
    logger.info(f"Output directory: {output_dir}")
    
    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")
    
    # Step 1: Migrate schema
    migrator = SchemaMigrator(db_path, dry_run=args.dry_run)
    migrator.migrate_schema()
    
    # Step 2: Convert images to video chunks (optional)
    if args.convert_images:
        converter = ImageToVideoConverter(
            db_path=db_path,
            output_dir=output_dir,
            fps=1.0,
            chunk_duration=60,
            dry_run=args.dry_run
        )
        converter.convert_images_to_chunks()
    
    logger.info("Migration completed!")


if __name__ == "__main__":
    main()
