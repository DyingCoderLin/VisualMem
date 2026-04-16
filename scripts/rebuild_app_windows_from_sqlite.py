#!/usr/bin/env python
"""
Scan SQLite storage to rebuild the app_name -> window_names mapping
and write the result into visualmem_storage/app_names.json.

Usage:
    python scripts/rebuild_app_windows_from_sqlite.py

This script is idempotent: it merges all discovered (app_name, window_name)
pairs into the existing JSON via AppNameManager.
"""

import sqlite3
from typing import Set, Tuple
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保可以导入 config 等模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import config
from utils.app_name_manager import app_name_manager


def collect_app_window_pairs(conn: sqlite3.Connection) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    cursor = conn.cursor()

    queries = [
        # New schema: sub_frames 表中的 app_name / window_name
        """
        SELECT DISTINCT app_name, window_name
        FROM sub_frames
        WHERE app_name IS NOT NULL AND app_name <> ''
          AND window_name IS NOT NULL AND window_name <> ''
        """,
        # New schema: window_chunks 表中的 app_name / window_name
        """
        SELECT DISTINCT app_name, window_name
        FROM window_chunks
        WHERE app_name IS NOT NULL AND app_name <> ''
          AND window_name IS NOT NULL AND window_name <> ''
        """,
        # Backward‑compatible: frames 表中可能已经直接写入 app_name / window_name
        """
        SELECT DISTINCT app_name, window_name
        FROM frames
        WHERE app_name IS NOT NULL AND app_name <> ''
          AND window_name IS NOT NULL AND window_name <> ''
        """,
    ]

    for sql in queries:
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
            for app_name, window_name in rows:
                if not app_name or not window_name:
                    continue
                pairs.add((str(app_name), str(window_name)))
        except sqlite3.OperationalError as e:
            # 某些旧库可能没有对应表，直接跳过
            print(f"[warn] query failed, skip: {e}")

    return pairs


def main() -> None:
    db_path = config.OCR_DB_PATH
    print(f"Using SQLite DB: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        pairs = collect_app_window_pairs(conn)
    finally:
        conn.close()

    print(f"Collected {len(pairs)} (app_name, window_name) pairs from SQLite")

    if not pairs:
        print("No app/window pairs found, nothing to update.")
        return

    # Merge into AppNameManager (this will also ensure all apps exist in the JSON)
    app_name_manager.add_window_pairs(list(pairs))

    print(f"Updated app/window mapping written to: {app_name_manager.storage_path}")


if __name__ == "__main__":
    main()

