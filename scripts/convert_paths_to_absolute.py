#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将数据库（SQLite 和 LanceDB）中的相对路径转换为绝对路径。
"""

import os
import sys
import sqlite3
from pathlib import Path

# 将项目根目录加入路径
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from config import config
from utils.logger import setup_logger

logger = setup_logger("convert_paths_to_absolute")

def convert_sqlite_paths():
    db_path = Path(config.OCR_DB_PATH)
    if not db_path.exists():
        logger.warning(f"SQLite database not found at {db_path}")
        return

    logger.info(f"Updating SQLite paths in {db_path}...")
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 获取所有记录
        cursor.execute("SELECT frame_id, image_path FROM frames")
        rows = cursor.fetchall()
        
        updated_count = 0
        for row in rows:
            frame_id = row['frame_id']
            old_path = row['image_path']
            
            if not old_path:
                continue
                
            # 检查是否已经是绝对路径
            if os.path.isabs(old_path):
                continue
                
            # 转换为绝对路径
            # 假设相对路径是相对于项目根目录的
            new_path = str((project_root / old_path).resolve())
            
            cursor.execute(
                "UPDATE frames SET image_path = ? WHERE frame_id = ?",
                (new_path, frame_id)
            )
            updated_count += 1
            
        conn.commit()
        conn.close()
        logger.info(f"Successfully updated {updated_count} paths in SQLite.")
    except Exception as e:
        logger.error(f"Error updating SQLite paths: {e}")

def convert_lancedb_paths(db_path_str, table_name):
    db_path = Path(db_path_str)
    if not db_path.exists():
        logger.warning(f"LanceDB not found at {db_path}")
        return

    logger.info(f"Updating LanceDB paths in {db_path} (table: {table_name})...")
    try:
        import lancedb
        import pandas as pd
        
        db = lancedb.connect(db_path)
        if table_name not in db.table_names():
            logger.warning(f"Table {table_name} not found in {db_path}")
            return
            
        table = db.open_table(table_name)
        df = table.to_pandas()
        
        if 'image_path' not in df.columns:
            logger.warning(f"Column 'image_path' not found in table {table_name}")
            return
            
        def make_abs(path):
            if not path or os.path.isabs(path):
                return path
            return str((project_root / path).resolve())
            
        df['image_path'] = df['image_path'].apply(make_abs)
        
        # 覆盖原表
        db.create_table(table_name, data=df, mode="overwrite")
        logger.info(f"Successfully updated paths in LanceDB table {table_name}.")
    except ImportError:
        logger.error("lancedb or pandas not installed, skipping LanceDB update.")
    except Exception as e:
        logger.error(f"Error updating LanceDB paths: {e}")

def main():
    logger.info(f"Project root: {project_root}")
    
    # 1. 更新 SQLite (OCR)
    convert_sqlite_paths()
    
    # 2. 更新 LanceDB (Frames)
    convert_lancedb_paths(config.LANCEDB_PATH, "frames")
    
    # 3. 更新 TextDB (OCR Text)
    # 尝试可能的表名：ocr_texts (rebuild_text_index 使用) 或 ocr_text
    convert_lancedb_paths(config.TEXT_LANCEDB_PATH, "ocr_texts")
    convert_lancedb_paths(config.TEXT_LANCEDB_PATH, "ocr_text")

if __name__ == "__main__":
    main()
