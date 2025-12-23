# core/storage/sqlite_storage.py
"""
SQLite 存储 - 用于 OCR fallback

存储结构：
- frames: 帧元数据（时间、设备等）
- ocr_text: OCR 识别的文本
"""

import sqlite3
import json
from typing import List, Dict, Optional, Union
from datetime import datetime
from pathlib import Path
from PIL import Image
from config import config
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SQLiteStorage:
    """
    SQLite 存储（用于 OCR fallback）
    
    表结构：
    1. frames: 帧元数据
       - frame_id (primary key)
       - timestamp
       - image_path
       - device_name
       - metadata (JSON)
    
    2. ocr_text: OCR 识别结果
       - id (primary key)
       - frame_id (foreign key)
       - text
       - text_json
       - ocr_engine
       - text_length
       - confidence
    """
    
    def __init__(self, db_path: str = None):
        """
        初始化 SQLite 存储
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path or config.OCR_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Initializing SQLite storage at: {self.db_path}")
        
        # 创建表
        self._create_tables()
        
        logger.debug("SQLite storage initialized successfully")
    
    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 使用 Row 工厂，方便字典访问
        return conn
    
    def _create_tables(self):
        """创建数据库表"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 创建 frames 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                frame_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                image_path TEXT NOT NULL,
                device_name TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建 ocr_text 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ocr_text (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id TEXT NOT NULL,
                text TEXT NOT NULL,
                text_json TEXT,
                ocr_engine TEXT NOT NULL,
                text_length INTEGER NOT NULL,
                confidence REAL DEFAULT 0.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (frame_id) REFERENCES frames(frame_id)
            )
        """)
        
        # 创建索引（加速查询）
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_frames_timestamp 
            ON frames(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ocr_frame_id 
            ON ocr_text(frame_id)
        """)
        
        # 创建全文搜索索引（FTS5）
        try:
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS ocr_text_fts 
                USING fts5(frame_id, text, content=ocr_text, content_rowid=id)
            """)
            
            # 创建触发器保持 FTS 索引同步
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS ocr_text_ai AFTER INSERT ON ocr_text BEGIN
                    INSERT INTO ocr_text_fts(rowid, frame_id, text) 
                    VALUES (new.id, new.frame_id, new.text);
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS ocr_text_ad AFTER DELETE ON ocr_text BEGIN
                    INSERT INTO ocr_text_fts(ocr_text_fts, rowid, frame_id, text) 
                    VALUES('delete', old.id, old.frame_id, old.text);
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS ocr_text_au AFTER UPDATE ON ocr_text BEGIN
                    INSERT INTO ocr_text_fts(ocr_text_fts, rowid, frame_id, text) 
                    VALUES('delete', old.id, old.frame_id, old.text);
                    INSERT INTO ocr_text_fts(rowid, frame_id, text) 
                    VALUES (new.id, new.frame_id, new.text);
                END
            """)
            
            logger.debug("FTS5 full-text search enabled")
            
        except sqlite3.OperationalError as e:
            logger.warning(f"FTS5 not available: {e}, using regular LIKE search")
        
        conn.commit()
        conn.close()
        
        logger.debug("Database tables created/verified")
    
    def store_frame_with_ocr(
        self,
        frame_id: str,
        timestamp: datetime,
        image_path: str,
        ocr_text: str,
        ocr_text_json: str = "",
        ocr_engine: str = "pytesseract",
        ocr_confidence: float = 0.0,
        device_name: str = "default",
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        存储帧和 OCR 结果
        
        Args:
            frame_id: 帧ID
            timestamp: 时间戳
            image_path: 图片路径
            ocr_text: OCR 识别的文本
            ocr_text_json: OCR 详细结果（JSON）
            ocr_engine: OCR 引擎名称
            ocr_confidence: OCR 置信度
            device_name: 设备名称
            metadata: 其他元数据
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 1. 插入 frame
            cursor.execute("""
                INSERT OR REPLACE INTO frames 
                (frame_id, timestamp, image_path, device_name, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                frame_id,
                timestamp.isoformat(),
                str(image_path),
                device_name,
                json.dumps(metadata) if metadata else "{}"
            ))
            
            # 2. 插入 OCR 结果（如果有文本）
            if ocr_text:
                text_length = len(ocr_text)
                cursor.execute("""
                    INSERT INTO ocr_text 
                    (frame_id, text, text_json, ocr_engine, text_length, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    frame_id,
                    ocr_text,
                    ocr_text_json,
                    ocr_engine,
                    text_length,
                    ocr_confidence
                ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Stored frame {frame_id} with OCR (text_length={len(ocr_text)})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store frame with OCR: {e}")
            return False
    
    def search_by_text(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.0
    ) -> List[Dict]:
        """
        通过文本搜索帧（OCR fallback）
        
        Args:
            query: 搜索关键词
            limit: 返回结果数量
            min_confidence: 最小置信度阈值
            
        Returns:
            匹配的帧列表
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 尝试使用 FTS5 全文搜索
            try:
                cursor.execute("""
                    SELECT 
                        f.frame_id,
                        f.timestamp,
                        f.image_path,
                        f.device_name,
                        f.metadata,
                        o.text,
                        o.text_length,
                        o.confidence,
                        o.ocr_engine
                    FROM ocr_text_fts fts
                    JOIN ocr_text o ON fts.rowid = o.id
                    JOIN frames f ON o.frame_id = f.frame_id
                    WHERE ocr_text_fts MATCH ?
                    AND o.confidence >= ?
                    ORDER BY rank
                    LIMIT ?
                """, (query, min_confidence, limit))
                
            except sqlite3.OperationalError:
                # FTS5 不可用，使用 LIKE 搜索
                cursor.execute("""
                    SELECT 
                        f.frame_id,
                        f.timestamp,
                        f.image_path,
                        f.device_name,
                        f.metadata,
                        o.text,
                        o.text_length,
                        o.confidence,
                        o.ocr_engine
                    FROM ocr_text o
                    JOIN frames f ON o.frame_id = f.frame_id
                    WHERE o.text LIKE ?
                    AND o.confidence >= ?
                    ORDER BY f.timestamp DESC
                    LIMIT ?
                """, (f"%{query}%", min_confidence, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            # 转换为字典列表
            results = []
            for row in rows:
                results.append({
                    "frame_id": row["frame_id"],
                    "timestamp": datetime.fromisoformat(row["timestamp"]),
                    "image_path": row["image_path"],
                    "device_name": row["device_name"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "ocr_text": row["text"],
                    "ocr_confidence": row["confidence"],
                    "ocr_engine": row["ocr_engine"]
                })
            
            logger.debug(f"Text search '{query}' found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []

    def search_text(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.0
    ) -> List[Dict]:
        """
        兼容接口：保持与调用方的 search_text 命名一致
        """
        return self.search_by_text(query=query, limit=limit, min_confidence=min_confidence)
    
    def get_frames_batch(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        分批获取帧（用于重建索引）
        
        Args:
            limit: 每批数量
            offset: 偏移量
            
        Returns:
            帧列表
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    f.frame_id,
                    f.timestamp,
                    f.image_path,
                    f.device_name,
                    f.metadata,
                    o.text as ocr_text,
                    o.confidence as ocr_confidence
                FROM frames f
                LEFT JOIN ocr_text o ON f.frame_id = o.frame_id
                ORDER BY f.timestamp ASC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            rows = cursor.fetchall()
            conn.close()
            
            results = []
            for row in rows:
                results.append({
                    "frame_id": row["frame_id"],
                    "timestamp": datetime.fromisoformat(row["timestamp"]),
                    "image_path": row["image_path"],
                    "device_name": row["device_name"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "ocr_text": row["ocr_text"] or "",
                    "ocr_confidence": row["ocr_confidence"] or 0.0
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get frames batch: {e}")
            return []
    
    def get_recent_frames(self, limit: int = 10) -> List[Dict]:
        """
        获取最近的帧
        
        Args:
            limit: 返回数量
            
        Returns:
            帧列表
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    f.frame_id,
                    f.timestamp,
                    f.image_path,
                    f.device_name,
                    f.metadata,
                    o.text as ocr_text,
                    o.confidence as ocr_confidence
                FROM frames f
                LEFT JOIN ocr_text o ON f.frame_id = o.frame_id
                ORDER BY f.timestamp DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            results = []
            for row in rows:
                results.append({
                    "frame_id": row["frame_id"],
                    "timestamp": datetime.fromisoformat(row["timestamp"]),
                    "image_path": row["image_path"],
                    "device_name": row["device_name"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "ocr_text": row["ocr_text"] or "",
                    "ocr_confidence": row["ocr_confidence"] or 0.0
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get recent frames: {e}")
            return []
    
    def get_frames_in_timerange(
        self,
        start_time: Union[datetime, str],
        end_time: Union[datetime, str],
        limit: int = 10000
    ) -> List[Dict]:
        """
        获取时间范围内的帧（直接在 SQL 中按时间范围查询，更高效）
        
        Args:
            start_time: 开始时间 (datetime 或 ISO 字符串)
            end_time: 结束时间 (datetime 或 ISO 字符串)
            limit: 最大返回数量（防止内存溢出）
            
        Returns:
            帧列表
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 统一转换为字符串
            start_str = start_time.isoformat() if isinstance(start_time, datetime) else start_time
            end_str = end_time.isoformat() if isinstance(end_time, datetime) else end_time
            
            cursor.execute("""
                SELECT 
                    f.frame_id,
                    f.timestamp,
                    f.image_path,
                    f.device_name,
                    f.metadata,
                    o.text as ocr_text,
                    o.confidence as ocr_confidence
                FROM frames f
                LEFT JOIN ocr_text o ON f.frame_id = o.frame_id
                WHERE f.timestamp >= ? AND f.timestamp < ?
                ORDER BY f.timestamp ASC
                LIMIT ?
            """, (start_str, end_str, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            results = []
            for row in rows:
                results.append({
                    "frame_id": row["frame_id"],
                    "timestamp": datetime.fromisoformat(row["timestamp"]),
                    "image_path": row["image_path"],
                    "device_name": row["device_name"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "ocr_text": row["ocr_text"] or "",
                    "ocr_confidence": row["ocr_confidence"] or 0.0
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get frames in time range: {e}")
            return []
    
    def get_earliest_frame(self) -> Optional[Dict]:
        """获取最早的帧"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    f.frame_id,
                    f.timestamp,
                    f.image_path
                FROM frames f
                ORDER BY f.timestamp ASC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "frame_id": row["frame_id"],
                    "timestamp": datetime.fromisoformat(row["timestamp"]),
                    "image_path": row["image_path"]
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get earliest frame: {e}")
            return None
    
    def get_latest_frame(self) -> Optional[Dict]:
        """获取最新的帧"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    f.frame_id,
                    f.timestamp,
                    f.image_path
                FROM frames f
                ORDER BY f.timestamp DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "frame_id": row["frame_id"],
                    "timestamp": datetime.fromisoformat(row["timestamp"]),
                    "image_path": row["image_path"]
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get latest frame: {e}")
            return None
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM frames")
            total_frames = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM ocr_text")
            total_ocr = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(text_length) FROM ocr_text")
            total_text_length = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                "total_frames": total_frames,
                "total_ocr_results": total_ocr,
                "total_text_length": total_text_length,
                "db_path": str(self.db_path),
                "storage_mode": "sqlite"
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "total_frames": 0,
                "storage_mode": "sqlite"
            }

