# core/storage/sqlite_storage.py
"""
SQLite 存储 - 用于 OCR fallback 和帧元数据

存储结构：
旧表（保持兼容）：
- frames: 帧元数据（时间、设备等）
- ocr_text: OCR 识别的文本

新表（视频chunk存储）：
- video_chunks: 全屏MP4视频块
- window_chunks: 窗口MP4视频块
- sub_frames: 窗口子帧
- frame_subframe_mapping: 帧与子帧的映射关系
"""

import sqlite3
import json
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
from PIL import Image
from config import config
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SQLiteStorage:
    """
    SQLite 存储
    
    表结构：
    
    === 旧表（保持兼容） ===
    1. frames: 帧元数据
       - frame_id (primary key)
       - timestamp
       - image_path (或 video_chunk引用)
       - device_name
       - metadata (JSON)
       - video_chunk_id, offset_index (新增，用于MP4存储)
    
    2. ocr_text: OCR 识别结果
       - id (primary key)
       - frame_id (foreign key, 可为NULL)
       - sub_frame_id (foreign key, 可为NULL)
       - text, text_json, ocr_engine, text_length, confidence
    
    === 新表（视频存储） ===
    3. video_chunks: 全屏视频块
       - id, file_path, monitor_id, device_name, fps, created_at
    
    4. window_chunks: 窗口视频块
       - id, file_path, app_name, window_name, monitor_id, fps, created_at
    
    5. sub_frames: 窗口子帧
       - sub_frame_id (primary key)
       - window_chunk_id, offset_index, timestamp
       - app_name, window_name, process_id, is_focused, image_hash
    
    6. frame_subframe_mapping: 帧与子帧映射
       - frame_id -> sub_frame_id (多对多)
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
        
        # ========== 旧表（保持兼容，扩展字段） ==========
        
        # 创建 frames 表（扩展支持video_chunk引用）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                frame_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                image_path TEXT NOT NULL,
                device_name TEXT,
                metadata TEXT,
                video_chunk_id INTEGER,
                offset_index INTEGER,
                monitor_id INTEGER DEFAULT 0,
                image_hash INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_chunk_id) REFERENCES video_chunks(id)
            )
        """)
        
        # 创建 ocr_text 表（扩展支持sub_frame_id）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ocr_text (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id TEXT,
                sub_frame_id TEXT,
                text TEXT NOT NULL,
                text_json TEXT,
                ocr_engine TEXT NOT NULL,
                text_length INTEGER NOT NULL,
                confidence REAL DEFAULT 0.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (frame_id) REFERENCES frames(frame_id),
                FOREIGN KEY (sub_frame_id) REFERENCES sub_frames(sub_frame_id)
            )
        """)
        
        # ========== 新表（视频存储） ==========
        
        # video_chunks: 全屏MP4视频块
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                monitor_id INTEGER NOT NULL DEFAULT 0,
                device_name TEXT,
                fps REAL DEFAULT 1.0,
                frame_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # window_chunks: 窗口MP4视频块
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS window_chunks (
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
        
        # sub_frames: 窗口子帧
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sub_frames (
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
        
        # frame_subframe_mapping: 帧与子帧映射
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS frame_subframe_mapping (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id TEXT NOT NULL,
                sub_frame_id TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (frame_id) REFERENCES frames(frame_id),
                FOREIGN KEY (sub_frame_id) REFERENCES sub_frames(sub_frame_id),
                UNIQUE(frame_id, sub_frame_id)
            )
        """)
        
        # ========== 索引 ==========
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_frames_timestamp 
            ON frames(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_frames_video_chunk
            ON frames(video_chunk_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ocr_frame_id 
            ON ocr_text(frame_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ocr_sub_frame_id
            ON ocr_text(sub_frame_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sub_frames_timestamp
            ON sub_frames(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sub_frames_window_chunk
            ON sub_frames(window_chunk_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sub_frames_app
            ON sub_frames(app_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_mapping_frame
            ON frame_subframe_mapping(frame_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_mapping_subframe
            ON frame_subframe_mapping(sub_frame_id)
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
            
            # 新增：统计视频chunks和子帧
            cursor.execute("SELECT COUNT(*) FROM video_chunks")
            total_video_chunks = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM window_chunks")
            total_window_chunks = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM sub_frames")
            total_sub_frames = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "total_frames": total_frames,
                "total_ocr_results": total_ocr,
                "total_text_length": total_text_length,
                "total_video_chunks": total_video_chunks,
                "total_window_chunks": total_window_chunks,
                "total_sub_frames": total_sub_frames,
                "db_path": str(self.db_path),
                "storage_mode": "sqlite"
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "total_frames": 0,
                "storage_mode": "sqlite"
            }
    
    # ========== 新增：视频Chunk相关方法 ==========
    
    def insert_video_chunk(
        self,
        file_path: str,
        monitor_id: int = 0,
        device_name: str = "default",
        fps: float = 1.0
    ) -> int:
        """
        插入视频chunk记录
        
        Args:
            file_path: MP4文件路径
            monitor_id: 显示器ID
            device_name: 设备名称
            fps: 帧率
            
        Returns:
            chunk_id
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO video_chunks (file_path, monitor_id, device_name, fps)
                VALUES (?, ?, ?, ?)
            """, (file_path, monitor_id, device_name, fps))
            
            chunk_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.debug(f"Inserted video chunk {chunk_id}: {file_path}")
            return chunk_id
            
        except Exception as e:
            logger.error(f"Failed to insert video chunk: {e}")
            return -1
    
    def insert_window_chunk(
        self,
        file_path: str,
        app_name: str,
        window_name: str,
        monitor_id: int = 0,
        fps: float = 1.0
    ) -> int:
        """
        插入窗口视频chunk记录
        
        Args:
            file_path: MP4文件路径
            app_name: 应用名称
            window_name: 窗口名称
            monitor_id: 显示器ID
            fps: 帧率
            
        Returns:
            chunk_id
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO window_chunks (file_path, app_name, window_name, monitor_id, fps)
                VALUES (?, ?, ?, ?, ?)
            """, (file_path, app_name, window_name, monitor_id, fps))
            
            chunk_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.debug(f"Inserted window chunk {chunk_id}: {app_name}/{window_name}")
            return chunk_id
            
        except Exception as e:
            logger.error(f"Failed to insert window chunk: {e}")
            return -1
    
    def update_chunk_frame_count(
        self,
        chunk_id: int,
        frame_count: int,
        chunk_type: str = "video"
    ) -> bool:
        """
        更新chunk的帧计数
        
        Args:
            chunk_id: chunk ID
            frame_count: 帧数量
            chunk_type: "video" 或 "window"
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            table = "video_chunks" if chunk_type == "video" else "window_chunks"
            cursor.execute(f"""
                UPDATE {table} SET frame_count = ? WHERE id = ?
            """, (frame_count, chunk_id))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update chunk frame count: {e}")
            return False
    
    # ========== 新增：帧与子帧相关方法 ==========
    
    def store_frame_with_video_ref(
        self,
        frame_id: str,
        timestamp: datetime,
        video_chunk_id: int,
        offset_index: int,
        monitor_id: int = 0,
        image_hash: int = 0,
        device_name: str = "default",
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        存储帧（使用video_chunk引用而非image_path）
        
        Args:
            frame_id: 帧ID
            timestamp: 时间戳
            video_chunk_id: 关联的视频chunk ID
            offset_index: 在视频中的帧索引
            monitor_id: 显示器ID
            image_hash: 图像哈希值
            device_name: 设备名称
            metadata: 其他元数据
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # image_path 设为空字符串或video chunk引用
            image_path = f"video_chunk:{video_chunk_id}:{offset_index}"
            
            cursor.execute("""
                INSERT OR REPLACE INTO frames 
                (frame_id, timestamp, image_path, device_name, metadata, 
                 video_chunk_id, offset_index, monitor_id, image_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                frame_id,
                timestamp.isoformat(),
                image_path,
                device_name,
                json.dumps(metadata) if metadata else "{}",
                video_chunk_id,
                offset_index,
                monitor_id,
                image_hash
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Stored frame {frame_id} with video ref")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store frame with video ref: {e}")
            return False
    
    def store_sub_frame(
        self,
        sub_frame_id: str,
        timestamp: datetime,
        window_chunk_id: int,
        offset_index: int,
        app_name: str,
        window_name: str,
        process_id: int = 0,
        is_focused: bool = False,
        image_hash: int = 0
    ) -> bool:
        """
        存储窗口子帧
        
        Args:
            sub_frame_id: 子帧ID
            timestamp: 时间戳
            window_chunk_id: 关联的窗口视频chunk ID
            offset_index: 在视频中的帧索引
            app_name: 应用名称
            window_name: 窗口名称
            process_id: 进程ID
            is_focused: 是否聚焦
            image_hash: 图像哈希值
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO sub_frames 
                (sub_frame_id, window_chunk_id, offset_index, timestamp,
                 app_name, window_name, process_id, is_focused, image_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sub_frame_id,
                window_chunk_id,
                offset_index,
                timestamp.isoformat(),
                app_name,
                window_name,
                process_id,
                1 if is_focused else 0,
                image_hash
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Stored sub_frame {sub_frame_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store sub_frame: {e}")
            return False
    
    def store_sub_frame_ocr(
        self,
        sub_frame_id: str,
        ocr_text: str,
        ocr_text_json: str = "",
        ocr_engine: str = "pytesseract",
        ocr_confidence: float = 0.0
    ) -> bool:
        """
        存储子帧的OCR结果
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if ocr_text:
                text_length = len(ocr_text)
                cursor.execute("""
                    INSERT INTO ocr_text 
                    (sub_frame_id, text, text_json, ocr_engine, text_length, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    sub_frame_id,
                    ocr_text,
                    ocr_text_json,
                    ocr_engine,
                    text_length,
                    ocr_confidence
                ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Stored OCR for sub_frame {sub_frame_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store sub_frame OCR: {e}")
            return False
    
    # ========== 新增：帧与子帧映射方法 ==========
    
    def add_frame_subframe_mapping(
        self,
        frame_id: str,
        sub_frame_id: str
    ) -> bool:
        """
        添加帧与子帧的映射关系
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR IGNORE INTO frame_subframe_mapping (frame_id, sub_frame_id)
                VALUES (?, ?)
            """, (frame_id, sub_frame_id))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to add frame-subframe mapping: {e}")
            return False
    
    def add_frame_subframe_mappings_batch(
        self,
        frame_id: str,
        sub_frame_ids: List[str]
    ) -> bool:
        """
        批量添加帧与子帧的映射关系
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            for sub_frame_id in sub_frame_ids:
                cursor.execute("""
                    INSERT OR IGNORE INTO frame_subframe_mapping (frame_id, sub_frame_id)
                    VALUES (?, ?)
                """, (frame_id, sub_frame_id))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to add frame-subframe mappings batch: {e}")
            return False
    
    def get_sub_frames_for_frame(self, frame_id: str) -> List[Dict]:
        """
        获取与帧关联的所有子帧
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT sf.*
                FROM sub_frames sf
                JOIN frame_subframe_mapping fsm ON sf.sub_frame_id = fsm.sub_frame_id
                WHERE fsm.frame_id = ?
                ORDER BY sf.app_name, sf.window_name
            """, (frame_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            results = []
            for row in rows:
                results.append({
                    "sub_frame_id": row["sub_frame_id"],
                    "window_chunk_id": row["window_chunk_id"],
                    "offset_index": row["offset_index"],
                    "timestamp": datetime.fromisoformat(row["timestamp"]),
                    "app_name": row["app_name"],
                    "window_name": row["window_name"],
                    "process_id": row["process_id"],
                    "is_focused": bool(row["is_focused"]),
                    "image_hash": row["image_hash"]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get sub_frames for frame: {e}")
            return []
    
    def get_frame_video_info(self, frame_id: str) -> Optional[Dict]:
        """
        获取帧的视频chunk信息（用于提取图像）
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT f.*, vc.file_path, vc.fps
                FROM frames f
                JOIN video_chunks vc ON f.video_chunk_id = vc.id
                WHERE f.frame_id = ?
            """, (frame_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "frame_id": row["frame_id"],
                    "video_chunk_id": row["video_chunk_id"],
                    "offset_index": row["offset_index"],
                    "file_path": row["file_path"],
                    "fps": row["fps"],
                    "timestamp": datetime.fromisoformat(row["timestamp"])
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get frame video info: {e}")
            return None
    
    def get_sub_frame_video_info(self, sub_frame_id: str) -> Optional[Dict]:
        """
        获取子帧的视频chunk信息（用于提取图像）
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT sf.*, wc.file_path, wc.fps
                FROM sub_frames sf
                JOIN window_chunks wc ON sf.window_chunk_id = wc.id
                WHERE sf.sub_frame_id = ?
            """, (sub_frame_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "sub_frame_id": row["sub_frame_id"],
                    "window_chunk_id": row["window_chunk_id"],
                    "offset_index": row["offset_index"],
                    "file_path": row["file_path"],
                    "fps": row["fps"],
                    "app_name": row["app_name"],
                    "window_name": row["window_name"],
                    "timestamp": datetime.fromisoformat(row["timestamp"])
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get sub_frame video info: {e}")
            return None
    
    def search_sub_frames_by_app(
        self,
        app_name: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        按应用名称搜索子帧
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT sf.*, o.text as ocr_text, o.confidence as ocr_confidence
                FROM sub_frames sf
                LEFT JOIN ocr_text o ON sf.sub_frame_id = o.sub_frame_id
                WHERE sf.app_name LIKE ?
                ORDER BY sf.timestamp DESC
                LIMIT ?
            """, (f"%{app_name}%", limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            results = []
            for row in rows:
                results.append({
                    "sub_frame_id": row["sub_frame_id"],
                    "timestamp": datetime.fromisoformat(row["timestamp"]),
                    "app_name": row["app_name"],
                    "window_name": row["window_name"],
                    "is_focused": bool(row["is_focused"]),
                    "ocr_text": row["ocr_text"] or "",
                    "ocr_confidence": row["ocr_confidence"] or 0.0
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search sub_frames by app: {e}")
            return []
