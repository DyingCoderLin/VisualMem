#!/usr/bin/env python3
"""
Demo: 5秒截屏存储到 demo.db

这个 demo 验证完整的截屏 -> OCR -> Embedding -> 存储流程：
1. 使用 screencap_rs 截取全屏和窗口
2. 对全屏做 OCR（并计时）
3. 对全屏和窗口做 Embedding（并计时）
4. 存储到 demo.db (SQLite) - 元数据和 OCR
5. 存储到 LanceDB - Embedding 向量（支持向量搜索）
6. 将临时图片压缩成 MP4

运行方式:
    python demo_capture_storage.py
"""

import os
import sys
import time
import sqlite3
import json
import threading
import queue
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import setup_logger
from core.preprocess.frame_diff import FrameDiffDetector
from utils.data_models import ScreenObject, WindowFrame
from core.capture.window_capturer import calculate_image_hash
import io

logger = setup_logger(__name__)

# ============================================================
# 配置
# ============================================================
DEMO_DB_PATH = PROJECT_ROOT / "demo_output" / "demo.db"
DEMO_TEMP_DIR = PROJECT_ROOT / "demo_output" / "temp_frames"
DEMO_VIDEO_DIR = PROJECT_ROOT / "demo_output" / "videos"
CAPTURE_DURATION = 5  # 秒
CAPTURE_INTERVAL = 1.0  # 每秒截一次
VIDEO_FPS = 1.0
BATCH_SIZE = 60  # 每60帧压缩一次（demo中不会触发，但保持一致）


# ============================================================
# 数据库初始化
# ============================================================
def init_demo_db(db_path: Path) -> sqlite3.Connection:
    """初始化 demo 数据库"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 删除旧的数据库
    if db_path.exists():
        db_path.unlink()
        logger.info(f"Deleted old database: {db_path}")
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 创建表 (与 SQLiteStorage 主 schema 保持一致)
    cursor.executescript("""
        -- 全屏帧表
        CREATE TABLE IF NOT EXISTS frames (
            frame_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            image_path TEXT NOT NULL,
            device_name TEXT,
            metadata TEXT,
            monitor_id INTEGER DEFAULT 0,
            video_chunk_id INTEGER,
            offset_index INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            app_name TEXT,
            window_name TEXT,
            FOREIGN KEY (video_chunk_id) REFERENCES video_chunks(id)
        );
        
        -- OCR 结果 (统一表：通过 frame_id 或 sub_frame_id 关联)
        CREATE TABLE IF NOT EXISTS ocr_text (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id TEXT,
            sub_frame_id TEXT,
            text TEXT NOT NULL,
            text_json TEXT,
            ocr_engine TEXT NOT NULL DEFAULT 'pytesseract',
            text_length INTEGER NOT NULL DEFAULT 0,
            confidence REAL DEFAULT 0.0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (frame_id) REFERENCES frames(frame_id),
            FOREIGN KEY (sub_frame_id) REFERENCES sub_frames(sub_frame_id)
        );
        
        -- 视频 chunks (全屏)
        CREATE TABLE IF NOT EXISTS video_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            monitor_id INTEGER NOT NULL DEFAULT 0,
            device_name TEXT,
            fps REAL DEFAULT 1.0,
            frame_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 窗口 chunks
        CREATE TABLE IF NOT EXISTS window_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            app_name TEXT NOT NULL,
            window_name TEXT NOT NULL,
            monitor_id INTEGER NOT NULL DEFAULT 0,
            fps REAL DEFAULT 1.0,
            frame_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 子帧表 (窗口截图)
        CREATE TABLE IF NOT EXISTS sub_frames (
            sub_frame_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            app_name TEXT NOT NULL,
            window_name TEXT NOT NULL,
            window_chunk_id INTEGER,
            offset_index INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (window_chunk_id) REFERENCES window_chunks(id)
        );
        
        -- 帧与子帧的映射关系
        CREATE TABLE IF NOT EXISTS frame_subframe_mapping (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id TEXT NOT NULL,
            sub_frame_id TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (frame_id) REFERENCES frames(frame_id),
            FOREIGN KEY (sub_frame_id) REFERENCES sub_frames(sub_frame_id),
            UNIQUE(frame_id, sub_frame_id)
        );
        
        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_frames_timestamp ON frames(timestamp);
        CREATE INDEX IF NOT EXISTS idx_frames_video_chunk ON frames(video_chunk_id);
        CREATE INDEX IF NOT EXISTS idx_ocr_frame_id ON ocr_text(frame_id);
        CREATE INDEX IF NOT EXISTS idx_ocr_sub_frame_id ON ocr_text(sub_frame_id);
        CREATE INDEX IF NOT EXISTS idx_sub_frames_timestamp ON sub_frames(timestamp);
        CREATE INDEX IF NOT EXISTS idx_sub_frames_window_chunk ON sub_frames(window_chunk_id);
        CREATE INDEX IF NOT EXISTS idx_sub_frames_app ON sub_frames(app_name);
        CREATE INDEX IF NOT EXISTS idx_mapping_frame ON frame_subframe_mapping(frame_id);
        CREATE INDEX IF NOT EXISTS idx_mapping_subframe ON frame_subframe_mapping(sub_frame_id);
    """)
    
    conn.commit()
    logger.info(f"Database initialized: {db_path}")
    return conn


# ============================================================
# 截屏模块
# ============================================================
def capture_screen_and_windows():
    """
    使用 screencap_rs 截取全屏和所有窗口（优化版：只返回原始bytes，不进行图片转换）
    
    Returns:
        dict: {
            "full_screen_bytes": bytes,  # PNG bytes，不转换
            "monitor_id": int,
            "windows": [
                {
                    "app_name": str,
                    "window_name": str,
                    "window_bytes": bytes  # PNG bytes，不转换
                },
                ...
            ]
        }
    """
    try:
        from screencap_rs import screencap_rs

        start_capture_time = time.time()
        
        # 截取全屏和窗口（不过滤系统窗口，包含最小化窗口以便测试）
        screen_result, window_results = screencap_rs.capture_screen_with_windows(
            monitor_id=None,  # 使用默认显示器
            include_minimized=True,  # 包含最小化的窗口
            filter_system=False  # 不过滤系统窗口，以便看到所有窗口
        )
        end_capture_time = time.time()
        logger.info(f"Real Capture time: {end_capture_time - start_capture_time:.2f}s")
        
        window_count = len(window_results) if window_results else 0
        logger.info(f"Captured {window_count} windows")
        if window_count > 0:
            logger.info(f"Window details: {[(w.info.app_name if hasattr(w, 'info') else 'N/A', w.info.title if hasattr(w, 'info') else 'N/A') for w in window_results[:5]]}")
        
        if screen_result is None:
            logger.error("Failed to capture screen")
            return None
        
        # 获取全屏图片 PNG bytes（不转换，延迟到异步处理）
        screen_bytes = screen_result.get_image_bytes()
        
        # 获取 monitor_id
        monitor_id = screen_result.monitor.id if screen_result.monitor else 0
        
        # 处理窗口（只获取bytes和元数据，不转换图片）
        windows = []
        for i, w in enumerate(window_results):
            try:
                # CapturedWindow 对象有 get_image_bytes() 方法
                window_bytes = w.get_image_bytes()
                
                if window_bytes and len(window_bytes) > 0:
                    # 获取窗口信息：CapturedWindow 有 info 属性，类型为 WindowInfo
                    # WindowInfo 包含: id, app_name, title, x, y, width, height, is_minimized
                    window_info = w.info
                    app_name = getattr(window_info, 'app_name', 'Unknown') or 'Unknown'
                    window_name = getattr(window_info, 'title', 'Unknown') or 'Unknown'
                    window_id = getattr(window_info, 'id', 0) or 0
                    
                    windows.append({
                        "app_name": app_name,
                        "window_name": window_name,
                        "window_bytes": window_bytes  # 只保存bytes，不转换
                    })
                    logger.debug(f"Processed window {i}: {app_name}/{window_name} (id={window_id})")
            except Exception as e:
                logger.warning(f"Failed to process window {i}: {e}", exc_info=True)
        
        return {
            "full_screen_bytes": screen_bytes,  # 返回原始bytes
            "monitor_id": monitor_id,
            "windows": windows
        }
        
    except ImportError as e:
        logger.error(f"screencap_rs not available: {e}, using fallback")
        # Fallback: 创建一个测试图片
        test_image = Image.new('RGB', (1920, 1080), color='blue')
        return {
            "full_screen": test_image,
            "monitor_id": 0,
            "windows": []
        }
    except Exception as e:
        logger.error(f"Capture failed: {e}")
        # Fallback
        test_image = Image.new('RGB', (1920, 1080), color='red')
        return {
            "full_screen": test_image,
            "monitor_id": 0,
            "windows": []
        }


# ============================================================
# OCR 模块
# ============================================================
def perform_ocr(image: Image.Image) -> tuple:
    """
    对图片进行 OCR
    
    Returns:
        (text, text_json, confidence)
    """
    try:
        import pytesseract
        
        # 简单 OCR
        text = pytesseract.image_to_string(image, lang='chi_sim+eng')
        
        # 详细 OCR (带位置信息)
        data = pytesseract.image_to_data(image, lang='chi_sim+eng', output_type=pytesseract.Output.DICT)
        
        # 计算平均置信度
        confidences = [c for c in data['conf'] if c > 0]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        # 构建 JSON 结果
        ocr_items = []
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                ocr_items.append({
                    "text": data['text'][i],
                    "x": data['left'][i],
                    "y": data['top'][i],
                    "width": data['width'][i],
                    "height": data['height'][i],
                    "conf": data['conf'][i]
                })
        
        text_json = json.dumps(ocr_items, ensure_ascii=False)
        
        return text.strip(), text_json, avg_conf
        
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return "", "[]", 0.0


# ============================================================
# Embedding 模块
# ============================================================
_encoder = None

def get_encoder():
    """获取或创建 encoder 单例"""
    global _encoder
    if _encoder is None:
        from core.encoder import create_encoder
        from config import config
        logger.info(f"Loading encoder: {config.EMBEDDING_MODEL}")
        _encoder = create_encoder(model_name=config.EMBEDDING_MODEL)
        logger.info(f"Encoder loaded. Dim: {_encoder.get_embedding_dim()}")
    return _encoder


def perform_embedding(image: Image.Image) -> list:
    """
    对图片进行 Embedding
    
    Returns:
        embedding vector (list of floats)
    """
    try:
        encoder = get_encoder()
        embedding = encoder.encode_image(image)
        return embedding
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return []


# ============================================================
# 异步批量处理器（OCR + Embedding）
# ============================================================
class AsyncProcessor:
    """异步批量处理 OCR 和 embedding，不阻塞截屏流程"""
    
    def __init__(self, storage: 'DemoStorage', db_path: str = None, batch_size: int = 8):
        """
        初始化异步处理器
        
        Args:
            storage: DemoStorage 实例
            db_path: 数据库文件路径（用于在后台线程创建独立连接）
            batch_size: 批量处理大小
        """
        self.storage = storage
        self.batch_size = batch_size
        self.task_queue = queue.Queue()
        self.encoder = None
        self.running = False
        self.thread = None
        
        # 数据库路径（用于在后台线程创建独立连接）
        if db_path:
            self.db_path = db_path
        elif hasattr(storage, 'conn') and storage.conn:
            # 尝试从连接对象获取数据库路径
            try:
                result = storage.conn.execute("PRAGMA database_list").fetchone()
                if result:
                    self.db_path = result[2]
                else:
                    self.db_path = None
            except Exception:
                self.db_path = None
        else:
            self.db_path = None
        
        # 统计信息
        self.total_processed = 0
        self.total_ocr_time = 0.0
        self.total_embedding_time = 0.0
        self.total_sub_embedding_time = 0.0
    
    def _get_db_conn(self):
        """在后台线程中创建独立的数据库连接"""
        if not self.db_path:
            return None
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    def start(self):
        """启动后台处理线程"""
        if self.running:
            return
        
        # 预加载 encoder
        self.encoder = get_encoder()
        
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        logger.info("AsyncProcessor started (OCR + Embedding)")
    
    def stop(self):
        """停止后台处理线程"""
        self.running = False
        # 放入停止信号
        self.task_queue.put(None)
        if self.thread:
            self.thread.join(timeout=10)
        logger.info("AsyncProcessor stopped")
    
    def add_full_screen_task(
        self,
        frame_id: str,
        timestamp: datetime,
        image_bytes: bytes,  # PNG bytes，不转换
        monitor_id: int,
        image_path: str
    ):
        """添加全屏帧的处理任务（OCR + Embedding）"""
        task = {
            "type": "full_screen",
            "frame_id": frame_id,
            "timestamp": timestamp,
            "image_bytes": image_bytes,  # 原始PNG bytes
            "monitor_id": monitor_id,
            "image_path": image_path
        }
        self.task_queue.put(task)
    
    def add_window_task(
        self,
        sub_frame_id: str,
        parent_frame_id: str,
        timestamp: datetime,
        window_bytes: bytes,  # PNG bytes，不转换
        app_name: str,
        window_name: str,
        image_path: str
    ):
        """添加窗口帧的处理任务（Embedding）"""
        task = {
            "type": "window",
            "sub_frame_id": sub_frame_id,
            "parent_frame_id": parent_frame_id,
            "timestamp": timestamp,
            "window_bytes": window_bytes,  # 原始PNG bytes
            "app_name": app_name,
            "window_name": window_name,
            "image_path": image_path
        }
        self.task_queue.put(task)
    
    def _process_loop(self):
        """后台处理循环"""
        batch = []
        
        while self.running:
            try:
                # 从队列获取任务，设置超时以便定期处理批次
                try:
                    task = self.task_queue.get(timeout=0.5)
                except queue.Empty:
                    # 如果队列为空，处理当前批次
                    if batch:
                        self._process_batch(batch)
                        batch = []
                    continue
                
                # 收到停止信号
                if task is None:
                    break
                
                batch.append(task)
                
                # 当批次达到指定大小或队列为空时，处理批次
                if len(batch) >= self.batch_size:
                    self._process_batch(batch)
                    batch = []
                    
            except Exception as e:
                logger.error(f"Error in embedding processor loop: {e}", exc_info=True)
        
        # 处理剩余的批次
        if batch:
            self._process_batch(batch)
    
    def _process_batch(self, batch: List[Dict[str, Any]]):
        """批量处理 embedding 任务"""
        if not batch:
            return
        
        # 分离全屏和窗口任务
        full_screen_tasks = [t for t in batch if t["type"] == "full_screen"]
        window_tasks = [t for t in batch if t["type"] == "window"]
        
        # 批量处理全屏帧
        if full_screen_tasks:
            self._process_full_screen_batch(full_screen_tasks)
        
        # 批量处理窗口帧
        if window_tasks:
            self._process_window_batch(window_tasks)
    
    def _process_full_screen_batch(self, tasks: List[Dict[str, Any]]):
        """批量处理全屏帧：图片转换 + OCR + Embedding"""
        try:
            import io
            
            # 1. 转换图片（从PNG bytes到PIL Image）
            convert_start = time.time()
            images = []
            for task in tasks:
                try:
                    image_bytes = task["image_bytes"]
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    images.append(image)
                except Exception as e:
                    logger.error(f"Failed to convert image for frame {task['frame_id']}: {e}")
                    images.append(None)
            convert_time = time.time() - convert_start
            logger.debug(f"Converted {len(tasks)} images in {convert_time:.2f}s")
            
            # 2. OCR处理（逐个处理，因为OCR通常不支持批量）
            ocr_start = time.time()
            ocr_results = []
            for task, image in zip(tasks, images):
                if image is None:
                    ocr_results.append(("", "[]", 0.0))
                    continue
                try:
                    ocr_text, ocr_json, ocr_conf = perform_ocr(image)
                    ocr_results.append((ocr_text, ocr_json, ocr_conf))
                except Exception as e:
                    logger.error(f"OCR failed for frame {task['frame_id']}: {e}")
                    ocr_results.append(("", "[]", 0.0))
            ocr_time = time.time() - ocr_start
            self.total_ocr_time += ocr_time
            
            # 3. 更新SQLite中的OCR结果（使用后台线程的独立连接）
            db_conn = self._get_db_conn()
            if db_conn:
                try:
                    cursor = db_conn.cursor()
                    for task, (ocr_text, ocr_json, ocr_conf) in zip(tasks, ocr_results):
                        try:
                            cursor.execute("""
                                UPDATE ocr_text 
                                SET text = ?, text_json = ?, confidence = ?
                                WHERE frame_id = ?
                            """, (ocr_text, ocr_json, ocr_conf, task["frame_id"]))
                        except Exception as e:
                            logger.error(f"Failed to update OCR for frame {task['frame_id']}: {e}")
                    db_conn.commit()
                finally:
                    db_conn.close()
            else:
                logger.warning("Database connection not available, skipping OCR update")
            
            # 4. 批量Embedding（只处理有效的图片）
            valid_images = [img for img in images if img is not None]
            valid_tasks = [task for task, img in zip(tasks, images) if img is not None]
            
            if valid_images:
                embedding_start = time.time()
                if hasattr(self.encoder, 'encode_image_batch'):
                    embeddings = self.encoder.encode_image_batch(valid_images)
                else:
                    # 如果不支持批量，逐个处理
                    embeddings = [self.encoder.encode_image(img) for img in valid_images]
                embedding_time = time.time() - embedding_start
                self.total_embedding_time += embedding_time
                self.total_processed += len(valid_tasks)
                
                # 5. 更新LanceDB和缓冲区
                # 获取对应的OCR结果
                valid_ocr_results = [(ocr_results[i]) for i, img in enumerate(images) if img is not None]
                
                for task, image, embedding, (ocr_text, ocr_json, ocr_conf) in zip(
                    valid_tasks, valid_images, embeddings, valid_ocr_results
                ):
                    try:
                        # 更新LanceDB
                        if self.storage.vector_storage:
                            frame_data = {
                                "frame_id": task["frame_id"],
                                "timestamp": task["timestamp"],
                                "image": image,
                                "embedding": embedding,
                                "ocr_text": ocr_text,
                                "image_path": task["image_path"],
                                "device_name": f"monitor_{task['monitor_id']}",
                                "metadata": {"monitor_id": task["monitor_id"]}
                            }
                            success = self.storage.vector_storage.store_frames_batch([frame_data])
                            if success:
                                logger.debug(f"Stored frame {task['frame_id']} to LanceDB (async)")
                            else:
                                logger.warning(f"Failed to store frame {task['frame_id']} to LanceDB")
                        
                        # 更新缓冲区中的embedding
                        for i, item in enumerate(self.storage.full_screen_buffer):
                            if len(item) >= 1 and item[0] == task["frame_id"]:
                                if len(item) >= 4:
                                    self.storage.full_screen_buffer[i] = (
                                        item[0], item[1], item[2], item[3], embedding
                                    )
                                break
                    except Exception as e:
                        logger.error(f"Error updating frame {task['frame_id']}: {e}")
                
                logger.debug(f"Processed {len(valid_tasks)} full_screen frames: OCR={ocr_time:.2f}s, Embedding={embedding_time:.2f}s (batch)")
            
        except Exception as e:
            logger.error(f"Error processing full_screen batch: {e}", exc_info=True)
    
    def _process_window_batch(self, tasks: List[Dict[str, Any]]):
        """批量处理窗口帧：图片转换 + Embedding"""
        try:
            import io
            
            # 1. 转换图片（从PNG bytes到PIL Image）
            convert_start = time.time()
            images = []
            for task in tasks:
                try:
                    window_bytes = task["window_bytes"]
                    image = Image.open(io.BytesIO(window_bytes)).convert('RGB')
                    images.append(image)
                except Exception as e:
                    logger.error(f"Failed to convert image for sub_frame {task['sub_frame_id']}: {e}")
                    images.append(None)
            convert_time = time.time() - convert_start
            logger.debug(f"Converted {len(tasks)} window images in {convert_time:.2f}s")
            
            # 2. 批量Embedding（只处理有效的图片）
            valid_images = [img for img in images if img is not None]
            valid_tasks = [task for task, img in zip(tasks, images) if img is not None]
            
            if valid_images:
                embedding_start = time.time()
                if hasattr(self.encoder, 'encode_image_batch'):
                    embeddings = self.encoder.encode_image_batch(valid_images)
                else:
                    # 如果不支持批量，逐个处理
                    embeddings = [self.encoder.encode_image(img) for img in valid_images]
                embedding_time = time.time() - embedding_start
                self.total_sub_embedding_time += embedding_time
                self.total_processed += len(valid_tasks)
                
                # 3. 更新LanceDB
                for task, image, embedding in zip(valid_tasks, valid_images, embeddings):
                    try:
                        if self.storage.vector_storage:
                            frame_data = {
                                "frame_id": task["sub_frame_id"],
                                "timestamp": task["timestamp"],
                                "image": image,
                                "embedding": embedding,
                                "ocr_text": "",
                                "image_path": task["image_path"],
                                "device_name": f"{task['app_name']}/{task['window_name']}",
                                "metadata": {
                                    "app_name": task["app_name"],
                                    "window_name": task["window_name"],
                                    "parent_frame_id": task["parent_frame_id"],
                                    "is_sub_frame": True
                                }
                            }
                            success = self.storage.vector_storage.store_frames_batch([frame_data])
                            if success:
                                logger.debug(f"Stored sub_frame {task['sub_frame_id']} to LanceDB (async)")
                            else:
                                logger.warning(f"Failed to store sub_frame {task['sub_frame_id']} to LanceDB")
                    except Exception as e:
                        logger.error(f"Error updating sub_frame {task['sub_frame_id']}: {e}")
                
                logger.debug(f"Processed {len(valid_tasks)} window frames: Embedding={embedding_time:.2f}s (batch)")
            
        except Exception as e:
            logger.error(f"Error processing window batch: {e}", exc_info=True)


# ============================================================
# 存储模块
# ============================================================
class DemoStorage:
    """Demo 存储管理器"""
    
    def __init__(self, conn: sqlite3.Connection, temp_dir: Path, video_dir: Path, lancedb_path: Path = None):
        self.conn = conn
        self.temp_dir = temp_dir
        self.video_dir = video_dir
        
        # 创建目录
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 LanceDB（用于存储 embedding）
        self.lancedb_path = lancedb_path or (PROJECT_ROOT / "demo_output" / "demo_lancedb")
        self.lancedb_path.mkdir(parents=True, exist_ok=True)
        
        try:
            from core.storage.lancedb_storage import LanceDBStorage
            from config import config
            
            # 根据模型名称确定 embedding 维度
            model_name = config.EMBEDDING_MODEL.lower()
            if "qwen" in model_name:
                embedding_dim = 2048  # Qwen: 2048
            else:
                embedding_dim = 1024  # CLIP: 1024
            
            self.vector_storage = LanceDBStorage(
                db_path=str(self.lancedb_path),
                embedding_dim=embedding_dim
            )
            logger.info(f"LanceDB initialized at: {self.lancedb_path} (embedding_dim={embedding_dim}, model={config.EMBEDDING_MODEL})")
        except Exception as e:
            logger.error(f"Failed to initialize LanceDB: {e}")
            self.vector_storage = None
        
        # 临时帧缓冲
        self.full_screen_buffer = []  # [(frame_id, timestamp, image_path, monitor_id, embedding)]
        self.window_buffers = {}  # {window_key: [(sub_frame_id, timestamp, image_path, ..., embedding)]}
    
    def store_full_screen_frame_bytes(
        self,
        frame_id: str,
        timestamp: datetime,
        image_bytes: bytes,
        monitor_id: int
    ) -> str:
        """存储全屏帧（从原始bytes，不转换图片）"""
        # 1. 直接保存PNG bytes（最快）
        temp_path = self.temp_dir / "full_screen" / f"monitor_{monitor_id}"
        temp_path.mkdir(parents=True, exist_ok=True)
        
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        image_path = (temp_path / f"{ts_str}.png").resolve()
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        
        # 2. 存入 SQLite 数据库（OCR结果稍后异步更新）
        cursor = self.conn.cursor()
        
        # 插入 frame
        cursor.execute("""
            INSERT INTO frames (frame_id, timestamp, image_path, monitor_id)
            VALUES (?, ?, ?, ?)
        """, (frame_id, timestamp.isoformat(), str(image_path), monitor_id))
        
        # 插入空的OCR结果占位符（稍后异步更新）
        cursor.execute("""
            INSERT INTO ocr_text (frame_id, text, text_json, ocr_engine, text_length, confidence)
            VALUES (?, ?, ?, 'pytesseract', 0, ?)
        """, (frame_id, "", "[]", 0.0))
        
        self.conn.commit()
        
        # 3. 添加到缓冲区（embedding稍后异步更新）
        self.full_screen_buffer.append((frame_id, timestamp, str(image_path), monitor_id, None))
        
        logger.info(f"Stored full_screen frame (bytes): {frame_id}, buffer_size={len(self.full_screen_buffer)}")
        return str(image_path)
    
    def store_full_screen_frame(
        self,
        frame_id: str,
        timestamp: datetime,
        image: Image.Image,
        monitor_id: int,
        ocr_text: str,
        ocr_json: str,
        ocr_conf: float,
        embedding: list = None
    ):
        """存储全屏帧（兼容旧接口）"""
        # 1. 保存临时图片
        temp_path = self.temp_dir / "full_screen" / f"monitor_{monitor_id}"
        temp_path.mkdir(parents=True, exist_ok=True)
        
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        image_path = (temp_path / f"{ts_str}.png").resolve()
        image.save(str(image_path), format='PNG')
        
        # 2. 存入 SQLite 数据库
        cursor = self.conn.cursor()
        
        # 插入 frame
        cursor.execute("""
            INSERT INTO frames (frame_id, timestamp, image_path, monitor_id)
            VALUES (?, ?, ?, ?)
        """, (frame_id, timestamp.isoformat(), str(image_path), monitor_id))
        
        # 插入 OCR 结果
        cursor.execute("""
            INSERT INTO ocr_text (frame_id, text, text_json, ocr_engine, text_length, confidence)
            VALUES (?, ?, ?, 'pytesseract', ?, ?)
        """, (frame_id, ocr_text, ocr_json, len(ocr_text), ocr_conf))
        
        self.conn.commit()
        
        # 3. 存入 LanceDB（如果提供了 embedding）
        if embedding and self.vector_storage:
            try:
                frame_data = {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "image": image,
                    "embedding": embedding,
                    "ocr_text": ocr_text,
                    "image_path": str(image_path),
                    "device_name": f"monitor_{monitor_id}",
                    "metadata": {"monitor_id": monitor_id}
                }
                # 使用批量写入接口
                success = self.vector_storage.store_frames_batch([frame_data])
                if success:
                    logger.debug(f"Stored frame {frame_id} embedding to LanceDB")
                else:
                    logger.warning(f"Failed to store frame {frame_id} embedding to LanceDB")
            except Exception as e:
                logger.error(f"Error storing frame {frame_id} to LanceDB: {e}")
        
        # 4. 添加到缓冲区
        self.full_screen_buffer.append((frame_id, timestamp, str(image_path), monitor_id, embedding))
        
        logger.info(f"Stored full_screen frame: {frame_id}, buffer_size={len(self.full_screen_buffer)}")
        return str(image_path)
    
    def store_window_frame_bytes(
        self,
        sub_frame_id: str,
        parent_frame_id: str,
        timestamp: datetime,
        window_bytes: bytes,
        app_name: str,
        window_name: str
    ) -> str:
        """存储窗口帧（从原始bytes，不转换图片）"""
        # 1. 直接保存PNG bytes（最快）
        safe_app = self._sanitize_name(app_name)
        safe_window = self._sanitize_name(window_name)
        window_key = f"{safe_app}_{safe_window}"
        
        temp_path = self.temp_dir / "windows" / window_key
        temp_path.mkdir(parents=True, exist_ok=True)
        
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        image_path = (temp_path / f"{ts_str}.png").resolve()
        with open(image_path, 'wb') as f:
            f.write(window_bytes)
        
        # 2. 存入数据库
        cursor = self.conn.cursor()
        
        # 插入 sub_frame (offset_index=0 placeholder, updated after compression)
        cursor.execute("""
            INSERT INTO sub_frames 
            (sub_frame_id, timestamp, app_name, window_name, offset_index)
            VALUES (?, ?, ?, ?, 0)
        """, (sub_frame_id, timestamp.isoformat(), app_name, window_name))
        
        # 插入映射关系
        cursor.execute("""
            INSERT OR IGNORE INTO frame_subframe_mapping (frame_id, sub_frame_id)
            VALUES (?, ?)
        """, (parent_frame_id, sub_frame_id))
        
        self.conn.commit()
        
        # 3. 添加到缓冲区
        if window_key not in self.window_buffers:
            self.window_buffers[window_key] = []
        self.window_buffers[window_key].append({
            "sub_frame_id": sub_frame_id,
            "timestamp": timestamp,
            "image_path": str(image_path),
            "app_name": app_name,
            "window_name": window_name
        })
        
        logger.info(f"Stored window frame (bytes): {app_name}/{window_name}, buffer_size={len(self.window_buffers[window_key])}")
        return str(image_path)
    
    def store_window_frame(
        self,
        sub_frame_id: str,
        parent_frame_id: str,
        timestamp: datetime,
        image: Image.Image,
        app_name: str,
        window_name: str,
        embedding: list = None
    ):
        """存储窗口帧（兼容旧接口）"""
        # 1. 保存临时图片
        safe_app = self._sanitize_name(app_name)
        safe_window = self._sanitize_name(window_name)
        window_key = f"{safe_app}_{safe_window}"
        
        temp_path = self.temp_dir / "windows" / window_key
        temp_path.mkdir(parents=True, exist_ok=True)
        
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        image_path = (temp_path / f"{ts_str}.png").resolve()
        image.save(str(image_path), format='PNG')
        
        # 2. 存入数据库
        cursor = self.conn.cursor()
        
        # 插入 sub_frame (offset_index=0 placeholder, updated after compression)
        cursor.execute("""
            INSERT INTO sub_frames 
            (sub_frame_id, timestamp, app_name, window_name, offset_index)
            VALUES (?, ?, ?, ?, 0)
        """, (sub_frame_id, timestamp.isoformat(), app_name, window_name))
        
        # 插入映射关系
        cursor.execute("""
            INSERT OR IGNORE INTO frame_subframe_mapping (frame_id, sub_frame_id)
            VALUES (?, ?)
        """, (parent_frame_id, sub_frame_id))
        
        self.conn.commit()
        
        # 存入 LanceDB（如果提供了 embedding）
        if embedding and self.vector_storage:
            try:
                # 将 sub_frame 也存储到 LanceDB 的 frames 表中（使用 sub_frame_id 作为 frame_id）
                frame_data = {
                    "frame_id": sub_frame_id,  # 使用 sub_frame_id
                    "timestamp": timestamp,
                    "image": image,
                    "embedding": embedding,
                    "ocr_text": "",  # 子帧通常不做 OCR
                    "image_path": str(image_path),
                    "device_name": f"{app_name}/{window_name}",
                    "metadata": {
                        "app_name": app_name,
                        "window_name": window_name,
                        "parent_frame_id": parent_frame_id,
                        "is_sub_frame": True  # 标记这是子帧
                    }
                }
                # 使用批量写入接口
                success = self.vector_storage.store_frames_batch([frame_data])
                if success:
                    logger.debug(f"Stored sub_frame {sub_frame_id} embedding to LanceDB")
                else:
                    logger.warning(f"Failed to store sub_frame {sub_frame_id} embedding to LanceDB")
            except Exception as e:
                logger.error(f"Error storing sub_frame {sub_frame_id} to LanceDB: {e}")
        
        # 3. 添加到缓冲区
        if window_key not in self.window_buffers:
            self.window_buffers[window_key] = []
        self.window_buffers[window_key].append({
            "sub_frame_id": sub_frame_id,
            "timestamp": timestamp,
            "image_path": str(image_path),
            "app_name": app_name,
            "window_name": window_name
        })
        
        logger.info(f"Stored window frame: {app_name}/{window_name}, buffer_size={len(self.window_buffers[window_key])}")
        return str(image_path)
    
    def flush_to_video(self):
        """将所有缓冲的帧压缩成视频"""
        from core.storage.ffmpeg_utils import FFmpegFrameCompressor
        
        compressor = FFmpegFrameCompressor(fps=VIDEO_FPS)
        cursor = self.conn.cursor()
        
        # 1. 压缩全屏帧
        if self.full_screen_buffer:
            logger.info(f"Compressing {len(self.full_screen_buffer)} full_screen frames...")
            
            # 按 monitor_id 分组
            by_monitor = {}
            for item in self.full_screen_buffer:
                # 处理 5 个元素的元组 (frame_id, timestamp, image_path, monitor_id, embedding)
                if len(item) == 5:
                    frame_id, ts, path, monitor_id, embedding = item
                else:
                    # 向后兼容：处理 4 个元素的元组
                    frame_id, ts, path, monitor_id = item[:4]
                
                if monitor_id not in by_monitor:
                    by_monitor[monitor_id] = []
                by_monitor[monitor_id].append((frame_id, ts, path))
            
            for monitor_id, frames in by_monitor.items():
                if not frames:
                    continue
                
                # 按时间排序
                frames.sort(key=lambda x: x[1])
                
                # 输出路径
                first_ts = frames[0][1]
                date_str = first_ts.strftime("%Y-%m-%d")
                time_str = first_ts.strftime("%H-%M-%S")
                output_dir = self.video_dir / f"full_screen_monitor_{monitor_id}" / date_str
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = (output_dir / f"monitor_{monitor_id}_{time_str}.mp4").resolve()
                
                # 压缩
                input_files = [f[2] for f in frames]
                success = compressor.compress_from_files(input_files, str(output_path))
                
                if success:
                    # 插入 video_chunk 记录
                    cursor.execute("""
                        INSERT INTO video_chunks (file_path, monitor_id, fps, frame_count)
                        VALUES (?, ?, ?, ?)
                    """, (str(output_path), monitor_id, VIDEO_FPS, len(frames)))
                    chunk_id = cursor.lastrowid
                    
                    # 更新所有帧的 video_chunk_id 和 offset_index
                    for i, (frame_id, ts, path) in enumerate(frames):
                        cursor.execute("""
                            UPDATE frames 
                            SET video_chunk_id = ?, offset_index = ?, image_path = ?
                            WHERE frame_id = ?
                        """, (chunk_id, i, f"video_chunk:{chunk_id}:{i}", frame_id))
                    
                    logger.info(f"✓ Compressed full_screen to {output_path}")
                else:
                    logger.error(f"✗ Failed to compress full_screen for monitor {monitor_id}")
            
            self.full_screen_buffer.clear()
        
        # 2. 压缩窗口帧
        for window_key, frames in self.window_buffers.items():
            if not frames:
                continue
            
            logger.info(f"Compressing {len(frames)} window frames for {window_key}...")
            
            # 按时间排序
            frames.sort(key=lambda x: x["timestamp"])
            
            # 输出路径
            first_ts = frames[0]["timestamp"]
            date_str = first_ts.strftime("%Y-%m-%d")
            time_str = first_ts.strftime("%H-%M-%S")
            output_dir = self.video_dir / date_str / window_key
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = (output_dir / f"window_{time_str}.mp4").resolve()
            
            # 压缩
            input_files = [f["image_path"] for f in frames]
            success = compressor.compress_from_files(input_files, str(output_path))
            
            if success:
                # 插入 window_chunk 记录
                app_name = frames[0]["app_name"]
                window_name = frames[0]["window_name"]
                cursor.execute("""
                    INSERT INTO window_chunks (file_path, app_name, window_name, fps, frame_count)
                    VALUES (?, ?, ?, ?, ?)
                """, (str(output_path), app_name, window_name, VIDEO_FPS, len(frames)))
                chunk_id = cursor.lastrowid
                
                # 更新所有子帧的 window_chunk_id 和 offset_index
                for i, f in enumerate(frames):
                    cursor.execute("""
                        UPDATE sub_frames 
                        SET window_chunk_id = ?, offset_index = ?
                        WHERE sub_frame_id = ?
                    """, (chunk_id, i, f["sub_frame_id"]))
                
                logger.info(f"✓ Compressed window to {output_path}")
            else:
                logger.error(f"✗ Failed to compress window {window_key}")
        
        self.window_buffers.clear()
        self.conn.commit()
    
    def cleanup_temp_files(self):
        """清理临时文件"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temp directory: {self.temp_dir}")
    
    def _sanitize_name(self, name: str) -> str:
        """清理文件名"""
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
            name = name.replace(char, '_')
        return name[:50]


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("Demo: 5秒截屏存储 (OCR + Embedding 计时)")
    print("=" * 60)
    
    # 初始化
    conn = init_demo_db(DEMO_DB_PATH)
    storage = DemoStorage(conn, DEMO_TEMP_DIR, DEMO_VIDEO_DIR)
    
    print(f"\n📁 Database: {DEMO_DB_PATH}")
    print(f"📁 Temp frames: {DEMO_TEMP_DIR}")
    print(f"📁 Videos: {DEMO_VIDEO_DIR}")
    print(f"⏱️  Duration: {CAPTURE_DURATION} seconds")
    print(f"📸 Interval: {CAPTURE_INTERVAL} seconds")
    print()
    
    # 预加载 Encoder（避免第一帧时间不准）
    print("🔄 Pre-loading encoder...")
    encoder_load_start = time.time()
    get_encoder()
    encoder_load_time = time.time() - encoder_load_start
    print(f"✓ Encoder loaded in {encoder_load_time:.2f}s")
    print()
    
    # 创建异步处理器（OCR + Embedding）
    print("🔄 Starting async processor (OCR + Embedding)...")
    async_processor = AsyncProcessor(storage, db_path=str(DEMO_DB_PATH), batch_size=8)
    async_processor.start()
    print("✓ Async processor started")
    print()
    
    # 初始化帧差检测器（用于去重）
    print("🔄 Initializing frame diff detector (deduplication)...")
    frame_diff_detector = FrameDiffDetector(
        screen_threshold=0.006,  # 全屏帧差阈值
        window_threshold=0.006   # 窗口帧差阈值
    )
    print("✓ Frame diff detector initialized")
    print()
    
    # 统计时间
    total_capture_time = 0.0
    
    # 开始截屏
    start_time = time.time()
    next_capture_time = start_time  # 下一次捕获的时间
    frame_count = 0
    sub_frame_count = 0
    skipped_frames = 0  # 跳过的重复帧数
    skipped_windows = 0  # 跳过的重复窗口数
    
    print("🎬 Starting capture...")
    index = 0
    
    while time.time() - start_time < CAPTURE_DURATION:
        # 等待到下一次捕获时间
        current_time = time.time()
        if current_time < next_capture_time:
            sleep_time = next_capture_time - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        loop_start = time.time()
        
        # 1. 截屏（关键路径，只获取bytes，不转换图片）
        print(f"Capturing frame {index}...")
        index += 1
        capture_start = time.time()
        result = capture_screen_and_windows()
        capture_time = time.time() - capture_start
        total_capture_time += capture_time
        
        if result is None:
            logger.error("Capture failed, skipping...")
            next_capture_time = time.time() + CAPTURE_INTERVAL
            continue
        
        timestamp = datetime.now(timezone.utc)
        
        # 1.5. 将全屏 bytes 转换为 PIL Image 用于帧差检测
        full_screen_image = Image.open(io.BytesIO(result["full_screen_bytes"])).convert('RGB')
        full_screen_hash = calculate_image_hash(full_screen_image)
        
        # 创建 ScreenObject 用于帧差检测
        screen_obj = ScreenObject(
            monitor_id=result["monitor_id"],
            device_name=f"monitor_{result['monitor_id']}",
            timestamp=timestamp,
            full_screen_image=full_screen_image,
            full_screen_hash=full_screen_hash,
            windows=[]
        )
        
        # 检查全屏是否有变化
        screen_diff_result = frame_diff_detector.check_screen_diff(screen_obj)
        
        if not screen_diff_result.should_store:
            # 跳过重复帧
            skipped_frames += 1
            logger.debug(f"Skipping duplicate full screen frame (diff={screen_diff_result.diff_score:.4f} < threshold)")
            next_capture_time = loop_start + CAPTURE_INTERVAL
            continue
        
        # 全屏有变化，继续处理
        frame_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        
        # 2. 存储全屏原始bytes
        image_path = storage.store_full_screen_frame_bytes(
            frame_id=frame_id,
            timestamp=timestamp,
            image_bytes=result["full_screen_bytes"],
            monitor_id=result["monitor_id"]
        )
        frame_count += 1
        
        # 3. 将全屏处理任务加入队列（图片转换 + OCR + Embedding，异步处理）
        async_processor.add_full_screen_task(
            frame_id=frame_id,
            timestamp=timestamp,
            image_bytes=result["full_screen_bytes"],
            monitor_id=result["monitor_id"],
            image_path=image_path
        )
        
        # 4. 处理窗口（存储原始bytes，处理任务异步）
        stored_windows = 0
        for i, window in enumerate(result.get("windows", [])):
            # 将窗口 bytes 转换为 PIL Image 用于帧差检测
            window_image = Image.open(io.BytesIO(window["window_bytes"])).convert('RGB')
            window_hash = calculate_image_hash(window_image)
            
            # 创建 WindowFrame 用于帧差检测
            window_frame = WindowFrame(
                app_name=window["app_name"],
                window_name=window["window_name"],
                image=window_image,
                timestamp=timestamp,
                image_hash=window_hash  # 使用 calculate_image_hash 计算的 hash
            )
            
            # 检查窗口是否有变化
            window_diff_result = frame_diff_detector.check_window_diff(window_frame)
            
            if not window_diff_result.should_store:
                # 跳过重复窗口帧
                skipped_windows += 1
                logger.debug(f"Skipping duplicate window frame {window['app_name']}/{window['window_name']} (diff={window_diff_result.diff_score:.4f} < threshold)")
                continue
            
            # 窗口有变化，继续处理
            sub_frame_id = f"{frame_id}_w{i}"
            
            # 存储窗口帧原始bytes
            window_image_path = storage.store_window_frame_bytes(
                sub_frame_id=sub_frame_id,
                parent_frame_id=frame_id,
                timestamp=timestamp,
                window_bytes=window["window_bytes"],
                app_name=window["app_name"],
                window_name=window["window_name"]
            )
            sub_frame_count += 1
            stored_windows += 1
            
            # 将窗口处理任务加入队列（图片转换 + Embedding，异步处理）
            async_processor.add_window_task(
                sub_frame_id=sub_frame_id,
                parent_frame_id=frame_id,
                timestamp=timestamp,
                window_bytes=window["window_bytes"],
                app_name=window["app_name"],
                window_name=window["window_name"],
                image_path=window_image_path
            )
        
        # 输出本帧的时间统计
        window_count = len(result.get("windows", []))
        loop_time = time.time() - loop_start
        print(f"  📸 Frame {frame_count}: capture={capture_time:.2f}s, total={loop_time:.2f}s, windows={stored_windows}/{window_count} stored (diff={screen_diff_result.diff_score:.4f})")
        
        if stored_windows > 0:
            print(f"  🪟 Windows ({stored_windows}/{window_count}): queued for async processing")
        
        # 设置下一次捕获时间（确保间隔为 CAPTURE_INTERVAL）
        next_capture_time = loop_start + CAPTURE_INTERVAL
    
    # 等待所有处理任务完成
    print("\n⏳ Waiting for all async tasks to complete (OCR + Embedding)...")
    max_wait_time = 60  # 最多等待60秒
    wait_start = time.time()
    while not async_processor.task_queue.empty():
        if time.time() - wait_start > max_wait_time:
            logger.warning("Timeout waiting for async tasks")
            break
        time.sleep(0.1)
    # 再等待一小段时间确保批次处理完成
    time.sleep(2.0)
    async_processor.stop()
    print("✓ All async tasks completed")
    
    # 时间统计总结
    total_captured = frame_count + skipped_frames
    print(f"\n✅ Capture complete: {frame_count} frames stored, {skipped_frames} frames skipped (deduplication)")
    print(f"   Windows: {sub_frame_count} stored, {skipped_windows} skipped")
    if total_captured > 0:
        dedup_rate = (skipped_frames / total_captured) * 100
        print(f"   Deduplication rate: {dedup_rate:.1f}% ({skipped_frames}/{total_captured} frames skipped)")
    print("\n" + "=" * 60)
    print("⏱️  TIME STATISTICS")
    print("=" * 60)
    if frame_count > 0:
        print(f"  Total capture time:        {total_capture_time:.2f}s (avg: {total_capture_time/frame_count:.2f}s/frame)")
        
        # 从 async_processor 获取统计信息
        total_ocr_time = async_processor.total_ocr_time
        total_embedding_time = async_processor.total_embedding_time
        total_sub_embedding_time = async_processor.total_sub_embedding_time
        
        print(f"  Total OCR time:            {total_ocr_time:.2f}s (async, avg: {total_ocr_time/frame_count:.2f}s/frame)")
        print(f"  Total Frame Embedding:    {total_embedding_time:.2f}s (async, avg: {total_embedding_time/frame_count:.2f}s/frame)")
        if sub_frame_count > 0:
            print(f"  Total Sub-Frame Embedding: {total_sub_embedding_time:.2f}s (async, avg: {total_sub_embedding_time/sub_frame_count:.2f}s/sub_frame)")
        print(f"  ---")
        
        # 关键路径时间（只包含截屏）
        critical_path_time = total_capture_time
        print(f"  Critical Path (Capture only): {critical_path_time:.2f}s (avg: {critical_path_time/frame_count:.2f}s/frame)")
        print(f"  Async Processing (OCR+Embedding): {total_ocr_time + total_embedding_time + total_sub_embedding_time:.2f}s (non-blocking)")
        
        total_processing_time = critical_path_time + total_ocr_time + total_embedding_time + total_sub_embedding_time
        print(f"  Total Processing:          {total_processing_time:.2f}s")
        print(f"    - Critical Path:         {critical_path_time:.2f}s ({critical_path_time/total_processing_time*100:.1f}%)")
        print(f"    - Async Processing:     {total_ocr_time + total_embedding_time + total_sub_embedding_time:.2f}s ({(total_ocr_time + total_embedding_time + total_sub_embedding_time)/total_processing_time*100:.1f}%)")
    print("=" * 60)
    
    # 压缩成视频
    print("\n🎥 Compressing to MP4...")
    storage.flush_to_video()
    
    # 清理临时文件
    print("\n🧹 Cleaning up temp files...")
    storage.cleanup_temp_files()
    
    # 验证结果
    print("\n📊 Verification:")
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM frames")
    print(f"  frames: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM ocr_text")
    print(f"  ocr_text: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM video_chunks")
    print(f"  video_chunks: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM sub_frames")
    print(f"  sub_frames: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM window_chunks")
    print(f"  window_chunks: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM frame_subframe_mapping")
    print(f"  frame_subframe_mapping: {cursor.fetchone()[0]}")
    
    # 检查 LanceDB 中的数据
    if storage.vector_storage:
        try:
            # 尝试查询 LanceDB 表
            table = storage.vector_storage.table
            if table:
                count = table.count_rows()
                print(f"  LanceDB embeddings: {count} (frames + sub_frames)")
            else:
                print(f"  LanceDB embeddings: table not available")
        except Exception as e:
            print(f"  LanceDB embeddings: error checking ({e})")
    else:
        print(f"  LanceDB embeddings: not initialized")
    
    # 显示一些示例数据
    print("\n📋 Sample data:")
    cursor.execute("SELECT frame_id, image_path, video_chunk_id, offset_index FROM frames LIMIT 3")
    for row in cursor.fetchall():
        print(f"  Frame: {row[0]}, path: {row[1][:50]}..., chunk: {row[2]}, offset: {row[3]}")
    
    cursor.execute("SELECT file_path, frame_count FROM video_chunks LIMIT 3")
    for row in cursor.fetchall():
        print(f"  Video: {row[0]}, frames: {row[1]}")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print(f"📁 Results saved to: {DEMO_DB_PATH.parent}")
    print("=" * 60)


if __name__ == "__main__":
    main()
