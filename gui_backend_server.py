#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend server for VisualMem GUI (remote mode).

Responsibilities:
- Receive frames from remote GUI via HTTP (frame diff + compression done on GUI)
- Store frames to server-side disk, SQLite (OCR DB), and LanceDB (vector DB)
- Provide RAG + rerank + VLM APIs for GUI queries
"""

from datetime import datetime, time, timezone, timedelta
from typing import List, Dict, Optional
import base64
import io
import threading
import time as time_module
from collections import deque
import json

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image as PILImage
from pathlib import Path

from config import config
from utils.logger import setup_logger
from utils.app_name_manager import app_name_manager
from core.encoder import create_encoder
from core.storage.lancedb_storage import LanceDBStorage
from core.storage.sqlite_storage import SQLiteStorage
from core.storage.temp_frame_buffer import TempFrameBuffer, FrameInfo
from core.storage.ffmpeg_utils import (
    FFmpegFrameCompressor,
    FFmpegFrameExtractor,
    get_video_frame_as_base64,
)
from core.retrieval.query_llm_utils import rewrite_and_time, filter_by_time
from core.retrieval.reranker import Reranker
from core.understand.api_vlm import ApiVLM
from core.ocr import create_ocr_engine
from core.capture.focused_window import get_focused_window
from utils.model_utils import ensure_model_downloaded


logger = setup_logger("gui_backend_server")

app = FastAPI(title="VisualMem Backend Server")

# 添加 CORS 中间件，允许 Electron 前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Electron 应用，允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)


# ============ 工具函数 ============

def _get_directory_size(directory: Path) -> int:
    """
    递归计算目录的总大小（字节）
    
    Args:
        directory: 目录路径
        
    Returns:
        目录总大小（字节）
    """
    total_size = 0
    try:
        if directory.exists() and directory.is_dir():
            for entry in directory.rglob('*'):
                try:
                    if entry.is_file():
                        total_size += entry.stat().st_size
                except (OSError, PermissionError):
                    # 忽略无法访问的文件
                    pass
    except (OSError, PermissionError):
        pass
    return total_size


def _format_size(bytes_size: int) -> str:
    """
    将字节大小格式化为人类可读的格式
    
    Args:
        bytes_size: 字节大小
        
    Returns:
        格式化后的字符串
    """
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f} KB"
    elif bytes_size < 1024 * 1024 * 1024:
        return f"{bytes_size / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes_size / (1024 * 1024 * 1024):.2f} GB"


def _ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """确保 datetime 对象具有 UTC 时区信息，如果是 naive 则视为本地时间并转换"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        # 如果没有时区信息，认为是本地时间，转换为 UTC
        return dt.astimezone(timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_local(dt_or_str) -> str:
    """UTC datetime/string → 本地时间 ISO 字符串（带时区偏移）。
    数据库中的时间都是 UTC（naive），此函数用于 API 响应。"""
    if dt_or_str is None:
        return ""
    if isinstance(dt_or_str, str):
        if not dt_or_str:
            return ""
        dt_or_str = datetime.fromisoformat(dt_or_str)
    if dt_or_str.tzinfo is None:
        dt_or_str = dt_or_str.replace(tzinfo=timezone.utc)
    return dt_or_str.astimezone().isoformat()


def _parent_watchdog():
    """
    监控父进程是否还在运行。如果父进程退出，则自动退出。
    这防止了 Electron 前端关闭后 Python 后端依然运行的问题。
    """
    import os
    import sys
    import time as time_module
    
    # 记录启动时的父进程 ID
    initial_ppid = os.getppid()
    if initial_ppid <= 1:
        # 如果父进程已经是 1 (init/launchd)，说明可能是独立启动的，不开启监控
        logger.info("Backend started without parent process or as orphan, skipping watchdog.")
        return

    logger.info(f"Parent process watchdog started (monitoring PPID: {initial_ppid})")
    
    while True:
        time_module.sleep(5)  # 每 5 秒检查一次
        current_ppid = os.getppid()
        
        # 如果父进程 ID 变为 1，或者与初始 ID 不同，说明原来的父进程已经退出
        if current_ppid != initial_ppid:
            logger.info(f"Parent process (PPID {initial_ppid}) has exited. Backend shutting down...")
            # 发送退出信号
            os._exit(0)  # 使用 os._exit(0) 强制退出，避免被 uvicorn 捕获


# ============ 全局单例组件 ============

encoder = None
vector_storage: Optional[LanceDBStorage] = None
sqlite_storage: Optional[SQLiteStorage] = None
reranker: Optional[Reranker] = None
vlm: Optional[ApiVLM] = None
ocr_engine = None
region_ocr_engine = None  # RegionOCREngine (UIED region detection + per-region OCR)

# ============ 视频存储相关组件 ============
temp_frame_buffer: Optional[TempFrameBuffer] = None
ffmpeg_compressor: Optional[FFmpegFrameCompressor] = None
ffmpeg_extractor: Optional[FFmpegFrameExtractor] = None

# 帧差检测（窗口级别去重）
from core.preprocess.frame_diff import FrameDiffDetector, is_solid_color_image
window_diff_detector: Optional[FrameDiffDetector] = None

# Activity clustering
from core.activity.cluster_manager import ClusterManager
cluster_manager: Optional[ClusterManager] = None

# 视频压缩配置
VIDEO_BATCH_SIZE = 60  # 每60帧压缩一次
VIDEO_FPS = 1.0  # 1帧/秒

# ============ 窗口捕获组件 ============
# 尝试导入 screencap_rs 用于窗口捕获
USE_SCREENCAP_RS = False
screencap_rs_module = None
try:
    import screencap_rs as screencap_rs_module
    USE_SCREENCAP_RS = True
    logger.info(f"screencap_rs available (platform: {screencap_rs_module.get_platform()})")
except ImportError:
    logger.warning("screencap_rs not available, window capture will be disabled")

# 是否在后端捕获窗口（如果前端没有提供窗口信息）
ENABLE_BACKEND_WINDOW_CAPTURE = True

# ============ 批量写入缓冲区 ============

class BatchWriteBuffer:
    """批量写入缓冲区：累积帧数据，达到阈值时批量写入"""
    
    def __init__(self, batch_size: int = 10, flush_interval_seconds: float = 60.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        self.buffer: deque = deque()
        self.buffer_lock = threading.Lock()
        self.last_flush_time = time_module.time()
        self.flush_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
    
    def add_frame(self, frame_data: dict):
        """添加帧数据到缓冲区"""
        with self.buffer_lock:
            self.buffer.append(frame_data)
            buf_len = len(self.buffer)
            should_flush = buf_len >= self.batch_size
        logger.info(f"BatchWriteBuffer: added frame {frame_data.get('frame_id', '?')}, buffer={buf_len}/{self.batch_size}")
        if should_flush:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """清空缓冲区并批量写入"""
        with self.buffer_lock:
            if not self.buffer:
                return
            frames_to_write = list(self.buffer)
            self.buffer.clear()
            self.last_flush_time = time_module.time()
        
        if not frames_to_write:
            return
        
        try:
            logger.info(f"BatchWriteBuffer: flushing {len(frames_to_write)} frames to LanceDB+SQLite...")
            # 批量写入到 LanceDB
            if vector_storage is not None:
                success = vector_storage.store_frames_batch(frames_to_write)
                if success:
                    logger.info(f"BatchWriteBuffer: ✓ LanceDB wrote {len(frames_to_write)} frames")
                else:
                    logger.error(f"BatchWriteBuffer: ✗ LanceDB batch write failed")
            
            # 批量写入到 SQLite（逐条写入，因为 SQLite 的批量写入接口可能不同）
            if sqlite_storage is not None:
                for frame_data in frames_to_write:
                    try:
                        # If OCR was already stored via store_ocr_with_regions,
                        # pass empty ocr_text to avoid duplicate ocr_text rows.
                        ocr_text_for_sqlite = "" if frame_data.get("_ocr_regions_stored") else frame_data.get("ocr_text", "")
                        sqlite_storage.store_frame_with_ocr(
                            frame_id=frame_data["frame_id"],
                            timestamp=frame_data["timestamp"],
                            image_path=frame_data["image_path"],
                            ocr_text=ocr_text_for_sqlite,
                            ocr_text_json=frame_data.get("ocr_text_json", "") if not frame_data.get("_ocr_regions_stored") else "",
                            ocr_engine=frame_data.get("ocr_engine", "pending"),
                            ocr_confidence=frame_data.get("ocr_confidence", 0.0),
                            device_name=frame_data.get("device_name", "remote-gui"),
                            metadata=frame_data.get("metadata", {}),
                            app_name=frame_data.get("app_name"),
                            window_name=frame_data.get("window_name"),
                            focused_app_name=frame_data.get("focused_app_name"),
                            focused_window_name=frame_data.get("focused_window_name"),
                        )
                    except Exception as e:
                        logger.error(f"写入 SQLite 失败 (frame_id={frame_data.get('frame_id')}): {e}")
        except Exception as e:
            logger.error(f"批量写入失败: {e}")
    
    def _periodic_flush(self):
        """定期检查并刷新缓冲区（后台线程）"""
        while not self.stop_event.is_set():
            time_module.sleep(1)  # 每秒检查一次
            with self.buffer_lock:
                elapsed = time_module.time() - self.last_flush_time
                should_flush = elapsed >= self.flush_interval and len(self.buffer) > 0
            
            if should_flush:
                logger.info(f"BatchWriteBuffer: periodic flush triggered ({elapsed:.0f}s elapsed, {len(self.buffer)} frames)")
                self._flush_buffer()
    
    def start(self):
        """启动后台刷新线程"""
        if self.flush_thread is None or not self.flush_thread.is_alive():
            self.stop_event.clear()
            self.flush_thread = threading.Thread(target=self._periodic_flush, daemon=True)
            self.flush_thread.start()
            logger.info(f"批量写入缓冲区后台线程已启动（批次大小: {self.batch_size}, 刷新间隔: {self.flush_interval}秒）")
    
    def stop(self):
        """停止后台刷新线程并清空缓冲区"""
        self.stop_event.set()
        if self.flush_thread is not None:
            self.flush_thread.join(timeout=5)
        # 清空剩余缓冲区
        self._flush_buffer()
        logger.info("批量写入缓冲区已停止")

# 全局批量写入缓冲区
batch_write_buffer: Optional[BatchWriteBuffer] = None


def _resolve_sub_frame_image_path(sf: dict) -> Optional[str]:
    """
    Build image_path for a sub_frame response.

    Normal sub_frames use ``window_chunk:{id}:{offset}``.
    Synthetic full-screen sub_frames (window_chunk_id==0) fall back
    to the ``frames`` table entry which holds the ``video_chunk:...`` reference.
    """
    wc_id = sf.get("window_chunk_id") or 0
    off = sf.get("offset_index")
    if wc_id > 0 and off is not None:
        return f"window_chunk:{wc_id}:{off}"
    if sqlite_storage is not None:
        try:
            conn = sqlite_storage._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT image_path FROM frames WHERE frame_id = ?",
                (sf["sub_frame_id"],),
            )
            row = cursor.fetchone()
            conn.close()
            if row and row["image_path"]:
                return row["image_path"]
        except Exception:
            pass
    return None


def _log_non_committed_cluster_result(app_name: str, sub_frame_id: str):
    """
    Log online clustering results for frames that did not directly land in an
    existing committed cluster. This is intentionally INFO-level so it shows up
    in backend_server.log during real runs.
    """
    if sqlite_storage is None:
        return
    try:
        conn = sqlite_storage._get_activity_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT activity_cluster_id, activity_label, provisional_label, cluster_status
            FROM activity_assignments
            WHERE sub_frame_id = ?
            """,
            (sub_frame_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return

        status = row["cluster_status"] or "pending"
        label = row["provisional_label"] or row["activity_label"]
        cluster_id = row["activity_cluster_id"]

        if status == "candidate":
            logger.debug(
                f"Activity clustering candidate for app='{app_name}': "
                f"frame={sub_frame_id}, cluster_id={cluster_id}, label='{label}'"
            )
        elif status == "pending":
            logger.debug(
                f"Activity clustering pending for app='{app_name}': "
                f"frame={sub_frame_id}, no committed match and no provisional label"
            )
    except Exception as e:
        logger.debug(f"Failed to log cluster result for {sub_frame_id}: {e}")


def _on_video_batch_ready(batch_type: str, identifier: str, frames: List[FrameInfo]):
    """
    当视频批次准备好压缩时的回调函数
    
    Args:
        batch_type: "full_screen" 或 "window"
        identifier: "monitor_{id}" 或 "{app_name}_{window_name}"
        frames: 帧信息列表
    """
    logger.info(f"Video batch ready: {batch_type}/{identifier}, {len(frames)} frames")
    # 实际压缩在 _compress_video_batch 中异步执行


def _compress_video_batch(batch_type: str, identifier: str, frames: List[FrameInfo]):
    """
    压缩视频批次
    
    Args:
        batch_type: "full_screen" 或 "window"
        identifier: "monitor_{id}" 或 "{app_name}_{window_name}"
        frames: 帧信息列表
    """
    global temp_frame_buffer, ffmpeg_compressor, sqlite_storage
    
    if not frames or ffmpeg_compressor is None:
        return
    
    try:
        # 获取输出路径
        first_frame = frames[0]
        output_path = temp_frame_buffer._get_video_output_path(
            batch_type, identifier, first_frame.timestamp
        )
        
        # 收集输入文件路径
        input_files = [f.image_path for f in frames]
        
        # 压缩视频
        success = ffmpeg_compressor.compress_from_files(input_files, str(output_path))
        
        if success:
            logger.debug(f"Video compression successful: {output_path}")
            
            # 插入视频chunk记录到数据库
            if sqlite_storage is not None:
                if batch_type == "full_screen":
                    monitor_id = int(identifier.split("_")[1]) if "_" in identifier else 0
                    chunk_id = sqlite_storage.insert_video_chunk(
                        file_path=str(output_path),
                        monitor_id=monitor_id,
                        device_name=identifier,
                        fps=VIDEO_FPS
                    )
                    
                    if chunk_id > 0:
                        sqlite_storage.update_chunk_frame_count(chunk_id, len(frames), "video")
                        for i, frame in enumerate(frames):
                            sqlite_storage.store_frame_with_video_ref(
                                frame_id=frame.frame_id,
                                timestamp=frame.timestamp,
                                video_chunk_id=chunk_id,
                                offset_index=i,
                                monitor_id=frame.monitor_id,
                                device_name=identifier,
                                metadata=frame.metadata,
                                app_name=None,
                                window_name=None
                            )
                            
                            # Fullscreen sub_frames are synced atomically inside
                            # store_frame_with_video_ref() — no back-fill needed here.
                else:
                    # 窗口视频
                    app_name = frames[0].app_name or "unknown"
                    window_name = frames[0].window_name or "unknown"
                    chunk_id = sqlite_storage.insert_window_chunk(
                        file_path=str(output_path),
                        app_name=app_name,
                        window_name=window_name,
                        monitor_id=0,
                        fps=VIDEO_FPS
                    )
                    
                    if chunk_id > 0:
                        sqlite_storage.update_chunk_frame_count(chunk_id, len(frames), "window")
                        for i, frame in enumerate(frames):
                            # 存储到 sub_frames 表（用于关联 window_chunk）
                            sqlite_storage.store_sub_frame(
                                sub_frame_id=frame.frame_id,
                                timestamp=frame.timestamp,
                                window_chunk_id=chunk_id,
                                offset_index=i,
                                app_name=frame.app_name or "",
                                window_name=frame.window_name or ""
                            )
                            
                            # 同时更新 frames 表中的记录（sub_frame 也存储在 frames 表中，用于统一查询）
                            # 使用 window_chunk 格式：window_chunk:{chunk_id}:{offset_index}
                            image_path = f"window_chunk:{chunk_id}:{i}"
                            # 对于 sub_frame，video_chunk_id 设为 NULL（因为它是 window_chunk）
                            # 我们需要直接更新 frames 表
                            try:
                                conn = sqlite_storage._get_connection()
                                cursor = conn.cursor()
                                
                                # First check if frame exists
                                cursor.execute("SELECT 1 FROM frames WHERE frame_id = ?", (frame.frame_id,))
                                if cursor.fetchone():
                                    cursor.execute("""
                                        UPDATE frames 
                                        SET image_path = ?,
                                            offset_index = ?,
                                            app_name = ?,
                                            window_name = ?
                                        WHERE frame_id = ?
                                    """, (
                                        image_path,
                                        i,
                                        frame.app_name or "",
                                        frame.window_name or "",
                                        frame.frame_id
                                    ))
                                else:
                                    # Insert if it doesn't exist
                                    cursor.execute("""
                                        INSERT INTO frames 
                                        (frame_id, timestamp, image_path, device_name, metadata, app_name, window_name, offset_index)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    """, (
                                        frame.frame_id,
                                        frame.timestamp.isoformat(),
                                        image_path,
                                        f"{frame.app_name}/{frame.window_name}",
                                        "{}",
                                        frame.app_name or "",
                                        frame.window_name or "",
                                        i
                                    ))
                                conn.commit()
                                conn.close()
                                logger.debug(f"Updated sub_frame {frame.frame_id} in frames table")
                            except Exception as e:
                                logger.error(f"Failed to update sub_frame {frame.frame_id} in frames table: {e}")
                            
                            # 创建帧与子帧的映射关系
                            if frame.parent_frame_id:
                                sqlite_storage.add_frame_subframe_mapping(
                                    frame_id=frame.parent_frame_id,
                                    sub_frame_id=frame.frame_id
                                )
            
            # 清理临时文件
            temp_frame_buffer.cleanup_batch_files(frames)
            
        else:
            logger.error(f"Video compression failed for {batch_type}/{identifier}")
            
    except Exception as e:
        logger.error(f"Error compressing video batch: {e}")


def _check_and_compress_batches():
    """
    检查并压缩所有就绪的批次
    """
    global temp_frame_buffer
    
    if temp_frame_buffer is None:
        return
    
    ready_batches = temp_frame_buffer.get_ready_batches()
    for batch_type, identifier, _ in ready_batches:
        frames = temp_frame_buffer.flush_batch(batch_type, identifier)
        if frames:
            _compress_video_batch(batch_type, identifier, frames)


def _flush_all_video_buffers():
    """
    刷新所有视频缓冲区（用于停止录制时）
    """
    global temp_frame_buffer
    
    if temp_frame_buffer is None:
        return
    
    all_batches = temp_frame_buffer.flush_all()
    for batch_type, identifier, frames in all_batches:
        if frames:
            _compress_video_batch(batch_type, identifier, frames)
    
    # 清理空目录
    temp_frame_buffer.cleanup_empty_dirs()


def _recover_temp_frames():
    """
    发现异常退出遗留的 temp_frames，将其压缩成 MP4 并更新 SQLite，然后删除临时文件。
    """
    global temp_frame_buffer, ffmpeg_compressor, sqlite_storage
    if not sqlite_storage or not ffmpeg_compressor or not temp_frame_buffer:
        return
        
    try:
        conn = sqlite_storage._get_connection()
        cursor = conn.cursor()
        
        # 1. 恢复全屏截图
        cursor.execute("SELECT frame_id, timestamp, image_path, device_name, metadata FROM frames WHERE image_path LIKE '%temp_frames%' AND frame_id LIKE 'frame_%' ORDER BY timestamp ASC")
        fs_records = cursor.fetchall()
        
        from collections import defaultdict
        from core.storage.temp_frame_buffer import FrameInfo
        import json
        
        fs_groups = defaultdict(list)
        for row in fs_records:
            path = Path(row["image_path"])
            if path.exists():
                monitor_id = 0
                if row["device_name"] and row["device_name"].startswith("monitor_"):
                    try:
                        monitor_id = int(row["device_name"].split("_")[1])
                    except ValueError:
                        pass
                
                ts_str = row["timestamp"]
                ts = datetime.fromisoformat(ts_str) if isinstance(ts_str, str) else ts_str
                
                fs_groups[monitor_id].append(FrameInfo(
                    frame_id=row["frame_id"],
                    timestamp=ts,
                    image_path=str(path),
                    monitor_id=monitor_id,
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {}
                ))
            else:
                # 文件丢失，清理僵尸路径
                cursor.execute("UPDATE frames SET image_path = '' WHERE frame_id = ?", (row["frame_id"],))

        # 2. 恢复窗口截图
        cursor.execute("""
            SELECT s.sub_frame_id, s.timestamp, f.image_path, s.app_name, s.window_name
            FROM sub_frames s
            JOIN frames f ON s.sub_frame_id = f.frame_id
            WHERE f.image_path LIKE '%temp_frames%'
            ORDER BY s.timestamp ASC
        """)
        win_records = cursor.fetchall()
        
        win_groups = defaultdict(list)
        for row in win_records:
            path = Path(row["image_path"])
            if path.exists():
                app_name = row["app_name"] or "unknown"
                window_name = row["window_name"] or "unknown"
                key = f"{app_name}_{window_name}"
                
                ts_str = row["timestamp"]
                ts = datetime.fromisoformat(ts_str) if isinstance(ts_str, str) else ts_str
                
                win_groups[key].append(FrameInfo(
                    frame_id=row["sub_frame_id"],
                    timestamp=ts,
                    image_path=str(path),
                    app_name=app_name,
                    window_name=window_name
                ))
            else:
                cursor.execute("UPDATE frames SET image_path = '' WHERE frame_id = ?", (row["sub_frame_id"],))

        # Commit zombie-path cleanups before compression. _compress_video_batch opens new
        # SQLite connections; an open uncommitted transaction on this conn would lock them.
        conn.commit()
        conn.close()

        for monitor_id, frames in fs_groups.items():
            logger.info(f"Recovering {len(frames)} leftover full_screen temp_frames for monitor_{monitor_id}...")
            _compress_video_batch("full_screen", f"monitor_{monitor_id}", frames)

        for key, frames in win_groups.items():
            logger.info(f"Recovering {len(frames)} leftover window temp_frames for {key}...")
            _compress_video_batch("window", key, frames)
    except Exception as e:
        logger.error(f"Error recovering temp frames: {e}")


def _init_components():
    """Lazy-init heavy components (called on first request)."""
    global encoder, vector_storage, sqlite_storage, reranker, vlm, ocr_engine, region_ocr_engine

    if encoder is None:
        logger.info(f"Loading encoder {config.EMBEDDING_MODEL} for gui_backend_server...")
        encoder = create_encoder(model_name=config.EMBEDDING_MODEL)
        logger.info(f"Encoder {config.EMBEDDING_MODEL} loaded.")

    if vector_storage is None:
        logger.info("Initializing LanceDB storage for gui_backend_server...")
        vector_storage = LanceDBStorage(
            db_path=config.LANCEDB_PATH,
            embedding_dim=encoder.embedding_dim,
        )
        logger.info("LanceDB storage initialized.")

    if sqlite_storage is None:
        logger.info("Initializing SQLite storage for gui_backend_server...")
        sqlite_storage = SQLiteStorage(db_path=config.OCR_DB_PATH)
        logger.info("SQLite storage initialized.")

    if reranker is None:
        reranker = Reranker()
        logger.info("Reranker initialized.")

    if vlm is None:
        vlm = ApiVLM()
        logger.info("VLM API client initialized.")

    global ocr_engine, region_ocr_engine
    if ocr_engine is None and config.ENABLE_OCR:
        try:
            ocr_engine = create_ocr_engine(config.OCR_ENGINE_TYPE, lang="chi_sim+eng")
            logger.info(f"OCR engine initialized ({config.OCR_ENGINE_TYPE}).")
        except Exception as e:
            logger.warning(f"Failed to init OCR engine, fallback to dummy: {e}")
            ocr_engine = create_ocr_engine("dummy")

    if region_ocr_engine is None and ocr_engine is not None:
        from core.ocr.region_detector import UIEDRegionDetector
        from core.ocr.region_ocr_engine import RegionOCREngine
        region_ocr_engine = RegionOCREngine(
            ocr_engine=ocr_engine,
            region_detector=UIEDRegionDetector(),
        )
        logger.info("RegionOCREngine initialized (UIED + per-region OCR).")


def _init_all_components():
    """
    强制初始化所有组件（用于服务器启动时预加载）。
    与 _init_components() 不同，这个函数会强制加载，不检查是否已加载。
    """
    global encoder, vector_storage, sqlite_storage, reranker, vlm, ocr_engine, region_ocr_engine, batch_write_buffer
    global temp_frame_buffer, ffmpeg_compressor, ffmpeg_extractor, window_diff_detector
    
    logger.info("=" * 60)
    logger.info("Initializing all backend components (startup preload)...")
    logger.info("=" * 60)

    # 0. Pre-flight check: Ensure models are downloaded
    # This provides better UX by explicitly showing download progress
    ensure_model_downloaded(config.EMBEDDING_MODEL, "Image Encoder")
    
    if config.ENABLE_RERANK:
        ensure_model_downloaded(config.RERANK_MODEL, "Reranker Model")
    
    # 1. Load encoder (embedding model)
    logger.info(f"[1/7] Loading encoder {config.EMBEDDING_MODEL}...")
    encoder = create_encoder(model_name=config.EMBEDDING_MODEL)
    
    # 2. Initialize LanceDB storage
    logger.info("[2/7] Initializing LanceDB storage...")
    vector_storage = LanceDBStorage(
        db_path=config.LANCEDB_PATH,
        embedding_dim=encoder.embedding_dim,
    )
    
    # 3. Initialize SQLite storage
    logger.info("[3/7] Initializing SQLite storage...")
    sqlite_storage = SQLiteStorage(db_path=config.OCR_DB_PATH)
    
    # 4. Load Reranker model
    if config.ENABLE_RERANK:
        logger.info("[4/7] Loading Reranker model...")
        reranker = Reranker()
    else:
        logger.info("[4/7] Reranker disabled (ENABLE_RERANK=False)")
        reranker = None
    
    # 5. Initialize VLM client
    logger.info("[5/7] Initializing VLM API client...")
    vlm = ApiVLM()
    
    # 6. Initialize OCR engine (if enabled)
    if config.ENABLE_OCR:
        logger.info(f"[6/7] Initializing OCR engine ({config.OCR_ENGINE_TYPE})...")
        try:
            ocr_engine = create_ocr_engine(config.OCR_ENGINE_TYPE, lang="chi_sim+eng")
        except Exception as e:
            logger.warning(f"Failed to init OCR engine ({config.OCR_ENGINE_TYPE}), fallback to dummy: {e}")
            ocr_engine = create_ocr_engine("dummy")
        # Build RegionOCREngine on top of the platform OCR engine
        from core.ocr.region_detector import UIEDRegionDetector
        from core.ocr.region_ocr_engine import RegionOCREngine
        region_ocr_engine = RegionOCREngine(
            ocr_engine=ocr_engine,
            region_detector=UIEDRegionDetector(),
        )
        logger.info("RegionOCREngine initialized (UIED + per-region OCR).")
    else:
        logger.info("[6/7] OCR engine disabled (ENABLE_OCR=False)")
        ocr_engine = None
    
    # 7. Optimize LanceDB (启动时优化：清理旧版本 + 压缩文件)
    logger.info("[7/7] Optimizing LanceDB (cleanup old versions + compact files)...")
    try:
        from datetime import timedelta
        if vector_storage is not None and vector_storage.table is not None:
            # 使用 optimize 方法（清理旧版本 + 压缩文件）
            # 清理 6 分钟前的版本（只保留最新的）
            logger.info("优化 LanceDB（清理旧版本 + 压缩文件）...")
            stats = vector_storage.cleanup_old_versions(
                older_than_hours=0.1,  # 清理 6 分钟前的版本（只保留最新的）
                delete_unverified=True
            )
            if stats:
                logger.info(f"✓ 优化完成（cleanup_older_than=0.1小时）")
        else:
            logger.info("LanceDB 表不存在，跳过优化")
    except Exception as e:
        logger.warning(f"优化 LanceDB 失败: {e}")
    
    # 8. Initialize batch write buffer
    logger.info("[8/10] Initializing batch write buffer...")
    batch_write_buffer = BatchWriteBuffer(batch_size=10, flush_interval_seconds=60.0)
    batch_write_buffer.start()
    
    # 9. Initialize temp frame buffer for video compression
    logger.info("[9/10] Initializing temp frame buffer for video compression...")
    temp_frame_buffer = TempFrameBuffer(
        storage_root=config.STORAGE_ROOT,
        batch_size=VIDEO_BATCH_SIZE,
        fps=VIDEO_FPS,
        on_batch_ready=_on_video_batch_ready
    )
    
    # 10. Initialize FFmpeg utilities
    logger.info("[10/11] Initializing FFmpeg utilities...")
    ffmpeg_compressor = FFmpegFrameCompressor(fps=VIDEO_FPS)
    ffmpeg_extractor = FFmpegFrameExtractor()
    
    # 11. Initialize window-level frame diff detector for dedup
    logger.info("[11/11] Initializing window frame diff detector...")
    window_diff_detector = FrameDiffDetector(
        screen_threshold=config.SIMPLE_FILTER_DIFF_THRESHOLD,
        window_threshold=config.SIMPLE_FILTER_DIFF_THRESHOLD,
    )
    
    # 12. Recover leftover temp frames from previous abnormal exit
    logger.info("[12/12] Recovering leftover temp frames...")
    _recover_temp_frames()

    # 13. Initialize activity cluster manager
    global cluster_manager
    if config.ENABLE_CLUSTERING:
        logger.info("[13/13] Initializing activity cluster manager...")
        try:
            cluster_manager = ClusterManager(
                activity_db_path=config.ACTIVITY_DB_PATH,
                main_db_path=config.OCR_DB_PATH,
            )
        except Exception as e:
            logger.warning(f"Failed to init ClusterManager (will retry later): {e}")
            cluster_manager = None
    else:
        logger.info("[13/13] Activity clustering disabled (ENABLE_CLUSTERING=False)")
        cluster_manager = None

    logger.info("=" * 60)
    logger.info("All backend components initialized successfully!")
    logger.info("=" * 60)


# ============ Pydantic models ============


class WindowInfo(BaseModel):
    """窗口信息（来自screencap_rs）"""
    app_name: str
    window_name: str
    image_base64: str  # 窗口截图的base64


class StoreFrameRequest(BaseModel):
    frame_id: str
    timestamp: str  # ISO string
    image_base64: str  # 全屏截图的base64
    monitor_id: int = 0  # 显示器ID
    metadata: Optional[Dict] = None
    windows: Optional[List[WindowInfo]] = None  # 窗口截图列表（可选）


class FrameResult(BaseModel):
    frame_id: str
    timestamp: str
    image_base64: Optional[str] = None
    image_path: Optional[str] = None
    ocr_text: Optional[str] = ""


class QueryRagWithTimeRequest(BaseModel):
    query: str
    start_time: Optional[str] = None  # ISO
    end_time: Optional[str] = None    # ISO
    search_type: str = "image"        # "image" or "text"
    ocr_mode: bool = False            # Legacy, kept for compatibility
    enable_hybrid: Optional[bool] = None
    enable_rerank: Optional[bool] = None
    activity_label: Optional[str] = None  # Filter by activity cluster label


class QueryRagWithTimeResponse(BaseModel):
    answer: str
    frames: List[FrameResult]


class GetFramesByDateRangeRequest(BaseModel):
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    offset: int = 0
    limit: int = 50


class GetFramesByDateRequest(BaseModel):
    date: str  # YYYY-MM-DD
    offset: int = 0
    limit: int = 50


class DateFrameCountResponse(BaseModel):
    date: str
    total_count: int


class DateRangeResponse(BaseModel):
    earliest_date: Optional[str]  # YYYY-MM-DD，最早的照片日期
    latest_date: Optional[str]    # YYYY-MM-DD，最新的照片日期


class ActivityClusterDebugAppStatus(BaseModel):
    app_name: str
    pending_unclassified: int
    candidate_frames: int
    committed_cluster_count: int
    threshold: int
    needs_recalc: bool


class ActivityClusterDebugResponse(BaseModel):
    threshold: int
    recalc_running: bool
    apps_needing_recalc: List[str]
    apps: List[ActivityClusterDebugAppStatus]


# ============ Startup Event ============


@app.on_event("startup")
async def startup_event():
    """
    服务器启动时预加载所有重型组件（embedding model, reranker, etc.）
    这样可以避免第一次请求时的延迟。
    """
    # 启动父进程监控线程
    import threading
    watchdog_thread = threading.Thread(target=_parent_watchdog, daemon=True)
    watchdog_thread.start()

    try:
        _init_all_components()
    except Exception as e:
        logger.critical(f"Fatal error during startup: {e}", exc_info=True)
        # 让进程以非零状态退出，Electron 端可以感知到后端启动失败
        import os
        os._exit(1)


@app.on_event("shutdown")
async def shutdown_event():
    """
    服务器关闭时清理资源，确保缓冲区数据写入磁盘。
    """
    logger.info("=" * 60)
    logger.info("Shutting down backend server...")
    
    global encoder, reranker, vlm, ocr_engine, region_ocr_engine, vector_storage, sqlite_storage
    
    # 1. 先刷新视频缓冲区
    try:
        if temp_frame_buffer is not None:
            logger.info("Flushing video frame buffers...")
            _flush_all_video_buffers()
    except Exception as e:
        logger.error(f"Error flushing video buffers: {e}")
    
    # 2. 再刷新批量写入缓冲区
    try:
        if batch_write_buffer is not None:
            logger.info("Flushing batch write buffer...")
            batch_write_buffer.stop()
    except Exception as e:
        logger.error(f"Error stopping batch write buffer: {e}")
    
    # 3. 显式释放大模型内存
    logger.info("Releasing models and clearing memory...")
    try:
        if encoder is not None:
            if hasattr(encoder, 'clear'):
                encoder.clear()
            encoder = None
        
        if reranker is not None:
            if hasattr(reranker, 'clear'):
                reranker.clear()
            reranker = None
            
        vlm = None
        ocr_engine = None
        region_ocr_engine = None

        # 强制垃圾回收
        import gc
        gc.collect()
        
        # 清理 PyTorch 缓存 (CUDA/MPS)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
        except Exception:
            pass
            
    except Exception as e:
        logger.error(f"Error during model release: {e}")
    
    # 4. 输出聚类统计
    if cluster_manager is not None:
        try:
            stats = cluster_manager.get_assignment_stats()
            logger.info("-" * 40)
            logger.info("Activity clustering session stats:")
            logger.info(f"  Total frames assigned : {stats['total_frames']}")
            logger.info(f"  VLM called (far from centroid): {stats['vlm_called_frames']}")
            logger.info(f"  VLM call ratio        : {stats['vlm_call_ratio']:.2%}")
        except Exception as e:
            logger.error(f"Error collecting cluster stats: {e}")

    logger.info("Backend server shutdown complete.")
    logger.info("=" * 60)


# ============ Endpoints ============


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/stats")
def get_stats():
    """
    获取存储统计信息
    
    Returns:
        包含总帧数、OCR帧数等统计信息的字典
    """
    # 组件已在启动时预加载，直接使用
    stats = {
        "total_frames": 0,
        "ocr_frames": 0,
        "storage_mode": "vector",
        "storage": "Local SQLite",
        "vlm_model": config.VLM_API_MODEL[:20] + "..." if len(config.VLM_API_MODEL) > 20 else config.VLM_API_MODEL,
        "disk_usage": "—",  # 将在下面计算
        "diff_threshold": config.SIMPLE_FILTER_DIFF_THRESHOLD,  # 帧差阈值配置
        "capture_interval_seconds": config.CAPTURE_INTERVAL_SECONDS,  # 截屏间隔（秒）
        "max_image_width": config.MAX_IMAGE_WIDTH,  # 最大图片宽度
        "image_quality": config.IMAGE_QUALITY  # 图片质量（1-100）
    }
    
    # 计算 visualmem_storage 文件夹的大小
    try:
        storage_root = Path(config.STORAGE_ROOT)
        storage_size = _get_directory_size(storage_root)
        stats["disk_usage"] = _format_size(storage_size)
    except Exception as e:
        logger.warning(f"Unable to get storage size: {e}")
        stats["disk_usage"] = "—"
    
    # 从 vector_storage 获取统计信息（主要统计源）
    if vector_storage is not None:
        try:
            vector_stats = vector_storage.get_stats()
            stats.update({
                "total_frames": vector_stats.get("total_frames", 0),
                "ocr_frames": vector_stats.get("ocr_frames", 0),
                "db_path": vector_stats.get("db_path", ""),
                "embedding_dim": vector_stats.get("embedding_dim", 0),
                "storage": "Vector DB"
            })
        except Exception as e:
            logger.warning(f"Failed to get vector storage stats: {e}")
    
    # 如果 vector_storage 没有 OCR 统计或为0，尝试从 sqlite_storage 获取
    if sqlite_storage is not None:
        try:
            sqlite_stats = sqlite_storage.get_stats()
            # 使用 SQLite 的 OCR 统计（更准确）
            stats["ocr_frames"] = sqlite_stats.get("total_ocr_results", 0)
            # 如果 vector_storage 的总帧数为0，也可以使用 SQLite 的帧数
            if stats.get("total_frames", 0) == 0:
                stats["total_frames"] = sqlite_stats.get("total_frames", 0)
        except Exception as e:
            logger.warning(f"Failed to get SQLite storage stats: {e}")
    
    return stats


@app.get("/api/activity_clusters/debug", response_model=ActivityClusterDebugResponse)
def get_activity_cluster_debug(app_name: Optional[str] = Query(default=None)):
    """Debug endpoint for persistent activity clustering trigger state."""
    if cluster_manager is None:
        raise HTTPException(status_code=503, detail="Cluster manager is not initialized")

    try:
        return cluster_manager.get_debug_status(app_name=app_name)
    except Exception as e:
        logger.error(f"Failed to get activity cluster debug status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/store_frame")
def store_frame(req: StoreFrameRequest):
    """
    Store a frame sent from remote GUI (支持新的视频存储模式).
    
    新逻辑:
    1. 全屏截图 -> 临时PNG文件 -> 每60帧压缩成MP4
    2. 窗口截图 -> 临时PNG文件 -> 每60帧压缩成MP4
    3. 同时进行embedding和OCR处理
    """
    global encoder, vector_storage, sqlite_storage, batch_write_buffer
    global temp_frame_buffer

    _t0 = time_module.time()
    logger.info(f"store_frame: START {req.frame_id}")

    # 组件检查
    assert encoder is not None
    assert vector_storage is not None
    assert sqlite_storage is not None
    assert batch_write_buffer is not None

    try:
        ts = datetime.fromisoformat(req.timestamp)
    except Exception as e:
        logger.error(f"Invalid timestamp '{req.timestamp}': {e}")
        raise HTTPException(status_code=400, detail=f"Invalid timestamp: {e}")

    # Decode full screen image
    img_bytes = base64.b64decode(req.image_base64)
    image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")

    # 生成 frame_id
    base_frame_id = ts.strftime("%Y%m%d_%H%M%S_") + f"{ts.microsecond:06d}"
    frame_id = f"frame_{base_frame_id}_{req.monitor_id}"

    # ========== 0. 检测当前聚焦窗口 ==========
    focused_app, focused_win = get_focused_window()

    # ========== 1. 处理全屏截图 ==========
    
    # 保存到临时文件用于视频压缩
    if temp_frame_buffer is not None:
        temp_image_path, batch_ready = temp_frame_buffer.add_full_screen_frame(
            frame_id=frame_id,
            image=image,
            timestamp=ts,
            monitor_id=req.monitor_id,
            metadata=req.metadata
        )
        
        # 如果批次就绪，触发压缩
        if batch_ready:
            _check_and_compress_batches()
    else:
        # 回退到原有的JPEG存储方式
        date_dir = config.IMAGE_STORAGE_PATH
        date_path = Path(date_dir) / ts.strftime("%Y%m%d")
        date_path.mkdir(parents=True, exist_ok=True)
        image_filename = f"{base_frame_id}.jpg"
        temp_image_path = str((date_path / image_filename).resolve())
        image.save(temp_image_path, format="JPEG", quality=config.IMAGE_QUALITY)

    # ========== 2. Embedding 和 OCR 处理 ==========
    logger.info(f"store_frame: {frame_id} step=embedding")
    # Compute image embedding
    embedding = encoder.encode_image(image)
    logger.info(f"store_frame: {frame_id} step=embedding_done")

    # Full-screen frame OCR is deferred: we collect sub_frame OCR results first,
    # then combine them as the frame's ocr_text (labeled by app_name).
    # This avoids mixing text from unrelated windows in a single OCR pass.
    ocr_engine_name = getattr(ocr_engine, 'engine_name', 'auto') if ocr_engine else "pending"

    # Collector for sub_frame OCR results: [(app_name, ocr_text, confidence), ...]
    sub_frame_ocr_parts = []

    # ========== 3. 处理窗口截图 ==========
    
    sub_frame_ids = []
    windows_to_process = []
    
    # 如果前端提供了窗口信息，使用前端数据
    if req.windows:
        for win in req.windows:
            try:
                win_bytes = base64.b64decode(win.image_base64)
                win_image = PILImage.open(io.BytesIO(win_bytes)).convert("RGB")
                windows_to_process.append({
                    "app_name": win.app_name,
                    "window_name": win.window_name,
                    "image": win_image
                })
            except Exception as e:
                logger.warning(f"Failed to decode window image for {win.app_name}: {e}")
    
    # 如果前端没有提供窗口信息，且启用了后端窗口捕获，使用screencap_rs
    elif ENABLE_BACKEND_WINDOW_CAPTURE and USE_SCREENCAP_RS and screencap_rs_module is not None:
        try:
            logger.info(f"store_frame: {frame_id} step=capture_windows_start")
            captured_windows = screencap_rs_module.capture_all_windows(
                include_minimized=False,
                filter_system=True
            )
            logger.info(f"store_frame: {frame_id} step=capture_windows_done count={len(captured_windows)}")
            for cw in captured_windows:
                try:
                    # 将PNG bytes转换为PIL Image
                    png_bytes = cw.get_image_bytes()
                    win_image = PILImage.open(io.BytesIO(png_bytes)).convert("RGB")
                    windows_to_process.append({
                        "app_name": cw.info.app_name,
                        "window_name": cw.info.title,
                        "image": win_image
                    })
                except Exception as e:
                    logger.debug(f"Failed to process captured window {cw.info.app_name}: {e}")
            logger.debug(f"Backend captured {len(windows_to_process)} windows")
        except Exception as e:
            logger.warning(f"Backend window capture failed: {e}")
    
    # 处理窗口（帧差去重 + embedding + app_name/window_name 记录）
    if windows_to_process and temp_frame_buffer is not None:
        from utils.data_models import WindowFrame as WF
        from core.capture.window_capturer import calculate_image_hash
        
        # 收集所有窗口的应用名称和窗口名称进行持久化
        current_apps = []
        app_window_pairs = []
        for win_data in windows_to_process:
            app_name = win_data.get("app_name")
            window_name = win_data.get("window_name")
            if app_name:
                current_apps.append(app_name)
                if window_name:
                    app_window_pairs.append((app_name, window_name))

        if current_apps:
            app_name_manager.add_apps(current_apps)
        if app_window_pairs:
            app_name_manager.add_window_pairs(app_window_pairs)
        
        for i, win_data in enumerate(windows_to_process):
            try:
                win_image = win_data["image"]
                app_name = win_data["app_name"]
                window_name = win_data["window_name"]
                
                # Window-level frame diff dedup
                if window_diff_detector is not None:
                    win_hash = calculate_image_hash(win_image)
                    wf = WF(
                        app_name=app_name,
                        window_name=window_name,
                        image=win_image,
                        image_hash=win_hash,
                        timestamp=ts,
                    )
                    diff_result = window_diff_detector.check_window_diff(wf)
                    if not diff_result.should_store:
                        logger.debug(
                            f"Skipping duplicate window {app_name}/{window_name} "
                            f"(diff={diff_result.diff_score:.4f})"
                        )
                        continue

                # Skip solid-color / black-screen window images
                if is_solid_color_image(win_image):
                    logger.debug(
                        f"Skipping solid-color window: {app_name}/{window_name}"
                    )
                    continue

                # 生成 sub_frame_id
                safe_app = app_name.replace(" ", "_").replace("/", "_")[:20]
                sub_frame_id = f"subframe_{safe_app}_{base_frame_id}_{i}"
                
                # 对窗口帧做 embedding
                logger.info(f"store_frame: {frame_id} win={app_name} step=win_embedding")
                win_embedding = encoder.encode_image(win_image)
                logger.info(f"store_frame: {frame_id} win={app_name} step=win_embedding_done")

                # 对窗口帧做 region OCR
                logger.info(f"store_frame: {frame_id} win={app_name} step=win_ocr_start")
                win_ocr_text = ""
                win_ocr_json = ""
                win_ocr_engine_name = "none"
                win_ocr_conf = 0.0
                win_ocr_regions_stored = False
                if region_ocr_engine is not None:
                    try:
                        win_regions = region_ocr_engine.recognize_regions(win_image)
                        if win_regions:
                            win_w, win_h = win_image.size
                            win_ocr_engine_name = getattr(ocr_engine, 'engine_name', 'auto')
                            sqlite_storage.store_ocr_with_regions(
                                sub_frame_id=sub_frame_id,
                                regions=win_regions,
                                ocr_engine=win_ocr_engine_name,
                                image_width=win_w,
                                image_height=win_h,
                            )
                            win_ocr_text = "\n".join(
                                r.get("text", "") for r in win_regions if r.get("text")
                            )
                            win_ocr_regions_stored = True
                            total_len = sum(len(r.get("text", "")) for r in win_regions)
                            if total_len > 0:
                                win_ocr_conf = sum(
                                    len(r.get("text", "")) * r.get("ocr_confidence", 0.0)
                                    for r in win_regions
                                ) / total_len
                    except Exception as e:
                        logger.warning(f"Region OCR failed for window {app_name}: {e}")

                logger.info(f"store_frame: {frame_id} win={app_name} step=win_ocr_done len={len(win_ocr_text)}")

                # Collect for frame-level combined OCR
                if win_ocr_text:
                    sub_frame_ocr_parts.append((app_name, win_ocr_text, win_ocr_conf))

                # 保存到临时文件
                win_temp_path, win_batch_ready = temp_frame_buffer.add_window_frame(
                    sub_frame_id=sub_frame_id,
                    image=win_image,
                    timestamp=ts,
                    app_name=app_name,
                    window_name=window_name,
                    parent_frame_id=frame_id  # 关联全屏帧ID
                )

                # Immediately create sub_frame record and mapping in SQLite
                # (the video_chunk reference will be back-filled on compression)
                if sqlite_storage is not None:
                    sqlite_storage.store_sub_frame(
                        sub_frame_id=sub_frame_id,
                        timestamp=ts,
                        window_chunk_id=0,
                        offset_index=0,
                        app_name=app_name,
                        window_name=window_name,
                    )
                    sqlite_storage.add_frame_subframe_mapping(
                        frame_id=frame_id,
                        sub_frame_id=sub_frame_id,
                    )

                # 将窗口帧的 embedding 也存储到 LanceDB（通过 batch_write_buffer）
                sub_frame_data = {
                    "frame_id": sub_frame_id,  # 使用 sub_frame_id 作为 frame_id
                    "timestamp": ts,
                    "image": win_image,
                    "embedding": win_embedding,
                    "ocr_text": win_ocr_text,
                    "image_path": win_temp_path,
                    "ocr_text_json": win_ocr_json,
                    "ocr_engine": win_ocr_engine_name,
                    "ocr_confidence": win_ocr_conf,
                    "device_name": f"{app_name}/{window_name}",
                    "metadata": {
                        "app_name": app_name,
                        "window_name": window_name,
                        "parent_frame_id": frame_id,
                        "is_sub_frame": True
                    },
                    "app_name": app_name,  # sub_frame 类型，填写 app_name
                    "window_name": window_name,  # sub_frame 类型，填写 window_name
                    "_ocr_regions_stored": win_ocr_regions_stored,
                }
                batch_write_buffer.add_frame(sub_frame_data)

                # Activity cluster assignment (real-time)
                if cluster_manager is not None:
                    try:
                        logger.info(f"store_frame: {frame_id} win={app_name} step=cluster_assign")
                        activity_label = cluster_manager.assign_frame(
                            app_name=app_name,
                            frame_id=sub_frame_id,
                            embedding=win_embedding,
                            image=win_image,
                            ocr_text=win_ocr_text,
                            timestamp=ts.isoformat(),
                            window_name=window_name,
                        )
                        logger.info(f"store_frame: {frame_id} win={app_name} step=cluster_done label={activity_label}")
                        if activity_label:
                            logger.debug(f"Assigned {sub_frame_id} -> '{activity_label}'")
                        _log_non_committed_cluster_result(app_name, sub_frame_id)
                    except Exception as e:
                        logger.debug(f"Cluster assign failed for {sub_frame_id}: {e}")

                sub_frame_ids.append(sub_frame_id)

                # 如果窗口批次就绪，触发压缩
                if win_batch_ready:
                    _check_and_compress_batches()
                    
            except Exception as e:
                logger.warning(f"Failed to process window {win_data.get('app_name', 'unknown')}: {e}")
                continue
    
    # ========== 4. 全屏应用检测：如果聚焦的应用不在已捕获的窗口中，创建合成子帧 ==========
    if focused_app and temp_frame_buffer is not None:
        captured_app_names = {w.get("app_name", "") for w in windows_to_process}
        if focused_app not in captured_app_names:
            try:
                safe_focused = focused_app.replace(" ", "_").replace("/", "_")[:20]
                syn_sub_id = f"subframe_{safe_focused}_{base_frame_id}_fullscreen"

                # Run independent region OCR for this synthetic sub_frame
                # (do NOT reuse full-screen OCR — it contains text from all windows)
                syn_ocr_text = ""
                syn_ocr_engine_name = "none"
                syn_ocr_conf = 0.0
                syn_ocr_regions_stored = False
                if region_ocr_engine is not None:
                    try:
                        syn_regions = region_ocr_engine.recognize_regions(image)
                        if syn_regions:
                            syn_img_w, syn_img_h = image.size
                            syn_ocr_engine_name = getattr(ocr_engine, 'engine_name', 'auto')
                            sqlite_storage.store_ocr_with_regions(
                                sub_frame_id=syn_sub_id,
                                regions=syn_regions,
                                ocr_engine=syn_ocr_engine_name,
                                image_width=syn_img_w,
                                image_height=syn_img_h,
                            )
                            syn_ocr_text = "\n".join(
                                r.get("text", "") for r in syn_regions if r.get("text")
                            )
                            syn_ocr_regions_stored = True
                            total_len = sum(len(r.get("text", "")) for r in syn_regions)
                            if total_len > 0:
                                syn_ocr_conf = sum(
                                    len(r.get("text", "")) * r.get("ocr_confidence", 0.0)
                                    for r in syn_regions
                                ) / total_len
                    except Exception as e:
                        logger.warning(f"Region OCR failed for synthetic sub_frame {focused_app}: {e}")

                # SQLite: sub_frames record (window_chunk_id=0 marks it as synthetic)
                if sqlite_storage is not None:
                    sqlite_storage.store_sub_frame(
                        sub_frame_id=syn_sub_id,
                        timestamp=ts,
                        window_chunk_id=0,
                        offset_index=0,
                        app_name=focused_app,
                        window_name=focused_win or focused_app,
                    )
                    sqlite_storage.add_frame_subframe_mapping(
                        frame_id=frame_id,
                        sub_frame_id=syn_sub_id,
                    )
                    # Pre-create frames table entry (image_path updated on compression)
                    # Pass empty ocr_text since region OCR already wrote to ocr_text
                    sqlite_storage.store_frame_with_ocr(
                        frame_id=syn_sub_id,
                        timestamp=ts,
                        image_path=temp_image_path,
                        ocr_text="" if syn_ocr_regions_stored else syn_ocr_text,
                        ocr_text_json="",
                        ocr_engine=syn_ocr_engine_name,
                        ocr_confidence=syn_ocr_conf,
                        device_name=f"{focused_app}/{focused_win}",
                        app_name=focused_app,
                        window_name=focused_win or focused_app,
                    )

                # LanceDB: reuse the same embedding (no re-encoding)
                syn_frame_data = {
                    "frame_id": syn_sub_id,
                    "timestamp": ts,
                    "image": image,
                    "embedding": embedding,
                    "ocr_text": syn_ocr_text,
                    "image_path": temp_image_path,
                    "ocr_text_json": "",
                    "ocr_engine": syn_ocr_engine_name,
                    "ocr_confidence": syn_ocr_conf,
                    "device_name": f"{focused_app}/{focused_win}",
                    "metadata": {"is_fullscreen_synthetic": True, "parent_frame_id": frame_id},
                    "app_name": focused_app,
                    "window_name": focused_win or focused_app,
                    "_ocr_regions_stored": syn_ocr_regions_stored,
                }
                batch_write_buffer.add_frame(syn_frame_data)

                if cluster_manager is not None:
                    try:
                        activity_label = cluster_manager.assign_frame(
                            app_name=focused_app,
                            frame_id=syn_sub_id,
                            embedding=embedding,
                            image=image,
                            ocr_text=syn_ocr_text,
                            timestamp=ts.isoformat(),
                            window_name=focused_win or "",
                        )
                        if activity_label:
                            logger.debug(f"Assigned {syn_sub_id} -> '{activity_label}'")
                        _log_non_committed_cluster_result(focused_app, syn_sub_id)
                    except Exception as e:
                        logger.debug(f"Cluster assign failed for {syn_sub_id}: {e}")

                sub_frame_ids.append(syn_sub_id)

                # Collect for frame-level combined OCR
                if syn_ocr_text:
                    sub_frame_ocr_parts.append((focused_app, syn_ocr_text, syn_ocr_conf))

                logger.debug(
                    f"Created synthetic sub_frame {syn_sub_id} for "
                    f"full-screen app {focused_app}/{focused_win}"
                )
            except Exception as e:
                logger.warning(f"Failed to create synthetic sub_frame for {focused_app}: {e}")

    # ========== 5. 组合全屏帧 OCR：拼接所有 sub_frame 的 OCR 文本 ==========
    combined_ocr_text = ""
    combined_ocr_conf = 0.0
    if sub_frame_ocr_parts:
        sections = []
        for app, text, conf in sub_frame_ocr_parts:
            sections.append(f"[{app}]\n{text}")
        combined_ocr_text = "\n\n".join(sections)
        # Weighted avg confidence
        total_len = sum(len(t) for _, t, _ in sub_frame_ocr_parts)
        if total_len > 0:
            combined_ocr_conf = sum(len(t) * c for _, t, c in sub_frame_ocr_parts) / total_len

    # Store frame's combined OCR to ocr_text (for FTS search)
    if combined_ocr_text and sqlite_storage is not None:
        sqlite_storage.store_frame_with_ocr(
            frame_id=frame_id,
            timestamp=ts,
            image_path=temp_image_path,
            ocr_text=combined_ocr_text,
            ocr_text_json="",
            ocr_engine=ocr_engine_name,
            ocr_confidence=combined_ocr_conf,
            device_name=f"monitor_{req.monitor_id}",
            metadata=req.metadata or {"size": image.size, "monitor_id": req.monitor_id},
            focused_app_name=focused_app or None,
            focused_window_name=focused_win or None,
        )

    # Write frame to batch buffer (LanceDB + frames table via BatchWriteBuffer)
    frame_data = {
        "frame_id": frame_id,
        "timestamp": ts,
        "image": image,
        "embedding": embedding,
        "ocr_text": combined_ocr_text,
        "image_path": temp_image_path,
        "ocr_text_json": "",
        "ocr_engine": ocr_engine_name,
        "ocr_confidence": combined_ocr_conf,
        "device_name": f"monitor_{req.monitor_id}",
        "metadata": req.metadata or {"size": image.size, "monitor_id": req.monitor_id},
        "app_name": None,
        "window_name": None,
        "focused_app_name": focused_app or None,
        "focused_window_name": focused_win or None,
        "_ocr_regions_stored": True,  # Already stored above, skip duplicate in BatchWriteBuffer
    }
    batch_write_buffer.add_frame(frame_data)

    _elapsed = time_module.time() - _t0
    logger.info(f"store_frame: {frame_id} done in {_elapsed:.2f}s, {len(sub_frame_ids)} windows (focused={focused_app})")

    # Check if cluster recalculation is needed (runs in background thread)
    if cluster_manager is not None and cluster_manager.should_recalculate():
        if vector_storage is not None and vector_storage.table is not None:
            threading.Thread(
                target=cluster_manager.recalculate,
                args=(vector_storage.table,),
                daemon=True,
            ).start()

    return {"status": "ok", "frame_id": frame_id, "sub_frame_count": len(sub_frame_ids)}


@app.post("/api/query_rag_with_time", response_model=QueryRagWithTimeResponse)
def query_rag_with_time(req: QueryRagWithTimeRequest):
    """
    Perform RAG query with time range filtering, rerank, and VLM analysis.
    Mirrors CLI / GUI RAG-with-time behavior, but returns JSON for remote GUI.
    """
    # 组件已在启动时预加载，直接使用
    assert encoder is not None
    assert vector_storage is not None
    assert sqlite_storage is not None
    assert vlm is not None

    enable_hybrid = req.enable_hybrid if req.enable_hybrid is not None else config.ENABLE_HYBRID
    enable_rerank = req.enable_rerank if req.enable_rerank is not None else config.ENABLE_RERANK
    
    # 如果启用了 rerank，检查 reranker 是否已初始化
    if enable_rerank and reranker is None:
        raise HTTPException(
            status_code=500, 
            detail="Reranker is enabled but not initialized. Please set ENABLE_RERANK=True or disable rerank in the request."
        )

    # 1) 显式时间（来自前端）
    explicit_start = _ensure_utc(datetime.fromisoformat(req.start_time)) if req.start_time else None
    explicit_end = _ensure_utc(datetime.fromisoformat(req.end_time)) if req.end_time else None

    # 2) 默认时间范围：先用显式时间，占位
    start_time = explicit_start
    end_time = explicit_end

    # 3) 调用 LLM 做 query rewrite + time_range 解析 + app_name 过滤
    #    - 无论是否开启 rewrite，都允许 LLM 解析 time_range 和 app_name
    #    - 是否采用扩写结果由 ENABLE_LLM_REWRITE 决定
    dense_queries = [req.query]
    sparse_queries = [req.query]
    llm_time_range = None
    related_apps = None
    unrelated_apps = None
    window_filters = None
    try:
        dense_llm, sparse_llm, llm_time_range, related_apps, unrelated_apps, window_filters = rewrite_and_time(
            req.query,
            enable_rewrite=config.ENABLE_LLM_REWRITE,
            enable_time=True,  # 总是允许解析时间范围
            expand_n=config.QUERY_REWRITE_NUM,
        )
        if config.ENABLE_LLM_REWRITE:
            dense_queries = dense_llm
            sparse_queries = sparse_llm
        
        # 确保 LLM 返回的时间也是 UTC 化的
        if llm_time_range:
            llm_time_range = (_ensure_utc(llm_time_range[0]), _ensure_utc(llm_time_range[1]))
    except Exception as e:
        logger.warning(f"rewrite_and_time failed, fallback to original query: {e}")
        llm_time_range = None

    # 4) 合并显式时间和 LLM 推理时间：取交集
    #    规则：
    #    - 如果两者都存在，start = max(explicit_start, llm_start), end = min(explicit_end, llm_end)
    #    - 如果只有显式时间，用显式时间
    #    - 如果只有 LLM 时间，用 LLM 时间
    if llm_time_range is not None:
        llm_start, llm_end = llm_time_range

        # 计算交集起点
        if explicit_start and llm_start:
            start_time = max(explicit_start, llm_start)
        elif explicit_start and not llm_start:
            start_time = explicit_start
        elif not explicit_start and llm_start:
            start_time = llm_start

        # 计算交集终点
        if explicit_end and llm_end:
            end_time = min(explicit_end, llm_end)
        elif explicit_end and not llm_end:
            end_time = explicit_end
        elif not explicit_end and llm_end:
            end_time = llm_end

        # 如果交集为空，优先保留显式时间；如果显式时间不存在，则保留 LLM 时间
        if start_time and end_time and start_time > end_time:
            if explicit_start or explicit_end:
                start_time = explicit_start
                end_time = explicit_end
            else:
                start_time, end_time = llm_start, llm_end

    # Log final search parameters
    logger.info(f"Final search parameters for query '{req.query}':")
    if start_time or end_time:
        logger.info(f"  • Time range: {start_time} to {end_time}")
    else:
        logger.info(f"  • Time range: Global (None)")
    logger.info(f"  • Related apps: {related_apps}")
    logger.info(f"  • Unrelated apps: {unrelated_apps}")
    if window_filters:
        logger.info(f"  • Included windows (by app): {window_filters.get('include')}")
        logger.info(f"  • Excluded windows (by app): {window_filters.get('exclude')}")
    else:
        logger.info(f"  • Window filters: None")

    top_k = config.MAX_IMAGES_TO_LOAD

    # Dense search
    def _dense_search() -> List[Dict]:
        frames: List[Dict] = []
        for q in dense_queries:
            emb = encoder.encode_text(q)
            
            # 根据 search_type 选择搜索表
            if req.search_type == "text":
                logger.info(f"Performing OCR text dense search for: {q}")
                res = vector_storage.search_ocr(
                    emb,
                    top_k=top_k,
                    start_time=start_time,
                    end_time=end_time,
                    related_apps=related_apps,
                    unrelated_apps=unrelated_apps,
                    window_filters=window_filters,
                )
            else:
                logger.info(f"Performing image dense search for: {q}")
                res = vector_storage.search(
                    emb,
                    top_k=top_k,
                    start_time=start_time,
                    end_time=end_time,
                    related_apps=related_apps,
                    unrelated_apps=unrelated_apps,
                    window_filters=window_filters,
                )
            frames.extend(res)
        return frames

    # Sparse search via SQLite FTS5
    def _sparse_search() -> List[Dict]:
        if not enable_hybrid:
            return []
        frames: List[Dict] = []
        for q in sparse_queries:
            # TODO: SQLiteStorage.search_by_text currently doesn't support related_apps/unrelated_apps
            # We filter by time first, then we could add app filtering here if needed.
            res = sqlite_storage.search_by_text(q, limit=top_k)
            if start_time or end_time:
                res = filter_by_time(res, (start_time, end_time))
            
            # Apply app filtering for sparse results
            if related_apps:
                res = [r for r in res if r.get("app_name") in related_apps]
            elif unrelated_apps:
                res = [r for r in res if r.get("app_name") not in unrelated_apps]
                
            for r in res:
                fid = r.get("frame_id")
                if not fid:
                    continue
                frame = {
                    "frame_id": fid,
                    "timestamp": r.get("timestamp"),
                    "image_path": r.get("image_path"),
                    "ocr_text": r.get("ocr_text", ""),
                    "distance": 1.0,
                    "metadata": r.get("metadata", {}),
                    "_from_sparse": True,
                }
                frames.append(frame)
        return frames

    dense_results = _dense_search()
    sparse_results = _sparse_search()

    # Merge & dedup
    frames: List[Dict] = []
    seen = set()
    for r in dense_results:
        fid = r.get("frame_id")
        if not fid or fid in seen:
            continue
        seen.add(fid)
        frames.append(r)
    for r in sparse_results:
        fid = r.get("frame_id")
        if not fid or fid in seen:
            continue
        seen.add(fid)
        frames.append(r)

    logger.info(
        f"RAG dense 原始 {len(dense_results)} 条, sparse 原始 {len(sparse_results)} 条, "
        f"去重后 {len(frames)} 张"
    )

    # Activity label post-filter (queries activity DB, not main DB)
    if req.activity_label and sqlite_storage is not None:
        try:
            conn_al = sqlite_storage._get_activity_connection()
            cursor_al = conn_al.cursor()
            cursor_al.execute(
                "SELECT sub_frame_id FROM activity_assignments WHERE activity_label LIKE ?",
                (f"%{req.activity_label}%",),
            )
            matching_ids = {r["sub_frame_id"] for r in cursor_al.fetchall()}
            conn_al.close()
            before = len(frames)
            frames = [f for f in frames if f.get("frame_id") in matching_ids]
            logger.info(f"activity_label filter '{req.activity_label}': {before} -> {len(frames)}")
        except Exception as e:
            logger.warning(f"activity_label filter failed: {e}")

    if not frames:
        return QueryRagWithTimeResponse(answer="在指定时间范围内未找到相关的屏幕记录。", frames=[])

    # Load images for rerank + VLM (path 可能是文件路径或 video_chunk:id:offset / window_chunk:id:offset)
    loaded_frames: List[Dict] = []
    for f in frames:
        path = f.get("image_path")
        if not path:
            continue
        img = _load_image_from_path(path)
        resolved_path = path
        if img is None and sqlite_storage is not None:
            # 可能是已压缩的帧：LanceDB 里仍是临时路径，用 frame_id 从 SQLite 取最新 image_path 再试
            fid = f.get("frame_id")
            if fid:
                try:
                    conn = sqlite_storage._get_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT image_path FROM frames WHERE frame_id = ?", (fid,))
                    row = cursor.fetchone()
                    conn.close()
                    if row and row["image_path"] and row["image_path"] != path:
                        resolved_path = row["image_path"]
                        img = _load_image_from_path(resolved_path)
                    if img is None:
                        # 子帧可能在 sub_frames 表，用 get_sub_frame_video_info 得到 file_path + offset
                        info = sqlite_storage.get_sub_frame_video_info(fid)
                        if info and info.get("file_path") and info.get("offset_index") is not None and ffmpeg_extractor:
                            img = ffmpeg_extractor.extract_frame_by_index(
                                info["file_path"], info["offset_index"], info.get("fps", VIDEO_FPS)
                            )
                            if img is not None:
                                resolved_path = f"window_chunk:{info['window_chunk_id']}:{info['offset_index']}"
                except Exception as e:
                    logger.debug(f"Resolve image_path for {fid}: {e}")
        if img is not None:
            f["image"] = img
            f["image_path"] = resolved_path
            loaded_frames.append(f)
        else:
            logger.debug(f"Skip frame (image not loadable): {f.get('frame_id')} path={path[:80]}")

    failed_count = len(frames) - len(loaded_frames)
    logger.info(f"图片加载成功 {len(loaded_frames)} 张，失败 {failed_count} 张")

    if not loaded_frames:
        return QueryRagWithTimeResponse(answer="检索到的图片无法加载。", frames=[])

    # Rerank
    frames_for_vlm = loaded_frames
    if enable_rerank:
        frames_for_vlm = reranker.rerank(
            query=req.query,
            frames=loaded_frames,
            top_k=config.RERANK_TOP_K,
        )
        if not frames_for_vlm:
            return QueryRagWithTimeResponse(answer="Rerank 后没有图片，无法进行 VLM 分析。", frames=[])

    # VLM analysis
    images = [f["image"] for f in frames_for_vlm]
    timestamps = [f.get("timestamp") for f in frames_for_vlm]

    system_prompt = (
        "You are a helpful visual assistant. You analyze screenshots to answer user questions. "
        "Always respond in Chinese (中文回答)."
    )
    prompt = f"""User Question: {req.query}

Please directly answer the user's question first, then provide supporting evidence from the screenshots below.
Focus on what the user was doing and how the visual content relates to their question."""

    answer = vlm._call_vlm(
        prompt,
        images,
        num_images=len(images),
        image_timestamps=timestamps if timestamps else None,
        system_prompt=system_prompt,
    )

    # Build response frames (with base64 thumbnails for GUI)
    resp_frames: List[FrameResult] = []
    for f in frames_for_vlm:
        img = f.get("image")
        img_b64 = None
        if img is not None:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        ts = f.get("timestamp")
        ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)
        resp_frames.append(
            FrameResult(
                frame_id=f.get("frame_id", ""),
                timestamp=ts_str,
                image_base64=img_b64,
                image_path=f.get("image_path"),
                ocr_text=f.get("ocr_text", ""),
            )
        )

    return QueryRagWithTimeResponse(answer=answer, frames=resp_frames)


def _load_image_from_path(path_str: str):
    """
    从 path 加载 PIL Image。path 可以是：
    - video_chunk:{chunk_id}:{offset_index}
    - window_chunk:{chunk_id}:{offset_index}
    - 绝对或相对文件路径
    返回 PIL.Image 或 None（加载失败或 chunk 尚未就绪）。
    """
    if not path_str or not path_str.strip():
        return None
    path_str = str(path_str).strip()
    # video_chunk / window_chunk：从数据库取文件路径再用 FFmpeg 抽帧
    if path_str.startswith("video_chunk:") or path_str.startswith("window_chunk:"):
        parts = path_str.split(":")
        if len(parts) != 3:
            return None
        try:
            chunk_id = int(parts[1])
            offset_index = int(parts[2])
        except ValueError:
            return None
        if chunk_id <= 0:
            return None
        if sqlite_storage is None or ffmpeg_extractor is None:
            return None
        try:
            conn = sqlite_storage._get_connection()
            cursor = conn.cursor()
            if path_str.startswith("video_chunk:"):
                cursor.execute("SELECT file_path, fps FROM video_chunks WHERE id = ?", (chunk_id,))
            else:
                cursor.execute("SELECT file_path, fps FROM window_chunks WHERE id = ?", (chunk_id,))
            row = cursor.fetchone()
            conn.close()
            if not row or not row["file_path"]:
                return None
            video_path = row["file_path"]
            fps = row["fps"] or VIDEO_FPS
            if not Path(video_path).exists():
                return None
            return ffmpeg_extractor.extract_frame_by_index(video_path, offset_index, fps)
        except Exception as e:
            logger.debug(f"_load_image_from_path chunk {path_str}: {e}")
            return None
    # 文件路径
    if Path(path_str).is_absolute():
        final_path = Path(path_str)
    else:
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir
        cwd = Path.cwd().absolute()
        if "visualmem_storage" in path_str:
            final_path = project_root / path_str
            if not final_path.exists():
                final_path = cwd / path_str
        else:
            base_path = Path(config.IMAGE_STORAGE_PATH)
            if base_path.is_absolute():
                final_path = base_path / path_str
            else:
                final_path = project_root / base_path / path_str
                if not final_path.exists():
                    final_path = cwd / base_path / path_str
    if not final_path.exists() or not final_path.is_file():
        return None
    try:
        return PILImage.open(str(final_path)).convert("RGB")
    except Exception as e:
        logger.debug(f"_load_image_from_path file {path_str}: {e}")
        return None


@app.get("/api/image")
def get_image(path: str = Query(..., description="Image file path")):
    """
    获取图片文件（用于前端显示）
    支持：
    1. 绝对路径
    2. 相对路径（相对于项目根目录）
    3. video_chunk:{chunk_id}:{offset_index} 格式（从视频中提取帧）
    """
    try:
        path_str = str(path)
        
        # 处理 video_chunk 或 window_chunk 引用格式
        if path_str.startswith("video_chunk:") or path_str.startswith("window_chunk:"):
            # 格式: video_chunk:{chunk_id}:{offset_index} 或 window_chunk:{chunk_id}:{offset_index}
            parts = path_str.split(":")
            if len(parts) != 3:
                raise HTTPException(status_code=400, detail=f"Invalid chunk format: {path}")
            
            try:
                chunk_id = int(parts[1])
                offset_index = int(parts[2])
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid chunk format: {path}")
            
            # 从数据库获取视频文件路径
            if sqlite_storage is None:
                raise HTTPException(status_code=500, detail="Storage not initialized")
            
            conn = sqlite_storage._get_connection()
            cursor = conn.cursor()
            
            # 根据 chunk 类型查询不同的表
            if path_str.startswith("video_chunk:"):
                cursor.execute("SELECT file_path, fps FROM video_chunks WHERE id = ?", (chunk_id,))
            else:  # window_chunk
                cursor.execute("SELECT file_path, fps FROM window_chunks WHERE id = ?", (chunk_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                chunk_type = "video" if path_str.startswith("video_chunk:") else "window"
                raise HTTPException(status_code=404, detail=f"{chunk_type} chunk {chunk_id} not found")
            
            video_path = row["file_path"]
            fps = row["fps"] or VIDEO_FPS
            
            # 从视频中提取帧
            if ffmpeg_extractor is None:
                raise HTTPException(status_code=500, detail="FFmpeg extractor not initialized")
            
            image = ffmpeg_extractor.extract_frame_by_index(video_path, offset_index, fps)
            if image is None:
                raise HTTPException(status_code=404, detail=f"Failed to extract frame {offset_index} from video")
            
            # 返回图片
            import io
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)
            
            return Response(
                content=buffer.getvalue(),
                media_type="image/jpeg",
                headers={
                    "Cache-Control": "public, max-age=3600",  # 缓存1小时
                }
            )
        
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir  # gui_backend_server.py 在项目根目录
        cwd = Path.cwd().absolute()

        # 如果是绝对路径，直接使用
        if Path(path_str).is_absolute():
            final_path = Path(path_str)
        else:
            # 相对路径处理
            # 获取项目根目录（脚本所在目录的父目录，或当前工作目录）
            # 尝试多种方式找到项目根目录
            
            # 如果路径包含 visualmem_storage，尝试相对于项目根目录
            if 'visualmem_storage' in path_str:
                # 尝试相对于脚本目录（项目根目录）
                final_path = project_root / path_str
                if not final_path.exists():
                    # 如果不存在，尝试相对于当前工作目录
                    final_path = cwd / path_str
            else:
                # 路径不包含存储目录，尝试相对于 IMAGE_STORAGE_PATH
                # IMAGE_STORAGE_PATH 可能是相对路径或绝对路径
                base_path = Path(config.IMAGE_STORAGE_PATH)
                if base_path.is_absolute():
                    final_path = base_path / path_str
                else:
                    # 如果是相对路径，尝试相对于项目根目录
                    final_path = project_root / base_path / path_str
                    if not final_path.exists():
                        # 再尝试相对于当前工作目录
                        final_path = cwd / base_path / path_str
        
        # 确保路径存在且是文件
        if not final_path.exists() or not final_path.is_file():
            # 特殊处理：如果路径包含 temp_frames 且不存在，说明可能刚刚被压缩成视频了
            if 'temp_frames' in path_str:
                logger.info(f"Temp image not found, trying to find updated path in DB: {path_str}")
                try:
                    # 从路径中尝试提取时间戳和 monitor_id
                    # 路径格式: .../temp_frames/full_screen/monitor_0/20260126_103850_579000.png
                    # 或者是: .../temp_frames/windows/AppName_WindowName/20260126_103850_579000.png
                    parts = Path(path_str).parts
                    filename = parts[-1]  # 20260126_103850_579000.png
                    
                    if filename.endswith('.png'):
                        ts_str = filename[:-4]  # 20260126_103850_579000
                        
                        if 'full_screen' in path_str:
                            # 全屏帧: frame_{ts_str}_{monitor_id}
                            monitor_part = parts[-2] if len(parts) >= 2 else "monitor_0"
                            monitor_id = monitor_part.split('_')[1] if '_' in monitor_part else "0"
                            frame_id = f"frame_{ts_str}_{monitor_id}"
                            
                            # 查询数据库获取最新路径
                            if sqlite_storage:
                                conn = sqlite_storage._get_connection()
                                cursor = conn.cursor()
                                cursor.execute("SELECT image_path FROM frames WHERE frame_id = ?", (frame_id,))
                                row = cursor.fetchone()
                                conn.close()
                                
                                if row and row['image_path'] and row['image_path'] != path_str:
                                    logger.info(f"Found updated path for {frame_id}: {row['image_path']}")
                                    # 递归调用 get_image 处理新路径（可能是 video_chunk）
                                    return get_image(row['image_path'])
                        
                        elif 'windows' in path_str:
                            # 窗口子帧: subframe_{safe_app}_{ts_str}_{index}
                            # 注意：由于 safe_app 和 index 难以从路径反推，我们尝试模糊匹配 timestamp
                            if sqlite_storage:
                                conn = sqlite_storage._get_connection()
                                cursor = conn.cursor()
                                # 转换 ts_str (20260126_103850_579000) 到 ISO 格式的一部分进行匹配
                                # 或者直接匹配 sub_frame_id 包含 ts_str 的记录
                                cursor.execute("SELECT window_chunk_id, offset_index FROM sub_frames WHERE sub_frame_id LIKE ?", (f"%{ts_str}%",))
                                row = cursor.fetchone()
                                conn.close()
                                
                                if row and row["window_chunk_id"]:
                                    new_path = f"window_chunk:{row['window_chunk_id']}:{row['offset_index']}"
                                    logger.info(f"Found updated window_chunk path for temp window frame: {new_path}")
                                    return get_image(new_path)
                except Exception as e:
                    logger.warning(f"Failed to redirect stale temp path: {e}")

            logger.warning(f"=== [Frontend Image Load Failed] ===")
            logger.warning(f"Frontend requested image path: '{path}'")
            logger.warning(f"Image not found: {path}")
            logger.warning(f"  Resolved path: {final_path} (exists: {final_path.exists()})")
            logger.warning(f"  Project root (script dir): {project_root}")
            logger.warning(f"  Current working directory: {cwd}")
            logger.warning(f"  IMAGE_STORAGE_PATH: {config.IMAGE_STORAGE_PATH}")
            raise HTTPException(status_code=404, detail=f"Image not found: {path}")
        
        # 返回图片文件
        return FileResponse(
            str(final_path),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=3600",  # 缓存1小时
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"=== [Frontend Image Load Failed] ===")
        logger.error(f"Frontend requested image path: '{path}'")
        logger.error(f"Failed to serve image {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve image: {str(e)}")


@app.get("/api/recent_frames")
def get_recent_frames(minutes: int = 5):
    """
    获取最近 X 分钟内的帧
    """
    if sqlite_storage is None:
        return {"frames": []}
    
    try:
        # 获取最近的 X 分钟内的帧
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=minutes)
        frames = sqlite_storage.get_frames_in_timerange(
            start_time=start_time, 
            end_time=end_time,
            only_full_screen=True  # 仅获取全屏帧
        )
        
        recent_frames = []
        for f in frames:
            sub_frames = sqlite_storage.get_sub_frames_for_frame(f["frame_id"])
            sub_list = []
            for sf in sub_frames:
                sub_list.append({
                    "sub_frame_id": sf["sub_frame_id"],
                    "timestamp": sf["timestamp"].isoformat(),
                    "app_name": sf.get("app_name", ""),
                    "window_name": sf.get("window_name", ""),
                    "image_path": _resolve_sub_frame_image_path(sf),
                })
            recent_frames.append({
                "frame_id": f["frame_id"],
                "timestamp": f["timestamp"].isoformat(),
                "image_path": f["image_path"],
                "ocr_text": f["ocr_text"],
                "sub_frames": sub_list,
            })
        # print(f"Found {len(recent_frames)} frames in the last {minutes} minutes.")
        
        return {"frames": recent_frames}
    except Exception as e:
        logger.error(f"Failed to get recent frames: {e}")
        return {"frames": []}


@app.get("/api/date-range")
def get_date_range():
    """
    获取数据库中最早和最新的照片日期
    用于前端确定加载范围
    """
    if sqlite_storage is None:
        raise HTTPException(status_code=500, detail="SQLite storage not initialized")
    
    try:
        earliest_frame = sqlite_storage.get_earliest_frame()
        latest_frame = sqlite_storage.get_latest_frame()
        
        earliest_date = None
        latest_date = None
        
        if earliest_frame and earliest_frame.get("timestamp"):
            ts = earliest_frame["timestamp"]
            if isinstance(ts, datetime):
                earliest_date = ts.date().isoformat()
            else:
                earliest_date = ts.split('T')[0] if 'T' in str(ts) else str(ts)[:10]
        
        if latest_frame and latest_frame.get("timestamp"):
            ts = latest_frame["timestamp"]
            if isinstance(ts, datetime):
                latest_date = ts.date().isoformat()
            else:
                latest_date = ts.split('T')[0] if 'T' in str(ts) else str(ts)[:10]

        # print(f"date range: from {earliest_date} to {latest_date}")
        
        return DateRangeResponse(earliest_date=earliest_date, latest_date=latest_date)
    except Exception as e:
        logger.error(f"Failed to get date range: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get date range: {str(e)}")


@app.post("/api/frames/date/count")
def get_frames_count_by_date(req: GetFramesByDateRequest):
    """
    获取某一天的照片总数
    """
    if sqlite_storage is None:
        raise HTTPException(status_code=500, detail="SQLite storage not initialized")
    
    try:
        # 使用更稳健的日期范围查询，确保包含所有时区偏移
        # 格式：timestamp >= '2025-12-23' AND timestamp < '2025-12-24'
        start_time_str = f"{req.date}"
        
        # 计算下一天
        date_obj = datetime.fromisoformat(req.date)
        next_day = date_obj + timedelta(days=1)
        end_time_str = next_day.strftime("%Y-%m-%d")
        
        # 使用 COUNT 查询获取总数（只统计主帧：frame_* 开头，有 image_path）
        conn = sqlite_storage._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) as count
            FROM frames f
            WHERE f.timestamp >= ? AND f.timestamp < ?
            AND f.image_path IS NOT NULL AND f.image_path != ''
            AND f.frame_id LIKE 'frame_%'
        """, (start_time_str, end_time_str))
        
        row = cursor.fetchone()
        conn.close()
        
        total_count = row["count"] if row else 0
        
        return DateFrameCountResponse(date=req.date, total_count=total_count)
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to get frame count for date: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get frame count: {str(e)}")


@app.post("/api/frames/date")
def get_frames_by_date(req: GetFramesByDateRequest):
    """
    获取某一天的照片（支持分页）
    
    参数：
    - date: 日期 (YYYY-MM-DD)
    - offset: 偏移量（默认0）
    - limit: 每页数量（默认50，可在前端修改）
    """
    if sqlite_storage is None:
        raise HTTPException(status_code=500, detail="SQLite storage not initialized")
    
    try:
        # 使用更稳健的日期范围查询
        start_time_str = f"{req.date}"
        date_obj = datetime.fromisoformat(req.date)
        next_day = date_obj + timedelta(days=1)
        end_time_str = next_day.strftime("%Y-%m-%d")
        
        # 验证 limit 范围
        limit = min(max(1, req.limit), 200)  # 限制在 1-200 之间
        offset = max(0, req.offset)  # 确保 offset 非负
        
        # 直接从 SQLite 获取该天的帧（使用 OFFSET 和 LIMIT 进行分页）
        # 注意：现在排序是 ASC（从早到晚），所以 offset=0 是最早的，offset=50 是第 51-100 张
        conn = sqlite_storage._get_connection()
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
            WHERE f.timestamp >= ? AND f.timestamp < ?
            AND f.frame_id LIKE 'frame_%'
            ORDER BY f.timestamp ASC
            LIMIT ? OFFSET ?
        """, (start_time_str, end_time_str, limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        # 转换为 API 响应格式（只返回路径，不返回 base64），并附带子帧（含可用的 image_path）
        result = []
        for row in rows:
            # 只返回有 image_path 的帧
            if not row["image_path"]:
                continue
            fid = row["frame_id"]
            sub_frames = sqlite_storage.get_sub_frames_for_frame(fid)
            sub_list = []
            for sf in sub_frames:
                sub_list.append({
                    "sub_frame_id": sf["sub_frame_id"],
                    "timestamp": sf["timestamp"].isoformat(),
                    "app_name": sf.get("app_name", ""),
                    "window_name": sf.get("window_name", ""),
                    "image_path": _resolve_sub_frame_image_path(sf),
                })
            ts = datetime.fromisoformat(row["timestamp"])
            ts_str = ts.isoformat()
            result.append({
                "frame_id": fid,
                "timestamp": ts_str,
                "image_path": row["image_path"],
                "ocr_text": row["ocr_text"] or "",
                "sub_frames": sub_list,
            })
        
        # logger.info(f"Returned {len(result)} frames for date {req.date} (offset={offset}, limit={limit})")
        return result
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to get frames by date: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get frames: {str(e)}")


@app.post("/api/frames")
def get_frames_by_date_range(req: GetFramesByDateRangeRequest):
    """
    根据日期范围获取帧列表（用于时间轴浏览）
    返回本地图片路径，不包含 base64 编码的图片数据
    注意：返回指定日期范围内的所有数据，不进行分页（前端通过调整日期范围来分页）
    """
    if sqlite_storage is None:
        raise HTTPException(status_code=500, detail="SQLite storage not initialized")
    
    try:
        # 使用更稳健的日期范围查询
        start_time_str = f"{req.start_date}"
        
        # 计算结束日期的下一天
        end_date_obj = datetime.fromisoformat(req.end_date)
        next_day = end_date_obj + timedelta(days=1)
        end_time_str = next_day.strftime("%Y-%m-%d")
        
        # 从 SQLite 获取时间范围内的所有帧
        # 不使用 offset/limit，因为前端通过调整日期范围来控制加载
        all_frames = sqlite_storage.get_frames_in_timerange(
            start_time=start_time_str, # 传递字符串，sqlite_storage 会处理
            end_time=end_time_str,
            limit=100000,  # 设置一个较大的 limit，确保获取所有数据
            only_full_screen=True  # 仅获取全屏帧用于时间轴
        )
        
        # 转换为 API 响应格式（只返回路径，不返回 base64），并附带子帧（含可用的 image_path）
        result = []
        for frame in all_frames:
            fid = frame.get("frame_id", "")
            sub_frames = sqlite_storage.get_sub_frames_for_frame(fid) if fid else []
            sub_list = []
            for sf in sub_frames:
                sub_list.append({
                    "sub_frame_id": sf["sub_frame_id"],
                    "timestamp": sf["timestamp"].isoformat(),
                    "app_name": sf.get("app_name", ""),
                    "window_name": sf.get("window_name", ""),
                    "image_path": _resolve_sub_frame_image_path(sf),
                })
            ts = frame.get("timestamp")
            ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)
            result.append({
                "frame_id": fid,
                "timestamp": ts_str,
                "image_path": frame.get("image_path", ""),
                "ocr_text": frame.get("ocr_text", ""),
                "sub_frames": sub_list,
            })
        
        # logger.info(f"Returned {len(result)} frames for date range {req.start_date} to {req.end_date}")
        return result
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to get frames by date range: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get frames: {str(e)}")


# 注意：在 remote 模式下，录屏功能由 Electron 前端完成
# 前端负责截屏和帧差过滤，然后通过 /api/store_frame 发送到后端
# 后端只负责 embedding 和 OCR 处理
@app.post("/api/recording/stop")
def stop_recording_api():
    """
    停止录制信号：触发后端所有缓冲区的强制刷新
    - 视频帧缓冲区 -> 压缩成MP4
    - 批量写入缓冲区 -> 写入LanceDB和SQLite
    """
    logger.info("收到前端录制停止信号...")
    
    # 1. 先刷新视频帧缓冲区（压缩剩余帧为MP4）
    if temp_frame_buffer is not None:
        logger.info("刷新视频帧缓冲区...")
        _flush_all_video_buffers()
    
    # 2. 再刷新批量写入缓冲区
    if batch_write_buffer is not None:
        logger.info("刷新批量写入缓冲区...")
        batch_write_buffer._flush_buffer()

    # 3. 输出本次录制的聚类统计并重置计数器
    if cluster_manager is not None:
        stats = cluster_manager.get_assignment_stats()
        total = stats["total_frames"]
        vlm = stats["vlm_called_frames"]
        ratio = stats["vlm_call_ratio"]
        logger.info(
            f"Recording cluster stats: {vlm}/{total} frames needed VLM labeling ({ratio:.1%})"
        )
        cluster_manager.reset_assignment_stats()

    return {"status": "success", "message": "All buffers flushed"}


@app.get("/api/video/extract_frame")
def extract_frame_from_video(
    video_path: str = Query(..., description="MP4视频文件路径"),
    frame_index: int = Query(0, description="帧索引（0开始）"),
    fps: float = Query(1.0, description="视频帧率")
):
    """
    从MP4视频中提取单帧并返回base64
    用于前端浏览历史时从视频中提取帧显示
    """
    global ffmpeg_extractor
    
    if ffmpeg_extractor is None:
        ffmpeg_extractor = FFmpegFrameExtractor()
    
    # 处理路径
    if not Path(video_path).is_absolute():
        video_path = str(Path(config.STORAGE_ROOT) / video_path)
    
    if not Path(video_path).exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
    
    # 提取帧
    image = ffmpeg_extractor.extract_frame_by_index(video_path, frame_index, fps)
    if image is None:
        raise HTTPException(status_code=500, detail="Failed to extract frame from video")
    
    # 转换为base64
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=80)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    
    return {
        "image_base64": img_b64,
        "width": image.width,
        "height": image.height
    }


@app.get("/api/video/frame_image")
def get_video_frame_image(
    video_path: str = Query(..., description="MP4视频文件路径"),
    frame_index: int = Query(0, description="帧索引（0开始）"),
    fps: float = Query(1.0, description="视频帧率")
):
    """
    从MP4视频中提取单帧并直接返回图片
    用于 <img> 标签直接引用
    """
    from fastapi.responses import Response
    
    global ffmpeg_extractor
    
    if ffmpeg_extractor is None:
        ffmpeg_extractor = FFmpegFrameExtractor()
    
    # 处理路径
    if not Path(video_path).is_absolute():
        video_path = str(Path(config.STORAGE_ROOT) / video_path)
    
    if not Path(video_path).exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
    
    # 提取帧
    image = ffmpeg_extractor.extract_frame_by_index(video_path, frame_index, fps)
    if image is None:
        raise HTTPException(status_code=500, detail="Failed to extract frame from video")
    
    # 转换为JPEG bytes
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=80)
    
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"}  # 缓存1小时
    )


@app.get("/api/video/buffer_stats")
def get_video_buffer_stats():
    """
    获取视频缓冲区统计信息
    """
    if temp_frame_buffer is None:
        return {"error": "Video buffer not initialized"}
    
    return temp_frame_buffer.get_stats()


def _check_clustering_health() -> tuple:
    """
    Check if activity clustering tables exist and timeline data is in sync with frames.
    Returns (is_healthy: bool, issue: str).
    Only relevant when ENABLE_CLUSTERING=True.
    """
    import sqlite3
    import os

    db_path = config.OCR_DB_PATH
    if not os.path.exists(db_path):
        return True, ""  # Fresh install, nothing to check

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Use activity DB for clustering health checks
        act_conn = sqlite_storage._get_activity_connection()
        act_cursor = act_conn.cursor()

        # Check if required tables exist in activity DB
        act_cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
            "('activity_clusters', 'activity_sessions', 'activity_assignments')"
        )
        existing = {row[0] for row in act_cursor.fetchall()}
        missing = {"activity_clusters", "activity_sessions", "activity_assignments"} - existing
        if missing:
            act_conn.close()
            conn.close()
            return False, f"Missing clustering tables in activity DB: {', '.join(sorted(missing))}"

        # Check if there are any frames to worry about
        cursor.execute("SELECT COUNT(*) FROM sub_frames")
        total_frames = cursor.fetchone()[0]
        if total_frames == 0:
            act_conn.close()
            conn.close()
            return True, ""

        # Check if timeline is empty while frames exist
        act_cursor.execute("SELECT MAX(end_time) FROM activity_sessions")
        latest_session = act_cursor.fetchone()[0]
        if not latest_session:
            act_conn.close()
            conn.close()
            return False, f"Timeline is empty but {total_frames} frames exist"

        # Check how many frames are uncovered after the last session
        act_cursor.execute(
            "SELECT COUNT(*) FROM activity_assignments WHERE timestamp > ? AND activity_cluster_id IS NULL",
            (latest_session,),
        )
        uncovered = act_cursor.fetchone()[0]
        act_conn.close()
        conn.close()

        if uncovered > 200:
            return False, (
                f"{uncovered} frames are not covered by the timeline "
                f"(latest session ended at {latest_session})"
            )

        return True, ""
    except Exception as e:
        return True, f"(health check skipped: {e})"


def _prompt_clustering_warning(issue: str) -> bool:
    """
    Print a warning about clustering health and ask the user whether to force-start.
    Returns True if the server should start, False if it should abort.
    """
    border = "=" * 68
    print(f"\n{border}")
    print("  WARNING: Activity clustering data issue detected")
    print(border)
    print(f"  Issue : {issue}")
    print()
    print("  To fix this, run the 3-phase timeline sync script:")
    print("    python scripts/backfill_activity_clusters.py --phases all")
    print()
    print("  Options:")
    print("    Press Enter or type 'y'  →  abort startup and run the script first")
    print("    Type 'n' + Enter         →  force-start anyway (timeline may be stale)")
    print(border)
    try:
        answer = input("  Your choice [y/n, default: y]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        # Non-interactive environment — default to force-start to avoid blocking the server
        print("\n  (Non-interactive mode detected, force-starting...)")
        return True
    if answer == "n":
        print("  Force-starting without timeline sync...")
        return True
    print("  Startup aborted. Please run the sync script and restart.\n")
    return False


if __name__ == "__main__":
    import uvicorn

    if config.ENABLE_CLUSTERING:
        is_healthy, issue = _check_clustering_health()
        if not is_healthy:
            should_start = _prompt_clustering_warning(issue)
            if not should_start:
                import sys
                sys.exit(0)

    uvicorn.run(
        "gui_backend_server:app",
        host="0.0.0.0",
        port=18080,
        reload=False,
        access_log=False,
    )
