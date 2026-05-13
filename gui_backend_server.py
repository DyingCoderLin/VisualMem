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
from typing import List, Dict, Optional, Any, Tuple
import base64
import io
import os
import re
import threading
import time as time_module
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import uuid

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image as PILImage
from pathlib import Path

from config import config
from utils.logger import setup_logger
from utils.eval_stats import emit_jsonl, get_eval_logger
from utils.eval_stability import start_stability_monitor
from utils.app_name_manager import app_name_manager
from core.api.router import router as data_platform_router, init_data_service
from core.api.daily_report_routes import router as daily_report_router
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
from core.capture.focused_window import get_focused_window, get_fullscreen_window_for_monitor
from core.worker import EnrichmentJob, FrameEnrichmentWorker
from utils.model_utils import ensure_model_downloaded


logger = setup_logger("gui_backend_server")

_EVAL_LATENCY_ENABLED = config.EVAL_LATENCY_STATS
_EVAL_STABILITY_ENABLED = config.EVAL_STABILITY_MONITOR
_RUNTIME_DIAGNOSTIC_LOGS_ENABLED = config.ENABLE_RUNTIME_DIAGNOSTIC_LOGS
_EVAL_LATENCY_LOGGER = (
    get_eval_logger("latency_frames", "logs/eval_latency_frames.jsonl")
    if _EVAL_LATENCY_ENABLED
    else None
)
_EVAL_STABILITY_STOP_EVENT: Optional[threading.Event] = None

app = FastAPI(title="VisualMem Backend Server")

# 添加 CORS 中间件，允许 Electron 前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Electron 应用，允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)

# Mount Data Platform API router (timeline, OCR, focus, reports)
app.include_router(data_platform_router)
app.include_router(daily_report_router)


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


def _emit_latency_event(event: Dict[str, Any]) -> None:
    if not _EVAL_LATENCY_ENABLED or _EVAL_LATENCY_LOGGER is None:
        return
    emit_jsonl(_EVAL_LATENCY_LOGGER, event)


def _runtime_diag(message: str) -> None:
    if _RUNTIME_DIAGNOSTIC_LOGS_ENABLED:
        logger.info(message)


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
_models_loaded = False  # Whether heavy models (encoder, reranker, OCR) have been loaded
_models_loading = False  # Whether models are currently being loaded
_models_lock = threading.Lock()

# ============ 视频存储相关组件 ============
temp_frame_buffer: Optional[TempFrameBuffer] = None
ffmpeg_compressor: Optional[FFmpegFrameCompressor] = None
ffmpeg_extractor: Optional[FFmpegFrameExtractor] = None

# 帧差检测（窗口级别去重）
from core.preprocess.frame_diff import FrameDiffDetector, is_solid_color_image
window_diff_detector: Optional[FrameDiffDetector] = None

# Cache last OCR text per fullscreen app (used when dedup skips the frame)
_fullscreen_ocr_cache: Dict[str, tuple] = {}  # app_name -> (ocr_text, confidence)

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

def _normalize_rect(bounds: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Normalize screen/window bounds dictionaries from Electron, Quartz, or xcap."""
    if not isinstance(bounds, dict):
        return None
    try:
        x = bounds.get("x", bounds.get("X"))
        y = bounds.get("y", bounds.get("Y"))
        width = bounds.get("width", bounds.get("Width"))
        height = bounds.get("height", bounds.get("Height"))
        if x is None or y is None or width is None or height is None:
            return None
        width_f = float(width)
        height_f = float(height)
        if width_f <= 0 or height_f <= 0:
            return None
        return {
            "x": float(x),
            "y": float(y),
            "width": width_f,
            "height": height_f,
        }
    except Exception:
        return None


def _extract_monitor_bounds(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Extract monitor bounds sent by the Electron recorder."""
    if not isinstance(metadata, dict):
        return None
    for key in ("monitor_bounds", "display_bounds", "bounds", "monitor_physical_bounds", "physical_bounds"):
        rect = _normalize_rect(metadata.get(key))
        if rect:
            return rect
    return _normalize_rect(metadata)


def _get_screencap_monitor_bounds(monitor_id: int) -> Optional[Dict[str, float]]:
    """Best-effort local monitor bounds fallback for older GUI clients."""
    if not (USE_SCREENCAP_RS and screencap_rs_module is not None):
        return None
    try:
        monitors = screencap_rs_module.get_monitors()
        if not monitors:
            return None
        monitor = monitors[monitor_id] if 0 <= monitor_id < len(monitors) else monitors[0]
        return _normalize_rect(
            {
                "x": getattr(monitor, "x", 0),
                "y": getattr(monitor, "y", 0),
                "width": getattr(monitor, "width", 0),
                "height": getattr(monitor, "height", 0),
            }
        )
    except Exception as e:
        logger.debug(f"Failed to read screencap monitor bounds: {e}")
        return None


def _intersection_area(a: Dict[str, float], b: Dict[str, float]) -> float:
    left = max(a["x"], b["x"])
    top = max(a["y"], b["y"])
    right = min(a["x"] + a["width"], b["x"] + b["width"])
    bottom = min(a["y"] + a["height"], b["y"] + b["height"])
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def _window_overlaps_monitor(
    window_data: Dict[str, Any],
    monitor_bounds: Optional[Dict[str, float]],
    min_window_overlap: float = 0.20,
) -> bool:
    """Return True when a captured window belongs to the target monitor."""
    if not monitor_bounds:
        return True
    window_rect = _normalize_rect(window_data)
    if not window_rect:
        # Older clients do not send window bounds. Keep those windows rather
        # than silently dropping potentially valid sub_frames.
        return True
    window_area = window_rect["width"] * window_rect["height"]
    if window_area <= 0:
        return False
    return (_intersection_area(window_rect, monitor_bounds) / window_area) >= min_window_overlap


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
        _runtime_diag(
            f"BatchWriteBuffer: added frame {frame_data.get('frame_id', '?')}, "
            f"buffer={buf_len}/{self.batch_size}"
        )
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

        n = len(frames_to_write)
        t_total0 = time_module.time()
        lance_ms = 0.0
        sqlite_ms_total = 0.0
        sqlite_failed = 0
        sqlite_ok = 0
        # Snapshot which frame_ids are in this flush so a mid-flush hang is
        # diagnosable from the log (matched against enrich heartbeat / stack dump).
        ids_preview = ",".join(f.get("frame_id", "?") for f in frames_to_write[:5])
        if n > 5:
            ids_preview += f",...(+{n - 5})"
        _runtime_diag(
            f"BatchWriteBuffer: flushing {n} frames to LanceDB+SQLite... ids=[{ids_preview}]"
        )

        try:
            if vector_storage is not None:
                t0 = time_module.time()
                try:
                    success = vector_storage.store_frames_batch(frames_to_write)
                except Exception as e:
                    logger.error(f"BatchWriteBuffer: LanceDB threw: {e}", exc_info=True)
                    success = False
                lance_ms = (time_module.time() - t0) * 1000.0
                if success:
                    _runtime_diag(
                        f"BatchWriteBuffer: ✓ LanceDB wrote {n} frames in {lance_ms:.0f}ms"
                    )
                else:
                    logger.error(
                        f"BatchWriteBuffer: ✗ LanceDB batch write failed in {lance_ms:.0f}ms"
                    )

            if sqlite_storage is not None:
                for frame_data in frames_to_write:
                    fid = frame_data.get("frame_id", "?")
                    t0 = time_module.time()
                    try:
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
                        sqlite_ok += 1
                    except Exception as e:
                        sqlite_failed += 1
                        logger.error(
                            f"BatchWriteBuffer: SQLite write failed for {fid}: {e}",
                            exc_info=True,
                        )
                    finally:
                        sqlite_ms_total += (time_module.time() - t0) * 1000.0
        except Exception as e:
            logger.error(f"BatchWriteBuffer: flush aborted: {e}", exc_info=True)

        total_ms = (time_module.time() - t_total0) * 1000.0
        avg_sqlite = (sqlite_ms_total / n) if n else 0.0
        _runtime_diag(
            f"BatchWriteBuffer: flush done n={n} total={total_ms:.0f}ms "
            f"lance={lance_ms:.0f}ms sqlite_total={sqlite_ms_total:.0f}ms "
            f"sqlite_avg={avg_sqlite:.0f}ms ok={sqlite_ok} failed={sqlite_failed}"
        )
        _emit_latency_event(
            {
                "event_type": "batch_flush",
                "n_frames": n,
                "total_ms": total_ms,
                "lance_ms": lance_ms,
                "sqlite_total_ms": sqlite_ms_total,
                "sqlite_ok": sqlite_ok,
                "sqlite_failed": sqlite_failed,
            }
        )

    def _periodic_flush(self):
        """定期检查并刷新缓冲区（后台线程）"""
        last_idle_log = time_module.time()
        IDLE_LOG_INTERVAL = 60.0  # heartbeat even when buffer empty
        while not self.stop_event.is_set():
            time_module.sleep(1)
            with self.buffer_lock:
                elapsed = time_module.time() - self.last_flush_time
                buf_len = len(self.buffer)
                should_flush = elapsed >= self.flush_interval and buf_len > 0

            if should_flush:
                _runtime_diag(
                    f"BatchWriteBuffer: periodic flush triggered ({elapsed:.0f}s elapsed, {buf_len} frames)"
                )
                self._flush_buffer()

            now = time_module.time()
            if _RUNTIME_DIAGNOSTIC_LOGS_ENABLED and now - last_idle_log >= IDLE_LOG_INTERVAL:
                # Cheap periodic visibility: if this stops appearing in the
                # log while frames are still being captured, the flush thread
                # itself has died (which previously went completely unnoticed).
                logger.info(
                    f"BatchWriteBuffer: heartbeat buffer={buf_len} "
                    f"since_last_flush={elapsed:.0f}s "
                    f"config(batch={self.batch_size}, interval={self.flush_interval}s)"
                )
                last_idle_log = now
    
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

# Long-lived enrichment worker: keeps heavy per-frame work
# (embedding, window OCR, cluster_assign, batch_write_buffer.add_frame)
# off the /api/store_frame HTTP request path. See core/worker/.
frame_enrichment_worker: Optional[FrameEnrichmentWorker] = None


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
            with sqlite_storage._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT image_path FROM frames WHERE frame_id = ?",
                    (sf["sub_frame_id"],),
                )
                row = cursor.fetchone()
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
        with sqlite_storage._activity_connection() as conn:
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
    
    t0 = time_module.time()
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
                                with sqlite_storage._connection() as conn:
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
    finally:
        _emit_latency_event(
            {
                "event_type": "video_compress_batch",
                "batch_type": batch_type,
                "identifier": identifier,
                "frames_in_batch": len(frames),
                "compress_ms": (time_module.time() - t0) * 1000.0,
            }
        )


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
        from collections import defaultdict
        from core.storage.temp_frame_buffer import FrameInfo
        import json

        # ── Phase 1: 恢复全屏截图 ──
        # 全屏帧压缩时 store_frame_with_video_ref 会同步更新 _fullscreen 全屏应用子帧的
        # image_path，所以必须先完成全屏压缩，再查窗口子帧，否则 _fullscreen 子帧会
        # 被窗口查询捞出来，而它们引用的 PNG 已在全屏 cleanup 中删除。
        fs_groups = defaultdict(list)

        with sqlite_storage._connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT frame_id, timestamp, image_path, device_name, metadata FROM frames WHERE image_path LIKE '%temp_frames%' AND frame_id LIKE 'frame_%' ORDER BY timestamp ASC")
            fs_records = cursor.fetchall()

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
                    cursor.execute("UPDATE frames SET image_path = '' WHERE frame_id = ?", (row["frame_id"],))

            conn.commit()

        for monitor_id, frames in fs_groups.items():
            logger.info(f"Recovering {len(frames)} leftover full_screen temp_frames for monitor_{monitor_id}...")
            _compress_video_batch("full_screen", f"monitor_{monitor_id}", frames)

        # ── Phase 2: 恢复窗口截图 ──
        # 在全屏压缩完成后再查询，此时 _fullscreen 子帧的 image_path 已被同步更新为
        # video_chunk 引用，不再匹配 '%temp_frames%'，自然被排除。
        win_groups = defaultdict(list)

        with sqlite_storage._connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT s.sub_frame_id, s.timestamp, f.image_path, s.app_name, s.window_name
                FROM sub_frames s
                JOIN frames f ON s.sub_frame_id = f.frame_id
                WHERE f.image_path LIKE '%temp_frames%'
                ORDER BY s.timestamp ASC
            """)
            win_records = cursor.fetchall()

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

            conn.commit()

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

    _init_region_ocr_engine()


_DIFF_STATE_TABLE = "window_diff_state"


_THUMBNAIL_SIZE = (128, 128)


def _ensure_diff_state_table():
    """Create or migrate the persistent diff state table."""
    global sqlite_storage
    if not sqlite_storage:
        return
    try:
        with sqlite_storage._connection() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {_DIFF_STATE_TABLE} (
                    window_key  TEXT PRIMARY KEY,
                    app_name    TEXT NOT NULL,
                    window_name TEXT NOT NULL,
                    image_hash  INTEGER NOT NULL,
                    thumbnail   BLOB,
                    updated_at  TEXT NOT NULL
                )
            """)
            # Migrate: add thumbnail column if missing (old schema)
            cursor = conn.execute(f"PRAGMA table_info({_DIFF_STATE_TABLE})")
            columns = {row[1] for row in cursor.fetchall()}
            if "thumbnail" not in columns:
                conn.execute(f"ALTER TABLE {_DIFF_STATE_TABLE} ADD COLUMN thumbnail BLOB")
            conn.commit()
    except Exception as e:
        logger.warning(f"Failed to create {_DIFF_STATE_TABLE} table: {e}")


_diff_state_flush_thread: Optional[threading.Thread] = None


def _start_periodic_diff_state_flush(interval: float = 300.0):
    """Start a daemon thread that flushes diff state every `interval` seconds."""
    global _diff_state_flush_thread
    import time

    def _loop():
        while True:
            time.sleep(interval)
            _flush_diff_state()

    _diff_state_flush_thread = threading.Thread(target=_loop, daemon=True)
    _diff_state_flush_thread.start()
    logger.info(f"Periodic diff state flush started (every {interval:.0f}s).")


def _flush_diff_state():
    """
    Batch-persist all window_diff_detector states (hash + thumbnail) to DB.
    Called on shutdown + periodically (crash safety).
    """
    global window_diff_detector, sqlite_storage
    if not window_diff_detector or not sqlite_storage:
        return
    states = window_diff_detector.window_states
    if not states:
        return
    try:
        rows = []
        for key, s in states.items():
            if s.previous_image is None:
                continue
            # Save thumbnail as lossless PNG to avoid compression artifacts
            # affecting cross-session diff comparison
            thumb = s.previous_image.resize(_THUMBNAIL_SIZE, PILImage.Resampling.LANCZOS)
            buf = io.BytesIO()
            thumb.save(buf, format="PNG")
            rows.append((
                key, s.app_name, s.window_name, s.previous_hash,
                buf.getvalue(),
            ))
        with sqlite_storage._connection() as conn:
            conn.execute(f"DELETE FROM {_DIFF_STATE_TABLE}")
            conn.executemany(f"""
                INSERT INTO {_DIFF_STATE_TABLE}
                    (window_key, app_name, window_name, image_hash, thumbnail, updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            """, rows)
            conn.commit()
        logger.info(f"Flushed {len(rows)} window diff states to DB.")
    except Exception as e:
        logger.warning(f"Failed to flush diff state: {e}")


def _warm_up_frame_diff_detector():
    """
    Restore window_diff_detector state from persisted thumbnails.
    Loads small JPEG thumbnails (not full images), fast startup.
    """
    global window_diff_detector, sqlite_storage
    if not window_diff_detector or not sqlite_storage:
        return

    from core.capture.window_capturer import calculate_image_hash

    _ensure_diff_state_table()

    try:
        with sqlite_storage._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT app_name, window_name, image_hash, thumbnail
                FROM {_DIFF_STATE_TABLE}
            """)
            rows = cursor.fetchall()

        if not rows:
            logger.info("No persisted diff state found, skipping warm-up.")
            return

        seeded = 0
        seeded_keys = []
        for row in rows:
            try:
                blob = row["thumbnail"]
                if not blob:
                    continue
                thumb = PILImage.open(io.BytesIO(blob)).convert("RGB")
                img_hash = calculate_image_hash(thumb)
                window_diff_detector.seed_window_state(
                    row["app_name"], row["window_name"], thumb, img_hash
                )
                seeded += 1
                seeded_keys.append(f"{row['app_name']}::{row['window_name']}")
            except Exception:
                pass

        logger.info(
            f"Warm-up: seeded {seeded}/{len(rows)} window diff states from DB.\n"
            + "\n".join(f"  {k}" for k in seeded_keys)
        )
    except Exception as e:
        logger.warning(f"Frame diff warm-up failed (non-fatal): {e}")


def _init_region_ocr_engine() -> None:
    """Initialize region OCR, but degrade gracefully when UIED/OpenCV is unavailable."""
    global ocr_engine, region_ocr_engine

    if region_ocr_engine is not None or ocr_engine is None:
        return

    detector = None
    if config.ENABLE_UIED:
        try:
            from core.ocr.region_detector import UIEDRegionDetector

            detector = UIEDRegionDetector()
        except Exception as e:
            logger.warning(
                "UIED region detector unavailable, falling back to whole-image OCR only: "
                f"{e}"
            )

    try:
        from core.ocr.region_ocr_engine import RegionOCREngine

        region_ocr_engine = RegionOCREngine(
            ocr_engine=ocr_engine,
            region_detector=detector,
        )
        if detector is not None:
            logger.info("RegionOCREngine initialized (UIED + per-region OCR).")
        else:
            logger.info("RegionOCREngine initialized (whole-image OCR only).")
    except Exception as e:
        logger.warning(
            "Failed to initialize RegionOCREngine; OCR will continue without region segmentation: "
            f"{e}"
        )
        region_ocr_engine = None


def _init_models():
    """
    Load heavy ML models: encoder, reranker, OCR engine.
    Called eagerly at startup (MODEL_LAZY_LOAD=false) or on-demand via /api/load_models.
    """
    global encoder, vector_storage, reranker, ocr_engine, region_ocr_engine
    global _models_loaded, _models_loading

    if _models_loaded:
        return
    with _models_lock:
        if _models_loaded:
            return

        _models_loading = True
        logger.info(
            "Starting model initialization on thread "
            f"{threading.current_thread().name} (lazy={config.MODEL_LAZY_LOAD})"
        )

        try:
            # 0. Pre-flight check: Ensure models are downloaded
            ensure_model_downloaded(config.EMBEDDING_MODEL, "Image Encoder")
            if config.ENABLE_RERANK:
                ensure_model_downloaded(config.RERANK_MODEL, "Reranker Model")

            # 1. Load encoder (embedding model)
            logger.info(f"[model 1/4] Loading encoder {config.EMBEDDING_MODEL}...")
            encoder = create_encoder(model_name=config.EMBEDDING_MODEL)

            # 2. Initialize LanceDB storage (needs encoder.embedding_dim)
            logger.info("[model 2/4] Initializing LanceDB storage...")
            vector_storage = LanceDBStorage(
                db_path=config.LANCEDB_PATH,
                embedding_dim=encoder.embedding_dim,
            )

            # 2b. Optimize LanceDB
            try:
                if vector_storage.table is not None:
                    logger.info("Optimizing LanceDB (cleanup old versions)...")
                    vector_storage.cleanup_old_versions(older_than_hours=0.1, delete_unverified=True)
                    logger.info("LanceDB optimization done.")
                else:
                    logger.info("LanceDB table does not exist, skipping optimization.")
            except Exception as e:
                logger.warning(f"LanceDB optimization failed: {e}")

            # 3. Load Reranker model
            if config.ENABLE_RERANK:
                logger.info("[model 3/4] Loading Reranker model...")
                reranker = Reranker()
            else:
                logger.info("[model 3/4] Reranker disabled (ENABLE_RERANK=False)")

            # 4. Initialize OCR engine (if enabled)
            if config.ENABLE_OCR:
                logger.info(f"[model 4/4] Initializing OCR engine ({config.OCR_ENGINE_TYPE})...")
                try:
                    ocr_engine = create_ocr_engine(config.OCR_ENGINE_TYPE, lang="chi_sim+eng")
                except Exception as e:
                    logger.warning(f"Failed to init OCR engine ({config.OCR_ENGINE_TYPE}), fallback to dummy: {e}")
                    ocr_engine = create_ocr_engine("dummy")
                _init_region_ocr_engine()
            else:
                logger.info("[model 4/4] OCR engine disabled (ENABLE_OCR=False)")

            _models_loaded = True
            logger.info("All ML models loaded successfully!")
        finally:
            _models_loading = False


def _init_infra():
    """
    Initialize lightweight infrastructure components (no ML models).
    Always called at startup regardless of MODEL_LAZY_LOAD.
    """
    global sqlite_storage, vlm, batch_write_buffer, frame_enrichment_worker
    global temp_frame_buffer, ffmpeg_compressor, ffmpeg_extractor, window_diff_detector
    global _EVAL_STABILITY_STOP_EVENT

    logger.info("=" * 60)
    logger.info("Initializing infrastructure components...")
    logger.info("=" * 60)

    # SQLite storage
    logger.info("[infra 1/7] Initializing SQLite storage...")
    sqlite_storage = SQLiteStorage(db_path=config.OCR_DB_PATH)

    # VLM client (lightweight API wrapper, no model loading)
    logger.info("[infra 2/7] Initializing VLM API client...")
    vlm = ApiVLM()

    # Batch write buffer
    # NOTE: After the /api/store_frame fast-path refactor, ``today_count`` and
    # ``/api/recent_frames`` both read from SQLite. Stale reads (counter-drift
    # and empty realtime view) appear whenever this buffer lags behind the
    # fast path by more than a few frames, so keep the parameters small.
    logger.info(
        f"[infra 3/7] Initializing batch write buffer "
        f"(size={config.BATCH_WRITE_BUFFER_SIZE}, "
        f"flush_interval={config.BATCH_WRITE_FLUSH_INTERVAL_SECONDS:.1f}s)..."
    )
    batch_write_buffer = BatchWriteBuffer(
        batch_size=config.BATCH_WRITE_BUFFER_SIZE,
        flush_interval_seconds=config.BATCH_WRITE_FLUSH_INTERVAL_SECONDS,
    )
    batch_write_buffer.start()

    # Temp frame buffer for video compression
    logger.info("[infra 4/7] Initializing temp frame buffer for video compression...")
    temp_frame_buffer = TempFrameBuffer(
        storage_root=config.STORAGE_ROOT,
        batch_size=VIDEO_BATCH_SIZE,
        fps=VIDEO_FPS,
        on_batch_ready=_on_video_batch_ready
    )

    # FFmpeg utilities
    logger.info("[infra 5/7] Initializing FFmpeg utilities...")
    ffmpeg_compressor = FFmpegFrameCompressor(fps=VIDEO_FPS)
    ffmpeg_extractor = FFmpegFrameExtractor()

    # Window-level frame diff detector for dedup
    logger.info("[infra 6/7] Initializing window frame diff detector...")
    window_diff_detector = FrameDiffDetector(
        screen_threshold=config.SIMPLE_FILTER_DIFF_THRESHOLD,
        window_threshold=config.SIMPLE_FILTER_DIFF_THRESHOLD,
    )

    # Recover leftover temp frames from previous abnormal exit
    logger.info("[infra 7/7] Recovering leftover temp frames...")
    _recover_temp_frames()

    # Warm up frame diff detector from DB (avoid cold-start re-capture)
    logger.info("Warming up frame diff detector from previous session...")
    _warm_up_frame_diff_detector()
    _start_periodic_diff_state_flush(interval=300.0)  # every 5 min, crash safety

    # Long-lived frame enrichment worker (embedding / OCR / sub_frame writes).
    # Started once here; drained on /api/recording/stop and on shutdown.
    logger.info(
        f"[infra 8/8] Starting frame enrichment worker "
        f"({config.FRAME_ENRICHMENT_WORKERS} threads)..."
    )
    frame_enrichment_worker = FrameEnrichmentWorker(
        num_workers=config.FRAME_ENRICHMENT_WORKERS,
        name="frame-enrich",
        enable_diagnostics=config.ENABLE_ENRICHMENT_STACK_DUMPS,
        heartbeat_interval_s=config.ENRICH_HEARTBEAT_SECONDS,
        stuck_job_threshold_s=config.ENRICH_STUCK_SECONDS,
    )
    frame_enrichment_worker.start()

    if _EVAL_STABILITY_ENABLED:
        interval_s = config.EVAL_STABILITY_INTERVAL_S
        _EVAL_STABILITY_STOP_EVENT = start_stability_monitor(
            interval_s=interval_s,
            log_path="logs/eval_stability.jsonl",
            db_paths=[
                config.OCR_DB_PATH,
                config.ACTIVITY_DB_PATH,
            ],
            dir_paths=[
                config.LANCEDB_PATH,
                str(Path(config.STORAGE_ROOT) / "temp_frames"),
                str(Path(config.STORAGE_ROOT) / "visualmem_video"),
            ],
        )
        logger.info(
            f"Eval stability monitor started (interval={interval_s:.1f}s, log=logs/eval_stability.jsonl)"
        )


def _init_all_components():
    """
    Initialize all components. Respects MODEL_LAZY_LOAD config:
    - When true: only loads infra at startup, models loaded on-demand
    - When false: loads everything eagerly at startup (original behavior)
    """
    _init_infra()

    if config.MODEL_LAZY_LOAD:
        logger.info("MODEL_LAZY_LOAD=true: Deferring ML model loading until recording starts.")
    else:
        logger.info("MODEL_LAZY_LOAD=false: Loading ML models eagerly at startup...")
        _init_models()

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
    x: Optional[int] = None
    y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None


class StoreFrameRequest(BaseModel):
    frame_id: str
    timestamp: str  # ISO string
    image_base64: str  # 全屏截图的base64
    monitor_id: int = 0  # 显示器ID
    metadata: Optional[Dict] = None
    windows: Optional[List[WindowInfo]] = None  # 窗口截图列表（可选）
    client_capture_ms: Optional[float] = None


class FrontendConfigResponse(BaseModel):
    theme: str


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


class RewindSubFrameResult(BaseModel):
    sub_frame_id: str = ""
    timestamp: str = ""
    app_name: str = ""
    window_name: str = ""
    image_path: Optional[str] = None


class RewindSegment(BaseModel):
    segment_id: Optional[str] = None
    frame_id: Optional[str] = None
    timestamp: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    title: Optional[str] = None
    app_name: Optional[str] = None
    window_name: Optional[str] = None
    activity_label: Optional[str] = None
    image_path: Optional[str] = None
    ocr_text: Optional[str] = ""
    sub_frames: List[RewindSubFrameResult] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RewindEvidenceRef(BaseModel):
    frame_id: Optional[str] = None
    sub_frame_id: Optional[str] = None
    timestamp: Optional[str] = None
    image_path: Optional[str] = None
    app_name: Optional[str] = None
    window_name: Optional[str] = None
    activity_label: Optional[str] = None
    ocr_snippet: Optional[str] = ""


class RewindSearchRequest(BaseModel):
    query: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    top_k: int = 12


class RewindSearchResponse(BaseModel):
    query: str
    segments: List[RewindSegment]


class RewindTimelineFramesRequest(BaseModel):
    start_time: str
    end_time: str
    offset: int = 0
    limit: int = 36


class RewindTimelineFrame(BaseModel):
    frame_id: str
    timestamp: str
    image_path: Optional[str] = None
    ocr_text: Optional[str] = ""
    sub_frames: List[RewindSubFrameResult] = Field(default_factory=list)


class RewindTimelineFramesResponse(BaseModel):
    start_time: str
    end_time: str
    offset: int
    limit: int
    total_count: int
    frames: List[RewindTimelineFrame]


class BuildRewindContextRequest(BaseModel):
    source_query: str
    selected_segments: List[RewindSegment] = Field(default_factory=list)
    evidence_refs: List[RewindEvidenceRef] = Field(default_factory=list)
    title: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    top_k: int = 12


class TaskMemoryResponse(BaseModel):
    task_memory_id: str
    title: str
    markdown: str
    source_query: str
    selected_segments: List[RewindSegment]
    evidence_refs: List[RewindEvidenceRef]
    created_at: str
    updated_at: str


class TaskMemoryListItem(BaseModel):
    task_memory_id: str
    title: str
    source_query: str
    created_at: str
    updated_at: str
    selected_segment_count: int = 0


class TaskMemoryListResponse(BaseModel):
    memories: List[TaskMemoryListItem]


class TaskMemoryPatchRequest(BaseModel):
    title: Optional[str] = None
    markdown: Optional[str] = None


class TaskMemoryAskRequest(BaseModel):
    question: str
    markdown: Optional[str] = None


class TaskMemoryAskResponse(BaseModel):
    task_memory_id: str
    answer: str
    evidence_refs: List[RewindEvidenceRef]


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
        # Initialize Data Platform API service (timeline, OCR, focus, analytics)
        init_data_service(config.OCR_DB_PATH, config.ACTIVITY_DB_PATH)
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
    global _EVAL_STABILITY_STOP_EVENT
    if _EVAL_STABILITY_STOP_EVENT is not None:
        _EVAL_STABILITY_STOP_EVENT.set()
        _EVAL_STABILITY_STOP_EVENT = None
    
    # 0. Persist frame diff state (before any cleanup)
    try:
        logger.info("Flushing window diff state...")
        _flush_diff_state()
    except Exception as e:
        logger.error(f"Error flushing diff state: {e}")

    # 0.5 Drain enrichment worker so any in-flight frames finish writing
    # before we flush video/batch buffers.
    try:
        if frame_enrichment_worker is not None:
            stop_timeout = getattr(config, "STOP_FLUSH_MAX_SECONDS", 30.0)
            logger.info(
                f"Draining frame enrichment queue on shutdown (max {stop_timeout}s)..."
            )
            drained = frame_enrichment_worker.drain(timeout=stop_timeout)
            stats = frame_enrichment_worker.stats()
            if drained:
                logger.info(
                    f"Frame enrichment drained on shutdown "
                    f"(completed={stats['completed']} failed={stats['failed']})"
                )
            else:
                logger.warning(
                    f"Frame enrichment drain TIMEOUT on shutdown — "
                    f"queue={stats['queue_depth']} inflight={stats['inflight']}; "
                    f"proceeding to flush anyway"
                )
            frame_enrichment_worker.shutdown(timeout=5.0)
    except Exception as e:
        logger.error(f"Error draining frame enrichment worker: {e}")

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


@app.get("/api/frontend_config", response_model=FrontendConfigResponse)
def get_frontend_config():
    return {"theme": config.FRONTEND_THEME}


@app.post("/api/load_models")
async def load_models_api():
    """
    On-demand loading of heavy ML models (encoder, reranker, OCR).
    Called by frontend before starting recording when MODEL_LAZY_LOAD=true.
    Returns immediately if models are already loaded.
    """
    try:
        _init_models()
        return {"status": "ok", "message": "Models loaded successfully"}
    except Exception as e:
        logger.error(f"Failed to load models on demand: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load models: {e}")


@app.get("/api/models_status")
def get_models_status():
    """Check whether ML models are loaded and ready."""
    return {
        "loaded": _models_loaded,
        "loading": _models_loading,
    }


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

    # Enrichment worker backlog (for remote GUI capture backpressure)
    if frame_enrichment_worker is not None:
        try:
            es = frame_enrichment_worker.stats()
            qd = int(es.get("queue_depth", 0) or 0)
            inf = int(es.get("inflight", 0) or 0)
            stats["enrichment_queue_depth"] = qd
            stats["enrichment_inflight"] = inf
            stats["enrichment_pipeline_depth"] = qd + inf
        except Exception as e:
            logger.debug(f"Enrichment stats for /api/stats: {e}")
            stats["enrichment_queue_depth"] = 0
            stats["enrichment_inflight"] = 0
            stats["enrichment_pipeline_depth"] = 0
    else:
        stats["enrichment_queue_depth"] = 0
        stats["enrichment_inflight"] = 0
        stats["enrichment_pipeline_depth"] = 0
    stats["enrichment_backpressure_high"] = config.ENRICHMENT_PIPELINE_BACKPRESS_HIGH
    stats["enrichment_backpressure_low"] = config.ENRICHMENT_PIPELINE_BACKPRESS_LOW

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
    Store a frame sent from remote GUI (video-storage mode).

    Request path is split in two:

    * **Sync fast path** (this function body, target < 500ms): decode + solid-
      color check + focused-window query + full-screen PNG persist. Returns a
      ``frame_summary`` so the frontend timeline can update immediately.
    * **Enrichment job** (``_enrich_frame_job``, runs on
      :data:`frame_enrichment_worker`): full-screen embedding, per-window
      embedding + OCR + sub_frame persistence, fullscreen-app sub_frame
      synthesis, cluster assignment, ``batch_write_buffer.add_frame`` for the
      main frame.

    Sub-frames are therefore initially returned empty and populated in SQLite /
    LanceDB asynchronously. TimelineView only consumes the main-frame
    image_path on the immediate ``recording-frame-stored`` event — sub-frames
    appear on the next page refresh or drill-down fetch.
    """
    global encoder, vector_storage, sqlite_storage, batch_write_buffer
    global temp_frame_buffer, frame_enrichment_worker

    _t0 = time_module.time()
    logger.info(f"store_frame: START {req.frame_id}")

    assert encoder is not None
    assert vector_storage is not None
    assert sqlite_storage is not None
    assert batch_write_buffer is not None
    assert frame_enrichment_worker is not None

    try:
        ts = datetime.fromisoformat(req.timestamp)
    except Exception as e:
        logger.error(f"Invalid timestamp '{req.timestamp}': {e}")
        raise HTTPException(status_code=400, detail=f"Invalid timestamp: {e}")

    # Decode full screen image (needed for solid-color check AND for embedding
    # inside the worker — decode once and hand the PIL off).
    img_bytes = base64.b64decode(req.image_base64)
    image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")

    # Skip solid-color / black-screen frames entirely (e.g. monitor off, lid closed)
    if is_solid_color_image(image):
        logger.debug("store_frame: skipping solid-color full screen image (screen off?)")
        es0 = frame_enrichment_worker.stats()
        q0 = int(es0.get("queue_depth", 0) or 0)
        i0 = int(es0.get("inflight", 0) or 0)
        return {
            "status": "skipped",
            "reason": "solid_color_frame",
            "frame_id": None,
            "enrichment_queue_depth": q0,
            "enrichment_inflight": i0,
            "enrichment_pipeline_depth": q0 + i0,
            "enrichment_backpressure_high": config.ENRICHMENT_PIPELINE_BACKPRESS_HIGH,
            "enrichment_backpressure_low": config.ENRICHMENT_PIPELINE_BACKPRESS_LOW,
        }

    base_frame_id = ts.strftime("%Y%m%d_%H%M%S_") + f"{ts.microsecond:06d}"
    frame_id = f"frame_{base_frame_id}_{req.monitor_id}"

    focused_app, focused_win = get_focused_window()

    # Persist full-screen PNG to the temp frame buffer. This is a cheap disk
    # write (no ffmpeg yet); batch compression runs later inside the worker
    # when a 60-frame batch is ready.
    if temp_frame_buffer is not None:
        temp_image_path, fs_batch_ready = temp_frame_buffer.add_full_screen_frame(
            frame_id=frame_id,
            image=image,
            timestamp=ts,
            monitor_id=req.monitor_id,
            metadata=req.metadata,
        )
    else:
        # 回退到原有的JPEG存储方式
        date_dir = config.IMAGE_STORAGE_PATH
        date_path = Path(date_dir) / ts.strftime("%Y%m%d")
        date_path.mkdir(parents=True, exist_ok=True)
        image_filename = f"{base_frame_id}.jpg"
        temp_image_path = str((date_path / image_filename).resolve())
        image.save(temp_image_path, format="JPEG", quality=config.IMAGE_QUALITY)
        fs_batch_ready = False

    sync_ms = (time_module.time() - _t0) * 1000.0

    # Build fast-path response BEFORE enqueuing so the returned frame_summary
    # reflects what's on disk right now.
    today_count = 0
    try:
        today_str = ts.strftime("%Y-%m-%d")
        next_day_str = (ts + timedelta(days=1)).strftime("%Y-%m-%d")
        with sqlite_storage._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*) as count FROM frames
                WHERE timestamp >= ? AND timestamp < ?
                  AND image_path IS NOT NULL AND image_path != ''
                  AND frame_id LIKE 'frame_%'
                """,
                (today_str, next_day_str),
            )
            row = cursor.fetchone()
            today_count = row["count"] if row else 0
    except Exception:
        pass

    queue_depth_before = frame_enrichment_worker.queue_depth()

    def _processor():
        _enrich_frame_job(
            req=req,
            image=image,
            ts=ts,
            frame_id=frame_id,
            base_frame_id=base_frame_id,
            temp_image_path=temp_image_path,
            focused_app=focused_app,
            focused_win=focused_win,
            fs_batch_ready=fs_batch_ready,
            sync_ms=sync_ms,
            client_capture_ms=float(req.client_capture_ms or 0.0),
        )

    frame_enrichment_worker.submit(
        EnrichmentJob(frame_id=frame_id, processor=_processor)
    )

    es_after = frame_enrichment_worker.stats()
    q_after = int(es_after.get("queue_depth", 0) or 0)
    i_after = int(es_after.get("inflight", 0) or 0)

    logger.info(
        f"store_frame: {frame_id} fastpath_done sync_ms={sync_ms:.0f} "
        f"enrich_queue_depth={queue_depth_before + 1}"
    )

    return {
        "status": "ok",
        "frame_id": frame_id,
        "sub_frame_count": 0,  # populated asynchronously
        "today_count": today_count,
        "enrichment_queue_depth": q_after,
        "enrichment_inflight": i_after,
        "enrichment_pipeline_depth": q_after + i_after,
        "enrichment_backpressure_high": config.ENRICHMENT_PIPELINE_BACKPRESS_HIGH,
        "enrichment_backpressure_low": config.ENRICHMENT_PIPELINE_BACKPRESS_LOW,
        "frame_summary": {
            "frame_id": frame_id,
            "timestamp": ts.isoformat(),
            "image_path": temp_image_path,
            "ocr_text": "",
            "sub_frames": [],
        },
    }


def _enrich_frame_job(
    *,
    req: "StoreFrameRequest",
    image: "PILImage.Image",
    ts: datetime,
    frame_id: str,
    base_frame_id: str,
    temp_image_path: str,
    focused_app: Optional[str],
    focused_win: Optional[str],
    fs_batch_ready: bool,
    sync_ms: float = 0.0,
    client_capture_ms: float = 0.0,
) -> None:
    """Heavy per-frame work, run on :data:`frame_enrichment_worker`.

    This used to be inline in ``store_frame`` and blocked the HTTP request for
    10–30 s in ``all`` recording mode. Moving it here keeps the fast path
    sub-second while the workers handle embedding, OCR, sub_frame persistence
    and cluster assignment at their own pace.
    """
    global encoder, vector_storage, sqlite_storage, batch_write_buffer
    global temp_frame_buffer

    _t0 = time_module.time()
    _HANDLER_MAX_SECONDS = 25
    _runtime_diag(f"enrich_frame: {frame_id} step=start")
    timings: Dict[str, float] = {
        "client_capture_ms": float(client_capture_ms or 0.0),
        "sync_ms": float(sync_ms or 0.0),
        "compress_batches_ms": 0.0,
        "embedding_ms": 0.0,
        "win_diff_ms": 0.0,
        "win_embedding_total_ms": 0.0,
        "win_ocr_total_ms": 0.0,
        "uied_total_ms": 0.0,
        "syn_ocr_ms": 0.0,
        "main_store_ocr_ms": 0.0,
        "cluster_assign_total_ms": 0.0,
        "add_frame_ms": 0.0,
    }

    # If the full-screen batch became ready during the sync path, compress it
    # here instead of blocking the HTTP response.
    if fs_batch_ready:
        _tc0 = time_module.time()
        try:
            _check_and_compress_batches()
            _runtime_diag(
                f"enrich_frame: {frame_id} step=compress_batches_done "
                f"t={((time_module.time() - _tc0) * 1000):.0f}ms"
            )
            timings["compress_batches_ms"] += (time_module.time() - _tc0) * 1000.0
        except Exception as e:
            logger.warning(
                f"Full-screen batch compression failed for {frame_id} after "
                f"{((time_module.time() - _tc0) * 1000):.0f}ms: {e}",
                exc_info=True,
            )

    _te0 = time_module.time()
    _runtime_diag(f"enrich_frame: {frame_id} step=embedding")
    embedding = encoder.encode_image(image)
    _runtime_diag(
        f"enrich_frame: {frame_id} step=embedding_done "
        f"t={((time_module.time() - _te0) * 1000):.0f}ms"
    )
    timings["embedding_ms"] = (time_module.time() - _te0) * 1000.0

    # Full-screen frame OCR is deferred: we collect sub_frame OCR results first,
    # then combine them as the frame's ocr_text (labeled by app_name).
    # This avoids mixing text from unrelated windows in a single OCR pass.
    ocr_engine_name = getattr(ocr_engine, 'engine_name', 'auto') if ocr_engine else "pending"
    monitor_bounds = _extract_monitor_bounds(req.metadata) or _get_screencap_monitor_bounds(req.monitor_id)
    fullscreen_app, fullscreen_win = get_fullscreen_window_for_monitor(monitor_bounds)
    if fullscreen_app:
        fullscreen_win = fullscreen_win or fullscreen_app
        app_name_manager.add_apps([fullscreen_app])
        if fullscreen_win:
            app_name_manager.add_window_pairs([(fullscreen_app, fullscreen_win)])
        _runtime_diag(
            f"enrich_frame: {frame_id} monitor={req.monitor_id} "
            f"fullscreen_label={fullscreen_app}/{fullscreen_win}"
        )

    # Collector for sub_frame OCR results: [(app_name, ocr_text, confidence), ...]
    sub_frame_ocr_parts = []

    # ========== 3. 处理窗口截图 ==========
    
    sub_frame_ids = []
    sub_frame_summaries = []  # Collect sub_frame info for response (avoid extra HTTP round-trips)
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
                    "image": win_image,
                    "x": win.x,
                    "y": win.y,
                    "width": win.width,
                    "height": win.height,
                })
            except Exception as e:
                logger.warning(f"Failed to decode window image for {win.app_name}: {e}")
    
    # 如果前端没有提供窗口信息，且启用了后端窗口捕获，使用screencap_rs
    elif ENABLE_BACKEND_WINDOW_CAPTURE and USE_SCREENCAP_RS and screencap_rs_module is not None:
        try:
            _runtime_diag(f"store_frame: {frame_id} step=capture_windows_start")
            captured_windows = screencap_rs_module.capture_all_windows(
                include_minimized=False,
                filter_system=True
            )
            _runtime_diag(f"store_frame: {frame_id} step=capture_windows_done count={len(captured_windows)}")
            for cw in captured_windows:
                try:
                    # 将PNG bytes转换为PIL Image
                    png_bytes = cw.get_image_bytes()
                    win_image = PILImage.open(io.BytesIO(png_bytes)).convert("RGB")
                    windows_to_process.append({
                        "app_name": cw.info.app_name,
                        "window_name": cw.info.title,
                        "image": win_image,
                        "x": getattr(cw.info, "x", None),
                        "y": getattr(cw.info, "y", None),
                        "width": getattr(cw.info, "width", None),
                        "height": getattr(cw.info, "height", None),
                    })
                except Exception as e:
                    logger.debug(f"Failed to process captured window {cw.info.app_name}: {e}")
            logger.debug(f"Backend captured {len(windows_to_process)} windows")
        except Exception as e:
            logger.warning(f"Backend window capture failed: {e}")

    if monitor_bounds and windows_to_process:
        before_filter = len(windows_to_process)
        windows_to_process = [
            w for w in windows_to_process
            if _window_overlaps_monitor(w, monitor_bounds)
        ]
        if len(windows_to_process) != before_filter:
            logger.debug(
                f"Filtered backend windows for monitor {req.monitor_id}: "
                f"{before_filter} -> {len(windows_to_process)}"
            )
    
    # Tracking for final summary log (shared across window batch + fullscreen app subframe)
    _processed_apps: List[str] = []
    _processed_reasons: List[str] = []
    _skipped_dedup_apps: List[str] = []
    _skipped_solid = 0

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
                    _td0 = time_module.time()
                    win_hash = calculate_image_hash(win_image)
                    wf = WF(
                        app_name=app_name,
                        window_name=window_name,
                        image=win_image,
                        image_hash=win_hash,
                        timestamp=ts,
                    )
                    diff_result = window_diff_detector.check_window_diff(wf)
                    timings["win_diff_ms"] += (time_module.time() - _td0) * 1000.0
                    if not diff_result.should_store:
                        logger.debug(
                            f"Skipping duplicate window {app_name}/{window_name} "
                            f"(diff={diff_result.diff_score:.4f})"
                        )
                        _skipped_dedup_apps.append(app_name)
                        continue

                # Skip solid-color / black-screen window images
                if is_solid_color_image(win_image):
                    logger.debug(
                        f"Skipping solid-color window: {app_name}/{window_name}"
                    )
                    _skipped_solid += 1
                    continue

                _processed_apps.append(app_name)
                _processed_reasons.append(
                    f"{app_name}/{window_name} diff={diff_result.diff_score:.4f} reason={diff_result.reason}"
                    if window_diff_detector is not None else f"{app_name}/{window_name} no_detector"
                )

                # 生成 sub_frame_id
                safe_app = app_name.replace(" ", "_").replace("/", "_")[:20]
                sub_frame_id = f"subframe_{safe_app}_{base_frame_id}_{i}"
                
                # 对窗口帧做 embedding
                _runtime_diag(f"store_frame: {frame_id} win={app_name} step=win_embedding")
                _twe0 = time_module.time()
                win_embedding = encoder.encode_image(win_image)
                _runtime_diag(f"store_frame: {frame_id} win={app_name} step=win_embedding_done")
                timings["win_embedding_total_ms"] += (time_module.time() - _twe0) * 1000.0

                # 对窗口帧做 region OCR
                _runtime_diag(f"store_frame: {frame_id} win={app_name} step=win_ocr_start")
                _two0 = time_module.time()
                win_ocr_text = ""
                win_ocr_json = ""
                win_ocr_engine_name = "none"
                win_ocr_conf = 0.0
                win_ocr_regions_stored = False
                win_layout_text = ""
                win_regions = []
                _elapsed_check = time_module.time() - _t0
                if _elapsed_check > _HANDLER_MAX_SECONDS:
                    logger.warning(
                        f"store_frame: {frame_id} TIMEOUT GUARD — {_elapsed_check:.1f}s elapsed, "
                        f"skipping remaining window OCR"
                    )
                    break
                if region_ocr_engine is not None:
                    try:
                        win_regions = region_ocr_engine.recognize_regions(win_image)
                        _region_timing = getattr(region_ocr_engine, "last_timing", {}) or {}
                        timings["uied_total_ms"] += float(_region_timing.get("detector_ms", 0.0) or 0.0)
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
                            try:
                                from core.activity.vlm_labeler import build_region_layout_text

                                win_layout_regions = []
                                for r in win_regions:
                                    bbox = r.get("bbox")
                                    text = r.get("text", "")
                                    if not bbox or not text:
                                        continue
                                    win_layout_regions.append(
                                        {
                                            "bbox": bbox,
                                            "text": text,
                                            "image_width": win_w,
                                            "image_height": win_h,
                                        }
                                    )
                                win_layout_text = build_region_layout_text(win_layout_regions)
                            except Exception:
                                win_layout_text = ""
                    except Exception as e:
                        logger.warning(f"Region OCR failed for window {app_name}: {e}")

                _runtime_diag(f"store_frame: {frame_id} win={app_name} step=win_ocr_done len={len(win_ocr_text)}")
                timings["win_ocr_total_ms"] += (time_module.time() - _two0) * 1000.0

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
                    _tca0 = time_module.time()
                    try:
                        _runtime_diag(f"store_frame: {frame_id} win={app_name} step=cluster_assign")
                        activity_label = cluster_manager.assign_frame(
                            app_name=app_name,
                            frame_id=sub_frame_id,
                            embedding=win_embedding,
                            image=win_image,
                            ocr_text=win_ocr_text,
                            layout_text=win_layout_text,
                            timestamp=ts.isoformat(),
                            window_name=window_name,
                        )
                        _runtime_diag(
                            f"store_frame: {frame_id} win={app_name} step=cluster_done "
                            f"label={activity_label} t={((time_module.time() - _tca0) * 1000):.0f}ms"
                        )
                        timings["cluster_assign_total_ms"] += (time_module.time() - _tca0) * 1000.0
                        if activity_label:
                            logger.debug(f"Assigned {sub_frame_id} -> '{activity_label}'")
                        _log_non_committed_cluster_result(app_name, sub_frame_id)
                    except Exception as e:
                        logger.warning(
                            f"store_frame: {frame_id} win={app_name} step=cluster_failed "
                            f"t={((time_module.time() - _tca0) * 1000):.0f}ms err={e}",
                            exc_info=True,
                        )

                sub_frame_ids.append(sub_frame_id)
                sub_frame_summaries.append({
                    "sub_frame_id": sub_frame_id,
                    "timestamp": ts.isoformat(),
                    "app_name": app_name,
                    "window_name": window_name,
                    "image_path": win_temp_path,
                })

                # 如果窗口批次就绪，触发压缩
                if win_batch_ready:
                    _check_and_compress_batches()
                    
            except Exception as e:
                logger.warning(f"Failed to process window {win_data.get('app_name', 'unknown')}: {e}")
                continue

        # (batch summary moved to final done log)

    # ========== 4. 全屏应用检测：只使用当前 monitor 的几何覆盖窗口来创建子帧 ==========
    # macOS native fullscreen windows may be missing from screencap_rs window
    # captures. The global focused app can be on another display, so the label
    # must come from monitor-local Quartz geometry instead.
    if fullscreen_app and temp_frame_buffer is not None:
        captured_app_names = {w.get("app_name", "") for w in windows_to_process}
        if fullscreen_app not in captured_app_names:
            _runtime_diag(f"store_frame: {frame_id} step=fullscreen_subframe_start app={fullscreen_app}")
            _syn_should_store = True
            if window_diff_detector is not None:
                from core.capture.window_capturer import calculate_image_hash as _calc_hash
                from utils.data_models import WindowFrame as _WF
                _syn_hash = _calc_hash(image)
                _syn_wf = _WF(
                    app_name=fullscreen_app,
                    window_name=fullscreen_win or fullscreen_app,
                    image=image,
                    image_hash=_syn_hash,
                    timestamp=ts,
                )
                _syn_diff = window_diff_detector.check_window_diff(_syn_wf)
                if not _syn_diff.should_store:
                    _skipped_dedup_apps.append(f"{fullscreen_app}(fullscreen)")
                    _syn_should_store = False
                    # Reuse cached OCR so the frame-level combined OCR still
                    # contains the fullscreen app's text
                    _cached = _fullscreen_ocr_cache.get(fullscreen_app)
                    if _cached:
                        sub_frame_ocr_parts.append((fullscreen_app, _cached[0], _cached[1]))
                else:
                    _processed_apps.append(f"{fullscreen_app}(fullscreen)")
                    _processed_reasons.append(
                        f"{fullscreen_app}(fullscreen) diff={_syn_diff.diff_score:.4f} "
                        f"reason={_syn_diff.reason}"
                    )

            if _syn_should_store:
                try:
                    safe_focused = fullscreen_app.replace(" ", "_").replace("/", "_")[:20]
                    syn_sub_id = f"subframe_{safe_focused}_{base_frame_id}_fullscreen"

                    _runtime_diag(f"store_frame: {frame_id} fullscreen_app={fullscreen_app} step=syn_ocr_start")
                    _tsyn0 = time_module.time()
                    syn_ocr_text = ""
                    syn_ocr_engine_name = "none"
                    syn_ocr_conf = 0.0
                    syn_ocr_regions_stored = False
                    syn_layout_text = ""
                    _elapsed_check = time_module.time() - _t0
                    if _elapsed_check > _HANDLER_MAX_SECONDS:
                        logger.warning(
                            f"store_frame: {frame_id} TIMEOUT GUARD — {_elapsed_check:.1f}s elapsed, "
                            f"skipping fullscreen OCR for {fullscreen_app}"
                        )
                    elif region_ocr_engine is not None:
                        try:
                            syn_regions = region_ocr_engine.recognize_regions(image)
                            _region_timing = getattr(region_ocr_engine, "last_timing", {}) or {}
                            timings["uied_total_ms"] += float(_region_timing.get("detector_ms", 0.0) or 0.0)
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
                                try:
                                    from core.activity.vlm_labeler import build_region_layout_text

                                    syn_layout_regions = []
                                    for r in syn_regions:
                                        bbox = r.get("bbox")
                                        text = r.get("text", "")
                                        if not bbox or not text:
                                            continue
                                        syn_layout_regions.append(
                                            {
                                                "bbox": bbox,
                                                "text": text,
                                                "image_width": syn_img_w,
                                                "image_height": syn_img_h,
                                            }
                                        )
                                    syn_layout_text = build_region_layout_text(syn_layout_regions)
                                except Exception:
                                    syn_layout_text = ""
                        except Exception as e:
                            logger.warning(f"Region OCR failed for fullscreen app sub_frame {fullscreen_app}: {e}")
                    timings["syn_ocr_ms"] += (time_module.time() - _tsyn0) * 1000.0

                    _runtime_diag(f"store_frame: {frame_id} fullscreen_app={fullscreen_app} step=syn_ocr_done len={len(syn_ocr_text)}")
                    # SQLite: sub_frames record (window_chunk_id=0 marks fullscreen app)
                    if sqlite_storage is not None:
                        sqlite_storage.store_sub_frame(
                            sub_frame_id=syn_sub_id,
                            timestamp=ts,
                            window_chunk_id=0,
                            offset_index=0,
                            app_name=fullscreen_app,
                            window_name=fullscreen_win or fullscreen_app,
                        )
                        sqlite_storage.add_frame_subframe_mapping(
                            frame_id=frame_id,
                            sub_frame_id=syn_sub_id,
                        )
                        sqlite_storage.store_frame_with_ocr(
                            frame_id=syn_sub_id,
                            timestamp=ts,
                            image_path=temp_image_path,
                            ocr_text="" if syn_ocr_regions_stored else syn_ocr_text,
                            ocr_text_json="",
                            ocr_engine=syn_ocr_engine_name,
                            ocr_confidence=syn_ocr_conf,
                            device_name=f"{fullscreen_app}/{fullscreen_win}",
                            app_name=fullscreen_app,
                            window_name=fullscreen_win or fullscreen_app,
                        )

                    _runtime_diag(f"store_frame: {frame_id} fullscreen_app={fullscreen_app} step=syn_sqlite_done")
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
                        "device_name": f"{fullscreen_app}/{fullscreen_win}",
                        "metadata": {
                            "is_fullscreen_synthetic": True,
                            "parent_frame_id": frame_id,
                            "fullscreen_label_source": "monitor_geometry",
                        },
                        "app_name": fullscreen_app,
                        "window_name": fullscreen_win or fullscreen_app,
                        "_ocr_regions_stored": syn_ocr_regions_stored,
                    }
                    batch_write_buffer.add_frame(syn_frame_data)

                    if cluster_manager is not None:
                        _tca0 = time_module.time()
                        try:
                            _runtime_diag(
                                f"store_frame: {frame_id} fullscreen_app={fullscreen_app} step=cluster_assign"
                            )
                            activity_label = cluster_manager.assign_frame(
                                app_name=fullscreen_app,
                                frame_id=syn_sub_id,
                                embedding=embedding,
                                image=image,
                                ocr_text=syn_ocr_text,
                                layout_text=syn_layout_text,
                                timestamp=ts.isoformat(),
                                window_name=fullscreen_win or "",
                            )
                            _runtime_diag(
                                f"store_frame: {frame_id} fullscreen_app={fullscreen_app} "
                                f"step=cluster_done label={activity_label} "
                                f"t={((time_module.time() - _tca0) * 1000):.0f}ms"
                            )
                            timings["cluster_assign_total_ms"] += (time_module.time() - _tca0) * 1000.0
                            if activity_label:
                                logger.debug(f"Assigned {syn_sub_id} -> '{activity_label}'")
                            _log_non_committed_cluster_result(fullscreen_app, syn_sub_id)
                        except Exception as e:
                            logger.warning(
                                f"store_frame: {frame_id} fullscreen_app={fullscreen_app} "
                                f"step=cluster_failed t={((time_module.time() - _tca0) * 1000):.0f}ms err={e}",
                                exc_info=True,
                            )

                    sub_frame_ids.append(syn_sub_id)
                    sub_frame_summaries.append({
                        "sub_frame_id": syn_sub_id,
                        "timestamp": ts.isoformat(),
                        "app_name": fullscreen_app,
                        "window_name": fullscreen_win or fullscreen_app,
                        "image_path": temp_image_path,
                    })

                    # Collect for frame-level combined OCR + update cache
                    if syn_ocr_text:
                        sub_frame_ocr_parts.append((fullscreen_app, syn_ocr_text, syn_ocr_conf))
                        _fullscreen_ocr_cache[fullscreen_app] = (syn_ocr_text, syn_ocr_conf)

                    logger.debug(
                        f"Created fullscreen app sub_frame {syn_sub_id} for "
                        f"full-screen app {fullscreen_app}/{fullscreen_win}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to create fullscreen app sub_frame for {fullscreen_app}: {e}")

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
        _tso0 = time_module.time()
        try:
            _runtime_diag(
                f"enrich_frame: {frame_id} step=main_store_ocr_start ocr_len={len(combined_ocr_text)}"
            )
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
                app_name=fullscreen_app or None,
                window_name=fullscreen_win or None,
                focused_app_name=focused_app or None,
                focused_window_name=focused_win or None,
            )
            _runtime_diag(
                f"enrich_frame: {frame_id} step=main_store_ocr_done "
                f"t={((time_module.time() - _tso0) * 1000):.0f}ms"
            )
            timings["main_store_ocr_ms"] = (time_module.time() - _tso0) * 1000.0
        except Exception as e:
            logger.error(
                f"enrich_frame: {frame_id} step=main_store_ocr_failed "
                f"t={((time_module.time() - _tso0) * 1000):.0f}ms err={e}",
                exc_info=True,
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
        "app_name": fullscreen_app or None,
        "window_name": fullscreen_win or None,
        "focused_app_name": focused_app or None,
        "focused_window_name": focused_win or None,
        "_ocr_regions_stored": True,  # Already stored above, skip duplicate in BatchWriteBuffer
    }
    _tba0 = time_module.time()
    _runtime_diag(f"enrich_frame: {frame_id} step=main_add_frame_start")
    batch_write_buffer.add_frame(frame_data)
    _runtime_diag(
        f"enrich_frame: {frame_id} step=main_add_frame_done "
        f"t={((time_module.time() - _tba0) * 1000):.0f}ms"
    )
    timings["add_frame_ms"] = (time_module.time() - _tba0) * 1000.0

    _elapsed = time_module.time() - _t0

    def _format_app_list(apps: List[str]) -> str:
        counts: Dict[str, int] = {}
        for a in apps:
            counts[a] = counts.get(a, 0) + 1
        return ", ".join(f"{a}×{c}" if c > 1 else a for a, c in counts.items())

    _summary_parts = [
        f"enrich_frame: {frame_id} done in {_elapsed:.2f}s "
        f"sub_frames={len(sub_frame_ids)} processed={len(_processed_apps)} "
        f"deduped={len(_skipped_dedup_apps)} solid={_skipped_solid} "
        f"focused={focused_app}"
    ]
    if _processed_reasons:
        for _r in _processed_reasons:
            _summary_parts.append(f"  + {_r}")
    if _skipped_dedup_apps:
        _summary_parts.append(f"  deduped: {_format_app_list(_skipped_dedup_apps)}")
    _runtime_diag("\n".join(_summary_parts))
    _emit_latency_event(
        {
            "event_type": "frame_latency",
            "frame_id": frame_id,
            "win_count": len(windows_to_process),
            "sub_frame_count": len(sub_frame_ids),
            "focused_app": focused_app or "",
            "total_enrich_ms": _elapsed * 1000.0,
            **timings,
        }
    )

    # Check if cluster recalculation is needed (runs in background thread).
    # Safe to launch from the enrichment worker — cluster_manager.recalculate
    # is idempotent and gated by should_recalculate().
    if cluster_manager is not None and cluster_manager.should_recalculate():
        if vector_storage is not None and vector_storage.table is not None:
            threading.Thread(
                target=cluster_manager.recalculate,
                args=(vector_storage.table,),
                daemon=True,
            ).start()


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
            with sqlite_storage._activity_connection() as conn_al:
                cursor_al = conn_al.cursor()
                cursor_al.execute(
                    "SELECT sub_frame_id FROM activity_assignments WHERE activity_label LIKE ?",
                    (f"%{req.activity_label}%",),
                )
                matching_ids = {r["sub_frame_id"] for r in cursor_al.fetchall()}
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
                    with sqlite_storage._connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT image_path FROM frames WHERE frame_id = ?", (fid,))
                        row = cursor.fetchone()
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


_TASK_MEMORY_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_TASK_MEMORY_OCR_LIMIT = 2000


def _task_memory_dir() -> Path:
    path = Path(config.STORAGE_ROOT) / "task_memories"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _task_memory_path(task_memory_id: str) -> Path:
    if not task_memory_id or not _TASK_MEMORY_ID_RE.match(task_memory_id):
        raise HTTPException(status_code=400, detail="Invalid task_memory_id")
    return _task_memory_dir() / f"{task_memory_id}.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip_text(text: Any, limit: int = _TASK_MEMORY_OCR_LIMIT) -> str:
    if text is None:
        return ""
    value = str(text).strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def _clip_one_line(text: Any, limit: int = 500) -> str:
    return _clip_text(" ".join(str(text or "").split()), limit)


def _parse_optional_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return _ensure_utc(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except Exception:
        return None


def _ts_to_iso(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid Task Memory JSON {path}: {e}")
    except OSError as e:
        logger.warning(f"Unable to read Task Memory {path}: {e}")
    return {}


def _write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _normalize_task_memory(raw: Dict[str, Any]) -> Dict[str, Any]:
    selected_segments = raw.get("selected_segments") or []
    evidence_refs = raw.get("evidence_refs") or []
    return {
        "task_memory_id": str(raw.get("task_memory_id") or ""),
        "title": str(raw.get("title") or "Untitled Task Memory"),
        "markdown": str(raw.get("markdown") or ""),
        "source_query": str(raw.get("source_query") or ""),
        "selected_segments": selected_segments if isinstance(selected_segments, list) else [],
        "evidence_refs": evidence_refs if isinstance(evidence_refs, list) else [],
        "created_at": str(raw.get("created_at") or ""),
        "updated_at": str(raw.get("updated_at") or ""),
    }


def _save_task_memory(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _normalize_task_memory(payload)
    path = _task_memory_path(normalized["task_memory_id"])
    _write_json_file(path, normalized)
    return normalized


def _load_task_memory(task_memory_id: str) -> Dict[str, Any]:
    path = _task_memory_path(task_memory_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Task Memory not found")
    memory = _normalize_task_memory(_read_json_file(path))
    if not memory["task_memory_id"]:
        memory["task_memory_id"] = task_memory_id
    return memory


def _list_task_memories() -> List[Dict[str, Any]]:
    memories = []
    for path in _task_memory_dir().glob("*.json"):
        data = _normalize_task_memory(_read_json_file(path))
        if not data["task_memory_id"]:
            data["task_memory_id"] = path.stem
        memories.append(data)
    memories.sort(key=lambda m: m.get("updated_at") or m.get("created_at") or "", reverse=True)
    return memories


def _activity_label_for_ids(frame_ids: List[str]) -> Optional[str]:
    ids = [fid for fid in frame_ids if fid]
    if not ids or sqlite_storage is None:
        return None
    try:
        placeholders = ",".join(["?"] * len(ids))
        with sqlite_storage._activity_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT activity_label, provisional_label
                FROM activity_assignments
                WHERE sub_frame_id IN ({placeholders})
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                ids,
            )
            row = cursor.fetchone()
        if not row:
            return None
        return row["activity_label"] or row["provisional_label"] or None
    except Exception as e:
        logger.debug(f"Task Memory activity label lookup failed: {e}")
        return None


_REWIND_QUESTION_FILLERS = (
    "你猜猜看",
    "猜猜看",
    "帮我看看",
    "帮我查查",
    "是谁啊",
    "是谁呀",
    "是谁呢",
    "是谁",
    "是什么",
    "是啥",
    "谁啊",
    "谁呀",
    "谁呢",
    "这个",
    "那个",
    "一下",
    "请问",
    "请",
    "啊",
    "呀",
    "呢",
    "吗",
    "？",
    "?",
)

_REWIND_RAG_SOURCE_WEIGHTS = {
    "dense_image": 1.0,
    "dense_ocr": 0.9,
    "sparse_fts": 0.85,
    "keyword_like": 0.75,
    "window_title": 0.65,
    "time_range_fallback": 0.45,
}


def _dedupe_strings(values: List[Any], limit: int = 20) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _rewind_local_rank_score(source: str, rank: int) -> float:
    safe_rank = max(int(rank or 1), 1)
    weight = _REWIND_RAG_SOURCE_WEIGHTS.get(source, 0.5)
    return float(weight) / (safe_rank ** 0.5)


def _clean_rewind_keyword(term: str) -> str:
    value = str(term or "").strip()
    for filler in sorted(_REWIND_QUESTION_FILLERS, key=len, reverse=True):
        value = value.replace(filler, " ")
    value = re.sub(r"[\s,，。；;：:!！?？\"'`“”‘’（）()【】\[\]{}<>《》]+", " ", value)
    return value.strip()


def _rewind_keyword_terms(query: str, sparse_queries: Optional[List[str]] = None) -> List[str]:
    candidates = [query]
    candidates.extend(sparse_queries or [])
    terms: List[str] = []
    for candidate in candidates:
        cleaned = _clean_rewind_keyword(candidate)
        if cleaned:
            terms.append(cleaned)
        for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9_@#.\-]{2,}", str(candidate or "")):
            token = _clean_rewind_keyword(token)
            if len(token) >= 2:
                terms.append(token)
    return _dedupe_strings(terms, limit=12)


def _merge_rewind_time_range(
    explicit_start: Optional[datetime],
    explicit_end: Optional[datetime],
    llm_time_range: Optional[Tuple[datetime, datetime]],
) -> Tuple[Optional[datetime], Optional[datetime]]:
    start_time = explicit_start
    end_time = explicit_end
    if llm_time_range is None:
        return start_time, end_time

    llm_start, llm_end = (_ensure_utc(llm_time_range[0]), _ensure_utc(llm_time_range[1]))
    if explicit_start and llm_start:
        start_time = max(explicit_start, llm_start)
    elif llm_start:
        start_time = llm_start

    if explicit_end and llm_end:
        end_time = min(explicit_end, llm_end)
    elif llm_end:
        end_time = llm_end

    if start_time and end_time and start_time > end_time:
        if explicit_start or explicit_end:
            return explicit_start, explicit_end
        return llm_start, llm_end
    return start_time, end_time


def _build_rewind_retrieval_plan(
    query: str,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
) -> Dict[str, Any]:
    dense_queries = [query] if query.strip() else []
    sparse_queries = [query] if query.strip() else []
    related_apps = None
    unrelated_apps = None
    window_filters = None
    llm_time_range = None
    controller_used = False
    controller_error = None

    if query.strip() and config.REWIND_ENABLE_AGENTIC_SEARCH:
        try:
            rewrite_result = rewrite_and_time(
                query,
                enable_rewrite=config.ENABLE_LLM_REWRITE,
                enable_time=config.ENABLE_TIME_FILTER,
                expand_n=config.QUERY_REWRITE_NUM,
                api_client=vlm,
            )
            if len(rewrite_result) == 6:
                (
                    dense_llm,
                    sparse_llm,
                    llm_time_range,
                    related_apps,
                    unrelated_apps,
                    window_filters,
                ) = rewrite_result
            else:
                dense_llm, sparse_llm, llm_time_range, related_apps, unrelated_apps = rewrite_result
                window_filters = None
            if config.ENABLE_LLM_REWRITE:
                dense_queries = dense_llm or dense_queries
                sparse_queries = sparse_llm or sparse_queries
            controller_used = True
        except Exception as e:
            controller_error = str(e)
            logger.warning(f"Rewind retrieval controller failed, using deterministic hops: {e}")

    merged_start, merged_end = _merge_rewind_time_range(start_dt, end_dt, llm_time_range)
    keyword_terms = _rewind_keyword_terms(query, sparse_queries)
    dense_queries = _dedupe_strings(dense_queries, limit=8)
    sparse_queries = _dedupe_strings([*sparse_queries, *keyword_terms], limit=16)

    return {
        "mode": "agentic_session" if config.REWIND_ENABLE_AGENTIC_SEARCH else "legacy",
        "controller_used": controller_used,
        "controller_error": controller_error,
        "dense_queries": dense_queries,
        "sparse_queries": sparse_queries,
        "keyword_terms": keyword_terms,
        "start_time": merged_start,
        "end_time": merged_end,
        "related_apps": related_apps,
        "unrelated_apps": unrelated_apps,
        "window_filters": window_filters,
        "hops": [
            "llm_rewrite_time_app_window",
            "activity_session_candidate_filter",
            "parallel_cloud_label_scoring",
            "parallel_local_multihop_rag",
            "rag_frame_to_activity_session_reverse_map",
            "label_rag_weighted_fusion",
        ],
    }


def _rewind_filter_app_sets(plan: Dict[str, Any]) -> Tuple[set, set, Dict[str, List[str]], Dict[str, List[str]]]:
    include_apps = set(plan.get("related_apps") or [])
    exclude_apps = set(plan.get("unrelated_apps") or [])
    window_filters = plan.get("window_filters") or {}
    include_windows = window_filters.get("include") or {}
    exclude_windows = window_filters.get("exclude") or {}
    include_apps.update(app for app in include_windows.keys() if app)
    exclude_apps.update(app for app in exclude_windows.keys() if app)
    return include_apps, exclude_apps, include_windows, exclude_windows


def _session_window_names(session: Dict[str, Any], limit: int = 8) -> List[str]:
    if sqlite_storage is None:
        return []
    start = session.get("start_time")
    end = session.get("end_time")
    app_name = session.get("app_name") or ""
    if not start or not end:
        return []
    try:
        with sqlite_storage._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT sf.window_name
                FROM sub_frames sf
                WHERE sf.timestamp >= ? AND sf.timestamp <= ?
                  AND (? = '' OR sf.app_name = ?)
                  AND sf.window_name IS NOT NULL AND sf.window_name != ''
                ORDER BY sf.timestamp DESC
                LIMIT ?
                """,
                (start, end, app_name, app_name, limit),
            )
            return [row["window_name"] for row in cursor.fetchall() if row["window_name"]]
    except Exception as e:
        logger.debug(f"Rewind session window-name lookup failed: {e}")
        return []


def _session_matches_plan_filters(session: Dict[str, Any], plan: Dict[str, Any]) -> bool:
    app = session.get("app_name") or ""
    include_apps, exclude_apps, include_windows, exclude_windows = _rewind_filter_app_sets(plan)
    if include_apps and app not in include_apps:
        return False
    if exclude_apps and app in exclude_apps:
        return False

    windows = _session_window_names(session, limit=12)
    if include_windows:
        included = False
        for include_app, allowed_windows in include_windows.items():
            if include_app and include_app != app:
                continue
            if not allowed_windows or any(w in allowed_windows for w in windows):
                included = True
                break
        if not included:
            return False
    for exclude_app, blocked_windows in exclude_windows.items():
        if exclude_app and exclude_app != app:
            continue
        if not blocked_windows or any(w in blocked_windows for w in windows):
            return False
    return True


def _candidate_activity_sessions_for_plan(
    plan: Dict[str, Any],
    limit: int,
) -> List[Dict[str, Any]]:
    """Collect app/window/label session candidates before the parallel loops."""
    if sqlite_storage is None:
        return []
    start_dt = plan.get("start_time")
    end_dt = plan.get("end_time")
    include_apps, exclude_apps, _include_windows, _exclude_windows = _rewind_filter_app_sets(plan)
    keyword_terms = plan.get("keyword_terms") or []
    sessions: List[Dict[str, Any]] = []

    try:
        with sqlite_storage._activity_connection() as conn:
            cursor = conn.cursor()
            where = ["session_status IN ('committed', 'candidate')"]
            params: List[Any] = []
            if start_dt:
                where.append("end_time >= ?")
                params.append(_sql_dt(start_dt))
            if end_dt:
                where.append("start_time <= ?")
                params.append(_sql_dt(end_dt))
            if include_apps:
                placeholders = ",".join("?" for _ in include_apps)
                where.append(f"app_name IN ({placeholders})")
                params.extend(sorted(include_apps))
            if exclude_apps:
                placeholders = ",".join("?" for _ in exclude_apps)
                where.append(f"app_name NOT IN ({placeholders})")
                params.extend(sorted(exclude_apps))

            if not include_apps and keyword_terms:
                term_clauses = []
                for term in keyword_terms[:6]:
                    term_clauses.append("(label LIKE ? ESCAPE '\\' OR app_name LIKE ? ESCAPE '\\')")
                    params.extend([_like_pattern(term), _like_pattern(term)])
                where.append("(" + " OR ".join(term_clauses) + ")")

            params.append(min(max(limit, 20), 160))
            cursor.execute(
                f"""
                SELECT id, app_name, cluster_id, label, start_time, end_time,
                       frame_count, session_status
                FROM activity_sessions
                WHERE {" AND ".join(where)}
                ORDER BY
                    CASE WHEN session_status = 'committed' THEN 0 ELSE 1 END,
                    frame_count DESC,
                    end_time DESC
                LIMIT ?
                """,
                tuple(params),
            )
            sessions = [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.debug(f"Rewind candidate activity-session collection failed: {e}")
        return []

    candidates: List[Dict[str, Any]] = []
    seen = set()
    for session in sessions:
        if not _session_matches_plan_filters(session, plan):
            continue
        key = _session_key(session)
        if key in seen:
            continue
        seen.add(key)
        enriched = dict(session)
        enriched["window_names"] = _session_window_names(session)
        candidates.append(enriched)
        if len(candidates) >= limit:
            break

    logger.info(
        "Rewind agentic candidate sessions: "
        f"count={len(candidates)} include_apps={sorted(include_apps)} "
        f"exclude_apps={sorted(exclude_apps)}"
    )
    for idx, candidate in enumerate(candidates[:20], start=1):
        logger.info(
            "Rewind candidate[%02d]: session=%s app=%s label=%s windows=%s time=%s->%s frames=%s",
            idx,
            candidate.get("id"),
            candidate.get("app_name"),
            candidate.get("label"),
            candidate.get("window_names") or [],
            candidate.get("start_time"),
            candidate.get("end_time"),
            candidate.get("frame_count"),
        )
    return candidates


def _session_identifier(session: Dict[str, Any]) -> str:
    if session.get("id") is not None:
        return str(session.get("id"))
    return "|".join(
        [
            str(session.get("app_name") or ""),
            str(session.get("label") or ""),
            str(session.get("start_time") or ""),
            str(session.get("end_time") or ""),
        ]
    )


def _parse_rewind_json_payload(text: Any) -> Optional[Any]:
    raw = str(text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL):
        candidates.insert(0, match.group(1).strip())
    object_start = raw.find("{")
    object_end = raw.rfind("}")
    if object_start >= 0 and object_end > object_start:
        candidates.append(raw[object_start : object_end + 1])
    array_start = raw.find("[")
    array_end = raw.rfind("]")
    if array_start >= 0 and array_end > array_start:
        candidates.append(raw[array_start : array_end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _fallback_rewind_label_score(
    query: str,
    plan: Dict[str, Any],
    session: Dict[str, Any],
) -> Tuple[float, str]:
    include_apps, exclude_apps, include_windows, exclude_windows = _rewind_filter_app_sets(plan)
    app = str(session.get("app_name") or "")
    label = str(session.get("label") or "")
    windows = [str(w or "") for w in session.get("window_names") or []]
    haystack = " ".join([app, label, *windows]).lower()
    terms = [str(term or "").strip().lower() for term in plan.get("keyword_terms") or [] if str(term or "").strip()]

    score = 0.05
    reasons: List[str] = []
    if include_apps and app in include_apps:
        score += 0.35
        reasons.append("include_app")
    if exclude_apps and app in exclude_apps:
        return 0.0, "excluded_app"

    matched_terms = [term for term in terms if term and term in haystack]
    if terms:
        score += 0.4 * (len(matched_terms) / max(len(terms), 1))
        if matched_terms:
            reasons.append("keyword:" + ",".join(matched_terms[:4]))
    elif query.strip():
        query_tokens = [
            token.lower()
            for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9_@#.\-]{2,}", query)
            if token.strip()
        ]
        overlap = [token for token in query_tokens if token in haystack]
        score += 0.25 * (len(overlap) / max(len(query_tokens), 1)) if query_tokens else 0.0
        if overlap:
            reasons.append("query_overlap:" + ",".join(overlap[:4]))

    if include_windows:
        for include_app, allowed_windows in include_windows.items():
            if include_app and include_app != app:
                continue
            if not allowed_windows:
                score += 0.15
                reasons.append("include_window_app")
                break
            matched_windows = [w for w in windows if w in allowed_windows]
            if matched_windows:
                score += 0.15
                reasons.append("include_window:" + ",".join(matched_windows[:2]))
                break
    for exclude_app, blocked_windows in exclude_windows.items():
        if exclude_app and exclude_app != app:
            continue
        if not blocked_windows or any(w in blocked_windows for w in windows):
            return 0.0, "excluded_window"

    frame_count = float(session.get("frame_count") or 0)
    if frame_count > 0:
        score += min(frame_count / 500.0, 0.1)
    return min(score, 1.0), "; ".join(reasons) or "fallback_prior"


def _score_rewind_label_candidates(
    query: str,
    plan: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    fallback_scores = {
        _session_identifier(session): _fallback_rewind_label_score(query, plan, session)
        for session in candidates
    }
    llm_scores: Dict[str, Tuple[float, str]] = {}
    llm_error = None

    compact_candidates = [
        {
            "id": _session_identifier(session),
            "app": session.get("app_name") or "",
            "windows": (session.get("window_names") or [])[:5],
            "label": session.get("label") or "",
            "start": session.get("start_time") or "",
            "end": session.get("end_time") or "",
            "frames": session.get("frame_count") or 0,
        }
        for session in candidates[:80]
    ]

    system_prompt = (
        "你是 VisualMem Rewind 的检索控制器。你只根据候选 activity sessions 的 app、window、label、"
        "时间段判断它们是否能回答用户问题。输出严格 JSON，不要输出 Markdown。"
    )
    prompt = {
        "task": "score_activity_session_labels_for_rewind_search",
        "user_query": query,
        "rewrite_dense_queries": plan.get("dense_queries") or [],
        "rewrite_sparse_queries": plan.get("sparse_queries") or [],
        "include_apps": plan.get("related_apps") or [],
        "exclude_apps": plan.get("unrelated_apps") or [],
        "window_filters": plan.get("window_filters") or {},
        "time_range": {
            "start": _ts_to_iso(plan.get("start_time")),
            "end": _ts_to_iso(plan.get("end_time")),
        },
        "candidates": compact_candidates,
        "output_schema": {
            "scores": [
                {
                    "id": "candidate id",
                    "score": "number from 0.0 to 1.0",
                    "reason": "short reason, mention label/app/window evidence",
                }
            ]
        },
    }

    try:
        ai = vlm or ApiVLM()
        response = ai._call_vlm_text_only(
            json.dumps(prompt, ensure_ascii=False),
            system_prompt=system_prompt,
        )
        payload = _parse_rewind_json_payload(response)
        score_items = payload.get("scores") if isinstance(payload, dict) else payload
        if isinstance(score_items, list):
            for item in score_items:
                if not isinstance(item, dict):
                    continue
                sid = str(item.get("id") or item.get("session_id") or "")
                if not sid:
                    continue
                try:
                    score = max(0.0, min(float(item.get("score") or 0.0), 1.0))
                except Exception:
                    score = 0.0
                reason = _clip_one_line(item.get("reason"), 220)
                llm_scores[sid] = (score, reason or "llm_selected")
    except Exception as e:
        llm_error = str(e)
        logger.debug(f"Rewind LLM label scoring failed, using fallback scores: {e}")

    scored: List[Dict[str, Any]] = []
    for session in candidates:
        sid = _session_identifier(session)
        fallback_score, fallback_reason = fallback_scores.get(sid, (0.0, "fallback_missing"))
        score, reason = llm_scores.get(sid, (fallback_score, fallback_reason))
        source = "cloud_llm" if sid in llm_scores else "fallback"
        scored.append(
            {
                "session": session,
                "session_id": sid,
                "llm_label_score": score,
                "llm_label_reason": reason,
                "llm_label_score_source": source,
                "fallback_label_score": fallback_score,
                "fallback_label_reason": fallback_reason,
                "llm_label_error": llm_error,
            }
        )

    scored.sort(key=lambda item: item.get("llm_label_score") or 0.0, reverse=True)
    logger.info(
        "Rewind agentic top label scores: %s",
        [
            {
                "session": item.get("session_id"),
                "score": round(float(item.get("llm_label_score") or 0.0), 3),
                "source": item.get("llm_label_score_source"),
                "app": item.get("session", {}).get("app_name"),
                "label": item.get("session", {}).get("label"),
                "reason": item.get("llm_label_reason"),
            }
            for item in scored[:12]
        ],
    )
    return scored[: max(limit, 1)]


def _safe_metadata_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _dt_from_any(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_utc(value)
    return _parse_optional_dt(str(value))


def _sql_dt(value: Optional[datetime]) -> Optional[str]:
    return _ensure_utc(value).isoformat() if value else None


def _like_pattern(term: str) -> str:
    escaped = str(term).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _app_window_allowed(
    app_name: Optional[str],
    window_name: Optional[str],
    related_apps: Optional[List[str]],
    unrelated_apps: Optional[List[str]],
    window_filters: Optional[Dict[str, Dict[str, List[str]]]],
) -> bool:
    app = app_name or ""
    window = window_name or ""
    include_map = (window_filters or {}).get("include") or {}
    exclude_map = (window_filters or {}).get("exclude") or {}

    if include_map:
        included = False
        for include_app, windows in include_map.items():
            if include_app and include_app != app:
                continue
            if not windows or window in windows:
                included = True
                break
        if not included:
            return False
    elif related_apps and app not in related_apps:
        return False

    for exclude_app, windows in exclude_map.items():
        if exclude_app and exclude_app != app:
            continue
        if not windows or window in windows:
            return False
    if unrelated_apps and app in unrelated_apps:
        return False
    return True


def _parent_frame_id_for_sub_frame(sub_frame_id: Optional[str]) -> Optional[str]:
    if not sub_frame_id or sqlite_storage is None:
        return None
    try:
        with sqlite_storage._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT frame_id
                FROM frame_subframe_mapping
                WHERE sub_frame_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (sub_frame_id,),
            )
            row = cursor.fetchone()
        return row["frame_id"] if row else None
    except Exception as e:
        logger.debug(f"Rewind parent-frame lookup failed for {sub_frame_id}: {e}")
        return None


def _normalize_rewind_hit(frame: Dict[str, Any], source: str, query_text: str, rank: int) -> Optional[Dict[str, Any]]:
    fid = frame.get("frame_id")
    metadata = _safe_metadata_dict(frame.get("metadata"))
    sub_frame_id = frame.get("sub_frame_id") or metadata.get("sub_frame_id")
    parent_frame_id = metadata.get("parent_frame_id")

    if fid and not parent_frame_id:
        parent_frame_id = _parent_frame_id_for_sub_frame(fid)
        if parent_frame_id:
            sub_frame_id = sub_frame_id or fid
            fid = parent_frame_id

    if not fid and sub_frame_id:
        fid = _parent_frame_id_for_sub_frame(sub_frame_id) or sub_frame_id
    if not fid:
        return None

    timestamp = _dt_from_any(frame.get("timestamp"))
    local_rank_score = _rewind_local_rank_score(source, rank)
    rag_detail = {
        "source": source,
        "query": query_text,
        "rank": rank,
        "score": round(local_rank_score, 4),
    }
    metadata.update(
        {
            "retrieval_sources": [source],
            "retrieval_queries": [query_text] if query_text else [],
            "retrieval_rank": rank,
            "rag_details": [rag_detail],
        }
    )
    if sub_frame_id:
        metadata["sub_frame_id"] = sub_frame_id
    if parent_frame_id:
        metadata["parent_frame_id"] = parent_frame_id

    normalized = dict(frame)
    normalized["frame_id"] = fid
    normalized["timestamp"] = timestamp or frame.get("timestamp")
    normalized["metadata"] = metadata
    if sub_frame_id:
        normalized["sub_frame_id"] = sub_frame_id
    normalized["_local_rag_score"] = float(frame.get("_local_rag_score") or 0.0) + local_rank_score
    normalized["_rewind_score"] = (
        float(frame.get("_rewind_score") or 0.0)
        + max(0.0, 10.0 - rank)
        + local_rank_score * 10.0
    )
    return normalized


def _add_rewind_hit(
    hits_by_key: Dict[Tuple[str, str], Dict[str, Any]],
    frame: Dict[str, Any],
    source: str,
    query_text: str,
    rank: int,
) -> None:
    hit = _normalize_rewind_hit(frame, source, query_text, rank)
    if not hit:
        return
    key = (hit.get("frame_id") or "", hit.get("sub_frame_id") or hit.get("metadata", {}).get("sub_frame_id") or "")
    existing = hits_by_key.get(key)
    if existing is None:
        hits_by_key[key] = hit
        return
    existing["_rewind_score"] = float(existing.get("_rewind_score") or 0.0) + float(hit.get("_rewind_score") or 0.0)
    existing["_local_rag_score"] = (
        float(existing.get("_local_rag_score") or 0.0)
        + float(hit.get("_local_rag_score") or 0.0)
    )
    existing_ocr = existing.get("ocr_text") or ""
    if len(hit.get("ocr_text") or "") > len(existing_ocr):
        existing["ocr_text"] = hit.get("ocr_text")
    metadata = existing.setdefault("metadata", {})
    hit_metadata = hit.get("metadata", {})
    if hit_metadata.get("retrieval_rank") is not None:
        current_rank = metadata.get("retrieval_rank")
        metadata["retrieval_rank"] = (
            min(int(current_rank), int(hit_metadata["retrieval_rank"]))
            if current_rank is not None
            else hit_metadata["retrieval_rank"]
        )
    for value in hit.get("metadata", {}).get("retrieval_sources") or []:
        if value not in metadata.setdefault("retrieval_sources", []):
            metadata["retrieval_sources"].append(value)
    for value in hit.get("metadata", {}).get("retrieval_queries") or []:
        if value not in metadata.setdefault("retrieval_queries", []):
            metadata["retrieval_queries"].append(value)
    seen_details = {
        (
            detail.get("source"),
            detail.get("query"),
            detail.get("rank"),
        )
        for detail in metadata.setdefault("rag_details", [])
        if isinstance(detail, dict)
    }
    for detail in hit_metadata.get("rag_details") or []:
        if not isinstance(detail, dict):
            continue
        detail_key = (detail.get("source"), detail.get("query"), detail.get("rank"))
        if detail_key in seen_details:
            continue
        seen_details.add(detail_key)
        metadata["rag_details"].append(detail)


def _rewind_keyword_like_search(
    term: str,
    limit: int,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
    related_apps: Optional[List[str]],
    unrelated_apps: Optional[List[str]],
    window_filters: Optional[Dict[str, Dict[str, List[str]]]],
) -> List[Dict[str, Any]]:
    if sqlite_storage is None or not term:
        return []
    results: List[Dict[str, Any]] = []
    try:
        with sqlite_storage._connection() as conn:
            cursor = conn.cursor()
            where = ["o.text LIKE ? ESCAPE '\\'"]
            params: List[Any] = [_like_pattern(term)]
            if start_dt:
                where.append("COALESCE(sf.timestamp, f.timestamp, pf.timestamp) >= ?")
                params.append(_sql_dt(start_dt))
            if end_dt:
                where.append("COALESCE(sf.timestamp, f.timestamp, pf.timestamp) <= ?")
                params.append(_sql_dt(end_dt))
            params.append(limit)
            cursor.execute(
                f"""
                SELECT
                    o.frame_id AS o_frame_id,
                    o.sub_frame_id AS o_sub_frame_id,
                    o.text AS ocr_text,
                    o.confidence AS ocr_confidence,
                    f.frame_id AS frame_id,
                    f.timestamp AS frame_timestamp,
                    f.image_path AS frame_image_path,
                    f.device_name AS device_name,
                    f.metadata AS frame_metadata,
                    f.app_name AS frame_app_name,
                    f.window_name AS frame_window_name,
                    f.focused_app_name AS focused_app_name,
                    f.focused_window_name AS focused_window_name,
                    sf.sub_frame_id AS sub_frame_id,
                    sf.timestamp AS sub_timestamp,
                    sf.app_name AS sub_app_name,
                    sf.window_name AS sub_window_name,
                    sf.window_chunk_id,
                    sf.offset_index,
                    pf.frame_id AS parent_frame_id,
                    pf.image_path AS parent_image_path,
                    pf.metadata AS parent_metadata
                FROM ocr_text o
                LEFT JOIN frames f ON o.frame_id = f.frame_id
                LEFT JOIN sub_frames sf ON o.sub_frame_id = sf.sub_frame_id
                LEFT JOIN frame_subframe_mapping fsm ON sf.sub_frame_id = fsm.sub_frame_id
                LEFT JOIN frames pf ON fsm.frame_id = pf.frame_id
                WHERE {" AND ".join(where)}
                ORDER BY COALESCE(sf.timestamp, f.timestamp, pf.timestamp) DESC
                LIMIT ?
                """,
                tuple(params),
            )
            rows = cursor.fetchall()

        for row in rows:
            app_name = row["sub_app_name"] or row["frame_app_name"] or row["focused_app_name"] or ""
            window_name = row["sub_window_name"] or row["frame_window_name"] or row["focused_window_name"] or ""
            if not _app_window_allowed(app_name, window_name, related_apps, unrelated_apps, window_filters):
                continue
            sf_id = row["sub_frame_id"] or row["o_sub_frame_id"]
            image_path = row["frame_image_path"] or row["parent_image_path"]
            if sf_id:
                image_path = _resolve_sub_frame_image_path(
                    {
                        "sub_frame_id": sf_id,
                        "window_chunk_id": row["window_chunk_id"],
                        "offset_index": row["offset_index"],
                    }
                ) or image_path
            results.append(
                {
                    "frame_id": row["frame_id"] or row["parent_frame_id"] or row["o_frame_id"] or sf_id,
                    "sub_frame_id": sf_id,
                    "timestamp": _dt_from_any(row["sub_timestamp"] or row["frame_timestamp"]),
                    "image_path": image_path,
                    "device_name": row["device_name"],
                    "metadata": {
                        **_safe_metadata_dict(row["frame_metadata"] or row["parent_metadata"]),
                        "sub_frame_id": sf_id,
                        "parent_frame_id": row["parent_frame_id"],
                    },
                    "app_name": app_name,
                    "window_name": window_name,
                    "ocr_text": row["ocr_text"] or "",
                    "ocr_confidence": row["ocr_confidence"] or 0.0,
                }
            )
    except Exception as e:
        logger.debug(f"Rewind keyword LIKE search failed for '{term}': {e}")
    return results


def _rewind_window_search(
    term: str,
    limit: int,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
) -> List[Dict[str, Any]]:
    if sqlite_storage is None or not term:
        return []
    results: List[Dict[str, Any]] = []
    try:
        with sqlite_storage._connection() as conn:
            cursor = conn.cursor()
            pattern = _like_pattern(term)
            frame_where = [
                "(f.app_name LIKE ? ESCAPE '\\' OR f.window_name LIKE ? ESCAPE '\\' "
                "OR f.focused_app_name LIKE ? ESCAPE '\\' OR f.focused_window_name LIKE ? ESCAPE '\\')"
            ]
            frame_params: List[Any] = [pattern, pattern, pattern, pattern]
            if start_dt:
                frame_where.append("f.timestamp >= ?")
                frame_params.append(_sql_dt(start_dt))
            if end_dt:
                frame_where.append("f.timestamp <= ?")
                frame_params.append(_sql_dt(end_dt))
            frame_params.append(limit)
            cursor.execute(
                f"""
                SELECT f.frame_id, f.timestamp, f.image_path, f.device_name, f.metadata,
                       f.app_name, f.window_name, f.focused_app_name, f.focused_window_name,
                       o.text AS ocr_text, o.confidence AS ocr_confidence
                FROM frames f
                LEFT JOIN ocr_text o ON f.frame_id = o.frame_id
                WHERE {" AND ".join(frame_where)}
                ORDER BY f.timestamp DESC
                LIMIT ?
                """,
                tuple(frame_params),
            )
            frame_rows = cursor.fetchall()

            sub_where = ["(sf.app_name LIKE ? ESCAPE '\\' OR sf.window_name LIKE ? ESCAPE '\\')"]
            sub_params: List[Any] = [pattern, pattern]
            if start_dt:
                sub_where.append("sf.timestamp >= ?")
                sub_params.append(_sql_dt(start_dt))
            if end_dt:
                sub_where.append("sf.timestamp <= ?")
                sub_params.append(_sql_dt(end_dt))
            sub_params.append(limit)
            cursor.execute(
                f"""
                SELECT sf.sub_frame_id, sf.timestamp, sf.app_name, sf.window_name,
                       sf.window_chunk_id, sf.offset_index, fsm.frame_id AS parent_frame_id,
                       f.image_path AS parent_image_path, o.text AS ocr_text, o.confidence AS ocr_confidence
                FROM sub_frames sf
                LEFT JOIN frame_subframe_mapping fsm ON sf.sub_frame_id = fsm.sub_frame_id
                LEFT JOIN frames f ON fsm.frame_id = f.frame_id
                LEFT JOIN ocr_text o ON sf.sub_frame_id = o.sub_frame_id
                WHERE {" AND ".join(sub_where)}
                ORDER BY sf.timestamp DESC
                LIMIT ?
                """,
                tuple(sub_params),
            )
            sub_rows = cursor.fetchall()

        for row in frame_rows:
            results.append(
                {
                    "frame_id": row["frame_id"],
                    "timestamp": _dt_from_any(row["timestamp"]),
                    "image_path": row["image_path"],
                    "device_name": row["device_name"],
                    "metadata": _safe_metadata_dict(row["metadata"]),
                    "app_name": row["app_name"] or row["focused_app_name"] or "",
                    "window_name": row["window_name"] or row["focused_window_name"] or "",
                    "ocr_text": row["ocr_text"] or "",
                    "ocr_confidence": row["ocr_confidence"] or 0.0,
                }
            )
        for row in sub_rows:
            image_path = _resolve_sub_frame_image_path(dict(row)) or row["parent_image_path"]
            results.append(
                {
                    "frame_id": row["parent_frame_id"] or row["sub_frame_id"],
                    "sub_frame_id": row["sub_frame_id"],
                    "timestamp": _dt_from_any(row["timestamp"]),
                    "image_path": image_path,
                    "metadata": {
                        "sub_frame_id": row["sub_frame_id"],
                        "parent_frame_id": row["parent_frame_id"],
                    },
                    "app_name": row["app_name"] or "",
                    "window_name": row["window_name"] or "",
                    "ocr_text": row["ocr_text"] or "",
                    "ocr_confidence": row["ocr_confidence"] or 0.0,
                }
            )
    except Exception as e:
        logger.debug(f"Rewind window-title search failed for '{term}': {e}")
    return results


def _rewind_activity_sessions_by_terms(
    terms: List[str],
    limit: int,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
) -> List[Dict[str, Any]]:
    if sqlite_storage is None or not terms:
        return []
    sessions: List[Dict[str, Any]] = []
    try:
        with sqlite_storage._activity_connection() as conn:
            cursor = conn.cursor()
            for term in terms:
                pattern = _like_pattern(term)
                where = [
                    "(label LIKE ? ESCAPE '\\' OR app_name LIKE ? ESCAPE '\\')",
                    "session_status IN ('committed', 'candidate')",
                ]
                params: List[Any] = [pattern, pattern]
                if start_dt:
                    where.append("end_time >= ?")
                    params.append(_sql_dt(start_dt))
                if end_dt:
                    where.append("start_time <= ?")
                    params.append(_sql_dt(end_dt))
                params.append(limit)
                cursor.execute(
                    f"""
                    SELECT id, app_name, cluster_id, label, start_time, end_time,
                           frame_count, session_status
                    FROM activity_sessions
                    WHERE {" AND ".join(where)}
                    ORDER BY frame_count DESC, end_time DESC
                    LIMIT ?
                    """,
                    tuple(params),
                )
                sessions.extend(dict(row) for row in cursor.fetchall())
    except Exception as e:
        logger.debug(f"Rewind activity-session term search failed: {e}")

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for session in sessions:
        key = (
            session.get("id"),
            session.get("app_name"),
            session.get("label"),
            session.get("start_time"),
            session.get("end_time"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(session)
        if len(deduped) >= limit:
            break
    return deduped


def _activity_session_for_hit(hit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if sqlite_storage is None:
        return None
    ts = _dt_from_any(hit.get("timestamp"))
    if not ts:
        return None
    ts_iso = _sql_dt(ts)
    app_name = hit.get("app_name") or ""
    sub_frame_id = hit.get("sub_frame_id") or hit.get("metadata", {}).get("sub_frame_id")
    cluster_id = None
    try:
        with sqlite_storage._activity_connection() as conn:
            cursor = conn.cursor()
            if sub_frame_id:
                cursor.execute(
                    """
                    SELECT app_name, activity_cluster_id, activity_label, provisional_label
                    FROM activity_assignments
                    WHERE sub_frame_id = ?
                    LIMIT 1
                    """,
                    (sub_frame_id,),
                )
                assignment = cursor.fetchone()
                if assignment:
                    app_name = app_name or assignment["app_name"] or ""
                    cluster_id = assignment["activity_cluster_id"]
            clauses = [
                "start_time <= ?",
                "end_time >= ?",
                "session_status IN ('committed', 'candidate')",
            ]
            params: List[Any] = [ts_iso, ts_iso]
            if cluster_id is not None:
                clauses.append("(cluster_id = ? OR app_name = ?)")
                params.extend([cluster_id, app_name])
            elif app_name:
                clauses.append("app_name = ?")
                params.append(app_name)
            cursor.execute(
                f"""
                SELECT id, app_name, cluster_id, label, start_time, end_time,
                       frame_count, session_status
                FROM activity_sessions
                WHERE {" AND ".join(clauses)}
                ORDER BY
                    CASE WHEN cluster_id = ? THEN 0 ELSE 1 END,
                    CASE WHEN session_status = 'committed' THEN 0 ELSE 1 END,
                    frame_count DESC
                LIMIT 1
                """,
                tuple([*params, cluster_id if cluster_id is not None else -1]),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)

            cursor.execute(
                """
                SELECT id, app_name, cluster_id, label, start_time, end_time,
                       frame_count, session_status
                FROM activity_sessions
                WHERE start_time <= ? AND end_time >= ?
                  AND session_status IN ('committed', 'candidate')
                ORDER BY frame_count DESC
                LIMIT 1
                """,
                (ts_iso, ts_iso),
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    except Exception as e:
        logger.debug(f"Rewind activity-session lookup failed: {e}")
        return None


def _sub_frames_for_activity_session(session: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    if sqlite_storage is None:
        return []
    start = session.get("start_time")
    end = session.get("end_time")
    app_name = session.get("app_name") or ""
    sub_frames: List[Dict[str, Any]] = []
    try:
        with sqlite_storage._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT sf.sub_frame_id, sf.timestamp, sf.app_name, sf.window_name,
                       sf.window_chunk_id, sf.offset_index, fsm.frame_id AS parent_frame_id
                FROM sub_frames sf
                LEFT JOIN frame_subframe_mapping fsm ON sf.sub_frame_id = fsm.sub_frame_id
                WHERE sf.timestamp >= ? AND sf.timestamp <= ?
                  AND (? = '' OR sf.app_name = ?)
                ORDER BY sf.timestamp ASC
                LIMIT ?
                """,
                (start, end, app_name, app_name, limit),
            )
            rows = cursor.fetchall()
        for row in rows:
            sf = {
                "sub_frame_id": row["sub_frame_id"],
                "timestamp": _ts_to_iso(row["timestamp"]),
                "app_name": row["app_name"] or "",
                "window_name": row["window_name"] or "",
                "window_chunk_id": row["window_chunk_id"],
                "offset_index": row["offset_index"],
                "parent_frame_id": row["parent_frame_id"],
            }
            sf["image_path"] = _resolve_sub_frame_image_path(sf)
            sub_frames.append(sf)
    except Exception as e:
        logger.debug(f"Rewind session sub-frame lookup failed: {e}")
    return sub_frames


def _frames_for_rewind_span(
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
    limit: int,
) -> List[Dict[str, Any]]:
    if sqlite_storage is None or not start_dt or not end_dt:
        return []
    try:
        with sqlite_storage._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT f.frame_id, f.timestamp, f.image_path, f.device_name, f.metadata,
                       f.app_name, f.window_name, f.focused_app_name, f.focused_window_name,
                       o.text AS ocr_text, o.confidence AS ocr_confidence
                FROM frames f
                LEFT JOIN ocr_text o ON f.frame_id = o.frame_id
                WHERE f.timestamp >= ? AND f.timestamp <= ?
                  AND f.frame_id LIKE 'frame_%'
                ORDER BY f.timestamp ASC
                LIMIT ?
                """,
                (_sql_dt(start_dt), _sql_dt(end_dt), limit),
            )
            rows = cursor.fetchall()
        return [
            {
                "frame_id": row["frame_id"],
                "timestamp": _dt_from_any(row["timestamp"]),
                "image_path": row["image_path"],
                "device_name": row["device_name"],
                "metadata": _safe_metadata_dict(row["metadata"]),
                "app_name": row["app_name"] or row["focused_app_name"] or "",
                "window_name": row["window_name"] or row["focused_window_name"] or "",
                "ocr_text": row["ocr_text"] or "",
                "ocr_confidence": row["ocr_confidence"] or 0.0,
            }
            for row in rows
        ]
    except Exception as e:
        logger.debug(f"Rewind span frame lookup failed: {e}")
        return []


def _ocr_snippets_for_rewind_span(
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
    app_name: Optional[str],
    terms: List[str],
    limit: int = 6,
) -> List[str]:
    if sqlite_storage is None or not start_dt or not end_dt:
        return []
    snippets: List[str] = []
    try:
        with sqlite_storage._connection() as conn:
            cursor = conn.cursor()
            where = ["COALESCE(sf.timestamp, f.timestamp, pf.timestamp) >= ?",
                     "COALESCE(sf.timestamp, f.timestamp, pf.timestamp) <= ?"]
            params: List[Any] = [_sql_dt(start_dt), _sql_dt(end_dt)]
            if app_name:
                where.append("(sf.app_name = ? OR f.app_name = ? OR f.focused_app_name = ?)")
                params.extend([app_name, app_name, app_name])
            if terms:
                term_clauses = []
                for term in terms[:4]:
                    term_clauses.append("o.text LIKE ? ESCAPE '\\'")
                    params.append(_like_pattern(term))
                where.append("(" + " OR ".join(term_clauses) + ")")
            params.append(limit)
            cursor.execute(
                f"""
                SELECT o.text
                FROM ocr_text o
                LEFT JOIN frames f ON o.frame_id = f.frame_id
                LEFT JOIN sub_frames sf ON o.sub_frame_id = sf.sub_frame_id
                LEFT JOIN frame_subframe_mapping fsm ON sf.sub_frame_id = fsm.sub_frame_id
                LEFT JOIN frames pf ON fsm.frame_id = pf.frame_id
                WHERE {" AND ".join(where)}
                ORDER BY COALESCE(sf.timestamp, f.timestamp, pf.timestamp) ASC
                LIMIT ?
                """,
                tuple(params),
            )
            snippets = [_clip_one_line(row["text"], 500) for row in cursor.fetchall() if row["text"]]
    except Exception as e:
        logger.debug(f"Rewind span OCR snippet lookup failed: {e}")
    return _dedupe_strings(snippets, limit=limit)


def _clip_rewind_session_window(
    session_start: Optional[datetime],
    session_end: Optional[datetime],
    hits: List[Dict[str, Any]],
) -> Tuple[Optional[datetime], Optional[datetime], bool]:
    if not session_start or not session_end:
        return session_start, session_end, False

    max_minutes = max(int(config.REWIND_SESSION_MAX_MINUTES or 45), 5)
    max_duration = timedelta(minutes=max_minutes)
    session_duration = session_end - session_start
    if session_duration <= max_duration:
        return session_start, session_end, False

    hit_times = [_dt_from_any(hit.get("timestamp")) for hit in hits]
    hit_times = [dt for dt in hit_times if dt is not None]
    padding = timedelta(minutes=max(int(config.REWIND_SESSION_PADDING_MINUTES or 5), 1))

    if hit_times:
        clipped_start = max(session_start, min(hit_times) - padding)
        clipped_end = min(session_end, max(hit_times) + padding)
        if clipped_end <= clipped_start:
            midpoint = hit_times[0]
            clipped_start = max(session_start, midpoint - max_duration / 2)
            clipped_end = min(session_end, clipped_start + max_duration)
        elif clipped_end - clipped_start > max_duration:
            midpoint = clipped_start + (clipped_end - clipped_start) / 2
            clipped_start = max(session_start, midpoint - max_duration / 2)
            clipped_end = min(session_end, clipped_start + max_duration)
    else:
        clipped_start = session_start
        clipped_end = min(session_end, session_start + max_duration)

    if clipped_end - clipped_start > max_duration:
        clipped_end = clipped_start + max_duration
    return clipped_start, clipped_end, True


def _session_key(session: Dict[str, Any]) -> Tuple[Any, str, str, str]:
    return (
        session.get("id"),
        session.get("app_name") or "",
        session.get("start_time") or "",
        session.get("end_time") or "",
    )


def _segment_from_activity_session(
    session: Dict[str, Any],
    hits: List[Dict[str, Any]],
    query_terms: List[str],
    retrieval_plan: Dict[str, Any],
) -> Dict[str, Any]:
    raw_start_dt = _dt_from_any(session.get("start_time"))
    raw_end_dt = _dt_from_any(session.get("end_time"))
    hits = sorted(hits, key=lambda h: float(h.get("_rewind_score") or 0.0), reverse=True)
    start_dt, end_dt, session_clipped = _clip_rewind_session_window(raw_start_dt, raw_end_dt, hits)
    clipped_session = dict(session)
    clipped_session["start_time"] = _ts_to_iso(start_dt)
    clipped_session["end_time"] = _ts_to_iso(end_dt)
    sub_frames = _sub_frames_for_activity_session(clipped_session, limit=config.REWIND_SESSION_FRAME_LIMIT)
    span_frames = _frames_for_rewind_span(start_dt, end_dt, limit=config.REWIND_SESSION_FRAME_LIMIT)

    representative = hits[0] if hits else None
    if representative is None and sub_frames:
        sf = sub_frames[0]
        representative = {
            "frame_id": sf.get("parent_frame_id") or sf.get("sub_frame_id"),
            "sub_frame_id": sf.get("sub_frame_id"),
            "timestamp": _dt_from_any(sf.get("timestamp")),
            "image_path": sf.get("image_path"),
            "app_name": sf.get("app_name"),
            "window_name": sf.get("window_name"),
            "metadata": {
                "sub_frame_id": sf.get("sub_frame_id"),
                "parent_frame_id": sf.get("parent_frame_id"),
            },
        }
    if representative is None and span_frames:
        representative = span_frames[0]

    snippet_parts = []
    for hit in hits[:4]:
        if hit.get("ocr_text"):
            snippet_parts.append(_clip_one_line(hit.get("ocr_text"), 500))
    snippet_parts.extend(
        _ocr_snippets_for_rewind_span(
            start_dt,
            end_dt,
            session.get("app_name"),
            query_terms,
            limit=6,
        )
    )
    if not snippet_parts:
        for frame in span_frames[:3]:
            if frame.get("ocr_text"):
                snippet_parts.append(_clip_one_line(frame.get("ocr_text"), 500))

    metadata = {
        "retrieval_mode": "activity_session",
        "session_id": session.get("id"),
        "session_status": session.get("session_status"),
        "cluster_id": session.get("cluster_id"),
        "session_frame_count": session.get("frame_count"),
        "raw_session_start_time": _ts_to_iso(raw_start_dt),
        "raw_session_end_time": _ts_to_iso(raw_end_dt),
        "session_clipped": session_clipped,
        "session_max_minutes": config.REWIND_SESSION_MAX_MINUTES,
        "hit_count": len(hits),
        "retrieval_sources": _dedupe_strings(
            [
                src
                for hit in hits
                for src in (hit.get("metadata", {}).get("retrieval_sources") or [])
            ],
            limit=16,
        ),
        "retrieval_queries": _dedupe_strings(
            [
                q
                for hit in hits
                for q in (hit.get("metadata", {}).get("retrieval_queries") or [])
            ],
            limit=16,
        ),
        "retrieval_plan": {
            "mode": retrieval_plan.get("mode"),
            "hops": retrieval_plan.get("hops"),
            "keyword_terms": retrieval_plan.get("keyword_terms"),
            "controller_used": retrieval_plan.get("controller_used"),
        },
    }
    rep_metadata = _safe_metadata_dict((representative or {}).get("metadata"))
    if rep_metadata.get("sub_frame_id") and not any(sf.get("sub_frame_id") == rep_metadata["sub_frame_id"] for sf in sub_frames):
        sub_frames.insert(
            0,
            {
                "sub_frame_id": rep_metadata["sub_frame_id"],
                "timestamp": _ts_to_iso((representative or {}).get("timestamp")),
                "app_name": (representative or {}).get("app_name") or session.get("app_name") or "",
                "window_name": (representative or {}).get("window_name") or "",
                "image_path": (representative or {}).get("image_path"),
            },
        )

    segment = {
        "segment_id": f"activity_session_{session.get('id') or uuid.uuid4().hex[:10]}",
        "frame_id": (representative or {}).get("frame_id"),
        "timestamp": _ts_to_iso((representative or {}).get("timestamp") or start_dt),
        "start_time": _ts_to_iso(start_dt),
        "end_time": _ts_to_iso(end_dt),
        "title": " · ".join(
            [p for p in [session.get("label"), session.get("app_name"), (representative or {}).get("window_name")] if p]
        ) or "Activity session",
        "app_name": session.get("app_name") or (representative or {}).get("app_name"),
        "window_name": (representative or {}).get("window_name"),
        "activity_label": session.get("label"),
        "image_path": (representative or {}).get("image_path"),
        "ocr_text": _clip_text("\n".join(_dedupe_strings(snippet_parts, limit=8))),
        "sub_frames": sub_frames[: config.REWIND_SESSION_FRAME_LIMIT],
        "metadata": metadata,
    }
    return _enrich_rewind_segment(segment)


def _segment_from_fallback_hits(
    hits: List[Dict[str, Any]],
    retrieval_plan: Dict[str, Any],
    explicit_start: Optional[datetime],
    explicit_end: Optional[datetime],
) -> Dict[str, Any]:
    hits = sorted(hits, key=lambda h: float(h.get("_rewind_score") or 0.0), reverse=True)
    representative = hits[0]
    hit_times = [_dt_from_any(hit.get("timestamp")) for hit in hits]
    hit_times = [dt for dt in hit_times if dt is not None]
    padding = timedelta(minutes=max(config.REWIND_SESSION_PADDING_MINUTES, 1))
    start_dt = (min(hit_times) - padding) if hit_times else _dt_from_any(representative.get("timestamp"))
    end_dt = (max(hit_times) + padding) if hit_times else _dt_from_any(representative.get("timestamp"))
    if explicit_start and start_dt:
        start_dt = max(start_dt, explicit_start)
    if explicit_end and end_dt:
        end_dt = min(end_dt, explicit_end)

    sub_frames: List[Dict[str, Any]] = []
    rep_metadata = _safe_metadata_dict(representative.get("metadata"))
    if representative.get("sub_frame_id") or rep_metadata.get("sub_frame_id"):
        sub_frames.append(
            {
                "sub_frame_id": representative.get("sub_frame_id") or rep_metadata.get("sub_frame_id"),
                "timestamp": _ts_to_iso(representative.get("timestamp")),
                "app_name": representative.get("app_name") or "",
                "window_name": representative.get("window_name") or "",
                "image_path": representative.get("image_path"),
            }
        )
    elif representative.get("frame_id"):
        sub_frames = _sub_frames_for_frame(representative.get("frame_id"))[: config.REWIND_SESSION_FRAME_LIMIT]

    snippet_parts = [
        _clip_one_line(hit.get("ocr_text"), 500)
        for hit in hits
        if hit.get("ocr_text")
    ]
    snippet_parts.extend(
        _ocr_snippets_for_rewind_span(
            start_dt,
            end_dt,
            representative.get("app_name"),
            retrieval_plan.get("keyword_terms") or [],
            limit=4,
        )
    )

    segment = {
        "segment_id": f"rewind_span_{uuid.uuid4().hex[:10]}",
        "frame_id": representative.get("frame_id"),
        "timestamp": _ts_to_iso(representative.get("timestamp")),
        "start_time": _ts_to_iso(start_dt),
        "end_time": _ts_to_iso(end_dt),
        "title": " · ".join(
            [p for p in [representative.get("app_name"), representative.get("window_name")] if p]
        ) or "Timeline span",
        "app_name": representative.get("app_name"),
        "window_name": representative.get("window_name"),
        "activity_label": representative.get("activity_label"),
        "image_path": representative.get("image_path"),
        "ocr_text": _clip_text("\n".join(_dedupe_strings(snippet_parts, limit=6))),
        "sub_frames": sub_frames,
        "metadata": {
            "retrieval_mode": "temporal_span",
            "hit_count": len(hits),
            "retrieval_sources": _dedupe_strings(
                [
                    src
                    for hit in hits
                    for src in (hit.get("metadata", {}).get("retrieval_sources") or [])
                ],
                limit=16,
            ),
            "retrieval_queries": _dedupe_strings(
                [
                    q
                    for hit in hits
                    for q in (hit.get("metadata", {}).get("retrieval_queries") or [])
                ],
                limit=16,
            ),
            "retrieval_plan": {
                "mode": retrieval_plan.get("mode"),
                "hops": retrieval_plan.get("hops"),
                "keyword_terms": retrieval_plan.get("keyword_terms"),
                "controller_used": retrieval_plan.get("controller_used"),
            },
        },
    }
    return _enrich_rewind_segment(segment)


def _group_rewind_hits_into_segments(
    hits: List[Dict[str, Any]],
    activity_sessions: List[Dict[str, Any]],
    retrieval_plan: Dict[str, Any],
    limit: int,
) -> List[Dict[str, Any]]:
    grouped_sessions: Dict[Tuple[Any, str, str, str], Dict[str, Any]] = {}
    fallback_groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}

    for session in activity_sessions:
        grouped_sessions.setdefault(_session_key(session), {"session": session, "hits": []})

    for hit in sorted(hits, key=lambda h: float(h.get("_rewind_score") or 0.0), reverse=True):
        session = _activity_session_for_hit(hit)
        if session:
            grouped_sessions.setdefault(_session_key(session), {"session": session, "hits": []})["hits"].append(hit)
            continue
        ts = _dt_from_any(hit.get("timestamp"))
        bucket_seconds = max(config.REWIND_SESSION_PADDING_MINUTES * 120, 600)
        bucket = int(ts.timestamp() // bucket_seconds) if ts else 0
        fallback_groups.setdefault((hit.get("app_name") or "", bucket), []).append(hit)

    segments: List[Dict[str, Any]] = []
    for group in grouped_sessions.values():
        segments.append(
            _segment_from_activity_session(
                group["session"],
                group["hits"],
                retrieval_plan.get("keyword_terms") or [],
                retrieval_plan,
            )
        )

    for group_hits in fallback_groups.values():
        segments.append(
            _segment_from_fallback_hits(
                group_hits,
                retrieval_plan,
                retrieval_plan.get("start_time"),
                retrieval_plan.get("end_time"),
            )
        )

    def score(segment: Dict[str, Any]) -> Tuple[float, str]:
        metadata = segment.get("metadata") or {}
        hit_count = float(metadata.get("hit_count") or 0)
        session_bonus = 10.0 if metadata.get("retrieval_mode") == "activity_session" else 0.0
        return session_bonus + hit_count, segment.get("start_time") or ""

    segments.sort(key=score, reverse=True)
    return segments[:limit]


def _rag_hit_debug_info(
    hit: Dict[str, Any],
    mapped_session: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = _safe_metadata_dict(hit.get("metadata"))
    return {
        "frame_id": hit.get("frame_id"),
        "sub_frame_id": hit.get("sub_frame_id") or metadata.get("sub_frame_id"),
        "timestamp": _ts_to_iso(hit.get("timestamp")),
        "app": hit.get("app_name"),
        "window": hit.get("window_name"),
        "rag_score": round(float(hit.get("_local_rag_score") or 0.0), 4),
        "sources": metadata.get("retrieval_sources") or [],
        "queries": metadata.get("retrieval_queries") or [],
        "rank": metadata.get("retrieval_rank"),
        "details": (metadata.get("rag_details") or [])[:4],
        "mapped_session_id": (mapped_session or {}).get("id"),
        "mapped_label": (mapped_session or {}).get("label"),
        "mapped_app": (mapped_session or {}).get("app_name"),
    }


def _aggregate_rewind_rag_score(hits: List[Dict[str, Any]]) -> float:
    if not hits:
        return 0.0
    scores = sorted(
        [float(hit.get("_local_rag_score") or 0.0) for hit in hits],
        reverse=True,
    )
    if not scores:
        return 0.0
    return min(1.0, scores[0] + sum(scores[1:8]) * 0.15)


def _run_rewind_local_rag(plan: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    start_dt = plan.get("start_time")
    end_dt = plan.get("end_time")
    candidate_limit = min(max(limit * 6, 36), 120)
    hits_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}

    if encoder is not None and vector_storage is not None:
        for q_idx, dense_query in enumerate(plan.get("dense_queries") or []):
            try:
                emb = encoder.encode_text(dense_query)
                for rank, frame in enumerate(
                    vector_storage.search(
                        emb,
                        top_k=candidate_limit,
                        start_time=start_dt,
                        end_time=end_dt,
                        related_apps=plan.get("related_apps"),
                        unrelated_apps=plan.get("unrelated_apps"),
                        window_filters=plan.get("window_filters"),
                    ),
                    start=1 + q_idx * candidate_limit,
                ):
                    _add_rewind_hit(hits_by_key, frame, "dense_image", dense_query, rank)
                for rank, frame in enumerate(
                    vector_storage.search_ocr(
                        emb,
                        top_k=max(candidate_limit // 2, limit),
                        start_time=start_dt,
                        end_time=end_dt,
                        related_apps=plan.get("related_apps"),
                        unrelated_apps=plan.get("unrelated_apps"),
                        window_filters=plan.get("window_filters"),
                    ),
                    start=1 + q_idx * candidate_limit,
                ):
                    _add_rewind_hit(hits_by_key, frame, "dense_ocr", dense_query, rank)
            except Exception as e:
                logger.debug(f"Rewind dense hop failed for '{dense_query}': {e}")

    if sqlite_storage is not None:
        for q_idx, sparse_query in enumerate(plan.get("sparse_queries") or []):
            try:
                for rank, frame in enumerate(
                    sqlite_storage.search_by_text(sparse_query, limit=max(candidate_limit // 2, limit)),
                    start=1 + q_idx * candidate_limit,
                ):
                    ts = _dt_from_any(frame.get("timestamp"))
                    if start_dt and ts and ts < start_dt:
                        continue
                    if end_dt and ts and ts > end_dt:
                        continue
                    sparse_app = frame.get("app_name") or frame.get("focused_app_name")
                    sparse_window = frame.get("window_name") or frame.get("focused_window_name")
                    if (sparse_app or sparse_window) and not _app_window_allowed(
                        sparse_app,
                        sparse_window,
                        plan.get("related_apps"),
                        plan.get("unrelated_apps"),
                        plan.get("window_filters"),
                    ):
                        continue
                    _add_rewind_hit(hits_by_key, frame, "sparse_fts", sparse_query, rank)
            except Exception as e:
                logger.debug(f"Rewind sparse FTS hop failed for '{sparse_query}': {e}")

        for term_idx, term in enumerate(plan.get("keyword_terms") or []):
            for rank, frame in enumerate(
                _rewind_keyword_like_search(
                    term,
                    limit=max(candidate_limit // 2, limit),
                    start_dt=start_dt,
                    end_dt=end_dt,
                    related_apps=plan.get("related_apps"),
                    unrelated_apps=plan.get("unrelated_apps"),
                    window_filters=plan.get("window_filters"),
                ),
                start=1 + term_idx * candidate_limit,
            ):
                _add_rewind_hit(hits_by_key, frame, "keyword_like", term, rank)
            for rank, frame in enumerate(
                _rewind_window_search(
                    term,
                    limit=max(candidate_limit // 3, limit),
                    start_dt=start_dt,
                    end_dt=end_dt,
                ),
                start=1 + term_idx * candidate_limit,
            ):
                if not _app_window_allowed(
                    frame.get("app_name"),
                    frame.get("window_name"),
                    plan.get("related_apps"),
                    plan.get("unrelated_apps"),
                    plan.get("window_filters"),
                ):
                    continue
                _add_rewind_hit(hits_by_key, frame, "window_title", term, rank)

    hits = sorted(
        hits_by_key.values(),
        key=lambda hit: (
            float(hit.get("_local_rag_score") or 0.0),
            float(hit.get("_rewind_score") or 0.0),
        ),
        reverse=True,
    )
    logger.info(
        "Rewind agentic top local RAG hits: %s",
        [_rag_hit_debug_info(hit) for hit in hits[:15]],
    )
    return hits


def _attach_rewind_fusion_metadata(
    segment: Dict[str, Any],
    *,
    final_score: float,
    llm_score: float,
    rag_score: float,
    intersection: bool,
    label_result: Optional[Dict[str, Any]],
    hits: List[Dict[str, Any]],
    mapped_session: Optional[Dict[str, Any]],
    fusion_mode: str,
) -> Dict[str, Any]:
    metadata = segment.setdefault("metadata", {})
    rag_top_hits = [_rag_hit_debug_info(hit, mapped_session) for hit in hits[:8]]
    metadata.update(
        {
            "fusion_mode": fusion_mode,
            "fusion_score": round(final_score, 4),
            "llm_label_score": round(llm_score, 4),
            "llm_label_reason": (label_result or {}).get("llm_label_reason"),
            "llm_label_score_source": (label_result or {}).get("llm_label_score_source"),
            "local_rag_score": round(rag_score, 4),
            "label_rag_intersection": intersection,
            "rag_hit_count": len(hits),
            "rag_top_hits": rag_top_hits,
            "rag_reverse_mapped_session": {
                "id": (mapped_session or {}).get("id"),
                "app_name": (mapped_session or {}).get("app_name"),
                "label": (mapped_session or {}).get("label"),
                "start_time": (mapped_session or {}).get("start_time"),
                "end_time": (mapped_session or {}).get("end_time"),
            }
            if mapped_session
            else None,
        }
    )
    return segment


def _fuse_rewind_label_and_rag(
    query: str,
    plan: Dict[str, Any],
    label_results: List[Dict[str, Any]],
    rag_hits: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    grouped_sessions: Dict[Tuple[Any, str, str, str], Dict[str, Any]] = {}
    fallback_groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}

    for label_result in label_results:
        session = label_result.get("session") or {}
        if not session:
            continue
        score = float(label_result.get("llm_label_score") or 0.0)
        if score < 0.12:
            continue
        grouped_sessions.setdefault(
            _session_key(session),
            {"session": session, "label_result": label_result, "hits": []},
        )["label_result"] = label_result

    for hit in rag_hits:
        session = _activity_session_for_hit(hit)
        if session:
            key = _session_key(session)
            group = grouped_sessions.setdefault(
                key,
                {"session": session, "label_result": None, "hits": []},
            )
            group["hits"].append(hit)
            continue
        ts = _dt_from_any(hit.get("timestamp"))
        bucket_seconds = max(config.REWIND_SESSION_PADDING_MINUTES * 120, 600)
        bucket = int(ts.timestamp() // bucket_seconds) if ts else 0
        fallback_groups.setdefault((hit.get("app_name") or "", bucket), []).append(hit)

    intersections = []
    scored_segments: List[Tuple[float, Dict[str, Any]]] = []
    for group in grouped_sessions.values():
        session = group["session"]
        hits = sorted(
            group["hits"],
            key=lambda hit: float(hit.get("_local_rag_score") or 0.0),
            reverse=True,
        )
        label_result = group.get("label_result")
        llm_score = float((label_result or {}).get("llm_label_score") or 0.0)
        rag_score = _aggregate_rewind_rag_score(hits)
        intersection = bool(label_result and hits)
        final_score = llm_score * 0.45 + rag_score * 0.45 + (0.10 if intersection else 0.0)
        if not hits and llm_score < 0.12:
            continue
        segment = _segment_from_activity_session(
            session,
            hits,
            plan.get("keyword_terms") or [],
            plan,
        )
        _attach_rewind_fusion_metadata(
            segment,
            final_score=final_score,
            llm_score=llm_score,
            rag_score=rag_score,
            intersection=intersection,
            label_result=label_result,
            hits=hits,
            mapped_session=session,
            fusion_mode="label_rag_fusion" if intersection else ("label_only_session" if label_result else "rag_session_reverse_map"),
        )
        if intersection:
            intersections.append(
                {
                    "session": session.get("id"),
                    "label": session.get("label"),
                    "app": session.get("app_name"),
                    "llm_score": round(llm_score, 3),
                    "rag_score": round(rag_score, 3),
                    "top_rag": [_rag_hit_debug_info(hit, session) for hit in hits[:3]],
                }
            )
        scored_segments.append((final_score, segment))

    for group_hits in fallback_groups.values():
        hits = sorted(
            group_hits,
            key=lambda hit: float(hit.get("_local_rag_score") or 0.0),
            reverse=True,
        )
        rag_score = _aggregate_rewind_rag_score(hits)
        final_score = rag_score * 0.45
        segment = _segment_from_fallback_hits(
            hits,
            plan,
            plan.get("start_time"),
            plan.get("end_time"),
        )
        _attach_rewind_fusion_metadata(
            segment,
            final_score=final_score,
            llm_score=0.0,
            rag_score=rag_score,
            intersection=False,
            label_result=None,
            hits=hits,
            mapped_session=None,
            fusion_mode="rag_only_temporal_span",
        )
        scored_segments.append((final_score, segment))

    logger.info("Rewind label/RAG intersections: %s", intersections[:20])
    scored_segments.sort(
        key=lambda item: (
            item[0],
            item[1].get("start_time") or item[1].get("timestamp") or "",
        ),
        reverse=True,
    )
    segments = [segment for _, segment in scored_segments[:limit]]
    for idx, segment in enumerate(segments, start=1):
        metadata = segment.get("metadata") if isinstance(segment.get("metadata"), dict) else {}
        logger.info(
            "Rewind fused timeline[%02d]: segment=%s mode=%s label=%s app=%s window=%s "
            "time=%s->%s final=%.3f llm=%.3f rag=%.3f intersect=%s reason=%s rag_top=%s",
            idx,
            segment.get("segment_id"),
            metadata.get("fusion_mode"),
            segment.get("activity_label"),
            segment.get("app_name"),
            segment.get("window_name"),
            segment.get("start_time"),
            segment.get("end_time"),
            float(metadata.get("fusion_score") or 0.0),
            float(metadata.get("llm_label_score") or 0.0),
            float(metadata.get("local_rag_score") or 0.0),
            metadata.get("label_rag_intersection"),
            metadata.get("llm_label_reason"),
            metadata.get("rag_top_hits") or [],
        )
    return segments


def _lookup_frame_record(frame_id: str) -> Optional[Dict[str, Any]]:
    if sqlite_storage is None or not frame_id:
        return None
    try:
        with sqlite_storage._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    f.frame_id,
                    f.timestamp,
                    f.image_path,
                    f.device_name,
                    f.metadata,
                    f.app_name,
                    f.window_name,
                    f.focused_app_name,
                    f.focused_window_name,
                    o.text as ocr_text,
                    o.confidence as ocr_confidence
                FROM frames f
                LEFT JOIN ocr_text o ON f.frame_id = o.frame_id
                WHERE f.frame_id = ?
                LIMIT 1
                """,
                (frame_id,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return {
            "frame_id": row["frame_id"],
            "timestamp": datetime.fromisoformat(row["timestamp"]),
            "image_path": row["image_path"],
            "device_name": row["device_name"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "app_name": row["app_name"] or row["focused_app_name"] or "",
            "window_name": row["window_name"] or row["focused_window_name"] or "",
            "ocr_text": row["ocr_text"] or "",
            "ocr_confidence": row["ocr_confidence"] or 0.0,
        }
    except Exception as e:
        logger.debug(f"Task Memory frame lookup failed for {frame_id}: {e}")
        return None


def _sub_frames_for_frame(frame_id: Optional[str]) -> List[Dict[str, Any]]:
    if sqlite_storage is None or not frame_id:
        return []
    sub_frames = []
    try:
        for sf in sqlite_storage.get_sub_frames_for_frame(frame_id):
            sub_frames.append(
                {
                    "sub_frame_id": sf.get("sub_frame_id", ""),
                    "timestamp": _ts_to_iso(sf.get("timestamp")),
                    "app_name": sf.get("app_name", "") or "",
                    "window_name": sf.get("window_name", "") or "",
                    "image_path": _resolve_sub_frame_image_path(sf),
                }
            )
    except Exception as e:
        logger.debug(f"Task Memory sub-frame lookup failed for {frame_id}: {e}")
    return sub_frames


def _normalize_rewind_segment(segment: Any) -> Dict[str, Any]:
    if isinstance(segment, BaseModel):
        data = segment.dict()
    elif isinstance(segment, dict):
        data = dict(segment)
    else:
        data = {}

    sub_frames = []
    for sf in data.get("sub_frames") or []:
        if isinstance(sf, BaseModel):
            sf_data = sf.dict()
        elif isinstance(sf, dict):
            sf_data = dict(sf)
        else:
            continue
        sub_frames.append(
            {
                "sub_frame_id": str(sf_data.get("sub_frame_id") or ""),
                "timestamp": _ts_to_iso(sf_data.get("timestamp")),
                "app_name": str(sf_data.get("app_name") or ""),
                "window_name": str(sf_data.get("window_name") or ""),
                "image_path": sf_data.get("image_path"),
            }
        )

    timestamp = _ts_to_iso(data.get("timestamp") or data.get("start_time") or data.get("end_time"))
    start_time = _ts_to_iso(data.get("start_time") or timestamp)
    end_time = _ts_to_iso(data.get("end_time") or timestamp)
    frame_id = data.get("frame_id")
    segment_id = data.get("segment_id") or frame_id or f"segment_{uuid.uuid4().hex[:10]}"

    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    normalized = {
        "segment_id": str(segment_id),
        "frame_id": str(frame_id) if frame_id else None,
        "timestamp": timestamp or None,
        "start_time": start_time or None,
        "end_time": end_time or None,
        "title": data.get("title"),
        "app_name": data.get("app_name") or None,
        "window_name": data.get("window_name") or None,
        "activity_label": data.get("activity_label") or None,
        "image_path": data.get("image_path"),
        "ocr_text": _clip_text(data.get("ocr_text")),
        "sub_frames": sub_frames,
        "metadata": metadata,
    }
    return normalized


def _enrich_rewind_segment(segment: Any) -> Dict[str, Any]:
    normalized = _normalize_rewind_segment(segment)
    frame_record = _lookup_frame_record(normalized.get("frame_id") or "")
    if frame_record:
        normalized["timestamp"] = normalized.get("timestamp") or _ts_to_iso(frame_record.get("timestamp"))
        normalized["start_time"] = normalized.get("start_time") or normalized["timestamp"]
        normalized["end_time"] = normalized.get("end_time") or normalized["timestamp"]
        normalized["image_path"] = normalized.get("image_path") or frame_record.get("image_path")
        normalized["app_name"] = normalized.get("app_name") or frame_record.get("app_name") or None
        normalized["window_name"] = normalized.get("window_name") or frame_record.get("window_name") or None
        normalized["ocr_text"] = normalized.get("ocr_text") or _clip_text(frame_record.get("ocr_text"))
        normalized["metadata"] = normalized.get("metadata") or frame_record.get("metadata") or {}

    if not normalized.get("sub_frames"):
        normalized["sub_frames"] = _sub_frames_for_frame(normalized.get("frame_id"))

    label_ids = [normalized.get("frame_id") or ""]
    label_ids.extend(sf.get("sub_frame_id", "") for sf in normalized.get("sub_frames") or [])
    normalized["activity_label"] = normalized.get("activity_label") or _activity_label_for_ids(label_ids)

    if not normalized.get("title"):
        title_parts = [
            normalized.get("activity_label"),
            normalized.get("app_name"),
            normalized.get("window_name"),
        ]
        normalized["title"] = " · ".join([str(p) for p in title_parts if p]) or (
            normalized.get("timestamp") or "Timeline evidence"
        )

    return normalized


def _frame_to_rewind_segment(frame: Dict[str, Any]) -> Dict[str, Any]:
    metadata = frame.get("metadata") if isinstance(frame.get("metadata"), dict) else {}
    sub_frame_id = frame.get("sub_frame_id") or metadata.get("sub_frame_id")
    sub_frames = []
    if sub_frame_id:
        sub_frames.append(
            {
                "sub_frame_id": sub_frame_id,
                "timestamp": _ts_to_iso(frame.get("timestamp")),
                "app_name": frame.get("app_name") or "",
                "window_name": frame.get("window_name") or "",
                "image_path": frame.get("image_path"),
            }
        )
    segment = {
        "segment_id": frame.get("frame_id") or f"segment_{uuid.uuid4().hex[:10]}",
        "frame_id": frame.get("frame_id"),
        "timestamp": _ts_to_iso(frame.get("timestamp")),
        "start_time": _ts_to_iso(frame.get("timestamp")),
        "end_time": _ts_to_iso(frame.get("timestamp")),
        "app_name": frame.get("app_name") or frame.get("focused_app_name"),
        "window_name": frame.get("window_name") or frame.get("focused_window_name"),
        "image_path": frame.get("image_path"),
        "ocr_text": _clip_text(frame.get("ocr_text")),
        "sub_frames": sub_frames,
        "metadata": metadata,
    }
    return _enrich_rewind_segment(segment)


def _segment_sort_key(segment: Dict[str, Any]) -> str:
    return str(segment.get("start_time") or segment.get("timestamp") or "")


def _merge_rewind_evidence_refs(
    segments: List[Dict[str, Any]],
    explicit_refs: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    seen = set()

    def add_ref(ref: Dict[str, Any]) -> None:
        key = (
            ref.get("frame_id") or "",
            ref.get("sub_frame_id") or "",
            ref.get("timestamp") or "",
            ref.get("image_path") or "",
        )
        if key in seen:
            return
        seen.add(key)
        refs.append(
            {
                "frame_id": ref.get("frame_id"),
                "sub_frame_id": ref.get("sub_frame_id"),
                "timestamp": ref.get("timestamp"),
                "image_path": ref.get("image_path"),
                "app_name": ref.get("app_name"),
                "window_name": ref.get("window_name"),
                "activity_label": ref.get("activity_label"),
                "ocr_snippet": _clip_one_line(ref.get("ocr_snippet") or ref.get("ocr_text"), 600),
            }
        )

    for segment in segments:
        add_ref(
            {
                "frame_id": segment.get("frame_id"),
                "timestamp": segment.get("timestamp") or segment.get("start_time"),
                "image_path": segment.get("image_path"),
                "app_name": segment.get("app_name"),
                "window_name": segment.get("window_name"),
                "activity_label": segment.get("activity_label"),
                "ocr_text": segment.get("ocr_text"),
            }
        )
        for sf in segment.get("sub_frames") or []:
            add_ref(
                {
                    "frame_id": segment.get("frame_id"),
                    "sub_frame_id": sf.get("sub_frame_id"),
                    "timestamp": sf.get("timestamp") or segment.get("timestamp"),
                    "image_path": sf.get("image_path"),
                    "app_name": sf.get("app_name"),
                    "window_name": sf.get("window_name"),
                    "activity_label": segment.get("activity_label"),
                    "ocr_text": segment.get("ocr_text"),
                }
            )
        metadata = segment.get("metadata") if isinstance(segment.get("metadata"), dict) else {}
        for frame_ref in metadata.get("timeline_frame_refs") or []:
            if not isinstance(frame_ref, dict):
                continue
            add_ref(
                {
                    "frame_id": frame_ref.get("frame_id") or segment.get("frame_id"),
                    "timestamp": frame_ref.get("timestamp"),
                    "image_path": frame_ref.get("image_path"),
                    "app_name": segment.get("app_name"),
                    "window_name": segment.get("window_name"),
                    "activity_label": segment.get("activity_label"),
                    "ocr_text": segment.get("ocr_text"),
                }
            )

    for ref in explicit_refs or []:
        ref_data = ref.dict() if isinstance(ref, BaseModel) else dict(ref)
        add_ref(ref_data)

    return refs


def _format_segment_time(segment: Dict[str, Any]) -> str:
    start = segment.get("start_time") or segment.get("timestamp") or ""
    end = segment.get("end_time") or segment.get("timestamp") or ""
    if not end or end == start:
        return start
    return f"{start} -> {end}"


def _format_timeline_evidence(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, segment in enumerate(sorted(segments, key=_segment_sort_key), start=1):
        lines.append(f"{idx}. Time: {_format_segment_time(segment)}")
        if segment.get("title"):
            lines.append(f"   Title: {segment['title']}")
        if segment.get("activity_label"):
            lines.append(f"   Activity: {segment['activity_label']}")
        app_window = " / ".join(
            [p for p in [segment.get("app_name"), segment.get("window_name")] if p]
        )
        if app_window:
            lines.append(f"   Window: {app_window}")
        if segment.get("frame_id"):
            lines.append(f"   Frame: {segment['frame_id']}")
        if segment.get("image_path"):
            lines.append(f"   Representative frame ref: {segment['image_path']}")
        metadata = segment.get("metadata") if isinstance(segment.get("metadata"), dict) else {}
        frame_refs = [ref for ref in metadata.get("timeline_frame_refs") or [] if isinstance(ref, dict)]
        if frame_refs:
            ref_times = [str(ref.get("timestamp") or ref.get("frame_id") or "") for ref in frame_refs[:6]]
            more = f"; +{len(frame_refs) - 6} more" if len(frame_refs) > 6 else ""
            lines.append(f"   Timeline frames: {'; '.join(ref_times)}{more}")
        ocr = _clip_one_line(segment.get("ocr_text"), 700)
        if ocr:
            lines.append(f"   OCR snippet: {ocr}")
        sub_frames = segment.get("sub_frames") or []
        if sub_frames:
            sub_parts = []
            for sf in sub_frames[:6]:
                label = " / ".join([p for p in [sf.get("app_name"), sf.get("window_name")] if p])
                sub_parts.append(f"{sf.get('sub_frame_id')} ({label or 'window'})")
            more = f"; +{len(sub_frames) - 6} more" if len(sub_frames) > 6 else ""
            lines.append(f"   Window evidence: {'; '.join(sub_parts)}{more}")
    return "\n".join(lines) if lines else "(No timeline evidence selected.)"


def _format_evidence_refs(refs: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, ref in enumerate(refs[:40], start=1):
        target = ref.get("sub_frame_id") or ref.get("frame_id") or ref.get("image_path") or "evidence"
        lines.append(f"{idx}. Ref: {target}")
        if ref.get("timestamp"):
            lines.append(f"   Time: {ref['timestamp']}")
        app_window = " / ".join([p for p in [ref.get("app_name"), ref.get("window_name")] if p])
        if app_window:
            lines.append(f"   Window: {app_window}")
        if ref.get("activity_label"):
            lines.append(f"   Activity: {ref['activity_label']}")
        if ref.get("image_path"):
            lines.append(f"   Frame ref: {ref['image_path']}")
        if ref.get("ocr_snippet"):
            lines.append(f"   OCR snippet: {_clip_one_line(ref['ocr_snippet'], 500)}")
    return "\n".join(lines) if lines else "(No evidence refs.)"


def _fallback_task_memory_markdown(
    title: str,
    source_query: str,
    segments: List[Dict[str, Any]],
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Source Query",
        source_query or "(empty)",
        "",
        "## Timeline Evidence",
    ]
    for segment in sorted(segments, key=_segment_sort_key):
        ocr = _clip_one_line(segment.get("ocr_text"), 260)
        label = segment.get("activity_label") or segment.get("title") or "activity"
        app_window = " / ".join(
            [p for p in [segment.get("app_name"), segment.get("window_name")] if p]
        )
        suffix = f" - {app_window}" if app_window else ""
        lines.append(f"- `{_format_segment_time(segment)}` {label}{suffix}")
        if ocr:
            lines.append(f"  OCR: {ocr}")
    lines.extend(
        [
            "",
            "## Working Context",
            "The selected evidence above is the persisted context for continuing this task.",
            "",
            "## Open Questions",
            "- What changed between the selected evidence segments?",
            "- Which file, window, or browser tab should be inspected next?",
            "- What is the next concrete action?",
        ]
    )
    return "\n".join(lines)


def _derive_task_memory_title(source_query: str, segments: List[Dict[str, Any]]) -> str:
    query = _clip_one_line(source_query, 80)
    if query:
        return query
    for segment in segments:
        if segment.get("title"):
            return _clip_one_line(segment["title"], 80)
    return "Untitled Task Memory"


def _load_rewind_images(segments: List[Dict[str, Any]], limit: int = 8) -> tuple[List[Any], List[Any]]:
    if limit <= 0:
        return [], []
    images: List[Any] = []
    timestamps: List[Any] = []
    seen_paths = set()
    for segment in sorted(segments, key=_segment_sort_key):
        candidates = [(segment.get("image_path"), segment.get("timestamp") or segment.get("start_time"))]
        metadata = segment.get("metadata") if isinstance(segment.get("metadata"), dict) else {}
        for ref in metadata.get("timeline_frame_refs") or []:
            if not isinstance(ref, dict):
                continue
            candidates.append((ref.get("image_path"), ref.get("timestamp")))
        for sf in segment.get("sub_frames") or []:
            candidates.append((sf.get("image_path"), sf.get("timestamp") or segment.get("timestamp")))
        for path, ts in candidates:
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            image = _load_image_from_path(path)
            if image is None:
                continue
            images.append(image)
            timestamps.append(_parse_optional_dt(ts) or ts)
            if len(images) >= limit:
                return images, timestamps
    return images, timestamps


def _call_rewind_ai(
    prompt: str,
    system_prompt: str,
    segments: List[Dict[str, Any]],
    image_limit: int = 8,
    timeout_seconds: Optional[float] = None,
) -> str:
    ai = vlm or ApiVLM()
    images, timestamps = _load_rewind_images(segments, limit=image_limit)
    if images:
        return ai._call_vlm(
            prompt,
            images,
            num_images=len(images),
            image_timestamps=timestamps,
            system_prompt=system_prompt,
            timeout_seconds=timeout_seconds,
        )
    return ai._call_vlm_text_only(
        prompt,
        system_prompt=system_prompt,
        timeout_seconds=timeout_seconds,
    )


def _search_rewind_segments_legacy(
    query: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    top_k: int = 12,
) -> List[Dict[str, Any]]:
    limit = min(max(int(top_k or 12), 1), 30)
    start_dt = _parse_optional_dt(start_time)
    end_dt = _parse_optional_dt(end_time)
    results: List[Dict[str, Any]] = []
    seen = set()

    def add_frames(frames: List[Dict[str, Any]]) -> None:
        for frame in frames:
            fid = frame.get("frame_id")
            if not fid or fid in seen:
                continue
            ts = frame.get("timestamp")
            ts_dt = ts if isinstance(ts, datetime) else _parse_optional_dt(_ts_to_iso(ts))
            if start_dt and ts_dt and ts_dt < start_dt:
                continue
            if end_dt and ts_dt and ts_dt > end_dt:
                continue
            seen.add(fid)
            results.append(frame)
            if len(results) >= limit:
                return

    if query.strip() and encoder is not None and vector_storage is not None:
        try:
            emb = encoder.encode_text(query.strip())
            add_frames(
                vector_storage.search(
                    emb,
                    top_k=limit,
                    start_time=start_dt,
                    end_time=end_dt,
                )
            )
        except Exception as e:
            logger.debug(f"Rewind dense evidence search failed: {e}")

    if query.strip() and sqlite_storage is not None and len(results) < limit:
        try:
            add_frames(sqlite_storage.search_by_text(query.strip(), limit=limit))
        except Exception as e:
            logger.debug(f"Rewind sparse evidence search failed: {e}")

    if sqlite_storage is not None and len(results) < limit and (start_dt or end_dt):
        try:
            start = start_dt or (end_dt - timedelta(hours=1))
            end = end_dt or (start_dt + timedelta(hours=1))
            add_frames(
                sqlite_storage.get_frames_in_timerange(
                    start_time=start,
                    end_time=end,
                    limit=limit,
                    only_full_screen=True,
                )
            )
        except Exception as e:
            logger.debug(f"Rewind time-range evidence search failed: {e}")

    if sqlite_storage is not None and len(results) < limit:
        try:
            add_frames(sqlite_storage.get_recent_frames(limit=limit))
        except Exception as e:
            logger.debug(f"Rewind recent evidence fallback failed: {e}")

    return [_frame_to_rewind_segment(frame) for frame in results[:limit]]


def _search_rewind_segments(
    query: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    top_k: int = 12,
) -> List[Dict[str, Any]]:
    limit = min(max(int(top_k or 12), 1), 30)
    if not config.REWIND_ENABLE_AGENTIC_SEARCH:
        return _search_rewind_segments_legacy(query, start_time, end_time, top_k)

    explicit_start = _parse_optional_dt(start_time)
    explicit_end = _parse_optional_dt(end_time)
    plan = _build_rewind_retrieval_plan(query, explicit_start, explicit_end)
    start_dt = plan.get("start_time")
    end_dt = plan.get("end_time")
    candidate_limit = min(max(limit * 6, 36), 120)

    logger.info(
        "Rewind agentic search plan: "
        f"dense={plan.get('dense_queries')} sparse={plan.get('sparse_queries')} "
        f"terms={plan.get('keyword_terms')} time={start_dt}->{end_dt} "
        f"include_apps={plan.get('related_apps')} exclude_apps={plan.get('unrelated_apps')} "
        f"window_filters={plan.get('window_filters')}"
    )

    candidate_sessions = _candidate_activity_sessions_for_plan(plan, limit=candidate_limit)
    label_results: List[Dict[str, Any]] = []
    rag_hits: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        label_future = (
            executor.submit(
                _score_rewind_label_candidates,
                query,
                plan,
                candidate_sessions,
                candidate_limit,
            )
            if candidate_sessions
            else None
        )
        rag_future = executor.submit(_run_rewind_local_rag, plan, limit)
        try:
            rag_hits = rag_future.result()
        except Exception as e:
            logger.warning(f"Rewind local RAG loop failed: {e}")
        if label_future is not None:
            try:
                label_results = label_future.result()
            except Exception as e:
                logger.warning(f"Rewind label scoring loop failed: {e}")

    if rag_hits or label_results:
        segments = _fuse_rewind_label_and_rag(
            query,
            plan,
            label_results,
            rag_hits,
            limit=limit,
        )
        if segments:
            return segments

    if sqlite_storage is not None and (start_dt or end_dt):
        try:
            start = start_dt or (end_dt - timedelta(hours=1))
            end = end_dt or (start_dt + timedelta(hours=1))
            fallback_frames = _frames_for_rewind_span(start, end, limit=limit)
            fallback_hits: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for rank, frame in enumerate(fallback_frames, start=1):
                _add_rewind_hit(fallback_hits, frame, "time_range_fallback", query, rank)
            segments = _group_rewind_hits_into_segments(
                list(fallback_hits.values()),
                [],
                plan,
                limit=limit,
            )
            if segments:
                return segments
        except Exception as e:
            logger.debug(f"Rewind agentic time fallback failed: {e}")

    return _search_rewind_segments_legacy(query, start_time, end_time, top_k)


def _generate_task_memory_markdown(
    title: str,
    source_query: str,
    segments: List[Dict[str, Any]],
) -> str:
    evidence = _format_timeline_evidence(segments)
    system_prompt = (
        "你是 VisualMem 的 Task Memory 生成器。你必须基于选中的 timeline evidence "
        "生成可持久化的工作上下文，帮助用户继续任务。回答使用 Markdown，中文为主，"
        "必须保留关键时间段、窗口/文件线索、已完成动作、未解决问题和下一步。"
    )
    prompt = f"""Source Query:
{source_query}

Timeline Evidence:
{evidence}

Generate a concise Task Memory Markdown document with these sections:
1. Task Summary
2. Evidence Timeline
3. Current State
4. Open Questions
5. Recommended Next Steps"""

    try:
        markdown = _call_rewind_ai(prompt, system_prompt, segments, image_limit=6)
        if markdown and not markdown.startswith(("API调用失败", "错误:")):
            return markdown
        logger.warning(f"Task Memory generation fell back after model response: {markdown[:120] if markdown else ''}")
    except Exception as e:
        logger.warning(f"Task Memory generation fallback: {e}")
    return _fallback_task_memory_markdown(title, source_query, segments)


@app.post("/api/rewind/search_segments", response_model=RewindSearchResponse)
def rewind_search_segments(req: RewindSearchRequest):
    if sqlite_storage is None and vector_storage is None:
        raise HTTPException(status_code=500, detail="Storage is not initialized")
    segments = _search_rewind_segments(
        query=req.query,
        start_time=req.start_time,
        end_time=req.end_time,
        top_k=req.top_k,
    )
    return RewindSearchResponse(query=req.query, segments=segments)


@app.post("/api/rewind/timeline_frames", response_model=RewindTimelineFramesResponse)
def rewind_timeline_frames(req: RewindTimelineFramesRequest):
    if sqlite_storage is None:
        raise HTTPException(status_code=500, detail="SQLite storage is not initialized")

    start_dt = _parse_optional_dt(req.start_time)
    end_dt = _parse_optional_dt(req.end_time)
    if not start_dt or not end_dt:
        raise HTTPException(status_code=400, detail="start_time and end_time are required")
    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start_time must be before end_time")

    limit = min(max(int(req.limit or 36), 1), 120)
    offset = max(int(req.offset or 0), 0)
    start_iso = _sql_dt(start_dt)
    end_iso = _sql_dt(end_dt)

    try:
        with sqlite_storage._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM frames f
                WHERE f.timestamp >= ? AND f.timestamp <= ?
                  AND f.frame_id LIKE 'frame_%'
                  AND f.image_path IS NOT NULL AND f.image_path != ''
                """,
                (start_iso, end_iso),
            )
            total_count = cursor.fetchone()["cnt"]

            cursor.execute(
                """
                SELECT
                    f.frame_id,
                    f.timestamp,
                    f.image_path,
                    o.text AS ocr_text
                FROM frames f
                LEFT JOIN ocr_text o ON f.frame_id = o.frame_id
                WHERE f.timestamp >= ? AND f.timestamp <= ?
                  AND f.frame_id LIKE 'frame_%'
                  AND f.image_path IS NOT NULL AND f.image_path != ''
                ORDER BY f.timestamp ASC
                LIMIT ? OFFSET ?
                """,
                (start_iso, end_iso, limit, offset),
            )
            rows = cursor.fetchall()

        frames: List[Dict[str, Any]] = []
        for row in rows:
            fid = row["frame_id"]
            sub_list = []
            for sf in sqlite_storage.get_sub_frames_for_frame(fid):
                sub_list.append(
                    {
                        "sub_frame_id": sf["sub_frame_id"],
                        "timestamp": _ts_to_iso(sf.get("timestamp")),
                        "app_name": sf.get("app_name", "") or "",
                        "window_name": sf.get("window_name", "") or "",
                        "image_path": _resolve_sub_frame_image_path(sf),
                    }
                )
            frames.append(
                {
                    "frame_id": fid,
                    "timestamp": _ts_to_iso(row["timestamp"]),
                    "image_path": row["image_path"],
                    "ocr_text": row["ocr_text"] or "",
                    "sub_frames": sub_list,
                }
            )

        return RewindTimelineFramesResponse(
            start_time=_ts_to_iso(start_dt),
            end_time=_ts_to_iso(end_dt),
            offset=offset,
            limit=limit,
            total_count=total_count,
            frames=frames,
        )
    except Exception as e:
        logger.error(f"Failed to load Rewind timeline frames: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load timeline frames: {e}")


@app.post("/api/rewind/build_context", response_model=TaskMemoryResponse)
def rewind_build_context(req: BuildRewindContextRequest):
    if not req.source_query.strip() and not req.selected_segments:
        raise HTTPException(status_code=400, detail="source_query or selected_segments is required")

    if req.selected_segments:
        segments = [_enrich_rewind_segment(segment) for segment in req.selected_segments]
    else:
        segments = _search_rewind_segments(
            query=req.source_query,
            start_time=req.start_time,
            end_time=req.end_time,
            top_k=req.top_k,
        )

    if not segments:
        raise HTTPException(status_code=404, detail="No timeline evidence found for Task Memory")

    title = req.title.strip() if req.title and req.title.strip() else _derive_task_memory_title(req.source_query, segments)
    markdown = _generate_task_memory_markdown(title, req.source_query, segments)
    now = _now_iso()
    task_memory_id = f"tm_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    memory = {
        "task_memory_id": task_memory_id,
        "title": title,
        "markdown": markdown,
        "source_query": req.source_query,
        "selected_segments": segments,
        "evidence_refs": _merge_rewind_evidence_refs(segments, req.evidence_refs),
        "created_at": now,
        "updated_at": now,
    }
    saved = _save_task_memory(memory)
    return TaskMemoryResponse(**saved)


@app.get("/api/rewind/task_memories", response_model=TaskMemoryListResponse)
def rewind_list_task_memories():
    items = []
    for memory in _list_task_memories():
        items.append(
            TaskMemoryListItem(
                task_memory_id=memory["task_memory_id"],
                title=memory["title"],
                source_query=memory["source_query"],
                created_at=memory["created_at"],
                updated_at=memory["updated_at"],
                selected_segment_count=len(memory.get("selected_segments") or []),
            )
        )
    return TaskMemoryListResponse(memories=items)


@app.get("/api/rewind/task_memories/{task_memory_id}", response_model=TaskMemoryResponse)
def rewind_get_task_memory(task_memory_id: str):
    return TaskMemoryResponse(**_load_task_memory(task_memory_id))


@app.patch("/api/rewind/task_memories/{task_memory_id}", response_model=TaskMemoryResponse)
def rewind_update_task_memory(task_memory_id: str, req: TaskMemoryPatchRequest):
    memory = _load_task_memory(task_memory_id)
    if req.title is not None:
        title = req.title.strip()
        if not title:
            raise HTTPException(status_code=400, detail="title cannot be empty")
        memory["title"] = title
    if req.markdown is not None:
        memory["markdown"] = req.markdown
    memory["updated_at"] = _now_iso()
    return TaskMemoryResponse(**_save_task_memory(memory))


@app.post("/api/rewind/task_memories/{task_memory_id}/ask", response_model=TaskMemoryAskResponse)
def rewind_ask_task_memory(task_memory_id: str, req: TaskMemoryAskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    memory = _load_task_memory(task_memory_id)
    segments = [_enrich_rewind_segment(segment) for segment in memory.get("selected_segments") or []]
    markdown = req.markdown if req.markdown is not None else memory.get("markdown", "")
    evidence_refs = _merge_rewind_evidence_refs(segments, memory.get("evidence_refs") or [])
    evidence = (
        f"{_format_timeline_evidence(segments)}\n\n"
        f"Persisted Evidence Refs:\n{_format_evidence_refs(evidence_refs)}"
    )
    system_prompt = (
        "你是 VisualMem 的任务继续助手。你必须基于 Task Memory 和它绑定的 "
        "timeline evidence 回答。如果信息不足，明确说不确定。回答要偏行动导向："
        "下一步做什么、检查什么文件/窗口/证据、有哪些未解决问题。"
    )
    prompt = f"""Task Memory:
{markdown}

Timeline Evidence:
{evidence}

Question:
{req.question}

    Answer requirements:
- Cite relevant timestamps or time ranges from Timeline Evidence.
- Prefer concrete next actions over generic advice.
- If the evidence does not support a claim, say that it is uncertain."""

    timeout_seconds = max(float(getattr(config, "REWIND_ASK_TIMEOUT_SECONDS", 90) or 90), 10.0)
    started_at = time_module.time()
    logger.info(
        "Task Memory ask started: task_memory_id=%s question_chars=%d markdown_chars=%d "
        "segments=%d evidence_refs=%d image_limit=%d timeout=%.0fs",
        task_memory_id,
        len(req.question or ""),
        len(markdown or ""),
        len(segments),
        len(evidence_refs),
        0,
        timeout_seconds,
    )
    try:
        answer = _call_rewind_ai(
            prompt,
            system_prompt,
            segments,
            image_limit=0,
            timeout_seconds=timeout_seconds,
        )
        if answer.startswith(("API调用失败", "错误:")):
            logger.warning("Task Memory ask model returned failure: %s", answer[:240])
            answer = (
                "AI 调用失败或超时，当前没有拿到可用回答。\n\n"
                f"错误信息：{_clip_one_line(answer, 500)}\n\n"
                "可以稍后重试，或先减少选中的 evidence segment 后重新 Build Task Memory。"
            )
    except Exception as e:
        logger.warning(f"Task Memory ask failed: {e}")
        answer = f"无法调用 AI 继续分析：{e}"
    finally:
        logger.info(
            "Task Memory ask finished: task_memory_id=%s elapsed=%.2fs",
            task_memory_id,
            time_module.time() - started_at,
        )

    return TaskMemoryAskResponse(
        task_memory_id=task_memory_id,
        answer=answer,
        evidence_refs=evidence_refs,
    )


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
            with sqlite_storage._connection() as conn:
                cursor = conn.cursor()
                if path_str.startswith("video_chunk:"):
                    cursor.execute("SELECT file_path, fps FROM video_chunks WHERE id = ?", (chunk_id,))
                else:
                    cursor.execute("SELECT file_path, fps FROM window_chunks WHERE id = ?", (chunk_id,))
                row = cursor.fetchone()
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
            
            with sqlite_storage._connection() as conn:
                cursor = conn.cursor()

                # 根据 chunk 类型查询不同的表
                if path_str.startswith("video_chunk:"):
                    cursor.execute("SELECT file_path, fps FROM video_chunks WHERE id = ?", (chunk_id,))
                else:  # window_chunk
                    cursor.execute("SELECT file_path, fps FROM window_chunks WHERE id = ?", (chunk_id,))

                row = cursor.fetchone()

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
                                with sqlite_storage._connection() as conn:
                                    cursor = conn.cursor()
                                    cursor.execute("SELECT image_path FROM frames WHERE frame_id = ?", (frame_id,))
                                    row = cursor.fetchone()

                                if row and row['image_path'] and row['image_path'] != path_str:
                                    logger.info(f"Found updated path for {frame_id}: {row['image_path']}")
                                    # 递归调用 get_image 处理新路径（可能是 video_chunk）
                                    return get_image(row['image_path'])
                        
                        elif 'windows' in path_str:
                            # 窗口子帧: subframe_{safe_app}_{ts_str}_{index}
                            # 注意：由于 safe_app 和 index 难以从路径反推，我们尝试模糊匹配 timestamp
                            if sqlite_storage:
                                with sqlite_storage._connection() as conn:
                                    cursor = conn.cursor()
                                    # 转换 ts_str (20260126_103850_579000) 到 ISO 格式的一部分进行匹配
                                    # 或者直接匹配 sub_frame_id 包含 ts_str 的记录
                                    cursor.execute("SELECT window_chunk_id, offset_index FROM sub_frames WHERE sub_frame_id LIKE ?", (f"%{ts_str}%",))
                                    row = cursor.fetchone()

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
        with sqlite_storage._connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) as count
                FROM frames f
                WHERE f.timestamp >= ? AND f.timestamp < ?
                AND f.image_path IS NOT NULL AND f.image_path != ''
                AND f.frame_id LIKE 'frame_%'
            """, (start_time_str, end_time_str))

            row = cursor.fetchone()

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
        with sqlite_storage._connection() as conn:
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
    停止录制信号：触发后端所有缓冲区的强制刷新。

    Order matters — the old implementation called ``_flush_all_video_buffers``
    and ``batch_write_buffer._flush_buffer`` first, but since the enrichment
    worker still had in-flight frames whose ``batch_write_buffer.add_frame``
    hadn't been called yet, those frames would miss the flush and stay
    uncommitted until the next periodic flush (or shutdown). That is the
    root cause of ``listDailyReports`` / ``recording.ts stopRecording``
    timing out — the HTTP worker pool was stuck inside the sync flush while
    the enrichment worker kept adding to the batch buffer behind its back.

    New order:
      1. Drain the enrichment queue (bounded by STOP_FLUSH_MAX_SECONDS) so
         all pending ``batch_write_buffer.add_frame`` calls complete.
      2. Flush video frame buffers (compress remaining temp PNGs into MP4).
      3. Flush the batch-write buffer (final commit to LanceDB + SQLite).
    """
    logger.info("收到前端录制停止信号...")

    stop_timeout = getattr(config, "STOP_FLUSH_MAX_SECONDS", 30.0)
    t0 = time_module.time()

    # 1. Drain the enrichment worker first so any in-flight frames land in
    # batch_write_buffer before we flush it below.
    if frame_enrichment_worker is not None:
        stats_before = frame_enrichment_worker.stats()
        logger.info(
            f"Draining frame enrichment queue "
            f"(queue={stats_before['queue_depth']} inflight={stats_before['inflight']}, "
            f"max {stop_timeout:.0f}s)..."
        )
        drained = frame_enrichment_worker.drain(timeout=stop_timeout)
        elapsed = time_module.time() - t0
        stats_after = frame_enrichment_worker.stats()
        if drained:
            logger.info(
                f"Frame enrichment drained in {elapsed:.2f}s "
                f"(completed_total={stats_after['completed']} failed_total={stats_after['failed']})"
            )
        else:
            logger.warning(
                f"Frame enrichment drain TIMEOUT after {elapsed:.2f}s — "
                f"queue={stats_after['queue_depth']} inflight={stats_after['inflight']}; "
                f"proceeding to flush anyway (remaining work will land on next flush)"
            )

    # 2. 刷新视频帧缓冲区（压缩剩余帧为MP4）
    if temp_frame_buffer is not None:
        logger.info("刷新视频帧缓冲区...")
        _flush_all_video_buffers()

    # 3. 刷新批量写入缓冲区
    if batch_write_buffer is not None:
        logger.info("刷新批量写入缓冲区...")
        batch_write_buffer._flush_buffer()

    # 4. 输出本次录制的聚类统计并重置计数器
    if cluster_manager is not None:
        stats = cluster_manager.get_assignment_stats()
        total = stats["total_frames"]
        vlm = stats["vlm_called_frames"]
        ratio = stats["vlm_call_ratio"]
        logger.info(
            f"Recording cluster stats: {vlm}/{total} frames needed VLM labeling ({ratio:.1%})"
        )
        cluster_manager.reset_assignment_stats()

    total_elapsed = time_module.time() - t0
    logger.info(f"Recording stop flush total elapsed={total_elapsed:.2f}s")
    return {
        "status": "success",
        "message": "All buffers flushed",
        "elapsed_seconds": round(total_elapsed, 2),
    }


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
    import os

    db_path = config.OCR_DB_PATH
    if not os.path.exists(db_path):
        return True, ""  # Fresh install, nothing to check

    try:
        with sqlite_storage._connection() as conn, sqlite_storage._activity_connection() as act_conn:
            cursor = conn.cursor()
            act_cursor = act_conn.cursor()

            # Check if required tables exist in activity DB
            act_cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
                "('activity_clusters', 'activity_sessions', 'activity_assignments')"
            )
            existing = {row[0] for row in act_cursor.fetchall()}
            missing = {"activity_clusters", "activity_sessions", "activity_assignments"} - existing
            if missing:
                return False, f"Missing clustering tables in activity DB: {', '.join(sorted(missing))}"

            # Check if there are any frames to worry about
            cursor.execute("SELECT COUNT(*) FROM sub_frames")
            total_frames = cursor.fetchone()[0]
            if total_frames == 0:
                return True, ""

            # Check if timeline is empty while frames exist
            act_cursor.execute("SELECT MAX(end_time) FROM activity_sessions")
            latest_session = act_cursor.fetchone()[0]
            if not latest_session:
                return False, f"Timeline is empty but {total_frames} frames exist"

            # Check how many frames are uncovered after the last session
            act_cursor.execute(
                "SELECT COUNT(*) FROM activity_assignments WHERE timestamp > ? AND activity_cluster_id IS NULL",
                (latest_session,),
            )
            uncovered = act_cursor.fetchone()[0]

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
