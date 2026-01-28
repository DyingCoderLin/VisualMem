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

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image as PILImage
from pathlib import Path

from config import config
from utils.logger import setup_logger
from core.encoder import create_encoder
from core.storage.lancedb_storage import LanceDBStorage
from core.storage.sqlite_storage import SQLiteStorage
from core.retrieval.query_llm_utils import rewrite_and_time, filter_by_time
from core.retrieval.reranker import Reranker
from core.understand.api_vlm import ApiVLM
from core.ocr import create_ocr_engine
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


# ============ 全局单例组件 ============

encoder = None
vector_storage: Optional[LanceDBStorage] = None
sqlite_storage: Optional[SQLiteStorage] = None
reranker: Optional[Reranker] = None
vlm: Optional[ApiVLM] = None
ocr_engine = None

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
            should_flush = len(self.buffer) >= self.batch_size
            if should_flush:
                logger.info(f"缓冲区达到批次大小 {self.batch_size}，触发批量写入")
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
            logger.info(f"批量写入 {len(frames_to_write)} 帧到 LanceDB...")
            # 批量写入到 LanceDB
            if vector_storage is not None:
                success = vector_storage.store_frames_batch(frames_to_write)
                if success:
                    logger.info(f"成功批量写入 {len(frames_to_write)} 帧")
                else:
                    logger.error("批量写入失败")
            
            # 批量写入到 SQLite（逐条写入，因为 SQLite 的批量写入接口可能不同）
            if sqlite_storage is not None:
                for frame_data in frames_to_write:
                    try:
                        sqlite_storage.store_frame_with_ocr(
                            frame_id=frame_data["frame_id"],
                            timestamp=frame_data["timestamp"],
                            image_path=frame_data["image_path"],
                            ocr_text=frame_data.get("ocr_text", ""),
                            ocr_text_json=frame_data.get("ocr_text_json", ""),
                            ocr_engine=frame_data.get("ocr_engine", "pending"),
                            ocr_confidence=frame_data.get("ocr_confidence", 0.0),
                            device_name=frame_data.get("device_name", "remote-gui"),
                            metadata=frame_data.get("metadata", {}),
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
                logger.info(f"达到刷新间隔 {self.flush_interval} 秒，触发批量写入")
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


def _init_components():
    """Lazy-init heavy components (called on first request)."""
    global encoder, vector_storage, sqlite_storage, reranker, vlm, ocr_engine

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

    global ocr_engine
    if ocr_engine is None and config.ENABLE_OCR:
        try:
            ocr_engine = create_ocr_engine("pytesseract", lang="chi_sim+eng")
            logger.info("OCR engine initialized (pytesseract).")
        except Exception as e:
            logger.warning(f"Failed to init OCR engine, fallback to dummy: {e}")
            ocr_engine = create_ocr_engine("dummy")


def _init_all_components():
    """
    强制初始化所有组件（用于服务器启动时预加载）。
    与 _init_components() 不同，这个函数会强制加载，不检查是否已加载。
    """
    global encoder, vector_storage, sqlite_storage, reranker, vlm, ocr_engine, batch_write_buffer
    
    logger.info("=" * 60)
    logger.info("Initializing all backend components (startup preload)...")
    logger.info("=" * 60)

    # 0. Pre-flight check: Ensure models are downloaded
    # This provides better UX by explicitly showing download progress
    ensure_model_downloaded(config.EMBEDDING_MODEL, "CLIP Encoder")
    
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
        logger.info("[6/7] Initializing OCR engine...")
        try:
            ocr_engine = create_ocr_engine("pytesseract", lang="chi_sim+eng")
        except Exception as e:
            ocr_engine = create_ocr_engine("dummy")
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
                logger.info("优化完成（cleanup_older_than=0.1小时）")
        else:
            logger.info("LanceDB 表不存在，跳过优化")
    except Exception as e:
        logger.warning(f"优化 LanceDB 失败: {e}")
    
    # 8. Initialize batch write buffer
    logger.info("[8/8] Initializing batch write buffer...")
    batch_write_buffer = BatchWriteBuffer(batch_size=10, flush_interval_seconds=60.0)
    batch_write_buffer.start()
    
    logger.info("=" * 60)
    logger.info("All backend components initialized successfully!")
    logger.info("=" * 60)


# ============ Pydantic models ============


class StoreFrameRequest(BaseModel):
    frame_id: str
    timestamp: str  # ISO string
    image_base64: str
    metadata: Optional[Dict] = None


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


# ============ Startup Event ============


@app.on_event("startup")
async def startup_event():
    """
    服务器启动时预加载所有重型组件（embedding model, reranker, etc.）
    这样可以避免第一次请求时的延迟。
    """
    _init_all_components()


@app.on_event("shutdown")
async def shutdown_event():
    """
    服务器关闭时清理资源，确保缓冲区数据写入磁盘。
    """
    logger.info("=" * 60)
    logger.info("Shutting down backend server...")
    if batch_write_buffer is not None:
        logger.info("Flushing batch write buffer...")
        batch_write_buffer.stop()
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


@app.post("/api/store_frame")
def store_frame(req: StoreFrameRequest):
    """
    Store a frame sent from remote GUI.
    - Decode base64 image
    - Save to IMAGE_STORAGE_PATH
    - Add to batch write buffer (批量写入，减少版本数量)
    - 当缓冲区达到10张图片或1分钟时，批量写入到 LanceDB 和 SQLite
    """
    # 组件已在启动时预加载，直接使用
    assert encoder is not None
    assert vector_storage is not None
    assert sqlite_storage is not None
    assert batch_write_buffer is not None

    try:
        ts = datetime.fromisoformat(req.timestamp)
    except Exception as e:
        logger.error(f"Invalid timestamp '{req.timestamp}': {e}")
        raise

    # Decode image
    img_bytes = base64.b64decode(req.image_base64)
    image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")

    # Build image path on server
    # 参考 gui_main.py 的逻辑，使用时间戳直接命名：YYYYMMDD_HHMMSS_ffffff.jpg
    date_dir = config.IMAGE_STORAGE_PATH
    date_path = Path(date_dir) / ts.strftime("%Y%m%d")
    date_path.mkdir(parents=True, exist_ok=True)
    
    # 生成基于时间戳的文件名和 frame_id：YYYYMMDD_HHMMSS_ffffff
    # frame_id 与文件名保持一致（不含 .jpg 扩展名）
    base_frame_id = ts.strftime("%Y%m%d_%H%M%S_") + f"{ts.microsecond:06d}"
    image_filename = f"{base_frame_id}.jpg"
    image_path = (date_path / image_filename).resolve()
    
    # 如果文件已存在（相同微秒时间戳），在微秒后追加序号
    if image_path.exists():
        counter = 1
        while image_path.exists():
            # 调整微秒，确保唯一性
            adjusted_microsecond = min(ts.microsecond + counter, 999999)
            adjusted_ts = ts.replace(microsecond=adjusted_microsecond)
            base_frame_id = adjusted_ts.strftime("%Y%m%d_%H%M%S_") + f"{adjusted_microsecond:06d}"
            image_filename = f"{base_frame_id}.jpg"
            image_path = (date_path / image_filename).resolve()
            counter += 1
    
    image.save(str(image_path), format="JPEG", quality=config.IMAGE_QUALITY)

    # 使用基于时间戳生成的 frame_id，而不是前端传来的旧格式
    frame_id = base_frame_id

    # Compute image embedding（但不立即写入，等待批量写入）
    embedding = encoder.encode_image(image)

    # OCR (if enabled)
    ocr_text = ""
    ocr_json = ""
    ocr_engine_name = "pending"
    ocr_conf = 0.0
    if ocr_engine is not None:
        try:
            result = ocr_engine.recognize(image)
            ocr_text = result.text
            ocr_json = result.text_json
            ocr_engine_name = result.engine
            ocr_conf = result.confidence
        except Exception as e:
            logger.warning(f"OCR failed for frame {frame_id}: {e}")

    # 添加到批量写入缓冲区（批量写入，不立即写入）
    frame_data = {
        "frame_id": frame_id,  # 使用新生成的 frame_id
        "timestamp": ts,
        "image": image,  # 保存 PIL Image 对象，批量写入时会保存
        "embedding": embedding,
        "ocr_text": ocr_text,
        "image_path": str(image_path),
        "ocr_text_json": ocr_json,
        "ocr_engine": ocr_engine_name,
        "ocr_confidence": ocr_conf,
        "device_name": "remote-gui",
        "metadata": req.metadata or {"size": image.size},
    }
    
    batch_write_buffer.add_frame(frame_data)
    
    logger.debug(f"Added frame to batch buffer: {frame_id} at {image_path}")
    return {"status": "ok"}


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

    # 3) 调用 LLM 做 query rewrite + time_range 解析
    #    - 无论是否开启 rewrite，都允许 LLM 解析 time_range
    #    - 是否采用扩写结果由 ENABLE_LLM_REWRITE 决定
    dense_queries = [req.query]
    sparse_queries = [req.query]
    llm_time_range = None
    try:
        dense_llm, sparse_llm, llm_time_range = rewrite_and_time(
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
                )
            else:
                logger.info(f"Performing image dense search for: {q}")
                res = vector_storage.search(
                    emb,
                    top_k=top_k,
                    start_time=start_time,
                    end_time=end_time,
                )
            frames.extend(res)
        return frames

    # Sparse search via SQLite FTS5
    def _sparse_search() -> List[Dict]:
        if not enable_hybrid:
            return []
        frames: List[Dict] = []
        for q in sparse_queries:
            res = sqlite_storage.search_by_text(q, limit=top_k)
            if start_time or end_time:
                res = filter_by_time(res, (start_time, end_time))
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

    if not frames:
        return QueryRagWithTimeResponse(answer="在指定时间范围内未找到相关的屏幕记录。", frames=[])

    # Load images for rerank + VLM
    loaded_frames: List[Dict] = []
    for f in frames:
        path = f.get("image_path")
        if not path:
            continue
        try:
            img = PILImage.open(path)
            f["image"] = img
            loaded_frames.append(f)
        except Exception as e:
            logger.warning(f"Failed to load image for frame {f.get('frame_id')}: {e}")

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


@app.get("/api/image")
def get_image(path: str = Query(..., description="Image file path")):
    """
    获取图片文件（用于前端显示）
    支持绝对路径或相对路径（相对于项目根目录）
    """
    try:
        path_str = str(path)
        
        # 如果是绝对路径，直接使用
        if Path(path_str).is_absolute():
            final_path = Path(path_str)
        else:
            # 相对路径处理
            # 获取项目根目录（脚本所在目录的父目录，或当前工作目录）
            # 尝试多种方式找到项目根目录
            script_dir = Path(__file__).parent.absolute()
            project_root = script_dir  # gui_backend_server.py 在项目根目录
            cwd = Path.cwd().absolute()
            
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
        frames = sqlite_storage.get_frames_in_timerange(start_time=start_time, end_time=end_time)
        
        recent_frames = []
        for f in frames:
            # 转换为前端需要的格式
            recent_frames.append({
                "frame_id": f["frame_id"],
                "timestamp": f["timestamp"].isoformat(),
                "image_path": f["image_path"],
                "ocr_text": f["ocr_text"]
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
        
        # 使用 COUNT 查询获取总数（只统计有 image_path 的帧）
        conn = sqlite_storage._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM frames f
            WHERE f.timestamp >= ? AND f.timestamp < ?
            AND f.image_path IS NOT NULL AND f.image_path != ''
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
            ORDER BY f.timestamp ASC
            LIMIT ? OFFSET ?
        """, (start_time_str, end_time_str, limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        # 转换为 API 响应格式（只返回路径，不返回 base64）
        result = []
        for row in rows:
            # 只返回有 image_path 的帧
            if not row["image_path"]:
                continue
                
            ts = datetime.fromisoformat(row["timestamp"])
            ts_str = ts.isoformat()
            result.append({
                "frame_id": row["frame_id"],
                "timestamp": ts_str,
                "image_path": row["image_path"],
                "ocr_text": row["ocr_text"] or "",
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
            limit=100000  # 设置一个较大的 limit，确保获取所有数据
        )
        
        # 转换为 API 响应格式（只返回路径，不返回 base64）
        result = []
        for frame in all_frames:
            ts = frame.get("timestamp")
            ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)
            result.append({
                "frame_id": frame.get("frame_id", ""),
                "timestamp": ts_str,
                "image_path": frame.get("image_path", ""),  # 只返回路径
                "ocr_text": frame.get("ocr_text", ""),
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
    停止录制信号：仅用于触发后端批量写入缓冲区的强制刷新
    """
    if batch_write_buffer is not None:
        logger.info("收到前端录制停止信号，强制刷新缓冲区...")
        batch_write_buffer._flush_buffer()
    return {"status": "success", "message": "Buffer flushed"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "gui_backend_server:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
    )
