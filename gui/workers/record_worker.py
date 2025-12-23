#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
录制工作线程 - 执行 main.py 的捕捉和存储逻辑
"""

import time
import queue
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional
import base64
import io

import requests

from PySide6.QtCore import QObject, Signal
from PIL import Image as PILImage
import numpy as np

from config import config
from utils.logger import setup_logger
from core.capture.screenshot_capturer import ScreenshotCapturer
from core.preprocess.simple_filter import calculate_normalized_rms_diff
from core.ocr import create_ocr_engine, OCRResult
from core.storage.sqlite_storage import SQLiteStorage

logger = setup_logger("record_worker")


class RecordWorker(QObject):
    """录制工作线程"""
    
    status_signal = Signal(str)  # 状态信号
    stats_signal = Signal(dict)  # 统计信号
    error_signal = Signal(str)  # 错误信号
    init_progress_signal = Signal(str)  # 初始化进度信号
    ready_signal = Signal()  # 准备就绪信号
    
    def __init__(self, storage_mode: str):
        super().__init__()
        self.storage_mode = storage_mode
        self.running = False
        self.initialized = False
        
        # GUI 模式（local / remote）
        self.gui_mode = config.GUI_MODE
        self.backend_url = config.GUI_REMOTE_BACKEND_URL.rstrip("/") if config.GUI_REMOTE_BACKEND_URL else ""
        
        # 延迟初始化的组件
        self.capturer: Optional[ScreenshotCapturer] = None
        self.encoder = None
        self.storage = None
        self.ocr_engine = None
        self.sqlite_storage: Optional[SQLiteStorage] = None
        self.ocr_queue: Optional[queue.Queue] = None
        self.ocr_thread: Optional[threading.Thread] = None
        self.ocr_thread_running: Optional[threading.Event] = None
        self.use_ocr = False
        self.last_frame_image: Optional[PILImage.Image] = None
    
    def initialize_components(self):
        """初始化所有组件 (在工作线程中调用)"""
        if self.initialized:
            return
        
        try:
            self._init_capturer()
            self._init_storage()
            self._init_sqlite()
            self._init_ocr()
            
            self.initialized = True
            self.init_progress_signal.emit("✅ 所有组件初始化完成!")
            self.ready_signal.emit()
            
            logger.info(f"RecordWorker 初始化完成 (模式: {self.storage_mode})")
            
        except Exception as e:
            error_msg = f"初始化失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.error_signal.emit(error_msg)
            raise
    
    def _init_capturer(self):
        """初始化截图器"""
        self.init_progress_signal.emit("正在初始化截图组件...")
        self.capturer = ScreenshotCapturer()
        time.sleep(0.1)
    
    def _init_storage(self):
        """初始化存储

        - GUI_MODE=local  : 使用原有本地存储逻辑（Simple / LanceDB）
        - GUI_MODE=remote : 不初始化本地存储，所有持久化由远程 backend_server 负责
        """
        if config.GUI_MODE == "remote":
            # 远程模式下不需要本地主存储，只做采集 + 帧差 + 压缩 + 上传
            self.init_progress_signal.emit("GUI 远程模式: 跳过本地存储初始化，使用 HTTP 上传到后端")
            self.storage = None
            self.encoder = None
            return

        if self.storage_mode == "simple":
            self.init_progress_signal.emit("正在初始化Simple存储...")
            from core.storage.simple_storage import SimpleStorage
            self.storage = SimpleStorage(storage_path=config.IMAGE_STORAGE_PATH)
            self.encoder = None
        else:
            # Vector模式: 预加载CLIP模型
            self.init_progress_signal.emit("正在加载CLIP模型... (首次加载较慢，请稍候)")
            from core.encoder.clip_encoder import CLIPEncoder
            from core.storage.lancedb_storage import LanceDBStorage
            
            self.encoder = CLIPEncoder(model_name=config.CLIP_MODEL)
            
            # 预热模型
            self.init_progress_signal.emit("正在预热CLIP模型...")
            dummy_image = PILImage.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            _ = self.encoder.encode_image(dummy_image)
            
            self.init_progress_signal.emit("正在初始化LanceDB存储...")
            self.storage = LanceDBStorage(
                db_path=config.LANCEDB_PATH,
                embedding_dim=self.encoder.embedding_dim
            )
    
    def _init_sqlite(self):
        """初始化SQLite

        - GUI_MODE=local  : 初始化本地 OCR SQLite（原有行为）
        - GUI_MODE=remote : 跳过本地 SQLite，OCR 与持久化由远程 backend_server 负责
        """
        if config.GUI_MODE == "remote":
            self.init_progress_signal.emit("GUI 远程模式: 跳过本地 SQLite 初始化，由后端负责 OCR/存储")
            self.sqlite_storage = None
            return

        self.init_progress_signal.emit("正在初始化SQLite...")
        self.sqlite_storage = SQLiteStorage(db_path=config.OCR_DB_PATH)
    
    def _init_ocr(self):
        """初始化OCR

        - GUI_MODE=local  : 本地初始化 OCR 引擎 + OCR 异步线程
        - GUI_MODE=remote : 跳过本地 OCR，OCR 由远程 backend_server 负责
        """
        if config.GUI_MODE == "remote":
            self.init_progress_signal.emit("GUI 远程模式: 跳过本地 OCR 初始化，由后端负责 OCR")
            self.ocr_engine = None
            self.use_ocr = False
            return

        if config.ENABLE_OCR:
            self.init_progress_signal.emit("正在初始化OCR引擎...")
            try:
                self.ocr_engine = create_ocr_engine("pytesseract", lang="chi_sim+eng")
                self.use_ocr = True
            except Exception as e:
                logger.warning(f"OCR初始化失败: {e}")
                self.ocr_engine = create_ocr_engine("dummy")
                self.use_ocr = False
        else:
            self.ocr_engine = create_ocr_engine("dummy")
            self.use_ocr = False
        
        if self.use_ocr:
            self.init_progress_signal.emit("正在启动OCR工作线程...")
            self.ocr_queue = queue.Queue(maxsize=100)
            self.ocr_thread_running = threading.Event()
            self.ocr_thread_running.set()
            self.ocr_thread = threading.Thread(target=self._ocr_worker, daemon=True)
            self.ocr_thread.start()
    
    def _ocr_worker(self):
        """OCR 异步工作线程"""
        while self.ocr_thread_running.is_set():
            try:
                task = self.ocr_queue.get(timeout=1.0)
                frame_id = task["frame_id"]
                timestamp = task["timestamp"]
                image = task["image"]
                image_path = task["image_path"]
                
                logger.debug(f"OCR worker处理frame {frame_id}...")
                ocr_result: OCRResult = self.ocr_engine.recognize(image)
                
                self.sqlite_storage.store_frame_with_ocr(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    image_path=image_path,
                    ocr_text=ocr_result.text,
                    ocr_text_json=ocr_result.text_json,
                    ocr_engine=ocr_result.engine,
                    ocr_confidence=ocr_result.confidence,
                    device_name="default",
                    metadata={"size": image.size}
                )
                
                logger.debug(
                    f"OCR worker完成 {frame_id}: "
                    f"文本长度={len(ocr_result.text)}, "
                    f"置信度={ocr_result.confidence:.2f}"
                )
                
                self.ocr_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"OCR worker错误: {e}", exc_info=True)
    
    def _should_keep_frame(self, image: PILImage.Image) -> bool:
        """帧差过滤"""
        if self.last_frame_image is None:
            self.last_frame_image = image.copy()
            return True
        
        diff_score = calculate_normalized_rms_diff(self.last_frame_image, image)
        
        if diff_score < config.SIMPLE_FILTER_DIFF_THRESHOLD:
            return False
        
        self.last_frame_image = image.copy()
        return True
    
    def _generate_frame_id(self, timestamp: datetime) -> str:
        """
        生成 frame ID（基于时间戳）
        
        格式: YYYYMMDD_HHMMSS_ffffff
        例如: 20251201_143025_123456
        
        优点：按文件名排序即为时间排序
        """
        return timestamp.strftime("%Y%m%d_%H%M%S_") + f"{timestamp.microsecond:06d}"
    
    def _enqueue_ocr_task(self, frame_id: str, timestamp: datetime, 
                         image: PILImage.Image, image_path: str):
        """将 OCR 任务加入队列"""
        if not self.use_ocr:
            return
        
        try:
            task = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "image": image.copy(),
                "image_path": image_path
            }
            self.ocr_queue.put(task, block=False)
        except queue.Full:
            logger.warning(f"OCR队列已满,丢弃frame {frame_id}")
        except Exception as e:
            logger.error(f"加入OCR任务失败: {e}")
    
    def start_recording(self):
        """开始录制"""
        # 先初始化组件
        if not self.initialized:
            logger.info("首次启动，正在初始化组件...")
            self.initialize_components()
            if not self.initialized:
                self.error_signal.emit("初始化失败，无法开始录制")
                return
        
        self.running = True
        logger.info("开始录制...")
        self.status_signal.emit("正在录制中...")
        
        frame_count = 0
        
        while self.running:
            try:
                # 1. 捕捉
                frame = self.capturer.capture()
                if not frame:
                    logger.warning("捕捉失败")
                    time.sleep(config.CAPTURE_INTERVAL_SECONDS)
                    continue
                
                # 2. 帧差过滤
                if not self._should_keep_frame(frame.image):
                    logger.debug("帧已过滤")
                    time.sleep(config.CAPTURE_INTERVAL_SECONDS)
                    continue
                
                logger.info("帧通过过滤,处理中...")
                
                # 3. 生成 frame_id
                frame_id = self._generate_frame_id(frame.timestamp)

                if config.GUI_MODE == "remote":
                    # 远程模式：不在本地写盘，压缩后通过 HTTP 上传到后端
                    try:
                        if not config.GUI_REMOTE_BACKEND_URL:
                            raise RuntimeError("GUI_REMOTE_BACKEND_URL 未配置，无法上传到后端")

                        # 压缩为 JPEG
                        buf = io.BytesIO()
                        frame.image.save(buf, format="JPEG", quality=config.IMAGE_QUALITY)
                        img_bytes = buf.getvalue()
                        img_b64 = base64.b64encode(img_bytes).decode("ascii")

                        payload = {
                            "frame_id": frame_id,
                            "timestamp": frame.timestamp.isoformat(),
                            "image_base64": img_b64,
                            "metadata": {
                                "size": frame.image.size,
                            },
                        }
                        url = config.GUI_REMOTE_BACKEND_URL.rstrip("/") + "/api/store_frame"
                        resp = requests.post(url, json=payload, timeout=30)
                        resp.raise_for_status()
                        logger.info(f"远程上传帧成功: {frame_id}")
                    except Exception as e:
                        logger.error(f"远程上传帧失败: {e}", exc_info=True)
                        self.error_signal.emit(f"远程上传帧失败: {e}")
                        time.sleep(config.CAPTURE_INTERVAL_SECONDS)
                        continue

                else:
                    # 本地模式：在本地写盘 + SQLite + OCR（原有行为）
                    # 4. 构造图片路径
                    date_dir = Path(config.IMAGE_STORAGE_PATH) / frame.timestamp.strftime("%Y%m%d")
                    image_path = str(date_dir / f"{frame_id}.jpg")
                    
                    # 5. 存储到主存储
                    success = self._store_frame(frame_id, frame, image_path)
                    
                    if not success:
                        logger.error("存储失败")
                        time.sleep(config.CAPTURE_INTERVAL_SECONDS)
                        continue
                    
                    # 6. 立即存储基础信息到SQLite
                    self._store_to_sqlite(frame_id, frame, image_path)
                    
                    # 7. 异步 OCR
                    if self.use_ocr:
                        self._enqueue_ocr_task(frame_id, frame.timestamp, frame.image, image_path)
                
                frame_count += 1
                logger.info(f"✓ 帧已存储 (总计: {frame_count})")
                
                # 获取统计信息（本地模式使用 storage.get_stats，远程模式使用 API）
                if self.gui_mode == "remote":
                    stats = self._get_stats_via_api()
                else:
                    if self.storage is not None:
                        stats = self.storage.get_stats()
                    else:
                        stats = {"total_frames": 0, "storage_mode": "unknown"}
                
                stats["recording_frames"] = frame_count
                self.stats_signal.emit(stats)
                
                time.sleep(config.CAPTURE_INTERVAL_SECONDS)
                
            except Exception as e:
                logger.error(f"录制错误: {e}", exc_info=True)
                self.error_signal.emit(f"录制错误: {str(e)}")
                time.sleep(config.CAPTURE_INTERVAL_SECONDS)
    
    def _store_frame(self, frame_id: str, frame, image_path: str) -> bool:
        """存储帧到主存储"""
        if self.storage_mode == "simple":
            return self.storage.store_frame(
                frame_id=frame_id,
                timestamp=frame.timestamp,
                image=frame.image,
                ocr_text="",
                metadata={"size": frame.image.size}
            )
        else:
            # Vector模式: 计算embedding并存储
            logger.debug("正在计算embedding...")
            embedding = self.encoder.encode_image(frame.image)
            return self.storage.store_frame(
                frame_id=frame_id,
                timestamp=frame.timestamp,
                image=frame.image,
                embedding=embedding,
                ocr_text="",
                metadata={"size": frame.image.size}
            )
    
    def _store_to_sqlite(self, frame_id: str, frame, image_path: str):
        """存储基础信息到SQLite"""
        try:
            self.sqlite_storage.store_frame_with_ocr(
                frame_id=frame_id,
                timestamp=frame.timestamp,
                image_path=image_path,
                ocr_text="",
                ocr_text_json="",
                ocr_engine="pending",
                ocr_confidence=0.0,
                device_name="default",
                metadata={"size": frame.image.size}
            )
            logger.debug(f"Frame {frame_id} 基础信息已存入SQLite")
        except Exception as e:
            logger.warning(f"存储基础信息到SQLite失败: {e}")
    
    def _get_stats_via_api(self) -> dict:
        """
        通过 API 从远程后端获取统计信息
        
        Returns:
            统计信息字典
        """
        try:
            if not self.backend_url:
                logger.warning("远程模式下 backend_url 未配置，返回默认统计信息")
                return {
                    "total_frames": 0,
                    "storage_mode": "remote",
                    "recording_frames": 0
                }
            
            url = self.backend_url + "/api/stats"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            logger.debug(f"从远程后端获取统计信息: {data}")
            return data
            
        except Exception as e:
            logger.warning(f"从远程后端获取统计信息失败: {e}，返回默认统计信息")
            return {
                "total_frames": 0,
                "storage_mode": "remote",
                "recording_frames": 0
            }
    
    def stop_recording(self):
        """停止录制"""
        self.running = False
        logger.info("停止录制...")
        
        # 停止 OCR 工作线程
        if self.use_ocr:
            self.ocr_thread_running.clear()
            if not self.ocr_queue.empty():
                logger.info(f"等待OCR队列处理完成 ({self.ocr_queue.qsize()} 个任务)...")
                self.ocr_queue.join()
            self.ocr_thread.join(timeout=5.0)
        
        self.status_signal.emit("录制已停止")


