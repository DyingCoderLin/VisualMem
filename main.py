# main.py - 最终版主程序
"""
VisualMem - 灵活的视觉记忆助手

支持两种模式:
1. Simple模式 (默认): 不需要CLIP/向量数据库，直接存储图片
2. Vector模式 (可选): 使用CLIP编码 + LanceDB向量检索

通过 .env 中的 STORAGE_MODE 配置选择
"""

import time
import sys
import queue
import threading
from datetime import datetime
from config import config
from utils.logger import setup_logger

logger = setup_logger("main")

# ==================== 模块加载 ====================

logger.info(f"Starting VisualMem in {config.STORAGE_MODE.upper()} mode")

# 1. Capture - 屏幕捕捉（两种模式都需要）
from core.capture.screenshot_capturer import ScreenshotCapturer
capturer = ScreenshotCapturer()

# 2. Preprocess - 帧差过滤（两种模式都需要）
from core.preprocess.simple_filter import calculate_normalized_rms_diff
from PIL import Image as PILImage

class SimpleFrameFilter:
    """简化的帧差过滤器"""
    def __init__(self, diff_threshold: float):
        self.last_frame_image = None
        self.diff_threshold = diff_threshold
        logger.info(f"FrameFilter initialized with threshold: {diff_threshold}")
    
    def should_keep(self, image: PILImage.Image) -> bool:
        if self.last_frame_image is None:
            self.last_frame_image = image.copy()
            return True
        
        diff_score = calculate_normalized_rms_diff(self.last_frame_image, image)
        
        if diff_score < self.diff_threshold:
            logger.debug(f"Frame filtered (diff={diff_score:.4f})")
            return False
        
        self.last_frame_image = image.copy()
        logger.debug(f"Frame kept (diff={diff_score:.4f})")
        return True

frame_filter = SimpleFrameFilter(diff_threshold=config.SIMPLE_FILTER_DIFF_THRESHOLD)

# 3. OCR - 异步 OCR 处理（fallback 到 SQLite）
from core.ocr import create_ocr_engine, OCRResult
from core.storage.sqlite_storage import SQLiteStorage

# 创建 OCR 引擎
if config.ENABLE_OCR:
    try:
        ocr_engine = create_ocr_engine("pytesseract", lang="chi_sim+eng")
        USE_OCR = True
        logger.info("OCR engine initialized (async processing)")
    except Exception as e:
        logger.warning(f"Failed to initialize OCR engine: {e}")
        ocr_engine = create_ocr_engine("dummy")
        USE_OCR = False
else:
    ocr_engine = create_ocr_engine("dummy")
    USE_OCR = False
    logger.info("OCR disabled (ENABLE_OCR=false)")

# 创建 SQLite 存储（用于 OCR fallback）
sqlite_storage = SQLiteStorage(db_path=config.OCR_DB_PATH)
logger.info("SQLite storage initialized for OCR fallback")

# 创建 OCR 处理队列
ocr_queue = queue.Queue(maxsize=100)
ocr_thread_running = threading.Event()
ocr_thread_running.set()

# 4. Storage - 根据配置选择存储方式
encoder = None

if config.STORAGE_MODE == "simple":
    # Simple模式：只需要简单存储
    logger.info("Loading Simple Storage...")
    from core.storage.simple_storage import SimpleStorage
    storage = SimpleStorage(storage_path=config.IMAGE_STORAGE_PATH)
    logger.info("✓ Simple mode ready (no CLIP encoder needed)")
    
elif config.STORAGE_MODE == "vector":
    # Vector模式：需要CLIP + LanceDB
    logger.info("Loading Vector mode components...")
    try:
        from core.encoder.clip_encoder import CLIPEncoder
        from core.storage.lancedb_storage import LanceDBStorage
        
        encoder = CLIPEncoder(model_name=config.CLIP_MODEL)
        storage = LanceDBStorage(
            db_path=config.LANCEDB_PATH,
            embedding_dim=encoder.embedding_dim
        )
        logger.info("✓ Vector mode ready (CLIP + LanceDB)")
    except ImportError as e:
        logger.error(f"Vector mode requires additional dependencies: {e}")
        logger.error("Install with: pip install transformers torch lancedb")
        sys.exit(1)
else:
    logger.error(f"Unknown STORAGE_MODE: {config.STORAGE_MODE}")
    sys.exit(1)

logger.info("All modules loaded successfully")

# ==================== OCR Worker Thread ====================

def ocr_worker():
    """
    OCR 工作线程
    
    异步处理 OCR 识别，将结果存入 SQLite（fallback 存储）
    参考 screenpipe 的 handle_new_transcript 逻辑
    """
    logger.info("OCR worker thread started")
    
    while ocr_thread_running.is_set():
        try:
            # 从队列获取任务（超时 1 秒避免阻塞）
            task = ocr_queue.get(timeout=1.0)
            
            frame_id = task["frame_id"]
            timestamp = task["timestamp"]
            image = task["image"]
            image_path = task["image_path"]
            
            # 执行 OCR 识别
            ocr_result: OCRResult = ocr_engine.recognize(image)
            
            # 存入 SQLite
            success = sqlite_storage.store_frame_with_ocr(
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
            
            if success:
                logger.debug(
                    f"OCR worker processed frame {frame_id} "
                    f"(text_length={len(ocr_result.text)}, "
                    f"confidence={ocr_result.confidence:.2f})"
                )
            else:
                logger.warning(f"OCR worker failed to store frame {frame_id}")
            
            ocr_queue.task_done()
            
        except queue.Empty:
            # 队列为空，继续等待
            continue
        except Exception as e:
            logger.error(f"OCR worker error: {e}", exc_info=True)
    
    logger.info("OCR worker thread stopped")

# 启动 OCR 工作线程
ocr_thread = threading.Thread(target=ocr_worker, daemon=True, name="OCRWorker")
ocr_thread.start()
logger.info("OCR worker thread launched")

# ==================== Helper Functions ====================

def generate_frame_id(timestamp: datetime) -> str:
    """
    生成唯一的frame ID（基于时间戳）
    
    格式: YYYYMMDD_HHMMSS_ffffff
    例如: 20251201_143025_123456
    
    优点：按文件名排序即为时间排序
    """
    return timestamp.strftime("%Y%m%d_%H%M%S_") + f"{timestamp.microsecond:06d}"

def enqueue_ocr_task(frame_id: str, timestamp: datetime, image: PILImage.Image, image_path: str):
    """
    将 OCR 任务加入队列（异步处理）
    
    参考 screenpipe 的 ocr_frame_queue.push() 逻辑
    """
    if not USE_OCR:
        return
    
    try:
        task = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "image": image.copy(),  # 复制图片避免并发问题
            "image_path": image_path
        }
        ocr_queue.put(task, block=False)
        logger.debug(f"Enqueued OCR task for frame {frame_id}")
    except queue.Full:
        logger.warning(f"OCR queue full, dropping frame {frame_id}")
    except Exception as e:
        logger.error(f"Failed to enqueue OCR task: {e}")

# ==================== Pipeline ====================

def capture_and_store_pipeline():
    """
    捕捉和存储Pipeline
    根据STORAGE_MODE自动选择处理方式
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Capture iteration at {time.ctime()}")
        
        # 1. 捕捉
        frame = capturer.capture()
        if not frame:
            logger.warning("Capture failed, skipping")
            return
        
        # 2. 帧差过滤
        if not frame_filter.should_keep(frame.image):
            logger.info("Frame filtered out (no significant change)")
            return
        
        logger.info("Frame passed filter, processing...")
        
        # 3. 生成 frame_id
        frame_id = generate_frame_id(frame.timestamp)
        
        # 4. 存储（根据模式不同）
        image_path = ""  # 将在存储时获得实际路径
        
        if config.STORAGE_MODE == "simple":
            # Simple模式：直接存储图片
            logger.info("Storing to Simple Storage...")
            success = storage.store_frame(
                frame_id=frame_id,
                timestamp=frame.timestamp,
                image=frame.image,
                ocr_text="",  # OCR 由异步线程处理
                metadata={"size": frame.image.size}
            )
            # 获取存储的图片路径（从 metadata 中）
            # 这里简化处理，实际路径由 storage 生成
            from pathlib import Path
            date_dir = Path(config.IMAGE_STORAGE_PATH) / frame.timestamp.strftime("%Y%m%d")
            image_path = str((date_dir / f"{frame_id}.jpg").resolve())
            
        else:
            # Vector模式：编码后存储
            logger.info("Encoding image with CLIP...")
            embedding = encoder.encode_image(frame.image)
            
            logger.info("Storing to LanceDB...")
            success = storage.store_frame(
                frame_id=frame_id,
                timestamp=frame.timestamp,
                image=frame.image,
                embedding=embedding,
                ocr_text="",  # OCR 由异步线程处理
                metadata={"size": frame.image.size}
            )
            # 获取存储的图片路径
            from pathlib import Path
            date_dir = Path(config.IMAGE_STORAGE_PATH) / frame.timestamp.strftime("%Y%m%d")
            image_path = str((date_dir / f"{frame_id}.jpg").resolve())
        
        # 5. 异步 OCR 处理（放入队列）
        if success and USE_OCR:
            enqueue_ocr_task(frame_id, frame.timestamp, frame.image, image_path)
        
        if success:
            logger.info("✓ Frame stored successfully")
        else:
            logger.error("✗ Failed to store frame")
            
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(f"Error in pipeline: {e}", exc_info=True)

# ==================== Main Loop ====================

def main_loop():
    """主后台循环：持续捕捉和存储"""
    print("\n" + "="*60)
    print("VisualMem - Visual Memory Assistant")
    print("="*60)
    print("\n配置:")
    print(f"  • Storage Mode: {config.STORAGE_MODE.upper()}")
    print(f"  • Capturer:     {config.CAPTURER_TYPE}")
    print(f"  • Filter:       帧差过滤 (阈值: {config.SIMPLE_FILTER_DIFF_THRESHOLD})")
    
    if config.STORAGE_MODE == "simple":
        print(f"  • Storage:      Simple (文件系统)")
        print(f"  • Path:         {config.IMAGE_STORAGE_PATH}")
    else:
        print(f"  • Encoder:      CLIP ({config.CLIP_MODEL})")
        print(f"  • Storage:      LanceDB")
        print(f"  • Path:         {config.LANCEDB_PATH}")
    
    print(f"  • Interval:     {config.CAPTURE_INTERVAL_SECONDS}s")
    print(f"  • OCR:          {'启用 (异步)' if USE_OCR else '禁用'}")
    if USE_OCR:
        print(f"  • OCR Fallback: SQLite ({config.OCR_DB_PATH})")
    print("="*60)
    
    # 显示统计
    stats = storage.get_stats()
    print(f"\n当前数据库状态:")
    print(f"  • 已存储帧数: {stats.get('total_frames', 0)}")
    if config.STORAGE_MODE == "vector":
        print(f"  • Embedding维度: {stats.get('embedding_dim', 0)}")
    
    print("\n开始捕捉循环... (按 Ctrl+C 停止)")
    print("="*60)
    print()
    
    try:
        while True:
            capture_and_store_pipeline()
            time.sleep(config.CAPTURE_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("正在关闭...")
        print("="*60)
        
        # 停止 OCR 工作线程
        logger.info("Stopping OCR worker thread...")
        ocr_thread_running.clear()
        
        # 等待队列处理完成
        if not ocr_queue.empty():
            print(f"等待 OCR 队列处理完成（剩余 {ocr_queue.qsize()} 个任务）...")
            ocr_queue.join()
        
        # 等待线程结束
        ocr_thread.join(timeout=5.0)
        logger.info("OCR worker thread stopped")
        
        # 显示最终统计
        stats = storage.get_stats()
        print(f"\n最终统计:")
        print(f"  • 总帧数: {stats.get('total_frames', 0)}")
        print(f"  • 模式: {config.STORAGE_MODE}")
        
        if USE_OCR:
            ocr_stats = sqlite_storage.get_stats()
            print(f"\nOCR Fallback 统计:")
            print(f"  • OCR 识别帧数: {ocr_stats.get('total_ocr_results', 0)}")
            print(f"  • 总文本长度: {ocr_stats.get('total_text_length', 0)}")
        
        logger.info("Received shutdown signal")
        sys.exit(0)

if __name__ == "__main__":
    main_loop()
