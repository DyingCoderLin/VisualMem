# core/capture/recording_coordinator.py
"""
Recording Coordinator Module

Orchestrates the complete capture-to-storage pipeline:
1. Capture screen and windows
2. Detect frame differences (independently for screen and each window)
3. Write changed frames to MP4 video chunks
4. Run OCR on changed frames
5. Generate embeddings
6. Store metadata and mappings to database

Reference: screenpipe's core.rs for the recording coordination pattern
"""
import uuid
import asyncio
import datetime
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image

from utils.logger import setup_logger
from utils.data_models import ScreenObject, WindowFrame
from config import config

from .window_capturer import WindowCapturer
from ..preprocess.frame_diff import FrameDiffDetector, FrameDiffResult
from ..storage.video_chunk_writer import VideoChunkManager
from ..storage.sqlite_storage import SQLiteStorage

logger = setup_logger(__name__)


@dataclass
class RecordingConfig:
    """Configuration for recording coordinator"""
    output_dir: str = ""
    monitor_id: int = 0
    fps: float = 1.0
    chunk_duration: int = 60
    capture_windows: bool = True
    capture_unfocused_windows: bool = True
    screen_diff_threshold: float = 0.006
    window_diff_threshold: float = 0.006
    run_ocr: bool = True
    run_embedding: bool = True
    max_image_width: int = 0
    
    def __post_init__(self):
        if not self.output_dir:
            self.output_dir = getattr(config, 'DATA_DIR', './data')


@dataclass
class RecordingStats:
    """Statistics for the recording session"""
    frames_captured: int = 0
    frames_stored: int = 0
    windows_captured: int = 0
    windows_stored: int = 0
    ocr_processed: int = 0
    embeddings_generated: int = 0
    errors: int = 0
    start_time: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )


class RecordingCoordinator:
    """
    Coordinates the complete recording pipeline
    
    Manages:
    - Screen and window capture
    - Independent frame difference detection
    - Video chunk encoding
    - OCR processing
    - Embedding generation
    - Database storage
    """
    
    def __init__(
        self,
        config: RecordingConfig,
        db: Optional[SQLiteStorage] = None,
        ocr_engine: Optional[Any] = None,
        embedding_encoder: Optional[Any] = None,
        on_frame_stored: Optional[Callable[[str, dict], None]] = None,
        on_subframe_stored: Optional[Callable[[str, dict], None]] = None
    ):
        """
        Args:
            config: Recording configuration
            db: SQLite storage instance (created if not provided)
            ocr_engine: OCR engine instance (optional)
            embedding_encoder: Embedding encoder instance (optional)
            on_frame_stored: Callback when a frame is stored
            on_subframe_stored: Callback when a sub-frame is stored
        """
        self.config = config
        self.db = db or SQLiteStorage()
        self.ocr_engine = ocr_engine
        self.embedding_encoder = embedding_encoder
        self.on_frame_stored = on_frame_stored
        self.on_subframe_stored = on_subframe_stored
        
        # Initialize components
        self.capturer = WindowCapturer(
            monitor_id=config.monitor_id,
            max_width=config.max_image_width,
            capture_windows=config.capture_windows,
            capture_unfocused_windows=config.capture_unfocused_windows
        )
        
        self.frame_diff = FrameDiffDetector(
            screen_threshold=config.screen_diff_threshold,
            window_threshold=config.window_diff_threshold
        )
        
        self.video_manager = VideoChunkManager(
            output_dir=config.output_dir,
            fps=config.fps,
            chunk_duration=config.chunk_duration,
            on_chunk_created=self._on_chunk_created
        )
        
        # State tracking
        self.stats = RecordingStats()
        self._is_running = False
        self._current_frame_id: Optional[str] = None
        
        # Track active windows for cleanup
        self._active_window_keys: set = set()
        
        logger.info(f"RecordingCoordinator initialized: monitor={config.monitor_id}")
    
    def _on_chunk_created(self, chunk_path: str, chunk_type: str, identifier: str):
        """Callback when a new video chunk is created"""
        logger.info(f"New {chunk_type} chunk created: {chunk_path}")
        
        # Insert chunk record to database
        if chunk_type == "screen":
            # Extract monitor_id from identifier (e.g., "monitor_0")
            monitor_id = int(identifier.split("_")[1]) if "_" in identifier else 0
            self.db.insert_video_chunk(
                file_path=chunk_path,
                monitor_id=monitor_id,
                device_name=identifier,
                fps=self.config.fps
            )
        else:  # window
            # Parse window key (e.g., "app::window::pid")
            parts = identifier.split("::")
            app_name = parts[0] if len(parts) > 0 else "unknown"
            window_name = parts[1] if len(parts) > 1 else "unknown"
            self.db.insert_window_chunk(
                file_path=chunk_path,
                app_name=app_name,
                window_name=window_name,
                monitor_id=self.config.monitor_id,
                fps=self.config.fps
            )
    
    def _generate_frame_id(self) -> str:
        """Generate a unique frame ID"""
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        return f"frame_{timestamp}_{uuid.uuid4().hex[:8]}"
    
    def _generate_sub_frame_id(self, app_name: str) -> str:
        """Generate a unique sub-frame ID"""
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        safe_app = app_name.replace(" ", "_").replace("/", "_")[:20]
        return f"subframe_{safe_app}_{timestamp}_{uuid.uuid4().hex[:8]}"
    
    async def _run_ocr(self, image: Image.Image) -> tuple:
        """
        Run OCR on an image
        
        Returns:
            Tuple of (text, text_json, confidence)
        """
        if not self.config.run_ocr or self.ocr_engine is None:
            return "", "", 0.0
        
        try:
            # OCR engine interface - adapt based on your implementation
            if hasattr(self.ocr_engine, 'extract_text_async'):
                result = await self.ocr_engine.extract_text_async(image)
            elif hasattr(self.ocr_engine, 'extract_text'):
                result = self.ocr_engine.extract_text(image)
            else:
                return "", "", 0.0
            
            if isinstance(result, dict):
                return (
                    result.get('text', ''),
                    result.get('text_json', ''),
                    result.get('confidence', 0.0)
                )
            elif isinstance(result, str):
                return result, "", 0.0
            
            return "", "", 0.0
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            self.stats.errors += 1
            return "", "", 0.0
    
    async def _run_embedding(self, image: Image.Image, text: str) -> Optional[List[float]]:
        """
        Generate embedding for an image/text
        
        Returns:
            Embedding vector or None
        """
        if not self.config.run_embedding or self.embedding_encoder is None:
            return None
        
        try:
            # Embedding encoder interface - adapt based on your implementation
            if hasattr(self.embedding_encoder, 'encode_image_async'):
                embedding = await self.embedding_encoder.encode_image_async(image)
            elif hasattr(self.embedding_encoder, 'encode_image'):
                embedding = self.embedding_encoder.encode_image(image)
            else:
                return None
            
            self.stats.embeddings_generated += 1
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            self.stats.errors += 1
            return None
    
    async def process_capture(self, screen_obj: ScreenObject) -> Dict[str, Any]:
        """
        Process a single capture: check diffs, store, OCR, embed
        
        Args:
            screen_obj: Captured screen object
            
        Returns:
            Processing result with frame_id, sub_frame_ids, etc.
        """
        result = {
            "frame_id": None,
            "frame_stored": False,
            "sub_frame_ids": [],
            "sub_frames_stored": 0,
            "timestamp": screen_obj.timestamp
        }
        
        self.stats.frames_captured += 1
        
        # 1. Check screen frame diff
        screen_diff_result = self.frame_diff.check_screen_diff(screen_obj)
        
        # 2. Track active windows
        current_window_keys = set()
        for window in screen_obj.windows:
            key = f"{window.app_name}::{window.window_name}::{window.process_id}"
            current_window_keys.add(key)
        
        # Cleanup stale window writers
        self.video_manager.cleanup_inactive_windows(current_window_keys)
        self._active_window_keys = current_window_keys
        
        # 3. Process screen if changed
        if screen_diff_result.should_store:
            frame_id = self._generate_frame_id()
            self._current_frame_id = frame_id
            
            # Write to video chunk
            chunk_result = self.video_manager.write_screen_frame(
                screen_obj.monitor_id,
                screen_obj.full_screen_image
            )
            
            if chunk_result:
                chunk_path, offset_index = chunk_result
                
                # Get chunk_id from database
                video_chunk_id = self._get_latest_video_chunk_id()
                
                # Store frame metadata
                self.db.store_frame_with_video_ref(
                    frame_id=frame_id,
                    timestamp=screen_obj.timestamp,
                    video_chunk_id=video_chunk_id,
                    offset_index=offset_index,
                    monitor_id=screen_obj.monitor_id,
                    image_hash=screen_obj.full_screen_hash,
                    device_name=screen_obj.device_name
                )
                
                # OCR (optional)
                if self.config.run_ocr and self.ocr_engine:
                    ocr_text, ocr_json, confidence = await self._run_ocr(
                        screen_obj.full_screen_image
                    )
                    if ocr_text:
                        self.db.store_frame_with_ocr(
                            frame_id=frame_id,
                            timestamp=screen_obj.timestamp,
                            image_path=f"video_chunk:{video_chunk_id}:{offset_index}",
                            ocr_text=ocr_text,
                            ocr_text_json=ocr_json,
                            ocr_confidence=confidence,
                            device_name=screen_obj.device_name
                        )
                        self.stats.ocr_processed += 1
                
                result["frame_id"] = frame_id
                result["frame_stored"] = True
                self.stats.frames_stored += 1
                
                # Callback
                if self.on_frame_stored:
                    self.on_frame_stored(frame_id, {
                        "timestamp": screen_obj.timestamp,
                        "monitor_id": screen_obj.monitor_id,
                        "diff_score": screen_diff_result.diff_score
                    })
                
                logger.debug(f"Stored frame {frame_id} (diff={screen_diff_result.diff_score:.4f})")
        
        # 4. Process each window independently
        sub_frame_ids = []
        for window in screen_obj.windows:
            self.stats.windows_captured += 1
            
            window_diff_result = self.frame_diff.check_window_diff(window)
            
            if window_diff_result.should_store:
                sub_frame_id = self._generate_sub_frame_id(window.app_name)
                
                # Write to video chunk
                chunk_result = self.video_manager.write_window_frame(
                    window.app_name,
                    window.window_name,
                    window.process_id,
                    window.image
                )
                
                if chunk_result:
                    chunk_path, offset_index = chunk_result
                    
                    # Get chunk_id from database
                    window_chunk_id = self._get_latest_window_chunk_id(
                        window.app_name,
                        window.window_name
                    )
                    
                    # Store sub-frame metadata
                    self.db.store_sub_frame(
                        sub_frame_id=sub_frame_id,
                        timestamp=screen_obj.timestamp,
                        window_chunk_id=window_chunk_id,
                        offset_index=offset_index,
                        app_name=window.app_name,
                        window_name=window.window_name,
                        process_id=window.process_id,
                        is_focused=window.is_focused,
                        image_hash=window.image_hash
                    )
                    
                    # OCR (optional)
                    if self.config.run_ocr and self.ocr_engine:
                        ocr_text, ocr_json, confidence = await self._run_ocr(window.image)
                        if ocr_text:
                            self.db.store_sub_frame_ocr(
                                sub_frame_id=sub_frame_id,
                                ocr_text=ocr_text,
                                ocr_text_json=ocr_json,
                                ocr_confidence=confidence
                            )
                            self.stats.ocr_processed += 1
                    
                    sub_frame_ids.append(sub_frame_id)
                    self.stats.windows_stored += 1
                    
                    # Callback
                    if self.on_subframe_stored:
                        self.on_subframe_stored(sub_frame_id, {
                            "timestamp": screen_obj.timestamp,
                            "app_name": window.app_name,
                            "window_name": window.window_name,
                            "diff_score": window_diff_result.diff_score
                        })
                    
                    logger.debug(
                        f"Stored sub_frame {sub_frame_id} "
                        f"({window.app_name}/{window.window_name})"
                    )
        
        # 5. Create frame-subframe mappings
        if result["frame_id"] and sub_frame_ids:
            self.db.add_frame_subframe_mappings_batch(result["frame_id"], sub_frame_ids)
        
        result["sub_frame_ids"] = sub_frame_ids
        result["sub_frames_stored"] = len(sub_frame_ids)
        
        return result
    
    def _get_latest_video_chunk_id(self) -> int:
        """Get the ID of the most recent video chunk"""
        try:
            conn = self.db._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM video_chunks 
                WHERE monitor_id = ?
                ORDER BY id DESC LIMIT 1
            """, (self.config.monitor_id,))
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else 0
        except Exception as e:
            logger.error(f"Failed to get latest video chunk id: {e}")
            return 0
    
    def _get_latest_window_chunk_id(self, app_name: str, window_name: str) -> int:
        """Get the ID of the most recent window chunk for an app/window"""
        try:
            conn = self.db._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM window_chunks 
                WHERE app_name = ? AND window_name = ?
                ORDER BY id DESC LIMIT 1
            """, (app_name, window_name))
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else 0
        except Exception as e:
            logger.error(f"Failed to get latest window chunk id: {e}")
            return 0
    
    async def capture_and_process(self) -> Optional[Dict[str, Any]]:
        """
        Perform one capture cycle: capture screen/windows and process
        
        Returns:
            Processing result or None if capture failed
        """
        screen_obj = self.capturer.capture_screen_with_windows()
        if screen_obj is None:
            self.stats.errors += 1
            return None
        
        return await self.process_capture(screen_obj)
    
    async def run_continuous(
        self,
        interval: float = None,
        max_iterations: Optional[int] = None
    ):
        """
        Run continuous capture loop
        
        Args:
            interval: Capture interval in seconds (defaults to 1/fps)
            max_iterations: Maximum iterations (None for infinite)
        """
        if interval is None:
            interval = 1.0 / self.config.fps
        
        self._is_running = True
        iteration = 0
        
        logger.info(
            f"Starting continuous capture: interval={interval}s, "
            f"max_iterations={max_iterations}"
        )
        
        try:
            while self._is_running:
                if max_iterations is not None and iteration >= max_iterations:
                    break
                
                start_time = datetime.datetime.now()
                
                result = await self.capture_and_process()
                
                if result and result.get("frame_stored"):
                    logger.debug(
                        f"Iteration {iteration}: stored frame {result['frame_id']}, "
                        f"{result['sub_frames_stored']} sub-frames"
                    )
                
                iteration += 1
                
                # Calculate sleep time
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                sleep_time = max(0, interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
        except asyncio.CancelledError:
            logger.info("Continuous capture cancelled")
        except Exception as e:
            logger.error(f"Continuous capture error: {e}")
            self.stats.errors += 1
        finally:
            self._is_running = False
            self.stop()
    
    def stop(self):
        """Stop recording and cleanup"""
        self._is_running = False
        self.video_manager.close_all()
        
        # Log final stats
        runtime = (
            datetime.datetime.now(datetime.timezone.utc) - self.stats.start_time
        ).total_seconds()
        
        logger.info(
            f"Recording stopped. Stats: "
            f"frames_captured={self.stats.frames_captured}, "
            f"frames_stored={self.stats.frames_stored}, "
            f"windows_captured={self.stats.windows_captured}, "
            f"windows_stored={self.stats.windows_stored}, "
            f"ocr_processed={self.stats.ocr_processed}, "
            f"errors={self.stats.errors}, "
            f"runtime={runtime:.1f}s"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current recording statistics"""
        runtime = (
            datetime.datetime.now(datetime.timezone.utc) - self.stats.start_time
        ).total_seconds()
        
        return {
            "frames_captured": self.stats.frames_captured,
            "frames_stored": self.stats.frames_stored,
            "windows_captured": self.stats.windows_captured,
            "windows_stored": self.stats.windows_stored,
            "ocr_processed": self.stats.ocr_processed,
            "embeddings_generated": self.stats.embeddings_generated,
            "errors": self.stats.errors,
            "runtime_seconds": runtime,
            "fps_actual": self.stats.frames_captured / runtime if runtime > 0 else 0,
            "is_running": self._is_running
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
