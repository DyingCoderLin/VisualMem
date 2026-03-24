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
from utils.app_name_manager import app_name_manager
from config import config

from .window_capturer import WindowCapturer
from ..preprocess.frame_diff import FrameDiffDetector, FrameDiffResult, is_solid_color_image
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
    use_region_ocr: bool = True
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
        on_subframe_stored: Optional[Callable[[str, dict], None]] = None,
        region_detector: Optional[Any] = None,
    ):
        """
        Args:
            config: Recording configuration
            db: SQLite storage instance (created if not provided)
            ocr_engine: OCR engine instance (optional)
            embedding_encoder: Embedding encoder instance (optional)
            on_frame_stored: Callback when a frame is stored
            on_subframe_stored: Callback when a sub-frame is stored
            region_detector: UIEDRegionDetector instance (optional, for region OCR)
        """
        self.config = config
        self.db = db or SQLiteStorage()
        self.ocr_engine = ocr_engine
        self.embedding_encoder = embedding_encoder
        self.on_frame_stored = on_frame_stored
        self.on_subframe_stored = on_subframe_stored
        self.region_detector = region_detector

        # Build RegionOCREngine if region OCR is enabled
        self.region_ocr_engine = None
        if config.use_region_ocr and ocr_engine is not None:
            from ..ocr.region_ocr_engine import RegionOCREngine
            self.region_ocr_engine = RegionOCREngine(
                ocr_engine=ocr_engine,
                region_detector=region_detector,
            )
        
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
        self._last_stored_frame_id: Optional[str] = None
        
        # chunk_id cache: populated by _on_chunk_created callback so we avoid
        # the race where we query the DB before the INSERT is committed.
        self._latest_video_chunk_ids: Dict[int, int] = {}       # monitor_id -> chunk_id
        self._latest_window_chunk_ids: Dict[str, int] = {}      # "app::window" -> chunk_id
        
        # Track active windows for cleanup
        self._active_window_keys: set = set()
        
        logger.info(f"RecordingCoordinator initialized: monitor={config.monitor_id}")
    
    def _on_chunk_created(self, chunk_path: str, chunk_type: str, identifier: str):
        """Callback when a new video chunk is created.
        
        Inserts the chunk record into the database and caches the returned
        chunk_id so that subsequent write_frame calls can reference it
        without a racy SELECT query.
        """
        logger.info(f"New {chunk_type} chunk created: {chunk_path}")
        
        if chunk_type == "screen":
            monitor_id = int(identifier.split("_")[1]) if "_" in identifier else 0
            chunk_id = self.db.insert_video_chunk(
                file_path=chunk_path,
                monitor_id=monitor_id,
                device_name=identifier,
                fps=self.config.fps
            )
            if chunk_id > 0:
                self._latest_video_chunk_ids[monitor_id] = chunk_id
        else:  # window
            parts = identifier.split("::")
            app_name = parts[0] if len(parts) > 0 else "unknown"
            window_name = parts[1] if len(parts) > 1 else "unknown"
            chunk_id = self.db.insert_window_chunk(
                file_path=chunk_path,
                app_name=app_name,
                window_name=window_name,
                monitor_id=self.config.monitor_id,
                fps=self.config.fps
            )
            if chunk_id > 0:
                key = f"{app_name}::{window_name}"
                self._latest_window_chunk_ids[key] = chunk_id
    
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
        Run OCR on an image using OCREngine.recognize() interface.

        Returns:
            Tuple of (text, text_json, confidence)
        """
        if not self.config.run_ocr or self.ocr_engine is None:
            return "", "", 0.0

        try:
            result = self.ocr_engine.recognize(image)
            return result.text, result.text_json, result.confidence
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            self.stats.errors += 1
            return "", "", 0.0
    
    def _run_region_ocr(self, image: Image.Image) -> List[dict]:
        """
        Run region-level OCR: detect regions + per-region OCR.
        Falls back to whole-image single region when region_detector is None.

        Returns:
            List of region dicts from RegionOCREngine.recognize_regions()
        """
        if self.region_ocr_engine is None:
            return []
        try:
            return self.region_ocr_engine.recognize_regions(image)
        except Exception as e:
            logger.error(f"Region OCR failed: {e}")
            self.stats.errors += 1
            return []

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
        focused_app = screen_obj.focused_app_name or ""
        focused_win = screen_obj.focused_window_name or ""
        
        result = {
            "frame_id": None,
            "frame_stored": False,
            "sub_frame_ids": [],
            "sub_frames_stored": 0,
            "timestamp": screen_obj.timestamp,
            "focused_app_name": focused_app,
            "focused_window_name": focused_win,
        }
        
        self.stats.frames_captured += 1
        
        # 0. Skip solid-color full screen (e.g. monitor off, lid closed)
        if is_solid_color_image(screen_obj.full_screen_image):
            logger.debug("Skipping solid-color full screen image")
            # Still process windows below (they might be valid)
            screen_diff_result = FrameDiffResult(
                should_store=False, diff_score=0.0,
                histogram_diff=0.0, ssim_diff=0.0,
                reason="Solid-color image skipped"
            )
        else:
            # 1. Check screen frame diff
            screen_diff_result = self.frame_diff.check_screen_diff(screen_obj)
        
        # 2. Track active windows and update app_name + window_name list
        current_window_keys = set()
        current_app_names = []
        app_window_pairs = []
        for window in screen_obj.windows:
            key = f"{window.app_name}::{window.window_name}"
            current_window_keys.add(key)
            if window.app_name:
                current_app_names.append(window.app_name)
                if window.window_name:
                    app_window_pairs.append((window.app_name, window.window_name))
        
        # Update app_name persistence
        if current_app_names:
            app_name_manager.add_apps(current_app_names)
        # Update window_name persistence grouped by app
        if app_window_pairs:
            app_name_manager.add_window_pairs(app_window_pairs)
        
        # Cleanup stale window writers
        self.video_manager.cleanup_inactive_windows(current_window_keys)
        self._active_window_keys = current_window_keys
        
        # Initialize sub_frame_ids early (may be populated by full-screen synthetic sub_frame)
        sub_frame_ids = []
        # Collector for sub_frame OCR results: [(app_name, text, confidence), ...]
        sub_frame_ocr_parts = []

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
                
                # Use cached chunk_id (populated by _on_chunk_created callback)
                video_chunk_id = self._latest_video_chunk_ids.get(
                    screen_obj.monitor_id,
                    self._get_latest_video_chunk_id()
                )
                
                # If the focused app is full-screen (not among captured windows),
                # tag this frame with its app_name so app-filtered searches find it.
                fullscreen_app = None
                fullscreen_win = None
                if focused_app:
                    focused_in_windows = any(
                        w.app_name == focused_app for w in screen_obj.windows
                    )
                    if not focused_in_windows:
                        fullscreen_app = focused_app
                        fullscreen_win = focused_win or focused_app

                # Store frame metadata
                self.db.store_frame_with_video_ref(
                    frame_id=frame_id,
                    timestamp=screen_obj.timestamp,
                    video_chunk_id=video_chunk_id,
                    offset_index=offset_index,
                    monitor_id=screen_obj.monitor_id,
                    device_name=screen_obj.device_name,
                    app_name=fullscreen_app,
                    window_name=fullscreen_win,
                    focused_app_name=focused_app or None,
                    focused_window_name=focused_win or None,
                )
                
                # Full-screen OCR is deferred — combined from sub_frame OCR results later.

                # If focused app is full-screen (not among captured windows),
                # create a synthetic sub_frame referencing the same video
                if fullscreen_app:
                    syn_id = self._generate_sub_frame_id(focused_app)
                    self.db.store_sub_frame(
                        sub_frame_id=syn_id,
                        timestamp=screen_obj.timestamp,
                        window_chunk_id=0,
                        offset_index=offset_index,
                        app_name=focused_app,
                        window_name=focused_win or focused_app,
                    )
                    self.db.add_frame_subframe_mapping(frame_id, syn_id)
                    self.db.store_frame_with_video_ref(
                        frame_id=syn_id,
                        timestamp=screen_obj.timestamp,
                        video_chunk_id=video_chunk_id,
                        offset_index=offset_index,
                        monitor_id=screen_obj.monitor_id,
                        device_name=screen_obj.device_name,
                        app_name=focused_app,
                        window_name=focused_win or focused_app,
                    )
                    # Region OCR for synthetic sub_frame
                    if self.config.run_ocr and self.region_ocr_engine is not None:
                        syn_regions = self._run_region_ocr(screen_obj.full_screen_image)
                        if syn_regions:
                            img_w, img_h = screen_obj.full_screen_image.size
                            engine_name = getattr(self.ocr_engine, 'engine_name', 'auto')
                            self.db.store_ocr_with_regions(
                                sub_frame_id=syn_id,
                                regions=syn_regions,
                                ocr_engine=engine_name,
                                image_width=img_w,
                                image_height=img_h,
                            )
                            syn_text = "\n".join(r.get("text", "") for r in syn_regions if r.get("text"))
                            syn_tl = sum(len(r.get("text", "")) for r in syn_regions)
                            syn_conf = sum(len(r.get("text", "")) * r.get("ocr_confidence", 0.0) for r in syn_regions) / syn_tl if syn_tl > 0 else 0.0
                            if syn_text:
                                sub_frame_ocr_parts.append((focused_app, syn_text, syn_conf))
                            self.stats.ocr_processed += 1
                    sub_frame_ids.append(syn_id)
                    logger.debug(
                        f"Created synthetic sub_frame {syn_id} for "
                        f"full-screen app {focused_app}"
                    )
                
                result["frame_id"] = frame_id
                result["frame_stored"] = True
                self._last_stored_frame_id = frame_id
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
        for window in screen_obj.windows:
            self.stats.windows_captured += 1

            # Skip solid-color window images
            if is_solid_color_image(window.image):
                logger.debug(
                    f"Skipping solid-color window: {window.app_name}/{window.window_name}"
                )
                continue

            window_diff_result = self.frame_diff.check_window_diff(window)
            
            if window_diff_result.should_store:
                sub_frame_id = self._generate_sub_frame_id(window.app_name)
                
                # Write to video chunk
                chunk_result = self.video_manager.write_window_frame(
                    window.app_name,
                    window.window_name,
                    window.image
                )
                
                if chunk_result:
                    chunk_path, offset_index = chunk_result
                    
                    # Use cached chunk_id (populated by _on_chunk_created callback)
                    win_key = f"{window.app_name}::{window.window_name}"
                    window_chunk_id = self._latest_window_chunk_ids.get(
                        win_key,
                        self._get_latest_window_chunk_id(window.app_name, window.window_name)
                    )
                    
                    # Store sub-frame metadata
                    self.db.store_sub_frame(
                        sub_frame_id=sub_frame_id,
                        timestamp=screen_obj.timestamp,
                        window_chunk_id=window_chunk_id,
                        offset_index=offset_index,
                        app_name=window.app_name,
                        window_name=window.window_name
                    )
                    
                    # OCR (optional) — prefer region OCR, fallback to whole-image
                    win_ocr_text = ""
                    win_ocr_conf = 0.0
                    if self.config.run_ocr and self.ocr_engine:
                        if self.region_ocr_engine is not None:
                            regions = self._run_region_ocr(window.image)
                            if regions:
                                img_w, img_h = window.image.size
                                engine_name = getattr(self.ocr_engine, 'engine_name', 'auto')
                                self.db.store_ocr_with_regions(
                                    sub_frame_id=sub_frame_id,
                                    regions=regions,
                                    ocr_engine=engine_name,
                                    image_width=img_w,
                                    image_height=img_h,
                                )
                                win_ocr_text = "\n".join(r.get("text", "") for r in regions if r.get("text"))
                                tl = sum(len(r.get("text", "")) for r in regions)
                                if tl > 0:
                                    win_ocr_conf = sum(len(r.get("text", "")) * r.get("ocr_confidence", 0.0) for r in regions) / tl
                                self.stats.ocr_processed += 1
                        else:
                            ocr_text, ocr_json, confidence = await self._run_ocr(window.image)
                            if ocr_text:
                                self.db.store_sub_frame_ocr(
                                    sub_frame_id=sub_frame_id,
                                    ocr_text=ocr_text,
                                    ocr_text_json=ocr_json,
                                    ocr_confidence=confidence,
                                )
                                win_ocr_text = ocr_text
                                win_ocr_conf = confidence
                                self.stats.ocr_processed += 1

                    # Collect for frame-level combined OCR
                    if win_ocr_text:
                        sub_frame_ocr_parts.append((window.app_name, win_ocr_text, win_ocr_conf))
                    
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
        mapping_frame_id = result["frame_id"] or self._last_stored_frame_id
        if mapping_frame_id and sub_frame_ids:
            self.db.add_frame_subframe_mappings_batch(mapping_frame_id, sub_frame_ids)

        # 6. Combine sub_frame OCR texts as the frame's ocr_text
        if result.get("frame_stored") and sub_frame_ocr_parts:
            sections = [f"[{app}]\n{text}" for app, text, _ in sub_frame_ocr_parts]
            combined_text = "\n\n".join(sections)
            total_len = sum(len(t) for _, t, _ in sub_frame_ocr_parts)
            combined_conf = sum(len(t) * c for _, t, c in sub_frame_ocr_parts) / total_len if total_len > 0 else 0.0
            engine_name = getattr(self.ocr_engine, 'engine_name', 'auto') if self.ocr_engine else "unknown"
            self.db.store_frame_with_ocr(
                frame_id=result["frame_id"],
                timestamp=screen_obj.timestamp,
                image_path="",
                ocr_text=combined_text,
                ocr_text_json="",
                ocr_engine=engine_name,
                ocr_confidence=combined_conf,
                device_name=screen_obj.device_name,
                focused_app_name=focused_app or None,
                focused_window_name=focused_win or None,
            )

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
