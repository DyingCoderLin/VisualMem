# core/storage/temp_frame_buffer.py
"""
Temporary Frame Buffer Module

Manages temporary storage of frames before they are compressed into MP4 video files.
Frames are stored as PNG files in a temp directory, then batch-compressed when:
1. Buffer reaches threshold (default: 60 frames)
2. Recording is stopped
3. Manual flush is triggered

Storage structure:
- visualmem_storage/temp_frames/
  - full_screen/
    - monitor_{id}/
      - {timestamp}.png
  - windows/
    - {app_name}_{window_name}/
      - {timestamp}.png
"""
import os
import shutil
import datetime
import threading
from typing import Optional, Dict, List, Callable, Tuple
from pathlib import Path
from PIL import Image
from dataclasses import dataclass, field
from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__)

# Default settings
DEFAULT_BATCH_SIZE = 60  # Compress every 60 frames
DEFAULT_FPS = 1.0


@dataclass
class FrameInfo:
    """Information about a buffered frame"""
    frame_id: str
    timestamp: datetime.datetime
    image_path: str
    monitor_id: int = 0
    app_name: Optional[str] = None
    window_name: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    parent_frame_id: Optional[str] = None  # 窗口帧关联的全屏帧ID


@dataclass
class BufferStats:
    """Statistics for the frame buffer"""
    full_screen_count: int = 0
    window_counts: Dict[str, int] = field(default_factory=dict)
    total_frames: int = 0
    compressions_triggered: int = 0


class TempFrameBuffer:
    """
    Temporary frame buffer for batching frames before video compression
    
    Features:
    - Stores frames as PNG files in temp directory
    - Tracks frame metadata for database insertion
    - Triggers compression when batch size is reached
    - Thread-safe operations
    """
    
    def __init__(
        self,
        storage_root: str = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        fps: float = DEFAULT_FPS,
        on_batch_ready: Optional[Callable[[str, str, List[FrameInfo]], None]] = None
    ):
        """
        Args:
            storage_root: Root directory for storage (default: config.STORAGE_ROOT)
            batch_size: Number of frames before triggering compression
            fps: Target FPS for video encoding
            on_batch_ready: Callback when a batch is ready for compression
                           signature: (batch_type, identifier, frames) -> None
                           batch_type: "full_screen" or "window"
                           identifier: "monitor_{id}" or "{app_name}_{window_name}"
        """
        self.storage_root = Path(storage_root or config.STORAGE_ROOT)
        self.batch_size = batch_size
        self.fps = fps
        self.on_batch_ready = on_batch_ready
        
        # Temp directory structure
        self.temp_dir = self.storage_root / "temp_frames"
        self.full_screen_temp_dir = self.temp_dir / "full_screen"
        self.windows_temp_dir = self.temp_dir / "windows"
        
        # Video output directories (new structure)
        self.video_dir = self.storage_root / "visualmem_video"
        
        # Frame buffers (in-memory tracking)
        self._full_screen_buffers: Dict[int, List[FrameInfo]] = {}  # monitor_id -> frames
        self._window_buffers: Dict[str, List[FrameInfo]] = {}  # window_key -> frames
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = BufferStats()
        
        # Initialize directories
        self._init_directories()
        
        logger.info(
            f"TempFrameBuffer initialized: batch_size={batch_size}, "
            f"temp_dir={self.temp_dir}"
        )
    
    def _init_directories(self):
        """Create necessary directories"""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.full_screen_temp_dir.mkdir(parents=True, exist_ok=True)
        self.windows_temp_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for use in file paths"""
        # Replace problematic characters
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
            name = name.replace(char, '_')
        # Limit length
        return name[:50]
    
    def _get_full_screen_dir(self, monitor_id: int) -> Path:
        """Get temp directory for full screen frames of a monitor"""
        dir_path = self.full_screen_temp_dir / f"monitor_{monitor_id}"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    def _get_window_dir(self, app_name: str, window_name: str) -> Path:
        """Get temp directory for window frames"""
        safe_name = f"{self._sanitize_name(app_name)}_{self._sanitize_name(window_name)}"
        dir_path = self.windows_temp_dir / safe_name
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    def _get_video_output_path(
        self,
        batch_type: str,
        identifier: str,
        timestamp: datetime.datetime
    ) -> Path:
        """
        Get output path for compressed video
        
        Full screen: visualmem_storage/visualmem_video/full_screen_<monitor_id>/<date>/
        Windows: visualmem_storage/visualmem_video/<date>/<window_name>/
        """
        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%H-%M-%S")
        
        if batch_type == "full_screen":
            # Format: visualmem_video/full_screen_<monitor_id>/<date>/monitor_<id>_<timestamp>.mp4
            output_dir = self.video_dir / f"full_screen_{identifier}" / date_str
            output_dir.mkdir(parents=True, exist_ok=True)
            return (output_dir / f"{identifier}_{time_str}.mp4").resolve()  # 绝对路径
        else:
            # Format: visualmem_video/<date>/<window_name>/monitor_<id>_<timestamp>.mp4
            # identifier format: "{app_name}_{window_name}"
            output_dir = self.video_dir / date_str / identifier
            output_dir.mkdir(parents=True, exist_ok=True)
            return (output_dir / f"monitor_0_{time_str}.mp4").resolve()  # 绝对路径
    
    def add_full_screen_frame(
        self,
        frame_id: str,
        image: Image.Image,
        timestamp: datetime.datetime,
        monitor_id: int = 0,
        metadata: Optional[Dict] = None
    ) -> Tuple[str, bool]:
        """
        Add a full screen frame to the buffer
        
        Args:
            frame_id: Unique frame identifier
            image: PIL Image
            timestamp: Capture timestamp
            monitor_id: Monitor identifier
            metadata: Additional metadata
            
        Returns:
            Tuple of (image_path, batch_ready)
            batch_ready is True if compression should be triggered
        """
        with self._lock:
            # Save frame to temp directory
            temp_dir = self._get_full_screen_dir(monitor_id)
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
            image_path = (temp_dir / f"{timestamp_str}.png").resolve()  # 使用绝对路径
            image.save(str(image_path), format='PNG')
            
            # Create frame info
            frame_info = FrameInfo(
                frame_id=frame_id,
                timestamp=timestamp,
                image_path=str(image_path),  # 绝对路径
                monitor_id=monitor_id,
                metadata=metadata or {}
            )
            
            # Add to buffer
            if monitor_id not in self._full_screen_buffers:
                self._full_screen_buffers[monitor_id] = []
            self._full_screen_buffers[monitor_id].append(frame_info)
            
            # Update stats
            self.stats.full_screen_count += 1
            self.stats.total_frames += 1
            
            # Check if batch is ready
            batch_ready = len(self._full_screen_buffers[monitor_id]) >= self.batch_size
            
            logger.debug(
                f"Added full_screen frame: monitor={monitor_id}, "
                f"buffer_size={len(self._full_screen_buffers[monitor_id])}, "
                f"batch_ready={batch_ready}"
            )
            
            return str(image_path), batch_ready
    
    def add_window_frame(
        self,
        sub_frame_id: str,
        image: Image.Image,
        timestamp: datetime.datetime,
        app_name: str,
        window_name: str,
        metadata: Optional[Dict] = None,
        parent_frame_id: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Add a window frame to the buffer
        
        Args:
            sub_frame_id: 子帧ID
            image: PIL Image
            timestamp: 时间戳
            app_name: 应用名称
            window_name: 窗口名称
            metadata: 元数据
            parent_frame_id: 关联的全屏帧ID
        
        Returns:
            Tuple of (image_path, batch_ready)
        """
        with self._lock:
            # Create window key
            window_key = f"{self._sanitize_name(app_name)}_{self._sanitize_name(window_name)}"
            
            # Save frame to temp directory
            temp_dir = self._get_window_dir(app_name, window_name)
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
            image_path = (temp_dir / f"{timestamp_str}.png").resolve()  # 使用绝对路径
            image.save(str(image_path), format='PNG')
            
            # Create frame info
            frame_info = FrameInfo(
                frame_id=sub_frame_id,
                timestamp=timestamp,
                image_path=str(image_path),  # 绝对路径
                app_name=app_name,
                window_name=window_name,
                metadata=metadata or {},
                parent_frame_id=parent_frame_id
            )
            
            # Add to buffer
            if window_key not in self._window_buffers:
                self._window_buffers[window_key] = []
            self._window_buffers[window_key].append(frame_info)
            
            # Update stats
            if window_key not in self.stats.window_counts:
                self.stats.window_counts[window_key] = 0
            self.stats.window_counts[window_key] += 1
            self.stats.total_frames += 1
            
            # Check if batch is ready
            batch_ready = len(self._window_buffers[window_key]) >= self.batch_size
            
            logger.debug(
                f"Added window frame: {window_key}, "
                f"buffer_size={len(self._window_buffers[window_key])}, "
                f"batch_ready={batch_ready}"
            )
            
            return str(image_path), batch_ready
    
    def get_ready_batches(self) -> List[Tuple[str, str, List[FrameInfo]]]:
        """
        Get all batches that are ready for compression
        
        Returns:
            List of (batch_type, identifier, frames) tuples
        """
        ready_batches = []
        
        with self._lock:
            # Check full screen batches
            for monitor_id, frames in self._full_screen_buffers.items():
                if len(frames) >= self.batch_size:
                    ready_batches.append(
                        ("full_screen", f"monitor_{monitor_id}", frames.copy())
                    )
            
            # Check window batches
            for window_key, frames in self._window_buffers.items():
                if len(frames) >= self.batch_size:
                    ready_batches.append(("window", window_key, frames.copy()))
        
        return ready_batches
    
    def flush_batch(self, batch_type: str, identifier: str) -> List[FrameInfo]:
        """
        Flush and return frames from a specific batch
        
        Args:
            batch_type: "full_screen" or "window"
            identifier: "monitor_{id}" or window_key
            
        Returns:
            List of FrameInfo that were flushed
        """
        with self._lock:
            if batch_type == "full_screen":
                # Extract monitor_id from identifier
                monitor_id = int(identifier.split("_")[1])
                frames = self._full_screen_buffers.pop(monitor_id, [])
            else:
                frames = self._window_buffers.pop(identifier, [])
            
            if frames:
                self.stats.compressions_triggered += 1
            
            return frames
    
    def flush_all(self) -> List[Tuple[str, str, List[FrameInfo]]]:
        """
        Flush all buffers (used when stopping recording)
        
        Returns:
            List of (batch_type, identifier, frames) tuples
        """
        all_batches = []
        
        with self._lock:
            # Flush all full screen batches
            for monitor_id, frames in list(self._full_screen_buffers.items()):
                if frames:
                    all_batches.append(
                        ("full_screen", f"monitor_{monitor_id}", frames.copy())
                    )
            self._full_screen_buffers.clear()
            
            # Flush all window batches
            for window_key, frames in list(self._window_buffers.items()):
                if frames:
                    all_batches.append(("window", window_key, frames.copy()))
            self._window_buffers.clear()
            
            self.stats.compressions_triggered += len(all_batches)
        
        logger.info(f"Flushed {len(all_batches)} batches from buffer")
        return all_batches
    
    def cleanup_batch_files(self, frames: List[FrameInfo]):
        """
        Clean up temporary files after successful compression
        
        Args:
            frames: List of FrameInfo whose temp files should be deleted
        """
        for frame in frames:
            try:
                if os.path.exists(frame.image_path):
                    os.remove(frame.image_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {frame.image_path}: {e}")
    
    def cleanup_empty_dirs(self):
        """Remove empty directories in temp folder"""
        try:
            for subdir in [self.full_screen_temp_dir, self.windows_temp_dir]:
                for dirpath, dirnames, filenames in os.walk(subdir, topdown=False):
                    if not dirnames and not filenames:
                        try:
                            os.rmdir(dirpath)
                        except OSError:
                            pass
        except Exception as e:
            logger.warning(f"Error cleaning up empty directories: {e}")
    
    def get_stats(self) -> Dict:
        """Get buffer statistics"""
        with self._lock:
            return {
                "full_screen_count": self.stats.full_screen_count,
                "window_counts": dict(self.stats.window_counts),
                "total_frames": self.stats.total_frames,
                "compressions_triggered": self.stats.compressions_triggered,
                "current_full_screen_buffers": {
                    f"monitor_{k}": len(v) 
                    for k, v in self._full_screen_buffers.items()
                },
                "current_window_buffers": {
                    k: len(v) for k, v in self._window_buffers.items()
                }
            }
    
    def get_buffer_sizes(self) -> Dict[str, int]:
        """Get current buffer sizes for all streams"""
        with self._lock:
            sizes = {}
            for monitor_id, frames in self._full_screen_buffers.items():
                sizes[f"full_screen_monitor_{monitor_id}"] = len(frames)
            for window_key, frames in self._window_buffers.items():
                sizes[f"window_{window_key}"] = len(frames)
            return sizes
