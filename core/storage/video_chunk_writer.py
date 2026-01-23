# core/storage/video_chunk_writer.py
"""
Video Chunk Writer Module

Writes frames to MP4 video files for efficient storage.
This is purely for storage compression - embedding is still done per-frame.
FFmpeg is used to extract frames when querying.

Reference: screenpipe's video.rs
"""
import os
import io
import subprocess
import datetime
from typing import Optional, Dict, Callable
from pathlib import Path
from PIL import Image
from threading import Lock
from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__)

# Default settings
DEFAULT_FPS = 1.0  # 1 frame per second for screenshot-like content
DEFAULT_CHUNK_DURATION = 60  # 60 seconds per chunk
MAX_FPS = 30.0


def find_ffmpeg_path() -> Optional[str]:
    """Find FFmpeg executable path"""
    try:
        result = subprocess.run(
            ["which", "ffmpeg"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    
    # Try common paths
    common_paths = [
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None


class VideoChunkWriter:
    """
    Writes frames to MP4 video chunks using FFmpeg
    
    Manages the lifecycle of video chunks:
    - Start new chunk when needed
    - Write frames to current chunk via FFmpeg stdin pipe
    - Finish chunk when duration reached or explicitly closed
    
    This is for storage only - frames are still processed individually
    for OCR and embedding.
    """
    
    def __init__(
        self,
        output_dir: str,
        chunk_type: str,  # "screen" or "window"
        identifier: str,  # e.g., "monitor_0" or "firefox::tab1::12345"
        fps: float = DEFAULT_FPS,
        chunk_duration: int = DEFAULT_CHUNK_DURATION,
        on_chunk_created: Optional[Callable[[str], None]] = None
    ):
        """
        Args:
            output_dir: Directory to store video chunks
            chunk_type: Type of chunk ("screen" or "window")
            identifier: Unique identifier for this stream
            fps: Frames per second
            chunk_duration: Duration of each chunk in seconds
            on_chunk_created: Callback when a new chunk file is created
        """
        self.output_dir = Path(output_dir)
        self.chunk_type = chunk_type
        self.identifier = identifier
        self.fps = min(fps if fps > 0 else DEFAULT_FPS, MAX_FPS)
        self.chunk_duration = chunk_duration
        self.on_chunk_created = on_chunk_created
        
        # FFmpeg process management
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.current_chunk_path: Optional[str] = None
        self.frame_count: int = 0
        self.frames_per_chunk: int = int(self.fps * self.chunk_duration)
        
        # Thread safety
        self._lock = Lock()
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find FFmpeg
        self.ffmpeg_path = find_ffmpeg_path()
        if not self.ffmpeg_path:
            logger.error("FFmpeg not found. Video chunk writing will not work.")
        
        logger.info(
            f"VideoChunkWriter initialized: {chunk_type}/{identifier}, "
            f"fps={self.fps}, chunk_duration={chunk_duration}s"
        )
    
    def _generate_chunk_filename(self) -> str:
        """Generate a unique filename for the chunk"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Sanitize identifier for filename
        safe_id = self.identifier.replace("/", "_").replace(":", "_").replace(" ", "_")
        return f"{self.chunk_type}_{safe_id}_{timestamp}.mp4"
    
    def _start_ffmpeg_process(self) -> bool:
        """Start a new FFmpeg process for writing"""
        if not self.ffmpeg_path:
            logger.error("FFmpeg not available")
            return False
        
        # Generate new chunk path
        filename = self._generate_chunk_filename()
        self.current_chunk_path = str(self.output_dir / filename)
        
        # Build FFmpeg command
        # Input: PNG images via pipe
        # Output: H.265/HEVC encoded MP4
        cmd = [
            self.ffmpeg_path,
            "-y",  # Overwrite output
            "-f", "image2pipe",
            "-vcodec", "png",
            "-r", str(self.fps),
            "-i", "-",  # Read from stdin
            # Pad to even dimensions (required for H.265)
            "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
            # H.265 encoding
            "-vcodec", "libx265",
            "-tag:v", "hvc1",
            "-preset", "ultrafast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            self.current_chunk_path
        ]
        
        try:
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.frame_count = 0
            
            logger.info(f"Started new video chunk: {self.current_chunk_path}")
            
            # Call callback if provided
            if self.on_chunk_created:
                self.on_chunk_created(self.current_chunk_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start FFmpeg process: {e}")
            self.ffmpeg_process = None
            return False
    
    def _finish_ffmpeg_process(self):
        """Finish the current FFmpeg process"""
        if self.ffmpeg_process is None:
            return
        
        try:
            # Close stdin to signal end of input
            if self.ffmpeg_process.stdin:
                self.ffmpeg_process.stdin.close()
            
            # Wait for process to complete
            stdout, stderr = self.ffmpeg_process.communicate(timeout=30)
            
            if self.ffmpeg_process.returncode != 0:
                logger.warning(f"FFmpeg finished with non-zero exit: {stderr.decode()}")
            else:
                logger.info(
                    f"Finished video chunk: {self.current_chunk_path} "
                    f"({self.frame_count} frames)"
                )
                
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg process timed out, killing...")
            self.ffmpeg_process.kill()
        except Exception as e:
            logger.error(f"Error finishing FFmpeg process: {e}")
        finally:
            self.ffmpeg_process = None
    
    def write_frame(self, image: Image.Image) -> Optional[int]:
        """
        Write a frame to the current video chunk
        
        Args:
            image: PIL Image to write
            
        Returns:
            The offset_index of this frame in the current chunk, or None if failed
        """
        with self._lock:
            # Check if we need to start a new chunk
            if self.ffmpeg_process is None or self.frame_count >= self.frames_per_chunk:
                # Finish current chunk if exists
                if self.ffmpeg_process is not None:
                    self._finish_ffmpeg_process()
                
                # Start new chunk
                if not self._start_ffmpeg_process():
                    return None
            
            # Encode frame as PNG
            try:
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                png_data = buffer.getvalue()
                
                # Write to FFmpeg stdin
                self.ffmpeg_process.stdin.write(png_data)
                self.ffmpeg_process.stdin.flush()
                
                offset_index = self.frame_count
                self.frame_count += 1
                
                return offset_index
                
            except BrokenPipeError:
                logger.error("FFmpeg pipe broken, restarting...")
                self._finish_ffmpeg_process()
                return None
            except Exception as e:
                logger.error(f"Failed to write frame: {e}")
                return None
    
    def get_current_chunk_path(self) -> Optional[str]:
        """Get the path to the current chunk being written"""
        return self.current_chunk_path
    
    def get_frame_count(self) -> int:
        """Get number of frames written to current chunk"""
        return self.frame_count
    
    def close(self):
        """Close the writer and finish any pending chunk"""
        with self._lock:
            if self.ffmpeg_process is not None:
                self._finish_ffmpeg_process()
        
        logger.info(f"VideoChunkWriter closed: {self.chunk_type}/{self.identifier}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class VideoChunkManager:
    """
    Manages multiple VideoChunkWriters for different streams
    
    Handles:
    - One writer per monitor (for full screen)
    - One writer per active window
    """
    
    def __init__(
        self,
        output_dir: str,
        fps: float = DEFAULT_FPS,
        chunk_duration: int = DEFAULT_CHUNK_DURATION,
        on_chunk_created: Optional[Callable[[str, str, str], None]] = None
    ):
        """
        Args:
            output_dir: Base directory for video storage
            fps: Frames per second
            chunk_duration: Duration of each chunk
            on_chunk_created: Callback(chunk_path, chunk_type, identifier)
        """
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.chunk_duration = chunk_duration
        self.on_chunk_created = on_chunk_created
        
        # Writers keyed by identifier
        self.screen_writers: Dict[int, VideoChunkWriter] = {}  # monitor_id -> writer
        self.window_writers: Dict[str, VideoChunkWriter] = {}  # window_key -> writer
        
        self._lock = Lock()
        
        logger.info(f"VideoChunkManager initialized: output_dir={output_dir}")
    
    def _make_callback(self, chunk_type: str, identifier: str) -> Callable[[str], None]:
        """Create a callback wrapper for a specific stream"""
        def callback(chunk_path: str):
            if self.on_chunk_created:
                self.on_chunk_created(chunk_path, chunk_type, identifier)
        return callback
    
    def get_screen_writer(self, monitor_id: int) -> VideoChunkWriter:
        """Get or create a writer for a monitor's full screen"""
        with self._lock:
            if monitor_id not in self.screen_writers:
                identifier = f"monitor_{monitor_id}"
                self.screen_writers[monitor_id] = VideoChunkWriter(
                    output_dir=str(self.output_dir / "screens"),
                    chunk_type="screen",
                    identifier=identifier,
                    fps=self.fps,
                    chunk_duration=self.chunk_duration,
                    on_chunk_created=self._make_callback("screen", identifier)
                )
            return self.screen_writers[monitor_id]
    
    def get_window_writer(
        self,
        app_name: str,
        window_name: str,
        process_id: int
    ) -> VideoChunkWriter:
        """Get or create a writer for a specific window"""
        window_key = f"{app_name}::{window_name}::{process_id}"
        
        with self._lock:
            if window_key not in self.window_writers:
                self.window_writers[window_key] = VideoChunkWriter(
                    output_dir=str(self.output_dir / "windows"),
                    chunk_type="window",
                    identifier=window_key,
                    fps=self.fps,
                    chunk_duration=self.chunk_duration,
                    on_chunk_created=self._make_callback("window", window_key)
                )
            return self.window_writers[window_key]
    
    def write_screen_frame(
        self,
        monitor_id: int,
        image: Image.Image
    ) -> Optional[tuple]:
        """
        Write a screen frame
        
        Returns:
            Tuple of (chunk_path, offset_index) or None if failed
        """
        writer = self.get_screen_writer(monitor_id)
        offset = writer.write_frame(image)
        if offset is not None:
            return (writer.get_current_chunk_path(), offset)
        return None
    
    def write_window_frame(
        self,
        app_name: str,
        window_name: str,
        process_id: int,
        image: Image.Image
    ) -> Optional[tuple]:
        """
        Write a window frame
        
        Returns:
            Tuple of (chunk_path, offset_index) or None if failed
        """
        writer = self.get_window_writer(app_name, window_name, process_id)
        offset = writer.write_frame(image)
        if offset is not None:
            return (writer.get_current_chunk_path(), offset)
        return None
    
    def cleanup_inactive_windows(self, active_keys: set):
        """Close writers for windows that are no longer active"""
        with self._lock:
            inactive_keys = set(self.window_writers.keys()) - active_keys
            for key in inactive_keys:
                self.window_writers[key].close()
                del self.window_writers[key]
            
            if inactive_keys:
                logger.debug(f"Cleaned up {len(inactive_keys)} inactive window writers")
    
    def close_all(self):
        """Close all writers"""
        with self._lock:
            for writer in self.screen_writers.values():
                writer.close()
            for writer in self.window_writers.values():
                writer.close()
            
            self.screen_writers.clear()
            self.window_writers.clear()
        
        logger.info("All VideoChunkWriters closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()
        return False
