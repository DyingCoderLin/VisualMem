# core/storage/ffmpeg_utils.py
"""
FFmpeg Utilities Module

Provides utilities for:
1. Compressing frames into MP4 video
2. Extracting frames from MP4 video
3. Getting video metadata

Uses H.265/HEVC for efficient compression.
"""
import os
import io
import subprocess
import tempfile
from typing import Optional, List, Tuple
from pathlib import Path
from PIL import Image
from utils.logger import setup_logger

logger = setup_logger(__name__)


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


def find_ffprobe_path() -> Optional[str]:
    """Find FFprobe executable path"""
    try:
        result = subprocess.run(
            ["which", "ffprobe"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    
    # Try common paths
    common_paths = [
        "/usr/bin/ffprobe",
        "/usr/local/bin/ffprobe",
        "/opt/homebrew/bin/ffprobe",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None


# Cache paths
_FFMPEG_PATH = find_ffmpeg_path()
_FFPROBE_PATH = find_ffprobe_path()


class FFmpegFrameCompressor:
    """
    Compress a batch of frames into an MP4 video file
    
    Uses H.265/HEVC encoding for efficient compression.
    Supports input as list of file paths or list of PIL Images.
    """
    
    def __init__(self, fps: float = 1.0, crf: int = 23, preset: str = "ultrafast"):
        """
        Args:
            fps: Frames per second
            crf: Constant Rate Factor (0-51, lower = better quality)
            preset: Encoding preset (ultrafast, veryfast, fast, medium, slow)
        """
        self.fps = fps
        self.crf = crf
        self.preset = preset
        self.ffmpeg_path = _FFMPEG_PATH
        
        if not self.ffmpeg_path:
            logger.error("FFmpeg not found. Video compression will not work.")
    
    def compress_from_files(
        self,
        input_files: List[str],
        output_path: str
    ) -> bool:
        """
        Compress a list of image files into MP4
        
        Args:
            input_files: List of image file paths (in order)
            output_path: Output MP4 file path
            
        Returns:
            True if successful
        """
        if not self.ffmpeg_path:
            logger.error("FFmpeg not available")
            return False
        
        if not input_files:
            logger.warning("No input files to compress")
            return False
        
        try:
            # Create concat file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                concat_file = f.name
                for img_path in input_files:
                    # FFmpeg concat demuxer format
                    f.write(f"file '{img_path}'\n")
                    f.write(f"duration {1.0/self.fps}\n")
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Build FFmpeg command
            cmd = [
                self.ffmpeg_path,
                "-y",  # Overwrite output
                "-f", "concat",
                "-safe", "0",  # Allow absolute paths
                "-i", concat_file,
                # Pad to even dimensions (required for H.265)
                "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
                # H.265 encoding
                "-vcodec", "libx265",
                "-tag:v", "hvc1",
                "-preset", self.preset,
                "-crf", str(self.crf),
                "-pix_fmt", "yuv420p",
                "-r", str(self.fps),
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=300  # 5 minute timeout
            )
            
            # Cleanup concat file
            os.unlink(concat_file)
            
            if result.returncode != 0:
                stderr = result.stderr.decode('utf-8', errors='ignore')
                logger.error(f"FFmpeg compression failed: {stderr[-500:]}")
                return False
            
            logger.info(f"Compressed {len(input_files)} frames to {output_path}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg compression timed out")
            return False
        except Exception as e:
            logger.error(f"FFmpeg compression error: {e}")
            return False
    
    def compress_from_images(
        self,
        images: List[Image.Image],
        output_path: str
    ) -> bool:
        """
        Compress a list of PIL Images into MP4
        
        Args:
            images: List of PIL Images (in order)
            output_path: Output MP4 file path
            
        Returns:
            True if successful
        """
        if not self.ffmpeg_path:
            logger.error("FFmpeg not available")
            return False
        
        if not images:
            logger.warning("No images to compress")
            return False
        
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Build FFmpeg command (pipe input)
            cmd = [
                self.ffmpeg_path,
                "-y",  # Overwrite output
                "-f", "image2pipe",
                "-vcodec", "png",
                "-r", str(self.fps),
                "-i", "-",  # Read from stdin
                # Pad to even dimensions
                "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
                # H.265 encoding
                "-vcodec", "libx265",
                "-tag:v", "hvc1",
                "-preset", self.preset,
                "-crf", str(self.crf),
                "-pix_fmt", "yuv420p",
                output_path
            ]
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Write images to stdin
            for img in images:
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                process.stdin.write(buffer.getvalue())
            
            process.stdin.close()
            stdout, stderr = process.communicate(timeout=300)
            
            if process.returncode != 0:
                stderr_text = stderr.decode('utf-8', errors='ignore')
                logger.error(f"FFmpeg compression failed: {stderr_text[-500:]}")
                return False
            
            logger.info(f"Compressed {len(images)} images to {output_path}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg compression timed out")
            return False
        except Exception as e:
            logger.error(f"FFmpeg compression error: {e}")
            return False


class FFmpegFrameExtractor:
    """
    Extract frames from MP4 video files
    
    Used for:
    1. Timeline browsing (full screen videos)
    2. Query result display (window videos)
    """
    
    def __init__(self):
        self.ffmpeg_path = _FFMPEG_PATH
        self.ffprobe_path = _FFPROBE_PATH
        
        if not self.ffmpeg_path:
            logger.error("FFmpeg not found. Frame extraction will not work.")
    
    def get_video_info(self, video_path: str) -> Optional[dict]:
        """
        Get video metadata
        
        Returns:
            Dict with fps, duration, frame_count, width, height
        """
        if not self.ffprobe_path:
            logger.error("FFprobe not available")
            return None
        
        try:
            cmd = [
                self.ffprobe_path,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
                "-of", "json",
                video_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return None
            
            import json
            data = json.loads(result.stdout)
            
            if not data.get("streams"):
                return None
            
            stream = data["streams"][0]
            
            # Parse frame rate (format: "30/1" or "30000/1001")
            fps_parts = stream.get("r_frame_rate", "1/1").split("/")
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 1.0
            
            return {
                "width": stream.get("width"),
                "height": stream.get("height"),
                "fps": fps,
                "frame_count": int(stream.get("nb_frames", 0)) if stream.get("nb_frames") else None,
                "duration": float(stream.get("duration", 0)) if stream.get("duration") else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return None
    
    def extract_frame_by_index(
        self,
        video_path: str,
        frame_index: int,
        fps: float = 1.0
    ) -> Optional[Image.Image]:
        """
        Extract a single frame by index
        
        Args:
            video_path: Path to MP4 video
            frame_index: 0-based frame index
            fps: Video FPS (used to calculate timestamp)
            
        Returns:
            PIL Image or None if failed
        """
        if not self.ffmpeg_path:
            logger.error("FFmpeg not available")
            return None
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
        
        try:
            # Calculate timestamp from index
            timestamp = frame_index / fps
            
            # Build FFmpeg command
            cmd = [
                self.ffmpeg_path,
                "-ss", str(timestamp),  # Seek to timestamp
                "-i", video_path,
                "-vframes", "1",  # Extract 1 frame
                "-f", "image2pipe",
                "-vcodec", "png",
                "-"  # Output to stdout
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode != 0:
                stderr = result.stderr.decode('utf-8', errors='ignore')
                logger.error(f"Frame extraction failed: {stderr[-200:]}")
                return None
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(result.stdout))
            return image.convert('RGB')
            
        except subprocess.TimeoutExpired:
            logger.error("Frame extraction timed out")
            return None
        except Exception as e:
            logger.error(f"Frame extraction error: {e}")
            return None
    
    def extract_frame_by_timestamp(
        self,
        video_path: str,
        timestamp_seconds: float
    ) -> Optional[Image.Image]:
        """
        Extract a single frame by timestamp
        
        Args:
            video_path: Path to MP4 video
            timestamp_seconds: Time in seconds from video start
            
        Returns:
            PIL Image or None if failed
        """
        if not self.ffmpeg_path:
            return None
        
        try:
            cmd = [
                self.ffmpeg_path,
                "-ss", str(timestamp_seconds),
                "-i", video_path,
                "-vframes", "1",
                "-f", "image2pipe",
                "-vcodec", "png",
                "-"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return None
            
            image = Image.open(io.BytesIO(result.stdout))
            return image.convert('RGB')
            
        except Exception as e:
            logger.error(f"Frame extraction error: {e}")
            return None
    
    def extract_multiple_frames(
        self,
        video_path: str,
        frame_indices: List[int],
        fps: float = 1.0
    ) -> List[Tuple[int, Optional[Image.Image]]]:
        """
        Extract multiple frames by indices
        
        Args:
            video_path: Path to MP4 video
            frame_indices: List of 0-based frame indices
            fps: Video FPS
            
        Returns:
            List of (index, image) tuples
        """
        results = []
        for idx in frame_indices:
            image = self.extract_frame_by_index(video_path, idx, fps)
            results.append((idx, image))
        return results
    
    def extract_thumbnail(
        self,
        video_path: str,
        max_width: int = 320,
        frame_index: int = 0,
        fps: float = 1.0
    ) -> Optional[Image.Image]:
        """
        Extract a thumbnail from video
        
        Args:
            video_path: Path to MP4 video
            max_width: Maximum width for thumbnail
            frame_index: Which frame to use
            fps: Video FPS
            
        Returns:
            PIL Image thumbnail or None
        """
        image = self.extract_frame_by_index(video_path, frame_index, fps)
        if image is None:
            return None
        
        # Resize if needed
        if image.width > max_width:
            ratio = max_width / image.width
            new_height = int(image.height * ratio)
            image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def frame_to_base64(self, image: Image.Image, quality: int = 80) -> str:
        """
        Convert PIL Image to base64 string
        
        Args:
            image: PIL Image
            quality: JPEG quality (1-100)
            
        Returns:
            Base64 encoded string
        """
        import base64
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        return base64.b64encode(buffer.getvalue()).decode('ascii')


# Convenience functions
def compress_frames_to_video(
    input_files: List[str],
    output_path: str,
    fps: float = 1.0
) -> bool:
    """Compress a list of image files to MP4"""
    compressor = FFmpegFrameCompressor(fps=fps)
    return compressor.compress_from_files(input_files, output_path)


def extract_frame_from_video(
    video_path: str,
    frame_index: int,
    fps: float = 1.0
) -> Optional[Image.Image]:
    """Extract a single frame from MP4 video"""
    extractor = FFmpegFrameExtractor()
    return extractor.extract_frame_by_index(video_path, frame_index, fps)


def get_video_frame_as_base64(
    video_path: str,
    frame_index: int,
    fps: float = 1.0,
    quality: int = 80
) -> Optional[str]:
    """Extract a frame and return as base64"""
    extractor = FFmpegFrameExtractor()
    image = extractor.extract_frame_by_index(video_path, frame_index, fps)
    if image:
        return extractor.frame_to_base64(image, quality)
    return None
