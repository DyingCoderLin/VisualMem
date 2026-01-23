# utils/video_utils.py
"""
Video Utilities Module

Provides utilities for extracting frames from video chunks at query time.
Used to retrieve individual frames from MP4 storage when needed for
VLM analysis or display.

Reference: screenpipe's video_utils.rs
"""
import os
import subprocess
import tempfile
import base64
from typing import Optional, List, Tuple
from pathlib import Path
from PIL import Image
from io import BytesIO
from .logger import setup_logger

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
    ffmpeg_path = find_ffmpeg_path()
    if ffmpeg_path:
        ffprobe_path = ffmpeg_path.replace("ffmpeg", "ffprobe")
        if os.path.exists(ffprobe_path):
            return ffprobe_path
    
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
    
    return None


def get_video_fps(video_path: str) -> float:
    """
    Get the frame rate of a video file
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Frame rate (fps), defaults to 1.0 if detection fails
    """
    ffprobe_path = find_ffprobe_path()
    if not ffprobe_path:
        logger.warning("ffprobe not found, using default fps=1.0")
        return 1.0
    
    try:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v", "quiet",
                "-print_format", "json",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                video_path
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            if streams:
                rate_str = streams[0].get("r_frame_rate", "1/1")
                if "/" in rate_str:
                    num, den = rate_str.split("/")
                    if float(den) != 0:
                        return float(num) / float(den)
    except Exception as e:
        logger.debug(f"Failed to get video fps: {e}")
    
    return 1.0


def get_video_metadata(video_path: str) -> dict:
    """
    Get metadata from a video file
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with fps, duration, creation_time, etc.
    """
    ffprobe_path = find_ffprobe_path()
    if not ffprobe_path:
        return {"fps": 1.0, "duration": 0.0}
    
    try:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            
            # Get fps from first video stream
            fps = 1.0
            streams = data.get("streams", [])
            for stream in streams:
                if stream.get("codec_type") == "video":
                    rate_str = stream.get("r_frame_rate", "1/1")
                    if "/" in rate_str:
                        num, den = rate_str.split("/")
                        if float(den) != 0:
                            fps = float(num) / float(den)
                    break
            
            # Get duration and creation time from format
            format_info = data.get("format", {})
            duration = float(format_info.get("duration", 0))
            
            tags = format_info.get("tags", {})
            creation_time = tags.get("creation_time")
            
            return {
                "fps": fps,
                "duration": duration,
                "creation_time": creation_time,
                "format_name": format_info.get("format_name"),
                "size": int(format_info.get("size", 0))
            }
            
    except Exception as e:
        logger.debug(f"Failed to get video metadata: {e}")
    
    return {"fps": 1.0, "duration": 0.0}


def extract_frame(
    video_path: str,
    offset_index: int,
    fps: Optional[float] = None,
    output_format: str = "pil"
) -> Optional[Image.Image]:
    """
    Extract a single frame from a video file
    
    Args:
        video_path: Path to the video file
        offset_index: Frame index to extract
        fps: Frames per second (auto-detected if None)
        output_format: "pil" for PIL Image, "base64" for base64 string
        
    Returns:
        PIL Image or None if extraction failed
    """
    ffmpeg_path = find_ffmpeg_path()
    if not ffmpeg_path:
        logger.error("FFmpeg not found")
        return None
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None
    
    # Get fps if not provided
    if fps is None:
        fps = get_video_fps(video_path)
    
    # Calculate timestamp
    timestamp = offset_index / fps
    
    try:
        # Use FFmpeg to extract frame to stdout as JPEG
        result = subprocess.run(
            [
                ffmpeg_path,
                "-ss", f"{timestamp:.3f}",
                "-i", video_path,
                "-vframes", "1",
                "-f", "image2pipe",
                "-c:v", "mjpeg",
                "-q:v", "2",
                "-"
            ],
            capture_output=True,
            timeout=30
        )
        
        if result.returncode != 0:
            logger.warning(f"FFmpeg failed: {result.stderr.decode()}")
            return None
        
        if not result.stdout:
            logger.warning("No frame data received from FFmpeg")
            return None
        
        # Parse image from bytes
        image = Image.open(BytesIO(result.stdout))
        image.load()  # Force load
        
        if output_format == "base64":
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return image
        
    except subprocess.TimeoutExpired:
        logger.error("Frame extraction timed out")
    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
    
    return None


def extract_frame_to_file(
    video_path: str,
    offset_index: int,
    output_path: str,
    fps: Optional[float] = None
) -> bool:
    """
    Extract a frame and save to file
    
    Args:
        video_path: Path to the video file
        offset_index: Frame index to extract
        output_path: Path to save the extracted frame
        fps: Frames per second (auto-detected if None)
        
    Returns:
        True if successful
    """
    ffmpeg_path = find_ffmpeg_path()
    if not ffmpeg_path:
        logger.error("FFmpeg not found")
        return False
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False
    
    # Get fps if not provided
    if fps is None:
        fps = get_video_fps(video_path)
    
    # Calculate timestamp
    timestamp = offset_index / fps
    
    try:
        result = subprocess.run(
            [
                ffmpeg_path,
                "-y",  # Overwrite
                "-ss", f"{timestamp:.3f}",
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",
                output_path
            ],
            capture_output=True,
            timeout=30
        )
        
        if result.returncode == 0 and os.path.exists(output_path):
            return True
        
        logger.warning(f"FFmpeg failed: {result.stderr.decode()}")
        return False
        
    except subprocess.TimeoutExpired:
        logger.error("Frame extraction timed out")
    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
    
    return False


def extract_frames_batch(
    video_path: str,
    offset_indices: List[int],
    fps: Optional[float] = None
) -> List[Tuple[int, Optional[Image.Image]]]:
    """
    Extract multiple frames from a video file
    
    Args:
        video_path: Path to the video file
        offset_indices: List of frame indices to extract
        fps: Frames per second (auto-detected if None)
        
    Returns:
        List of (offset_index, image) tuples
    """
    if fps is None:
        fps = get_video_fps(video_path)
    
    results = []
    for offset in offset_indices:
        image = extract_frame(video_path, offset, fps)
        results.append((offset, image))
    
    return results


def extract_frame_base64(
    video_path: str,
    offset_index: int,
    fps: Optional[float] = None,
    quality: int = 85
) -> Optional[str]:
    """
    Extract a frame and return as base64 string
    
    Args:
        video_path: Path to the video file
        offset_index: Frame index to extract
        fps: Frames per second (auto-detected if None)
        quality: JPEG quality (1-100)
        
    Returns:
        Base64 encoded JPEG string or None
    """
    image = extract_frame(video_path, offset_index, fps)
    if image is None:
        return None
    
    try:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Base64 encoding failed: {e}")
        return None


def validate_video_file(video_path: str) -> bool:
    """
    Check if a video file is valid and readable
    
    Args:
        video_path: Path to the video file
        
    Returns:
        True if valid
    """
    ffmpeg_path = find_ffmpeg_path()
    if not ffmpeg_path:
        return False
    
    if not os.path.exists(video_path):
        return False
    
    try:
        result = subprocess.run(
            [
                ffmpeg_path,
                "-v", "error",
                "-i", video_path,
                "-f", "null",
                "-"
            ],
            capture_output=True,
            timeout=30
        )
        return result.returncode == 0
    except Exception:
        return False


def get_frame_count(video_path: str) -> int:
    """
    Get the total number of frames in a video
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Number of frames, or 0 if detection fails
    """
    ffprobe_path = find_ffprobe_path()
    if not ffprobe_path:
        return 0
    
    try:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v", "error",
                "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=nb_read_packets",
                "-of", "csv=p=0",
                video_path
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            count = result.stdout.strip()
            if count.isdigit():
                return int(count)
    except Exception as e:
        logger.debug(f"Failed to get frame count: {e}")
    
    # Fallback: estimate from duration and fps
    metadata = get_video_metadata(video_path)
    if metadata["duration"] > 0:
        return int(metadata["duration"] * metadata["fps"])
    
    return 0
