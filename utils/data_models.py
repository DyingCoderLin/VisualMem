# utils/data_models.py
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Dict, Any
from PIL.Image import Image
import datetime


class ScreenFrame(BaseModel):
    """Legacy single-level screen capture model (kept for backward compatibility)"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    timestamp: datetime.datetime
    image: Image  # PIL 图像对象
    ocr_text: Optional[str] = None


class WindowFrame(BaseModel):
    """
    Individual application window capture
    
    Represents a single window captured from the screen, with its own
    frame difference detection and storage pipeline.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    sub_frame_id: Optional[str] = None  # Assigned after frame diff check confirms storage
    app_name: str
    window_name: str
    process_id: int
    is_focused: bool = False
    image: Image  # PIL 图像对象
    image_hash: int = 0  # Hash for quick comparison
    ocr_text: Optional[str] = None
    ocr_text_json: Optional[str] = None
    ocr_confidence: float = 0.0
    timestamp: Optional[datetime.datetime] = None
    
    # Video chunk reference (set after encoding)
    window_chunk_id: Optional[int] = None
    offset_index: Optional[int] = None


class ScreenObject(BaseModel):
    """
    Screen capture with all visible windows
    
    Represents a complete capture of a monitor, including both the full screen
    image and individual window captures. The full screen and each window have
    independent frame difference detection.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    monitor_id: int
    device_name: str = "default"
    timestamp: datetime.datetime
    
    # Full screen capture
    full_screen_image: Image
    full_screen_hash: int = 0  # Hash for quick comparison
    
    # Individual window captures
    windows: List[WindowFrame] = Field(default_factory=list)
    
    # Frame ID (assigned after frame diff check confirms storage)
    frame_id: Optional[str] = None
    
    # Video chunk reference (set after encoding)
    video_chunk_id: Optional[int] = None
    offset_index: Optional[int] = None
    
    # OCR results for full screen (optional)
    ocr_text: Optional[str] = None
    ocr_text_json: Optional[str] = None


class FrameSubFrameMapping(BaseModel):
    """
    Mapping between a screen frame and its associated window sub-frames
    
    A new frame_id is generated when the full screen changes significantly.
    Sub-frame IDs may be new or reused from previous captures if the specific
    window content hasn't changed.
    """
    frame_id: str
    sub_frame_ids: List[str] = Field(default_factory=list)
    timestamp: datetime.datetime
    monitor_id: int


class VideoChunkInfo(BaseModel):
    """Metadata for a video chunk (MP4 file)"""
    chunk_id: int
    file_path: str
    chunk_type: str  # "screen" or "window"
    monitor_id: int
    app_name: Optional[str] = None  # For window chunks
    window_name: Optional[str] = None  # For window chunks
    fps: float = 1.0
    frame_count: int = 0
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))


class VLMAnalysis(BaseModel):
    """VLM analysis result for a frame"""
    frame_id: str
    timestamp: datetime.datetime
    description: str
    visual_elements: List[Dict[str, Any]]
    layout_summary: str
    entities: List[str]
    embedding: Optional[List[float]] = None


