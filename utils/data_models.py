# utils/data_models.py
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from PIL.Image import Image
import datetime

class ScreenFrame(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    timestamp: datetime.datetime
    image: Image  # PIL 图像对象
    ocr_text: Optional[str] = None

class VLMAnalysis(BaseModel):
    frame_id: str
    timestamp: datetime.datetime
    description: str
    visual_elements: List[Dict[str, Any]]
    layout_summary: str
    entities: List[str]
    embedding: Optional[List[float]] = None


