# core/capture/screenshot_capturer.py
import datetime
from typing import Optional
from PIL import ImageGrab, Image
from .base_capturer import AbstractCapturer
from utils.data_models import ScreenFrame
from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__)

class ScreenshotCapturer(AbstractCapturer):
    """
    使用 PIL 的 ImageGrab 来捕获屏幕截图
    支持自动压缩以节省存储和VLM开销
    """
    
    def __init__(self, max_width: int = None):
        """
        Args:
            max_width: 图片最大宽度，0或None表示不压缩
        """
        self.max_width = max_width if max_width is not None else config.MAX_IMAGE_WIDTH
        if self.max_width > 0:
            logger.info(f"ScreenshotCapturer initialized (max_width={self.max_width})")
        else:
            logger.info("ScreenshotCapturer initialized (no compression)")
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """
        按比例压缩图片
        """
        if self.max_width <= 0 or image.width <= self.max_width:
            return image
        
        # 计算新的尺寸（保持宽高比）
        ratio = self.max_width / image.width
        new_height = int(image.height * ratio)
        
        # 使用高质量的缩放算法
        resized = image.resize((self.max_width, new_height), Image.Resampling.LANCZOS)
        logger.debug(f"Resized image: {image.size} -> {resized.size}")
        return resized
    
    def capture(self) -> Optional[ScreenFrame]:
        """
        捕获当前屏幕的截图（保持原始高清分辨率）
        """
        try:
            # 捕获整个屏幕
            screenshot: Image.Image = ImageGrab.grab()
            original_size = screenshot.size
            
            # 转换为RGB模式（JPEG不支持透明通道）
            if screenshot.mode in ('RGBA', 'LA', 'P'):
                screenshot = screenshot.convert('RGB')
                logger.debug(f"Converted image to RGB mode for JPEG compatibility")
            
            # 不再在这里压缩图片，改为在存储阶段压缩，
            # 这样 OCR 和 Embedding 可以使用高清图片
            # screenshot = self._resize_image(screenshot)
            
            # 创建 ScreenFrame 对象
            frame = ScreenFrame(
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                image=screenshot,
                ocr_text=None  # OCR 文本将在预处理阶段填充
            )
            
            logger.debug(f"Captured screenshot: {screenshot.size}")
            
            return frame
            
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            return None


