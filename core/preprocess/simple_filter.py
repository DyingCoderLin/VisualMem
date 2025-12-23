# core/preprocess/simple_filter.py
import pytesseract
from PIL import Image, ImageChops
import numpy as np
from typing import Optional
from .base_preprocessor import AbstractPreprocessor
from utils.data_models import ScreenFrame
from utils.logger import setup_logger

logger = setup_logger(__name__)

def calculate_normalized_rms_diff(img1: Image.Image, img2: Image.Image) -> float:
    """
    计算两张 PIL 图像的归一化均方根(RMS)差异。
    返回一个 0.0 (相同) 到 1.0 (完全不同) 之间的浮点数。
    """
    # 确保图像模式和大小相同
    if img1.size != img2.size or img1.mode != img2.mode:
        img2 = img2.resize(img1.size).convert(img1.mode)
        
    # 1. 计算差异图像
    diff_img = ImageChops.difference(img1, img2)
    
    # 2. 转换为 numpy 数组
    diff_array = np.array(diff_img)
    
    # 3. 计算 RMS
    rms = np.sqrt(np.mean(np.square(diff_array)))
    
    # 4. 归一化 (RMS 的最大值是 255)
    normalized_rms = rms / 255.0
    return normalized_rms


class SimpleFilter(AbstractPreprocessor):
    """
    一个简单的"粗暴"过滤器 (Naive 实现):
    1. 检查与上一帧的图像差异，如果差异小于阈值，则丢弃 (返回 None)。
    2. (可选) 如果通过，运行 OCR 丰富上下文。
    """
    def __init__(self, diff_threshold: float):
        self.last_frame_image: Optional[Image.Image] = None
        self.diff_threshold = diff_threshold
        logger.info(f"SimpleFilter initialized with diff_threshold: {self.diff_threshold}")

    def process(self, frame: ScreenFrame) -> Optional[ScreenFrame]:
        # 1. 计算与上一帧的差异
        if self.last_frame_image:
            diff_score = calculate_normalized_rms_diff(self.last_frame_image, frame.image)
            
            logger.debug(f"Frame diff score: {diff_score}")
            
            if diff_score < self.diff_threshold:
                # 差异太小，丢弃这一帧
                logger.debug("Frame filtered out (diff too small)")
                return None
        
        # 2. 更新上一帧为当前帧 (通过了差异检测)
        # 注意：需要复制图像，以防后续步骤修改 frame.image
        self.last_frame_image = frame.image.copy()

        # 3. (可选) 运行 OCR 并填充，为 VLM 提供额外上下文
        logger.info("Frame passed filter. Running OCR...")
        try:
            frame.ocr_text = pytesseract.image_to_string(frame.image)
            logger.debug(f"OCR extracted {len(frame.ocr_text)} characters")
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            frame.ocr_text = ""
        
        # 4. 通过该帧，准备发送给VLM
        return frame


