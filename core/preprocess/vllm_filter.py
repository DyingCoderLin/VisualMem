# core/preprocess/vllm_filter.py
import pytesseract
from .base_preprocessor import AbstractPreprocessor
from utils.data_models import ScreenFrame
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

class VLLMFilter(AbstractPreprocessor):
    """
    (未来实现) 使用一个轻量级VLM或视觉模型来决定
    这个帧是否"有趣"或"重要"，而不仅仅是看像素差异。
    """
    def __init__(self):
        # TODO: 在此初始化你的 VLLM 过滤模型
        logger.info("VLLMFilter initialized (Not yet implemented).")

    def process(self, frame: ScreenFrame) -> Optional[ScreenFrame]:
        # TODO: 调用 VLLM 过滤模型
        # is_important = self.vllm_filter_model.predict(frame.image)
        # if not is_important:
        #     return None
        
        logger.info("Processing frame (passthrough for now).")
        
        # (在实现前，可以先填充OCR)
        try:
            frame.ocr_text = pytesseract.image_to_string(frame.image)
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            frame.ocr_text = ""

        return frame


