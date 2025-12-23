# core/ocr/ocr_engine.py
"""
OCR 引擎封装

支持多种 OCR 引擎，提供统一接口
"""

from typing import Optional, Dict, List
from PIL import Image
import time
from utils.logger import setup_logger

logger = setup_logger(__name__)


class OCRResult:
    """OCR 识别结果"""
    
    def __init__(
        self,
        text: str,
        confidence: float = 0.0,
        text_json: Optional[str] = None,
        engine: str = "pytesseract"
    ):
        self.text = text
        self.confidence = confidence
        self.text_json = text_json or ""
        self.engine = engine


class OCREngine:
    """
    OCR 引擎基类
    """
    
    def recognize(self, image: Image.Image) -> OCRResult:
        """
        识别图像中的文本
        
        Args:
            image: PIL Image 对象
            
        Returns:
            OCRResult 对象
        """
        raise NotImplementedError


class PytesseractOCR(OCREngine):
    """
    使用 Pytesseract 的 OCR 引擎
    
    优点：免费、易用
    缺点：识别率一般
    """
    
    def __init__(self, lang: str = "chi_sim+eng"):
        """
        初始化 Pytesseract OCR
        
        Args:
            lang: 语言设置，默认中英文
        """
        self.lang = lang
        self.engine_name = "pytesseract"
        
        try:
            import pytesseract
            self.pytesseract = pytesseract
            logger.info(f"Pytesseract OCR initialized with lang={lang}")
        except ImportError:
            logger.error("pytesseract not installed. Run: pip install pytesseract")
            raise
    
    def recognize(self, image: Image.Image) -> OCRResult:
        """
        使用 Pytesseract 识别文本
        """
        try:
            start_time = time.perf_counter()
            
            # 识别文本
            text = self.pytesseract.image_to_string(image, lang=self.lang)
            
            # 获取详细信息（包含置信度）
            try:
                data = self.pytesseract.image_to_data(
                    image, 
                    lang=self.lang, 
                    output_type=self.pytesseract.Output.DICT
                )
                
                # 计算平均置信度
                confidences = [float(conf) for conf in data['conf'] if conf != '-1']
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                # 构建 JSON（简化版）
                import json
                text_json = json.dumps({
                    "words": [
                        {
                            "text": word,
                            "confidence": float(conf)
                        }
                        for word, conf in zip(data['text'], data['conf'])
                        if word.strip() and conf != '-1'
                    ]
                }, ensure_ascii=False)
                
            except Exception as e:
                logger.warning(f"Failed to get detailed OCR data: {e}")
                avg_confidence = 0.0
                text_json = ""
            
            elapsed = time.perf_counter() - start_time
            logger.debug(f"OCR completed in {elapsed*1000:.2f}ms, text length: {len(text)}")
            
            return OCRResult(
                text=text.strip(),
                confidence=avg_confidence / 100.0,  # 归一化到 0-1
                text_json=text_json,
                engine=self.engine_name
            )
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return OCRResult(text="", confidence=0.0, engine=self.engine_name)


class DummyOCR(OCREngine):
    """
    虚拟 OCR（用于测试或禁用 OCR 时）
    """
    
    def __init__(self):
        self.engine_name = "dummy"
        logger.info("Dummy OCR initialized (OCR disabled)")
    
    def recognize(self, image: Image.Image) -> OCRResult:
        """不进行任何识别，返回空结果"""
        return OCRResult(text="", confidence=0.0, engine=self.engine_name)


def create_ocr_engine(engine_type: str = "pytesseract", **kwargs) -> OCREngine:
    """
    工厂函数：创建 OCR 引擎
    
    Args:
        engine_type: OCR 引擎类型
            - "pytesseract": Pytesseract OCR
            - "dummy": 虚拟 OCR（禁用）
        **kwargs: 引擎特定参数
        
    Returns:
        OCR 引擎实例
    """
    if engine_type == "pytesseract":
        return PytesseractOCR(**kwargs)
    elif engine_type == "dummy":
        return DummyOCR()
    else:
        logger.warning(f"Unknown OCR engine: {engine_type}, using dummy")
        return DummyOCR()

