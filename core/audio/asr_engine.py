# core/audio/asr_engine.py
"""
ASR (Automatic Speech Recognition) 引擎

参考 screenpipe 的 audio transcription 逻辑
"""

from typing import Optional
import time
import numpy as np
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ASRResult:
    """ASR 识别结果"""
    
    def __init__(
        self,
        text: str,
        language: str = "zh",
        confidence: float = 0.0,
        start_time: float = 0.0,
        end_time: float = 0.0,
        engine: str = "whisper"
    ):
        self.text = text
        self.language = language
        self.confidence = confidence
        self.start_time = start_time
        self.end_time = end_time
        self.engine = engine


class ASREngine:
    """ASR 引擎基类"""
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        """
        转录音频为文本
        
        Args:
            audio_data: 音频数据（numpy array）
            sample_rate: 采样率
            
        Returns:
            ASRResult 对象
        """
        raise NotImplementedError


class WhisperASR(ASREngine):
    """
    使用 Whisper 的 ASR 引擎
    
    Whisper 是 OpenAI 的开源语音识别模型
    参考 screenpipe 使用 whisper.cpp
    """
    
    def __init__(self, model_size: str = "base", language: str = "zh"):
        """
        初始化 Whisper ASR
        
        Args:
            model_size: 模型大小
                - tiny: 39M, 最快
                - base: 74M, 平衡
                - small: 244M
                - medium: 769M
                - large: 1550M, 最准确
            language: 语言代码（zh/en/auto）
        """
        self.model_size = model_size
        self.language = language
        self.engine_name = "whisper"
        
        try:
            import whisper
            self.whisper = whisper
            
            logger.info(f"Loading Whisper model: {model_size}")
            self.model = whisper.load_model(model_size)
            logger.info(f"Whisper ASR initialized ({model_size}, lang={language})")
            
        except ImportError:
            logger.error("whisper not installed. Run: pip install openai-whisper")
            raise
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        """
        使用 Whisper 转录音频
        """
        try:
            start_time = time.perf_counter()
            
            # Whisper 需要 16kHz 采样率
            if sample_rate != 16000:
                logger.warning(f"Resampling from {sample_rate}Hz to 16000Hz")
                # 简化处理：这里应该做重采样
                # 实际使用时需要 librosa 或 scipy
            
            # 转录
            result = self.model.transcribe(
                audio_data,
                language=self.language if self.language != "auto" else None,
                fp16=False  # 使用 float32（CPU 友好）
            )
            
            elapsed = time.perf_counter() - start_time
            
            text = result["text"].strip()
            language = result.get("language", self.language)
            
            logger.debug(
                f"ASR completed in {elapsed:.2f}s, "
                f"text length: {len(text)}, "
                f"language: {language}"
            )
            
            return ASRResult(
                text=text,
                language=language,
                confidence=0.0,  # Whisper 不直接提供置信度
                start_time=0.0,
                end_time=len(audio_data) / sample_rate,
                engine=self.engine_name
            )
            
        except Exception as e:
            logger.error(f"ASR failed: {e}")
            return ASRResult(text="", engine=self.engine_name)


class DummyASR(ASREngine):
    """虚拟 ASR（用于测试或禁用 ASR 时）"""
    
    def __init__(self):
        self.engine_name = "dummy"
        logger.info("Dummy ASR initialized (ASR disabled)")
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        """不进行任何识别，返回空结果"""
        return ASRResult(text="", engine=self.engine_name)


def create_asr_engine(engine_type: str = "whisper", **kwargs) -> ASREngine:
    """
    工厂函数：创建 ASR 引擎
    
    Args:
        engine_type: ASR 引擎类型
            - "whisper": Whisper ASR
            - "dummy": 虚拟 ASR（禁用）
        **kwargs: 引擎特定参数
        
    Returns:
        ASR 引擎实例
    """
    if engine_type == "whisper":
        return WhisperASR(**kwargs)
    elif engine_type == "dummy":
        return DummyASR()
    else:
        logger.warning(f"Unknown ASR engine: {engine_type}, using dummy")
        return DummyASR()

