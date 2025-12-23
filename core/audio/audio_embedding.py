# core/audio/audio_embedding.py
"""
音频 Embedding 模块

用于说话人识别（Speaker Diarization）
参考 screenpipe 的 speaker embedding 逻辑
"""

from typing import List, Optional
import numpy as np
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SpeakerEmbedding:
    """说话人 Embedding 结果"""
    
    def __init__(
        self,
        embedding: np.ndarray,
        speaker_id: Optional[int] = None,
        confidence: float = 0.0
    ):
        self.embedding = embedding
        self.speaker_id = speaker_id
        self.confidence = confidence


class AudioEmbedding:
    """音频 Embedding 基类"""
    
    def extract_speaker_embedding(self, audio_data: np.ndarray, sample_rate: int = 16000) -> SpeakerEmbedding:
        """
        提取说话人 embedding
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            SpeakerEmbedding 对象
        """
        raise NotImplementedError


class PyannoteEmbedding(AudioEmbedding):
    """
    使用 pyannote.audio 的 speaker embedding
    
    pyannote.audio 是强大的说话人分离工具
    """
    
    def __init__(self, model_name: str = "pyannote/embedding"):
        """
        初始化 pyannote embedding
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        
        try:
            from pyannote.audio import Model
            from pyannote.audio.pipelines import SpeakerEmbedding as PyannoteModel
            
            logger.info(f"Loading pyannote model: {model_name}")
            self.model = PyannoteModel(Model.from_pretrained(model_name))
            logger.info("Pyannote embedding initialized")
            
        except ImportError:
            logger.error("pyannote.audio not installed. Run: pip install pyannote.audio")
            raise
    
    def extract_speaker_embedding(self, audio_data: np.ndarray, sample_rate: int = 16000) -> SpeakerEmbedding:
        """
        提取说话人 embedding
        """
        try:
            # pyannote 需要特定格式的输入
            # 这里简化处理，实际使用需要更复杂的预处理
            
            embedding = self.model({"waveform": audio_data, "sample_rate": sample_rate})
            
            logger.debug(f"Extracted speaker embedding: dim={len(embedding)}")
            
            return SpeakerEmbedding(
                embedding=np.array(embedding),
                confidence=1.0
            )
            
        except Exception as e:
            logger.error(f"Failed to extract speaker embedding: {e}")
            # 返回零向量
            return SpeakerEmbedding(
                embedding=np.zeros(512),  # 假设维度为 512
                confidence=0.0
            )


class DummyEmbedding(AudioEmbedding):
    """虚拟 Embedding（用于测试）"""
    
    def __init__(self):
        logger.info("Dummy audio embedding initialized")
    
    def extract_speaker_embedding(self, audio_data: np.ndarray, sample_rate: int = 16000) -> SpeakerEmbedding:
        """返回零向量"""
        return SpeakerEmbedding(
            embedding=np.zeros(512),
            confidence=0.0
        )


def create_audio_embedding(embedding_type: str = "pyannote", **kwargs) -> AudioEmbedding:
    """
    工厂函数：创建音频 embedding 模型
    
    Args:
        embedding_type: 类型
            - "pyannote": Pyannote embedding
            - "dummy": 虚拟 embedding
        **kwargs: 特定参数
        
    Returns:
        AudioEmbedding 实例
    """
    if embedding_type == "pyannote":
        return PyannoteEmbedding(**kwargs)
    elif embedding_type == "dummy":
        return DummyEmbedding()
    else:
        logger.warning(f"Unknown embedding type: {embedding_type}, using dummy")
        return DummyEmbedding()


def compare_speaker_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    比较两个说话人 embedding 的相似度
    
    Args:
        emb1: 第一个 embedding
        emb2: 第二个 embedding
        
    Returns:
        余弦相似度（0-1）
    """
    # 余弦相似度
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    # 归一化到 0-1
    return (similarity + 1) / 2

