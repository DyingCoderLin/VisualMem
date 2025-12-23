# core/audio/__init__.py
"""
音频处理模块

包含 ASR（语音识别）和音频 embedding 功能
注意：这些是核心代码，暂未应用到 main.py 中
"""

from .asr_engine import (
    ASREngine,
    ASRResult,
    WhisperASR,
    DummyASR,
    create_asr_engine
)

from .audio_embedding import (
    AudioEmbedding,
    SpeakerEmbedding,
    create_audio_embedding
)

__all__ = [
    # ASR
    'ASREngine',
    'ASRResult',
    'WhisperASR',
    'DummyASR',
    'create_asr_engine',
    # Embedding
    'AudioEmbedding',
    'SpeakerEmbedding',
    'create_audio_embedding'
]

