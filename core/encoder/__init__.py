"""
编码器模块

提供统一的编码接口，支持文本和图像编码
"""

from .base_encoder import (
    BaseEncoder,
    TextEncoderInterface,
    ImageEncoderInterface,
    MultiModalEncoderInterface
)
from .text_encoder import TextEncoder, create_text_encoder
from .clip_encoder import CLIPEncoder

__all__ = [
    # 抽象基类
    "BaseEncoder",
    "TextEncoderInterface",
    "ImageEncoderInterface",
    "MultiModalEncoderInterface",
    # 具体实现
    "TextEncoder",
    "CLIPEncoder",
    # 工厂函数
    "create_text_encoder",
]
