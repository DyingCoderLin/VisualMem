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
from .qwen_encoder import QwenEncoder

def create_encoder(model_name: str, device: str = None) -> MultiModalEncoderInterface:
    """
    编码器工厂函数：根据模型名称创建相应的编码器
    
    Args:
        model_name: 模型名称或路径
        device: 计算设备
        
    Returns:
        编码器实例
    """
    if "qwen" in model_name.lower():
        return QwenEncoder(model_name=model_name, device=device)
    else:
        # 默认使用 CLIP/SigLIP 编码器
        return CLIPEncoder(model_name=model_name, device=device)

__all__ = [
    # 抽象基类
    "BaseEncoder",
    "TextEncoderInterface",
    "ImageEncoderInterface",
    "MultiModalEncoderInterface",
    # 具体实现
    "TextEncoder",
    "CLIPEncoder",
    "QwenEncoder",
    # 工厂函数
    "create_text_encoder",
    "create_encoder",
]
