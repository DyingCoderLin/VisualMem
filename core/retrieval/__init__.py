"""
检索器模块

提供统一的检索接口，支持 Dense、Sparse、Hybrid 三种检索模式
"""

from .base_retriever import (
    BaseRetriever,
    TextRetrieverInterface,
    ImageRetrieverInterface,
    MultiModalRetrieverInterface
)
from .text_retriever import TextRetriever, create_text_retriever
from .image_retriever import ImageRetriever

__all__ = [
    # 抽象基类
    "BaseRetriever",
    "TextRetrieverInterface",
    "ImageRetrieverInterface",
    "MultiModalRetrieverInterface",
    # 具体实现
    "TextRetriever",
    "ImageRetriever",
    # 工厂函数
    "create_text_retriever",
]
