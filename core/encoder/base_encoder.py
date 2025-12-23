"""
编码器抽象基类

为所有编码器（文本、图像、多模态）提供统一接口
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional
from PIL import Image


class BaseEncoder(ABC):
    """
    编码器抽象基类
    
    所有编码器（TextEncoder, CLIPEncoder 等）都应继承此基类
    并实现其抽象方法，以提供统一的编码接口
    """
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        初始化编码器
        
        Args:
            model_name: 模型名称或路径
            device: 计算设备
        """
        self.model_name = model_name
        self.device = device
        self.embedding_dim = None
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        获取 embedding 维度
        
        Returns:
            embedding 向量的维度
        """
        pass
    
    @abstractmethod
    def encode(self, input_data: Union[str, Image.Image]) -> List[float]:
        """
        编码单个输入（文本或图像）
        
        Args:
            input_data: 输入数据（文本字符串或 PIL Image）
            
        Returns:
            embedding 向量
        """
        pass
    
    @abstractmethod
    def encode_batch(self, input_data_list: List[Union[str, Image.Image]]) -> List[List[float]]:
        """
        批量编码
        
        Args:
            input_data_list: 输入数据列表
            
        Returns:
            embedding 向量列表
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, device={self.device}, dim={self.embedding_dim})"


class TextEncoderInterface(BaseEncoder):
    """
    文本编码器接口
    
    专门用于纯文本编码的编码器应继承此接口
    """
    
    @abstractmethod
    def encode_text(self, text: str) -> List[float]:
        """
        编码文本
        
        Args:
            text: 输入文本
            
        Returns:
            embedding 向量
        """
        pass
    
    def encode(self, input_data: str) -> List[float]:
        """实现基类的 encode 方法"""
        return self.encode_text(input_data)
    
    def encode_batch(self, input_data_list: List[str]) -> List[List[float]]:
        """实现基类的 encode_batch 方法"""
        return self.encode_text_batch(input_data_list)
    
    @abstractmethod
    def encode_text_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            
        Returns:
            embedding 向量列表
        """
        pass


class ImageEncoderInterface(BaseEncoder):
    """
    图像编码器接口
    
    专门用于图像编码的编码器应继承此接口
    """
    
    @abstractmethod
    def encode_image(self, image: Image.Image) -> List[float]:
        """
        编码图像
        
        Args:
            image: PIL Image 对象
            
        Returns:
            embedding 向量
        """
        pass
    
    def encode(self, input_data: Image.Image) -> List[float]:
        """实现基类的 encode 方法"""
        return self.encode_image(input_data)
    
    def encode_batch(self, input_data_list: List[Image.Image]) -> List[List[float]]:
        """实现基类的 encode_batch 方法"""
        return self.encode_image_batch(input_data_list)
    
    @abstractmethod
    def encode_image_batch(self, images: List[Image.Image]) -> List[List[float]]:
        """
        批量编码图像
        
        Args:
            images: PIL Image 对象列表
            
        Returns:
            embedding 向量列表
        """
        pass


class MultiModalEncoderInterface(TextEncoderInterface, ImageEncoderInterface):
    """
    多模态编码器接口
    
    同时支持文本和图像编码的编码器（如 CLIP）应继承此接口
    """
    
    def encode(self, input_data: Union[str, Image.Image]) -> List[float]:
        """
        智能编码：根据输入类型自动选择编码方法
        
        Args:
            input_data: 文本字符串或 PIL Image
            
        Returns:
            embedding 向量
        """
        if isinstance(input_data, str):
            return self.encode_text(input_data)
        elif isinstance(input_data, Image.Image):
            return self.encode_image(input_data)
        else:
            raise TypeError(f"不支持的输入类型: {type(input_data)}")
    
    def encode_batch(self, input_data_list: List[Union[str, Image.Image]]) -> List[List[float]]:
        """
        批量编码：自动处理混合类型
        
        Args:
            input_data_list: 混合输入列表（文本和图像）
            
        Returns:
            embedding 向量列表
        """
        results = []
        for data in input_data_list:
            results.append(self.encode(data))
        return results

