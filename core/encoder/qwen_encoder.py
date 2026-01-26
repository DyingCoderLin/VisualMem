# core/encoder/qwen_encoder.py
import torch
import numpy as np
from typing import List, Union, Dict, Any, Optional
from PIL import Image
from core.encoder.base_encoder import MultiModalEncoderInterface
from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__)

class QwenEncoder(MultiModalEncoderInterface):
    """
    Qwen Encoder: 直接使用官方提供的 Qwen3VLEmbedder 类实现
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-Embedding-2B", device: str = None):
        """
        初始化 Qwen 编码器
        """
        super().__init__(model_name, device)
        
        try:
            from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
            
            logger.info(f"Loading Qwen3VLEmbedder from {model_name}")
            
            # 按照官方方式直接加载模型
            # 默认启用 bfloat16 和 flash_attention_2 优化（如果硬件支持）
            self.model = Qwen3VLEmbedder(
                model_name_or_path=model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa"
            )
            
            # 如果显式指定了设备且模型支持 .to()
            if device and device != "auto" and hasattr(self.model, 'to'):
                self.model.to(device)
                
            # 维度固定为 2048
            self.embedding_dim = 2048
            logger.info(f"QwenEncoder loaded. Dim: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen encoder: {e}")
            raise
            
    def get_embedding_dim(self) -> int:
        return self.embedding_dim
        
    def encode_image(self, image: Image.Image) -> List[float]:
        """编码图像"""
        inputs = [{"image": image}]
        embeddings = self.model.process(inputs)
        # 归一化并转为 float32 后再转为 list (避免 NumPy 不支持 bfloat16 的问题)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings[0].detach().cpu().to(torch.float32).numpy().tolist()
        
    def encode_text(self, text: str) -> List[float]:
        """编码文本（查询）"""
        inputs = [{
            "text": text,
            "instruction": "Retrieve images or text relevant to the user's query."
        }]
        embeddings = self.model.process(inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings[0].detach().cpu().to(torch.float32).numpy().tolist()
        
    def encode_text_batch(self, texts: List[str]) -> List[List[float]]:
        """批量编码文本"""
        inputs = [{
            "text": text,
            "instruction": "Retrieve images or text relevant to the user's query."
        } for text in texts]
        embeddings = self.model.process(inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.detach().cpu().to(torch.float32).numpy().tolist()
        
    def encode_image_batch(self, images: List[Image.Image]) -> List[List[float]]:
        """批量编码图像"""
        inputs = [{"image": img} for img in images]
        embeddings = self.model.process(inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.detach().cpu().to(torch.float32).numpy().tolist()
