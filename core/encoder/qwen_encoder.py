# core/encoder/qwen_encoder.py
import torch
import numpy as np
from typing import List, Union, Dict, Any, Optional
from PIL import Image
from core.encoder.base_encoder import MultiModalEncoderInterface
from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__)

class Qwen3VLEmbedder:
    """
    Qwen3-VL-Embedding-2B 模型封装类
    参考用户提供的示例代码实现
    """
    def __init__(self, model_name_or_path: str, device: str = "auto", **kwargs):
        from transformers import AutoProcessor, AutoModel
        
        # 自动选择设备
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"Loading Qwen3-VL-Embedding model from {model_name_or_path} on {self.device}")
        
        # 加载处理器和模型
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            **kwargs
        ).to(self.device)
        self.model.eval()
        
    def process(self, inputs: List[Dict[str, Any]]) -> torch.Tensor:
        """
        处理输入并返回 embedding
        
        Args:
            inputs: 包含 text, instruction, image 等字段的字典列表
            
        Returns:
            torch.Tensor: Embedding 矩阵
        """
        # 注意：这里的具体实现可能取决于模型的 remote_code
        # 如果模型已经在本地下载并包含 modeling_qwen3_vl_embedding.py，
        # 则此方法可能由模型类直接提供。
        # 这里的实现尝试模拟标准 multimodal embedding 流程
        
        # 如果模型类有 process 方法，直接调用
        if hasattr(self.model, 'process'):
            return self.model.process(inputs)
            
        # 否则尝试通用流程
        with torch.no_grad():
            # 这里简化处理，实际 Qwen3-VL-Embedding 可能有更复杂的 prompt 构造
            processed_inputs = self.processor(
                text=[i.get("text", "") for i in inputs],
                images=[i.get("image") for i in inputs if "image" in i],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            outputs = self.model(**processed_inputs)
            # 通常取最后一个 token 或者 pooler_output
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output
            return outputs.last_hidden_state[:, 0, :] # 简化的 CLS pooling

class QwenEncoder(MultiModalEncoderInterface):
    """
    Qwen Encoder: 用于 Qwen3-VL-Embedding-2B 模型
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-Embedding-2B", device: str = None):
        """
        初始化 Qwen 编码器
        """
        # 自动选择设备
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
                
        super().__init__(model_name, device)
        
        try:
            # 优先尝试从本地或库导入 Qwen3VLEmbedder
            # 如果用户环境中已经有这个类，则直接使用
            try:
                from src.models.qwen3_vl_embedding import Qwen3VLEmbedder as OriginalEmbedder
                self.embedder = OriginalEmbedder(model_name_or_path=model_name, device=device)
            except ImportError:
                # 否则使用我们定义的包装类
                self.embedder = Qwen3VLEmbedder(model_name_or_path=model_name, device=device)
            
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
        embeddings = self.embedder.process(inputs)
        # 归一化并转为 list
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings[0].cpu().numpy().tolist()
        
    def encode_text(self, text: str) -> List[float]:
        """编码文本（查询）"""
        inputs = [{
            "text": text,
            "instruction": "Retrieve images or text relevant to the user's query."
        }]
        embeddings = self.embedder.process(inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings[0].cpu().numpy().tolist()
        
    def encode_text_batch(self, texts: List[str]) -> List[List[float]]:
        inputs = [{
            "text": text,
            "instruction": "Retrieve images or text relevant to the user's query."
        } for text in texts]
        embeddings = self.embedder.process(inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().tolist()
        
    def encode_image_batch(self, images: List[Image.Image]) -> List[List[float]]:
        inputs = [{"image": img} for img in images]
        embeddings = self.embedder.process(inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().tolist()
