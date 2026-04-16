"""
文本 Embedding 编码器（基于 CLIP）

使用 CLIP 模型的文本编码能力，与 CLIPEncoder 使用相同的底层模型
这样可以控制变量，便于对比 Dense/Sparse/Hybrid 检索的性能
"""

from typing import List, Optional
from core.encoder.base_encoder import TextEncoderInterface
from utils.logger import setup_logger

logger = setup_logger(__name__)


class TextEncoder(TextEncoderInterface):
    """
    文本 Embedding 编码器（基于 CLIP）
    
    使用 CLIP 模型的文本塔进行文本编码
    
    优势：
    - 与 CLIPEncoder 使用相同的底层模型（控制变量）
    - Embedding 与图像在同一空间（支持跨模态）
    - 轻量高效，与图像检索保持一致
    """
    
    def __init__(
        self,
        model_name: str = "google/siglip-large-patch16-384",
        device: Optional[str] = None
    ):
        """
        初始化文本编码器
        
        Args:
            model_name: CLIP 模型名称
            device: 计算设备 ("cuda", "cpu", "mps" 或 None 自动选择)
        """
        # 延迟导入（避免模块导入时立即需要）
        import torch
        from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoProcessor
        
        # 自动选择设备
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        # 调用父类构造函数
        super().__init__(model_name, device)
        
        self.model_name = model_name
        self.device = device
        
        logger.info(f"正在加载 TextEncoder (基于 CLIP): {model_name} on {device}")
        
        try:
            # 根据模型名称判断使用 CLIP 还是 SigLIP
            if "siglip" in model_name.lower():
                # SigLIP 模型使用 AutoModel
                self.model = AutoModel.from_pretrained(model_name)
                try:
                    self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
                except ImportError as e:
                    logger.warning(f"Fast processor not available ({e}), falling back to slow processor")
                    self.processor = AutoProcessor.from_pretrained(model_name)
            else:
                # 标准 CLIP 模型
                self.model = CLIPModel.from_pretrained(model_name)
                try:
                    self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
                except ImportError as e:
                    logger.warning(f"Fast processor not available ({e}), falling back to slow processor")
                    self.processor = CLIPProcessor.from_pretrained(model_name)
            
            self.model.to(self.device)
            self.model.eval()
            
            # 获取 embedding 维度
            # CLIP 使用 projection_dim，SigLIP 使用 text_config.hidden_size
            if "siglip" in model_name.lower():
                self.embedding_dim = self.model.config.text_config.hidden_size
            else:
                self.embedding_dim = self.model.config.projection_dim
            
            logger.info(f"TextEncoder 加载成功")
            logger.info(f"  - 模型: {model_name}")
            logger.info(f"  - 设备: {device}")
            logger.info(f"  - 向量维度: {self.embedding_dim}")
            logger.info(f"  - 底层: CLIP 文本塔（与 CLIPEncoder 共享模型）")
            
        except Exception as e:
            logger.error(f"加载 TextEncoder 失败: {e}")
            raise
    
    def encode_text(self, text: str) -> List[float]:
        """
        编码单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            embedding 向量（List[float]）
        """
        if not text or not text.strip():
            logger.warning("输入文本为空，返回零向量")
            return [0.0] * self.embedding_dim
        
        try:
            import torch  # 延迟导入
            with torch.no_grad():
                # 预处理文本
                inputs = self.processor(
                    text=text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # 获取文本特征
                text_features = self.model.get_text_features(**inputs)
                
                # 归一化（CLIP 标准做法）
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # 转换为 list
                embedding = text_features[0].cpu().numpy().tolist()
                
                logger.debug(f"生成文本 embedding: dim={len(embedding)}")
                return embedding
                
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            return [0.0] * self.embedding_dim
    
    def encode_text_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小（CLIP 不支持太大的 batch）
            
        Returns:
            embedding 向量列表
        """
        if not texts:
            logger.warning("输入文本列表为空")
            return []
        
        # 过滤空文本
        valid_texts = [t if t and t.strip() else " " for t in texts]
        
        try:
            logger.debug(f"批量编码 {len(texts)} 个文本 (batch_size={batch_size})")
            
            all_embeddings = []
            
            # 分批处理
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i+batch_size]
                
                import torch  # 延迟导入
                with torch.no_grad():
                    # 预处理文本批次
                    inputs = self.processor(
                        text=batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.device)
                    
                    # 获取文本特征
                    text_features = self.model.get_text_features(**inputs)
                    
                    # 归一化
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # 转换为 list
                    batch_embeddings = text_features.cpu().numpy().tolist()
                    all_embeddings.extend(batch_embeddings)
            
            logger.debug(f"批量编码完成: {len(all_embeddings)} 个 embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"批量编码失败: {e}")
            # 返回零向量
            return [[0.0] * self.embedding_dim for _ in texts]
    
    def get_embedding_dim(self) -> int:
        """获取 embedding 维度"""
        return self.embedding_dim


def create_text_encoder(
    model_name: str = "google/siglip-large-patch16-384",
    device: Optional[str] = None,
    **kwargs
) -> TextEncoder:
    """
    创建文本编码器的便捷函数
    
    Args:
        model_name: CLIP 模型名称
        device: 计算设备
        **kwargs: 其他参数
        
    Returns:
        TextEncoder 实例
        
    Examples:
        # 使用默认 CLIP 模型
        encoder = create_text_encoder()
        
        # 使用指定模型
        encoder = create_text_encoder(model_name="openai/clip-vit-large-patch14")
    """
    return TextEncoder(model_name=model_name, device=device)


if __name__ == "__main__":
    # 测试代码
    print("\n" + "="*60)
    print("文本编码器测试（基于 CLIP）")
    print("="*60)
    
    # 1. 创建编码器
    encoder = create_text_encoder()
    
    # 2. 单个文本编码
    text = "这是一段测试文本，用于验证 CLIP 文本编码器的功能。"
    embedding = encoder.encode_text(text)
    print(f"\n文本: {text}")
    print(f"Embedding 维度: {len(embedding)}")
    print(f"Embedding 前5维: {embedding[:5]}")
    
    # 3. 批量编码
    texts = [
        "机器学习是人工智能的一个分支",
        "深度学习使用神经网络进行训练",
        "自然语言处理处理人类语言"
    ]
    embeddings = encoder.encode_text_batch(texts)
    print(f"\n批量编码 {len(texts)} 个文本")
    print(f"每个 embedding 维度: {len(embeddings[0])}")
    
    # 4. 计算相似度
    import numpy as np
    sim_01 = np.dot(embeddings[0], embeddings[1])
    sim_02 = np.dot(embeddings[0], embeddings[2])
    print(f"\n相似度测试:")
    print(f"  文本0 vs 文本1: {sim_01:.4f}")
    print(f"  文本0 vs 文本2: {sim_02:.4f}")
    
    print("\n" + "="*60)
    print("TextEncoder 使用 CLIP 模型，与 CLIPEncoder 共享底层")
    print("="*60)
