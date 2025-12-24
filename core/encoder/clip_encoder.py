# core/encoder/clip_encoder.py
from typing import List
from PIL.Image import Image
from core.encoder.base_encoder import MultiModalEncoderInterface
from utils.logger import setup_logger

logger = setup_logger(__name__)

class CLIPEncoder(MultiModalEncoderInterface):
    """
    CLIP Encoder: 用于生成图像和文本的embedding
    模型轻量级（~600MB），可以在本地快速运行
    """
    
    def __init__(self, model_name: str = "google/siglip-large-patch16-384", device: str = None):
        """
        初始化CLIP模型
        
        Args:
            model_name: 模型名称或路径
            device: 计算设备（None 为自动选择）
        
        推荐模型:
        - openai/clip-vit-base-patch32 (512维, 快速)
        - google/siglip-large-patch16-384 (1024维, 86M参数)
        - google/siglip-base-patch16-224 (768维, 更强)
        - OFA-Sys/chinese-clip-vit-base-patch16 (中文友好)
        """
        # 延迟导入（避免模块导入时立即需要）
        import torch
        from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoProcessor
        
        # 自动选择设备
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        # 调用父类构造函数
        super().__init__(model_name, device)
        
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading CLIP model: {model_name} (this may take a while if downloading for the first time)")
        logger.info(f"Using device: {self.device}")
        
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
                    self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
                except ImportError as e:
                    logger.warning(f"Fast processor not available ({e}), falling back to slow processor")
                    self.processor = AutoProcessor.from_pretrained(model_name)
            
            self.model.to(self.device)
            self.model.eval()  # 设置为评估模式
            
            # 获取embedding维度
            # CLIP 使用 projection_dim，SigLIP 使用 text_config.hidden_size
            if "siglip" in model_name.lower():
                self.embedding_dim = self.model.config.text_config.hidden_size
            else:
                self.embedding_dim = self.model.config.projection_dim
            logger.info(f"CLIP model loaded successfully. Embedding dim: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def get_embedding_dim(self) -> int:
        """获取 embedding 维度"""
        return self.embedding_dim
    
    def encode_text_batch(self, texts: List[str]) -> List[List[float]]:
        """批量编码文本"""
        return [self.encode_text(text) for text in texts]
    
    def encode_image_batch(self, images: List[Image]) -> List[List[float]]:
        """批量编码图像"""
        return [self.encode_image(image) for image in images]
    
    def encode_image(self, image: Image) -> List[float]:
        """
        将图像编码为embedding向量
        
        Args:
            image: PIL Image对象
            
        Returns:
            embedding向量 (List[float])
        """
        try:
            import torch  # 延迟导入
            with torch.no_grad():
                # 预处理图像
                inputs = self.processor(
                    images=image, 
                    return_tensors="pt"
                ).to(self.device)
                
                # 获取图像特征
                image_features = self.model.get_image_features(**inputs)
                
                # 归一化（CLIP的标准做法）
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # 转换为list
                embedding = image_features[0].cpu().numpy().tolist()
                
                logger.debug(f"Generated image embedding: dim={len(embedding)}")
                return embedding
                
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            # 返回零向量作为fallback
            return [0.0] * self.embedding_dim
    
    def encode_text(self, text: str) -> List[float]:
        """
        将文本编码为embedding向量（用于查询）
        
        Args:
            text: 查询文本
            
        Returns:
            embedding向量 (List[float])
        """
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
                
                # 归一化
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # 转换为list
                embedding = text_features[0].cpu().numpy().tolist()
                
                logger.debug(f"Generated text embedding for query: '{text[:50]}...'")
                return embedding
                
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            # 返回零向量作为fallback
            return [0.0] * self.embedding_dim
    
    def compute_similarity(self, image_embedding: List[float], text_embedding: List[float]) -> float:
        """
        计算图像和文本embedding的相似度
        
        Args:
            image_embedding: 图像embedding
            text_embedding: 文本embedding
            
        Returns:
            相似度分数 (0-1)
        """
        import torch  # 延迟导入
        # 转换为tensor
        img_tensor = torch.tensor(image_embedding)
        txt_tensor = torch.tensor(text_embedding)
        
        # 计算余弦相似度
        similarity = torch.cosine_similarity(img_tensor, txt_tensor, dim=0).item()
        
        # 归一化到0-1
        similarity = (similarity + 1) / 2
        
        return similarity

