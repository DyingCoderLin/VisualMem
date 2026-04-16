"""
检索器抽象基类

为所有检索器（文本、图像、多模态）提供统一接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Literal, Union
from pathlib import Path
from PIL import Image
from config import config
from core.encoder.base_encoder import BaseEncoder
from utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseRetriever(ABC):
    """
    检索器抽象基类
    
    所有检索器（TextRetriever, ImageRetriever 等）都应继承此基类
    并实现其抽象方法，以提供统一的检索接口
    
    支持三种检索模式：
    - Dense: 纯语义搜索（基于 embedding 相似度）
    - Sparse: 全文搜索（FTS, BM25）
    - Hybrid: 混合搜索（Dense + Sparse + Reranker）
    """
    
    # 默认数据库路径（子类应覆盖）
    DEFAULT_DB_PATH = "./visualmem_db"
    DEFAULT_TABLE_NAME = "data"
    
    def __init__(
        self,
        encoder: Optional[BaseEncoder] = None,
        db_path: Optional[str] = None,
        table_name: Optional[str] = None,
        default_reranker: str = "linear"
    ):
        """
        初始化检索器
        
        Args:
            encoder: 编码器实例（可选，仅 dense/hybrid 搜索需要）
            db_path: 数据库路径（如果为 None，使用默认路径）
            table_name: 表名（如果为 None，使用默认表名）
            default_reranker: 默认 reranker
        """
        self.encoder = encoder
        self.db_path = Path(db_path) if db_path else Path(self.DEFAULT_DB_PATH)
        self.table_name = table_name if table_name else self.DEFAULT_TABLE_NAME
        self.default_reranker_name = default_reranker
        
        # 子类需要初始化的属性
        self.db = None
        self.table = None
        self._rerankers = {}
        
        logger.info(f"{self.__class__.__name__} 初始化")
        logger.info(f"  - 数据库路径: {self.db_path}")
        logger.info(f"  - 表名: {self.table_name}")
        if encoder:
            logger.info(f"  - Encoder: {encoder.__class__.__name__}")
        else:
            logger.info(f"  - Encoder: None (仅支持 Sparse/FTS 搜索)")
    
    @abstractmethod
    def _connect_db(self):
        """连接数据库（子类实现）"""
        pass
    
    @abstractmethod
    def _get_reranker(self, reranker_name: str):
        """获取或创建 reranker 实例（子类实现）"""
        pass
    
    @abstractmethod
    def ensure_fts_index(self, text_field: str = "text") -> bool:
        """
        确保 FTS 索引存在
        
        Args:
            text_field: 要建立索引的文本字段
            
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    def retrieve_dense(
        self,
        query: Union[str, Image.Image],
        top_k: int = 10,
        filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Dense 检索（纯语义搜索）
        
        Args:
            query: 查询输入（文本或图像）
            top_k: 返回结果数量
            filter: SQL 风格的过滤条件
            
        Returns:
            检索结果列表
        """
        pass
    
    @abstractmethod
    def retrieve_sparse(
        self,
        query: str,
        top_k: int = 10,
        text_field: str = "text",
        filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Sparse 检索（FTS 全文搜索）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            text_field: 要搜索的文本字段
            filter: SQL 风格的过滤条件
            
        Returns:
            检索结果列表
        """
        pass
    
    @abstractmethod
    def retrieve_hybrid(
        self,
        query: Union[str, Image.Image],
        top_k: int = 10,
        text_field: str = "text",
        reranker: Optional[str] = None,
        filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Hybrid 检索（混合搜索）
        
        Args:
            query: 查询输入（文本或图像）
            top_k: 返回结果数量
            text_field: 要搜索的文本字段
            reranker: Reranker 名称
            filter: SQL 风格的过滤条件
            
        Returns:
            检索结果列表
        """
        pass
    
    def retrieve(
        self,
        query: Union[str, Image.Image],
        top_k: int = 10,
        mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
        **kwargs
    ) -> List[Dict]:
        """
        统一检索接口
        
        Args:
            query: 查询输入
            top_k: 返回结果数量
            mode: 检索模式
            **kwargs: 其他参数
            
        Returns:
            检索结果列表
        """
        if mode == "dense":
            return self.retrieve_dense(query, top_k, **kwargs)
        elif mode == "sparse":
            if not isinstance(query, str):
                logger.warning("Sparse 检索仅支持文本查询，回退到 Dense 检索")
                return self.retrieve_dense(query, top_k, **kwargs)
            return self.retrieve_sparse(query, top_k, **kwargs)
        elif mode == "hybrid":
            return self.retrieve_hybrid(query, top_k, **kwargs)
        else:
            logger.error(f"未知检索模式: {mode}")
            return []
    
    def get_stats(self) -> Dict:
        """
        获取数据库统计信息
        
        Returns:
            统计信息字典
        """
        if self.table is None:
            return {"status": "表不存在"}
        
        try:
            count = self.table.count_rows()
            return {
                "retriever_type": self.__class__.__name__,
                "db_path": str(self.db_path),
                "table_name": self.table_name,
                "total_rows": count,
                "embedding_dim": self.encoder.get_embedding_dim(),
                "encoder": self.encoder.__class__.__name__,
                "default_reranker": self.default_reranker_name
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"status": "错误", "error": str(e)}
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"db={self.db_path}, "
            f"table={self.table_name}, "
            f"encoder={self.encoder.__class__.__name__})"
        )


class TextRetrieverInterface(BaseRetriever):
    """
    文本检索器接口
    
    专门用于纯文本检索的检索器应继承此接口
    """
    
    DEFAULT_DB_PATH = config.TEXT_LANCEDB_PATH
    DEFAULT_TABLE_NAME = "ocr_texts"
    
    def retrieve_dense(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[str] = None
    ) -> List[Dict]:
        """文本语义检索"""
        pass


class ImageRetrieverInterface(BaseRetriever):
    """
    图像检索器接口
    
    专门用于图像检索的检索器应继承此接口
    """
    
    DEFAULT_DB_PATH = "./visualmem_db"
    DEFAULT_TABLE_NAME = "screen_analyses"
    
    def retrieve_dense(
        self,
        query: Image.Image,
        top_k: int = 10,
        filter: Optional[str] = None
    ) -> List[Dict]:
        """图像相似度检索"""
        pass


class MultiModalRetrieverInterface(BaseRetriever):
    """
    多模态检索器接口
    
    同时支持文本和图像检索的检索器（如 CLIP）应继承此接口
    """
    
    DEFAULT_DB_PATH = "./visualmem_db"
    DEFAULT_TABLE_NAME = "screen_analyses"
    
    def retrieve_dense(
        self,
        query: Union[str, Image.Image],
        top_k: int = 10,
        filter: Optional[str] = None
    ) -> List[Dict]:
        """
        多模态语义检索
        
        根据输入类型自动选择：
        - 文本 -> 文本到图像检索
        - 图像 -> 图像到图像检索
        """
        pass
    
    def retrieve_by_text(self, text: str, top_k: int = 10, **kwargs) -> List[Dict]:
        """文本到图像检索"""
        return self.retrieve_dense(text, top_k, **kwargs)
    
    def retrieve_by_image(self, image: Image.Image, top_k: int = 10, **kwargs) -> List[Dict]:
        """图像到图像检索"""
        return self.retrieve_dense(image, top_k, **kwargs)
    
    def retrieve_by_image_path(self, image_path: str, top_k: int = 10, **kwargs) -> List[Dict]:
        """通过图像路径检索"""
        image = Image.open(image_path)
        return self.retrieve_by_image(image, top_k, **kwargs)
