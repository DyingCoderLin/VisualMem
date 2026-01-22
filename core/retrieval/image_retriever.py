# core/retrieval/image_retriever.py
"""
图像检索器：只使用 dense 向量搜索
基于 CLIP + LanceDB 实现多模态检索（文本→图像，图像→图像）
"""

import time
from typing import List, Dict, Optional, Union
from datetime import datetime
from PIL import Image
from pathlib import Path

from core.encoder import MultiModalEncoderInterface
from core.storage.lancedb_storage import LanceDBStorage
from core.retrieval.base_retriever import MultiModalRetrieverInterface
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ImageRetriever(MultiModalRetrieverInterface):
    """
    多模态图像检索器：基于 CLIP/Qwen + LanceDB 的 dense 向量搜索
    
    特点：
    - 只使用 dense 向量搜索（去掉 hybrid/sparse/BM25）
    - 使用 CLIP/Qwen 进行图像和文本编码（图文对齐）
    - 使用 LanceDB 作为向量数据库
    - 支持文本查询图像、图像查询图像
    """
    
    # 设置默认路径
    DEFAULT_DB_PATH = "./visualmem_db"
    DEFAULT_TABLE_NAME = "screen_analyses"
    
    def __init__(
        self,
        encoder: MultiModalEncoderInterface,
        storage: LanceDBStorage,
        db_path: Optional[str] = None,
        table_name: Optional[str] = None,
        top_k: int = 10,
    ):
        """
        初始化检索器
        
        Args:
            encoder: CLIP 编码器实例
            storage: LanceDB 存储实例
            db_path: 数据库路径（为了兼容抽象基类）
            table_name: 表名（为了兼容抽象基类）
            top_k: 默认返回的结果数量
        """
        # 调用父类构造函数
        super().__init__(encoder, db_path, table_name, default_reranker="linear")
        
        self.encoder = encoder
        self.storage = storage
        self.top_k = top_k
        
        logger.info(f"ImageRetriever initialized with top_k={top_k}")
        logger.info(f"Encoder model: {encoder.model_name}")
        logger.info(f"Embedding dim: {encoder.embedding_dim}")
    
    def _connect_db(self):
        """连接数据库（使用 storage，不直接连接）"""
        # ImageRetriever 使用 LanceDBStorage，不需要额外连接
        pass
    
    def _get_reranker(self, reranker_name: str):
        """获取 reranker（ImageRetriever 不使用 reranker）"""
        # ImageRetriever 只使用 dense 搜索，不需要 reranker
        return None
    
    def ensure_fts_index(self, text_field: str = "text") -> bool:
        """确保 FTS 索引（ImageRetriever 不支持 FTS）"""
        logger.warning("ImageRetriever 不支持 FTS 索引")
        return False
    
    def retrieve_dense(
        self,
        query: Union[str, Image.Image],
        top_k: Optional[int] = None,
        filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Dense 检索（支持文本或图像查询）
        
        Args:
            query: 查询输入（文本或图像）
            top_k: 返回结果数量
            filter: 过滤条件（暂不支持）
            
        Returns:
            检索结果列表
        """
        if isinstance(query, str):
            return self.retrieve_by_text(query, top_k)
        elif isinstance(query, Image.Image):
            return self.retrieve_by_image(query, top_k)
        else:
            logger.error(f"不支持的查询类型: {type(query)}")
            return []
    
    def retrieve_sparse(
        self,
        query: str,
        top_k: int = 10,
        text_field: str = "text",
        filter: Optional[str] = None
    ) -> List[Dict]:
        """Sparse 检索（ImageRetriever 不支持）"""
        logger.warning("ImageRetriever 不支持 Sparse 检索")
        return []
    
    def retrieve_hybrid(
        self,
        query: Union[str, Image.Image],
        top_k: int = 10,
        text_field: str = "text",
        reranker: Optional[str] = None,
        filter: Optional[str] = None
    ) -> List[Dict]:
        """Hybrid 检索（ImageRetriever 不支持，回退到 Dense）"""
        logger.warning("ImageRetriever 不支持 Hybrid 检索，使用 Dense 检索")
        return self.retrieve_dense(query, top_k, filter)
    
    def retrieve(
        self,
        query: Optional[str] = None,
        query_image: Optional[Image.Image] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        通用检索方法：支持文本或图像查询
        
        Args:
            query: 文本查询（可选）
            query_image: 图像查询（可选）
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
            
        Note:
            - query 和 query_image 至少提供一个
            - 如果都提供，优先使用 query_image
        """
        if query is None and query_image is None:
            logger.error("Must provide either query text or query image")
            return []
        
        if top_k is None:
            top_k = self.top_k
        
        start_time = time.perf_counter()
        time_breakdown = {}
        
        try:
            # 1. 编码查询
            if query_image is not None:
                logger.info(f"Retrieving by image, top_k={top_k}")
                encoding_start = time.perf_counter()
                query_embedding = self.encoder.encode_image(query_image)
                time_breakdown['encoding'] = (time.perf_counter() - encoding_start) * 1000
                logger.debug(f"Image encoding took {time_breakdown['encoding']:.2f}ms")
            else:
                logger.info(f"Retrieving by text: '{query[:50]}...', top_k={top_k}")
                encoding_start = time.perf_counter()
                query_embedding = self.encoder.encode_text(query)
                time_breakdown['encoding'] = (time.perf_counter() - encoding_start) * 1000
                logger.debug(f"Text encoding took {time_breakdown['encoding']:.2f}ms")
            
            # 2. Dense 向量搜索
            search_start = time.perf_counter()
            results = self.storage.search(query_embedding, top_k=top_k)
            time_breakdown['search'] = (time.perf_counter() - search_start) * 1000
            
            total_time = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"Retrieved {len(results)} results in {total_time:.2f}ms "
                f"(encoding: {time_breakdown['encoding']:.2f}ms, "
                f"search: {time_breakdown['search']:.2f}ms)"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve: {e}", exc_info=True)
            return []
    
    def retrieve_by_text(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        根据文本查询检索相似图像
        
        Args:
            query: 文本查询
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        return self.retrieve(query=query, query_image=None, top_k=top_k)
    
    def retrieve_by_image(
        self,
        query_image: Image.Image,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        根据查询图像检索相似图像
        
        Args:
            query_image: PIL Image 查询图像
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        return self.retrieve(query=None, query_image=query_image, top_k=top_k)
    
    def retrieve_by_image_path(
        self,
        image_path: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        根据图像文件路径检索相似图像（便捷方法）
        
        Args:
            image_path: 图像文件路径
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        try:
            image = Image.open(image_path)
            logger.info(f"Loaded query image from: {image_path}")
            return self.retrieve_by_image(image, top_k=top_k)
        except Exception as e:
            logger.error(f"Failed to load image from {image_path}: {e}")
            return []
    
    def retrieve_recent(
        self,
        limit: int = None,
    ) -> List[Dict]:
        """
        检索最近的图像（按时间排序）
        
        这是一个便捷方法，不需要查询图像，直接返回最近的帧
        
        Args:
            limit: 返回数量
            
        Returns:
            最近的帧列表
        """
        if limit is None:
            limit = self.top_k
        
        logger.info(f"Retrieving recent {limit} frames")
        
        try:
            results = self.storage.load_recent(limit=limit)
            logger.info(f"Retrieved {len(results)} recent frames")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve recent frames: {e}", exc_info=True)
            return []
    
    def retrieve_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict]:
        """
        按时间范围检索图像
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            时间范围内的帧列表
        """
        logger.info(f"Retrieving frames from {start_time} to {end_time}")
        
        try:
            results = self.storage.load_by_time_range(start_time, end_time)
            logger.info(f"Retrieved {len(results)} frames in time range")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve by time range: {e}", exc_info=True)
            return []
    
    def get_stats(self) -> Dict:
        """
        获取检索器统计信息
        
        Returns:
            统计信息字典
        """
        storage_stats = self.storage.get_stats()
        
        return {
            "retriever_type": "dense_only",
            "encoder_model": self.encoder.model_name,
            "embedding_dim": self.encoder.embedding_dim,
            "top_k": self.top_k,
            "supports_text_query": True,
            "supports_image_query": True,
            "storage_stats": storage_stats,
        }

