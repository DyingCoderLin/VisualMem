"""
文本检索器 - 支持 Dense、Sparse、Hybrid 三种检索方式

模仿 screenpipe 和 LanceDB 的混合搜索实现
- Dense: 纯向量语义搜索
- Sparse: FTS (BM25) 关键词搜索
- Hybrid: 结合 Dense + Sparse，使用 Reranker 重排序
"""

from typing import List, Dict, Optional, Literal
from config import config
from pathlib import Path
from core.encoder.text_encoder import TextEncoder
from core.retrieval.base_retriever import TextRetrieverInterface
from utils.logger import setup_logger

logger = setup_logger(__name__)


class TextRetriever(TextRetrieverInterface):
    """
    文本检索器
    
    支持三种检索模式：
    1. Dense (Vector): 纯语义搜索，基于 embedding 相似度
    2. Sparse (FTS): 全文搜索，基于 BM25 算法
    3. Hybrid: 混合搜索，结合 Dense + Sparse 并重排序
    
    Reranker 选项：
    - linear: LinearCombinationReranker（快速，默认）
    - rrf: RRFReranker（Reciprocal Rank Fusion）
    - cross-encoder: CrossEncoderReranker（最准确，但较慢）
    """
    
    # 设置默认路径
    from config import config
    DEFAULT_DB_PATH = config.TEXT_LANCEDB_PATH
    DEFAULT_TABLE_NAME = "ocr_texts"
    
    def __init__(
        self,
        encoder: Optional[TextEncoder] = None,
        db_path: Optional[str] = None,
        table_name: Optional[str] = None,
        default_reranker: str = "linear"
    ):
        """
        初始化文本检索器
        
        Args:
            encoder: 文本编码器（可选，仅 dense/hybrid 搜索需要，sparse 搜索不需要）
            db_path: LanceDB 数据库路径（None 使用默认）
            table_name: 表名（None 使用默认）
            default_reranker: 默认 reranker ("linear", "rrf", "cross-encoder")
        """
        # 调用父类构造函数
        super().__init__(encoder, db_path, table_name, default_reranker)
        
        # 连接数据库
        self._connect_db()
    
    def _connect_db(self):
        """连接数据库"""
        import lancedb  # 延迟导入
        self.db = lancedb.connect(str(self.db_path))
        
        # 检查表是否存在
        self.table = None
        if self.table_name in self.db.table_names():
            self.table = self.db.open_table(self.table_name)
            logger.info(f"已连接到表: {self.table_name}")
        else:
            logger.warning(f"表 {self.table_name} 不存在，请先创建或添加数据")
    
    def _get_reranker(self, reranker_name: str):
        """获取或创建 reranker 实例"""
        from lancedb.rerankers import (
            LinearCombinationReranker,
            RRFReranker,
            CrossEncoderReranker,
        )  # 延迟导入
        
        if reranker_name not in self._rerankers:
            if reranker_name == "linear":
                # LinearCombinationReranker: 默认 weight=0.7 (70% vector, 30% FTS)
                self._rerankers[reranker_name] = LinearCombinationReranker(weight=0.7)
                logger.info("使用 LinearCombinationReranker (weight=0.7)")
                
            elif reranker_name == "rrf":
                # RRFReranker: Reciprocal Rank Fusion
                self._rerankers[reranker_name] = RRFReranker()
                logger.info("使用 RRFReranker")
                
            elif reranker_name == "cross-encoder":
                # CrossEncoderReranker: 使用 cross-encoder 模型
                self._rerankers[reranker_name] = CrossEncoderReranker()
                logger.info("使用 CrossEncoderReranker (最准确但较慢)")
                
            else:
                logger.warning(f"未知 reranker: {reranker_name}，使用默认 linear")
                self._rerankers[reranker_name] = LinearCombinationReranker(weight=0.7)
        
        return self._rerankers[reranker_name]
    
    def ensure_fts_index(self, text_field: str = "text") -> bool:
        """
        确保 FTS 索引存在
        
        Args:
            text_field: 要建立索引的文本字段
            
        Returns:
            是否成功
        """
        if self.table is None:
            logger.error("表不存在，无法创建 FTS 索引")
            return False
        
        try:
            # LanceDB 会自动检查索引是否已存在
            self.table.create_fts_index(text_field, replace=False)
            logger.info(f"FTS 索引已就绪: {text_field}")
            return True
        except Exception as e:
            # 如果索引已存在，LanceDB 会抛出异常但不影响使用
            logger.debug(f"FTS 索引可能已存在: {e}")
            return True
    
    def retrieve_dense(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Dense 检索（纯向量语义搜索）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter: SQL 风格的过滤条件（可选）
            
        Returns:
            检索结果列表
        """
        if self.table is None:
            logger.error("表不存在，无法检索")
            return []
        
        logger.info(f"[Dense Search] 查询: '{query}' (top_k={top_k})")
        
        if self.encoder is None:
            logger.error("Dense 检索需要 encoder，但 encoder 未初始化")
            return []
        
        try:
            # 1. 生成查询 embedding
            query_embedding = self.encoder.encode_text(query)
            
            # 2. 向量搜索
            search_query = self.table.search(query_embedding, query_type="vector")
            
            # 3. 应用过滤条件
            if filter:
                search_query = search_query.where(filter)
            
            # 4. 限制结果数量并返回
            results = search_query.limit(top_k).to_list()
            
            logger.info(f"[Dense Search] 找到 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"Dense 检索失败: {e}")
            return []
    
    def retrieve_sparse(
        self,
        query: str,
        top_k: int = 10,
        text_field: str = "text",
        filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Sparse 检索（FTS 全文搜索，BM25）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            text_field: 要搜索的文本字段
            filter: SQL 风格的过滤条件（可选）
            
        Returns:
            检索结果列表
        """
        if self.table is None:
            logger.error("表不存在，无法检索")
            return []
        
        logger.info(f"[Sparse Search / FTS] 查询: '{query}' (top_k={top_k})")
        
        try:
            # 1. 确保 FTS 索引存在
            self.ensure_fts_index(text_field)
            
            # 2. FTS 搜索
            search_query = self.table.search(query, query_type="fts")
            
            # 3. 应用过滤条件
            if filter:
                search_query = search_query.where(filter)
            
            # 4. 限制结果数量并返回
            results = search_query.limit(top_k).to_list()
            
            logger.info(f"[Sparse Search] 找到 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"Sparse 检索失败: {e}")
            return []
    
    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 10,
        text_field: str = "text",
        reranker: Optional[str] = None,
        filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Hybrid 检索（混合搜索：Dense + Sparse + Reranker）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            text_field: 要搜索的文本字段
            reranker: Reranker 名称 ("linear", "rrf", "cross-encoder")
            filter: SQL 风格的过滤条件（可选）
            
        Returns:
            检索结果列表
        """
        if self.table is None:
            logger.error("表不存在，无法检索")
            return []
        
        # 使用默认 reranker
        if reranker is None:
            reranker = self.default_reranker_name
        
        logger.info(f"[Hybrid Search] 查询: '{query}' (top_k={top_k}, reranker={reranker})")
        
        if self.encoder is None:
            logger.error("Hybrid 检索需要 encoder，但 encoder 未初始化")
            return []
        
        try:
            # 1. 确保 FTS 索引存在
            self.ensure_fts_index(text_field)
            
            # 2. 生成查询 embedding
            query_embedding = self.encoder.encode_text(query)
            
            # 3. 获取 reranker
            reranker_instance = self._get_reranker(reranker)
            
            # 4. 混合搜索（需要同时传递 vector 和 text）
            # LanceDB hybrid search 需要 vector（dense）和 text（sparse）两部分
            search_query = (
                self.table.search(query_type="hybrid")
                .vector(query_embedding)  # Dense search
                .text(query)              # Sparse search (FTS)
            )
            
            # 5. 应用过滤条件
            if filter:
                search_query = search_query.where(filter)
            
            # 6. Rerank 并限制结果数量
            results = (
                search_query
                .rerank(reranker=reranker_instance)
                .limit(top_k)
                .to_list()
            )
            
            logger.info(f"[Hybrid Search] 找到 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"Hybrid 检索失败: {e}")
            return []
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
        **kwargs
    ) -> List[Dict]:
        """
        统一检索接口
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            mode: 检索模式 ("dense", "sparse", "hybrid")
            **kwargs: 其他参数传递给具体的检索方法
            
        Returns:
            检索结果列表
        """
        if mode == "dense":
            return self.retrieve_dense(query, top_k, **kwargs)
        elif mode == "sparse":
            return self.retrieve_sparse(query, top_k, **kwargs)
        elif mode == "hybrid":
            return self.retrieve_hybrid(query, top_k, **kwargs)
        else:
            logger.error(f"未知检索模式: {mode}")
            return []


def create_text_retriever(
    encoder: Optional[TextEncoder] = None,
    db_path: str = config.TEXT_LANCEDB_PATH,
    table_name: str = "ocr_texts",
    create_encoder: bool = False,
    **kwargs
) -> TextRetriever:
    """
    创建文本检索器的便捷函数
    
    Args:
        encoder: 文本编码器（如果为 None 且 create_encoder=False，则不创建 encoder，仅支持 sparse 搜索）
        db_path: 数据库路径
        table_name: 表名
        create_encoder: 如果为 True 且 encoder 为 None，则自动创建 encoder（用于 dense/hybrid 搜索）
        **kwargs: 其他参数
        
    Returns:
        TextRetriever 实例
        
    Note:
        - 如果只做 sparse 搜索（FTS），不需要 encoder，设置 create_encoder=False 即可
        - 如果要做 dense/hybrid 搜索，需要 encoder，设置 create_encoder=True 或传入 encoder
    """
    if encoder is None and create_encoder:
        from core.encoder.text_encoder import create_text_encoder
        encoder = create_text_encoder()
    
    return TextRetriever(
        encoder=encoder,
        db_path=db_path,
        table_name=table_name,
        **kwargs
    )


if __name__ == "__main__":
    # 测试代码
    print("\n" + "="*60)
    print("文本检索器测试")
    print("="*60)
    
    from core.encoder.text_encoder import create_text_encoder
    
    # 1. 创建编码器和检索器
    encoder = create_text_encoder()
    retriever = create_text_retriever(encoder=encoder, db_path="./test_text.db")
    
    # 2. 检查表是否存在
    if retriever.table is None:
        print("\n⚠️ 表不存在，请先运行数据准备脚本")
    else:
        print("\n✅ 表已连接")
        
        # 3. 测试统计信息
        stats = retriever.get_stats()
        print(f"\n数据库统计:")
        for k, v in stats.items():
            print(f"  • {k}: {v}")
        
        # 4. 测试检索（如果有数据）
        if stats.get("total_rows", 0) > 0:
            query = "机器学习"
            
            print(f"\n测试查询: '{query}'")
            
            # Dense
            print("\n--- Dense 检索 ---")
            results = retriever.retrieve_dense(query, top_k=3)
            for i, r in enumerate(results, 1):
                print(f"{i}. {r.get('text', 'N/A')[:50]}")
            
            # Sparse
            print("\n--- Sparse 检索 ---")
            results = retriever.retrieve_sparse(query, top_k=3)
            for i, r in enumerate(results, 1):
                print(f"{i}. {r.get('text', 'N/A')[:50]}")
            
            # Hybrid
            print("\n--- Hybrid 检索 ---")
            results = retriever.retrieve_hybrid(query, top_k=3)
            for i, r in enumerate(results, 1):
                print(f"{i}. {r.get('text', 'N/A')[:50]}")
    
    print("\n" + "="*60)

