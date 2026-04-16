#!/usr/bin/env python3
"""
查询方式自动选择示例

根据查询类型和配置，自动选择最佳查询方式
这个是cursor自己生成的，还没研究它是干啥的
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

from config import config
from utils.logger import setup_logger

logger = setup_logger("query_selector")


class SmartQuerySelector:
    """
    智能查询选择器
    
    根据配置和查询类型，自动选择最佳查询方式
    """
    
    def __init__(self):
        """初始化查询器"""
        self.storage_mode = config.STORAGE_MODE
        self.enable_ocr = config.ENABLE_OCR
        
        # 初始化可用的存储
        self.lancedb_available = False
        self.sqlite_available = False
        
        # 初始化 LanceDB（如果是 vector 模式）
        if self.storage_mode == "vector":
            try:
                from core.encoder import create_encoder
                from core.storage.lancedb_storage import LanceDBStorage
                from core.retrieval.image_retriever import ImageRetriever
                
                self.encoder = create_encoder(model_name=config.EMBEDDING_MODEL)
                self.lancedb_storage = LanceDBStorage(
                    db_path=config.LANCEDB_PATH,
                    embedding_dim=self.encoder.embedding_dim
                )
                self.retriever = ImageRetriever(self.encoder, self.lancedb_storage)
                self.lancedb_available = True
                logger.info("LanceDB retriever initialized")
            except Exception as e:
                logger.warning(f"LanceDB not available: {e}")
        
        # 初始化 SQLite（如果启用了 OCR）
        if self.enable_ocr:
            try:
                from core.storage.sqlite_storage import SQLiteStorage
                self.sqlite_storage = SQLiteStorage(db_path=config.OCR_DB_PATH)
                self.sqlite_available = True
                logger.info("SQLite storage initialized")
            except Exception as e:
                logger.warning(f"SQLite not available: {e}")
        
        # 打印可用的查询方式
        print("\n" + "="*60)
        print("智能查询选择器")
        print("="*60)
        print(f"\n配置:")
        print(f"  - Storage Mode: {self.storage_mode}")
        print(f"  - OCR Enabled: {self.enable_ocr}")
        print(f"\n可用查询方式:")
        if self.lancedb_available:
            print("  - LanceDB 语义搜索（文本→图像、图像→图像）")
        if self.sqlite_available:
            print("  - SQLite 文本搜索（OCR Fallback）")
        if not self.lancedb_available and not self.sqlite_available:
            print("  警告: 仅支持时间范围查询")
        print("="*60)
    
    def query(
        self,
        query_text: Optional[str] = None,
        query_image: Optional[Image.Image] = None,
        top_k: int = 10,
        strategy: str = "auto"
    ) -> List[Dict]:
        """
        智能查询
        
        Args:
            query_text: 查询文本
            query_image: 查询图像
            top_k: 返回结果数量
            strategy: 查询策略
                - "auto": 自动选择
                - "semantic": 强制语义搜索（LanceDB）
                - "text": 强制文本搜索（SQLite）
                - "hybrid": 混合搜索
                
        Returns:
            查询结果列表
        """
        # 自动选择策略
        if strategy == "auto":
            strategy = self._select_strategy(query_text, query_image)
            print(f"\n自动选择策略: {strategy}")
        
        # 执行查询
        if strategy == "semantic":
            return self._query_semantic(query_text, query_image, top_k)
        elif strategy == "text":
            return self._query_text(query_text, top_k)
        elif strategy == "hybrid":
            return self._query_hybrid(query_text, top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _select_strategy(
        self,
        query_text: Optional[str],
        query_image: Optional[Image.Image]
    ) -> str:
        """
        自动选择查询策略
        
        决策逻辑：
        1. 如果是图像查询 → semantic
        2. 如果包含精确关键词（error, command等） → text
        3. 如果都可用 → hybrid
        4. 否则用可用的方式
        """
        # 图像查询 → 必须用语义
        if query_image is not None:
            if self.lancedb_available:
                return "semantic"
            else:
                raise ValueError("Image query requires LanceDB (vector mode)")
        
        # 文本查询
        if query_text:
            # 检测是否是精确关键词查询
            exact_keywords = ["error", "warning", "exception", "git", "command", "def ", "class "]
            is_exact_query = any(kw in query_text.lower() for kw in exact_keywords)
            
            if is_exact_query and self.sqlite_available:
                return "text"  # 精确查询用 SQLite
            
            # 两者都可用 → 混合策略
            if self.lancedb_available and self.sqlite_available:
                return "hybrid"
            
            # 只有一个可用
            if self.lancedb_available:
                return "semantic"
            elif self.sqlite_available:
                return "text"
        
        raise ValueError("No query method available")
    
    def _query_semantic(
        self,
        query_text: Optional[str],
        query_image: Optional[Image.Image],
        top_k: int
    ) -> List[Dict]:
        """语义搜索（LanceDB）"""
        if not self.lancedb_available:
            raise ValueError("LanceDB not available")
        
        print(f"\n使用 LanceDB 语义搜索")
        
        if query_image:
            results = self.retriever.retrieve_by_image(query_image, top_k=top_k)
        elif query_text:
            results = self.retriever.retrieve_by_text(query_text, top_k=top_k)
        else:
            raise ValueError("Either query_text or query_image must be provided")
        
        return results
    
    def _query_text(self, query_text: str, top_k: int) -> List[Dict]:
        """文本搜索（SQLite）"""
        if not self.sqlite_available:
            raise ValueError("SQLite not available")
        
        print(f"\n使用 SQLite 文本搜索")
        
        results = self.sqlite_storage.search_by_text(query_text, limit=top_k)
        return results
    
    def _query_hybrid(self, query_text: str, top_k: int) -> List[Dict]:
        """混合搜索（SQLite 快速过滤 + LanceDB 语义排序）"""
        if not (self.lancedb_available and self.sqlite_available):
            raise ValueError("Hybrid search requires both LanceDB and SQLite")
        
        print(f"\n使用混合搜索策略")
        print(f"  Step 1: SQLite 快速过滤候选集...")
        
        # Step 1: SQLite 文本过滤（快速）
        ocr_results = self.sqlite_storage.search_by_text(query_text, limit=100)
        print(f"  → 找到 {len(ocr_results)} 个候选")
        
        if len(ocr_results) == 0:
            print(f"  Step 2: SQLite 无结果，使用纯语义搜索...")
            return self._query_semantic(query_text, None, top_k)
        
        # Step 2: LanceDB 语义搜索
        print(f"  Step 2: LanceDB 语义排序...")
        semantic_results = self.retriever.retrieve_by_text(query_text, top_k=top_k * 2)
        
        # Step 3: 合并结果（优先语义相关的候选）
        candidate_paths = {r['image_path'] for r in ocr_results}
        ocr_dict = {r['image_path']: r for r in ocr_results}
        
        final_results = []
        for sem_r in semantic_results:
            if sem_r['image_path'] in candidate_paths:
                # 合并 OCR 数据
                ocr_data = ocr_dict[sem_r['image_path']]
                sem_r['ocr_text'] = ocr_data.get('ocr_text', '')
                sem_r['ocr_confidence'] = ocr_data.get('ocr_confidence', 0.0)
                final_results.append(sem_r)
                
                if len(final_results) >= top_k:
                    break
        
        print(f"  → 返回 {len(final_results)} 个结果")
        return final_results


def example_usage():
    """示例：不同类型的查询"""
    
    selector = SmartQuerySelector()
    
    print("\n" + "="*60)
    print("示例1：精确文本查询（自动选择 SQLite）")
    print("="*60)
    
    try:
        results = selector.query(
            query_text="Error: connection timeout",
            top_k=5,
            strategy="auto"
        )
        
        print(f"\n找到 {len(results)} 个结果：")
        for i, r in enumerate(results[:3], 1):
            print(f"\n{i}. {r.get('image_path', 'N/A')}")
            if 'distance' in r:
                print(f"   距离: {r['distance']:.4f}")
            if 'ocr_text' in r:
                print(f"   文本: {r['ocr_text'][:80]}...")
    
    except Exception as e:
        print(f"错误: {e}")
    
    print("\n" + "="*60)
    print("示例2：模糊概念查询（自动选择 LanceDB 或混合）")
    print("="*60)
    
    try:
        results = selector.query(
            query_text="机器学习 模型训练",
            top_k=5,
            strategy="auto"
        )
        
        print(f"\n找到 {len(results)} 个结果：")
        for i, r in enumerate(results[:3], 1):
            print(f"\n{i}. {r.get('image_path', 'N/A')}")
            if 'distance' in r:
                print(f"   距离: {r['distance']:.4f}")
    
    except Exception as e:
        print(f"错误: {e}")
    
    print("\n" + "="*60)
    print("示例3：强制使用特定策略")
    print("="*60)
    
    try:
        # 强制使用语义搜索
        results = selector.query(
            query_text="python code",
            top_k=5,
            strategy="semantic"  # 强制语义
        )
        print(f"语义搜索返回 {len(results)} 个结果")
    except Exception as e:
        print(f"语义搜索不可用: {e}")
    
    try:
        # 强制使用文本搜索
        results = selector.query(
            query_text="python",
            top_k=5,
            strategy="text"  # 强制文本
        )
        print(f"文本搜索返回 {len(results)} 个结果")
    except Exception as e:
        print(f"文本搜索不可用: {e}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("智能查询选择器 - 自动选择最佳查询方式")
    print("="*60)
    print("\n根据以下因素自动选择：")
    print("  1. 配置（STORAGE_MODE, ENABLE_OCR）")
    print("  2. 查询类型（精确文本 vs 模糊概念）")
    print("  3. 可用性（LanceDB, SQLite）")
    
    try:
        example_usage()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        logger.error(f"示例运行失败: {e}", exc_info=True)
        print(f"\n错误: {e}")


if __name__ == "__main__":
    main()
