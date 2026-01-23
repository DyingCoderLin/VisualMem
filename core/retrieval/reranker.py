# core/retrieval/reranker.py
"""
Reranker Module: 使用官方 Qwen3-VL-Reranker-2B 模型进行重排
"""
from typing import List, Dict
from .local_reranker import LocalReranker

class Reranker:
    def __init__(self):
        # 始终使用本地加载的模型
        self.instance = LocalReranker()
    
    def rerank(self, query: str, frames: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        对搜索结果进行重排
        """
        return self.instance.rerank(query, frames, top_k)
