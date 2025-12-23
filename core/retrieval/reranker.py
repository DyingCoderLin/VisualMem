# core/retrieval/reranker.py
"""
Reranker Module: Re-rank search results using multimodal models

Uses local model for reranking (directly loads model and uses function calls)

Based on paper method: Uses VLM to judge whether images contain query content
Score = exp(logit_Yes) / (exp(logit_Yes) + exp(logit_No))
"""
from typing import List, Dict

from utils.logger import setup_logger

logger = setup_logger("reranker")

# Lazy import to avoid loading model at import time
_local_reranker = None


class Reranker:
    """
    Rerank using multimodal models (local mode)
    """
    
    def __init__(self):
        """
        Initialize Reranker (always uses local mode)
        """
        logger.debug("Reranker using local mode (function calls)")
        self.local_reranker = self._get_local_reranker()
    
    def _get_local_reranker(self):
        """Get local reranker instance (singleton)"""
        global _local_reranker
        if _local_reranker is None:
            from .local_reranker import LocalReranker
            _local_reranker = LocalReranker()
        return _local_reranker
    
    def rerank(
        self,
        query: str,
        frames: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Re-rank search results
        
        Args:
            query: Query text
            frames: Search result list (each frame must contain 'image' field)
            top_k: Return top-k results (default 10)
        
        Returns:
            Sorted top-k frames list (sorted by score in descending order)
        """
        if not frames:
            return []
        
        return self.local_reranker.rerank(query, frames, top_k)

