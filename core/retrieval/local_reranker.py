# core/retrieval/local_reranker.py
import torch
from typing import List, Dict
from src.models.qwen3_vl_reranker import Qwen3VLReranker
from config import config
from utils.logger import setup_logger

logger = setup_logger("local_reranker")

# 全局单例，确保模型只加载一次
_model_instance = None

def get_rerank_model():
    global _model_instance
    if _model_instance is None:
        model_path = config.RERANK_MODEL  # 默认已在 config.py 改为 Qwen/Qwen3-VL-Reranker-2B
        logger.info(f"Loading Qwen3VLReranker from {model_path}")
        _model_instance = Qwen3VLReranker(
            model_name_or_path=model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa"
        )
    return _model_instance

class LocalReranker:
    def __init__(self):
        self.model = get_rerank_model()

    def rerank(self, query: str, frames: List[Dict], top_k: int = 10) -> List[Dict]:
        if not frames:
            return []

        # 构造输入，完全对齐官方示例格式
        documents = []
        for f in frames:
            doc = {}
            if f.get("image"):
                doc["image"] = f["image"]
            if f.get("ocr_text"):
                doc["text"] = f["ocr_text"]
            documents.append(doc)

        inputs = {
            "instruction": "Retrieve images or text relevant to the user's query.",
            "query": {"text": query},
            "documents": documents,
            "fps": 1.0, 
            "max_frames": 64
        }

        # 直接调用官方 process 方法
        scores = self.model.process(inputs)

        # 组装分数并排序
        for i, f in enumerate(frames):
            f["rerank_score"] = scores[i]

        frames.sort(key=lambda x: x["rerank_score"], reverse=True)
        return frames[:top_k]
