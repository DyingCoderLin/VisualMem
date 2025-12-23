# core/storage/base_storage.py
from abc import ABC, abstractmethod
from typing import List, Optional
from utils.data_models import VLMAnalysis

class AbstractStorage(ABC):
    """
    抽象存储基类
    定义了所有存储后端必须实现的接口
    """
    
    @abstractmethod
    def store(self, analysis: VLMAnalysis) -> bool:
        """
        存储一条VLM分析结果
        返回是否成功
        """
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[VLMAnalysis]:
        """
        根据查询搜索相关的分析结果
        返回最相关的top_k条结果
        """
        pass
    
    @abstractmethod
    def search_by_time(self, start_time, end_time) -> List[VLMAnalysis]:
        """
        根据时间范围搜索
        """
        pass


