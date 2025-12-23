# core/preprocess/base_preprocessor.py
from abc import ABC, abstractmethod
from typing import Optional
from utils.data_models import ScreenFrame

class AbstractPreprocessor(ABC):
    """
    抽象预处理器基类
    """
    
    @abstractmethod
    def process(self, frame: ScreenFrame) -> Optional[ScreenFrame]:
        """
        处理一个帧。
        如果返回 None，表示该帧被过滤掉，不应被送入VLM。
        如果返回 ScreenFrame，表示该帧通过，进行下一步处理。
        """
        pass


