# core/capture/base_capturer.py
from abc import ABC, abstractmethod
from typing import Optional
from utils.data_models import ScreenFrame

class AbstractCapturer(ABC):
    """
    抽象捕捉器基类
    定义了所有捕捉器必须实现的接口
    """
    
    @abstractmethod
    def capture(self) -> Optional[ScreenFrame]:
        """
        捕捉当前屏幕/窗口
        返回 ScreenFrame 对象，如果捕捉失败则返回 None
        """
        pass


