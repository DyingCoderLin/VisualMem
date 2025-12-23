# utils/logger.py
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# 日志级别映射
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}


def get_log_level() -> int:
    """获取日志级别（从环境变量或 config）"""
    try:
        from config import config
        log_level_str = config.LOG_LEVEL
    except ImportError:
        log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    return LOG_LEVEL_MAP.get(log_level_str, logging.INFO)


def setup_logger(name: str = "visualmem", level: int = None) -> logging.Logger:
    """
    设置统一的日志记录器
    - 输出到终端（stdout）
    - 默认INFO级别，可通过环境变量 LOG_LEVEL 或 config.LOG_LEVEL 修改
    
    Args:
        name: logger 名称
        level: 日志级别（如果不指定，则从配置读取）
    """
    logger = logging.getLogger(name)
    
    # 获取日志级别
    log_level = level if level is not None else get_log_level()
    logger.setLevel(log_level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    return logger


# ============================================
# 全局 logger 实例
# ============================================
# 使用方法:
#   from utils.logger import logger
#   logger.info("信息")
#   logger.debug("调试信息")
#   logger.warning("警告")
#   logger.error("错误")
# ============================================
logger = setup_logger("visualmem")


def setup_generate_logger(log_file: str = "logs/generate_info.log") -> logging.Logger:
    """
    设置专门记录VLM生成信息的日志记录器
    - 只写入文件，不输出到终端
    - 记录VLM调用的响应时间等关键信息
    """
    logger = logging.getLogger("generate_info")
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 确保日志目录存在
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建简洁的格式化器（专门用于generate信息）
    formatter = logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    # 防止日志传播到root logger（避免输出到终端）
    logger.propagate = False
    
    return logger


