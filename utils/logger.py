# utils/logger.py
import logging
import logging.handlers
import sys
import os
import fcntl
from datetime import datetime
from pathlib import Path
from queue import Queue

# 日志级别映射
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "backend_server.log"


class ColorFormatter(logging.Formatter):
    """Colorize store_frame log lines by phase.

    - START / done lines: cyan (bold)
    - step= detail lines (embedding, capture_windows, win_ocr, win_embedding): dim white
    - cluster_assign / cluster_done / label=: yellow (bold)
    - ERROR/WARNING: red/yellow as usual
    """

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    # Colors
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    WHITE = "\033[37m"

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        levelno = record.levelno

        # ERROR / WARNING always override
        if levelno >= logging.ERROR:
            return f"{self.RED}{self.BOLD}{msg}{self.RESET}"
        if levelno >= logging.WARNING:
            return f"{self.YELLOW}{msg}{self.RESET}"

        # Colorize store_frame lines by phase
        raw = record.getMessage()
        if "store_frame:" in raw:
            if "START " in raw or " done in " in raw:
                return f"{self.CYAN}{self.BOLD}{msg}{self.RESET}"
            if "cluster_assign" in raw or "cluster_done" in raw:
                return f"{self.YELLOW}{self.BOLD}{msg}{self.RESET}"
            if "step=" in raw or "win=" in raw:
                return f"{self.DIM}{msg}{self.RESET}"

        return msg


class NonBlockingStreamHandler(logging.StreamHandler):
    """StreamHandler that never blocks the caller.

    Sets the underlying file descriptor to non-blocking mode so that a full
    pipe buffer (e.g. when Electron stops reading stdout) raises EAGAIN instead
    of blocking the Python process.  On EAGAIN the log message is silently
    dropped — this is far better than freezing the entire backend.
    """

    def __init__(self, stream=None):
        super().__init__(stream)
        self._tried_nonblock = False

    def _ensure_nonblocking(self):
        if self._tried_nonblock:
            return
        self._tried_nonblock = True
        try:
            fd = self.stream.fileno()
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        except (AttributeError, OSError):
            pass

    def emit(self, record: logging.LogRecord) -> None:
        self._ensure_nonblocking()
        try:
            msg = self.format(record)
            self.stream.write(msg + self.terminator)
            self.stream.flush()
        except BlockingIOError:
            pass  # pipe full — drop this message rather than blocking
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


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
    - 主输出: 直接写文件 (logs/backend_server.log)，不依赖 Electron stdout 管道
    - 辅助输出: stdout（非阻塞模式），管道满时自动丢弃而非卡死
    - 默认INFO级别，可通过环境变量 LOG_LEVEL 或 config.LOG_LEVEL 修改
    """
    logger = logging.getLogger(name)

    log_level = level if level is not None else get_log_level()
    logger.setLevel(log_level)

    if logger.handlers:
        # Allow raising/lowering level when the same logger is reconfigured (e.g. CLI -v)
        if level is not None:
            for h in logger.handlers:
                h.setLevel(log_level)
        return logger

    fmt_str = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'

    # --- Primary handler: direct file write (never blocks on pipe) ---
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(_LOG_FILE), encoding='utf-8')
    file_handler.setLevel(log_level)
    force_color = os.environ.get("LOG_COLOR", "").lower() in ("1", "true", "yes")
    if force_color:
        file_handler.setFormatter(ColorFormatter(fmt_str, datefmt=datefmt))
    else:
        file_handler.setFormatter(logging.Formatter(fmt_str, datefmt=datefmt))
    logger.addHandler(file_handler)

    # --- Secondary handler: stdout (non-blocking, for Electron pipe / terminal) ---
    stdout_handler = NonBlockingStreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    use_color = force_color or (hasattr(sys.stdout, 'isatty') and sys.stdout.isatty())
    if use_color:
        stdout_handler.setFormatter(ColorFormatter(fmt_str, datefmt=datefmt))
    else:
        stdout_handler.setFormatter(logging.Formatter(fmt_str, datefmt=datefmt))
    logger.addHandler(stdout_handler)

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


