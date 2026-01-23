# core/capture/__init__.py
from .base_capturer import AbstractCapturer
from .screenshot_capturer import ScreenshotCapturer
from .window_capturer import (
    WindowCapturer,
    calculate_image_hash,
    should_skip_window,
    _USE_RUST as USE_RUST_CAPTURE,
)
from .recording_coordinator import (
    RecordingCoordinator,
    RecordingConfig,
    RecordingStats,
)

# Try to import Rust module directly for advanced usage
try:
    import screencap_rs
except ImportError:
    screencap_rs = None

__all__ = [
    "AbstractCapturer",
    "ScreenshotCapturer",
    "WindowCapturer",
    "calculate_image_hash",
    "should_skip_window",
    "RecordingCoordinator",
    "RecordingConfig",
    "RecordingStats",
    "USE_RUST_CAPTURE",
    "screencap_rs",
]
