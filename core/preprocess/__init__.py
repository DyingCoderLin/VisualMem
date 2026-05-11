"""Frame preprocessing helpers and factory functions."""

from typing import Optional

from .base_preprocessor import AbstractPreprocessor
from .frame_diff import (
    FrameDiffDetector,
    FrameDiffResult,
    calculate_histogram,
    compare_histograms,
    calculate_ssim,
)


def create_preprocessor(preprocessor_type: Optional[str] = None) -> AbstractPreprocessor:
    """Create a configured frame preprocessor without importing experiments eagerly."""
    from config import config

    kind = (preprocessor_type or config.PREPROCESSOR_TYPE or "simple").strip().lower()
    if kind == "simple":
        from .simple_filter import SimpleFilter

        return SimpleFilter(config.SIMPLE_FILTER_DIFF_THRESHOLD)
    if kind in {"vllm", "vlm"}:
        from .vllm_filter import VLLMFilter

        return VLLMFilter()
    raise ValueError(f"Unknown preprocessor type: {kind}")


def __getattr__(name: str):
    if name in {"SimpleFilter", "calculate_normalized_rms_diff"}:
        from .simple_filter import SimpleFilter, calculate_normalized_rms_diff

        return {
            "SimpleFilter": SimpleFilter,
            "calculate_normalized_rms_diff": calculate_normalized_rms_diff,
        }[name]
    if name == "VLLMFilter":
        from .vllm_filter import VLLMFilter

        return VLLMFilter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AbstractPreprocessor",
    "FrameDiffDetector",
    "FrameDiffResult",
    "calculate_histogram",
    "compare_histograms",
    "calculate_ssim",
    "SimpleFilter",
    "VLLMFilter",
    "calculate_normalized_rms_diff",
    "create_preprocessor",
]
