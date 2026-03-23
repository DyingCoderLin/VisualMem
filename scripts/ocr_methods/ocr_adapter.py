"""
Platform-adaptive OCR adapter.

Re-exports from core.ocr for backward compatibility with scripts.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.ocr.platform_ocr import AppleVisionOCR, WindowsOCR
from core.ocr.ocr_engine import OCREngine, OCRResult, PytesseractOCR, create_ocr_engine


def create_platform_ocr(lang: str = "chi_sim+eng", force_engine: str = None) -> OCREngine:
    """
    Create the best available OCR engine for the current platform.

    Args:
        lang: Language hint (pytesseract format like "chi_sim+eng").
        force_engine: Force a specific engine: "apple_vision", "windows_ocr",
                      "pytesseract". None = auto-detect.

    Returns:
        OCREngine instance
    """
    if force_engine:
        return create_ocr_engine(force_engine, lang=lang)
    return create_ocr_engine("auto", lang=lang)
