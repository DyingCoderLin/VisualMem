# core/ocr/__init__.py

from .ocr_engine import (
    OCREngine,
    OCRResult,
    PytesseractOCR,
    DummyOCR,
    create_ocr_engine
)

__all__ = [
    'OCREngine',
    'OCRResult',
    'PytesseractOCR',
    'DummyOCR',
    'create_ocr_engine',
]

# platform_ocr classes are imported lazily by create_ocr_engine

