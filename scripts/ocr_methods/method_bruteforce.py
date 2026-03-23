"""
Method 1: Brute-force OCR
Directly run pytesseract on the full image without any region segmentation.
Same approach as the current VisualMem project.
"""
import json
import time
from PIL import Image

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.ocr_methods.ocr_adapter import create_platform_ocr


_ocr_engine = None

def _get_ocr(lang: str, force_engine: str = None):
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = create_platform_ocr(lang=lang, force_engine=force_engine)
    return _ocr_engine


def run_bruteforce_ocr(image: Image.Image, output_path: str, lang: str = "chi_sim+eng", force_engine: str = None) -> dict:
    """
    Run full-image OCR without any region segmentation.

    Args:
        image: PIL Image to OCR
        output_path: Path to save JSON result
        lang: Tesseract language string

    Returns:
        dict with text, confidence, words, timing
    """
    ocr = _get_ocr(lang, force_engine)

    start = time.perf_counter()
    result = ocr.recognize(image)
    elapsed = time.perf_counter() - start

    output = {
        "method": "bruteforce",
        "elapsed_ms": round(elapsed * 1000, 2),
        "text": result.text,
        "confidence": result.confidence,
        "text_json": json.loads(result.text_json) if result.text_json else {},
        "engine": result.engine,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output
