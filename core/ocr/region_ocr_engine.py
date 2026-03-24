# core/ocr/region_ocr_engine.py
"""
Region-level OCR orchestrator.

Combines a region detector (UIED) with an OCR engine to perform
per-region text recognition. Falls back to whole-image OCR when
no region detector is provided.
"""
import json
from typing import List, Optional

from PIL import Image

from core.ocr.ocr_engine import OCREngine
from core.ocr.region_detector import UIEDRegionDetector
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RegionOCREngine:
    """Detects regions, then runs OCR on each region independently."""

    def __init__(
        self,
        ocr_engine: OCREngine,
        region_detector: Optional[UIEDRegionDetector] = None,
    ):
        self.ocr_engine = ocr_engine
        self.region_detector = region_detector

    def recognize_regions(self, image: Image.Image) -> List[dict]:
        """
        Detect regions + per-region OCR.

        Returns:
            List of dicts, each with:
                region_index, bbox, text, text_json, ocr_confidence
        When region_detector is None, returns a single whole-image region.
        """
        w, h = image.size

        if self.region_detector is None:
            # Whole-image fallback
            result = self.ocr_engine.recognize(image)
            return [{
                "region_index": 0,
                "bbox": [0, 0, w, h],
                "text": result.text,
                "text_json": result.text_json,
                "ocr_confidence": result.confidence,
            }]

        regions_meta = self.region_detector.detect(image)

        if not regions_meta:
            # No regions detected, fall back to whole image
            result = self.ocr_engine.recognize(image)
            return [{
                "region_index": 0,
                "bbox": [0, 0, w, h],
                "text": result.text,
                "text_json": result.text_json,
                "ocr_confidence": result.confidence,
            }]

        results = []
        for meta in regions_meta:
            bbox = meta["bbox"]
            x1, y1, x2, y2 = bbox
            crop = image.crop((x1, y1, x2, y2))
            ocr_result = self.ocr_engine.recognize(crop)
            results.append({
                "region_index": meta["region_index"],
                "bbox": bbox,
                "text": ocr_result.text,
                "text_json": ocr_result.text_json,
                "ocr_confidence": ocr_result.confidence,
            })

        # Remainder OCR: mask out detected regions, OCR the rest
        # to capture text in areas UIED missed.
        from PIL import ImageDraw
        remainder_img = image.copy()
        draw = ImageDraw.Draw(remainder_img)
        for meta in regions_meta:
            x1, y1, x2, y2 = meta["bbox"]
            draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))

        remainder_result = self.ocr_engine.recognize(remainder_img)
        if remainder_result.text.strip():
            next_index = max(r["region_index"] for r in results) + 1 if results else 0
            results.append({
                "region_index": next_index,
                "bbox": [0, 0, w, h],
                "text": remainder_result.text,
                "text_json": remainder_result.text_json,
                "ocr_confidence": remainder_result.confidence,
                "is_remainder": True,
            })

        logger.debug(f"RegionOCR: {len(results)} regions (incl. remainder), image {w}x{h}")
        return results
