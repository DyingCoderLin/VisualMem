# core/ocr/region_ocr_engine.py
"""
Region-level OCR orchestrator.

Strategy: single whole-image OCR → detect UIED regions → assign text
lines to regions by spatial overlap. This avoids N+1 separate OCR calls
(per-region + remainder) and is ~1.3x faster with identical quality.
"""
import json
from typing import List, Optional

from PIL import Image

from core.ocr.ocr_engine import OCREngine
from core.ocr.region_detector import UIEDRegionDetector
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _assign_text_to_regions(
    text_bboxes: List[dict],
    region_bboxes: List[List[int]],
) -> tuple:
    """Assign OCR text lines to UIED regions by center-point containment.

    Returns (region_texts, unassigned):
        region_texts: {region_index: [text_bbox, ...]}
        unassigned: [text_bbox, ...]
    """
    region_texts = {i: [] for i in range(len(region_bboxes))}
    unassigned = []

    for tb in text_bboxes:
        tx1, ty1, tx2, ty2 = tb["bbox"]
        t_cx, t_cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2

        best_region = -1
        best_overlap = 0
        for i, (rx1, ry1, rx2, ry2) in enumerate(region_bboxes):
            if rx1 <= t_cx <= rx2 and ry1 <= t_cy <= ry2:
                overlap = (
                    max(0, min(tx2, rx2) - max(tx1, rx1))
                    * max(0, min(ty2, ry2) - max(ty1, ry1))
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_region = i

        if best_region >= 0:
            region_texts[best_region].append(tb)
        else:
            unassigned.append(tb)

    return region_texts, unassigned


class RegionOCREngine:
    """Detects UI regions, then assigns whole-image OCR text to each region."""

    def __init__(
        self,
        ocr_engine: OCREngine,
        region_detector: Optional[UIEDRegionDetector] = None,
    ):
        self.ocr_engine = ocr_engine
        self.region_detector = region_detector

    def recognize_regions(self, image: Image.Image) -> List[dict]:
        """
        Whole-image OCR + UIED region detection + spatial text assignment.

        Returns:
            List of dicts, each with:
                region_index, bbox, text, text_json, ocr_confidence
        """
        w, h = image.size
        has_bbox_support = hasattr(self.ocr_engine, "recognize_with_bboxes")

        # --- Fast path: OCR engine supports bboxes AND we have a region detector ---
        if has_bbox_support and self.region_detector is not None:
            return self._recognize_via_assign(image, w, h)

        # --- Fallback: no region detector or no bbox support ---
        if self.region_detector is None:
            result = self.ocr_engine.recognize(image)
            return [{
                "region_index": 0,
                "bbox": [0, 0, w, h],
                "text": result.text,
                "text_json": result.text_json,
                "ocr_confidence": result.confidence,
            }]

        # Fallback: region detector but no bbox support → per-region OCR
        return self._recognize_per_region(image, w, h)

    def _recognize_via_assign(self, image: Image.Image, w: int, h: int) -> List[dict]:
        """Strategy B: single whole-image OCR → assign text to UIED regions."""
        # 1. Detect UIED regions
        regions_meta = self.region_detector.detect(image)

        if not regions_meta:
            result = self.ocr_engine.recognize(image)
            return [{
                "region_index": 0,
                "bbox": [0, 0, w, h],
                "text": result.text,
                "text_json": result.text_json,
                "ocr_confidence": result.confidence,
            }]

        # 2. Single whole-image OCR with bounding boxes
        text_bboxes = self.ocr_engine.recognize_with_bboxes(image)

        if not text_bboxes:
            return []

        # 3. Assign text lines to regions by spatial overlap
        region_bboxes = [m["bbox"] for m in regions_meta]
        region_texts, unassigned = _assign_text_to_regions(text_bboxes, region_bboxes)

        # 4. Build results
        results = []
        for i, meta in enumerate(regions_meta):
            texts = region_texts.get(i, [])
            combined_text = "\n".join(t["text"] for t in texts)
            avg_conf = (
                sum(t["confidence"] for t in texts) / len(texts) if texts else 0.0
            )
            words = [
                {"text": t["text"], "confidence": round(t["confidence"], 4)}
                for t in texts
            ]
            results.append({
                "region_index": meta["region_index"],
                "bbox": meta["bbox"],
                "text": combined_text,
                "text_json": json.dumps({"words": words}, ensure_ascii=False),
                "ocr_confidence": avg_conf,
            })

        # Unassigned text → remainder region
        if unassigned:
            remainder_text = "\n".join(t["text"] for t in unassigned)
            avg_conf = sum(t["confidence"] for t in unassigned) / len(unassigned)
            next_index = max(r["region_index"] for r in results) + 1 if results else 0
            words = [
                {"text": t["text"], "confidence": round(t["confidence"], 4)}
                for t in unassigned
            ]
            results.append({
                "region_index": next_index,
                "bbox": [0, 0, w, h],
                "text": remainder_text,
                "text_json": json.dumps({"words": words}, ensure_ascii=False),
                "ocr_confidence": avg_conf,
                "is_remainder": True,
            })

        logger.debug(
            f"RegionOCR: {len(results)} regions "
            f"({len(text_bboxes)} OCR lines, {len(unassigned)} unassigned), "
            f"image {w}x{h}"
        )
        return results

    def _recognize_per_region(self, image: Image.Image, w: int, h: int) -> List[dict]:
        """Fallback: per-region crop OCR + remainder (for engines without bbox support)."""
        regions_meta = self.region_detector.detect(image)

        if not regions_meta:
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

        # Remainder OCR
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

        logger.debug(f"RegionOCR (per-region fallback): {len(results)} regions, image {w}x{h}")
        return results
