# core/ocr/region_ocr_engine.py
"""
Region-level OCR orchestrator.

Strategy: single whole-image OCR → detect UIED regions → assign text
lines to regions by spatial overlap. This avoids N+1 separate OCR calls
(per-region + remainder) and is ~1.3x faster with identical quality.

When region_detector is None (e.g. ENABLE_UIED=false), output is always
exactly **one** full-frame region so ocr_regions / storage shape matches
the multi-region path (region_index=0, bbox = full image).
"""
import json
import multiprocessing
import os
import tempfile
import time
from typing import List, Optional

from PIL import Image

from core.ocr.ocr_engine import OCRResult, OCREngine
from core.ocr.region_detector import UIEDRegionDetector
from utils.logger import setup_logger

logger = setup_logger(__name__)

_mp_ctx = multiprocessing.get_context("spawn")


def _single_full_image_regions(w: int, h: int, result: OCRResult) -> List[dict]:
    """One canonical region covering the full image (UIED off or no UIED regions detected).

    Keeps the same list[dict] shape as the multi-region path: always length 1 for this mode.
    """
    tj = (result.text_json or "").strip()
    if not tj:
        tj = json.dumps({"words": []}, ensure_ascii=False)
    return [
        {
            "region_index": 0,
            "bbox": [0, 0, w, h],
            "text": result.text or "",
            "text_json": tj,
            "ocr_confidence": float(result.confidence)
            if result.confidence is not None
            else 0.0,
        }
    ]


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


def _persistent_ocr_worker(req_conn, res_conn, engine_type: str, use_region_detector: bool):
    """Long-running OCR worker in a child process.

    Parent can still kill this worker on timeout; subsequent requests recreate it.
    """
    from PIL import Image as _Image
    from core.ocr.ocr_engine import create_ocr_engine as _create

    ocr_eng = _create(engine_type)
    detector = None
    if use_region_detector:
        from core.ocr.region_detector import UIEDRegionDetector as _Det
        detector = _Det()

    engine = RegionOCREngine.__new__(RegionOCREngine)
    engine.ocr_engine = ocr_eng
    engine.region_detector = detector
    engine._last_timeout_ts = 0.0
    engine._worker_proc = None

    while True:
        try:
            tmp_path = req_conn.recv()
            if tmp_path is None:
                break
            img = _Image.open(tmp_path)
            results = engine._recognize_regions_inner(img)
            res_conn.send(results)
        except EOFError:
            break
        except Exception:
            try:
                res_conn.send([])
            except Exception:
                break


class RegionOCREngine:
    """Detects UI regions, then assigns whole-image OCR text to each region."""

    def __init__(
        self,
        ocr_engine: OCREngine,
        region_detector: Optional[UIEDRegionDetector] = None,
    ):
        self.ocr_engine = ocr_engine
        self.region_detector = region_detector
        self._last_timeout_ts: float = 0.0
        self._worker_proc = None
        self._parent_req_conn = None
        self._parent_res_conn = None

    _OCR_TIMEOUT_SECONDS = 15
    _COOLDOWN_AFTER_TIMEOUT = 90

    def _ensure_worker(self):
        """Start or restart persistent OCR worker."""
        if (
            self._worker_proc is not None
            and self._worker_proc.is_alive()
            and self._parent_req_conn is not None
            and self._parent_res_conn is not None
        ):
            return
        self._kill_worker()
        engine_name = getattr(self.ocr_engine, "engine_name", "auto")
        has_detector = self.region_detector is not None
        self._parent_req_conn, child_req_conn = _mp_ctx.Pipe()
        child_res_conn, self._parent_res_conn = _mp_ctx.Pipe()
        self._worker_proc = _mp_ctx.Process(
            target=_persistent_ocr_worker,
            args=(child_req_conn, child_res_conn, engine_name, has_detector),
            daemon=True,
        )
        self._worker_proc.start()
        logger.info(f"OCR worker subprocess started (pid={self._worker_proc.pid})")

    def recognize_regions(self, image: Image.Image) -> List[dict]:
        """
        Whole-image OCR + UIED region detection + spatial text assignment.

        Uses Pipe-based IPC with Connection.poll() for reliable timeout
        (Queue.get(timeout) is unreliable on macOS).
        Circuit breaker skips OCR for a cooldown period after a timeout.
        """
        if self._last_timeout_ts > 0:
            elapsed_since_timeout = time.monotonic() - self._last_timeout_ts
            if elapsed_since_timeout < self._COOLDOWN_AFTER_TIMEOUT:
                logger.warning(
                    f"RegionOCR circuit breaker OPEN — skipping OCR "
                    f"({elapsed_since_timeout:.0f}s / {self._COOLDOWN_AFTER_TIMEOUT}s cooldown)"
                )
                return []
            else:
                logger.info("RegionOCR circuit breaker CLOSED — resuming OCR")
                self._last_timeout_ts = 0.0

        self._ensure_worker()
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.png')
        try:
            os.close(tmp_fd)
            image.save(tmp_path, format="PNG")
            self._parent_req_conn.send(tmp_path)

            # watchdog loop: do short polls, and enforce wall-clock deadline ourselves
            # (avoids relying on a single long poll(timeout) call).
            deadline = time.monotonic() + self._OCR_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                if self._parent_res_conn.poll(0.2):
                    return self._parent_res_conn.recv()
                if self._worker_proc is None or (not self._worker_proc.is_alive()):
                    logger.error("RegionOCR: worker exited unexpectedly; restarting")
                    self._activate_circuit_breaker()
                    return []

            img_w, img_h = image.size
            logger.error(
                f"RegionOCR: timeout after {self._OCR_TIMEOUT_SECONDS}s "
                f"(image={img_w}x{img_h}), killing worker and activating circuit breaker"
            )
            self._activate_circuit_breaker()
            return []
        except (BrokenPipeError, OSError, EOFError) as e:
            logger.error(f"RegionOCR: pipe/process error ({e}), activating circuit breaker")
            self._activate_circuit_breaker()
            return []
        except Exception as e:
            logger.error(f"RegionOCR: unexpected error ({e}), activating circuit breaker")
            self._activate_circuit_breaker()
            return []
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _kill_worker(self):
        if self._worker_proc is not None:
            try:
                if self._worker_proc.is_alive():
                    self._worker_proc.terminate()
                    self._worker_proc.join(timeout=2)
                    if self._worker_proc.is_alive():
                        self._worker_proc.kill()
                        self._worker_proc.join(timeout=2)
            except Exception:
                pass
        self._worker_proc = None
        for conn in (self._parent_req_conn, self._parent_res_conn):
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        self._parent_req_conn = None
        self._parent_res_conn = None

    def _activate_circuit_breaker(self):
        """Kill worker and enter cooldown."""
        self._last_timeout_ts = time.monotonic()
        self._kill_worker()

    def _recognize_regions_inner(self, image: Image.Image) -> List[dict]:
        """Actual OCR logic (may block)."""
        w, h = image.size
        has_bbox_support = hasattr(self.ocr_engine, "recognize_with_bboxes")

        if has_bbox_support and self.region_detector is not None:
            return self._recognize_via_assign(image, w, h)

        if self.region_detector is None:
            result = self.ocr_engine.recognize(image)
            return _single_full_image_regions(w, h, result)

        return self._recognize_per_region(image, w, h)

    def _recognize_via_assign(self, image: Image.Image, w: int, h: int) -> List[dict]:
        """Strategy B: single whole-image OCR → assign text to UIED regions."""
        regions_meta = self.region_detector.detect(image)

        if not regions_meta:
            result = self.ocr_engine.recognize(image)
            return _single_full_image_regions(w, h, result)

        text_bboxes = self.ocr_engine.recognize_with_bboxes(image)

        if not text_bboxes:
            return []

        region_bboxes = [m["bbox"] for m in regions_meta]
        region_texts, unassigned = _assign_text_to_regions(text_bboxes, region_bboxes)

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
            return _single_full_image_regions(w, h, result)

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
