# core/ocr/platform_ocr.py
"""
Platform-native OCR engines.

- macOS: Apple Vision framework (via pyobjc)
- Windows: Windows.Media.Ocr (UWP OCR via winocr)
"""
import platform
import time
import json

from PIL import Image
from core.ocr.ocr_engine import OCREngine, OCRResult
from utils.logger import setup_logger

logger = setup_logger(__name__)

_SYSTEM = platform.system()


class AppleVisionOCR(OCREngine):
    """macOS native OCR using Apple Vision framework via pyobjc."""

    def __init__(self, lang: str = "zh-Hans"):
        self.engine_name = "apple_vision"
        self.lang = lang
        try:
            import Vision
            import Quartz
            self._Vision = Vision
            self._Quartz = Quartz
            logger.info("Apple Vision OCR initialized")
        except ImportError:
            raise ImportError(
                "pyobjc-framework-Vision required. Install with:\n"
                "  pip install pyobjc-framework-Vision pyobjc-framework-Quartz"
            )

    def _run_request(self, image: Image.Image):
        """Run Apple Vision text request and return raw observations."""
        import io
        from Foundation import NSData

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        ns_data = NSData.dataWithBytes_length_(png_bytes, len(png_bytes))
        handler = self._Vision.VNImageRequestHandler.alloc().initWithData_options_(
            ns_data, None
        )

        request = self._Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(
            self._Vision.VNRequestTextRecognitionLevelAccurate
        )
        request.setRecognitionLanguages_(["zh-Hans", "zh-Hant", "en"])
        request.setUsesLanguageCorrection_(True)

        success = handler.performRequests_error_([request], None)
        if not success[0]:
            logger.warning(f"Apple Vision OCR failed: {success[1]}")
            return []
        return request.results() or []

    def recognize(self, image: Image.Image) -> OCRResult:
        """Run Apple Vision text recognition on a PIL Image."""
        start = time.perf_counter()

        try:
            observations = self._run_request(image)
            texts = []
            confidences = []
            words = []

            for obs in observations:
                candidate = obs.topCandidates_(1)
                if candidate and len(candidate) > 0:
                    text = candidate[0].string()
                    conf = candidate[0].confidence()
                    texts.append(text)
                    confidences.append(conf)
                    words.append({"text": text, "confidence": round(conf, 4)})

            full_text = "\n".join(texts)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            elapsed = time.perf_counter() - start
            logger.debug(f"Apple Vision OCR: {elapsed*1000:.1f}ms, {len(full_text)} chars")

            return OCRResult(
                text=full_text,
                confidence=avg_conf,
                text_json=json.dumps({"words": words}, ensure_ascii=False),
                engine=self.engine_name,
            )

        except Exception as e:
            logger.error(f"Apple Vision OCR error: {e}")
            return OCRResult(text="", confidence=0.0, engine=self.engine_name)

    def recognize_with_bboxes(self, image: Image.Image) -> list:
        """Run OCR and return text lines with pixel bounding boxes.

        Returns list of {"text", "confidence", "bbox": [x1, y1, x2, y2]}.
        """
        try:
            observations = self._run_request(image)
            img_w, img_h = image.size
            results = []
            for obs in observations:
                candidate = obs.topCandidates_(1)
                if not candidate or len(candidate) == 0:
                    continue
                text = candidate[0].string()
                conf = float(candidate[0].confidence())
                bb = obs.boundingBox()
                # Apple Vision: normalized coords, origin at bottom-left
                x = bb.origin.x * img_w
                y = (1 - bb.origin.y - bb.size.height) * img_h
                w = bb.size.width * img_w
                h = bb.size.height * img_h
                results.append({
                    "text": text,
                    "confidence": conf,
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                })
            return results
        except Exception as e:
            logger.error(f"Apple Vision OCR (with bboxes) error: {e}")
            return []


class WindowsOCR(OCREngine):
    """Windows native OCR using Windows.Media.Ocr (UWP) via winocr."""

    def __init__(self, lang: str = "zh-Hans-CN"):
        self.engine_name = "windows_ocr"
        self.lang = lang
        try:
            import winocr
            self._winocr = winocr
            logger.info(f"Windows OCR initialized with lang={lang}")
        except ImportError:
            raise ImportError(
                "winocr required. Install with:\n"
                "  pip install winocr"
            )

    def recognize(self, image: Image.Image) -> OCRResult:
        """Run Windows UWP OCR on a PIL Image."""
        import asyncio
        import io

        start = time.perf_counter()
        try:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(
                self._winocr.recognize_png(png_bytes, lang=self.lang)
            )
            loop.close()

            texts = []
            words = []
            for line in result.get("lines", []):
                line_text = line.get("text", "")
                texts.append(line_text)
                for word in line.get("words", []):
                    words.append({
                        "text": word.get("text", ""),
                        "confidence": 1.0,
                    })

            full_text = "\n".join(texts)

            elapsed = time.perf_counter() - start
            logger.debug(f"Windows OCR: {elapsed*1000:.1f}ms, {len(full_text)} chars")

            return OCRResult(
                text=full_text,
                confidence=1.0,
                text_json=json.dumps({"words": words}, ensure_ascii=False),
                engine=self.engine_name,
            )
        except Exception as e:
            logger.error(f"Windows OCR error: {e}")
            return OCRResult(text="", confidence=0.0, engine=self.engine_name)
