# Session Planning & Debug Log

This file tracks all modifications and debugging within development sessions.
New entries are prepended (newest first).

---

## 2026-03-23: OCR Compare Demo

### Modifications

#### Fix: objc duplicate libavdevice warning in YOLO-World method
- **Problem**: `opencv-python` and `av` (PyAV) both bundle `libavdevice.dylib`. macOS runtime emits ObjC warnings about duplicate class implementations (`AVFFrameReceiver`, `AVFAudioReceiver`).
- **Root cause**: `cv2/.dylibs/libavdevice.61.x` and `av/.dylibs/libavdevice.62.x` both register the same ObjC classes. Warning fires at C level when the second dylib is loaded.
- **Attempted**: Swapping to `opencv-python-headless` — still bundles dylibs. Deleting the dylib — cv2 links against it, breaks import.
- **Final fix**: In `demo_ocr_compare.py`, pre-import both `cv2` and `av` with stderr (fd 2) redirected to `/dev/null` during the import. After both are loaded, restore stderr. Warning only fires once per process on first load.
- **File**: `scripts/demo_ocr_compare.py` (top-level, before other imports)

#### Fix: UIED int32 JSON serialization error
- **Problem**: `cv2.connectedComponentsWithStats` returns numpy `int32` values, which `json.dump` cannot serialize.
- **Fix**: Cast bbox coordinates to Python `int()` in `_detect_regions()`.
- **File**: `scripts/ocr_methods/method_uied.py`

#### Feature: Platform-adaptive OCR engine
- **Problem**: pytesseract is slow and low-quality. macOS/Windows have native OCR engines.
- **Fix**: Created `scripts/ocr_methods/ocr_adapter.py` with auto-detection:
  - macOS → Apple Vision (pyobjc-framework-Vision)
  - Windows → UWP OCR (winocr)
  - Linux → pytesseract fallback
- **Files modified**: All method files updated to use `create_platform_ocr()` instead of hardcoded `PytesseractOCR`.
- **New arg**: `--ocr-engine` in `demo_ocr_compare.py` (auto/apple_vision/windows_ocr/pytesseract)

#### Feature: OCR comparison demo
- **New files created**:
  - `scripts/demo_ocr_compare.py` — Main script
  - `scripts/ocr_methods/__init__.py`
  - `scripts/ocr_methods/method_bruteforce.py` — Method 1: full-image OCR
  - `scripts/ocr_methods/method_yoloworld.py` — Method 2: YOLO-World v2 region detection + OCR
  - `scripts/ocr_methods/method_uied.py` — Method 3: Traditional CV (UIED-inspired) region detection + OCR
  - `scripts/ocr_methods/ocr_adapter.py` — Platform-adaptive OCR engine selector
- **Dependencies added**: `ultralytics`, `pyobjc-framework-Vision`, `pyobjc-framework-Quartz`

### Known Issues
- YOLO-World v2 (`yolov8l-worldv2.pt`) pretrained on COCO cannot meaningfully detect UI elements. It typically finds 0-1 regions covering the entire screen. Need fine-tuned model — see `training_guide.md`.
- UIED-CV misses text in open/unbounded regions (areas not enclosed by edges). This is inherent to the gradient+connected-component approach.
- `screencap_rs.capture_screen_with_windows()` takes ~60s on this machine and returns 0 windows — may be a permissions or Rust module issue.
