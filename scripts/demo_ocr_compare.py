#!/usr/bin/env python3
"""
OCR Comparison Demo: compare three OCR strategies on screen captures.

Methods:
  1. Brute-force: pytesseract on full image (current VisualMem approach)
  2. YOLO-World: open-vocabulary UI region detection + per-region OCR
  3. UIED-CV: traditional CV region detection (UIED-inspired) + per-region OCR

Usage:
  conda activate mobiagent
  python scripts/demo_ocr_compare.py --method all
  python scripts/demo_ocr_compare.py --method 2 --conf 0.15
  python scripts/demo_ocr_compare.py --method 1 3
"""
import argparse
import json
import os
import sys
import time
import re

# Suppress macOS ObjC duplicate class warnings from cv2 + av (PyAV) dylib conflict.
# Both packages bundle libavdevice with identical ObjC classes. Cosmetic only.
# The warning fires at C level when the second dylib loads, so we must suppress
# stderr (fd 2) during the import of both cv2 and av/torchvision.
if sys.platform == "darwin":
    _stderr_fd = os.dup(2)
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 2)
    try:
        import cv2  # noqa: F401
        import av  # noqa: F401
    except ImportError:
        pass
    finally:
        os.dup2(_stderr_fd, 2)
        os.close(_stderr_fd)
        os.close(_devnull_fd)

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from PIL import Image
from utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Screen capture
# ---------------------------------------------------------------------------

def capture_screen() -> dict:
    """
    Capture full screen + all application windows using the project's capturer.

    Returns:
        dict with 'full_screen' (PIL.Image) and 'windows' (list of dicts with
        'app_name', 'window_name', 'image').
    """
    from core.capture.window_capturer import WindowCapturer

    capturer = WindowCapturer(monitor_id=0)
    screen_obj = capturer.capture_screen_with_windows()

    if screen_obj is None:
        logger.error("Failed to capture screen")
        sys.exit(1)

    windows = []
    for w in screen_obj.windows:
        windows.append({
            "app_name": w.app_name,
            "window_name": w.window_name,
            "image": w.image,
        })

    logger.info(f"Captured full screen ({screen_obj.full_screen_image.size}) + {len(windows)} windows")
    return {
        "full_screen": screen_obj.full_screen_image,
        "windows": windows,
    }


def _safe_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    return re.sub(r'[^\w\-.]', '_', name)[:80]


# ---------------------------------------------------------------------------
# Save raw screenshots
# ---------------------------------------------------------------------------

def save_raw_screenshots(capture: dict, output_dir: str):
    """Save original screenshots to raw_screenshots/."""
    raw_dir = os.path.join(output_dir, "raw_screenshots")
    os.makedirs(raw_dir, exist_ok=True)

    capture["full_screen"].save(os.path.join(raw_dir, "full_screen.png"))

    win_dir = os.path.join(raw_dir, "windows")
    os.makedirs(win_dir, exist_ok=True)
    for w in capture["windows"]:
        fname = _safe_filename(f"{w['app_name']}_{w['window_name']}") + ".png"
        w["image"].save(os.path.join(win_dir, fname))

    logger.info(f"Raw screenshots saved to {raw_dir}")


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def run_method1(capture: dict, output_dir: str, lang: str, force_engine: str = None):
    """Method 1: Brute-force OCR."""
    from ocr_methods.method_bruteforce import run_bruteforce_ocr

    method_dir = os.path.join(output_dir, "method1_bruteforce")

    # Full screen
    result_full = run_bruteforce_ocr(
        capture["full_screen"],
        os.path.join(method_dir, "full_screen.json"),
        lang=lang,
        force_engine=force_engine,
    )

    # Windows
    results_win = []
    for w in capture["windows"]:
        fname = _safe_filename(f"{w['app_name']}_{w['window_name']}")
        r = run_bruteforce_ocr(
            w["image"],
            os.path.join(method_dir, "windows", f"{fname}.json"),
            lang=lang,
            force_engine=force_engine,
        )
        results_win.append(r)

    total_time = result_full["elapsed_ms"] + sum(r["elapsed_ms"] for r in results_win)
    total_chars = len(result_full["text"]) + sum(len(r["text"]) for r in results_win)

    return {
        "method": "bruteforce",
        "total_time_ms": round(total_time, 2),
        "total_chars": total_chars,
        "num_regions": 0,
        "full_screen_chars": len(result_full["text"]),
        "window_count": len(results_win),
    }


def run_method2(capture: dict, output_dir: str, lang: str, model_name: str, conf: float, force_engine: str = None):
    """Method 2: YOLO-World region detection + OCR."""
    from ocr_methods.method_yoloworld import run_yoloworld_ocr

    method_dir = os.path.join(output_dir, "method2_yoloworld")

    # Full screen
    result_full = run_yoloworld_ocr(
        capture["full_screen"],
        os.path.join(method_dir, "full_screen.json"),
        os.path.join(method_dir, "full_screen_regions.png"),
        lang=lang,
        model_name=model_name,
        conf_threshold=conf,
        force_engine=force_engine,
    )

    # Windows
    results_win = []
    total_regions = result_full["num_regions"]
    for w in capture["windows"]:
        fname = _safe_filename(f"{w['app_name']}_{w['window_name']}")
        r = run_yoloworld_ocr(
            w["image"],
            os.path.join(method_dir, "windows", f"{fname}.json"),
            os.path.join(method_dir, "windows", f"{fname}_regions.png"),
            lang=lang,
            model_name=model_name,
            conf_threshold=conf,
            force_engine=force_engine,
        )
        results_win.append(r)
        total_regions += r["num_regions"]

    total_time = result_full["elapsed_total_ms"] + sum(r["elapsed_total_ms"] for r in results_win)
    total_chars = sum(len(reg["ocr_text"]) for reg in result_full["regions"])
    total_chars += sum(len(reg["ocr_text"]) for r in results_win for reg in r["regions"])

    return {
        "method": "yoloworld",
        "model": model_name,
        "conf_threshold": conf,
        "total_time_ms": round(total_time, 2),
        "total_chars": total_chars,
        "num_regions": total_regions,
        "full_screen_regions": result_full["num_regions"],
        "window_count": len(results_win),
    }


def run_method3(capture: dict, output_dir: str, lang: str, min_grad: int, min_area: int, force_engine: str = None):
    """Method 3: UIED-inspired CV detection + OCR."""
    from ocr_methods.method_uied import run_uied_ocr

    method_dir = os.path.join(output_dir, "method3_uied")

    # Full screen
    result_full = run_uied_ocr(
        capture["full_screen"],
        os.path.join(method_dir, "full_screen.json"),
        os.path.join(method_dir, "full_screen_regions.png"),
        lang=lang,
        min_grad=min_grad,
        min_area=min_area,
        force_engine=force_engine,
    )

    # Windows
    results_win = []
    total_regions = result_full["num_regions"]
    for w in capture["windows"]:
        fname = _safe_filename(f"{w['app_name']}_{w['window_name']}")
        r = run_uied_ocr(
            w["image"],
            os.path.join(method_dir, "windows", f"{fname}.json"),
            os.path.join(method_dir, "windows", f"{fname}_regions.png"),
            lang=lang,
            min_grad=min_grad,
            min_area=min_area,
            force_engine=force_engine,
        )
        results_win.append(r)
        total_regions += r["num_regions"]

    total_time = result_full["elapsed_total_ms"] + sum(r["elapsed_total_ms"] for r in results_win)
    total_chars = sum(len(reg["ocr_text"]) for reg in result_full["regions"])
    total_chars += sum(len(reg["ocr_text"]) for r in results_win for reg in r["regions"])

    return {
        "method": "uied_cv",
        "total_time_ms": round(total_time, 2),
        "total_chars": total_chars,
        "num_regions": total_regions,
        "full_screen_regions": result_full["num_regions"],
        "window_count": len(results_win),
    }


# ---------------------------------------------------------------------------
# Training guide generation
# ---------------------------------------------------------------------------

TRAINING_GUIDE = """\
# YOLO UI Element Detection Model Training Guide

## Goal
Train a YOLO model specifically for detecting UI components in desktop screenshots
(text blocks, sidebars, toolbars, code editors, terminals, dialogs, etc.).

## 1. Dataset Options

### Option A: Public Datasets
- **Rico** (Android UI): ~66k screenshots with UI component annotations.
  Download: https://interactionmining.org/rico
  Note: Mobile UI, needs adaptation for desktop.
- **Enrico**: Curated subset of Rico with design topic labels.
- **WebUI**: Web page screenshots with layout annotations.
  Paper: "WebUI: A Dataset for Enhancing Visual UI Understanding"
- **Screen2Words / Widget Captioning**: Datasets with UI element bounding boxes.

### Option B: Build Your Own Desktop UI Dataset (Recommended for best results)
1. Use VisualMem's capture pipeline to collect diverse desktop screenshots.
2. Include varied applications: browsers, IDEs (VS Code, Cursor), terminals,
   document editors, chat apps, email clients, file managers.
3. Aim for 500-2000 annotated images for a good starting point.

## 2. Annotation

### Tool: Label Studio (recommended, free & open source)
```bash
pip install label-studio
label-studio start
```
Or use Roboflow (web-based, has free tier with auto-labeling).

### Annotation Classes (recommended set)
```
text_block, sidebar, toolbar, menu_bar, tab_bar, code_editor,
terminal, dialog, button, input_field, table, image_area,
navigation_bar, status_bar, card, panel, list_view, search_bar,
header, footer, scroll_bar, dropdown, modal
```

### Annotation Tips
- Label the functional region, not individual characters.
- For nested elements (e.g., sidebar containing a list), label the outermost
  logical container. Add inner elements only if they represent distinct UI types.
- Maintain consistent bounding box tightness across annotators.

## 3. Training

### Base Model
Use `yolov8l-worldv2.pt` as the base for transfer learning.
YOLO-World v2 already understands visual-text alignment, so fine-tuning on
UI data will adapt it quickly.

### Data Format
Convert annotations to YOLO format:
```
# Each image has a corresponding .txt file:
# class_id center_x center_y width height (all normalized 0-1)
0 0.45 0.12 0.90 0.05
1 0.10 0.55 0.18 0.85
```

### Directory Structure
```
dataset/
  train/
    images/
    labels/
  val/
    images/
    labels/
  data.yaml
```

### data.yaml
```yaml
path: ./dataset
train: train/images
val: val/images

names:
  0: text_block
  1: sidebar
  2: toolbar
  3: menu_bar
  4: tab_bar
  5: code_editor
  6: terminal
  7: dialog
  8: button
  9: input_field
  10: table
  11: image_area
  12: navigation_bar
  13: status_bar
  14: card
  15: panel
```

### Training Command
```python
from ultralytics import YOLO

# Load pretrained YOLO-World v2 as base
model = YOLO("yolov8l-worldv2.pt")

# Fine-tune on your UI dataset
results = model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=1280,        # Desktop screenshots are high-res
    batch=4,            # Adjust based on GPU memory
    patience=20,        # Early stopping
    lr0=0.001,          # Lower LR for fine-tuning
    lrf=0.01,
    warmup_epochs=5,
    device=0,           # GPU index
    project="yolo_ui",
    name="desktop_ui_v1",
)

# Export for deployment
model.export(format="onnx")
```

### CLI Alternative
```bash
yolo train model=yolov8l-worldv2.pt data=dataset/data.yaml \\
    epochs=100 imgsz=1280 batch=4 patience=20 lr0=0.001 \\
    project=yolo_ui name=desktop_ui_v1
```

## 4. Evaluation & Iteration

```python
# Validate
metrics = model.val(data="dataset/data.yaml")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")

# Inference on new screenshot
results = model.predict("test_screenshot.png", conf=0.25)
results[0].show()
```

### Key Metrics to Track
- mAP50 > 0.7 is a good target for UI detection
- Per-class AP to identify weak categories
- Visual inspection of predictions on diverse screenshots

## 5. Integration into VisualMem

After training, replace `yolov8l-worldv2.pt` with your fine-tuned model:
```python
# In scripts/ocr_methods/method_yoloworld.py, change model loading:
model = YOLO("yolo_ui/desktop_ui_v1/weights/best.pt")
```

Or add a config option in `.env`:
```
UI_DETECTION_MODEL=yolo_ui/desktop_ui_v1/weights/best.pt
```
"""


def save_training_guide(output_dir: str):
    """Save the YOLO UI model training guide."""
    path = os.path.join(output_dir, "training_guide.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(TRAINING_GUIDE)
    logger.info(f"Training guide saved to {path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list):
    """Print comparison summary table."""
    print("\n" + "=" * 78)
    print(" OCR Comparison Summary")
    print("=" * 78)
    print(f"{'Method':<20} {'Time (ms)':>12} {'Regions':>10} {'Chars':>10} {'Windows':>10}")
    print("-" * 78)

    for r in results:
        method = r["method"]
        total_time = r["total_time_ms"]
        regions = r.get("num_regions", 0)
        chars = r["total_chars"]
        wins = r["window_count"]
        print(f"{method:<20} {total_time:>12.1f} {regions:>10} {chars:>10} {wins:>10}")

    print("=" * 78)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OCR Comparison Demo: compare three OCR strategies on screen captures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/demo_ocr_compare.py --method all
  python scripts/demo_ocr_compare.py --method 2 --conf 0.15
  python scripts/demo_ocr_compare.py --method 1 3
        """,
    )
    parser.add_argument(
        "--method", nargs="+", default=["all"],
        help="Methods to run: 1, 2, 3, or 'all' (default: all)",
    )
    parser.add_argument("--output-dir", default="ocr_compare_output", help="Output directory")
    parser.add_argument("--lang", default="chi_sim+eng", help="OCR language (default: chi_sim+eng)")
    parser.add_argument("--yolo-model", default="yolov8l-worldv2.pt", help="YOLO-World model name")
    parser.add_argument("--conf", type=float, default=0.01, help="YOLO detection confidence threshold")
    parser.add_argument("--min-grad", type=int, default=10, help="UIED: minimum gradient threshold")
    parser.add_argument("--min-area", type=int, default=500, help="UIED: minimum region area")
    parser.add_argument(
        "--ocr-engine", default=None,
        choices=["auto", "apple_vision", "windows_ocr", "pytesseract"],
        help="OCR engine: auto (default, platform-native), apple_vision, windows_ocr, pytesseract",
    )

    args = parser.parse_args()

    # Normalize ocr engine
    force_engine = args.ocr_engine if args.ocr_engine != "auto" else None

    methods = set()
    for m in args.method:
        if m == "all":
            methods = {1, 2, 3}
            break
        methods.add(int(m))

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Capture screen
    print("Capturing screen...")
    capture = capture_screen()

    # Step 2: Save raw screenshots
    save_raw_screenshots(capture, output_dir)

    # Step 3: Run selected methods
    results = []

    if 1 in methods:
        print("\n--- Method 1: Brute-force OCR ---")
        r = run_method1(capture, output_dir, args.lang, force_engine)
        results.append(r)
        print(f"  Done: {r['total_time_ms']:.0f}ms, {r['total_chars']} chars")

    if 2 in methods:
        print("\n--- Method 2: YOLO-World + OCR ---")
        r = run_method2(capture, output_dir, args.lang, args.yolo_model, args.conf, force_engine)
        results.append(r)
        print(f"  Done: {r['total_time_ms']:.0f}ms, {r['num_regions']} regions, {r['total_chars']} chars")

    if 3 in methods:
        print("\n--- Method 3: UIED-CV + OCR ---")
        r = run_method3(capture, output_dir, args.lang, args.min_grad, args.min_area, force_engine)
        results.append(r)
        print(f"  Done: {r['total_time_ms']:.0f}ms, {r['num_regions']} regions, {r['total_chars']} chars")

    # Step 4: Summary
    if results:
        print_summary(results)

        # Save summary JSON
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Summary saved to {summary_path}")

    # Step 5: Training guide
    save_training_guide(output_dir)

    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
