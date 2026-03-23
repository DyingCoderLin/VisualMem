"""
Method 2: YOLO-World v2 open-vocabulary UI region detection + per-region OCR

Uses yolov8l-worldv2.pt with custom UI class prompts to detect logical regions
in screenshots (sidebar, toolbar, text block, etc.), then runs pytesseract on
each detected region independently.
"""
import json
import time
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.ocr_methods.ocr_adapter import create_platform_ocr

# UI classes for open-vocabulary detection
UI_CLASSES = [
    "text block", "sidebar", "toolbar", "menu bar", "navigation bar",
    "tab bar", "code editor", "terminal", "dialog", "card",
    "header", "footer", "input field", "table", "content area",
    "panel", "list", "search bar", "status bar", "tooltip",
]

# Lazy-loaded globals
_model = None
_ocr_engine = None


def _get_model(model_name: str = "yolov8l-worldv2.pt"):
    global _model
    if _model is None:
        from ultralytics import YOLOWorld
        _model = YOLOWorld(model_name)
        _model.set_classes(UI_CLASSES)
    return _model


def _get_ocr(lang: str, force_engine: str = None):
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = create_platform_ocr(lang=lang, force_engine=force_engine)
    return _ocr_engine


def _draw_regions(image: Image.Image, regions: list) -> Image.Image:
    """Draw detected regions on the image for visualization."""
    vis = image.copy()
    draw = ImageDraw.Draw(vis)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
        "#F1948A", "#82E0AA", "#F8C471", "#AED6F1", "#D2B4DE",
        "#A3E4D7", "#FAD7A0", "#D5DBDB", "#ABEBC6", "#F9E79F",
    ]

    for i, region in enumerate(regions):
        bbox = region["bbox"]
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = bbox

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        label = f"{region['class_name']} ({region['det_confidence']:.2f})"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([text_bbox[0] - 1, text_bbox[1] - 1, text_bbox[2] + 1, text_bbox[3] + 1], fill=color)
        draw.text((x1, y1), label, fill="black", font=font)

    return vis


def run_yoloworld_ocr(
    image: Image.Image,
    output_json_path: str,
    output_vis_path: str,
    lang: str = "chi_sim+eng",
    model_name: str = "yolov8l-worldv2.pt",
    conf_threshold: float = 0.01,
    force_engine: str = None,
) -> dict:
    """
    Detect UI regions with YOLO-World, then OCR each region.

    Args:
        image: PIL Image to process
        output_json_path: Path to save JSON result
        output_vis_path: Path to save visualization image
        lang: Tesseract language string
        model_name: YOLO-World model file
        conf_threshold: Detection confidence threshold

    Returns:
        dict with regions, timing info
    """
    model = _get_model(model_name)
    ocr = _get_ocr(lang, force_engine)

    # Detection
    start_det = time.perf_counter()
    img_array = np.array(image)
    results = model.predict(img_array, conf=conf_threshold, verbose=False)
    elapsed_det = time.perf_counter() - start_det

    # Parse detections and OCR each region
    regions = []
    start_ocr = time.perf_counter()

    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            cls_name = UI_CLASSES[cls_id] if cls_id < len(UI_CLASSES) else f"class_{cls_id}"

            # Crop and OCR the region
            crop = image.crop((x1, y1, x2, y2))
            ocr_result = ocr.recognize(crop)

            regions.append({
                "bbox": [x1, y1, x2, y2],
                "class_name": cls_name,
                "det_confidence": round(conf, 4),
                "ocr_text": ocr_result.text,
                "ocr_confidence": round(ocr_result.confidence, 4),
            })

    elapsed_ocr = time.perf_counter() - start_ocr

    output = {
        "method": "yoloworld",
        "model": model_name,
        "conf_threshold": conf_threshold,
        "ui_classes": UI_CLASSES,
        "elapsed_detection_ms": round(elapsed_det * 1000, 2),
        "elapsed_ocr_ms": round(elapsed_ocr * 1000, 2),
        "elapsed_total_ms": round((elapsed_det + elapsed_ocr) * 1000, 2),
        "num_regions": len(regions),
        "regions": regions,
    }

    # Save JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Save visualization
    vis = _draw_regions(image, regions)
    os.makedirs(os.path.dirname(output_vis_path), exist_ok=True)
    vis.save(output_vis_path)

    return output
