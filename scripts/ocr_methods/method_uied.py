"""
Method 3: Traditional CV region detection (UIED-inspired) + per-region OCR

Implements the core ideas from UIED (UI Element Detection) using OpenCV:
- Gradient-based binarization to find edges
- Connected component analysis to detect UI regions
- Contour hierarchy analysis for nested components
- Region filtering and merging

Then runs pytesseract on each detected region independently.

Reference: https://github.com/MulongXie/UIED
"""
import json
import time
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.ocr_methods.ocr_adapter import create_platform_ocr


_ocr_engine = None


def _get_ocr(lang: str, force_engine: str = None):
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = create_platform_ocr(lang=lang, force_engine=force_engine)
    return _ocr_engine


def _gradient_binarization(gray: np.ndarray, min_grad: int = 10) -> np.ndarray:
    """
    UIED-style gradient-based binarization.
    Compute Sobel gradients and threshold to get edge map.
    """
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(grad_x) + cv2.convertScaleAbs(grad_y)
    _, binary = cv2.threshold(grad, min_grad, 255, cv2.THRESH_BINARY)
    return binary


def _flood_fill_regions(binary: np.ndarray, block_size: int = 5) -> np.ndarray:
    """
    UIED-style flood fill block division.
    Dilate edges to connect nearby components, then find connected regions.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (block_size, block_size))
    # Close gaps between nearby edges
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Dilate to merge nearby regions
    dilated = cv2.dilate(closed, kernel, iterations=1)
    return dilated


def _detect_regions(
    image_cv: np.ndarray,
    min_grad: int = 10,
    block_size: int = 5,
    min_area: int = 500,
    min_width: int = 20,
    min_height: int = 20,
    merge_overlap_threshold: float = 0.5,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect UI regions using UIED-inspired CV pipeline.

    Returns:
        List of (x1, y1, x2, y2) bounding boxes
    """
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Step 1: Gradient binarization
    binary = _gradient_binarization(gray, min_grad)

    # Step 2: Flood fill / morphological closing to form regions
    region_map = _flood_fill_regions(binary, block_size)

    # Step 3: Invert to find enclosed regions (white areas become regions)
    inverted = cv2.bitwise_not(region_map)

    # Step 4: Connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    bboxes = []
    for i in range(1, num_labels):  # skip background (label 0)
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Filter: minimum area, width, height
        if area < min_area or bw < min_width or bh < min_height:
            continue
        # Filter: skip regions that span the entire image (background)
        if bw >= w * 0.95 and bh >= h * 0.95:
            continue

        bboxes.append((int(x), int(y), int(x + bw), int(y + bh)))

    # Step 5: Also detect via contour hierarchy for nested components
    contours, hierarchy = cv2.findContours(region_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours and hierarchy is not None:
        for i, cnt in enumerate(contours):
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < min_area or bw < min_width or bh < min_height:
                continue
            if bw >= w * 0.95 and bh >= h * 0.95:
                continue
            bboxes.append((int(x), int(y), int(x + bw), int(y + bh)))

    # Step 6: Merge overlapping bounding boxes
    bboxes = _merge_overlapping(bboxes, merge_overlap_threshold)

    return bboxes


def _iou(box1: Tuple, box2: Tuple) -> float:
    """Compute intersection over union of two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def _containment(inner: Tuple, outer: Tuple) -> float:
    """Compute how much of inner is contained within outer."""
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_inner = (inner[2] - inner[0]) * (inner[3] - inner[1])

    return inter / area_inner if area_inner > 0 else 0


def _merge_overlapping(bboxes: List[Tuple], threshold: float = 0.5) -> List[Tuple]:
    """Merge highly overlapping or contained bounding boxes."""
    if not bboxes:
        return []

    # Sort by area descending
    bboxes = sorted(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    merged = []
    used = [False] * len(bboxes)

    for i in range(len(bboxes)):
        if used[i]:
            continue
        current = list(bboxes[i])
        for j in range(i + 1, len(bboxes)):
            if used[j]:
                continue
            # If j is mostly contained in current, absorb it
            if _containment(bboxes[j], tuple(current)) > threshold:
                used[j] = True
                continue
            # If high IoU, merge
            if _iou(tuple(current), bboxes[j]) > threshold:
                current[0] = min(current[0], bboxes[j][0])
                current[1] = min(current[1], bboxes[j][1])
                current[2] = max(current[2], bboxes[j][2])
                current[3] = max(current[3], bboxes[j][3])
                used[j] = True

        merged.append(tuple(current))

    return merged


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
    ]

    for i, region in enumerate(regions):
        bbox = region["bbox"]
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = bbox

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        label = f"region_{i}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([text_bbox[0] - 1, text_bbox[1] - 1, text_bbox[2] + 1, text_bbox[3] + 1], fill=color)
        draw.text((x1, y1), label, fill="black", font=font)

    return vis


def run_uied_ocr(
    image: Image.Image,
    output_json_path: str,
    output_vis_path: str,
    lang: str = "chi_sim+eng",
    min_grad: int = 10,
    block_size: int = 5,
    min_area: int = 500,
    force_engine: str = None,
) -> dict:
    """
    Detect UI regions with traditional CV (UIED-inspired), then OCR each region.

    Args:
        image: PIL Image to process
        output_json_path: Path to save JSON result
        output_vis_path: Path to save visualization image
        lang: Tesseract language string
        min_grad: Minimum gradient for edge detection (higher = finer segmentation)
        block_size: Morphological kernel size for region merging
        min_area: Minimum region area in pixels

    Returns:
        dict with regions, timing info
    """
    ocr = _get_ocr(lang, force_engine)

    # Convert to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Detect regions
    start_det = time.perf_counter()
    bboxes = _detect_regions(
        img_cv,
        min_grad=min_grad,
        block_size=block_size,
        min_area=min_area,
    )
    elapsed_det = time.perf_counter() - start_det

    # OCR each region
    regions = []
    start_ocr = time.perf_counter()

    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        crop = image.crop((x1, y1, x2, y2))
        ocr_result = ocr.recognize(crop)

        regions.append({
            "bbox": [x1, y1, x2, y2],
            "region_id": i,
            "ocr_text": ocr_result.text,
            "ocr_confidence": round(ocr_result.confidence, 4),
        })

    elapsed_ocr = time.perf_counter() - start_ocr

    output = {
        "method": "uied_cv",
        "params": {
            "min_grad": min_grad,
            "block_size": block_size,
            "min_area": min_area,
        },
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
