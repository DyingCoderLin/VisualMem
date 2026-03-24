# core/ocr/region_detector.py
"""
UIED-inspired region detector using traditional CV.

Detects UI regions via gradient binarization, morphological operations,
connected component analysis, and contour hierarchy.
"""
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from utils.logger import setup_logger

logger = setup_logger(__name__)


def _gradient_binarization(gray: np.ndarray, min_grad: int = 10) -> np.ndarray:
    """Gradient-based binarization: Sobel edges thresholded."""
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(grad_x) + cv2.convertScaleAbs(grad_y)
    _, binary = cv2.threshold(grad, min_grad, 255, cv2.THRESH_BINARY)
    return binary


def _flood_fill_regions(binary: np.ndarray, block_size: int = 5) -> np.ndarray:
    """Morphological close + dilate to form connected regions."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (block_size, block_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    return dilated


def _iou(box1: Tuple, box2: Tuple) -> float:
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
            if _containment(bboxes[j], tuple(current)) > threshold:
                used[j] = True
                continue
            if _iou(tuple(current), bboxes[j]) > threshold:
                current[0] = min(current[0], bboxes[j][0])
                current[1] = min(current[1], bboxes[j][1])
                current[2] = max(current[2], bboxes[j][2])
                current[3] = max(current[3], bboxes[j][3])
                used[j] = True
        merged.append(tuple(current))
    return merged


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

    binary = _gradient_binarization(gray, min_grad)
    region_map = _flood_fill_regions(binary, block_size)
    inverted = cv2.bitwise_not(region_map)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    bboxes = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or bw < min_width or bh < min_height:
            continue
        if bw >= w * 0.95 and bh >= h * 0.95:
            continue
        bboxes.append((int(x), int(y), int(x + bw), int(y + bh)))

    contours, hierarchy = cv2.findContours(region_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours and hierarchy is not None:
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < min_area or bw < min_width or bh < min_height:
                continue
            if bw >= w * 0.95 and bh >= h * 0.95:
                continue
            bboxes.append((int(x), int(y), int(x + bw), int(y + bh)))

    bboxes = _merge_overlapping(bboxes, merge_overlap_threshold)
    return bboxes


class UIEDRegionDetector:
    """UIED-inspired region detector for UI screenshots."""

    def __init__(
        self,
        min_grad: int = 10,
        block_size: int = 5,
        min_area: int = 500,
        min_width: int = 20,
        min_height: int = 20,
        merge_overlap_threshold: float = 0.5,
    ):
        self.min_grad = min_grad
        self.block_size = block_size
        self.min_area = min_area
        self.min_width = min_width
        self.min_height = min_height
        self.merge_overlap_threshold = merge_overlap_threshold

    def detect(self, image: Image.Image) -> List[dict]:
        """
        Detect UI regions in image.

        Returns:
            List of {"bbox": [x1, y1, x2, y2], "region_index": i}
        """
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        bboxes = _detect_regions(
            img_cv,
            min_grad=self.min_grad,
            block_size=self.block_size,
            min_area=self.min_area,
            min_width=self.min_width,
            min_height=self.min_height,
            merge_overlap_threshold=self.merge_overlap_threshold,
        )
        return [
            {"bbox": list(bbox), "region_index": i}
            for i, bbox in enumerate(bboxes)
        ]
