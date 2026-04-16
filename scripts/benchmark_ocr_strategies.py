#!/usr/bin/env python3
"""
Benchmark OCR strategies: per-region vs whole-image-assign vs no-region.

Takes one screenshot, runs all strategies on the same image,
outputs timing breakdown and text results to /tmp/ for comparison.

Usage:
    python scripts/benchmark_ocr_strategies.py
    python scripts/benchmark_ocr_strategies.py --image /path/to/screenshot.png
"""
import argparse
import io
import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Apple Vision OCR with bounding boxes
# ---------------------------------------------------------------------------

def apple_vision_ocr_with_bboxes(img: Image.Image) -> list:
    """Single Apple Vision OCR call returning text + pixel bounding boxes."""
    import Vision
    from Foundation import NSData

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    ns_data = NSData.dataWithBytes_length_(buf.getvalue(), buf.tell())

    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setRecognitionLanguages_(["zh-Hans", "zh-Hant", "en"])
    request.setUsesLanguageCorrection_(True)

    success = handler.performRequests_error_([request], None)
    if not success[0]:
        return []

    img_w, img_h = img.size
    results = []
    for obs in (request.results() or []):
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


def assign_text_to_regions(text_bboxes, region_bboxes):
    """Assign OCR text lines to UIED regions by center-point containment."""
    region_texts = {i: [] for i in range(len(region_bboxes))}
    unassigned = []

    for tb in text_bboxes:
        tx1, ty1, tx2, ty2 = tb["bbox"]
        t_cx, t_cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2

        best_region = -1
        best_overlap = 0
        for i, (rx1, ry1, rx2, ry2) in enumerate(region_bboxes):
            if rx1 <= t_cx <= rx2 and ry1 <= t_cy <= ry2:
                overlap = max(0, min(tx2, rx2) - max(tx1, rx1)) * max(0, min(ty2, ry2) - max(ty1, ry1))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_region = i

        if best_region >= 0:
            region_texts[best_region].append(tb)
        else:
            unassigned.append(tb)

    return region_texts, unassigned


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def strategy_a_per_region(image, detector, ocr):
    """Current: UIED detect → per-region crop OCR → remainder OCR."""
    timings = {}

    t0 = time.perf_counter()
    regions_meta = detector.detect(image)
    timings["uied_detect"] = (time.perf_counter() - t0) * 1000

    W, H = image.size
    results = []

    # Per-region OCR
    t0 = time.perf_counter()
    for meta in regions_meta:
        bbox = meta["bbox"]
        crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        r = ocr.recognize(crop)
        results.append({"bbox": bbox, "text": r.text, "conf": r.confidence})
    timings["per_region_ocr"] = (time.perf_counter() - t0) * 1000
    timings["per_region_count"] = len(regions_meta)

    # Remainder OCR
    t0 = time.perf_counter()
    remainder_img = image.copy()
    draw = ImageDraw.Draw(remainder_img)
    for meta in regions_meta:
        x1, y1, x2, y2 = meta["bbox"]
        draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
    remainder = ocr.recognize(remainder_img)
    timings["remainder_ocr"] = (time.perf_counter() - t0) * 1000

    if remainder.text.strip():
        results.append({"bbox": [0, 0, W, H], "text": remainder.text, "conf": remainder.confidence, "is_remainder": True})

    timings["total"] = timings["uied_detect"] + timings["per_region_ocr"] + timings["remainder_ocr"]
    return results, timings


def strategy_b_whole_assign(image, detector):
    """Optimized: single whole-image OCR → assign text to UIED regions."""
    timings = {}

    t0 = time.perf_counter()
    regions_meta = detector.detect(image)
    timings["uied_detect"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    text_bboxes = apple_vision_ocr_with_bboxes(image)
    timings["whole_image_ocr"] = (time.perf_counter() - t0) * 1000
    timings["ocr_lines"] = len(text_bboxes)

    t0 = time.perf_counter()
    region_bboxes = [m["bbox"] for m in regions_meta]
    region_texts, unassigned = assign_text_to_regions(text_bboxes, region_bboxes)
    timings["assign"] = (time.perf_counter() - t0) * 1000

    W, H = image.size
    results = []
    for i, meta in enumerate(regions_meta):
        texts = region_texts.get(i, [])
        combined = "\n".join(t["text"] for t in texts)
        avg_conf = sum(t["confidence"] for t in texts) / len(texts) if texts else 0
        results.append({"bbox": meta["bbox"], "text": combined, "conf": avg_conf})
    for t in unassigned:
        results.append({"bbox": t["bbox"], "text": t["text"], "conf": t["confidence"], "is_unassigned": True})

    timings["total"] = timings["uied_detect"] + timings["whole_image_ocr"] + timings["assign"]
    timings["unassigned"] = len(unassigned)
    return results, timings


def strategy_c_whole_only(image):
    """Baseline: single whole-image OCR, no regions."""
    timings = {}

    t0 = time.perf_counter()
    text_bboxes = apple_vision_ocr_with_bboxes(image)
    timings["whole_image_ocr"] = (time.perf_counter() - t0) * 1000
    timings["ocr_lines"] = len(text_bboxes)
    timings["total"] = timings["whole_image_ocr"]

    results = [{"bbox": tb["bbox"], "text": tb["text"], "conf": tb["confidence"]} for tb in text_bboxes]
    return results, timings


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results(path, strategy_name, results, timings, image_size):
    lines = [f"=== {strategy_name} ==="]
    lines.append(f"Image: {image_size[0]}x{image_size[1]}")
    lines.append(f"Timings: {json.dumps(timings, indent=2)}")
    lines.append("")

    for i, r in enumerate(results):
        bbox = r["bbox"]
        tag = ""
        if r.get("is_remainder"):
            tag = " [REMAINDER]"
        elif r.get("is_unassigned"):
            tag = " [UNASSIGNED]"
        lines.append(f"--- Region {i} [{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]{tag} ---")
        lines.append(r["text"] if r["text"] else "(empty)")
        lines.append("")

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark OCR strategies")
    parser.add_argument("--image", default=None, help="Path to screenshot image (default: take screenshot)")
    args = parser.parse_args()

    if args.image:
        image = Image.open(args.image).convert("RGB")
        print(f"Loaded image: {image.size}")
    else:
        from PIL import ImageGrab
        image = ImageGrab.grab().convert("RGB")
        image.save("/tmp/ocr_benchmark_screenshot.png")
        print(f"Screenshot taken: {image.size}")
        print(f"Saved to /tmp/ocr_benchmark_screenshot.png")

    W, H = image.size

    # Init components
    from core.ocr.region_detector import UIEDRegionDetector
    from core.ocr.platform_ocr import AppleVisionOCR

    detector = UIEDRegionDetector()
    ocr = AppleVisionOCR()

    # Warmup Apple Vision
    ocr.recognize(image.crop((0, 0, 200, 200)))

    # Run strategies
    print("\nRunning Strategy A (per-region OCR)...")
    results_a, timings_a = strategy_a_per_region(image, detector, ocr)

    print("Running Strategy B (whole-image OCR + assign)...")
    results_b, timings_b = strategy_b_whole_assign(image, detector)

    print("Running Strategy C (whole-image OCR only)...")
    results_c, timings_c = strategy_c_whole_only(image)

    # Write results
    text_a = write_results("/tmp/ocr_strategy_A_per_region.txt", "Strategy A: Per-Region OCR", results_a, timings_a, (W, H))
    text_b = write_results("/tmp/ocr_strategy_B_whole_assign.txt", "Strategy B: Whole-Image OCR + Assign", results_b, timings_b, (W, H))
    text_c = write_results("/tmp/ocr_strategy_C_whole_only.txt", "Strategy C: Whole-Image OCR Only", results_c, timings_c, (W, H))

    # Print summary
    print(f"\n{'=' * 70}")
    print("TIMING COMPARISON")
    print(f"{'=' * 70}")

    print(f"\n--- Strategy A: Per-Region OCR (current) ---")
    print(f"  UIED detect:        {timings_a['uied_detect']:>7.0f}ms")
    print(f"  Per-region OCR ×{timings_a['per_region_count']:>2}: {timings_a['per_region_ocr']:>7.0f}ms"
          f"  ({timings_a['per_region_ocr']/max(timings_a['per_region_count'],1):.0f}ms each)")
    print(f"  Remainder OCR:      {timings_a['remainder_ocr']:>7.0f}ms  ← 这是大头！对遮蔽全图再做一次 OCR")
    print(f"  Total:              {timings_a['total']:>7.0f}ms")

    print(f"\n--- Strategy B: Whole-Image OCR + Assign ---")
    print(f"  UIED detect:        {timings_b['uied_detect']:>7.0f}ms")
    print(f"  Whole-image OCR:    {timings_b['whole_image_ocr']:>7.0f}ms  ({timings_b['ocr_lines']} lines)")
    print(f"  Assign to regions:  {timings_b['assign']:>7.0f}ms")
    print(f"  Total:              {timings_b['total']:>7.0f}ms")
    print(f"  Unassigned lines:   {timings_b['unassigned']}")

    print(f"\n--- Strategy C: Whole-Image OCR Only ---")
    print(f"  Whole-image OCR:    {timings_c['whole_image_ocr']:>7.0f}ms  ({timings_c['ocr_lines']} lines)")
    print(f"  Total:              {timings_c['total']:>7.0f}ms")

    speedup_b = timings_a["total"] / timings_b["total"] if timings_b["total"] > 0 else 0
    speedup_c = timings_a["total"] / timings_c["total"] if timings_c["total"] > 0 else 0

    print(f"\n--- Speedup ---")
    print(f"  A → B: {speedup_b:.1f}x")
    print(f"  A → C: {speedup_c:.1f}x")

    # Why the speedup is limited
    print(f"\n--- Why A is slow ---")
    pct_remainder = timings_a["remainder_ocr"] / timings_a["total"] * 100
    pct_per_region = timings_a["per_region_ocr"] / timings_a["total"] * 100
    print(f"  Remainder OCR 占 A 总时间的 {pct_remainder:.0f}%")
    print(f"  Per-region OCR 占 {pct_per_region:.0f}%")
    print(f"  A 的 remainder = 对遮蔽全图做一次完整 OCR ≈ B 的 whole-image OCR")
    print(f"  所以 A ≈ B + per-region OCR 额外开销")
    print(f"  如果去掉 remainder，A 变成 {timings_a['total'] - timings_a['remainder_ocr']:.0f}ms（但会丢失 UIED 漏检区域的文字）")

    # Text quality comparison
    chars_a = sum(len(r["text"]) for r in results_a)
    chars_b = sum(len(r["text"]) for r in results_b)
    chars_c = sum(len(r["text"]) for r in results_c)
    print(f"\n--- Text Output ---")
    print(f"  A: {chars_a} chars  ({len(results_a)} regions)")
    print(f"  B: {chars_b} chars  ({len(results_b)} regions)")
    print(f"  C: {chars_c} chars  ({len(results_c)} lines)")

    print(f"\nResult files:")
    print(f"  /tmp/ocr_strategy_A_per_region.txt")
    print(f"  /tmp/ocr_strategy_B_whole_assign.txt")
    print(f"  /tmp/ocr_strategy_C_whole_only.txt")
    print(f"  /tmp/ocr_benchmark_screenshot.png")


if __name__ == "__main__":
    main()
