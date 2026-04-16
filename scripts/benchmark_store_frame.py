#!/usr/bin/env python3
"""
Analyze store_frame bottleneck from actual log data + live benchmarks.

Usage:
    python scripts/benchmark_store_frame.py
"""
import os
import re
import sys
import time
import statistics
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


def parse_log_timings(log_path="logs/backend_server.log"):
    """Extract store_frame timings from backend log."""
    pattern = re.compile(
        r"store_frame: (\S+) done in ([\d.]+)s, (\d+) windows"
    )
    entries = []
    try:
        with open(log_path) as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    entries.append({
                        "frame_id": m.group(1),
                        "elapsed_s": float(m.group(2)),
                        "windows": int(m.group(3)),
                    })
    except FileNotFoundError:
        pass
    return entries


def benchmark_components():
    """Benchmark individual pipeline components."""
    from PIL import ImageGrab

    print("Taking screenshot...")
    image = ImageGrab.grab().convert("RGB")
    print(f"  Image size: {image.size}\n")

    results = {}

    # --- Embedding ---
    print("--- Embedding ---")
    from core.encoder.clip_encoder import CLIPEncoder
    encoder = CLIPEncoder()
    # Warmup
    encoder.encode_image(image)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        emb = encoder.encode_image(image)
        times.append((time.perf_counter() - t0) * 1000)
    avg = statistics.mean(times)
    print(f"  encode_image: avg={avg:.0f}ms  range=[{min(times):.0f}, {max(times):.0f}]ms")
    results["embedding_ms"] = avg

    # --- Region OCR ---
    print("\n--- Region OCR ---")
    try:
        from core.ocr.platform_ocr import AppleVisionOCR
        from core.ocr.region_ocr_engine import RegionOCREngine
        ocr_eng = AppleVisionOCR()
        region_ocr = RegionOCREngine(ocr_engine=ocr_eng)

        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            regions = region_ocr.recognize_regions(image)
            times.append((time.perf_counter() - t0) * 1000)
        avg = statistics.mean(times)
        n_regions = len(regions) if regions else 0
        print(f"  region_ocr: avg={avg:.0f}ms  range=[{min(times):.0f}, {max(times):.0f}]ms  ({n_regions} regions)")
        results["ocr_ms"] = avg
    except Exception as e:
        print(f"  OCR skipped: {e}")
        results["ocr_ms"] = 0

    # --- Cluster assignment (committed hit) ---
    print("\n--- Cluster assignment ---")
    try:
        from core.activity.cluster_manager import ClusterManager
        import numpy as np
        cm = ClusterManager()

        times_committed = []
        times_pending = []
        for _ in range(5):
            t0 = time.perf_counter()
            label = cm.assign_frame(
                app_name="BenchmarkApp_" + str(time.time_ns()),
                frame_id=f"subframe_bench_{time.time_ns()}",
                embedding=np.array(emb, dtype=np.float32),
                image=image,
                ocr_text="benchmark test text",
                timestamp=datetime.now(timezone.utc).isoformat(),
                window_name="BenchmarkWindow",
            )
            elapsed = (time.perf_counter() - t0) * 1000
            if elapsed > 2000:
                times_pending.append(elapsed)
            else:
                times_committed.append(elapsed)

        if times_committed:
            avg_c = statistics.mean(times_committed)
            print(f"  assign (no VLM): avg={avg_c:.0f}ms  count={len(times_committed)}")
            results["cluster_no_vlm_ms"] = avg_c
        if times_pending:
            avg_p = statistics.mean(times_pending)
            print(f"  assign (VLM):    avg={avg_p:.0f}ms  count={len(times_pending)}")
            results["cluster_vlm_ms"] = avg_p
        if not times_committed and not times_pending:
            print(f"  no results")
            results["cluster_no_vlm_ms"] = 0
    except Exception as e:
        print(f"  Cluster skipped: {e}")
        results["cluster_no_vlm_ms"] = 0

    # --- SQLite write ---
    print("\n--- SQLite write ---")
    from core.storage.sqlite_storage import SQLiteStorage
    sqlite_db = SQLiteStorage(db_path=config.OCR_DB_PATH, activity_db_path=config.ACTIVITY_DB_PATH)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        sqlite_db.store_sub_frame(
            sub_frame_id=f"subframe_bench_{time.time_ns()}",
            timestamp=datetime.now(timezone.utc),
            window_chunk_id=0, offset_index=0,
            app_name="Bench", window_name="Bench",
        )
        times.append((time.perf_counter() - t0) * 1000)
    avg = statistics.mean(times)
    print(f"  store_sub_frame: avg={avg:.0f}ms")
    results["sqlite_ms"] = avg

    return results


def main():
    print("=" * 70)
    print("store_frame Bottleneck Analysis")
    print("=" * 70)

    # ===== Part 1: Analyze existing log data =====
    entries = parse_log_timings()
    if entries:
        print(f"\n[Log Analysis] Found {len(entries)} store_frame completions\n")

        by_windows = {}
        for e in entries:
            w = e["windows"]
            by_windows.setdefault(w, []).append(e["elapsed_s"])

        print(f"{'Windows':>8} {'Count':>6} {'Avg(s)':>8} {'Min(s)':>8} {'Max(s)':>8} {'P95(s)':>8}")
        print("-" * 55)
        for w in sorted(by_windows.keys()):
            times = by_windows[w]
            avg = statistics.mean(times)
            p95 = sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else max(times)
            print(f"{w:>8} {len(times):>6} {avg:>8.2f} {min(times):>8.2f} {max(times):>8.2f} {p95:>8.2f}")

        # Calculate per-window marginal cost
        if 0 in by_windows and len(by_windows) > 1:
            base = statistics.mean(by_windows[0])
            costs = []
            for w, times in sorted(by_windows.items()):
                if w > 0:
                    per_win = (statistics.mean(times) - base) / w
                    costs.append(per_win)
            if costs:
                avg_per_win = statistics.mean(costs)
                print(f"\n  Base overhead (0 windows): {base:.2f}s")
                print(f"  Per-window marginal cost:  {avg_per_win:.2f}s")
                print(f"  Formula: total ≈ {base:.1f} + {avg_per_win:.1f} × N_windows")
    else:
        print("\n[Log Analysis] No store_frame entries found in log")

    # ===== Part 2: Throughput simulation =====
    interval = float(os.environ.get("CAPTURE_INTERVAL_SECONDS", getattr(config, "CAPTURE_INTERVAL_SECONDS", 3)))
    print(f"\n{'=' * 70}")
    print(f"Throughput Simulation (interval={interval}s)")
    print(f"{'=' * 70}")

    if entries:
        avg_all = statistics.mean([e["elapsed_s"] for e in entries])
        avg_multi_win = statistics.mean([e["elapsed_s"] for e in entries if e["windows"] > 0]) if any(e["windows"] > 0 for e in entries) else 0

        frames_per_5min = 5 * 60 / interval * 2  # 2 monitors
        processing_capacity = 5 * 60 / avg_all if avg_all > 0 else float('inf')

        print(f"\n  Capture rate:     {2/interval:.1f} frames/sec ({frames_per_5min:.0f} per 5min)")
        print(f"  Avg processing:   {avg_all:.2f}s/frame")
        print(f"  Processing rate:  {1/avg_all:.2f} frames/sec ({processing_capacity:.0f} per 5min)")
        print(f"  Utilization:      {(2/interval) / (1/avg_all) * 100:.0f}%")

        if (2/interval) > (1/avg_all):
            deficit = (2/interval) - (1/avg_all)
            print(f"\n  ⚠ OVERLOADED: producing {deficit:.2f} frames/sec faster than processing")
            print(f"    After 5 min: ~{deficit * 300:.0f} frames backlogged")
            print(f"    Each backlogged frame adds ~{avg_all:.1f}s to queue latency")
        else:
            headroom = (1/avg_all) - (2/interval)
            print(f"\n  ✓ OK: {headroom:.2f} frames/sec headroom")

    # ===== Part 3: Live benchmark =====
    print(f"\n{'=' * 70}")
    print("Live Component Benchmarks")
    print("=" * 70)
    component_results = benchmark_components()

    # ===== Part 4: Per-window breakdown =====
    print(f"\n{'=' * 70}")
    print("Per-Window Cost Breakdown (from benchmarks)")
    print("=" * 70)
    emb = component_results.get("embedding_ms", 250)
    ocr = component_results.get("ocr_ms", 1500)
    cluster = component_results.get("cluster_no_vlm_ms", 20)
    sqlite = component_results.get("sqlite_ms", 2)
    vlm = component_results.get("cluster_vlm_ms", 5000)
    total = emb + ocr + cluster + sqlite

    print(f"\n  Embedding:     {emb:>7.0f}ms  ({emb/total*100:>4.0f}%)")
    print(f"  Region OCR:    {ocr:>7.0f}ms  ({ocr/total*100:>4.0f}%)")
    print(f"  Cluster:       {cluster:>7.0f}ms  ({cluster/total*100:>4.0f}%)")
    print(f"  SQLite:        {sqlite:>7.0f}ms  ({sqlite/total*100:>4.0f}%)")
    print(f"  ─────────────────────────")
    print(f"  Total/window:  {total:>7.0f}ms")
    if vlm:
        print(f"  + VLM call:    {vlm:>7.0f}ms  (only for new pending leaders)")

    print(f"\n  Budget per interval ({interval}s): {interval*1000:.0f}ms")
    max_windows = int(interval * 1000 / total) if total > 0 else 999
    print(f"  Max windows/interval without pileup: {max_windows}")


if __name__ == "__main__":
    main()
