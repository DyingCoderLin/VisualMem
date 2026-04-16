#!/usr/bin/env python3
"""
Compression Benchmark: MP4 video chunks vs individual JPEG frames

Captures the screen for 5 minutes with frame-diff deduplication enabled,
then compares storage size between:
  A) MP4 video chunks (H.265 encoded via FFmpeg)
  B) Individual JPEG frames (quality=80, the original approach)

Usage:
    python scripts/benchmark_compression.py [--duration 300] [--interval 1.0] [--threshold 0.006]

Output:
    benchmark_output/
    ├── mp4_mode/          # MP4 video chunks
    │   ├── screens/       # Full-screen video chunks
    │   └── windows/       # Per-window video chunks
    ├── jpeg_mode/         # Individual JPEG frames
    │   ├── screens/       # Full-screen JPEG images
    │   └── windows/       # Per-window JPEG images
    └── benchmark_report.txt
"""

import os
import sys
import io
import time
import shutil
import argparse
import datetime
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from utils.logger import setup_logger
from core.preprocess.frame_diff import FrameDiffDetector
from utils.data_models import ScreenObject, WindowFrame

logger = setup_logger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "benchmark_output"
MP4_DIR = OUTPUT_DIR / "mp4_mode"
JPEG_DIR = OUTPUT_DIR / "jpeg_mode"
JPEG_QUALITY = 80


def get_dir_size_bytes(path: Path) -> int:
    """Sum of file sizes under path (for subdir breakdown)."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def get_dir_size_via_du(path: Path) -> int:
    """
    Get directory size using `du -sk` (same as `du -sh` in bytes).
    Returns size in bytes. Used for compression ratio so it matches shell `du`.
    """
    path = path.resolve()
    if not path.exists():
        return 0
    try:
        # -k = 1024-byte units; output is "12345\t/path"
        out = subprocess.run(
            ["du", "-sk", str(path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return get_dir_size_bytes(path)
        size_k = int(out.stdout.strip().split()[0])
        return size_k * 1024
    except (ValueError, IndexError, FileNotFoundError, subprocess.TimeoutExpired):
        return get_dir_size_bytes(path)


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def capture_screen_and_windows() -> Optional[dict]:
    """Capture full screen and all windows using screencap_rs or fallback."""
    try:
        from screencap_rs import screencap_rs
        screen_result, window_results = screencap_rs.capture_screen_with_windows(
            monitor_id=None,
            include_minimized=False,
            filter_system=True,
        )
        if screen_result is None:
            return None

        full_screen_bytes = screen_result.get_image_bytes()
        monitor_id = screen_result.monitor.id if screen_result.monitor else 0

        windows = []
        for w in (window_results or []):
            try:
                wb = w.get_image_bytes()
                if wb and len(wb) > 0:
                    info = w.info
                    windows.append({
                        "app_name": getattr(info, "app_name", "Unknown") or "Unknown",
                        "window_name": getattr(info, "title", "Unknown") or "Unknown",
                        "bytes": wb,
                    })
            except Exception:
                pass

        return {
            "full_screen_bytes": full_screen_bytes,
            "monitor_id": monitor_id,
            "windows": windows,
        }
    except ImportError:
        logger.warning("screencap_rs not available, generating synthetic test frames")
        img = Image.new("RGB", (1920, 1080), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return {
            "full_screen_bytes": buf.getvalue(),
            "monitor_id": 0,
            "windows": [],
        }
    except Exception as e:
        logger.error(f"Capture failed: {e}")
        return None


def calculate_image_hash(image: Image.Image) -> int:
    import hashlib
    small = image.resize((64, 64), Image.Resampling.LANCZOS)
    return int(hashlib.md5(small.tobytes()).hexdigest()[:15], 16)


def run_capture_loop(
    duration_seconds: int,
    interval: float,
    diff_threshold: float,
) -> Tuple[List[dict], int, int, int, int]:
    """
    Run capture loop, return deduplicated frames and statistics.

    Returns:
        (frames_list, total_captured, skipped_screen, stored_screen, stored_windows)

    Each entry in frames_list:
        {
            "timestamp": datetime,
            "screen_image": Image,
            "windows": [{"app_name", "window_name", "image": Image}, ...]
        }
    """
    diff_detector = FrameDiffDetector(
        screen_threshold=diff_threshold,
        window_threshold=diff_threshold,
    )

    frames: List[dict] = []
    total_captured = 0
    skipped_screen = 0
    stored_screen = 0
    stored_windows = 0

    start = time.time()
    next_tick = start

    while time.time() - start < duration_seconds:
        now = time.time()
        if now < next_tick:
            time.sleep(next_tick - now)

        loop_start = time.time()
        total_captured += 1

        result = capture_screen_and_windows()
        if result is None:
            next_tick = loop_start + interval
            continue

        ts = datetime.datetime.now(datetime.timezone.utc)
        screen_image = Image.open(io.BytesIO(result["full_screen_bytes"])).convert("RGB")
        screen_hash = calculate_image_hash(screen_image)

        screen_obj = ScreenObject(
            monitor_id=result["monitor_id"],
            device_name=f"monitor_{result['monitor_id']}",
            timestamp=ts,
            full_screen_image=screen_image,
            full_screen_hash=screen_hash,
            windows=[],
        )

        screen_diff = diff_detector.check_screen_diff(screen_obj)
        if not screen_diff.should_store:
            skipped_screen += 1
            next_tick = loop_start + interval
            continue

        stored_screen += 1

        kept_windows = []
        for w in result.get("windows", []):
            win_image = Image.open(io.BytesIO(w["bytes"])).convert("RGB")
            win_hash = calculate_image_hash(win_image)
            wf = WindowFrame(
                app_name=w["app_name"],
                window_name=w["window_name"],
                image=win_image,
                image_hash=win_hash,
                timestamp=ts,
            )
            win_diff = diff_detector.check_window_diff(wf)
            if win_diff.should_store:
                kept_windows.append({
                    "app_name": w["app_name"],
                    "window_name": w["window_name"],
                    "image": win_image,
                })
                stored_windows += 1

        frames.append({
            "timestamp": ts,
            "screen_image": screen_image,
            "windows": kept_windows,
        })

        elapsed = time.time() - loop_start
        print(
            f"\r  Captured {stored_screen} / {total_captured} "
            f"(skipped {skipped_screen}), "
            f"windows kept this tick: {len(kept_windows)}, "
            f"elapsed: {time.time() - start:.0f}s / {duration_seconds}s",
            end="",
            flush=True,
        )

        next_tick = loop_start + interval

    print()
    return frames, total_captured, skipped_screen, stored_screen, stored_windows


def sanitize(name: str) -> str:
    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " "]:
        name = name.replace(ch, "_")
    return name[:50]


def store_as_jpeg(frames: List[dict]):
    """Store all frames as individual JPEG files."""
    screen_dir = JPEG_DIR / "screens"
    window_dir = JPEG_DIR / "windows"
    screen_dir.mkdir(parents=True, exist_ok=True)

    for entry in frames:
        ts_str = entry["timestamp"].strftime("%Y%m%d_%H%M%S_%f")
        img: Image.Image = entry["screen_image"]
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        img.save(str(screen_dir / f"{ts_str}.jpg"), "JPEG", quality=JPEG_QUALITY, optimize=True)

        for w in entry["windows"]:
            key = f"{sanitize(w['app_name'])}_{sanitize(w['window_name'])}"
            wdir = window_dir / key
            wdir.mkdir(parents=True, exist_ok=True)
            wimg: Image.Image = w["image"]
            if wimg.mode in ("RGBA", "LA", "P"):
                wimg = wimg.convert("RGB")
            wimg.save(str(wdir / f"{ts_str}.jpg"), "JPEG", quality=JPEG_QUALITY, optimize=True)


def store_as_mp4(frames: List[dict], fps: float = 1.0):
    """Store all frames as MP4 video chunks using the project's VideoChunkWriter."""
    from core.storage.video_chunk_writer import VideoChunkManager

    screen_dir = MP4_DIR / "screens"
    window_dir = MP4_DIR / "windows"
    screen_dir.mkdir(parents=True, exist_ok=True)
    window_dir.mkdir(parents=True, exist_ok=True)

    manager = VideoChunkManager(
        output_dir=str(MP4_DIR),
        fps=fps,
        chunk_duration=60,
    )

    for entry in frames:
        img: Image.Image = entry["screen_image"]
        manager.write_screen_frame(0, img)

        for w in entry["windows"]:
            manager.write_window_frame(w["app_name"], w["window_name"], w["image"])

    manager.close_all()


def main():
    parser = argparse.ArgumentParser(description="Benchmark: MP4 vs JPEG compression ratio")
    parser.add_argument("--duration", type=int, default=300, help="Capture duration in seconds (default: 300 = 5 min)")
    parser.add_argument("--interval", type=float, default=1.0, help="Capture interval in seconds (default: 1.0)")
    parser.add_argument("--threshold", type=float, default=0.006, help="Frame diff threshold (default: 0.006)")
    parser.add_argument("--fps", type=float, default=1.0, help="MP4 FPS (default: 1.0)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Compression Benchmark: MP4 video chunks vs Individual JPEG frames")
    print("=" * 70)
    print(f"  Duration:   {args.duration}s ({args.duration / 60:.1f} min)")
    print(f"  Interval:   {args.interval}s")
    print(f"  Threshold:  {args.threshold}")
    print(f"  JPEG qual:  {JPEG_QUALITY}")
    print(f"  MP4 codec:  H.265 / HEVC  (CRF 23, ultrafast)")
    print(f"  MP4 FPS:    {args.fps}")
    print(f"  Output:     {OUTPUT_DIR}")
    print("=" * 70)

    # Clean up previous run
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Capture
    print("\n[Phase 1] Capturing screen frames with deduplication...")
    cap_start = time.time()
    frames, total_captured, skipped, stored_screen, stored_windows = run_capture_loop(
        args.duration, args.interval, args.threshold
    )
    cap_time = time.time() - cap_start
    print(f"  Capture done in {cap_time:.1f}s")
    print(f"  Total ticks:          {total_captured}")
    print(f"  Screen frames stored: {stored_screen}")
    print(f"  Screen frames skipped:{skipped}")
    print(f"  Window frames stored: {stored_windows}")

    if not frames:
        print("\nNo frames captured. Exiting.")
        return

    # Phase 2: Store as JPEG
    print("\n[Phase 2] Storing frames as individual JPEG files...")
    jpeg_start = time.time()
    store_as_jpeg(frames)
    jpeg_time = time.time() - jpeg_start
    print(f"  JPEG storage done in {jpeg_time:.1f}s")

    # Phase 3: Store as MP4
    print("\n[Phase 3] Storing frames as MP4 video chunks (H.265)...")
    mp4_start = time.time()
    store_as_mp4(frames, fps=args.fps)
    mp4_time = time.time() - mp4_start
    print(f"  MP4 storage done in {mp4_time:.1f}s")

    # Phase 4: Measure sizes (via du -sk, same as shell `du -sh`)
    print("\n[Phase 4] Calculating storage sizes (du)...")
    jpeg_total = get_dir_size_via_du(JPEG_DIR)
    mp4_total = get_dir_size_via_du(MP4_DIR)
    jpeg_screen_size = get_dir_size_via_du(JPEG_DIR / "screens")
    jpeg_window_size = get_dir_size_via_du(JPEG_DIR / "windows") if (JPEG_DIR / "windows").exists() else 0
    mp4_screen_size = get_dir_size_via_du(MP4_DIR / "screens")
    mp4_window_size = get_dir_size_via_du(MP4_DIR / "windows") if (MP4_DIR / "windows").exists() else 0

    ratio = jpeg_total / mp4_total if mp4_total > 0 else float("inf")

    # Count files
    jpeg_file_count = sum(1 for _ in JPEG_DIR.rglob("*.jpg"))
    mp4_file_count = sum(1 for _ in MP4_DIR.rglob("*.mp4"))

    # Report
    report_lines = [
        "=" * 70,
        "  COMPRESSION BENCHMARK REPORT",
        "=" * 70,
        "",
        "Configuration:",
        f"  Duration:          {args.duration}s ({args.duration / 60:.1f} min)",
        f"  Capture interval:  {args.interval}s",
        f"  Diff threshold:    {args.threshold}",
        f"  JPEG quality:      {JPEG_QUALITY}",
        f"  MP4 codec:         H.265 (CRF 23, ultrafast)",
        "",
        "Capture Statistics:",
        f"  Total ticks:          {total_captured}",
        f"  Screen frames stored: {stored_screen} (skipped: {skipped})",
        f"  Window frames stored: {stored_windows}",
        f"  Dedup rate (screen):  {skipped / total_captured * 100:.1f}%" if total_captured > 0 else "  N/A",
        "",
        "JPEG Mode (individual files):",
        f"  Screen size:  {format_size(jpeg_screen_size)}",
        f"  Window size:  {format_size(jpeg_window_size)}",
        f"  Total size:   {format_size(jpeg_total)}",
        f"  File count:   {jpeg_file_count} files",
        f"  Write time:   {jpeg_time:.1f}s",
        "",
        "MP4 Mode (video chunks):",
        f"  Screen size:  {format_size(mp4_screen_size)}",
        f"  Window size:  {format_size(mp4_window_size)}",
        f"  Total size:   {format_size(mp4_total)}",
        f"  File count:   {mp4_file_count} files",
        f"  Write time:   {mp4_time:.1f}s",
        "",
        "Comparison:",
        f"  Compression ratio:  {ratio:.2f}x  (JPEG / MP4)",
        f"  Space saved:        {format_size(jpeg_total - mp4_total)}  ({(1 - mp4_total / jpeg_total) * 100:.1f}%)" if jpeg_total > 0 else "  N/A",
        "",
        "=" * 70,
    ]

    report = "\n".join(report_lines)
    print(report)

    report_path = OUTPUT_DIR / "benchmark_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
