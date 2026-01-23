#!/usr/bin/env python3
"""
Test script for the new capture and recording system
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.capture import WindowCapturer, RecordingCoordinator, RecordingConfig, USE_RUST_CAPTURE
from core.storage import SQLiteStorage
from utils.logger import setup_logger

logger = setup_logger(__name__)


def test_capture_once():
    """Test single capture"""
    print("=" * 50)
    print("Testing single capture...")
    print(f"Using Rust module: {USE_RUST_CAPTURE}")
    print("=" * 50)
    
    capturer = WindowCapturer(monitor_id=0)
    
    # Capture screen with windows
    screen_obj = capturer.capture_screen_with_windows()
    
    if screen_obj is None:
        print("ERROR: Failed to capture screen")
        return False
    
    print(f"\nFull screen: {screen_obj.full_screen_image.size}")
    print(f"Monitor ID: {screen_obj.monitor_id}")
    print(f"Timestamp: {screen_obj.timestamp}")
    print(f"Windows captured: {len(screen_obj.windows)}")
    
    for i, w in enumerate(screen_obj.windows):
        print(f"  {i+1}. {w.app_name}: {w.window_name} ({w.image.size})")
    
    # Save full screen for verification
    output_dir = Path("./test_output")
    output_dir.mkdir(exist_ok=True)
    
    screen_path = output_dir / "full_screen.png"
    screen_obj.full_screen_image.save(screen_path)
    print(f"\nSaved full screen to: {screen_path}")
    
    # Save windows
    for i, w in enumerate(screen_obj.windows[:5]):  # Save first 5 windows
        safe_name = w.app_name.replace("/", "_").replace(" ", "_")[:20]
        window_path = output_dir / f"window_{i}_{safe_name}.png"
        w.image.save(window_path)
        print(f"Saved window to: {window_path}")
    
    print("\nSingle capture test PASSED!")
    return True


async def test_recording(iterations=5, interval=2.0):
    """Test recording coordinator"""
    print("\n" + "=" * 50)
    print(f"Testing recording coordinator ({iterations} iterations)...")
    print("=" * 50)
    
    # Config
    config = RecordingConfig(
        output_dir="./test_output/video_chunks",
        monitor_id=0,
        fps=1.0,
        chunk_duration=60,
        capture_windows=True,
        capture_unfocused_windows=True,
        screen_diff_threshold=0.006,
        window_diff_threshold=0.006,
        run_ocr=False,  # Disable OCR for quick test
        run_embedding=False,  # Disable embedding for quick test
    )
    
    # Create storage
    db = SQLiteStorage(db_path="./test_output/test.db")
    
    # Callbacks
    def on_frame_stored(frame_id, info):
        print(f"  Frame stored: {frame_id}")
    
    def on_subframe_stored(sub_frame_id, info):
        print(f"  Sub-frame stored: {sub_frame_id} ({info['app_name']})")
    
    # Run recording
    with RecordingCoordinator(
        config=config,
        db=db,
        on_frame_stored=on_frame_stored,
        on_subframe_stored=on_subframe_stored
    ) as recorder:
        await recorder.run_continuous(
            interval=interval,
            max_iterations=iterations
        )
        
        # Print stats
        stats = recorder.get_stats()
        print(f"\nRecording stats:")
        print(f"  Frames captured: {stats['frames_captured']}")
        print(f"  Frames stored: {stats['frames_stored']}")
        print(f"  Windows captured: {stats['windows_captured']}")
        print(f"  Windows stored: {stats['windows_stored']}")
        print(f"  Runtime: {stats['runtime_seconds']:.1f}s")
    
    # Print DB stats
    db_stats = db.get_stats()
    print(f"\nDatabase stats:")
    print(f"  Total frames: {db_stats['total_frames']}")
    print(f"  Video chunks: {db_stats.get('total_video_chunks', 0)}")
    print(f"  Window chunks: {db_stats.get('total_window_chunks', 0)}")
    print(f"  Sub-frames: {db_stats.get('total_sub_frames', 0)}")
    
    print("\nRecording test PASSED!")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test capture and recording system")
    parser.add_argument("--capture", action="store_true", help="Test single capture")
    parser.add_argument("--record", action="store_true", help="Test recording")
    parser.add_argument("--iterations", type=int, default=5, help="Recording iterations")
    parser.add_argument("--interval", type=float, default=2.0, help="Capture interval (seconds)")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # Default to --all if no specific test selected
    if not (args.capture or args.record or args.all):
        args.all = True
    
    success = True
    
    if args.capture or args.all:
        if not test_capture_once():
            success = False
    
    if args.record or args.all:
        if not asyncio.run(test_recording(args.iterations, args.interval)):
            success = False
    
    print("\n" + "=" * 50)
    if success:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 50)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
