#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisualMem Command Line Interface (for remote/no-GUI scenarios)

Interactive steps:
1) Select retrieval source: 0=Semantic/Vector retrieval, 1=OCR full-text retrieval
2) Select work mode: 0=RAG (full database retrieval), 1=Real-time Q&A
3) Enter query text

System resource status will be displayed at startup for remote machine monitoring.
"""

import os
import sys
import shutil
import threading
import asyncio
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone

from config import config
from utils.logger import setup_logger
from core.understand.api_vlm import ApiVLM
from core.storage.sqlite_storage import SQLiteStorage
from gui.workers.record_worker import RecordWorker
from core.retrieval.query_llm_utils import (
    rewrite_and_time,
    filter_by_time,
)
from core.retrieval.reranker import Reranker
from core.encoder.qwen_encoder import QwenEncoder
from core.encoder.clip_encoder import CLIPEncoder
from core.storage.sqlite_storage import SQLiteStorage

logger = setup_logger("cli_main")

# Optional dependencies
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - only executed when dependency is missing
    psutil = None

try:
    from core.encoder import create_encoder
    from core.storage.lancedb_storage import LanceDBStorage
except Exception:
    create_encoder = None  # type: ignore
    LanceDBStorage = None  # type: ignore

try:
    from core.capture.screenshot_capturer import ScreenshotCapturer
except Exception:
    ScreenshotCapturer = None  # type: ignore

try:
    from core.ocr.paddleocr_worker import PaddleOCRWorker
except Exception:
    PaddleOCRWorker = None  # type: ignore

# Recording state
_record_worker: Optional[RecordWorker] = None
_record_thread: Optional[threading.Thread] = None
_record_lock = threading.Lock()

# Vector retrieval components (lazy initialization, reused)
_vector_encoder: Optional[QwenEncoder] = None
_vector_storage: Optional[LanceDBStorage] = None
_vector_initialized = False


def _get_directory_size(directory: Path) -> int:
    """
    Recursively calculate the total size of a directory (in bytes)
    
    Args:
        directory: Directory path
        
    Returns:
        Total directory size (in bytes)
    """
    total_size = 0
    try:
        if directory.exists() and directory.is_dir():
            for entry in directory.rglob('*'):
                try:
                    if entry.is_file():
                        total_size += entry.stat().st_size
                except (OSError, PermissionError):
                    # Ignore inaccessible files
                    pass
    except (OSError, PermissionError):
        pass
    return total_size


def _format_size(bytes_size: int) -> str:
    """
    Format byte size to human-readable format
    
    Args:
        bytes_size: Byte size
        
    Returns:
        Formatted string
    """
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f} KB"
    elif bytes_size < 1024 * 1024 * 1024:
        return f"{bytes_size / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes_size / (1024 * 1024 * 1024):.2f} GB"


def _print_system_status():
    """Print system load information (minimize additional dependencies)"""
    print("=" * 60)
    print("System Status:")
    print("-" * 60)
    # Actually don't need to print CPU status, it's not the heaviest load
    # if psutil:
    #     cpu = psutil.cpu_percent(interval=0.2)
    #     mem = psutil.virtual_memory()
    #     print(f"CPU: {cpu:.1f}% | Memory: {mem.percent:.1f}% ({mem.used/1024/1024/1024:.1f}GB/{mem.total/1024/1024/1024:.1f}GB)")
    # else:
    #     try:
    #         load1, load5, load15 = os.getloadavg()
    #         print(f"Load Avg (1/5/15): {load1:.2f} / {load5:.2f} / {load15:.2f}")
    #     except Exception:
    #         print("Unable to get CPU load (install psutil for better display)")
    
    # Calculate visualmem_storage directory size
    try:
        storage_root = Path(config.STORAGE_ROOT)
        storage_size = _get_directory_size(storage_root)
        print(f"Disk Usage: {_format_size(storage_size)} ({config.STORAGE_ROOT})")
    except Exception as e:
        logger.warning(f"Unable to get project storage size: {e}")
        print("Unable to get project storage information")
    print("=" * 60)


def _prompt_binary(msg: str) -> int:
    """Read 0/1 choice"""
    while True:
        choice = input(f"{msg} (0/1): ").strip()
        if choice in ("0", "1"):
            return int(choice)
        print("Please enter 0 or 1")


def _prompt_query() -> str:
    """Read a query, empty line continues waiting, q/quit/exit exits loop; supports start/stop to control recording"""
    query = input("Please enter your query : ").strip()
    return query


def _ensure_vector_components():
    """Ensure vector retrieval components are initialized (lazy loading, initialize only once)"""
    global _vector_encoder, _vector_storage, _vector_initialized
    
    if not create_encoder or not LanceDBStorage:
        print("Vector retrieval dependencies not installed, please ensure transformers/torch/lancedb are available.")
        sys.exit(1)
    
    if _vector_initialized:
        return
    
    print(f"\n\n🔄 Loading encoder model {config.EMBEDDING_MODEL}... (first load is slow)")
    _vector_encoder = create_encoder(model_name=config.EMBEDDING_MODEL)
    print(f"📦 Initializing LanceDB storage at {config.LANCEDB_PATH}...")
    _vector_storage = LanceDBStorage(
        db_path=config.LANCEDB_PATH,
        embedding_dim=_vector_encoder.embedding_dim
    )
    _vector_initialized = True
    print(f"✅ Encoder model {config.EMBEDDING_MODEL} loaded")


def _vector_rag(query: str, top_k: int = None):
    """Semantic/Vector RAG: Full database retrieval (reuses GUI's time filtering logic)"""
    _ensure_vector_components()
    if top_k is None:
        top_k = config.MAX_IMAGES_TO_LOAD
    print("Mode: Semantic Retrieval -> RAG (Full Database Vector Retrieval)")

    # Reuse initialized encoder and storage
    if _vector_encoder is None or _vector_storage is None:
        print("Error: Vector retrieval components not properly initialized")
        return
    encoder = _vector_encoder
    storage = _vector_storage

    dense_queries = [query]
    sparse_queries = [query]
    time_range = None
    if config.ENABLE_LLM_REWRITE or config.ENABLE_TIME_FILTER:
        print(f"rewriting query: {query}")
        dense_queries, sparse_queries, time_range = rewrite_and_time(
            query,
            enable_rewrite=config.ENABLE_LLM_REWRITE,
            enable_time=config.ENABLE_TIME_FILTER,
            expand_n=config.QUERY_REWRITE_NUM,
        )
        print(f"dense_queries: {dense_queries}")
        print(f"sparse_queries: {sparse_queries}")
        print(f"time_range: {time_range}")

    # Extract time range (for LanceDB Pre-filtering)
    start_time = None
    end_time = None
    if time_range:
        start_time, end_time = time_range
        print(f"⏰ Time Range: {start_time} - {end_time}")
        print("🔍 Using LanceDB Pre-filtering for vector retrieval...")

    # Define helper function: Dense search
    def _dense_search_task():
        """Dense search task (synchronous function)"""
        dense_frames = []
        for q in dense_queries:
            embedding = encoder.encode_text(q)
            # print(f"[Dense] searching for query '{q}' for {top_k} relevant frames")
            res = storage.search(
                embedding, 
                top_k=top_k,
                start_time=start_time,
                end_time=end_time
            )
            # print(f"[Dense] res num: {len(res)}")
            dense_frames.extend(res)
        return dense_frames
    
    # Define helper function: Sparse search (using SQLite FTS5, no textdb)
    def _sparse_search_task():
        """Sparse search task (synchronous function, pure SQLite FTS5)"""
        if not config.ENABLE_HYBRID:
            return []
        
        try:
            sqlite_storage = SQLiteStorage(db_path=config.OCR_DB_PATH)
            sparse_frames = []
            # Search for each sparse query via SQLite FTS5
            for q in sparse_queries:
                # print(f"[Sparse] searching for query '{q}' for {top_k} relevant frames")
                sparse_results = sqlite_storage.search_by_text(q, limit=top_k)
                # Apply time filter (in Python, result集很小影响不大)
                if time_range:
                    sparse_results = filter_by_time(sparse_results, time_range)
                # print(f"[Sparse] res num: {len(sparse_results)}")
                
                # Convert sparse results to frame format
                for result in sparse_results:
                    fid = result.get("frame_id")
                    if not fid:
                        continue
                    
                    # SQLite FTS5 不直接给打分，这里用占位 distance=1.0（后续 rerank 会重新打分）
                    frame = {
                        "frame_id": fid,
                        "timestamp": datetime.fromisoformat(result.get("timestamp")) if isinstance(result.get("timestamp"), str) else result.get("timestamp"),
                        "image_path": result.get("image_path"),
                        "ocr_text": result.get("ocr_text") or result.get("text", ""),
                        "distance": 1.0,  # placeholder, real relevance will be decided by reranker
                        "metadata": result.get("metadata", {}),
                        "_from_sparse": True  # Mark as from sparse search
                    }
                    
                    # Load image
                    if frame.get("image_path"):
                        try:
                            from pathlib import Path
                            img_path = Path(frame["image_path"])
                            
                            if img_path.exists():
                                from PIL import Image
                                frame["image"] = Image.open(img_path)
                            else:
                                logger.warning(f"Image not found: {frame['image_path']} (tried: {img_path})")
                        except Exception as e:
                            logger.warning(f"Failed to load image for frame {fid}: {e}")
                    
                    sparse_frames.append(frame)
            
            return sparse_frames
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}", exc_info=True)
            print(f"⚠️ Sparse retrieval failed, using Dense results only: {e}")
            return []
    
    # Execute Dense and Sparse searches in parallel (using asyncio)
    if config.ENABLE_HYBRID:
        print("🔍 Enabling Hybrid Search, executing Dense and Sparse retrieval in parallel...")
        # Use asyncio to execute in parallel
        async def _run_parallel_searches():
            dense_task = asyncio.to_thread(_dense_search_task)
            sparse_task = asyncio.to_thread(_sparse_search_task)
            return await asyncio.gather(dense_task, sparse_task)
        
        dense_results, sparse_results = asyncio.run(_run_parallel_searches())
    else:
        # Only execute Dense search
        dense_results = _dense_search_task()
        sparse_results = []
    
    # Merge results and deduplicate
    frames = []
    seen = set()
    
    # Add Dense results
    for r in dense_results:
        fid = r.get("frame_id")
        if fid in seen:
            continue
        seen.add(fid)
        frames.append(r)
    
    # Add Sparse results
    sparse_added = 0
    sparse_skipped_duplicate = 0
    sparse_no_image = 0
    for r in sparse_results:
        fid = r.get("frame_id")
        if fid in seen:
            sparse_skipped_duplicate += 1
            continue
        seen.add(fid)
        # Check if image loaded successfully
        if r.get("image") is None:
            sparse_no_image += 1
        frames.append(r)
        sparse_added += 1
    
    if config.ENABLE_HYBRID:
        logger.debug(f"[Debug] Sparse results: total={len(sparse_results)}, added={sparse_added}, duplicate_skipped={sparse_skipped_duplicate}, no_image={sparse_no_image}")
    
    if not frames:
        print("No relevant screenshots found.")
        return
    
    # Ensure all frames have images
    frames_with_images = [f for f in frames if f.get("image") is not None]
    if not frames_with_images:
        print("Retrieved images cannot be loaded, check paths or storage.")
        return
    
    # Statistics use frames_with_images (only count successfully loaded images)
    dense_count = len([f for f in frames_with_images if not f.get('_from_sparse', False)])
    sparse_count = len([f for f in frames_with_images if f.get('_from_sparse', False)])
    if config.ENABLE_HYBRID:
        logger.info(f"Retrieved {len(frames_with_images)} relevant images (Dense: {dense_count}, Sparse: {sparse_count})")
    else:
        print(f"Retrieved {len(frames_with_images)} relevant images")
    
    # Rerank step (if enabled)
    if config.ENABLE_RERANK:
        print(f"🔄 Reranking (returning top-{config.RERANK_TOP_K})...")
        reranker = Reranker()
        frames_with_images = reranker.rerank(
            query=query,
            frames=frames_with_images,
            top_k=config.RERANK_TOP_K
        )
        
        if not frames_with_images:
            print("❌ No images after rerank, cannot perform VLM analysis.")
            return
    
    images = [f["image"] for f in frames_with_images]
    logger.debug(f"Calling VLM to analyze {len(images)} images...")
    
    # Extract timestamps
    timestamps = [f.get("timestamp") for f in frames_with_images]
    
    # System prompt: Define assistant role
    system_prompt = "You are a helpful visual assistant. You analyze screenshots to answer user questions. Always respond in Chinese (中文回答)."
    
    # User prompt: Answer question directly, then provide supporting evidence from images
    prompt = f"""User Question: {query}

Please directly answer the user's question first, then provide supporting evidence from the {len(images)} screenshots below. Focus on what the user was doing and how the visual content relates to their question."""
    
    # Update frames reference (may change after rerank)
    frames = frames_with_images
    
    vlm = ApiVLM()
    response = vlm._call_vlm(
        prompt, 
        images, 
        num_images=len(images),
        image_timestamps=timestamps if timestamps else None,
        system_prompt=system_prompt
    )
    print("\n=== VLM Response ===")
    print(response)


def _ocr_rag(query: str, limit: int = 20):
    """OCR Full-text RAG: Default full database, no time range filtering"""
    print("Mode: OCR Full-text Retrieval -> Text Understanding")
    sqlite_storage = SQLiteStorage(db_path=config.OCR_DB_PATH)

    time_range = None
    if config.ENABLE_LLM_REWRITE or config.ENABLE_TIME_FILTER:
        # OCR mode only uses original query for FTS, but can use LLM to parse time
        _, _, time_range = rewrite_and_time(
            query,
            enable_rewrite=False,
            enable_time=config.ENABLE_TIME_FILTER,
            expand_n=config.QUERY_REWRITE_NUM,
        )

    results = sqlite_storage.search_text(query, limit=limit)
    if time_range:
        results = filter_by_time(results, time_range)
    if not results:
        print("No relevant OCR text found.")
        return
    
    snippets = []
    for i, r in enumerate(results[:15], 1):
        ts = r.get("timestamp")
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else str(ts)
        text = (r.get("ocr_text", "") or "")[:500]
        snippets.append(f"[{i}] Time: {ts_str}\nContent: {text}")
    
    vlm = ApiVLM()
    
    # System prompt: Define assistant role
    system_prompt = "You are a helpful assistant. You analyze OCR text to answer user questions. Always respond in Chinese (中文回答)."
    
    # User prompt: Answer question directly
    prompt = f"""User Question: {query}

Please directly answer the user's question first, then provide supporting evidence from the OCR text below.

OCR Records (sorted by relevance):

{chr(10).join(snippets)}
"""
    response = vlm._call_vlm_text_only(prompt, system_prompt=system_prompt)
    print("\n=== VLM Response ===")
    print(response)


def _collect_realtime_images() -> Tuple[List, List]:
    """Get current screenshot + last 5 historical screenshots and their timestamps"""
    if not ScreenshotCapturer:
        print("Screenshot module unavailable, please check dependencies.")
        return [], []
    
    capturer = ScreenshotCapturer()
    current = capturer.capture()
    if not current:
        print("Unable to get current screen screenshot.")
        return [], []
    
    images = [current.image]
    timestamps = [current.timestamp]
    try:
        sqlite_storage = SQLiteStorage(db_path=config.OCR_DB_PATH)
        recent = sqlite_storage.get_recent_frames(limit=5)
        for frame_meta in recent:
            try:
                path = Path(frame_meta["image_path"])
                if path.exists():
                    from PIL import Image
                    images.append(Image.open(path))
                    timestamps.append(frame_meta["timestamp"])
            except Exception as e:  # pragma: no cover - failure doesn't affect main flow
                logger.warning(f"Failed to load historical screenshot: {e}")
    except Exception as e:
        logger.warning(f"Failed to read historical screenshots: {e}")
    
    return images, timestamps


def _realtime_visual(query: str):
    """Real-time Q&A (Visual)"""
    print("Mode: Real-time Q&A (Visual)")
    images, timestamps = _collect_realtime_images()
    if not images:
        print("No images obtained, cannot perform real-time Q&A.")
        return
    
    # System prompt: Define assistant role
    system_prompt = "You are a helpful visual assistant. You analyze screenshots to answer user questions. Always respond in Chinese (中文回答)."
    
    # User prompt: Answer question directly
    prompt = f"""User Question: {query}

The first image is the current screen, and the remaining images are recent history. Please directly answer the user's question first, then provide supporting evidence from the screenshots."""
    
    vlm = ApiVLM()
    response = vlm._call_vlm(
        prompt, 
        images, 
        num_images=len(images),
        image_timestamps=timestamps,
        system_prompt=system_prompt
    )
    print("\n=== Real-time Response ===")
    print(response)
    print(f"\nAnalyzed {len(images)} images (1 current + {len(images)-1} historical)")


def _realtime_ocr(query: str):
    """Real-time Q&A (OCR Text)"""
    print("Mode: Real-time Q&A (OCR Text)")
    ocr_texts: List[Dict] = []
    
    # Current screen OCR
    if PaddleOCRWorker and ScreenshotCapturer:
        try:
            capturer = ScreenshotCapturer()
            current = capturer.capture()
            if current:
                worker = PaddleOCRWorker()
                text = worker.process(current.image)
                if text:
                    ocr_texts.append({"time": "Current", "text": text})
        except Exception as e:
            logger.warning(f"Current screen OCR failed: {e}")
    else:
        print("OCR dependencies missing (PaddleOCR or screenshot module), skipping current screen OCR.")
    
    # Historical OCR
    try:
        sqlite_storage = SQLiteStorage(db_path=config.OCR_DB_PATH)
        recent = sqlite_storage.get_recent_frames(limit=10)
        for frame in recent:
            text = frame.get("ocr_text", "")
            if text:
                # 将存储的 UTC 时间转换为本地时间
                ts_obj = frame["timestamp"]
                if ts_obj.tzinfo is None:
                    ts_obj = ts_obj.replace(tzinfo=timezone.utc)
                ts = ts_obj.astimezone().strftime("%H:%M:%S")
                
                ocr_texts.append({"time": ts, "text": text[:500]})
    except Exception as e:
        logger.warning(f"Failed to get historical OCR: {e}")
    
    if not ocr_texts:
        print("No OCR text available.")
        return
    
    context = "\n\n---\n\n".join(f"[{item['time']}]\n{item['text']}" for item in ocr_texts)
    
    # System prompt: Define assistant role
    system_prompt = "You are a helpful assistant. You analyze OCR text to answer user questions. Always respond in Chinese (中文回答)."
    
    # User prompt: Answer question directly
    prompt = f"""User Question: {query}

Please directly answer the user's question first, then provide supporting evidence from the OCR text below.

Current and Historical Screen OCR Text:

{context}
"""
    vlm = ApiVLM()
    response = vlm._call_vlm_text_only(prompt, system_prompt=system_prompt)
    print("\n=== Real-time OCR Response ===")
    print(response)
    print(f"\nAnalyzed {len(ocr_texts)} OCR texts")


def _start_recording():
    """Start background recording thread"""
    global _record_worker, _record_thread
    with _record_lock:
        if _record_thread and _record_thread.is_alive():
            print("Recording is already in progress, enter stop to stop.")
            return
        try:
            _record_worker = RecordWorker(config.STORAGE_MODE)
            _record_thread = threading.Thread(
                target=_record_worker.start_recording, daemon=True
            )
            _record_thread.start()
            print("Recording started, enter stop to stop.")
        except Exception as e:
            print(f"Failed to start recording: {e}")
            _record_worker = None
            _record_thread = None


def _stop_recording():
    """Stop background recording thread"""
    global _record_worker, _record_thread
    with _record_lock:
        if not _record_thread or not _record_thread.is_alive():
            print("Not currently recording, enter start to start recording.")
            return
        try:
            if _record_worker:
                _record_worker.stop_recording()
            _record_thread.join(timeout=5)
            print("Recording stopped.")
        except Exception as e:
            print(f"Failed to stop recording: {e}")
        finally:
            _record_worker = None
            _record_thread = None


def main():
    _print_system_status()
    print("Select retrieval source: 0=Semantic/Vector retrieval  1=OCR full-text retrieval")
    source_choice = _prompt_binary("Please enter choice")
    
    print("Select work mode: 0=RAG (full database)  1=Real-time Q&A")
    mode_choice = _prompt_binary("Please enter choice")
    
    print("\n" + "=" * 70)
    print("📝 User Guide:")
    print("=" * 70)
    print("  • Enter your query to search")
    print("  • Type 'start' to begin recording")
    print("  • Type 'stop' to stop recording")
    print("  • Type 'q', 'quit', or 'exit' to exit")
    print("=" * 70 + "\n")
    try:
        while True:
            query = _prompt_query()
            if not query:
                continue
            if query.lower() in ("q", "quit", "exit"):
                print("Received exit command, ending.")
                _stop_recording()
                break
            if query.lower() == "start":
                _start_recording()
                continue
            if query.lower() == "stop":
                _stop_recording()
                continue
            
            if source_choice == 0 and mode_choice == 0:
                _vector_rag(query)
            elif source_choice == 0 and mode_choice == 1:
                _realtime_visual(query)
            elif source_choice == 1 and mode_choice == 0:
                _ocr_rag(query)
            else:
                _realtime_ocr(query)
            print(f"="*70)
            print("query answered, you can continue asking other questions")
            print(f"="*70)
    except KeyboardInterrupt:
        _stop_recording()
        print("\nUser interrupted, exited.")
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        print(f"Execution failed: {e}")


if __name__ == "__main__":
    main()
