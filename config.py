# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file

class Config:
    # ============================================
    # Storage Mode Selection (Core)
    # ============================================
    # Options:
    #   - simple: Simple file storage (default, Naive implementation)
    #   - vector: Vector database storage (advanced, requires CLIP+LanceDB)
    STORAGE_MODE = os.environ.get("STORAGE_MODE", "simple")

    # ============================================
    # Module Selection
    # ============================================
    CAPTURER_TYPE = os.environ.get("CAPTURER_TYPE", "screenshot")
    PREPROCESSOR_TYPE = os.environ.get("PREPROCESSOR_TYPE", "simple") 
    # VLM backend type:
    #   - vllm: Use OpenAI format interface (/v1/chat/completions)
    #   - transformer: Use generate interface (/generate)
    VLM_BACKEND_TYPE = os.environ.get("VLM_BACKEND_TYPE", "vllm") 
    
    # ============================================
    # Simple Mode Configuration
    # ============================================
    STORAGE_ROOT = os.environ.get("STORAGE_ROOT", "./visualmem_storage")
    IMAGE_STORAGE_PATH = os.environ.get(
        "IMAGE_STORAGE_PATH",
        os.path.join(STORAGE_ROOT, "visualmem_image"),
    )
    # Benchmark name (if set, automatically switches to the benchmark dataset paths)
    BENCHMARK_NAME = os.environ.get("BENCHMARK_NAME", "").strip() or None
    # Benchmark dataset image root directory, default: IMAGE_STORAGE_PATH/benchmarks
    BENCHMARK_IMAGE_ROOT = os.environ.get(
        "BENCHMARK_IMAGE_ROOT",
        os.path.join(IMAGE_STORAGE_PATH, "benchmarks"),
    )
    # Benchmark dataset database root directory, default: STORAGE_ROOT/dbs_benchmark
    BENCHMARK_DB_ROOT = os.environ.get(
        "BENCHMARK_DB_ROOT",
        os.path.join(STORAGE_ROOT, "dbs_benchmark"),
    )
    # OCR SQLite database path (can be automatically redirected by BENCHMARK_NAME)
    OCR_DB_PATH = os.environ.get(
        "OCR_DB_PATH",
        os.path.join(STORAGE_ROOT, "visualmem_ocr.db"),
    )
    # Text index LanceDB path (can be automatically redirected by BENCHMARK_NAME)
    TEXT_LANCEDB_PATH = os.environ.get(
        "TEXT_LANCEDB_PATH",
        os.path.join(STORAGE_ROOT, "visualmem_textdb"),
    )
    MAX_IMAGES_TO_LOAD = int(os.environ.get("MAX_IMAGES_TO_LOAD", "19"))
    
    # ============================================
    # Vector Mode Configuration (if STORAGE_MODE=vector)
    # ============================================
    ENABLE_CLIP_ENCODER = os.environ.get("ENABLE_CLIP_ENCODER", "false").lower() == "true"
    # For multimodal RAG, we need image-text alignment, so must use CLIP/ALIGN series embedding models
    CLIP_MODEL = os.environ.get("CLIP_MODEL", "google/siglip-large-patch16-384")
    LANCEDB_PATH = os.environ.get(
        "LANCEDB_PATH",
        os.path.join(STORAGE_ROOT, "visualmem_lancedb"),
    )

    # ============================================
    # Query Enhancement
    # ============================================
    ENABLE_LLM_REWRITE = os.environ.get("ENABLE_LLM_REWRITE", "false").lower() == "true"
    ENABLE_TIME_FILTER = os.environ.get("ENABLE_TIME_FILTER", "false").lower() == "true"
    QUERY_REWRITE_NUM = int(os.environ.get("QUERY_REWRITE_NUM", "3"))

    # ============================================
    # GUI Mode (local disk vs remote backend)
    # ============================================
    # GUI_MODE:
    #   - "local": GUI writes to local disk (default, current behavior)
    #   - "remote": GUI uploads frames via HTTP to a backend server
    GUI_MODE = os.environ.get("GUI_MODE", "local").lower()
    # When GUI_MODE="remote", GUI will send HTTP requests to this backend
    GUI_REMOTE_BACKEND_URL = os.environ.get("GUI_REMOTE_BACKEND_URL", "").strip()
    # ============================================
    # Hybrid Search Configuration
    # ============================================
    ENABLE_HYBRID = os.environ.get("ENABLE_HYBRID", "false").lower() == "true"
    

    # Query Rewrite Independent API Configuration (optional, defaults to VLM config)
    # If these values are set, query rewrite will use an independent API, otherwise use VLM config
    QUERY_REWRITE_API_KEY = os.environ.get("QUERY_REWRITE_API_KEY", "")
    QUERY_REWRITE_BASE_URL = os.environ.get("QUERY_REWRITE_BASE_URL", "")
    QUERY_REWRITE_MODEL = os.environ.get("QUERY_REWRITE_MODEL", "")
    
    # ============================================
    # Reranker Configuration
    # ============================================
    ENABLE_RERANK = os.environ.get("ENABLE_RERANK", "false").lower() == "true"
    RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", "10"))
    
    # Reranker model configuration (local mode)
    RERANK_MODEL = os.environ.get("RERANK_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
    
    # ============================================
    # Image Compression Configuration
    # ============================================
    # Maximum screenshot width (maintains aspect ratio), 0 means no compression
    # Recommended values: 1280 (720p) or 1920 (1080p)
    # although clip model uses 384x384, image should be larger to improve ocr and VLM performance
    MAX_IMAGE_WIDTH = int(os.environ.get("MAX_IMAGE_WIDTH", "1280"))
    # JPEG quality (1-100), used for compressed storage
    IMAGE_QUALITY = int(os.environ.get("IMAGE_QUALITY", "80"))
    # Image storage format (JPEG or PNG)
    IMAGE_FORMAT = os.environ.get("IMAGE_FORMAT", "JPEG")
    
    # ============================================
    # Preprocessing Parameters
    # ============================================
    SIMPLE_FILTER_DIFF_THRESHOLD = float(os.environ.get("SIMPLE_FILTER_DIFF_THRESHOLD", "0.006"))
    
    # OCR configuration (enabled by default)
    ENABLE_OCR = os.environ.get("ENABLE_OCR", "true").lower() == "true"
    
    # Frame difference filtering during query (enabled by default)
    # If enabled: only feed images with frame difference > 0.006 to VLM
    # If disabled: directly feed all recent images to VLM
    ENABLE_QUERY_FRAME_DIFF = os.environ.get("ENABLE_QUERY_FRAME_DIFF", "true").lower() == "true"

    # ============================================
    # VLM Configuration (required for both modes)
    # ============================================
    VLM_API_KEY = os.environ.get("VLM_API_KEY", "")
    # API base address (only needs host:port, endpoint path will be automatically added based on VLM_BACKEND_TYPE)
    VLM_API_URI = os.environ.get("VLM_API_URI", "http://localhost:8081")
    VLM_API_MODEL = os.environ.get("VLM_API_MODEL", "Qwen2.5-VL-3B-Instruct")

    # ============================================
    # Runtime Parameters
    # ============================================
    CAPTURE_INTERVAL_SECONDS = int(os.environ.get("CAPTURE_INTERVAL_SECONDS", "3"))
    
    # ============================================
    # Logging Configuration
    # ============================================
    # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

    # ============================================
    # Benchmark Auto-redirect
    # If BENCHMARK_NAME is set, IMAGE_STORAGE_PATH, LANCEDB_PATH,
    # OCR_DB_PATH, TEXT_LANCEDB_PATH will point to resources in the corresponding benchmark directory.
    # ============================================
    if BENCHMARK_NAME:
        _benchmark_dir = os.path.join(BENCHMARK_DB_ROOT, BENCHMARK_NAME)
        IMAGE_STORAGE_PATH = os.environ.get(
            "IMAGE_STORAGE_PATH",
            os.path.join(BENCHMARK_IMAGE_ROOT, BENCHMARK_NAME),
        )
        LANCEDB_PATH = os.environ.get(
            "LANCEDB_PATH",
            os.path.join(_benchmark_dir, "lancedb"),
        )
        OCR_DB_PATH = os.environ.get(
            "OCR_DB_PATH",
            os.path.join(_benchmark_dir, "ocr.db"),
        )
        TEXT_LANCEDB_PATH = os.environ.get(
            "TEXT_LANCEDB_PATH",
            os.path.join(_benchmark_dir, "textdb"),
        )

# Export a singleton instance
config = Config()


