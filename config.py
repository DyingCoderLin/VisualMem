# config.py
import os
from dotenv import load_dotenv

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(path: str) -> str:
    """Resolve a path relative to the project root if it is not absolute."""
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(_PROJECT_ROOT, path))


_default_env = os.path.join(_PROJECT_ROOT, ".env")
ENV_FILE = os.environ.get("VISUALMEM_ENV_FILE", _default_env)
# If user specified a relative path via env var, resolve it relative to project root
if not os.path.isabs(ENV_FILE):
    ENV_FILE = os.path.join(_PROJECT_ROOT, ENV_FILE)
load_dotenv(dotenv_path=ENV_FILE)  # Load selected env file (default: <project_root>/.env)

# Default image embedding (SigLIP lighter; set EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B for stronger GPU)
_DEFAULT_EMBEDDING_MODEL = "google/siglip-large-patch16-384"

class Config:
    # ============================================
    # Storage Mode Selection (Core)
    # ============================================
    # Options:
    #   - simple: Simple file storage (Naive implementation)
    #   - vector: Vector database storage (advanced, requires CLIP+LanceDB)
    STORAGE_MODE = os.environ.get("STORAGE_MODE", "vector")

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
    STORAGE_ROOT = _resolve_path(os.environ.get("STORAGE_ROOT", "./visualmem_storage"))
    IMAGE_STORAGE_PATH = _resolve_path(os.environ.get(
        "IMAGE_STORAGE_PATH",
        os.path.join(STORAGE_ROOT, "visualmem_image"),
    ))
    # Benchmark name (if set, automatically switches to the benchmark dataset paths)
    BENCHMARK_NAME = os.environ.get("BENCHMARK_NAME", "").strip() or None
    # Benchmark dataset image root directory, default: IMAGE_STORAGE_PATH/benchmarks
    BENCHMARK_IMAGE_ROOT = _resolve_path(os.environ.get(
        "BENCHMARK_IMAGE_ROOT",
        os.path.join(IMAGE_STORAGE_PATH, "benchmarks"),
    ))
    # Benchmark dataset database root directory, default: STORAGE_ROOT/dbs_benchmark
    BENCHMARK_DB_ROOT = _resolve_path(os.environ.get(
        "BENCHMARK_DB_ROOT",
        os.path.join(STORAGE_ROOT, "dbs_benchmark"),
    ))
    # OCR SQLite database path (can be automatically redirected by BENCHMARK_NAME)
    OCR_DB_PATH = _resolve_path(os.environ.get(
        "OCR_DB_PATH",
        os.path.join(STORAGE_ROOT, "visualmem_ocr.db"),
    ))
    # Activity clustering database (separate DB to avoid write lock contention with capture pipeline)
    ACTIVITY_DB_PATH = _resolve_path(os.environ.get(
        "ACTIVITY_DB_PATH",
        os.path.join(STORAGE_ROOT, "visualmem_activity.db"),
    ))
    # Text index LanceDB path (can be automatically redirected by BENCHMARK_NAME)
    TEXT_LANCEDB_PATH = _resolve_path(os.environ.get(
        "TEXT_LANCEDB_PATH",
        os.path.join(STORAGE_ROOT, "visualmem_textdb"),
    ))
    MAX_IMAGES_TO_LOAD = int(os.environ.get("MAX_IMAGES_TO_LOAD", "20"))
    
    # ============================================
    # Model Loading Configuration
    # ============================================
    # When true, heavy models (encoder, reranker, OCR) are NOT loaded at startup.
    # They are loaded on-demand when recording starts (via /api/load_models).
    MODEL_LAZY_LOAD = os.environ.get("MODEL_LAZY_LOAD", "true").lower() == "true"

    # ============================================
    # Vector Mode Configuration (if STORAGE_MODE=vector)
    # ============================================
    ENABLE_CLIP_ENCODER = os.environ.get("ENABLE_CLIP_ENCODER", "false").lower() == "true"
    # Multimodal embeddings: SigLIP (default, lighter) or Qwen3-VL-Embedding-2B (heavier, stronger).
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)

    # LanceDB Path selection based on model
    def _compute_lancedb_path():
        raw = os.environ.get("STORAGE_ROOT", "./visualmem_storage")
        storage_root = _resolve_path(raw)
        model = os.environ.get("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)

        if "qwen" in model.lower():
            return os.path.join(storage_root, "visualmem_qwen_lancedb")
        else:
            return os.path.join(storage_root, "visualmem_clip_lancedb")

    LANCEDB_PATH = _resolve_path(os.environ.get("LANCEDB_PATH", _compute_lancedb_path()))

    # ============================================
    # Query Enhancement
    # ============================================
    ENABLE_LLM_REWRITE = os.environ.get("ENABLE_LLM_REWRITE", "true").lower() == "true"
    ENABLE_TIME_FILTER = os.environ.get("ENABLE_TIME_FILTER", "true").lower() == "true"
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
    ENABLE_HYBRID = os.environ.get("ENABLE_HYBRID", "true").lower() == "true"
    

    # Query Rewrite Independent API Configuration (optional, defaults to VLM config)
    # If these values are set, query rewrite will use an independent API, otherwise use VLM config
    QUERY_REWRITE_API_KEY = os.environ.get("QUERY_REWRITE_API_KEY", "")
    QUERY_REWRITE_BASE_URL = os.environ.get("QUERY_REWRITE_BASE_URL", "")
    QUERY_REWRITE_MODEL = os.environ.get("QUERY_REWRITE_MODEL", "")
    
    # ============================================
    # Reranker Configuration
    # ============================================
    ENABLE_RERANK = os.environ.get("ENABLE_RERANK", "false").lower() == "true"
    RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", "20"))
    
    # Reranker model configuration (local mode)
    RERANK_MODEL = os.environ.get("RERANK_MODEL", "Qwen/Qwen3-VL-Reranker-2B")
    
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
    # OCR engine type: "auto" (platform-native), "apple_vision", "windows_ocr", "pytesseract", "dummy"
    OCR_ENGINE_TYPE = os.environ.get("OCR_ENGINE_TYPE", "auto")
    # UIED-inspired CV region detection before OCR assignment (extra CPU vs whole-image OCR only).
    # When false, RegionOCREngine still returns one full-frame region per image (storage shape unchanged).
    ENABLE_UIED = os.environ.get("ENABLE_UIED", "true").lower() == "true"
    
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
    VLM_API_MODEL = os.environ.get("VLM_API_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

    # ============================================
    # Runtime Parameters
    # ============================================
    CAPTURE_INTERVAL_SECONDS = int(os.environ.get("CAPTURE_INTERVAL_SECONDS", "3"))
    
    # ============================================
    # Activity Clustering
    # ============================================
    # Enable activity clustering and timeline generation (requires extra compute + VLM for labeling)
    ENABLE_CLUSTERING = os.environ.get("ENABLE_CLUSTERING", "false").lower() == "true"
    CLUSTER_SIMILARITY_THRESHOLD = float(os.environ.get("CLUSTER_SIMILARITY_THRESHOLD", "0.82"))
    CLUSTER_STRONG_ASSIGNMENT_THRESHOLD = float(os.environ.get("CLUSTER_STRONG_ASSIGNMENT_THRESHOLD", "0.90"))
    CLUSTER_CANDIDATE_SIMILARITY_THRESHOLD = float(os.environ.get("CLUSTER_CANDIDATE_SIMILARITY_THRESHOLD", "0.78"))
    # Deprecated: recalculation is now only triggered by per-app unclassified counts.
    # Keep this env var for backward compatibility with existing .env files.
    CLUSTER_RECALC_INTERVAL_HOURS = float(os.environ.get("CLUSTER_RECALC_INTERVAL_HOURS", "24"))
    CLUSTER_UNCLASSIFIED_TRIGGER = int(os.environ.get("CLUSTER_UNCLASSIFIED_TRIGGER", "50"))
    CLUSTER_CANDIDATE_PROMOTION_COUNT = int(os.environ.get("CLUSTER_CANDIDATE_PROMOTION_COUNT", "3"))
    CLUSTER_CENTROID_EMA_ALPHA = float(os.environ.get("CLUSTER_CENTROID_EMA_ALPHA", "0.05"))
    CLUSTER_FREEZE_WINDOW_MINUTES = int(os.environ.get("CLUSTER_FREEZE_WINDOW_MINUTES", "30"))
    # Pending pool: similarity threshold to match an in-flight VLM leader
    CLUSTER_PENDING_SIMILARITY_THRESHOLD = float(os.environ.get("CLUSTER_PENDING_SIMILARITY_THRESHOLD", "0.78"))
    # Pending pool: TTL in seconds for pending entries before auto-expiry
    CLUSTER_PENDING_TTL_SECONDS = int(os.environ.get("CLUSTER_PENDING_TTL_SECONDS", "1800"))
    # VLM base for cluster labeling (OpenAI-compatible). Example: https://models.sjtu.edu.cn/api
    # Request URL is {CLUSTER_VLM_URL}/v1/chat/completions
    CLUSTER_VLM_URL = os.environ.get("CLUSTER_VLM_URL", "")
    # Optional API key for cluster labeling endpoint. Falls back to VLM_API_KEY when empty.
    CLUSTER_VLM_API_KEY = os.environ.get("CLUSTER_VLM_API_KEY", "").strip()
    # Whether cluster labeling should send image to VLM (token expensive).
    # false => prefer text-only prompt using OCR + optional region layout.
    CLUSTER_VLM_USE_VISION = os.environ.get("CLUSTER_VLM_USE_VISION", "true").lower() == "true"
    # Default model id for cluster labeling (vision + text if TEXT not set). Empty => Qwen/Qwen3.5-9B.
    CLUSTER_VLM_MODEL = os.environ.get("CLUSTER_VLM_MODEL", "").strip()
    # Optional text-only model id override. Empty => CLUSTER_VLM_MODEL or default.
    CLUSTER_VLM_TEXT_MODEL = os.environ.get("CLUSTER_VLM_TEXT_MODEL", "").strip()
    # Max OCR chars included in cluster-labeling prompt.
    CLUSTER_VLM_TEXT_OCR_MAX_CHARS = int(os.environ.get("CLUSTER_VLM_TEXT_OCR_MAX_CHARS", "2000"))
    # Include OCR region layout (bbox + text) in text-only prompt.
    CLUSTER_VLM_INCLUDE_REGION_LAYOUT = os.environ.get("CLUSTER_VLM_INCLUDE_REGION_LAYOUT", "true").lower() == "true"

    # ============================================
    # Report Generation (Map-Reduce daily report pipeline)
    # Defaults match SJTU OpenAI-compatible gateway; set REPORT_API_KEY in .env.
    # Model ids must be litellm "provider/model" (e.g. openai/qwen3coder, openai/minimax-m2.5).
    # Bare names are normalized in core/report/llm_caller.py. Native MiniMax: minimax/MiniMax-M2.5.
    # Daily report HTTP fetcher: same host as VisualMem data API (gui_backend_server).
    # Auto: GUI_MODE=remote + GUI_REMOTE_BACKEND_URL -> that URL; else http://localhost:18080.
    # Set REPORT_DATA_API_BASE only to override (e.g. CLI on another host).
    # ============================================
    REPORT_MAP_MODEL = os.environ.get("REPORT_MAP_MODEL", "openai/minimax-m2.5")
    REPORT_REDUCE_MODEL = os.environ.get("REPORT_REDUCE_MODEL", "openai/minimax-m2.5")
    REPORT_API_KEY = os.environ.get("REPORT_API_KEY", "")
    REPORT_API_BASE = os.environ.get(
        "REPORT_API_BASE",
        "https://models.sjtu.edu.cn/api/v1",
    )
    REPORT_CHUNK_TOKEN_LIMIT = int(os.environ.get("REPORT_CHUNK_TOKEN_LIMIT", "30000"))
    REPORT_GAP_MINUTES = int(os.environ.get("REPORT_GAP_MINUTES", "10"))
    REPORT_DAILY_GOAL = os.environ.get("REPORT_DAILY_GOAL", "")
    _report_data_env = os.environ.get("REPORT_DATA_API_BASE", "").strip()
    if _report_data_env:
        REPORT_DATA_API_BASE = _report_data_env.rstrip("/")
    elif GUI_MODE == "remote" and GUI_REMOTE_BACKEND_URL:
        REPORT_DATA_API_BASE = GUI_REMOTE_BACKEND_URL.rstrip("/")
    else:
        REPORT_DATA_API_BASE = "http://localhost:18080"
    # Reduce: read prior daily_report_*.json from this directory (project-relative ok).
    REPORT_LOG_DIR = _resolve_path(os.environ.get("REPORT_LOG_DIR", "logs"))
    # How many calendar days to walk backward when looking for prior report files.
    REPORT_HISTORY_DAYS = int(os.environ.get("REPORT_HISTORY_DAYS", "14"))
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
        IMAGE_STORAGE_PATH = _resolve_path(os.environ.get(
            "IMAGE_STORAGE_PATH",
            os.path.join(BENCHMARK_IMAGE_ROOT, BENCHMARK_NAME),
        ))
        LANCEDB_PATH = _resolve_path(os.environ.get(
            "LANCEDB_PATH",
            os.path.join(_benchmark_dir, "lancedb"),
        ))
        OCR_DB_PATH = _resolve_path(os.environ.get(
            "OCR_DB_PATH",
            os.path.join(_benchmark_dir, "ocr.db"),
        ))
        ACTIVITY_DB_PATH = _resolve_path(os.environ.get(
            "ACTIVITY_DB_PATH",
            os.path.join(_benchmark_dir, "activity.db"),
        ))
        TEXT_LANCEDB_PATH = _resolve_path(os.environ.get(
            "TEXT_LANCEDB_PATH",
            os.path.join(_benchmark_dir, "textdb"),
        ))

# Export a singleton instance
config = Config()
