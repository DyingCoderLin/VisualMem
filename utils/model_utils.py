# utils/model_utils.py
import os
from utils.logger import setup_logger

logger = setup_logger("model_utils")

def is_model_cached(model_id: str) -> bool:
    """
    Check if a Hugging Face model is cached locally.
    Uses huggingface_hub to scan the cache.
    """
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_id:
                # Check if there are any revisions with files
                if repo.revisions:
                    return True
        return False
    except ImportError:
        logger.warning("huggingface_hub not installed, cannot check cache accurately.")
        return False
    except Exception as e:
        logger.warning(f"Failed to check cache for {model_id}: {e}")
        return False

def ensure_model_downloaded(model_id: str, model_name_label: str = "Model"):
    """
    Check if model exists, if not, print a clear message and download it.
    """
    if is_model_cached(model_id):
        logger.info(f"{model_name_label} ({model_id}) is already cached.")
        return

    # Use print instead of logger to ensure it's captured by Electron's stdout/stderr
    # and stands out in the terminal
    print(f"\n{'='*60}")
    print(f"{model_name_label} NOT FOUND: {model_id}")
    print("Starting download... This may take several minutes depending on your internet speed.")
    print(f"{'='*60}\n", flush=True)
    
    try:
        from huggingface_hub import snapshot_download
        # This will show progress bars to stdout/stderr
        snapshot_download(repo_id=model_id)
        print(f"\n{model_name_label} download complete!\n", flush=True)
    except Exception as e:
        print(f"\nFailed to download {model_id}: {e}\n", flush=True)
        # We don't raise here, let the actual model loading fail later with its own error
