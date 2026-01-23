# core/preprocess/frame_diff.py
"""
Frame Difference Detection Module

Implements independent frame difference detection for:
1. Full screen images
2. Individual application windows

Uses a combination of histogram comparison and SSIM for accuracy.
Reference: screenpipe's utils.rs compare_images_histogram and compare_images_ssim
"""
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from PIL import Image
from utils.logger import setup_logger
from utils.data_models import ScreenObject, WindowFrame

logger = setup_logger(__name__)

# Try to import skimage for SSIM, fall back to simpler methods if not available
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    logger.warning("scikit-image not available, using simplified SSIM calculation")


@dataclass
class FrameDiffResult:
    """Result of frame difference detection"""
    should_store: bool  # Whether the frame should be stored
    diff_score: float   # The difference score (0.0 = identical, 1.0 = completely different)
    histogram_diff: float
    ssim_diff: float
    reason: str = ""


@dataclass
class ScreenDiffState:
    """State for tracking previous screen captures per monitor"""
    previous_image: Optional[Image.Image] = None
    previous_hash: int = 0
    frame_count: int = 0
    # Track max average frame for selecting best frame in a sequence
    max_average_frame: Optional[Image.Image] = None
    max_average_value: float = 0.0
    max_average_frame_number: int = 0


@dataclass
class WindowDiffState:
    """State for tracking previous window captures"""
    previous_image: Optional[Image.Image] = None
    previous_hash: int = 0
    frame_count: int = 0
    app_name: str = ""
    window_name: str = ""


def calculate_histogram(image: Image.Image, bins: int = 256) -> np.ndarray:
    """
    Calculate grayscale histogram of an image
    
    Args:
        image: PIL Image
        bins: Number of histogram bins
        
    Returns:
        Normalized histogram array
    """
    # Convert to grayscale
    gray = image.convert('L')
    # Calculate histogram
    hist = np.array(gray.histogram())
    # Normalize
    hist = hist.astype(float) / hist.sum()
    return hist


def compare_histograms(hist1: np.ndarray, hist2: np.ndarray, metric: str = "hellinger") -> float:
    """
    Compare two histograms using the specified metric
    
    Args:
        hist1, hist2: Histogram arrays
        metric: Comparison metric ("hellinger", "correlation", "chi_square")
        
    Returns:
        Difference score (0.0 = identical, higher = more different)
    """
    if metric == "hellinger":
        # Hellinger distance (similar to screenpipe)
        # Range: [0, 1] where 0 = identical, 1 = completely different
        return np.sqrt(0.5 * np.sum((np.sqrt(hist1) - np.sqrt(hist2)) ** 2))
    
    elif metric == "correlation":
        # Correlation coefficient
        # Range: [-1, 1] where 1 = identical
        # We return (1 - corr) / 2 to normalize to [0, 1]
        mean1, mean2 = np.mean(hist1), np.mean(hist2)
        std1, std2 = np.std(hist1), np.std(hist2)
        if std1 == 0 or std2 == 0:
            return 1.0
        corr = np.sum((hist1 - mean1) * (hist2 - mean2)) / (len(hist1) * std1 * std2)
        return (1.0 - corr) / 2.0
    
    elif metric == "chi_square":
        # Chi-square distance
        epsilon = 1e-10
        chi_sq = np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + epsilon))
        # Normalize to roughly [0, 1]
        return min(chi_sq / 2.0, 1.0)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def calculate_ssim(image1: Image.Image, image2: Image.Image) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images
    
    Args:
        image1, image2: PIL Images
        
    Returns:
        SSIM score in range [0, 1] where 1 = identical
    """
    # Convert to grayscale numpy arrays
    gray1 = np.array(image1.convert('L'), dtype=np.float64)
    gray2 = np.array(image2.convert('L'), dtype=np.float64)
    
    # Resize to same dimensions if needed
    if gray1.shape != gray2.shape:
        # Resize both to smaller dimensions
        min_h = min(gray1.shape[0], gray2.shape[0])
        min_w = min(gray1.shape[1], gray2.shape[1])
        gray1 = np.array(image1.convert('L').resize((min_w, min_h)), dtype=np.float64)
        gray2 = np.array(image2.convert('L').resize((min_w, min_h)), dtype=np.float64)
    
    if HAS_SKIMAGE:
        # Use skimage's implementation
        try:
            # Determine appropriate win_size based on image dimensions
            min_dim = min(gray1.shape)
            win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
            if win_size < 3:
                win_size = 3
            
            score = ssim(gray1, gray2, data_range=255.0, win_size=win_size)
            return float(score)
        except Exception as e:
            logger.debug(f"SSIM calculation failed, using fallback: {e}")
    
    # Fallback: Simple SSIM approximation
    return _simple_ssim(gray1, gray2)


def _simple_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Simplified SSIM calculation without skimage dependency
    
    Uses the basic SSIM formula with default constants.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    if denominator == 0:
        return 1.0
    
    return numerator / denominator


class FrameDiffDetector:
    """
    Detects significant changes between consecutive frames
    
    Implements independent frame difference detection for:
    - Full screen images (per monitor)
    - Individual application windows
    
    Uses a combination of:
    - Hash comparison (quick reject for identical images)
    - Histogram comparison (Hellinger distance)
    - SSIM comparison (structural similarity)
    
    The average of histogram diff and SSIM diff is used as the final score.
    """
    
    def __init__(
        self,
        screen_threshold: float = 0.006,
        window_threshold: float = 0.006,
        use_max_average: bool = True,
        max_average_window: int = 10
    ):
        """
        Args:
            screen_threshold: Minimum difference to consider screen changed
            window_threshold: Minimum difference to consider window changed
            use_max_average: If True, track max average frame in a sequence
            max_average_window: Number of frames to track for max average
        """
        self.screen_threshold = screen_threshold
        self.window_threshold = window_threshold
        self.use_max_average = use_max_average
        self.max_average_window = max_average_window
        
        # State tracking
        self.screen_states: Dict[int, ScreenDiffState] = {}  # monitor_id -> state
        self.window_states: Dict[str, WindowDiffState] = {}  # window_key -> state
        
        logger.info(
            f"FrameDiffDetector initialized "
            f"(screen_threshold={screen_threshold}, window_threshold={window_threshold})"
        )
    
    def _get_window_key(self, app_name: str, window_name: str, process_id: int) -> str:
        """Generate a unique key for a window"""
        return f"{app_name}::{window_name}::{process_id}"
    
    def _compare_images(
        self,
        current: Image.Image,
        previous: Image.Image,
        current_hash: int,
        previous_hash: int
    ) -> Tuple[float, float, float]:
        """
        Compare two images and return difference scores
        
        Returns:
            Tuple of (combined_diff, histogram_diff, ssim_diff)
        """
        # Quick check: if hashes match, images are likely identical
        if current_hash == previous_hash:
            return 0.0, 0.0, 0.0
        
        # Calculate histogram difference
        hist1 = calculate_histogram(current)
        hist2 = calculate_histogram(previous)
        histogram_diff = compare_histograms(hist1, hist2, "hellinger")
        
        # Calculate SSIM difference (1 - SSIM gives difference)
        ssim_score = calculate_ssim(current, previous)
        ssim_diff = 1.0 - ssim_score
        
        # Combined score (average of both metrics)
        combined_diff = (histogram_diff + ssim_diff) / 2.0
        
        return combined_diff, histogram_diff, ssim_diff
    
    def check_screen_diff(
        self,
        screen_obj: ScreenObject
    ) -> FrameDiffResult:
        """
        Check if the full screen image has changed significantly
        
        Args:
            screen_obj: ScreenObject containing full screen capture
            
        Returns:
            FrameDiffResult indicating whether to store this frame
        """
        monitor_id = screen_obj.monitor_id
        current_image = screen_obj.full_screen_image
        current_hash = screen_obj.full_screen_hash
        
        # Get or create state for this monitor
        if monitor_id not in self.screen_states:
            self.screen_states[monitor_id] = ScreenDiffState()
        
        state = self.screen_states[monitor_id]
        state.frame_count += 1
        
        # First frame is always stored
        if state.previous_image is None:
            state.previous_image = current_image.copy()
            state.previous_hash = current_hash
            return FrameDiffResult(
                should_store=True,
                diff_score=1.0,
                histogram_diff=1.0,
                ssim_diff=1.0,
                reason="First frame"
            )
        
        # Compare with previous frame
        combined_diff, histogram_diff, ssim_diff = self._compare_images(
            current_image,
            state.previous_image,
            current_hash,
            state.previous_hash
        )
        
        logger.debug(
            f"Screen diff (monitor {monitor_id}): "
            f"combined={combined_diff:.4f}, "
            f"histogram={histogram_diff:.4f}, "
            f"ssim={ssim_diff:.4f}"
        )
        
        # Check threshold
        should_store = combined_diff >= self.screen_threshold
        
        if should_store:
            # Update state with current frame
            state.previous_image = current_image.copy()
            state.previous_hash = current_hash
            
            # Reset max average tracking
            if self.use_max_average:
                state.max_average_frame = None
                state.max_average_value = 0.0
                state.max_average_frame_number = 0
            
            return FrameDiffResult(
                should_store=True,
                diff_score=combined_diff,
                histogram_diff=histogram_diff,
                ssim_diff=ssim_diff,
                reason=f"Changed (diff={combined_diff:.4f} >= threshold={self.screen_threshold})"
            )
        else:
            # Track max average frame if enabled
            if self.use_max_average and combined_diff > state.max_average_value:
                state.max_average_frame = current_image.copy()
                state.max_average_value = combined_diff
                state.max_average_frame_number = state.frame_count
            
            return FrameDiffResult(
                should_store=False,
                diff_score=combined_diff,
                histogram_diff=histogram_diff,
                ssim_diff=ssim_diff,
                reason=f"Unchanged (diff={combined_diff:.4f} < threshold={self.screen_threshold})"
            )
    
    def check_window_diff(
        self,
        window: WindowFrame
    ) -> FrameDiffResult:
        """
        Check if a window image has changed significantly
        
        Args:
            window: WindowFrame containing window capture
            
        Returns:
            FrameDiffResult indicating whether to store this window
        """
        window_key = self._get_window_key(
            window.app_name,
            window.window_name,
            window.process_id
        )
        
        current_image = window.image
        current_hash = window.image_hash
        
        # Get or create state for this window
        if window_key not in self.window_states:
            self.window_states[window_key] = WindowDiffState(
                app_name=window.app_name,
                window_name=window.window_name
            )
        
        state = self.window_states[window_key]
        state.frame_count += 1
        
        # First frame for this window is always stored
        if state.previous_image is None:
            state.previous_image = current_image.copy()
            state.previous_hash = current_hash
            return FrameDiffResult(
                should_store=True,
                diff_score=1.0,
                histogram_diff=1.0,
                ssim_diff=1.0,
                reason="First frame for this window"
            )
        
        # Compare with previous frame
        combined_diff, histogram_diff, ssim_diff = self._compare_images(
            current_image,
            state.previous_image,
            current_hash,
            state.previous_hash
        )
        
        logger.debug(
            f"Window diff ({window.app_name}/{window.window_name}): "
            f"combined={combined_diff:.4f}"
        )
        
        # Check threshold
        should_store = combined_diff >= self.window_threshold
        
        if should_store:
            # Update state
            state.previous_image = current_image.copy()
            state.previous_hash = current_hash
            
            return FrameDiffResult(
                should_store=True,
                diff_score=combined_diff,
                histogram_diff=histogram_diff,
                ssim_diff=ssim_diff,
                reason=f"Changed (diff={combined_diff:.4f} >= threshold={self.window_threshold})"
            )
        else:
            return FrameDiffResult(
                should_store=False,
                diff_score=combined_diff,
                histogram_diff=histogram_diff,
                ssim_diff=ssim_diff,
                reason=f"Unchanged (diff={combined_diff:.4f} < threshold={self.window_threshold})"
            )
    
    def process_screen_object(
        self,
        screen_obj: ScreenObject
    ) -> Tuple[FrameDiffResult, List[Tuple[WindowFrame, FrameDiffResult]]]:
        """
        Process a complete ScreenObject and check diffs for screen and all windows
        
        Args:
            screen_obj: ScreenObject to process
            
        Returns:
            Tuple of:
            - FrameDiffResult for full screen
            - List of (WindowFrame, FrameDiffResult) for each window
        """
        # Check screen diff
        screen_result = self.check_screen_diff(screen_obj)
        
        # Check each window diff independently
        window_results: List[Tuple[WindowFrame, FrameDiffResult]] = []
        for window in screen_obj.windows:
            window_result = self.check_window_diff(window)
            window_results.append((window, window_result))
        
        return screen_result, window_results
    
    def get_changed_windows(
        self,
        screen_obj: ScreenObject
    ) -> List[WindowFrame]:
        """
        Get list of windows that have changed significantly
        
        Args:
            screen_obj: ScreenObject to process
            
        Returns:
            List of WindowFrame objects that should be stored
        """
        changed_windows = []
        for window in screen_obj.windows:
            result = self.check_window_diff(window)
            if result.should_store:
                changed_windows.append(window)
        
        return changed_windows
    
    def reset_state(self, monitor_id: Optional[int] = None):
        """
        Reset tracking state
        
        Args:
            monitor_id: If provided, only reset state for this monitor.
                       If None, reset all state.
        """
        if monitor_id is not None:
            if monitor_id in self.screen_states:
                del self.screen_states[monitor_id]
            # Also remove window states associated with this monitor
            # (Note: window states don't track monitor_id, so we can't filter them)
        else:
            self.screen_states.clear()
            self.window_states.clear()
        
        logger.info(f"Reset frame diff state (monitor_id={monitor_id})")
    
    def cleanup_stale_windows(self, active_window_keys: List[str]):
        """
        Remove tracking state for windows that are no longer active
        
        Args:
            active_window_keys: List of window keys that are currently active
        """
        stale_keys = set(self.window_states.keys()) - set(active_window_keys)
        for key in stale_keys:
            del self.window_states[key]
        
        if stale_keys:
            logger.debug(f"Cleaned up {len(stale_keys)} stale window states")
