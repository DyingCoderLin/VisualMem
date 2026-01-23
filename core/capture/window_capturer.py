# core/capture/window_capturer.py
"""
Window-level screen capture module

Captures both full screen and individual application windows.
Uses Rust bindings (screencap_rs) for cross-platform support (macOS, Windows, Linux).
Falls back to pure Python implementation on Linux if Rust module not available.

Reference: screenpipe's capture_screenshot_by_window.rs
"""
import datetime
import hashlib
from typing import Optional, List, Set
from PIL import Image
from io import BytesIO
from dataclasses import dataclass
from .base_capturer import AbstractCapturer
from utils.data_models import ScreenFrame, WindowFrame, ScreenObject
from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__)

# Try to import Rust module
_USE_RUST = False
try:
    import screencap_rs
    _USE_RUST = True
    logger.info(f"Using Rust screencap_rs module (platform: {screencap_rs.get_platform()})")
except ImportError:
    logger.warning("screencap_rs not available, using pure Python fallback (Linux only)")


def calculate_image_hash(image: Image.Image) -> int:
    """Calculate a hash of the image for quick comparison"""
    # Resize to small size for faster hashing
    small = image.resize((64, 64), Image.Resampling.LANCZOS)
    # Convert to bytes and hash
    img_bytes = small.tobytes()
    return int(hashlib.md5(img_bytes).hexdigest()[:16], 16)


def should_skip_window(app_name: str, title: str) -> bool:
    """
    Check if a window should be skipped based on app name or title
    
    Note: When using Rust module, filtering is done in Rust.
    This function is for the pure Python fallback.
    """
    # System windows to skip
    SKIP_APPS: Set[str] = {
        # Linux desktop environments
        "gnome-shell", "plasma", "xfdesktop", "polybar", "i3bar",
        "plank", "dock", "panel", "desktop", "activities",
        "top bar", "status bar", "notification area", "system tray",
        "xdg-desktop-portal", "gsd-", "ibus-", "fcitx",
        # macOS
        "window server", "systemuiserver", "controlcenter", "dock",
        "notificationcenter", "loginwindow", "windowmanager", "spotlight",
        # Windows
        "windows shell experience host", "microsoft text input application",
        "windows explorer", "program manager", "taskbar",
    }
    
    SKIP_TITLES: Set[str] = {
        "desktop", "panel", "top bar", "status bar", "dock",
        "dashboard", "activities", "system tray", "notification area",
        "menu bar", "control center", "mission control", "",
    }
    
    app_lower = app_name.lower()
    title_lower = title.lower()
    
    for skip_app in SKIP_APPS:
        if skip_app in app_lower:
            return True
    
    for skip_title in SKIP_TITLES:
        if skip_title == title_lower:
            return True
    
    return False


# ============================================================================
# Rust-based implementation (cross-platform)
# ============================================================================

class RustWindowCapturer(AbstractCapturer):
    """
    Cross-platform window capturer using Rust bindings
    
    Supports macOS, Windows, and Linux via xcap library.
    """
    
    def __init__(
        self,
        monitor_id: int = 0,
        max_width: int = None,
        capture_windows: bool = True,
        capture_unfocused_windows: bool = True
    ):
        self.monitor_id = monitor_id
        self.max_width = max_width if max_width is not None else getattr(config, 'MAX_IMAGE_WIDTH', 0)
        self.capture_windows = capture_windows
        self.capture_unfocused_windows = capture_unfocused_windows
        
        # Get actual monitor ID from xcap
        monitors = screencap_rs.get_monitors()
        if monitors:
            if monitor_id < len(monitors):
                self._monitor_xcap_id = monitors[monitor_id].id
            else:
                self._monitor_xcap_id = monitors[0].id
        else:
            self._monitor_xcap_id = None
        
        logger.info(
            f"RustWindowCapturer initialized (monitor={monitor_id}, "
            f"platform={screencap_rs.get_platform()})"
        )
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image if needed"""
        if self.max_width <= 0 or image.width <= self.max_width:
            return image
        
        ratio = self.max_width / image.width
        new_height = int(image.height * ratio)
        return image.resize((self.max_width, new_height), Image.Resampling.LANCZOS)
    
    def _bytes_to_pil(self, png_bytes: bytes) -> Image.Image:
        """Convert PNG bytes to PIL Image"""
        image = Image.open(BytesIO(png_bytes))
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        return image
    
    def capture(self) -> Optional[ScreenFrame]:
        """Legacy interface for backward compatibility"""
        screen_obj = self.capture_screen_with_windows()
        if screen_obj is None:
            return None
        
        return ScreenFrame(
            timestamp=screen_obj.timestamp,
            image=screen_obj.full_screen_image,
            ocr_text=None
        )
    
    def capture_screen_with_windows(self) -> Optional[ScreenObject]:
        """
        Capture full screen and all visible application windows
        """
        try:
            timestamp = datetime.datetime.now(datetime.timezone.utc)
            
            # Capture screen and windows in one call
            screen_result, window_results = screencap_rs.capture_screen_with_windows(
                monitor_id=self._monitor_xcap_id,
                include_minimized=False,
                filter_system=True
            )
            
            # Process full screen
            if screen_result is None:
                logger.error("Failed to capture screen")
                return None
            
            full_screen = self._bytes_to_pil(screen_result.get_image_bytes())
            full_screen = self._resize_image(full_screen)
            full_screen_hash = calculate_image_hash(full_screen)
            
            # Process windows
            windows: List[WindowFrame] = []
            
            if self.capture_windows:
                for w in window_results:
                    # Skip unfocused if not capturing them
                    # Note: xcap doesn't provide is_focused, so we capture all
                    
                    try:
                        window_image = self._bytes_to_pil(w.get_image_bytes())
                        window_image = self._resize_image(window_image)
                        window_hash = calculate_image_hash(window_image)
                        
                        windows.append(WindowFrame(
                            app_name=w.info.app_name,
                            window_name=w.info.title,
                            process_id=0,  # xcap doesn't provide pid
                            is_focused=False,  # xcap doesn't provide this
                            image=window_image,
                            image_hash=window_hash,
                            timestamp=timestamp
                        ))
                    except Exception as e:
                        logger.debug(f"Failed to process window {w.info.app_name}: {e}")
                        continue
                
                logger.debug(f"Captured {len(windows)} windows")
            
            # Create ScreenObject
            screen_obj = ScreenObject(
                monitor_id=self.monitor_id,
                device_name=f"monitor_{self.monitor_id}",
                timestamp=timestamp,
                full_screen_image=full_screen,
                full_screen_hash=full_screen_hash,
                windows=windows
            )
            
            logger.debug(f"Screen capture complete: {full_screen.size}, {len(windows)} windows")
            return screen_obj
            
        except Exception as e:
            logger.error(f"Failed to capture screen with windows: {e}")
            return None
    
    def get_available_monitors(self) -> List[int]:
        """Get list of available monitor IDs"""
        try:
            monitors = screencap_rs.get_monitors()
            return list(range(len(monitors)))
        except Exception:
            return [0]


# ============================================================================
# Pure Python fallback (Linux only)
# ============================================================================

# Only import Linux-specific modules when needed
if not _USE_RUST:
    import subprocess
    
    @dataclass
    class WindowInfo:
        """Raw window information from the window manager"""
        window_id: int
        title: str
        app_name: str
        process_id: int
        x: int
        y: int
        width: int
        height: int
        is_focused: bool = False
        is_minimized: bool = False
    
    class LinuxWindowEnumerator:
        """Enumerate windows on Linux using wmctrl/xdotool"""
        
        def __init__(self):
            self._check_available_tools()
        
        def _check_available_tools(self):
            self.has_wmctrl = self._command_exists("wmctrl")
            self.has_xdotool = self._command_exists("xdotool")
            self.has_xprop = self._command_exists("xprop")
            
            if not (self.has_wmctrl or self.has_xdotool):
                logger.warning("Neither wmctrl nor xdotool found.")
                logger.info("Install with: sudo apt install wmctrl xdotool")
        
        def _command_exists(self, cmd: str) -> bool:
            try:
                subprocess.run(["which", cmd], capture_output=True, check=True)
                return True
            except subprocess.CalledProcessError:
                return False
        
        def get_focused_window_id(self) -> Optional[int]:
            if self.has_xdotool:
                try:
                    result = subprocess.run(
                        ["xdotool", "getactivewindow"],
                        capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0:
                        return int(result.stdout.strip())
                except Exception:
                    pass
            return None
        
        def get_all_windows(self) -> List[WindowInfo]:
            focused_id = self.get_focused_window_id()
            
            if self.has_wmctrl:
                return self._get_windows_wmctrl(focused_id)
            elif self.has_xdotool:
                return self._get_windows_xdotool(focused_id)
            return []
        
        def _get_windows_wmctrl(self, focused_id: Optional[int]) -> List[WindowInfo]:
            windows = []
            try:
                result = subprocess.run(
                    ["wmctrl", "-l", "-p", "-G"],
                    capture_output=True, text=True, timeout=5
                )
                
                if result.returncode != 0:
                    return windows
                
                for line in result.stdout.strip().split('\n'):
                    if not line:
                        continue
                    
                    parts = line.split(None, 8)
                    if len(parts) < 9:
                        continue
                    
                    try:
                        window_id = int(parts[0], 16)
                        pid = int(parts[2])
                        x, y = int(parts[3]), int(parts[4])
                        width, height = int(parts[5]), int(parts[6])
                        title = parts[8] if len(parts) > 8 else ""
                        
                        app_name = self._get_window_class(window_id) or "Unknown"
                        is_minimized = self._is_window_minimized(window_id)
                        
                        windows.append(WindowInfo(
                            window_id=window_id,
                            title=title,
                            app_name=app_name,
                            process_id=pid,
                            x=x, y=y,
                            width=width, height=height,
                            is_focused=(window_id == focused_id),
                            is_minimized=is_minimized
                        ))
                    except (ValueError, IndexError):
                        continue
                        
            except subprocess.TimeoutExpired:
                logger.warning("wmctrl timed out")
            except Exception as e:
                logger.error(f"Error getting windows: {e}")
            
            return windows
        
        def _get_windows_xdotool(self, focused_id: Optional[int]) -> List[WindowInfo]:
            windows = []
            try:
                result = subprocess.run(
                    ["xdotool", "search", "--onlyvisible", "--name", ""],
                    capture_output=True, text=True, timeout=5
                )
                
                if result.returncode != 0:
                    return windows
                
                for wid_str in result.stdout.strip().split('\n'):
                    if not wid_str:
                        continue
                    
                    try:
                        window_id = int(wid_str)
                        info = self._get_window_info_xdotool(window_id, focused_id)
                        if info:
                            windows.append(info)
                    except ValueError:
                        continue
                        
            except Exception as e:
                logger.error(f"Error getting windows: {e}")
            
            return windows
        
        def _get_window_info_xdotool(self, window_id: int, focused_id: Optional[int]) -> Optional[WindowInfo]:
            try:
                geom_result = subprocess.run(
                    ["xdotool", "getwindowgeometry", "--shell", str(window_id)],
                    capture_output=True, text=True, timeout=2
                )
                
                if geom_result.returncode != 0:
                    return None
                
                geom = {}
                for line in geom_result.stdout.strip().split('\n'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        geom[key] = value
                
                name_result = subprocess.run(
                    ["xdotool", "getwindowname", str(window_id)],
                    capture_output=True, text=True, timeout=2
                )
                title = name_result.stdout.strip() if name_result.returncode == 0 else ""
                
                pid_result = subprocess.run(
                    ["xdotool", "getwindowpid", str(window_id)],
                    capture_output=True, text=True, timeout=2
                )
                pid = int(pid_result.stdout.strip()) if pid_result.returncode == 0 else 0
                
                app_name = self._get_window_class(window_id) or "Unknown"
                
                return WindowInfo(
                    window_id=window_id,
                    title=title,
                    app_name=app_name,
                    process_id=pid,
                    x=int(geom.get('X', 0)),
                    y=int(geom.get('Y', 0)),
                    width=int(geom.get('WIDTH', 0)),
                    height=int(geom.get('HEIGHT', 0)),
                    is_focused=(window_id == focused_id),
                    is_minimized=False
                )
            except Exception:
                return None
        
        def _get_window_class(self, window_id: int) -> Optional[str]:
            if not self.has_xprop:
                return None
            
            try:
                result = subprocess.run(
                    ["xprop", "-id", str(window_id), "WM_CLASS"],
                    capture_output=True, text=True, timeout=2
                )
                
                if result.returncode == 0 and "WM_CLASS" in result.stdout:
                    parts = result.stdout.split('"')
                    if len(parts) >= 4:
                        return parts[3]
                    elif len(parts) >= 2:
                        return parts[1]
            except Exception:
                pass
            
            return None
        
        def _is_window_minimized(self, window_id: int) -> bool:
            if not self.has_xprop:
                return False
            
            try:
                result = subprocess.run(
                    ["xprop", "-id", str(window_id), "_NET_WM_STATE"],
                    capture_output=True, text=True, timeout=2
                )
                
                if result.returncode == 0:
                    return "_NET_WM_STATE_HIDDEN" in result.stdout
            except Exception:
                pass
            
            return False
    
    class WindowScreenshotCapture:
        """Capture screenshots of individual windows on Linux"""
        
        def __init__(self):
            self.has_import = self._command_exists("import")
        
        def _command_exists(self, cmd: str) -> bool:
            try:
                subprocess.run(["which", cmd], capture_output=True, check=True)
                return True
            except subprocess.CalledProcessError:
                return False
        
        def capture_window(self, window_id: int) -> Optional[Image.Image]:
            if self.has_import:
                return self._capture_with_import(window_id)
            return None
        
        def _capture_with_import(self, window_id: int) -> Optional[Image.Image]:
            try:
                import tempfile
                import os
                
                fd, temp_path = tempfile.mkstemp(suffix='.png')
                os.close(fd)
                
                try:
                    result = subprocess.run(
                        ["import", "-window", hex(window_id), temp_path],
                        capture_output=True, timeout=5
                    )
                    
                    if result.returncode == 0 and os.path.exists(temp_path):
                        image = Image.open(temp_path)
                        image.load()
                        return image.copy()
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
            except Exception as e:
                logger.error(f"Failed to capture window {window_id}: {e}")
            
            return None
        
        def capture_full_screen(self, monitor_id: int = 0) -> Optional[Image.Image]:
            try:
                from PIL import ImageGrab
                screenshot = ImageGrab.grab()
                
                if screenshot.mode in ('RGBA', 'LA', 'P'):
                    screenshot = screenshot.convert('RGB')
                
                return screenshot
            except Exception as e:
                logger.error(f"Failed to capture full screen: {e}")
                return None
    
    class PythonWindowCapturer(AbstractCapturer):
        """Pure Python window capturer (Linux only)"""
        
        def __init__(
            self,
            monitor_id: int = 0,
            max_width: int = None,
            capture_windows: bool = True,
            capture_unfocused_windows: bool = True
        ):
            self.monitor_id = monitor_id
            self.max_width = max_width if max_width is not None else getattr(config, 'MAX_IMAGE_WIDTH', 0)
            self.capture_windows = capture_windows
            self.capture_unfocused_windows = capture_unfocused_windows
            
            self.window_enumerator = LinuxWindowEnumerator()
            self.screenshot_capture = WindowScreenshotCapture()
            
            logger.info(f"PythonWindowCapturer initialized (Linux fallback)")
        
        def _resize_image(self, image: Image.Image) -> Image.Image:
            if self.max_width <= 0 or image.width <= self.max_width:
                return image
            
            ratio = self.max_width / image.width
            new_height = int(image.height * ratio)
            return image.resize((self.max_width, new_height), Image.Resampling.LANCZOS)
        
        def capture(self) -> Optional[ScreenFrame]:
            screen_obj = self.capture_screen_with_windows()
            if screen_obj is None:
                return None
            
            return ScreenFrame(
                timestamp=screen_obj.timestamp,
                image=screen_obj.full_screen_image,
                ocr_text=None
            )
        
        def capture_screen_with_windows(self) -> Optional[ScreenObject]:
            try:
                timestamp = datetime.datetime.now(datetime.timezone.utc)
                
                # Capture full screen
                full_screen = self.screenshot_capture.capture_full_screen(self.monitor_id)
                if full_screen is None:
                    logger.error("Failed to capture full screen")
                    return None
                
                full_screen = self._resize_image(full_screen)
                full_screen_hash = calculate_image_hash(full_screen)
                
                # Capture windows
                windows: List[WindowFrame] = []
                
                if self.capture_windows:
                    window_infos = self.window_enumerator.get_all_windows()
                    
                    for win_info in window_infos:
                        if should_skip_window(win_info.app_name, win_info.title):
                            continue
                        
                        if win_info.is_minimized:
                            continue
                        
                        if not self.capture_unfocused_windows and not win_info.is_focused:
                            continue
                        
                        if win_info.width <= 0 or win_info.height <= 0:
                            continue
                        
                        window_image = self.screenshot_capture.capture_window(win_info.window_id)
                        if window_image is None:
                            continue
                        
                        window_image = self._resize_image(window_image)
                        window_hash = calculate_image_hash(window_image)
                        
                        windows.append(WindowFrame(
                            app_name=win_info.app_name,
                            window_name=win_info.title,
                            process_id=win_info.process_id,
                            is_focused=win_info.is_focused,
                            image=window_image,
                            image_hash=window_hash,
                            timestamp=timestamp
                        ))
                    
                    logger.debug(f"Captured {len(windows)} windows")
                
                screen_obj = ScreenObject(
                    monitor_id=self.monitor_id,
                    device_name=f"monitor_{self.monitor_id}",
                    timestamp=timestamp,
                    full_screen_image=full_screen,
                    full_screen_hash=full_screen_hash,
                    windows=windows
                )
                
                return screen_obj
                
            except Exception as e:
                logger.error(f"Failed to capture screen with windows: {e}")
                return None
        
        def get_available_monitors(self) -> List[int]:
            return [0]


# ============================================================================
# Main exported class - automatically selects best implementation
# ============================================================================

if _USE_RUST:
    WindowCapturer = RustWindowCapturer
else:
    WindowCapturer = PythonWindowCapturer
