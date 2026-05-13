# core/capture/focused_window.py
"""
Cross-platform focused/active window detection.

Detects which application window currently has user focus.
- macOS: AppKit/Quartz (fast) with osascript fallback
- Linux: xdotool + xprop
- Windows: Win32 API
"""
import platform
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import setup_logger

logger = setup_logger(__name__)


def get_focused_window() -> Tuple[str, str]:
    """
    Get the currently focused window's app name and title.

    Returns:
        (app_name, window_title). Both empty strings on failure.
    """
    system = platform.system()
    try:
        if system == "Darwin":
            return _get_focused_window_macos()
        elif system == "Linux":
            return _get_focused_window_linux()
        elif system == "Windows":
            return _get_focused_window_windows()
    except Exception as e:
        logger.debug(f"Failed to get focused window: {e}")
    return "", ""


def get_fullscreen_window_for_monitor(
    monitor_bounds: Optional[Dict[str, Any]],
    min_monitor_overlap: float = 0.80,
) -> Tuple[str, str]:
    """
    Return the visible app/window that occupies a specific monitor.

    This is intentionally monitor-scoped. A global focused window is not enough
    on multi-display setups because the focused app may live on another screen.
    """
    if not monitor_bounds:
        return "", ""

    system = platform.system()
    try:
        if system == "Darwin":
            return _get_fullscreen_window_for_monitor_macos(
                monitor_bounds,
                min_monitor_overlap=min_monitor_overlap,
            )
        if system == "Windows":
            return _get_fullscreen_window_for_monitor_windows(
                monitor_bounds,
                min_monitor_overlap=min_monitor_overlap,
            )
    except Exception as e:
        logger.debug(f"Failed to get monitor fullscreen window: {e}")
    return "", ""


def _rect_from_mapping(bounds: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Normalize Electron/Quartz/xcap bound dictionaries."""
    try:
        x = bounds.get("x", bounds.get("X"))
        y = bounds.get("y", bounds.get("Y"))
        width = bounds.get("width", bounds.get("Width"))
        height = bounds.get("height", bounds.get("Height"))
        if x is None or y is None or width is None or height is None:
            return None
        width_f = float(width)
        height_f = float(height)
        if width_f <= 0 or height_f <= 0:
            return None
        return {
            "x": float(x),
            "y": float(y),
            "width": width_f,
            "height": height_f,
        }
    except Exception:
        return None


def _intersection_area(a: Dict[str, float], b: Dict[str, float]) -> float:
    left = max(a["x"], b["x"])
    top = max(a["y"], b["y"])
    right = min(a["x"] + a["width"], b["x"] + b["width"])
    bottom = min(a["y"] + a["height"], b["y"] + b["height"])
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def _is_system_window_owner(owner: str) -> bool:
    owner_lower = (owner or "").strip().lower()
    if not owner_lower:
        return True
    skip_owners = {
        "window server",
        "systemuiserver",
        "controlcenter",
        "notificationcenter",
        "dock",
        "loginwindow",
        "windowmanager",
        "spotlight",
    }
    return owner_lower in skip_owners


def _get_fullscreen_window_for_monitor_macos(
    monitor_bounds: Dict[str, Any],
    min_monitor_overlap: float,
) -> Tuple[str, str]:
    """macOS: use Quartz window geometry to identify a monitor-local fullscreen window."""
    import Quartz  # type: ignore[import-untyped]

    monitor_rect = _rect_from_mapping(monitor_bounds)
    if not monitor_rect:
        return "", ""

    monitor_area = monitor_rect["width"] * monitor_rect["height"]
    if monitor_area <= 0:
        return "", ""

    window_list = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly
        | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    )
    if not window_list:
        return "", ""

    candidates: List[Tuple[float, str, str]] = []
    for win in window_list:
        owner = str(win.get("kCGWindowOwnerName", "") or "")
        title = str(win.get("kCGWindowName", "") or "")
        if _is_system_window_owner(owner):
            continue

        # Layer 0 is the normal application-window layer. Higher layers include
        # menu bar, overlays, tooltips and system UI, which can otherwise cover
        # large portions of the display.
        try:
            layer = int(win.get("kCGWindowLayer", 0) or 0)
        except Exception:
            layer = 0
        if layer != 0:
            continue

        bounds = win.get("kCGWindowBounds") or {}
        window_rect = _rect_from_mapping(bounds)
        if not window_rect:
            continue

        overlap = _intersection_area(monitor_rect, window_rect)
        if overlap <= 0:
            continue

        monitor_overlap = overlap / monitor_area
        if monitor_overlap < min_monitor_overlap:
            continue

        candidates.append((monitor_overlap, owner, title or owner))

    if not candidates:
        return "", ""

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def _get_focused_window_macos() -> Tuple[str, str]:
    """macOS: use AppKit + Quartz (pyobjc), fall back to osascript."""
    try:
        from AppKit import NSWorkspace  # type: ignore[import-untyped]
        import Quartz  # type: ignore[import-untyped]

        # activeApplication() returns the real foreground app even when
        # called from a background process.  frontmostApplication() would
        # return the *calling* process's own app, which is wrong when the
        # recorder runs as a daemon / background service.
        active_info = NSWorkspace.sharedWorkspace().activeApplication()
        if not active_info:
            return "", ""

        app_name = str(active_info.get("NSApplicationName", "") or "")
        pid = active_info.get("NSApplicationProcessIdentifier", 0)

        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly
            | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID,
        )

        window_title = ""
        if window_list:
            for win in window_list:
                if win.get("kCGWindowOwnerPID", 0) == pid:
                    title = win.get("kCGWindowName", "")
                    if title:
                        window_title = str(title)
                        break

        return app_name, window_title
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Quartz/AppKit focused window detection failed: {e}")

    return _get_focused_window_macos_osascript()


def _get_focused_window_macos_osascript() -> Tuple[str, str]:
    """Fallback: use osascript to query System Events."""
    import subprocess

    app_name = ""
    window_title = ""

    try:
        result = subprocess.run(
            [
                "osascript", "-e",
                'tell application "System Events" to get name of first '
                'application process whose frontmost is true',
            ],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            app_name = result.stdout.strip()
    except Exception as e:
        logger.debug(f"osascript app name failed: {e}")

    try:
        result = subprocess.run(
            [
                "osascript", "-e",
                'tell application "System Events" to get name of front window '
                'of (first application process whose frontmost is true)',
            ],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            window_title = result.stdout.strip()
    except Exception as e:
        logger.debug(f"osascript window title failed: {e}")

    return app_name, window_title


def _get_focused_window_linux() -> Tuple[str, str]:
    """Linux: use xdotool + xprop."""
    import subprocess

    try:
        result = subprocess.run(
            ["xdotool", "getactivewindow"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode != 0:
            return "", ""

        window_id = result.stdout.strip()

        name_result = subprocess.run(
            ["xdotool", "getwindowname", window_id],
            capture_output=True, text=True, timeout=2,
        )
        window_title = name_result.stdout.strip() if name_result.returncode == 0 else ""

        app_name = ""
        try:
            class_result = subprocess.run(
                ["xprop", "-id", window_id, "WM_CLASS"],
                capture_output=True, text=True, timeout=2,
            )
            if class_result.returncode == 0 and "WM_CLASS" in class_result.stdout:
                parts = class_result.stdout.split('"')
                if len(parts) >= 4:
                    app_name = parts[3]
                elif len(parts) >= 2:
                    app_name = parts[1]
        except Exception:
            pass

        return app_name, window_title
    except Exception as e:
        logger.debug(f"Linux focused window detection failed: {e}")
        return "", ""


def _get_focused_window_windows() -> Tuple[str, str]:
    """Windows: use Win32 GetForegroundWindow."""
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]

        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return "", ""

        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        window_title = buf.value

        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

        app_name = ""
        try:
            import psutil
            proc = psutil.Process(pid.value)
            app_name = proc.name()
        except Exception:
            pass

        return app_name, window_title
    except Exception as e:
        logger.debug(f"Windows focused window detection failed: {e}")
        return "", ""


def _get_fullscreen_window_for_monitor_windows(
    monitor_bounds: Dict[str, Any],
    min_monitor_overlap: float,
) -> Tuple[str, str]:
    """Windows: enumerate visible top-level windows and match by monitor geometry."""
    import ctypes
    import os
    from ctypes import wintypes

    monitor_rect = _rect_from_mapping(monitor_bounds)
    if not monitor_rect:
        return "", ""

    monitor_area = monitor_rect["width"] * monitor_rect["height"]
    if monitor_area <= 0:
        return "", ""

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    GWL_EXSTYLE = -20
    WS_EX_TOOLWINDOW = 0x00000080
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    skip_classes = {
        "Progman",
        "WorkerW",
        "Shell_TrayWnd",
        "Shell_SecondaryTrayWnd",
        "NotifyIconOverflowWindow",
    }
    skip_processes = {
        "explorer.exe",
        "searchhost.exe",
        "startmenuexperiencehost.exe",
        "shellexperiencehost.exe",
        "textinputhost.exe",
        "widgets.exe",
    }

    def _window_text(hwnd: int) -> str:
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value

    def _class_name(hwnd: int) -> str:
        buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, buf, 256)
        return buf.value

    def _process_name(pid: int) -> str:
        try:
            import psutil  # type: ignore[import-untyped]

            return psutil.Process(pid).name()
        except Exception:
            pass

        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return ""
        try:
            size = wintypes.DWORD(32768)
            buf = ctypes.create_unicode_buffer(size.value)
            if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return os.path.basename(buf.value)
        except Exception:
            return ""
        finally:
            kernel32.CloseHandle(handle)
        return ""

    candidates: List[Tuple[float, int, str, str]] = []

    def _enum(hwnd: int, _lparam: int) -> bool:
        try:
            if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
                return True

            class_name = _class_name(hwnd)
            if class_name in skip_classes:
                return True

            ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            if ex_style & WS_EX_TOOLWINDOW:
                return True

            rect = wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return True

            window_rect = _rect_from_mapping(
                {
                    "x": int(rect.left),
                    "y": int(rect.top),
                    "width": int(rect.right - rect.left),
                    "height": int(rect.bottom - rect.top),
                }
            )
            if not window_rect:
                return True

            overlap = _intersection_area(monitor_rect, window_rect)
            if overlap <= 0:
                return True

            monitor_overlap = overlap / monitor_area
            if monitor_overlap < min_monitor_overlap:
                return True

            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            process_name = _process_name(int(pid.value))
            if process_name.lower() in skip_processes:
                return True

            title = _window_text(hwnd)
            app_name = process_name or class_name or title
            window_name = title or class_name or app_name
            if app_name:
                candidates.append((monitor_overlap, len(candidates), app_name, window_name))
        except Exception:
            return True
        return True

    user32.EnumWindows(EnumWindowsProc(_enum), 0)
    if not candidates:
        return "", ""

    # EnumWindows walks top-level windows in z-order, so keep that order as the
    # tiebreaker when multiple windows have the same monitor coverage.
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2], candidates[0][3]
