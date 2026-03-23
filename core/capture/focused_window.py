# core/capture/focused_window.py
"""
Cross-platform focused/active window detection.

Detects which application window currently has user focus.
- macOS: AppKit/Quartz (fast) with osascript fallback
- Linux: xdotool + xprop
- Windows: Win32 API
"""
import platform
from typing import Tuple

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


def _get_focused_window_macos() -> Tuple[str, str]:
    """macOS: use AppKit + Quartz (pyobjc), fall back to osascript."""
    try:
        from AppKit import NSWorkspace  # type: ignore[import-untyped]
        import Quartz  # type: ignore[import-untyped]

        active_app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if not active_app:
            return "", ""

        app_name = str(active_app.localizedName() or "")
        pid = active_app.processIdentifier()

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
