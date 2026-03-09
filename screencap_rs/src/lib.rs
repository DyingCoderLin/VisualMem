//! Cross-platform screen and window capture module with Python bindings
//!
//! Supports: macOS, Windows, Linux
//! Reference: screenpipe's capture_screenshot_by_window.rs

use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashSet;
use std::io::Cursor;
use xcap::{Monitor, Window};

// ============================================================================
// Platform-specific skip lists
// ============================================================================

// Common keywords to skip across all platforms (user configurable defaults)
static SKIP_KEYWORDS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from([
        "bit", "vpn", "trash", "private", "incognito", "wallpaper",
        "settings", "keepass", "recorder", "vaults", "obs studio",
        "screenpipe", "visualmem",
    ])
});

// apps that should be sk
#[cfg(target_os = "macos")]
static SKIP_APPS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from([
        // System apps (English)
        "Window Server",
        "SystemUIServer",
        "ControlCenter",
        "Dock",
        "NotificationCenter",
        "loginwindow",
        "WindowManager",
        "Contexts",
        "Screenshot",
        "Spotlight",
        // System apps (Chinese/localized)
        "控制中心",  // Control Center
        "程序坞",    // Dock
        "聚焦",      // Spotlight
        "系统设置",  // System Settings
        // User configurable defaults
        "DeepL",
    ])
});

#[cfg(target_os = "windows")]
static SKIP_APPS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from([
        // System apps
        "Windows Shell Experience Host",
        "Microsoft Text Input Application",
        "Windows Explorer",
        "Program Manager",
        "Microsoft Store",
        "Search",
        "TaskBar",
        // User configurable defaults
        "Nvidia",
    ])
});

#[cfg(target_os = "linux")]
static SKIP_APPS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from([
        // System apps
        "Gnome-shell",
        "gnome-shell",
        "Plasma",
        "plasma",
        "Xfdesktop",
        "Polybar",
        "i3bar",
        "Plank",
        "Dock",
    ])
});

#[cfg(target_os = "macos")]
static SKIP_TITLES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from([
        // System window titles (English)
        "Item-0",
        "App Icon Window",
        "Dock",
        "NowPlaying",
        "FocusModes",
        "Shortcuts",
        "AudioVideoModule",
        "Clock",
        "WiFi",
        "Battery",
        "BentoBox",
        "Menu Bar",
        "Menubar",  // Alternative naming
        "Notification Center",
        "Control Center",
        "Spotlight",
        "Mission Control",
        "Desktop",
        "Screen Sharing",
        "Touch Bar",
        "Status Bar",
        "Menu Extra",
        "System Settings",
        // System window titles (Chinese/localized)
        "程序坞",       // Dock
        "控制中心",     // Control Center
        "通知中心",     // Notification Center
        "调度中心",     // Mission Control
        "聚焦",         // Spotlight
        "电池",         // Battery
        "无线局域网",   // WiFi
        "蓝牙",         // Bluetooth
        "系统设置",     // System Settings
        "辅助功能",     // Accessibility
        "屏幕共享",     // Screen Sharing
        "菜单栏",       // Menu Bar
        "桌面",         // Desktop
        // Empty title
        "",
    ])
});

#[cfg(target_os = "windows")]
static SKIP_TITLES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from([
        // System window titles
        "Program Manager",
        "Windows Input Experience",
        "Microsoft Text Input Application",
        "Task View",
        "Start",
        "System Tray",
        "Notification Area",
        "Action Center",
        "Task Bar",
        "Desktop",
        // User configurable defaults
        "Control Panel",
        "System Properties",
        // Empty title
        "",
    ])
});

#[cfg(target_os = "linux")]
static SKIP_TITLES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from([
        // System window titles
        "Desktop",
        "Panel",
        "Top Bar",
        "Status Bar",
        "Dock",
        "Dashboard",
        "Activities",
        "System Tray",
        "Notification Area",
        // User configurable defaults
        "Info center",
        "Discover",
        "Parted",
        // Empty title
        "",
    ])
});

// ============================================================================
// Python-exposed data structures
// ============================================================================

/// Information about a window
#[pyclass]
#[derive(Clone, Debug)]
pub struct WindowInfo {
    #[pyo3(get)]
    pub id: u32,
    #[pyo3(get)]
    pub app_name: String,
    #[pyo3(get)]
    pub title: String,
    #[pyo3(get)]
    pub x: i32,
    #[pyo3(get)]
    pub y: i32,
    #[pyo3(get)]
    pub width: u32,
    #[pyo3(get)]
    pub height: u32,
    #[pyo3(get)]
    pub is_minimized: bool,
}

#[pymethods]
impl WindowInfo {
    fn __repr__(&self) -> String {
        format!(
            "WindowInfo(id={}, app='{}', title='{}', size={}x{})",
            self.id, self.app_name, self.title, self.width, self.height
        )
    }
}

/// Information about a monitor
#[pyclass]
#[derive(Clone, Debug)]
pub struct MonitorInfo {
    #[pyo3(get)]
    pub id: u32,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub x: i32,
    #[pyo3(get)]
    pub y: i32,
    #[pyo3(get)]
    pub width: u32,
    #[pyo3(get)]
    pub height: u32,
    #[pyo3(get)]
    pub is_primary: bool,
}

#[pymethods]
impl MonitorInfo {
    fn __repr__(&self) -> String {
        format!(
            "MonitorInfo(id={}, name='{}', size={}x{}, primary={})",
            self.id, self.name, self.width, self.height, self.is_primary
        )
    }
}

/// Captured window with image data
#[pyclass]
#[derive(Clone)]
pub struct CapturedWindow {
    #[pyo3(get)]
    pub info: WindowInfo,
    image_data: Vec<u8>,
}

#[pymethods]
impl CapturedWindow {
    /// Get image as PNG bytes
    fn get_image_bytes<'py>(&self, py: Python<'py>) -> &'py PyBytes {
        PyBytes::new(py, &self.image_data)
    }

    #[getter]
    fn image_width(&self) -> u32 {
        self.info.width
    }

    #[getter]
    fn image_height(&self) -> u32 {
        self.info.height
    }

    fn __repr__(&self) -> String {
        format!(
            "CapturedWindow(app='{}', title='{}', bytes={})",
            self.info.app_name, self.info.title, self.image_data.len()
        )
    }
}

/// Captured screen with image data
#[pyclass]
#[derive(Clone)]
pub struct CapturedScreen {
    #[pyo3(get)]
    pub monitor: MonitorInfo,
    image_data: Vec<u8>,
}

#[pymethods]
impl CapturedScreen {
    /// Get image as PNG bytes
    fn get_image_bytes<'py>(&self, py: Python<'py>) -> &'py PyBytes {
        PyBytes::new(py, &self.image_data)
    }

    fn __repr__(&self) -> String {
        format!(
            "CapturedScreen(monitor='{}', bytes={})",
            self.monitor.name, self.image_data.len()
        )
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn should_skip_window(app_name: &str, title: &str) -> bool {
    let app_lower = app_name.to_lowercase();
    let title_lower = title.to_lowercase();

    // Skip "Unknown" app names
    if app_lower == "unknown" || app_name == "Unknown" {
        return true;
    }

    // Check platform-specific app names (contains match)
    for skip_app in SKIP_APPS.iter() {
        if app_lower.contains(&skip_app.to_lowercase()) {
            return true;
        }
    }

    // Check platform-specific window titles (contains match for better coverage)
    // This handles cases like "BentoBox-0", "BentoBox-1", etc.
    for skip_title in SKIP_TITLES.iter() {
        let skip_lower = skip_title.to_lowercase();
        if !skip_lower.is_empty() {
            // Use contains match for non-empty titles
            if title_lower.contains(&skip_lower) {
                return true;
            }
        } else {
            // Exact match for empty titles
            if title_lower.is_empty() {
                return true;
            }
        }
    }

    // Check common keywords (contains match in both app name and title)
    for keyword in SKIP_KEYWORDS.iter() {
        let keyword_lower = keyword.to_lowercase();
        if app_lower.contains(&keyword_lower) || title_lower.contains(&keyword_lower) {
            return true;
        }
    }

    false
}

fn window_to_info(window: &Window) -> Result<WindowInfo, xcap::XCapError> {
    Ok(WindowInfo {
        id: window.id()?,
        app_name: window.app_name()?,
        title: window.title()?,
        x: window.x()?,
        y: window.y()?,
        width: window.width()?,
        height: window.height()?,
        is_minimized: window.is_minimized()?,
    })
}

fn monitor_to_info(monitor: &Monitor, index: u32) -> Result<MonitorInfo, xcap::XCapError> {
    Ok(MonitorInfo {
        id: monitor.id()?,
        name: monitor.name()?,
        x: monitor.x()?,
        y: monitor.y()?,
        width: monitor.width()?,
        height: monitor.height()?,
        is_primary: index == 0,
    })
}

fn image_to_png_bytes(image: &image::RgbaImage) -> Result<Vec<u8>, String> {
    let mut buffer = Cursor::new(Vec::new());
    let dynamic_image = image::DynamicImage::ImageRgba8(image.clone());
    dynamic_image
        .write_to(&mut buffer, image::ImageFormat::Png)
        .map_err(|e| format!("Failed to encode image: {}", e))?;
    Ok(buffer.into_inner())
}

// ============================================================================
// Python-exposed functions
// ============================================================================

/// Get platform name
#[pyfunction]
fn get_platform() -> &'static str {
    #[cfg(target_os = "macos")]
    return "macos";
    #[cfg(target_os = "windows")]
    return "windows";
    #[cfg(target_os = "linux")]
    return "linux";
    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    return "unknown";
}

/// Get list of all available monitors
#[pyfunction]
fn get_monitors() -> PyResult<Vec<MonitorInfo>> {
    let monitors = Monitor::all().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to get monitors: {}", e))
    })?;

    let mut result = Vec::new();
    for (i, m) in monitors.iter().enumerate() {
        if let Ok(info) = monitor_to_info(m, i as u32) {
            result.push(info);
        }
    }

    Ok(result)
}

/// Get list of all visible windows
#[pyfunction]
#[pyo3(signature = (include_minimized=false, filter_system=true))]
fn get_windows(include_minimized: bool, filter_system: bool) -> PyResult<Vec<WindowInfo>> {
    let windows = Window::all().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to get windows: {}", e))
    })?;

    let mut result = Vec::new();
    for window in windows {
        // Skip windows that fail to provide properties (e.g. just closed)
        let is_minimized = match window.is_minimized() {
            Ok(b) => b,
            Err(_) => continue,
        };

        if !include_minimized && is_minimized {
            continue;
        }

        let app_name = match window.app_name() {
            Ok(s) => s,
            Err(_) => continue,
        };
        let title = match window.title() {
            Ok(s) => s,
            Err(_) => continue,
        };

        if filter_system && should_skip_window(&app_name, &title) {
            continue;
        }

        let width = match window.width() {
            Ok(w) => w,
            Err(_) => continue,
        };
        let height = match window.height() {
            Ok(h) => h,
            Err(_) => continue,
        };

        if width == 0 || height == 0 {
            continue;
        }

        if let Ok(info) = window_to_info(&window) {
            result.push(info);
        }
    }

    Ok(result)
}

/// Capture a specific window by ID
#[pyfunction]
fn capture_window(window_id: u32) -> PyResult<Option<CapturedWindow>> {
    let windows = Window::all().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to get windows: {}", e))
    })?;

    for window in windows {
        let current_id = match window.id() {
            Ok(id) => id,
            Err(_) => continue,
        };

        if current_id == window_id {
            let image = window.capture_image().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to capture: {}", e))
            })?;

            let image_data = image_to_png_bytes(&image).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(e)
            })?;

            if let Ok(info) = window_to_info(&window) {
                return Ok(Some(CapturedWindow {
                    info,
                    image_data,
                }));
            }
        }
    }

    Ok(None)
}

/// Capture all visible windows
#[pyfunction]
#[pyo3(signature = (include_minimized=false, filter_system=true))]
fn capture_all_windows(include_minimized: bool, filter_system: bool) -> PyResult<Vec<CapturedWindow>> {
    let windows = Window::all().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to get windows: {}", e))
    })?;

    let mut result = Vec::new();
    for window in windows {
        let is_minimized = match window.is_minimized() {
            Ok(b) => b,
            Err(_) => continue,
        };

        if !include_minimized && is_minimized {
            continue;
        }

        let app_name = match window.app_name() {
            Ok(s) => s,
            Err(_) => continue,
        };
        let title = match window.title() {
            Ok(s) => s,
            Err(_) => continue,
        };

        if filter_system && should_skip_window(&app_name, &title) {
            continue;
        }

        let width = match window.width() {
            Ok(w) => w,
            Err(_) => continue,
        };
        let height = match window.height() {
            Ok(h) => h,
            Err(_) => continue,
        };

        if width == 0 || height == 0 {
            continue;
        }

        if let Ok(image) = window.capture_image() {
            if let Ok(image_data) = image_to_png_bytes(&image) {
                if let Ok(info) = window_to_info(&window) {
                    result.push(CapturedWindow {
                        info,
                        image_data,
                    });
                }
            }
        }
    }

    Ok(result)
}

/// Capture a specific monitor (full screen)
#[pyfunction]
#[pyo3(signature = (monitor_id=None))]
fn capture_screen(monitor_id: Option<u32>) -> PyResult<Option<CapturedScreen>> {
    let monitors = Monitor::all().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to get monitors: {}", e))
    })?;

    let found = if let Some(id) = monitor_id {
        monitors
            .iter()
            .enumerate()
            .find(|(_, m)| m.id().map(|mid| mid == id).unwrap_or(false))
    } else {
        monitors.first().map(|m| (0, m))
    };

    let (index, monitor) = match found {
        Some(pair) => pair,
        None => return Ok(None),
    };

    let image = monitor.capture_image().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to capture screen: {}", e))
    })?;

    let image_data = image_to_png_bytes(&image).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(e)
    })?;

    if let Ok(info) = monitor_to_info(monitor, index as u32) {
        return Ok(Some(CapturedScreen {
            monitor: info,
            image_data,
        }));
    }

    Ok(None)
}

/// Capture full screen and all visible windows in one call
#[pyfunction]
#[pyo3(signature = (monitor_id=None, include_minimized=false, filter_system=true))]
fn capture_screen_with_windows(
    monitor_id: Option<u32>,
    include_minimized: bool,
    filter_system: bool,
) -> PyResult<(Option<CapturedScreen>, Vec<CapturedWindow>)> {
    let screen = capture_screen(monitor_id)?;
    let windows = capture_all_windows(include_minimized, filter_system)?;
    Ok((screen, windows))
}

// ============================================================================
// Python module definition (PyO3 0.20 API)
// ============================================================================

#[pymodule]
fn screencap_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<WindowInfo>()?;
    m.add_class::<MonitorInfo>()?;
    m.add_class::<CapturedWindow>()?;
    m.add_class::<CapturedScreen>()?;

    m.add_function(wrap_pyfunction!(get_platform, m)?)?;
    m.add_function(wrap_pyfunction!(get_monitors, m)?)?;
    m.add_function(wrap_pyfunction!(get_windows, m)?)?;
    m.add_function(wrap_pyfunction!(capture_window, m)?)?;
    m.add_function(wrap_pyfunction!(capture_all_windows, m)?)?;
    m.add_function(wrap_pyfunction!(capture_screen, m)?)?;
    m.add_function(wrap_pyfunction!(capture_screen_with_windows, m)?)?;

    Ok(())
}
