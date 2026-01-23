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

#[cfg(target_os = "macos")]
static SKIP_APPS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from([
        "Window Server", "SystemUIServer", "ControlCenter", "Dock",
        "NotificationCenter", "loginwindow", "WindowManager", "Spotlight",
    ])
});

#[cfg(target_os = "windows")]
static SKIP_APPS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from([
        "Windows Shell Experience Host", "Microsoft Text Input Application",
        "Windows Explorer", "Program Manager", "TaskBar",
    ])
});

#[cfg(target_os = "linux")]
static SKIP_APPS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from([
        "Gnome-shell", "gnome-shell", "Plasma", "plasma",
        "Polybar", "i3bar", "Plank", "Dock", "Panel",
    ])
});

#[cfg(target_os = "macos")]
static SKIP_TITLES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from([
        "Menu Bar", "Notification Center", "Control Center",
        "Spotlight", "Mission Control", "Desktop", "",
    ])
});

#[cfg(target_os = "windows")]
static SKIP_TITLES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from([
        "Program Manager", "Task View", "Start",
        "System Tray", "Task Bar", "Desktop", "",
    ])
});

#[cfg(target_os = "linux")]
static SKIP_TITLES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from([
        "Desktop", "Panel", "Top Bar", "Status Bar",
        "Dock", "Activities", "System Tray", "",
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

    for skip_app in SKIP_APPS.iter() {
        if app_lower.contains(&skip_app.to_lowercase()) {
            return true;
        }
    }

    for skip_title in SKIP_TITLES.iter() {
        if skip_title.to_lowercase() == title_lower {
            return true;
        }
    }

    false
}

fn window_to_info(window: &Window) -> WindowInfo {
    WindowInfo {
        id: window.id(),
        app_name: window.app_name().to_string(),
        title: window.title().to_string(),
        x: window.x(),
        y: window.y(),
        width: window.width(),
        height: window.height(),
        is_minimized: window.is_minimized(),
    }
}

fn monitor_to_info(monitor: &Monitor, index: u32) -> MonitorInfo {
    MonitorInfo {
        id: monitor.id(),
        name: monitor.name().to_string(),
        x: monitor.x(),
        y: monitor.y(),
        width: monitor.width(),
        height: monitor.height(),
        is_primary: index == 0,
    }
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

    Ok(monitors
        .iter()
        .enumerate()
        .map(|(i, m)| monitor_to_info(m, i as u32))
        .collect())
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
        if !include_minimized && window.is_minimized() {
            continue;
        }

        if filter_system && should_skip_window(window.app_name(), window.title()) {
            continue;
        }

        if window.width() == 0 || window.height() == 0 {
            continue;
        }

        result.push(window_to_info(&window));
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
        if window.id() == window_id {
            let image = window.capture_image().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to capture: {}", e))
            })?;

            let image_data = image_to_png_bytes(&image).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(e)
            })?;

            return Ok(Some(CapturedWindow {
                info: window_to_info(&window),
                image_data,
            }));
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
        if !include_minimized && window.is_minimized() {
            continue;
        }

        if filter_system && should_skip_window(window.app_name(), window.title()) {
            continue;
        }

        if window.width() == 0 || window.height() == 0 {
            continue;
        }

        if let Ok(image) = window.capture_image() {
            if let Ok(image_data) = image_to_png_bytes(&image) {
                result.push(CapturedWindow {
                    info: window_to_info(&window),
                    image_data,
                });
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

    let (index, monitor) = if let Some(id) = monitor_id {
        monitors
            .iter()
            .enumerate()
            .find(|(_, m)| m.id() == id)
            .map(|(i, m)| (i, m))
    } else {
        monitors.first().map(|m| (0, m))
    }
    .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No monitor found"))?;

    let image = monitor.capture_image().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to capture screen: {}", e))
    })?;

    let image_data = image_to_png_bytes(&image).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(e)
    })?;

    Ok(Some(CapturedScreen {
        monitor: monitor_to_info(monitor, index as u32),
        image_data,
    }))
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
