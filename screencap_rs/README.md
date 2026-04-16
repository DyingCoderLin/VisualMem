# screencap_rs

Cross-platform screen and window capture library with Python bindings.

## Supported Platforms

- **macOS** - Uses CoreGraphics APIs via xcap
- **Windows** - Uses Windows Graphics Capture API via xcap
- **Linux** - Uses X11/XCB via xcap

## Features

- Enumerate all visible windows
- Capture individual window screenshots
- Capture full screen screenshots
- Filter out system UI windows automatically
- Returns images as PNG bytes (can be loaded with PIL)

## Building

### Prerequisites

1. Install Rust: https://rustup.rs/
2. Install maturin: `pip install maturin`

### Linux Dependencies

```bash
sudo apt install libxcb1-dev libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev
```

### Build

```bash
# Development build (in-place)
cd screencap_rs
maturin develop --release

# Or build wheel
maturin build --release
```

## Usage

```python
import screencap_rs
from PIL import Image
from io import BytesIO

# Get platform
print(screencap_rs.get_platform())  # "linux", "macos", or "windows"

# List monitors
monitors = screencap_rs.get_monitors()
for m in monitors:
    print(f"Monitor {m.id}: {m.name} ({m.width}x{m.height})")

# List windows
windows = screencap_rs.get_windows(include_minimized=False, filter_system=True)
for w in windows:
    print(f"Window {w.id}: {w.app_name} - {w.title}")

# Capture full screen
screen = screencap_rs.capture_screen()
if screen:
    image = Image.open(BytesIO(screen.get_image_bytes()))
    image.save("screenshot.png")

# Capture a specific window
captured = screencap_rs.capture_window(window_id=12345)
if captured:
    image = Image.open(BytesIO(captured.get_image_bytes()))
    image.save("window.png")

# Capture all windows
windows = screencap_rs.capture_all_windows()
for w in windows:
    image = Image.open(BytesIO(w.get_image_bytes()))
    image.save(f"{w.info.app_name}_{w.info.id}.png")

# Capture screen + all windows at once
screen, windows = screencap_rs.capture_screen_with_windows()
```

## API Reference

### Functions

- `get_platform() -> str` - Returns "macos", "windows", or "linux"
- `get_monitors() -> List[MonitorInfo]` - List all monitors
- `get_windows(include_minimized=False, filter_system=True) -> List[WindowInfo]` - List windows
- `capture_window(window_id: int) -> Optional[CapturedWindow]` - Capture specific window
- `capture_all_windows(include_minimized=False, filter_system=True) -> List[CapturedWindow]` - Capture all windows
- `capture_screen(monitor_id=None) -> Optional[CapturedScreen]` - Capture full screen
- `capture_screen_with_windows(monitor_id=None, include_minimized=False, filter_system=True)` - Capture both

### Classes

- `MonitorInfo` - id, name, x, y, width, height, is_primary
- `WindowInfo` - id, app_name, title, process_id, x, y, width, height, is_minimized
- `CapturedWindow` - info: WindowInfo, get_image_bytes() -> bytes
- `CapturedScreen` - monitor: MonitorInfo, get_image_bytes() -> bytes
