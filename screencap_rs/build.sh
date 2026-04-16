#!/bin/bash
# Build script for screencap_rs

set -e

cd "$(dirname "$0")"

echo "=== Building screencap_rs ==="

# Add cargo to PATH if installed but not in PATH
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# Also try direct path
if [ -d "$HOME/.cargo/bin" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Check for Rust
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust not installed. Install from https://rustup.rs/"
    exit 1
fi

echo "Found cargo: $(which cargo)"

# Check for maturin
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin
fi

# Install platform-specific dependencies
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "=== Linux detected ==="
    echo "Installing X11 dependencies..."
    if command -v apt &> /dev/null; then
        sudo apt install -y libxcb1-dev libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y libxcb-devel
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm libxcb
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "=== macOS detected ==="
    # No additional dependencies needed
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "=== Windows detected ==="
    # No additional dependencies needed
fi

# Build
echo "=== Building with maturin ==="
maturin develop --release

echo "=== Build complete! ==="
echo ""
echo "Test with:"
echo "  python -c \"import screencap_rs; print(screencap_rs.get_platform())\""
