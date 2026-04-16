#!/bin/bash
# macOS Setup Script for VisualMem
set -e

echo "=========================================="
echo "  VisualMem macOS Setup"
echo "=========================================="

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Check macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: This script is for macOS only"
    exit 1
fi

# Check Homebrew
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install system dependencies
echo ""
echo "=== Installing system dependencies ==="
brew install ffmpeg tesseract || true

# Check/Install Rust
echo ""
echo "=== Checking Rust ==="
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

if ! command -v cargo &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
echo "Rust version: $(rustc --version)"

# Check Python
echo ""
echo "=== Checking Python ==="
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found. Install with: brew install python3"
    exit 1
fi
echo "Python version: $(python3 --version)"

# Install maturin
echo ""
echo "=== Installing maturin ==="
pip install maturin

# Build Rust module
echo ""
echo "=== Building screencap_rs ==="
cd "$PROJECT_ROOT/screencap_rs"
maturin develop --release

# Verify
echo ""
echo "=== Verifying installation ==="
python3 -c "
import screencap_rs
print('Platform:', screencap_rs.get_platform())
print('Monitors:', len(screencap_rs.get_monitors()))
print('screencap_rs OK!')
"

# Install Python dependencies
echo ""
echo "=== Installing Python dependencies ==="
cd "$PROJECT_ROOT"
pip install -r requirements_macos.txt

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: Grant screen recording permission!"
echo "  System Preferences → Privacy & Security → Screen Recording"
echo "  Add your terminal app (Terminal/iTerm2/VS Code)"
echo ""
echo "Test with:"
echo "  python test_capture.py --capture"
echo ""
