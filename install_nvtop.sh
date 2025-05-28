#!/bin/bash

# nvtop Installation Script
# This script downloads and installs nvtop from GitHub releases
# Designed for environments without snap or where nvtop isn't in repositories

set -e  # Exit on any error

echo "🚀 Installing nvtop..."

# Configuration
NVTOP_VERSION="3.1.0"
DOWNLOAD_URL="https://github.com/Syllo/nvtop/releases/download/${NVTOP_VERSION}/nvtop-x86_64.AppImage"
TEMP_DIR="/tmp/nvtop_install"
BINARY_PATH="/usr/local/bin/nvtop"
LIB_PATH="/usr/local/lib"

# Check if running as root for system installation
if [[ $EUID -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
    echo "📝 Note: This script requires sudo privileges for system installation"
fi

# Create temporary directory
echo "📁 Creating temporary directory..."
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

# Download nvtop AppImage
echo "⬇️  Downloading nvtop v${NVTOP_VERSION}..."
wget "$DOWNLOAD_URL" -O nvtop.AppImage

# Make it executable
chmod +x nvtop.AppImage

# Extract AppImage contents (for environments without FUSE)
echo "📦 Extracting AppImage..."
./nvtop.AppImage --appimage-extract

# Install binary
echo "💾 Installing nvtop binary..."
$SUDO cp squashfs-root/usr/bin/nvtop "$BINARY_PATH"

# Install required libraries
echo "📚 Installing required libraries..."
$SUDO mkdir -p "$LIB_PATH"
$SUDO cp squashfs-root/usr/lib/* "$LIB_PATH/" 2>/dev/null || true

# Update library cache
$SUDO ldconfig 2>/dev/null || true

# Clean up
echo "🧹 Cleaning up..."
cd /
rm -rf "$TEMP_DIR"

# Test installation
echo "🔍 Testing installation..."
if nvtop --version > /dev/null 2>&1; then
    echo "✅ nvtop successfully installed!"
    echo "📊 Run 'nvtop' to start GPU monitoring"
    nvtop --version
else
    echo "❌ Installation failed - nvtop not working properly"
    exit 1
fi

echo ""
echo "🎉 Installation complete!"
echo "Usage: nvtop"
echo "Press 'q' to quit nvtop, 'h' for help" 