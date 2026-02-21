#!/bin/bash
# Setup script for SGLang + Unsloth two-environment architecture
#
# Creates TWO COMPLETELY ISOLATED virtual environments:
# - .venv: Main training env (ART + unsloth + openai>=2.14)
# - .venv-sglang-server: SGLang server ONLY (sglang + openai==2.6.1)
#
# They communicate via HTTP (localhost:8000), NOT Python imports.
# This avoids ALL dependency conflicts (torchao, openai, etc.)
#
# Usage:
#   chmod +x scripts/setup_sglang.sh
#   ./scripts/setup_sglang.sh
#
# Then activate the main env to run training:
#   source .venv/bin/activate
#   python your_training_script.py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "SGLang + Unsloth Two-Environment Setup"
echo "=========================================="
echo ""
echo "This will create TWO ISOLATED environments:"
echo "  1. .venv               - Main: ART + Unsloth (openai>=2.14, torchao>=0.13)"
echo "  2. .venv-sglang-server - Server: SGLang ONLY (openai==2.6.1, torchao==0.9)"
echo ""
echo "They communicate via HTTP only. No shared dependencies."
echo ""

# Check for python3.11
PYTHON_CMD=""
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 11 ]; then
        PYTHON_CMD="python3"
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3.11+ required."
    echo ""
    echo "Install with:"
    echo "  apt update && apt install -y software-properties-common"
    echo "  add-apt-repository -y ppa:deadsnakes/ppa"
    echo "  apt update && apt install -y python3.11 python3.11-venv python3.11-dev"
    exit 1
fi

echo "Using: $PYTHON_CMD ($($PYTHON_CMD --version))"

echo ""
echo "Step 1/4: Creating main training environment (.venv)..."
echo "--------------------------------------------------------"
if [ -d ".venv" ]; then
    echo "  .venv already exists, removing..."
    rm -rf .venv
fi
$PYTHON_CMD -m venv .venv
echo "  Created .venv"

echo ""
echo "Step 2/4: Installing ART + training dependencies..."
echo "----------------------------------------------------"
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -e ".[sglang]"
deactivate
echo "  Main environment ready (ART + Unsloth)"

echo ""
echo "Step 3/4: Creating SGLang server environment (.venv-sglang-server)..."
echo "----------------------------------------------------------------------"
if [ -d ".venv-sglang-server" ]; then
    echo "  .venv-sglang-server already exists, removing..."
    rm -rf .venv-sglang-server
fi
$PYTHON_CMD -m venv .venv-sglang-server
echo "  Created .venv-sglang-server"

echo ""
echo "Step 4/4: Installing SGLang server (ISOLATED - no ART)..."
echo "----------------------------------------------------------"
source .venv-sglang-server/bin/activate
pip install --upgrade pip wheel
# Install ONLY sglang - nothing else! No ART, no shared deps.
pip install "sglang[srt]>=0.5.5"
deactivate
echo "  SGLang server environment ready (sglang ONLY)"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Architecture:"
echo "  .venv (main)           <--HTTP-->  .venv-sglang-server"
echo "  - ART + Unsloth                    - sglang[srt] ONLY"
echo "  - openai>=2.14                     - openai==2.6.1"
echo "  - torchao>=0.13                    - torchao==0.9"
echo ""
echo "Usage:"
echo ""
echo "  # Activate main training environment"
echo "  source .venv/bin/activate"
echo ""
echo "  # Run your script (SGLang server auto-detected)"
echo "  python your_script.py"
echo ""
echo "The SGLang backend automatically finds .venv-sglang-server/bin/python"
echo "and uses it to spawn the inference server subprocess."
echo ""
