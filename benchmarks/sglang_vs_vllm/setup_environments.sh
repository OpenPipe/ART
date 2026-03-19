#!/usr/bin/env bash
# =============================================================================
# Setup script for SGLang + Megatron vs vLLM + Megatron benchmark
#
# Creates separate Python environments for SGLang (to avoid conflicts with
# vLLM which is already installed in the ART environment).
#
# Prerequisites:
#   - CUDA 12.x installed
#   - Python 3.10+ available
#   - nvidia-smi working
#   - uv package manager installed
#
# Usage:
#   bash benchmarks/sglang_vs_vllm/setup_environments.sh
#
# After setup:
#   python benchmarks/sglang_vs_vllm/run_benchmark.py
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# =============================================================================
# 1. Validate prerequisites
# =============================================================================

info "Checking prerequisites..."

# Check CUDA
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. CUDA drivers required."
fi
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
info "NVIDIA driver version: $CUDA_VERSION"

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
info "GPUs detected: $GPU_COUNT"

# Check Python
if ! command -v python3 &>/dev/null; then
    error "python3 not found"
fi
PYTHON_VERSION=$(python3 --version 2>&1)
info "Python: $PYTHON_VERSION"

# Check uv
if ! command -v uv &>/dev/null; then
    warn "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
info "uv: $(uv --version)"

success "Prerequisites OK"

# =============================================================================
# 2. Verify vLLM environment (existing ART environment)
# =============================================================================

info ""
info "=== Checking vLLM (ART) environment ==="

cd "$PROJECT_ROOT"

# Check if vLLM is installed in the current environment
if python3 -c "import vllm; print(f'vLLM {vllm.__version__}')" 2>/dev/null; then
    VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)")
    success "vLLM $VLLM_VERSION is already installed in ART environment"
else
    info "vLLM not found. Installing with ART backend extras..."
    uv pip install vllm
    success "vLLM installed"
fi

# Check if Megatron deps are available
if python3 -c "import megatron.bridge" 2>/dev/null; then
    success "Megatron bridge is installed"
else
    info "Megatron bridge not found (will be installed on first training run)"
    info "  You can install manually: bash src/art/megatron/setup.sh"
fi

# =============================================================================
# 3. Create SGLang environment
# =============================================================================

info ""
info "=== Setting up SGLang environment ==="

SGLANG_ENV="$HOME/.venvs/sglang-bench"

if [ -d "$SGLANG_ENV" ] && "$SGLANG_ENV/bin/python" -c "import sglang" 2>/dev/null; then
    SGLANG_VERSION=$("$SGLANG_ENV/bin/python" -c "import sglang; print(sglang.__version__)" 2>/dev/null || echo "unknown")
    success "SGLang environment already exists at $SGLANG_ENV (version: $SGLANG_VERSION)"
    info "  To recreate: rm -rf $SGLANG_ENV && bash $0"
else
    info "Creating SGLang virtual environment at $SGLANG_ENV..."
    mkdir -p "$(dirname "$SGLANG_ENV")"

    # Create venv
    uv venv "$SGLANG_ENV" --python python3

    info "Installing SGLang and dependencies..."

    # uv venv doesn't include pip — use "uv pip install --python <path>"
    # Install PyTorch first (matching CUDA version)
    uv pip install --python "$SGLANG_ENV/bin/python" \
        torch torchvision --index-url https://download.pytorch.org/whl/cu124

    # Install SGLang with all extras (server, router, all backends)
    # Ref: https://docs.sglang.ai/start/install.html
    uv pip install --python "$SGLANG_ENV/bin/python" \
        "sglang[all]>=0.4.6.post1"

    # Install additional dependencies for benchmark + bench_serving.py
    uv pip install --python "$SGLANG_ENV/bin/python" \
        aiohttp openai numpy tqdm datasets

    # Verify installation
    if "$SGLANG_ENV/bin/python" -c "import sglang; print(f'SGLang {sglang.__version__}')" 2>/dev/null; then
        SGLANG_VERSION=$("$SGLANG_ENV/bin/python" -c "import sglang; print(sglang.__version__)")
        success "SGLang $SGLANG_VERSION installed successfully"
    else
        error "SGLang installation failed. Check logs above."
    fi
fi

# =============================================================================
# 4. Install benchmark dependencies in ART environment
# =============================================================================

info ""
info "=== Installing benchmark dependencies in ART environment ==="

cd "$PROJECT_ROOT"
uv pip install aiohttp 2>/dev/null || pip install aiohttp

success "Benchmark dependencies installed"

# =============================================================================
# 5. Verify everything works
# =============================================================================

info ""
info "=== Verification ==="

# vLLM check — use uv run to pick up the ART .venv
VLLM_PYTHON=$(cd "$PROJECT_ROOT" && uv run python -c "import sys; print(sys.executable)" 2>/dev/null || echo "python3")
info "vLLM Python: $VLLM_PYTHON"
cd "$PROJECT_ROOT" && uv run python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU count: {torch.cuda.device_count()}')
import vllm
print(f'  vLLM: {vllm.__version__}')
"

# SGLang check
SGLANG_PYTHON="$SGLANG_ENV/bin/python"
info "SGLang Python: $SGLANG_PYTHON"
"$SGLANG_PYTHON" -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU count: {torch.cuda.device_count()}')
import sglang
print(f'  SGLang: {sglang.__version__}')
"

# =============================================================================
# 6. Print usage instructions
# =============================================================================

info ""
success "=== Setup Complete ==="
echo ""
echo "Environment paths:"
echo "  vLLM Python:   $VLLM_PYTHON"
echo "  SGLang Python: $SGLANG_PYTHON"
echo ""
echo "Run the benchmark (use 'uv run python' so the ART .venv is used):"
echo ""
echo "  uv run python benchmarks/sglang_vs_vllm/run_benchmark.py \\"
echo "    --sglang-python $SGLANG_PYTHON \\"
echo "    --model Qwen/Qwen2.5-7B-Instruct \\"
echo "    --dataset gsm8k \\"
echo "    --num-steps 3 \\"
echo "    --num-rollouts 16 \\"
echo "    --concurrency 8 \\"
echo "    --tp 2"
echo ""
