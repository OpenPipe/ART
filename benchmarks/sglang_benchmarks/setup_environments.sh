#!/usr/bin/env bash
# =============================================================================
# Setup script for Unsloth + SGLang benchmark
#
# Creates a Python environment for SGLang and ensures Unsloth is available.
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
# Helper: uv pip install that works with or without a venv
# =============================================================================

uv_pip_install() {
    if [ -n "${VIRTUAL_ENV:-}" ] || [ -d "$PROJECT_ROOT/.venv" ]; then
        uv pip install "$@"
    else
        uv pip install --system "$@"
    fi
}

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
# 2. Create SGLang environment
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

    uv pip install --python "$SGLANG_ENV/bin/python" \
        torch torchvision --index-url https://download.pytorch.org/whl/cu124

    # Install SGLang with all extras
    # Ref: https://docs.sglang.ai/start/install.html
    uv pip install --python "$SGLANG_ENV/bin/python" \
        "sglang[all]>=0.4.6.post1"

    # Install additional dependencies for benchmark
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
# 3. Verify Unsloth environment (for --backends unsloth)
# =============================================================================

info ""
info "=== Checking Unsloth (MoE training) environment ==="

cd "$PROJECT_ROOT"

# Step 1: Install Unsloth packages
if python3 -c "import importlib.metadata; print(importlib.metadata.version('unsloth'))" 2>/dev/null; then
    UNSLOTH_VERSION=$(python3 -c "import importlib.metadata; print(importlib.metadata.version('unsloth'))")
    success "Unsloth $UNSLOTH_VERSION package is installed"
else
    info "Unsloth not found. Installing..."
    uv_pip_install --upgrade unsloth unsloth_zoo
fi

# Step 2: Force-upgrade transformers and trl to versions required for MoE
info "Ensuring transformers>=5.0.0 and trl>=0.27.1 for MoE support..."
NEED_TF_UPGRADE=$(python3 -c "
import importlib.metadata as meta
v = meta.version('transformers')
print('yes' if tuple(int(x) for x in v.split('.')[:2]) < (5, 0) else 'no')
" 2>/dev/null || echo "yes")

NEED_TRL_UPGRADE=$(python3 -c "
import importlib.metadata as meta
v = meta.version('trl')
parts = v.split('.')
print('yes' if (int(parts[0]), int(parts[1])) < (0, 27) else 'no')
" 2>/dev/null || echo "yes")

if [ "$NEED_TF_UPGRADE" = "yes" ] || [ "$NEED_TRL_UPGRADE" = "yes" ]; then
    info "Upgrading: transformers>=5.0.0 trl>=0.27.1 ..."
    uv_pip_install --upgrade "transformers>=5.0.0" "trl>=0.27.1"
fi

# Unsloth 2026.2.x blocks datasets>=4.5.0
NEED_DS_DOWNGRADE=$(python3 -c "
import importlib.metadata as meta
v = meta.version('datasets')
parts = [int(x) for x in v.split('.')[:2]]
print('yes' if parts[0] > 4 or (parts[0] == 4 and parts[1] >= 5) else 'no')
" 2>/dev/null || echo "no")

if [ "$NEED_DS_DOWNGRADE" = "yes" ]; then
    info "Downgrading datasets to 4.3.0 (Unsloth requires <4.5.0)..."
    uv_pip_install "datasets==4.3.0"
fi

# torchvision must match PyTorch
TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "0.0.0")
TV_VER=$(python3 -c "import torchvision; print(torchvision.__version__.split('+')[0])" 2>/dev/null || echo "0.0.0")
info "torch=$TORCH_VER  torchvision=$TV_VER"

NEED_TV_UPGRADE=$(python3 -c "
import importlib.metadata as meta
t = meta.version('torch').split('+')[0]
tv = meta.version('torchvision').split('+')[0]
t_major, t_minor = int(t.split('.')[0]), int(t.split('.')[1])
tv_major, tv_minor = int(tv.split('.')[0]), int(tv.split('.')[1])
expected_tv_minor = t_minor + 12
if tv_major == 0 and tv_minor < expected_tv_minor:
    print('yes')
else:
    print('no')
" 2>/dev/null || echo "yes")

if [ "$NEED_TV_UPGRADE" = "yes" ]; then
    info "Upgrading torchvision to match PyTorch $TORCH_VER..."
    uv_pip_install --upgrade torchvision
fi

# Step 3: Verify Unsloth import
info "Verifying Unsloth import..."
if python3 -c "
import sys, types

vllm_ok = False
try:
    import vllm._C
    vllm_ok = True
except (ImportError, OSError, AttributeError):
    pass

if not vllm_ok:
    sys.modules['vllm._C'] = types.ModuleType('vllm._C')
    class _Stub(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith('__') and name.endswith('__'):
                raise AttributeError(name)
            return lambda *a, **kw: None
    sys.modules['unsloth_zoo.vllm_utils'] = _Stub('unsloth_zoo.vllm_utils')
    print('  (mocked vllm internals — ABI mismatch with PyTorch, using SGLang)')

import unsloth
print(f'Unsloth {unsloth.__version__}')
"; then
    success "Unsloth ready"
else
    warn "Unsloth import failed (see error above). Unsloth backend may not work."
    warn "  Try: pip install --upgrade unsloth unsloth_zoo transformers>=5.0.0 trl>=0.27.1"
fi

# Show final versions
python3 -c "
import importlib.metadata as meta
for pkg in ['unsloth', 'unsloth-zoo', 'transformers', 'trl', 'torch', 'triton']:
    try:
        print(f'  {pkg}: {meta.version(pkg)}')
    except meta.PackageNotFoundError:
        print(f'  {pkg}: not installed')
" 2>/dev/null || true

# Check MoE backend support
python3 -c "
import torch
print(f'  MoE backend auto-detection:')
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    if 'H100' in name or 'B200' in name:
        print(f'    GPU: {name} → grouped_mm (optimal)')
    elif 'A100' in name:
        print(f'    GPU: {name} → unsloth_triton (optimal)')
    else:
        print(f'    GPU: {name} → native_torch (fallback)')
else:
    print('    No CUDA GPU available')
" 2>/dev/null || true

# =============================================================================
# 4. Install benchmark dependencies
# =============================================================================

info ""
info "=== Installing benchmark dependencies ==="

cd "$PROJECT_ROOT"
uv_pip_install aiohttp 2>/dev/null || pip install aiohttp

success "Benchmark dependencies installed"

# =============================================================================
# 5. Verify everything works
# =============================================================================

info ""
info "=== Verification ==="

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
" 2>/dev/null || warn "SGLang verification failed (non-fatal)"

# =============================================================================
# 6. Print usage instructions
# =============================================================================

info ""
success "=== Setup Complete ==="
echo ""
echo "Environment paths:"
echo "  SGLang Python: $SGLANG_PYTHON"
echo ""
echo "Run the benchmark:"
echo ""
echo "  # Unsloth MoE + SGLang"
echo "  python benchmarks/sglang_vs_vllm/run_benchmark.py \\"
echo "    --sglang-python $SGLANG_PYTHON \\"
echo "    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\"
echo "    --backends unsloth \\"
echo "    --num-steps 3 \\"
echo "    --num-rollouts 16 \\"
echo "    --unsloth-lora-rank 16 \\"
echo "    --tp 2"
echo ""
echo "  # Unsloth MoE backend options:"
echo "  #   --unsloth-lora-rank 16     Higher rank for MoE (default: 16, Unsloth recommends 16-64)"
echo "  #   --unsloth-moe-backend auto  auto|grouped_mm|unsloth_triton|native_torch"
echo "  #   --unsloth-port 8300        SGLang inference port for Unsloth backend"
echo ""
