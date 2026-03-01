#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
# install missing cudnn headers & ninja build tools
apt-get update
apt-get install -y libcudnn9-headers-cuda-12 ninja-build

# Python dependencies are declared in pyproject.toml extras.
# Keep backend + megatron together so setup does not prune runtime deps (e.g. vllm).
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd "${repo_root}"
uv sync --extra backend --extra megatron --frozen --active
