#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
# Install missing cudnn headers, DeepEP RDMA headers, and ninja build tools.
missing_packages=()
for package in libcudnn9-headers-cuda-12 libibverbs-dev ninja-build; do
    if ! dpkg-query -W "${package}" >/dev/null 2>&1; then
        missing_packages+=("${package}")
    fi
done

if [ "${#missing_packages[@]}" -gt 0 ]; then
    if [ "$(id -u)" -eq 0 ]; then
        apt-get update
        apt-get install -y "${missing_packages[@]}"
    elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y "${missing_packages[@]}"
    else
        echo "Missing required packages: ${missing_packages[*]}" >&2
        echo "Install them as root or run with passwordless sudo available." >&2
        exit 1
    fi
fi

# Python dependencies are declared in pyproject.toml extras.
# Megatron setup still needs the shared backend extras, but the vLLM runtime now
# lives in its own project and venv under vllm_runtime/.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd "${repo_root}"
uv_bin="uv"
if [ -x "${HOME}/.local/bin/uv" ]; then
    uv_bin="${HOME}/.local/bin/uv"
fi
"${uv_bin}" sync --extra backend --extra megatron --frozen --active

if [ "${INSTALL_VLLM_RUNTIME:-true}" = "true" ]; then
    "${uv_bin}" sync --project vllm_runtime --frozen --no-dev
fi
