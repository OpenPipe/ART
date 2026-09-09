#!/usr/bin/env bash
set -euo pipefail

log() {
    echo "[art-megatron-setup] $*"
}

fail() {
    log "$*" >&2
    exit 1
}

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
[ -x "${cuda_home}/bin/nvcc" ] || fail "CUDA_HOME must contain bin/nvcc: ${cuda_home}"
for command in gcc g++ ninja nvidia-smi uv; do
    command -v "${command}" >/dev/null || fail "Supported trainer image is missing ${command}"
done

cuda_major="$("${cuda_home}/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -1)"
case "${cuda_major}" in
    12)
        root_extra=megatron
        runtime_extra=cuda12
        ;;
    13)
        root_extra=megatron-cu130
        runtime_extra=cuda13
        ;;
    *)
        fail "Unsupported CUDA major ${cuda_major:-unknown}; expected 12 or 13"
        ;;
esac

if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | sort -u | paste -sd ';')"
fi
[ -n "${TORCH_CUDA_ARCH_LIST}" ] || fail "Could not determine TORCH_CUDA_ARCH_LIST"
export CUDA_HOME="${cuda_home}"
export CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
export TORCH_CUDA_ARCH_LIST

cd "${repo_root}"
log "installing root=${root_extra} trainer=${runtime_extra} arch=${TORCH_CUDA_ARCH_LIST}"
uv sync --extra "${root_extra}" --frozen --inexact
uv sync \
    --project megatron_runtime \
    --extra "${runtime_extra}" \
    --frozen \
    --no-dev \
    --no-install-project \
    --python "${repo_root}/.venv/bin/python"
uv pip install \
    --python "${repo_root}/megatron_runtime/.venv/bin/python" \
    --no-deps \
    --editable "${repo_root}"

multinode=0
if [ "${INSTALL_MULTINODE:-false}" = "true" ]; then
    multinode=1
fi
HYBRID_EP_MULTINODE="${multinode}" USE_NIXL="${multinode}" \
    "${repo_root}/megatron_runtime/.venv/bin/python" \
    -m art.megatron.hybrid_ep_setup

if [ "${INSTALL_VLLM_RUNTIME:-true}" = "true" ]; then
    CUDA_HOME="${cuda_home}" bash vllm_runtime/setup.sh
fi
