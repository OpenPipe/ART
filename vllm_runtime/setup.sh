#!/usr/bin/env bash
set -euo pipefail

cuda_home="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -x "${cuda_home}/bin/nvcc" ]; then
    echo "[art-vllm-runtime-setup] CUDA_HOME does not contain nvcc: ${cuda_home}" >&2
    exit 1
fi
cuda_major="$("${cuda_home}/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -1)"
case "${cuda_major}" in
    12) runtime_extra="cuda12" ;;
    13) runtime_extra="cuda13" ;;
    *)
        echo "[art-vllm-runtime-setup] Unsupported CUDA major ${cuda_major}; expected 12 or 13." >&2
        exit 1
        ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"
uv_bin="uv"
if [ -x "${HOME}/.local/bin/uv" ]; then
    uv_bin="${HOME}/.local/bin/uv"
fi
echo "[art-vllm-runtime-setup] CUDA_HOME=${cuda_home}, profile=${runtime_extra}"
"${uv_bin}" sync --extra "${runtime_extra}" --frozen --no-dev
".venv/bin/python" - <<'PY'
import torch
import vllm

print(f"[art-vllm-runtime-setup] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[art-vllm-runtime-setup] vllm={vllm.__version__}")
print(f"[art-vllm-runtime-setup] device={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}")
PY
