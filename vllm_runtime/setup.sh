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

cutlass_cu13_intact() {
    ".venv/bin/python" - <<'PY'
import base64
import hashlib
from importlib.metadata import PackageNotFoundError, distribution

try:
    files = distribution("nvidia-cutlass-dsl-libs-cu13").files
except PackageNotFoundError:
    raise SystemExit(1)
if not files:
    raise SystemExit(1)
for path in files:
    expected = path.hash
    if expected is None or expected.mode != "sha256" or not expected.value:
        continue
    try:
        actual = base64.urlsafe_b64encode(
            hashlib.sha256(path.locate().read_bytes()).digest()
        ).decode().rstrip("=")
    except OSError:
        raise SystemExit(1)
    if actual != expected.value:
        raise SystemExit(1)
PY
}

if [ "${cuda_major}" = 13 ] && ! cutlass_cu13_intact; then
    echo "[art-vllm-runtime-setup] Repairing CUTLASS DSL install-order race"
    site_packages="$(".venv/bin/python" -c \
        'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    # Overlay the wheel because its files share directories with libs-base;
    # uninstalling either wheel first can delete files owned by the other.
    "${uv_bin}" pip install --python .venv/bin/python --target "${site_packages}" \
        --reinstall --no-deps \
        nvidia-cutlass-dsl-libs-cu13==4.5.2
    cutlass_cu13_intact || {
        echo "[art-vllm-runtime-setup] CUTLASS DSL integrity check failed" >&2
        exit 1
    }
fi

".venv/bin/python" - <<'PY'
import torch
import vllm

print(f"[art-vllm-runtime-setup] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[art-vllm-runtime-setup] vllm={vllm.__version__}")
print(f"[art-vllm-runtime-setup] device={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}")
PY
