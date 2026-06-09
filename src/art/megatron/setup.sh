#!/usr/bin/env bash
set -euo pipefail

log() {
    echo "[art-megatron-setup] $*"
}

fail() {
    echo "[art-megatron-setup] $*" >&2
    exit 1
}

detect_cuda_home() {
    if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
        echo "${CUDA_HOME}"
        return
    fi
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        echo /usr/local/cuda
        return
    fi

    local candidate latest
    latest=""
    for candidate in /usr/local/cuda-*; do
        if [ -x "${candidate}/bin/nvcc" ]; then
            latest="${candidate}"
        fi
    done
    if [ -n "${latest}" ]; then
        echo "${latest}"
        return
    fi
    fail "Could not find CUDA nvcc. Set CUDA_HOME to the CUDA toolkit path."
}

detect_cuda_major() {
    local cuda_home="$1"
    local major
    major="$("${cuda_home}/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -n 1)"
    if [ -z "${major}" ]; then
        fail "Could not parse CUDA major version from ${cuda_home}/bin/nvcc --version."
    fi
    echo "${major}"
}

detect_cuda_minor() {
    local cuda_home="$1"
    local minor
    minor="$("${cuda_home}/bin/nvcc" --version | sed -n 's/.*release [0-9][0-9]*\.\([0-9][0-9]*\).*/\1/p' | head -n 1)"
    if [ -z "${minor}" ]; then
        fail "Could not parse CUDA minor version from ${cuda_home}/bin/nvcc --version."
    fi
    echo "${minor}"
}

detect_torch_cuda_arch_list() {
    if [ "${ART_MEGATRON_SETUP_RESPECT_TORCH_CUDA_ARCH_LIST:-0}" = "1" ] && [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
        echo "${TORCH_CUDA_ARCH_LIST}"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local cap
        cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
        if [ -n "${cap}" ]; then
            echo "${cap}"
            return
        fi
    fi
    echo "9.0"
}

cuda_compute_from_arch_list() {
    local arch="$1"
    arch="${arch%%[ ,;]*}"
    arch="${arch%%+PTX}"
    arch="${arch//./}"
    if [ -z "${arch}" ]; then
        fail "Could not derive CUDA compute capability from TORCH_CUDA_ARCH_LIST."
    fi
    echo "${arch}"
}

configure_cuda_home_wrapper() {
    local real_cuda_home="$1"
    local compute="$2"
    local wrapper_root="$3"
    local wrapper_bin="${wrapper_root}/bin"
    mkdir -p "${wrapper_bin}"
    ln -sfn "${real_cuda_home}/include" "${wrapper_root}/include"
    ln -sfn "${real_cuda_home}/lib64" "${wrapper_root}/lib64"
    ln -sfn "${real_cuda_home}/lib" "${wrapper_root}/lib"
    ln -sfn "${real_cuda_home}/targets" "${wrapper_root}/targets"
    cat >"${wrapper_bin}/nvcc" <<EOF
#!/usr/bin/env bash
set -euo pipefail
real_nvcc="${real_cuda_home}/bin/nvcc"
target_compute="${compute}"
args=()
saw_gencode=0
kept_gencode=0
while [ "\$#" -gt 0 ]; do
    case "\$1" in
        -gencode|--generate-code)
            saw_gencode=1
            key="\$1"
            value="\${2:-}"
            if [[ "\${value}" == *"arch=compute_\${target_compute},"* ]]; then
                args+=("\${key}" "\${value}")
                kept_gencode=1
            fi
            shift 2
            ;;
        -gencode=*|--generate-code=*)
            saw_gencode=1
            if [[ "\$1" == *"arch=compute_\${target_compute},"* ]]; then
                args+=("\$1")
                kept_gencode=1
            fi
            shift
            ;;
        *)
            args+=("\$1")
            shift
            ;;
    esac
done
if [ "\${saw_gencode}" -eq 1 ] && [ "\${kept_gencode}" -eq 0 ]; then
    args+=("-gencode" "arch=compute_\${target_compute},code=sm_\${target_compute}")
fi
exec "\${real_nvcc}" "\${args[@]}"
EOF
    chmod +x "${wrapper_bin}/nvcc"
}

select_cudnn_headers_package() {
    local cuda_major="$1"
    local candidate="libcudnn9-headers-cuda-${cuda_major}"
    if apt-cache show "${candidate}" >/dev/null 2>&1; then
        echo "${candidate}"
        return
    fi
    fail "Missing apt package ${candidate}. Ensure the CUDA ${cuda_major} NVIDIA apt repo is configured."
}

select_cuda_cccl_package() {
    local cuda_major="$1"
    local cuda_minor="$2"
    local candidate="cuda-cccl-${cuda_major}-${cuda_minor}"
    if apt-cache show "${candidate}" >/dev/null 2>&1; then
        echo "${candidate}"
        return
    fi
    if [ ! -f "${CUDA_HOME}/include/cuda/std/tuple" ]; then
        fail "Missing cuda/std headers and apt package ${candidate} is unavailable."
    fi
}

detect_cuda_cccl_include() {
    if [ -f "${CUDA_HOME}/include/cuda/std/tuple" ]; then
        return
    fi

    local tuple_path
    tuple_path="$(find "${CUDA_HOME}/targets" -path '*/include/cccl/cuda/std/tuple' -print -quit 2>/dev/null || true)"
    if [ -n "${tuple_path}" ]; then
        dirname "$(dirname "$(dirname "${tuple_path}")")"
        return
    fi

    fail "Could not find CUDA CCCL include path containing cuda/std/tuple."
}

run_as_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
        sudo "$@"
    else
        fail "Need root or passwordless sudo to run: $*"
    fi
}

link_cuda_cccl_headers() {
    local cccl_include="$1"
    local name
    mkdir -p "${CUDA_HOME}/include"
    for name in cuda cub thrust; do
        if [ -e "${cccl_include}/${name}" ] && [ ! -e "${CUDA_HOME}/include/${name}" ]; then
            run_as_root ln -s "${cccl_include}/${name}" "${CUDA_HOME}/include/${name}"
        fi
    done
}

install_missing_packages() {
    local missing_packages=("$@")
    if [ "${#missing_packages[@]}" -eq 0 ]; then
        return
    fi

    log "Installing missing Megatron apt dependencies: ${missing_packages[*]}"
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
}

real_cuda_home="$(detect_cuda_home)"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
export CUDA_HOME="${real_cuda_home}"
cuda_major="$(detect_cuda_major "${CUDA_HOME}")"
cuda_minor="$(detect_cuda_minor "${CUDA_HOME}")"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="$(detect_torch_cuda_arch_list)"
export CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
cuda_compute="$(cuda_compute_from_arch_list "${TORCH_CUDA_ARCH_LIST}")"

case "${cuda_major}" in
    12)
        backend_extra="backend"
        megatron_extra="megatron"
        export APEX_CUDA_EXT="${APEX_CUDA_EXT:-1}"
        export APEX_FAST_LAYER_NORM="${APEX_FAST_LAYER_NORM:-1}"
        ;;
    13)
        backend_extra="backend-cu130"
        megatron_extra="megatron-cu130"
        export APEX_CUDA_EXT="${APEX_CUDA_EXT:-0}"
        export APEX_FAST_LAYER_NORM="${APEX_FAST_LAYER_NORM:-0}"
        export CAUSAL_CONV1D_FORCE_BUILD="${CAUSAL_CONV1D_FORCE_BUILD:-TRUE}"
        export CAUSAL_CONV1D_SKIP_CUDA_BUILD="${CAUSAL_CONV1D_SKIP_CUDA_BUILD:-FALSE}"
        ;;
    *)
        fail "Unsupported CUDA major version ${cuda_major}; expected CUDA 12 or 13."
        ;;
esac

log "CUDA_HOME=${CUDA_HOME}"
log "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
log "Using uv extras: ${backend_extra}, ${megatron_extra}"

# Install missing cudnn headers, DeepEP RDMA headers, and ninja build tools.
cudnn_headers_package="$(select_cudnn_headers_package "${cuda_major}")"
cuda_cccl_package="$(select_cuda_cccl_package "${cuda_major}" "${cuda_minor}")"
missing_packages=()
for package in "${cudnn_headers_package}" "${cuda_cccl_package}" libibverbs-dev ninja-build; do
    if [ -z "${package}" ]; then
        continue
    fi
    if ! dpkg-query -W "${package}" >/dev/null 2>&1; then
        missing_packages+=("${package}")
    fi
done
install_missing_packages "${missing_packages[@]}"

cuda_cccl_include="$(detect_cuda_cccl_include)"
if [ -n "${cuda_cccl_include}" ]; then
    link_cuda_cccl_headers "${cuda_cccl_include}"
    export CPATH="${cuda_cccl_include}:${CPATH:-}"
    log "CPATH includes CUDA CCCL headers: ${cuda_cccl_include}"
fi

if [ "${cuda_major}" = "13" ]; then
    cuda_wrapper_root="${repo_root}/scratch/megatron_setup_cuda${cuda_major}_sm${cuda_compute}"
    configure_cuda_home_wrapper "${real_cuda_home}" "${cuda_compute}" "${cuda_wrapper_root}"
    export CUDA_HOME="${cuda_wrapper_root}"
    export PATH="${CUDA_HOME}/bin:${real_cuda_home}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
    cublaslt_preload="${real_cuda_home}/targets/sbsa-linux/lib/libcublasLt.so.${cuda_major}"
    if [ ! -f "${cublaslt_preload}" ]; then
        cublaslt_preload="${real_cuda_home}/lib64/libcublasLt.so.${cuda_major}"
    fi
    if [ -f "${cublaslt_preload}" ]; then
        export LD_PRELOAD="${cublaslt_preload}${LD_PRELOAD:+:${LD_PRELOAD}}"
        log "LD_PRELOAD includes CUDA ${cuda_major} cuBLASLt: ${cublaslt_preload}"
    fi
    log "CUDA13 nvcc wrapper filters extension builds to sm_${cuda_compute}"
fi

cd "${repo_root}"
uv_bin="uv"
if [ -x "${HOME}/.local/bin/uv" ]; then
    uv_bin="${HOME}/.local/bin/uv"
fi
"${uv_bin}" sync \
    --extra "${backend_extra}" \
    --extra "${megatron_extra}" \
    --no-sources-package transformer-engine \
    --frozen \
    --active

venv_site_packages="$("${repo_root}/.venv/bin/python" - <<'PY'
import sysconfig

print(sysconfig.get_paths()["purelib"])
PY
)"
venv_torch_lib="${venv_site_packages}/torch/lib"
if [ -d "${venv_torch_lib}" ]; then
    export LD_LIBRARY_PATH="${venv_torch_lib}:${LD_LIBRARY_PATH:-}"
fi

if [ "${cuda_major}" = "13" ] && ! "${repo_root}/.venv/bin/python" -c "import torch; import causal_conv1d_cuda" >/dev/null 2>&1; then
    log "Building causal-conv1d CUDA extension for sm_${cuda_compute}"
    "${repo_root}/.venv/bin/python" -m pip install \
        --force-reinstall \
        --no-build-isolation \
        --no-cache-dir \
        --no-deps \
        "causal-conv1d==1.6.1"
fi

"${repo_root}/.venv/bin/python" - <<PY
import torch
import causal_conv1d_cuda
from transformer_engine.pytorch.quantization import check_fp8_block_scaling_support

expected = "${cuda_major}"
actual = str(torch.version.cuda).split(".")[0]
if actual != expected:
    raise SystemExit(f"torch CUDA major {actual} does not match detected CUDA major {expected}")
print(f"[art-megatron-setup] torch={torch.__version__} cuda={torch.version.cuda}")
print("[art-megatron-setup] causal-conv1d CUDA extension=available")
print(f"[art-megatron-setup] transformer-engine fp8 block scaling={check_fp8_block_scaling_support()[0]}")
PY
