#!/usr/bin/env bash
set -euo pipefail

log() {
    echo "[art-megatron-setup] $*"
}

fail() {
    log "$*" >&2
    exit 1
}

detect_cuda_home() {
    local candidate latest=""
    if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
        echo "${CUDA_HOME}"
        return
    fi
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        echo /usr/local/cuda
        return
    fi
    for candidate in /usr/local/cuda-*; do
        if [ -x "${candidate}/bin/nvcc" ]; then
            latest="${candidate}"
        fi
    done
    [ -n "${latest}" ] || fail "Could not find CUDA nvcc; set CUDA_HOME."
    echo "${latest}"
}

cuda_version() {
    local version
    version="$("$1/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2/p' | head -1)"
    [ -n "${version}" ] || fail "Could not parse CUDA version from $1/bin/nvcc."
    echo "${version}"
}

detect_cuda_arch() {
    local arch
    if [ "${ART_MEGATRON_SETUP_RESPECT_TORCH_CUDA_ARCH_LIST:-0}" = "1" ] && [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
        echo "${TORCH_CUDA_ARCH_LIST}"
        return
    fi
    arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')"
    [ -n "${arch}" ] || fail "Could not detect GPU compute capability; set TORCH_CUDA_ARCH_LIST."
    echo "${arch}"
}

configure_nvcc_wrapper() {
    local real_cuda_home="$1" compute="$2" wrapper_root="$3"
    mkdir -p "${wrapper_root}/bin"
    for path in include lib lib64 targets; do
        ln -sfn "${real_cuda_home}/${path}" "${wrapper_root}/${path}"
    done
    cat >"${wrapper_root}/bin/nvcc" <<EOF
#!/usr/bin/env bash
set -euo pipefail
args=()
saw_gencode=0
kept_gencode=0
while [ "\$#" -gt 0 ]; do
    case "\$1" in
        -gencode|--generate-code)
            saw_gencode=1
            if [[ "\${2:-}" == *"arch=compute_${compute},"* ]]; then
                args+=("\$1" "\$2")
                kept_gencode=1
            fi
            shift 2
            ;;
        -gencode=*|--generate-code=*)
            saw_gencode=1
            if [[ "\$1" == *"arch=compute_${compute},"* ]]; then
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
    args+=("-gencode" "arch=compute_${compute},code=sm_${compute}")
fi
exec "${real_cuda_home}/bin/nvcc" "\${args[@]}"
EOF
    chmod +x "${wrapper_root}/bin/nvcc"
}

install_packages() {
    local missing=() package
    for package in "$@"; do
        if ! dpkg-query -W "${package}" >/dev/null 2>&1; then
            apt-cache show "${package}" >/dev/null 2>&1 || fail "Required apt package ${package} is unavailable."
            missing+=("${package}")
        fi
    done
    [ "${#missing[@]}" -gt 0 ] || return 0
    log "Installing apt dependencies: ${missing[*]}"
    if [ "$(id -u)" -eq 0 ]; then
        apt-get update
        apt-get install -y "${missing[@]}"
    elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y "${missing[@]}"
    else
        fail "Need root or passwordless sudo to install: ${missing[*]}"
    fi
}

install_runtime_profile() {
    local source="$1" destination="${ART_MEGATRON_RUNTIME_PROFILE:-/etc/profile.d/50-art-megatron-env.sh}"
    if [ -w "$(dirname "${destination}")" ]; then
        install -m 0644 "${source}" "${destination}"
    elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
        sudo install -m 0644 "${source}" "${destination}"
    else
        fail "Need root or passwordless sudo to install ${destination}."
    fi
    echo "${destination}"
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
real_cuda_home="$(detect_cuda_home)"
read -r cuda_major cuda_minor <<<"$(cuda_version "${real_cuda_home}")"
export TORCH_CUDA_ARCH_LIST="$(detect_cuda_arch)"
export CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
cuda_compute="${TORCH_CUDA_ARCH_LIST%%[ ,;]*}"
cuda_compute="${cuda_compute%%+PTX}"
cuda_compute="${cuda_compute//./}"

case "${cuda_major}" in
    12)
        distributed_extra="distributed"
        megatron_extra="megatron"
        export APEX_CUDA_EXT="${APEX_CUDA_EXT:-1}"
        export APEX_FAST_LAYER_NORM="${APEX_FAST_LAYER_NORM:-1}"
        ;;
    13)
        distributed_extra="distributed-cu130"
        megatron_extra="megatron-cu130"
        export APEX_CUDA_EXT="${APEX_CUDA_EXT:-0}"
        export APEX_FAST_LAYER_NORM="${APEX_FAST_LAYER_NORM:-0}"
        ;;
    *)
        fail "Unsupported CUDA major ${cuda_major}; expected 12 or 13."
        ;;
esac

export CUDA_HOME="${real_cuda_home}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
install_packages "libcudnn9-headers-cuda-${cuda_major}" "cuda-cccl-${cuda_major}-${cuda_minor}" libibverbs-dev ninja-build

if [ ! -f "${CUDA_HOME}/include/cuda/std/tuple" ]; then
    tuple_path="$(find "${CUDA_HOME}/targets" -path '*/include/cccl/cuda/std/tuple' -print -quit 2>/dev/null)"
    [ -n "${tuple_path}" ] || fail "Could not find CUDA CCCL headers."
    cccl_include="$(dirname "$(dirname "$(dirname "${tuple_path}")")")"
    export CPATH="${cccl_include}:${CPATH:-}"
fi

if [ "${cuda_major}" = "13" ]; then
    cuda_wrapper="$(mktemp -d "${TMPDIR:-/tmp}/art-megatron-cuda13-sm${cuda_compute}.XXXXXX")"
    trap 'rm -rf "${cuda_wrapper}"' EXIT
    configure_nvcc_wrapper "${real_cuda_home}" "${cuda_compute}" "${cuda_wrapper}"
    export CUDA_HOME="${cuda_wrapper}"
    export PATH="${CUDA_HOME}/bin:${real_cuda_home}/bin:${PATH}"
fi

log "CUDA_HOME=${real_cuda_home}, profiles=${distributed_extra}+${megatron_extra}, arch=${TORCH_CUDA_ARCH_LIST}"
cd "${repo_root}"
uv_bin="uv"
if [ -x "${HOME}/.local/bin/uv" ]; then
    uv_bin="${HOME}/.local/bin/uv"
fi
"${uv_bin}" sync --extra "${distributed_extra}" --extra "${megatron_extra}" --no-sources-package transformer-engine --frozen --inexact

runtime_library_path=""
for library_dir in \
    "${repo_root}"/.venv/lib/python*/site-packages/nvidia/*/lib \
    /usr/local/art-multinode/nixl/lib/x86_64-linux-gnu \
    /usr/local/art-multinode/ucx/lib \
    "${real_cuda_home}/lib64" \
    "${real_cuda_home}/lib"; do
    [ ! -d "${library_dir}" ] || runtime_library_path="${runtime_library_path:+${runtime_library_path}:}${library_dir}"
done
[ -n "${runtime_library_path}" ] || fail "Could not find the installed CUDA runtime libraries."
export LD_LIBRARY_PATH="${runtime_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
runtime_profile="${repo_root}/.venv/art-megatron-env.sh"
{
    echo '# Generated by ART Megatron setup. Re-run setup after moving this checkout.'
    printf 'export CUDA_HOME=%q\n' "${real_cuda_home}"
    printf 'export TORCH_CUDA_ARCH_LIST=%q\n' "${TORCH_CUDA_ARCH_LIST}"
    printf 'export CUDA_ARCH_LIST=%q\n' "${CUDA_ARCH_LIST}"
    printf 'export PATH=%q/bin${PATH:+:${PATH}}\n' "${real_cuda_home}"
    printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}\n' "${runtime_library_path}"
} >"${runtime_profile}"
runtime_profile="$(install_runtime_profile "${runtime_profile}")"
bash -n "${runtime_profile}"
env -u LD_LIBRARY_PATH bash --noprofile --norc -c \
    'source "$1"; "$2" -c "import torch; import transformer_engine.pytorch"' \
    bash "${runtime_profile}" "${repo_root}/.venv/bin/python"

"${repo_root}/.venv/bin/python" - <<PY
import torch
from transformer_engine.pytorch.quantization import check_fp8_block_scaling_support

expected_cuda = "${cuda_major}"
actual_cuda = str(torch.version.cuda).split(".")[0]
if actual_cuda != expected_cuda:
    raise SystemExit(f"torch CUDA major {actual_cuda} != toolkit CUDA major {expected_cuda}")
print(f"[art-megatron-setup] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[art-megatron-setup] device={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}")
print(f"[art-megatron-setup] transformer-engine fp8 block scaling={check_fp8_block_scaling_support()[0]}")
PY

"${uv_bin}" run --frozen --no-sync python -m art.megatron.hybrid_ep_setup
if [ "${INSTALL_VLLM_RUNTIME:-true}" = "true" ]; then
    CUDA_HOME="${real_cuda_home}" bash vllm_runtime/setup.sh
fi
