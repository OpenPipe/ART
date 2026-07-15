import os

from art.megatron.runtime.te_cutlass_grouped_gemm import (
    force_te_cutlass_grouped_gemm_env,
    install_te_cutlass_grouped_gemm_guard,
)


def _set_cache_dir(env_var: str, default_path: str) -> None:
    if not os.environ.get(env_var):
        os.environ[env_var] = os.path.expanduser(default_path)
    os.makedirs(os.environ[env_var], exist_ok=True)


def configure_megatron_runtime_env() -> None:
    force_te_cutlass_grouped_gemm_env()
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = os.environ.get(
        "ART_MEGATRON_CUDA_DEVICE_MAX_CONNECTIONS",
        os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
    )
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # The currently validated ART MoE grouped-GEMM runtime is SM90. Future
    # SM100 support should come from the TE grouped-GEMM implementation, not
    # ART-side kernel special casing.
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
    _set_cache_dir("TORCHINDUCTOR_CACHE_DIR", "~/.cache/torchinductor")
    _set_cache_dir("TRITON_CACHE_DIR", "~/.triton/cache")
    install_te_cutlass_grouped_gemm_guard()
