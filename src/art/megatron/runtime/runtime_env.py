import os

from art.megatron.runtime.te_cutlass_grouped_gemm import (
    force_te_cutlass_grouped_gemm_env,
    install_te_cutlass_grouped_gemm_guard,
)
from art.utils.cache_dirs import compiler_cache_root, configure_model_cache_env


def _set_cache_dir(env_var: str, default_path: str) -> None:
    path = os.path.expanduser(os.environ.get(env_var) or default_path)
    os.environ[env_var] = path
    os.makedirs(path, exist_ok=True)


def _cache_path(name: str, cache_root: str) -> str:
    return os.path.join(cache_root, name)


def _set_inductor_cache_dir(cache_root: str) -> None:
    from torch._inductor.runtime.cache_dir_utils import default_cache_dir

    if os.environ.get("TORCHINDUCTOR_CACHE_DIR") == default_cache_dir():
        del os.environ["TORCHINDUCTOR_CACHE_DIR"]
    _set_cache_dir(
        "TORCHINDUCTOR_CACHE_DIR",
        _cache_path("torchinductor", cache_root),
    )


def configure_megatron_runtime_env() -> None:
    cache_root = str(configure_model_cache_env())
    compiled_root = str(compiler_cache_root(cache_root))
    force_te_cutlass_grouped_gemm_env()
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = os.environ.get(
        "ART_MEGATRON_CUDA_DEVICE_MAX_CONNECTIONS",
        os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
    )
    _set_inductor_cache_dir(compiled_root)
    _set_cache_dir("TRITON_CACHE_DIR", _cache_path("triton", compiled_root))
    os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
    _set_cache_dir(
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR",
        _cache_path("flash_attention_cute_dsl", compiled_root),
    )
    install_te_cutlass_grouped_gemm_guard()
