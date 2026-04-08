import os


def _set_cache_dir(env_var: str, default_path: str) -> None:
    if not os.environ.get(env_var):
        os.environ[env_var] = os.path.expanduser(default_path)
    os.makedirs(os.environ[env_var], exist_ok=True)


def configure_megatron_runtime_env() -> None:
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
    _set_cache_dir("TORCHINDUCTOR_CACHE_DIR", "~/.cache/torchinductor")
    _set_cache_dir("TRITON_CACHE_DIR", "~/.triton/cache")
