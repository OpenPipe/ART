from collections.abc import MutableMapping
import os
from pathlib import Path
import re

_DEFAULT_CACHE_ROOT = Path("/tmp/art-cache")


def compiler_cache_root(
    cache_root: str | Path,
    environ: MutableMapping[str, str] | None = None,
) -> Path:
    environ = os.environ if environ is None else environ
    arch = environ.get("TORCH_CUDA_ARCH_LIST") or environ.get("CUDA_ARCH_LIST")
    arch_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", arch or "unknown")
    return Path(cache_root) / "compiled" / arch_tag


def _set_path(
    environ: MutableMapping[str, str],
    name: str,
    default: str | Path,
    *,
    previous_default: Path | None = None,
) -> Path:
    value = environ.get(name)
    path = Path(value or default).expanduser()
    if previous_default is not None and path == previous_default:
        path = Path(default).expanduser()
    environ[name] = str(path)
    return path


def configure_model_cache_env(
    environ: MutableMapping[str, str] | None = None,
    *,
    cache_root: str | Path | None = None,
) -> Path:
    """Set node-local cache defaults while preserving explicit paths."""
    environ = os.environ if environ is None else environ
    previous_art = environ.get("ART_MEGATRON_CACHE_ROOT")
    previous_root = (
        Path(previous_art).expanduser() if previous_art else _DEFAULT_CACHE_ROOT
    )
    previous_xdg = Path(environ.get("XDG_CACHE_HOME") or previous_root).expanduser()
    previous_hf = Path(
        environ.get("HF_HOME") or previous_xdg / "huggingface"
    ).expanduser()
    previous_hub = Path(
        environ.get("HF_HUB_CACHE")
        or environ.get("HUGGINGFACE_HUB_CACHE")
        or previous_hf / "hub"
    ).expanduser()

    selected_root = cache_root if cache_root is not None else previous_art
    art_root = Path(selected_root).expanduser() if selected_root is not None else None
    if art_root is not None:
        environ["ART_MEGATRON_CACHE_ROOT"] = str(art_root)
    rebase = cache_root is not None
    xdg_root = _set_path(
        environ,
        "XDG_CACHE_HOME",
        art_root or _DEFAULT_CACHE_ROOT,
        previous_default=previous_root if rebase else None,
    )
    hf_home = _set_path(
        environ,
        "HF_HOME",
        xdg_root / "huggingface",
        previous_default=previous_xdg / "huggingface" if rebase else None,
    )
    legacy_hub_cache = environ.get("HUGGINGFACE_HUB_CACHE")
    hub_default = (
        Path(legacy_hub_cache).expanduser()
        if legacy_hub_cache
        and (not rebase or Path(legacy_hub_cache).expanduser() != previous_hf / "hub")
        else hf_home / "hub"
    )
    hub_cache = _set_path(
        environ,
        "HF_HUB_CACHE",
        hub_default,
        previous_default=previous_hf / "hub" if rebase else None,
    )
    compiled_root = compiler_cache_root(xdg_root, environ)
    previous_compiled_root = compiler_cache_root(previous_xdg, environ)
    for name, default, previous_default in (
        ("HUGGINGFACE_HUB_CACHE", hub_cache, previous_hf / "hub"),
        ("TRANSFORMERS_CACHE", hub_cache, previous_hub),
        ("TORCH_HOME", xdg_root / "torch", previous_xdg / "torch"),
        (
            "TORCH_EXTENSIONS_DIR",
            compiled_root / "torch_extensions",
            previous_compiled_root / "torch_extensions",
        ),
        (
            "TORCHINDUCTOR_CACHE_DIR",
            compiled_root / "torchinductor",
            previous_compiled_root / "torchinductor",
        ),
        ("TRITON_HOME", xdg_root, previous_xdg),
        (
            "TRITON_CACHE_DIR",
            compiled_root / "triton",
            previous_compiled_root / "triton",
        ),
        (
            "VLLM_CACHE_ROOT",
            compiled_root / "vllm",
            previous_compiled_root / "vllm",
        ),
        ("VLLM_CONFIG_ROOT", xdg_root / "vllm_config", previous_xdg / "vllm_config"),
    ):
        _set_path(
            environ,
            name,
            default,
            previous_default=previous_default if rebase else None,
        )
    return art_root or xdg_root
