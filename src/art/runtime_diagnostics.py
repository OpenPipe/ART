from collections.abc import Callable
import os
from pathlib import Path
import shlex
import shutil
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from art.utils.cache_dirs import cache_filesystem_type, network_cache_warning

RuntimeProgress = Callable[[str], None]


class RuntimeEnvironmentStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: Literal["vllm", "megatron"]
    mode: Literal["managed", "development", "override", "unavailable"]
    profile: str | None = None
    variant: str | None = None
    ready: bool
    path: Path | None = None
    cache_root: Path | None = None
    filesystem: str | None = None
    disk_bytes: int | None = Field(default=None, ge=0)
    free_bytes: int | None = Field(default=None, ge=0)
    warning: str | None = None


class RuntimePreparation(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: Literal["vllm", "megatron"]
    profile: str
    variant: str | None = None
    path: Path
    elapsed_seconds: float = Field(ge=0)


def _disk_bytes(path: Path) -> int:
    total = 0
    seen: set[tuple[int, int]] = set()
    for root, _, files in os.walk(path):
        for name in files:
            try:
                stat = (Path(root) / name).stat(follow_symlinks=False)
            except OSError:
                continue
            identity = (stat.st_dev, stat.st_ino)
            if identity not in seen:
                seen.add(identity)
                total += stat.st_blocks * 512
    return total


def _existing_parent(path: Path) -> Path:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _managed_status(
    *,
    name: Literal["vllm", "megatron"],
    profile: str,
    variant: str | None,
    path: Path,
    cache_root: Path,
    ready: bool,
    include_size: bool,
) -> RuntimeEnvironmentStatus:
    existing = _existing_parent(cache_root)
    return RuntimeEnvironmentStatus(
        name=name,
        mode="managed",
        profile=profile,
        variant=variant,
        ready=ready,
        path=path,
        cache_root=cache_root,
        filesystem=cache_filesystem_type(cache_root),
        disk_bytes=(
            _disk_bytes(cache_root) if include_size and cache_root.exists() else 0
        ),
        free_bytes=shutil.disk_usage(existing).free if existing.exists() else None,
        warning=network_cache_warning(cache_root),
    )


def inspect_vllm_runtime(*, include_size: bool = True) -> RuntimeEnvironmentStatus:
    from art import vllm_runtime as runtime

    profile = runtime.MANAGED_RUNTIME_EXTRA
    if override := os.environ.get("ART_VLLM_RUNTIME_BIN"):
        path = Path(shlex.split(override)[0]).expanduser()
        return RuntimeEnvironmentStatus(
            name="vllm",
            mode="override",
            profile=profile,
            ready=path.is_file(),
            path=path,
        )
    source = runtime._source_runtime_bin()
    if source.is_file():
        return RuntimeEnvironmentStatus(
            name="vllm", mode="development", profile=profile, ready=True, path=source
        )
    bundle = runtime._bundled_runtime_dir()
    if not (bundle / "manifest.json").is_file():
        return RuntimeEnvironmentStatus(
            name="vllm", mode="unavailable", profile=profile, ready=False
        )
    manifest = runtime._load_bundled_manifest(bundle)
    manifest_hash = runtime._manifest_hash(manifest)
    cache_root = runtime.get_vllm_runtime_cache_root().expanduser().resolve()
    path = cache_root / manifest_hash
    ready = runtime._validate_managed_runtime(
        path,
        cache_root=cache_root,
        manifest=manifest,
        manifest_hash=manifest_hash,
    )
    return _managed_status(
        name="vllm",
        profile=profile,
        variant=None,
        path=path,
        cache_root=cache_root,
        ready=ready is not None,
        include_size=include_size,
    )


def inspect_megatron_runtime(
    *,
    require_hybrid_ep: bool = False,
    multinode: bool = False,
    include_size: bool = True,
) -> RuntimeEnvironmentStatus:
    from art.distributed.host_admission import _art_build_sha256
    from art.megatron.runtime import managed as runtime

    if multinode and not require_hybrid_ep:
        raise ValueError("multi-node preparation requires HybridEP")
    profile = runtime._runtime_profile()
    variant = (
        "hybrid_ep_multinode"
        if multinode
        else "hybrid_ep"
        if require_hybrid_ep
        else "base"
    )
    if override := os.environ.get("ART_MEGATRON_RUNTIME_PYTHON"):
        path = Path(override).expanduser()
        return RuntimeEnvironmentStatus(
            name="megatron",
            mode="override",
            profile=profile,
            variant=variant,
            ready=os.access(path, os.X_OK),
            path=path,
        )
    source = runtime._source_runtime_python()
    if source.is_file():
        return RuntimeEnvironmentStatus(
            name="megatron",
            mode="development",
            profile=profile,
            variant=variant,
            ready=os.access(source, os.X_OK),
            path=source,
        )
    bundle = runtime._bundled_runtime_dir()
    if not (bundle / "manifest.json").is_file():
        return RuntimeEnvironmentStatus(
            name="megatron",
            mode="unavailable",
            profile=profile,
            variant=variant,
            ready=False,
        )
    manifest = runtime._load_manifest(bundle)
    art_build_sha256 = _art_build_sha256()
    manifest_hash = runtime._manifest_hash(manifest, profile, variant, art_build_sha256)
    cache_root = runtime._runtime_cache_root().expanduser().resolve()
    path = cache_root / manifest_hash
    ready = runtime._valid_runtime(
        path,
        cache_root=cache_root,
        manifest_hash=manifest_hash,
        profile=profile,
        variant=variant,
    )
    return _managed_status(
        name="megatron",
        profile=profile,
        variant=variant,
        path=path,
        cache_root=cache_root,
        ready=ready is not None,
        include_size=include_size,
    )


def prepare_runtime_environments(
    *,
    require_hybrid_ep: bool = False,
    multinode: bool = False,
    progress: RuntimeProgress,
) -> tuple[RuntimePreparation, RuntimePreparation]:
    from art import vllm_runtime
    from art.distributed.host_admission import _art_build_sha256
    from art.megatron.runtime.managed import ensure_megatron_runtime

    if multinode and not require_hybrid_ep:
        raise ValueError("multi-node preparation requires HybridEP")
    started = time.perf_counter()
    vllm_path = Path(vllm_runtime._runtime_command_prefix(progress=progress)[0])
    vllm = RuntimePreparation(
        name="vllm",
        profile=vllm_runtime.MANAGED_RUNTIME_EXTRA,
        path=vllm_path,
        elapsed_seconds=time.perf_counter() - started,
    )
    started = time.perf_counter()
    megatron_info = ensure_megatron_runtime(
        art_build_sha256=_art_build_sha256(),
        require_hybrid_ep=require_hybrid_ep,
        multinode=multinode,
        progress=progress,
    )
    megatron = RuntimePreparation(
        name="megatron",
        profile=megatron_info.profile,
        variant=megatron_info.variant,
        path=Path(megatron_info.python),
        elapsed_seconds=time.perf_counter() - started,
    )
    return vllm, megatron
