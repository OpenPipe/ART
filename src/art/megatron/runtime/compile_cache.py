from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field

from .specs import TrainerRuntimeSpec

_PACKAGES = ("megatron-core", "torchmonarch", "transformer-engine", "transformers")
_MAX_CACHE_ENTRIES = 16
_MAX_CACHE_BYTES = 16 * 1024**3


class CompileCacheEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["miss", "hit", "published", "existing", "rejected"]
    key: str = Field(pattern=r"^[0-9a-f]{64}$")
    elapsed_s: float = Field(ge=0)
    artifact_bytes: int = Field(default=0, ge=0)


def _package_versions() -> dict[str, str]:
    result = {}
    for package in _PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "missing"
    return result


def _compile_cache_key(spec: TrainerRuntimeSpec, rank: int) -> str:
    import torch
    import triton

    runtime = spec.model_dump(
        mode="json",
        exclude={
            "cache_root",
            "compile_cache",
            "compile_fingerprint",
            "optimizer_layout_fingerprint",
            "snapshot_pool_capacity",
            "trainer_mesh",
        },
    )
    runtime.update(
        {
            "rank": rank,
            "topology": spec.trainer_mesh.topology.model_dump(mode="json"),
            "hybrid_ep": (
                None
                if spec.hybrid_ep is None
                else {
                    "multinode": spec.hybrid_ep.multinode,
                    "ranks_per_nvlink_domain": spec.hybrid_ep.ranks_per_nvlink_domain,
                }
            ),
        }
    )
    payload: dict[str, Any] = {
        "schema": 1,
        "runtime": runtime,
        "environment": {
            "python": sys.implementation.cache_tag,
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "sm": torch.cuda.get_device_capability(),
            "packages": _package_versions(),
            "compile_workarounds": os.environ.get(
                "ART_MEGATRON_COMPILE_WORKAROUNDS", "1"
            ),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _prune_cache(root: Path, *, protected: Path | None = None) -> None:
    entries = []
    for path in root.iterdir():
        if len(path.name) != 64 or any(
            char not in "0123456789abcdef" for char in path.name
        ):
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if path.is_file():
            entries.append((path, stat.st_mtime_ns, stat.st_size))
    entries.sort(key=lambda value: value[1])
    total = sum(size for _, _, size in entries)
    remaining = len(entries)
    for path, _modified, size in entries:
        if remaining <= _MAX_CACHE_ENTRIES and total <= _MAX_CACHE_BYTES:
            break
        if path == protected:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        total -= size
        remaining -= 1


class TrainerCompileCache:
    """Trusted rank-local PyTorch compiler cache for one exact runtime shape."""

    def __init__(
        self, spec: TrainerRuntimeSpec, *, rank: int, cache_root: Path
    ) -> None:
        self.key = _compile_cache_key(spec, rank)
        self.path = cache_root / "megatron" / "compile_cache" / "v1" / self.key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.loaded = False

    def load(self) -> CompileCacheEvent:
        import torch

        started = time.perf_counter()
        _prune_cache(self.path.parent, protected=self.path)
        if not self.path.is_file():
            return CompileCacheEvent(
                status="miss", key=self.key, elapsed_s=time.perf_counter() - started
            )
        try:
            artifact = self.path.read_bytes()
        except FileNotFoundError:
            return CompileCacheEvent(
                status="miss", key=self.key, elapsed_s=time.perf_counter() - started
            )
        if len(artifact) > _MAX_CACHE_BYTES:
            self.path.unlink(missing_ok=True)
            return CompileCacheEvent(
                status="rejected",
                key=self.key,
                elapsed_s=time.perf_counter() - started,
                artifact_bytes=len(artifact),
            )
        if torch.compiler.load_cache_artifacts(artifact) is None:
            raise RuntimeError(f"PyTorch rejected compile cache {self.key}")
        try:
            self.path.touch()
        except FileNotFoundError:
            pass
        self.loaded = True
        return CompileCacheEvent(
            status="hit",
            key=self.key,
            elapsed_s=time.perf_counter() - started,
            artifact_bytes=len(artifact),
        )

    def publish(self) -> CompileCacheEvent:
        import torch

        started = time.perf_counter()
        try:
            existing_bytes = self.path.stat().st_size
        except FileNotFoundError:
            existing_bytes = 0
        if self.loaded or existing_bytes:
            return CompileCacheEvent(
                status="existing",
                key=self.key,
                elapsed_s=time.perf_counter() - started,
                artifact_bytes=existing_bytes,
            )
        saved = torch.compiler.save_cache_artifacts()
        if saved is None:
            raise RuntimeError("PyTorch produced no compiler cache after training")
        artifact, _info = saved
        if len(artifact) > _MAX_CACHE_BYTES:
            return CompileCacheEvent(
                status="rejected",
                key=self.key,
                elapsed_s=time.perf_counter() - started,
                artifact_bytes=len(artifact),
            )
        staging = self.path.with_name(f".{self.key}.{os.getpid()}.{uuid.uuid4().hex}")
        try:
            staging.write_bytes(artifact)
            os.replace(staging, self.path)
            _prune_cache(self.path.parent, protected=self.path)
        finally:
            staging.unlink(missing_ok=True)
        self.loaded = True
        return CompileCacheEvent(
            status="published",
            key=self.key,
            elapsed_s=time.perf_counter() - started,
            artifact_bytes=len(artifact),
        )
