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


class CompileCacheEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["miss", "hit", "published", "existing"]
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


def _deserialize_cache_artifacts(serialized: bytes) -> dict[str, list[Any]]:
    from torch.compiler._cache import CacheArtifactManager, _deserialize_single_cache
    from torch.utils._appending_byte_serializer import AppendingByteSerializer

    CacheArtifactManager._ensure_cache_artifacts_registered()
    merged: dict[str, dict[str, Any]] = {}
    for kind, artifacts in AppendingByteSerializer.to_list(
        serialized, deserialize_fn=_deserialize_single_cache
    ):
        by_key = merged.setdefault(kind, {})
        by_key.update((artifact.key, artifact) for artifact in artifacts)
    return {
        kind: [by_key[key] for key in sorted(by_key)]
        for kind, by_key in sorted(merged.items())
    }


def _serialize_cache_artifacts(artifacts: dict[str, list[Any]]) -> bytes:
    from torch.compiler._cache import _serialize_single_cache
    from torch.utils._appending_byte_serializer import AppendingByteSerializer

    serializer = AppendingByteSerializer(serialize_fn=_serialize_single_cache)
    serializer.extend((kind, artifacts[kind]) for kind in sorted(artifacts))
    return serializer.to_bytes()


def _merge_cache_artifacts(
    target: dict[str, list[Any]], incoming: dict[str, list[Any]]
) -> None:
    for kind, artifacts in incoming.items():
        by_key = {artifact.key: artifact for artifact in target.get(kind, ())}
        by_key.update((artifact.key, artifact) for artifact in artifacts)
        target[kind] = [by_key[key] for key in sorted(by_key)]


class TrainerCompileCache:
    """Trusted rank-local PyTorch compiler cache for one exact runtime shape."""

    def __init__(
        self, spec: TrainerRuntimeSpec, *, rank: int, cache_root: Path
    ) -> None:
        self.key = _compile_cache_key(spec, rank)
        self.path = cache_root / "megatron" / "compile_cache" / "v1" / self.key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.loaded = False
        self._artifacts: dict[str, list[Any]] = {}

    def load(self) -> CompileCacheEvent:
        import torch
        from torch.compiler._cache import CacheArtifactManager

        started = time.perf_counter()
        if not self.path.is_file():
            return CompileCacheEvent(
                status="miss", key=self.key, elapsed_s=time.perf_counter() - started
            )
        artifact = self.path.read_bytes()
        artifacts = _deserialize_cache_artifacts(artifact)
        if torch.compiler.load_cache_artifacts(artifact) is None:
            raise RuntimeError(f"PyTorch rejected compile cache {self.key}")
        self._artifacts = artifacts
        CacheArtifactManager._seen_artifacts.update(
            artifact for values in artifacts.values() for artifact in values
        )
        self.loaded = True
        return CompileCacheEvent(
            status="hit",
            key=self.key,
            elapsed_s=time.perf_counter() - started,
            artifact_bytes=len(artifact),
        )

    def publish(self) -> CompileCacheEvent:
        import fcntl

        import torch
        from torch.compiler._cache import CacheArtifactManager

        started = time.perf_counter()
        if not CacheArtifactManager.need_serialize():
            if not self.path.is_file():
                raise RuntimeError("PyTorch produced no compiler cache after training")
            return CompileCacheEvent(
                status="existing",
                key=self.key,
                elapsed_s=time.perf_counter() - started,
                artifact_bytes=self.path.stat().st_size,
            )
        saved = torch.compiler.save_cache_artifacts()
        if saved is None:
            raise RuntimeError("PyTorch produced no compiler cache after training")
        artifact, _info = saved
        incoming = _deserialize_cache_artifacts(artifact)
        lock_path = self.path.with_name(f".{self.key}.lock")
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                merged: dict[str, list[Any]] = {}
                _merge_cache_artifacts(merged, self._artifacts)
                if self.path.is_file():
                    _merge_cache_artifacts(
                        merged, _deserialize_cache_artifacts(self.path.read_bytes())
                    )
                _merge_cache_artifacts(merged, incoming)
                artifact = _serialize_cache_artifacts(merged)
                staging = self.path.with_name(
                    f".{self.key}.{os.getpid()}.{uuid.uuid4().hex}"
                )
                try:
                    with staging.open("wb") as output:
                        output.write(artifact)
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(staging, self.path)
                finally:
                    staging.unlink(missing_ok=True)
                self._artifacts = merged
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        self.loaded = True
        return CompileCacheEvent(
            status="published",
            key=self.key,
            elapsed_s=time.perf_counter() - started,
            artifact_bytes=len(artifact),
        )
