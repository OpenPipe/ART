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


class TrainerCompileCache:
    """Trusted rank-local PyTorch compiler cache for one exact runtime shape."""

    def __init__(
        self, spec: TrainerRuntimeSpec, *, rank: int, cache_root: Path
    ) -> None:
        if os.environ.get("TORCH_CACHING_PRECOMPILE", "0") == "1":
            raise RuntimeError("TORCH_CACHING_PRECOMPILE is not rank-qualified")
        self.key = _compile_cache_key(spec, rank)
        self.path = cache_root / "megatron" / "compile_cache" / "v1" / self.key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.loaded = False

    def load(self) -> CompileCacheEvent:
        import torch

        started = time.perf_counter()
        if not self.path.is_file():
            return CompileCacheEvent(
                status="miss", key=self.key, elapsed_s=time.perf_counter() - started
            )
        artifact = self.path.read_bytes()
        if torch.compiler.load_cache_artifacts(artifact) is None:
            raise RuntimeError(f"PyTorch rejected compile cache {self.key}")
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
        if self.loaded or self.path.is_file():
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
        staging = self.path.with_name(f".{self.key}.{os.getpid()}.{uuid.uuid4().hex}")
        try:
            staging.write_bytes(artifact)
            os.replace(staging, self.path)
        finally:
            staging.unlink(missing_ok=True)
        self.loaded = True
        return CompileCacheEvent(
            status="published",
            key=self.key,
            elapsed_s=time.perf_counter() - started,
            artifact_bytes=len(artifact),
        )
