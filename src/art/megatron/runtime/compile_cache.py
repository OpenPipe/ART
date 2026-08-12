from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType
from typing import Any, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field

from .specs import TrainerRuntimeSpec

_PACKAGES = ("megatron-core", "torchmonarch", "transformer-engine", "transformers")


def _support_precompile_serialization() -> None:
    from torch._dynamo.guards import CheckFunctionManager, GuardsStatePickler

    original = GuardsStatePickler.reducer_override
    if getattr(original, "_art_supports_instance_method_aliases", False):
        return

    def reducer_override(self: Any, obj: Any) -> Any:
        # PyTorch assumes a bound method is installed under its function name.
        if isinstance(obj, MethodType) and not hasattr(
            obj.__self__, obj.__func__.__name__
        ):
            return type(self)._unpickle_bound_method, (obj.__func__, obj.__self__)
        return original(self, obj)

    setattr(reducer_override, "_art_supports_instance_method_aliases", True)
    GuardsStatePickler.reducer_override = reducer_override

    serialize = CheckFunctionManager.serialize_guards

    def serialize_guards(
        self: Any, builder: Any, sorted_guards: list[Any], output_graph: Any
    ) -> bytes:
        # PyTorch's first pass filters identity guards, but its second pass can add more.
        unsupported = set(self.UNSUPPORTED_SERIALIZATION_GUARD_TYPES)
        sorted_guards = [
            guard
            for guard in sorted_guards
            if guard.create_fn_name() not in unsupported
            and unsupported.isdisjoint(guard.guard_types or ())
        ]
        return serialize(self, builder, sorted_guards, output_graph)

    CheckFunctionManager.serialize_guards = serialize_guards


def _support_non_strict_package_bypass() -> None:
    from torch._dynamo.convert_frame import DynamoOutput

    original = DynamoOutput.build_guards
    if getattr(original, "_art_supports_package_bypass", False):
        return

    def build_guards(self: Any, *args: Any, **kwargs: Any) -> Any:
        manager = original(self, *args, **kwargs)
        output = self.tracer_output.output_graph
        if (
            kwargs.get("save")
            and manager.guards_state is None
            and output is not None
            and output.package is None
        ):
            # The package entry is already marked as bypassed, so this is not stored.
            manager.guards_state = b""
        return manager

    setattr(build_guards, "_art_supports_package_bypass", True)
    DynamoOutput.build_guards = build_guards


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
        "schema": 2,
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
            "dynamo_precompile": True,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class TrainerCompileCache:
    """Trusted rank-local PyTorch compiler cache for one exact runtime shape."""

    def __init__(
        self, spec: TrainerRuntimeSpec, *, rank: int, cache_root: Path
    ) -> None:
        from torch._dynamo import config

        if config.caching_precompile:
            raise RuntimeError("Dynamo precompile was enabled before trainer imports")
        _support_precompile_serialization()
        _support_non_strict_package_bypass()
        os.environ["TORCH_CACHING_PRECOMPILE"] = "1"
        config.caching_precompile = True
        self.key = _compile_cache_key(spec, rank)
        self.path = cache_root / "megatron" / "compile_cache" / "v2" / self.key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.loaded = False

    def _require_precompile(self, info: Any) -> None:
        if not info.artifacts.get("precompile"):
            raise RuntimeError(f"compile cache {self.key} has no Dynamo artifact")

    def load(self) -> CompileCacheEvent:
        import torch

        started = time.perf_counter()
        if not self.path.is_file():
            return CompileCacheEvent(
                status="miss", key=self.key, elapsed_s=time.perf_counter() - started
            )
        artifact = self.path.read_bytes()
        info = torch.compiler.load_cache_artifacts(artifact)
        if info is None:
            raise RuntimeError(f"PyTorch rejected compile cache {self.key}")
        self._require_precompile(info)
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
        artifact, info = saved
        self._require_precompile(info)
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
