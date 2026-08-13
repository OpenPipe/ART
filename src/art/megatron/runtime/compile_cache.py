from __future__ import annotations

from contextvars import ContextVar
import hashlib
from importlib import import_module
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

from ..training.compile import TrainingCompilePlan
from .specs import TrainerRuntimeSpec

_PACKAGES = ("megatron-core", "torchmonarch", "transformer-engine", "transformers")


def _load_module_global(module_name: str, attribute: str) -> Any:
    return getattr(import_module(module_name), attribute)


def _load_autograd_backward(module_name: str, attribute: str) -> Any:
    backward = _load_module_global(module_name, attribute)._backward_cls
    return backward.__new__(backward)


def _module_global_reference(value: Any) -> tuple[str, str]:
    matches = sorted(
        (module_name, attribute)
        for module_name, module in tuple(sys.modules.items())
        if module is not None and module_name != "__main__"
        for attribute, candidate in vars(module).items()
        if candidate is value and attribute.isidentifier()
    )
    if not matches:
        raise TypeError(f"compile guard has no stable module global for {value!r}")
    return matches[0]


def _support_precompile_serialization() -> None:
    import torch
    from torch._dynamo.guards import CheckFunctionManager, GuardsStatePickler, _Missing

    original = GuardsStatePickler.reducer_override
    if getattr(original, "_art_supports_instance_method_aliases", False):
        return

    def reducer_override(self: Any, obj: Any) -> Any:
        # Timers and routing replay retain transient events behind compiled modules.
        if isinstance(obj, torch.cuda.Event) and id(obj) not in self.guard_tree_values:
            return _Missing, ("unguarded CUDA event",)
        if isinstance(obj, ContextVar):
            return _load_module_global, _module_global_reference(obj)
        if isinstance(obj, torch.autograd.graph.Node):
            forward = getattr(type(obj), "_forward_cls", None)
            if forward is None:
                raise TypeError(f"guarded autograd node has no forward class: {obj!r}")
            return _load_autograd_backward, _module_global_reference(forward)
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


def _compile_cache_key(
    spec: TrainerRuntimeSpec,
    rank: int,
    plan: TrainingCompilePlan,
) -> str:
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
        "schema": 3,
        "runtime": runtime,
        "compile_plan": plan.model_dump(mode="json"),
        "environment": {
            "python": sys.implementation.cache_tag,
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "sm": torch.cuda.get_device_capability(),
            "packages": _package_versions(),
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
        self._spec = spec
        self._rank = rank
        self._cache_root = cache_root
        self.key: str | None = None
        self.path: Path | None = None
        self.loaded = False

    def _configure(self, plan: TrainingCompilePlan) -> tuple[str, Path]:
        key = _compile_cache_key(self._spec, self._rank, plan)
        if self.key is not None and self.key != key:
            raise RuntimeError("trainer compile plan changed after cache load")
        self.key = key
        self.path = self._cache_root / "megatron" / "compile_cache" / "v3" / self.key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self.key, self.path

    def _configured(self) -> tuple[str, Path]:
        if self.key is None or self.path is None:
            raise RuntimeError("trainer compile cache was not given a compile plan")
        return self.key, self.path

    def _require_precompile(self, info: Any) -> None:
        if not info.artifacts.get("precompile"):
            raise RuntimeError(f"compile cache {self.key} has no Dynamo artifact")

    def load(self, plan: TrainingCompilePlan) -> CompileCacheEvent:
        import torch

        started = time.perf_counter()
        key, path = self._configure(plan)
        if not path.is_file():
            return CompileCacheEvent(
                status="miss", key=key, elapsed_s=time.perf_counter() - started
            )
        artifact = path.read_bytes()
        info = torch.compiler.load_cache_artifacts(artifact)
        if info is None:
            raise RuntimeError(f"PyTorch rejected compile cache {key}")
        self._require_precompile(info)
        self.loaded = True
        return CompileCacheEvent(
            status="hit",
            key=key,
            elapsed_s=time.perf_counter() - started,
            artifact_bytes=len(artifact),
        )

    def publish(self) -> CompileCacheEvent:
        import torch

        started = time.perf_counter()
        key, path = self._configured()
        if self.loaded or path.is_file():
            return CompileCacheEvent(
                status="existing",
                key=key,
                elapsed_s=time.perf_counter() - started,
                artifact_bytes=path.stat().st_size,
            )
        saved = torch.compiler.save_cache_artifacts()
        if saved is None:
            raise RuntimeError("PyTorch produced no compiler cache after training")
        artifact, info = saved
        self._require_precompile(info)
        staging = path.with_name(f".{key}.{os.getpid()}.{uuid.uuid4().hex}")
        try:
            staging.write_bytes(artifact)
            os.replace(staging, path)
        finally:
            staging.unlink(missing_ok=True)
        self.loaded = True
        return CompileCacheEvent(
            status="published",
            key=key,
            elapsed_s=time.perf_counter() - started,
            artifact_bytes=len(artifact),
        )
