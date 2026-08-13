from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import copyreg
import hashlib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import inspect
from io import BytesIO
import json
import os
from pathlib import Path
import pickle
import sys
import time
from types import CellType, CodeType, FunctionType, MethodType, ModuleType
from typing import TYPE_CHECKING, Any, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field

from .specs import TrainerRuntimeSpec

if TYPE_CHECKING:
    from ..training.compile import TrainingCompilePlan

_PACKAGES = ("megatron-core", "torchmonarch", "transformer-engine", "transformers")
_DEFERRED_PRECOMPILE_INSTALLS: list[tuple[Any, dict[Any, Any]]] = []
_CodeIdentity = tuple[str, str, str, int, CodeType]


def _code_identity(code: CodeType) -> _CodeIdentity:
    return (
        code.co_filename,
        code.co_qualname,
        code.co_name,
        code.co_firstlineno,
        code,
    )


def _canonicalize_code(
    code: CodeType, canonical_codes: dict[_CodeIdentity, CodeType]
) -> CodeType:
    if canonical := canonical_codes.get(_code_identity(code)):
        return canonical
    constants = tuple(
        _canonicalize_code(value, canonical_codes)
        if isinstance(value, CodeType)
        else value
        for value in code.co_consts
    )
    return (
        code
        if all(left is right for left, right in zip(constants, code.co_consts))
        else code.replace(co_consts=constants)
    )


def _canonicalize_serialized_code(
    serialized_code: Any, canonical_codes: dict[_CodeIdentity, CodeType]
) -> None:
    from torch._dynamo import package

    code = package.SerializedCode.to_code_object(serialized_code)
    package._CODE_CACHE[serialized_code] = _canonicalize_code(code, canonical_codes)


def _load_module_global(module_name: str, attribute: str) -> Any:
    return getattr(import_module(module_name), attribute)


def _load_autograd_backward(module_name: str, attribute: str) -> Any:
    backward = _load_module_global(module_name, attribute)._backward_cls
    return backward.__new__(backward)


def _load_import_source(module_name: str) -> Any:
    from torch._dynamo.source import ImportSource

    # __init__ registers a trace-time guard; hydration restores already-saved guards.
    source = ImportSource.__new__(ImportSource)
    object.__setattr__(source, "module_name", module_name)
    return source


def _load_python_code(value: Any) -> Any:
    from torch._dynamo.package import SerializedCode

    return SerializedCode.to_code_object(value)


def _reduce_python_code(code: CodeType) -> tuple[Any, tuple[Any, ...]]:
    from torch._dynamo.package import SerializedCode

    return _load_python_code, (SerializedCode.from_code_object(code),)


def _load_cell(value: Any) -> CellType:
    def cell() -> Any:
        return value

    assert cell.__closure__ is not None
    return cell.__closure__[0]


def _load_python_function(
    code: CodeType,
    module_name: str,
    name: str,
    qualname: str,
    defaults: tuple[Any, ...] | None,
    closure: tuple[CellType, ...] | None,
    kwdefaults: dict[str, Any] | None,
    annotations: dict[str, Any],
    state: dict[str, Any],
) -> FunctionType:
    function = FunctionType(
        code, import_module(module_name).__dict__, name, defaults, closure
    )
    function.__qualname__ = qualname
    function.__kwdefaults__ = kwdefaults
    function.__annotations__ = annotations
    function.__dict__.update(state)
    return function


def _globally_resolves(function: FunctionType) -> bool:
    try:
        value: Any = import_module(function.__module__)
        for part in function.__qualname__.split("."):
            value = getattr(value, part)
    except (AttributeError, ImportError):
        return False
    return value is function


class _CompilePackagePickler(pickle.Pickler):
    dispatch_table = copyreg.dispatch_table | {
        ModuleType: lambda module: (import_module, (module.__name__,)),
        CodeType: _reduce_python_code,
        CellType: lambda cell: (_load_cell, (cell.cell_contents,)),
    }

    def reducer_override(self, value: Any) -> Any:
        value_type = type(value)
        if value_type.__name__ == "HybridEPHandle" and value_type.__module__.endswith(
            ".hybrid_ep_buffer"
        ):
            return value_type, (tuple(value), value.logical_num_tokens)
        if isinstance(value, FunctionType) and not _globally_resolves(value):
            return _load_python_function, (
                value.__code__,
                value.__module__,
                value.__name__,
                value.__qualname__,
                value.__defaults__,
                value.__closure__,
                value.__kwdefaults__,
                value.__annotations__,
                value.__dict__,
            )
        return NotImplemented


class _CompilePackagePickle:
    _art_scoped_reducers = True

    @staticmethod
    def dumps(value: Any, *args: Any, **kwargs: Any) -> bytes:
        buffer = BytesIO()
        pickler = _CompilePackagePickler(buffer, *args, **kwargs)
        pickler.dump(value)
        return buffer.getvalue()

    @staticmethod
    def loads(value: bytes) -> Any:
        return pickle.loads(value)


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
    from torch._dynamo import package
    from torch._dynamo.eval_frame import innermost_fn
    from torch._dynamo.guards import CheckFunctionManager, GuardsStatePickler, _Missing
    from torch._dynamo.package import CompilePackage
    from torch._dynamo.source import DefaultsSource, ImportSource

    if not getattr(package.pickle, "_art_scoped_reducers", False):
        setattr(package, "pickle", _CompilePackagePickle())

    original_source_id_from_fn = CompilePackage.source_id_from_fn
    if not getattr(
        original_source_id_from_fn, "_art_supports_regional_instances", False
    ):

        def precompile_function(fn: Any) -> Any:
            fn = innermost_fn(fn)
            if inspect.isclass(fn):
                fn = innermost_fn(fn.__call__)
            if not hasattr(fn, "__code__"):
                raise TypeError(f"precompile target has no Python code: {fn!r}")
            return fn

        def source_id_from_fn(fn: Any) -> str:
            function = precompile_function(fn)
            source_id = original_source_id_from_fn(function)
            owner = getattr(function, "__self__", None)
            namespace = getattr(owner, "_art_compile_cache_namespace", None)
            if namespace is None:
                return source_id
            return hashlib.sha256(f"{source_id}\0{namespace}".encode()).hexdigest()

        setattr(source_id_from_fn, "_art_supports_regional_instances", True)
        setattr(CompilePackage, "source_id_from_fn", staticmethod(source_id_from_fn))

        original_initialize = CompilePackage.initialize

        def initialize(self: Any, fn: Any, *args: Any, **kwargs: Any) -> None:
            original_initialize(self, precompile_function(fn), *args, **kwargs)

        CompilePackage.initialize = initialize

    original = GuardsStatePickler.reducer_override
    if getattr(original, "_art_supports_instance_method_aliases", False):
        return

    def reducer_override(self: Any, obj: Any) -> Any:
        obj_type = type(obj)
        if obj_type.__name__ == "HybridEPHandle" and obj_type.__module__.endswith(
            ".hybrid_ep_buffer"
        ):
            return obj_type, (tuple(obj), obj.logical_num_tokens)
        # Dtypes are singletons: a missing local dtype may also belong to a guarded tensor.
        if isinstance(obj, torch.dtype):
            return NotImplemented
        # Timers and routing replay retain transient events behind compiled modules.
        if isinstance(obj, torch.cuda.Event) and id(obj) not in self.guard_tree_values:
            return _Missing, ("unguarded CUDA event",)
        if (
            type(obj).__module__ == "hybrid_ep_cpp"
            and id(obj) not in self.guard_tree_values
        ):
            return _Missing, ("unguarded HybridEP runtime state",)
        if isinstance(obj, ContextVar):
            return _load_module_global, _module_global_reference(obj)
        if isinstance(obj, ImportSource):
            return _load_import_source, (obj.module_name,)
        if isinstance(obj, DefaultsSource):
            return type(obj), (obj.base, obj.idx_key, obj.is_kw)
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


def _support_shared_code_precompile() -> None:
    from torch._dynamo import output_graph, package
    from torch._dynamo.package import CompilePackage

    original_install = CompilePackage.install
    if not getattr(original_install, "_art_supports_shared_code", False):
        original_uninstall = CompilePackage.uninstall
        # Regional layer packages share one code object, so hydration must be additive.
        installing = ContextVar("art_precompile_installing", default=False)

        def uninstall(self: Any) -> None:
            if not installing.get():
                original_uninstall(self)

        def install(self: Any, backends: dict[Any, Any]) -> None:
            if getattr(self, "_art_precompile_installed", False):
                raise RuntimeError("compile package was installed more than once")
            try:
                for entry in self._codes.values():
                    for guarded_code in entry.guarded_codes:
                        package.load_guards_state(guarded_code.guards_state)
            except (AttributeError, ImportError):
                # A decorated method can load before its enclosing class is published.
                if getattr(self, "_art_precompile_deferred", False):
                    raise
                self._art_precompile_deferred = True
                _DEFERRED_PRECOMPILE_INSTALLS.append((self, backends))
                return
            # Resume frames embed reconstructed source code in co_consts; execution
            # strategies are attached by identity to the canonical code objects.
            canonical_codes = {}
            for code, entry in self._codes.items():
                if not entry.code_source:
                    continue
                canonical = package._lookup_code(entry)
                if code != canonical:
                    raise RuntimeError("compile package code does not match its source")
                canonical_codes[_code_identity(code)] = canonical

            normalized = {
                _canonicalize_code(code, canonical_codes): entry
                for code, entry in self._codes.items()
            }
            if len(normalized) != len(self._codes):
                raise RuntimeError(
                    "compile package has ambiguous canonical code entries"
                )
            self._codes = normalized
            for entry in self._codes.values():
                for guarded_code in entry.guarded_codes:
                    _canonicalize_serialized_code(
                        guarded_code.dynamo_code, canonical_codes
                    )
            token = installing.set(True)
            try:
                original_install(self, backends)
            finally:
                installing.reset(token)
            self._art_precompile_installed = True

        setattr(install, "_art_supports_shared_code", True)
        CompilePackage.uninstall = uninstall
        CompilePackage.install = install

    original_global = output_graph.OutputGraph.install_global
    if not getattr(original_global, "_art_avoids_hydrated_globals", False):

        def install_global(self: Any, prefix: str, value: Any) -> str:
            # Hydration does not advance Dynamo's process-local unique-id counter.
            while True:
                name = output_graph.unique_id(prefix)
                if name not in self.global_scope:
                    self.install_global_unsafe(name, value)
                    return name

        setattr(install_global, "_art_avoids_hydrated_globals", True)
        output_graph.OutputGraph.install_global = install_global


def finalize_precompile_imports() -> None:
    pending = tuple(_DEFERRED_PRECOMPILE_INSTALLS)
    _DEFERRED_PRECOMPILE_INSTALLS.clear()
    for package, backends in pending:
        package.install(backends)


def _support_serialized_code_resolution() -> None:
    from torch._dynamo import package

    original = package._get_code_source
    if getattr(original, "_art_resolves_serialized_code", False):
        return

    def get_code_source(code: CodeType) -> tuple[str, str]:
        try:
            return original(code)
        except package.PackageError:
            module = inspect.getmodule(code)
            owner: Any = module
            for part in code.co_qualname.split("."):
                if owner is None or not hasattr(owner, part):
                    break
                owner = getattr(owner, part)
                if isinstance(owner, FunctionType):
                    break
            if not isinstance(owner, FunctionType):
                raise

            seen: set[int] = set()

            def candidates(value: Any) -> Iterator[CodeType]:
                if id(value) in seen:
                    return
                seen.add(id(value))
                if isinstance(value, CodeType):
                    if value == code:
                        yield value
                    for constant in value.co_consts:
                        if isinstance(constant, CodeType):
                            yield from candidates(constant)
                elif isinstance(value, FunctionType):
                    yield from candidates(value.__code__)
                    for cell in value.__closure__ or ():
                        try:
                            contents = cell.cell_contents
                        except ValueError:
                            continue
                        if isinstance(contents, (CodeType, FunctionType)):
                            yield from candidates(contents)

            resolved = set()
            for candidate in candidates(owner):
                try:
                    resolved.add(original(candidate))
                except package.PackageError:
                    pass
            if len(resolved) != 1:
                raise
            return resolved.pop()

    setattr(get_code_source, "_art_resolves_serialized_code", True)
    setattr(package, "_get_code_source", get_code_source)


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


def _support_cross_package_resume_codes() -> None:
    from torch._dynamo.package import CompilePackage
    from torch._dynamo.resume_execution import ContinueExecutionCache

    original = CompilePackage.code_context
    if getattr(original, "_art_supports_cross_package_resumes", False):
        return

    @contextmanager
    def code_context(self: Any, code: Any) -> Any:
        if (
            code not in self._codes
            and code in ContinueExecutionCache.generated_code_metadata
        ):
            metadata = ContinueExecutionCache.generated_code_metadata[code]
            module = inspect.getmodule(metadata.code)
            if module is not None:
                names = [
                    name
                    for name, value in vars(module).items()
                    if isinstance(value, FunctionType) and value.__code__ is code
                ]
                for name in names or [None]:
                    self.add_resume_function(code, module.__name__, name)
        with original(self, code):
            yield

    setattr(code_context, "_art_supports_cross_package_resumes", True)
    CompilePackage.code_context = code_context


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
        "schema": 8,
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
    return {kind: list(by_key.values()) for kind, by_key in merged.items()}


def _serialize_cache_artifacts(
    artifacts: dict[str, list[Any]],
) -> tuple[bytes, Any]:
    from torch.compiler._cache import CacheInfo, _serialize_single_cache
    from torch.utils._appending_byte_serializer import AppendingByteSerializer

    info = CacheInfo()
    serializer = AppendingByteSerializer(serialize_fn=_serialize_single_cache)
    serializer.extend(artifacts.items())
    for values in artifacts.values():
        for artifact in values:
            info.add(artifact)
    return serializer.to_bytes(), info


def _merge_cache_artifacts(
    target: dict[str, list[Any]], incoming: dict[str, list[Any]]
) -> None:
    for kind, artifacts in incoming.items():
        by_key = {artifact.key: artifact for artifact in target.get(kind, ())}
        by_key.update((artifact.key, artifact) for artifact in artifacts)
        target[kind] = list(by_key.values())


class TrainerCompileCache:
    """Trusted rank-local PyTorch compiler cache for one exact runtime shape."""

    def __init__(
        self, spec: TrainerRuntimeSpec, *, rank: int, cache_root: Path
    ) -> None:
        from torch._dynamo import config
        from torch._functorch import config as functorch_config

        if not config.caching_precompile:
            raise RuntimeError(
                "trainer compile cache requires Dynamo precompile at process bootstrap"
            )
        _support_precompile_serialization()
        _support_shared_code_precompile()
        _support_serialized_code_resolution()
        _support_non_strict_package_bypass()
        _support_cross_package_resume_codes()
        functorch_config.bundled_autograd_cache = True
        self._spec = spec
        self._rank = rank
        self._cache_root = cache_root
        self.key: str | None = None
        self.path: Path | None = None
        self.loaded = False
        self._artifacts: dict[str, list[Any]] = {}

    def _configure(self, plan: TrainingCompilePlan) -> tuple[str, Path]:
        key = _compile_cache_key(self._spec, self._rank, plan)
        if self.key is not None and self.key != key:
            raise RuntimeError("trainer compile plan changed after cache load")
        self.key = key
        self.path = self._cache_root / "megatron" / "compile_cache" / "v8" / self.key
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
        finalize_precompile_imports()
        started = time.perf_counter()
        key, path = self._configure(plan)
        if not path.is_file():
            return CompileCacheEvent(
                status="miss", key=key, elapsed_s=time.perf_counter() - started
            )
        artifact = path.read_bytes()
        self._artifacts = _deserialize_cache_artifacts(artifact)
        from torch.compiler._cache import CacheArtifactManager

        info = CacheArtifactManager.populate_caches(self._artifacts)
        CacheArtifactManager._seen_artifacts.update(
            artifact for artifacts in self._artifacts.values() for artifact in artifacts
        )
        self._require_precompile(info)
        self.loaded = True
        if os.environ.get("TORCH_STRICT_PRECOMPILE", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            import torch

            torch.compiler.set_stance("fail_on_recompile")
        return CompileCacheEvent(
            status="hit",
            key=key,
            elapsed_s=time.perf_counter() - started,
            artifact_bytes=len(artifact),
        )

    def publish(self) -> CompileCacheEvent:
        import torch
        from torch.compiler._cache import CacheArtifactManager

        started = time.perf_counter()
        key, path = self._configured()
        if not CacheArtifactManager.need_serialize():
            if not path.is_file():
                raise RuntimeError("PyTorch produced no compiler cache after training")
            return CompileCacheEvent(
                status="existing",
                key=key,
                elapsed_s=time.perf_counter() - started,
                artifact_bytes=path.stat().st_size,
            )
        saved = torch.compiler.save_cache_artifacts()
        if saved is None:
            raise RuntimeError("PyTorch produced no compiler cache after training")
        new_artifact, _ = saved
        _merge_cache_artifacts(
            self._artifacts, _deserialize_cache_artifacts(new_artifact)
        )
        artifact, info = _serialize_cache_artifacts(self._artifacts)
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
