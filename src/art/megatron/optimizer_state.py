from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterator, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from ..utils.get_model_step import get_step_from_dir
from ..utils.output_dirs import get_step_checkpoint_dir

ALLOW_UNPAIRED_MEGATRON_RESUME_ENV = "ART_ALLOW_UNPAIRED_MEGATRON_RESUME"
OPTIMIZER_GENERATIONS_DIR = "generations"
OPTIMIZER_MANIFEST = "manifest.json"
OPTIMIZER_POINTER = "committed.json"
OPTIMIZER_WRITER_LOCK = ".writer.lock"
ADAPTER_PUBLICATION_ACK = ".optimizer-published.json"
ADAPTER_PUBLICATION_TIMEOUT_ENV = "ART_ADAPTER_PUBLICATION_TIMEOUT_S"
_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
_GENERATION_PATTERN = r"step-\d{8,}-[0-9a-f]{32}"
_GENERATION_RE = re.compile(f"^{_GENERATION_PATTERN}$")
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _OptimizerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MegatronResumeStep(_OptimizerRecord):
    step: int
    latest_lora_step: int
    optimizer_step: int | None
    used_unpaired_override: bool = False
    quarantined_lora_steps: tuple[int, ...] = ()


class OptimizerAdapter(_OptimizerRecord):
    identity: str = Field(min_length=1)
    step: int = Field(ge=0)
    sha256: str = Field(pattern=_SHA256_PATTERN)


class OptimizerTopology(_OptimizerRecord):
    world_size: int = Field(gt=0)
    tp: int = Field(gt=0)
    cp: int = Field(gt=0)
    ep: int = Field(gt=0)
    etp: int = Field(gt=0)
    pp: int = Field(gt=0)
    vpp: int = Field(gt=0)


class OptimizerShard(_OptimizerRecord):
    rank: int = Field(ge=0)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    layout_sha256: str = Field(pattern=_SHA256_PATTERN)


class _PairedOptimizerRecord(_OptimizerRecord):
    step: int = Field(ge=0)
    adapter: OptimizerAdapter

    @model_validator(mode="after")
    def _validate_adapter_step(self) -> "_PairedOptimizerRecord":
        if self.step != self.adapter.step:
            raise ValueError("optimizer and adapter steps must match")
        return self


class OptimizerGenerationManifest(_PairedOptimizerRecord):
    format_version: Literal[2] = 2
    generation: str = Field(pattern=f"^{_GENERATION_PATTERN}$")
    runtime_sha256: str = Field(pattern=_SHA256_PATTERN)
    topology: OptimizerTopology
    shards: tuple[OptimizerShard, ...]


class OptimizerGenerationPointer(_PairedOptimizerRecord):
    format_version: Literal[2] = 2
    generation: str = Field(pattern=f"^{_GENERATION_PATTERN}$")


def optimizer_shard_name(rank: int, world_size: int) -> str:
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(
            f"Invalid optimizer shard rank {rank} for world size {world_size}"
        )
    return f"{rank + 1:02d}-of-{world_size:02d}.pt"


def current_optimizer_topology(world_size: int) -> OptimizerTopology:
    from megatron.core import parallel_state as ps

    return OptimizerTopology(
        world_size=world_size,
        tp=int(ps.get_tensor_model_parallel_world_size()),
        cp=int(ps.get_context_parallel_world_size()),
        ep=int(ps.get_expert_model_parallel_world_size()),
        etp=int(ps.get_expert_tensor_parallel_world_size()),
        pp=int(ps.get_pipeline_model_parallel_world_size()),
        vpp=int(ps.get_virtual_pipeline_model_parallel_world_size() or 1),
    )


def new_optimizer_generation(step: int) -> str:
    if step < 0:
        raise ValueError(f"Optimizer step must be non-negative, got {step}")
    return f"step-{step:08d}-{uuid4().hex}"


def _validate_generation_name(generation: str) -> None:
    if _GENERATION_RE.fullmatch(generation) is None:
        raise ValueError(f"Invalid optimizer generation name: {generation!r}")


def optimizer_pending_generation_path(
    optimizer_state_path: str, generation: str
) -> Path:
    _validate_generation_name(generation)
    return (
        Path(optimizer_state_path)
        / OPTIMIZER_GENERATIONS_DIR
        / f".pending-{generation}"
    )


def optimizer_generation_path(optimizer_state_path: str, generation: str) -> Path:
    _validate_generation_name(generation)
    return Path(optimizer_state_path) / OPTIMIZER_GENERATIONS_DIR / generation


def optimizer_shard_path(generation_path: Path, *, rank: int, world_size: int) -> Path:
    return generation_path / optimizer_shard_name(rank, world_size)


def hash_optimizer_shard(path: Path) -> str:
    return _hash_files((path,))


def hash_adapter_checkpoint(path: str | Path) -> str:
    adapter_path = Path(path)
    files = tuple(adapter_path / name for name in _ADAPTER_FILES)
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        raise RuntimeError(f"Adapter checkpoint is incomplete; missing {missing}")
    return _hash_files(files, relative_to=adapter_path)


def optimizer_adapter(path: str | Path, step: int) -> OptimizerAdapter:
    if step < 0:
        raise ValueError(f"Adapter step must be non-negative, got {step}")
    identity = str(Path(path).absolute())
    return OptimizerAdapter(
        identity=identity,
        step=step,
        sha256=hash_adapter_checkpoint(identity),
    )


def canonical_adapter_path(staging_path: str | Path, step: int) -> Path:
    staging = Path(staging_path).absolute()
    if (
        staging.parent.name != "staging"
        or staging.parent.parent.name != "megatron_runtime"
    ):
        raise RuntimeError(
            "Megatron adapter publication requires the managed staging layout: "
            f"{staging}"
        )
    return Path(
        get_step_checkpoint_dir(str(staging.parent.parent.parent), step)
    ).absolute()


def _canonical_adapter_path(path: str | Path, step: int) -> Path:
    candidate = Path(path).absolute()
    if (
        candidate.parent.name == "staging"
        and candidate.parent.parent.name == "megatron_runtime"
    ):
        return canonical_adapter_path(candidate, step)
    return candidate


def publish_adapter_checkpoint(
    staging_path: str | Path, *, step: int
) -> OptimizerAdapter:
    staging = Path(staging_path).absolute()
    canonical = canonical_adapter_path(staging, step)
    if canonical.exists():
        raise RuntimeError(f"Refusing to replace canonical adapter {canonical}")
    digest = hash_adapter_checkpoint(staging)
    for name in _ADAPTER_FILES:
        with (staging / name).open("rb") as adapter_file:
            os.fsync(adapter_file.fileno())
    _fsync_directory(staging)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, canonical)
    _fsync_directory(canonical.parent)
    adapter = optimizer_adapter(canonical, step)
    if adapter.sha256 != digest:
        raise RuntimeError(
            "Canonical adapter digest changed during publication: "
            f"staged={digest}, canonical={adapter.sha256}"
        )
    _write_model_atomic(canonical / ADAPTER_PUBLICATION_ACK, adapter)
    return adapter


def read_adapter_publication(
    adapter_path: str | Path, *, step: int
) -> OptimizerAdapter | None:
    canonical = _canonical_adapter_path(adapter_path, step)
    acknowledgment = canonical / ADAPTER_PUBLICATION_ACK
    if not acknowledgment.is_file():
        if acknowledgment.exists():
            raise RuntimeError(
                f"Invalid adapter publication acknowledgment: {acknowledgment}"
            )
        return None
    try:
        adapter = OptimizerAdapter.model_validate_json(
            acknowledgment.read_text("utf-8")
        )
    except Exception as exc:
        raise RuntimeError(
            f"Invalid adapter publication acknowledgment: {acknowledgment}"
        ) from exc
    current = optimizer_adapter(canonical, step)
    if adapter != current or "staging" in Path(adapter.identity).parts:
        raise RuntimeError(
            "Adapter publication acknowledgment identity or digest does not match "
            "the canonical adapter: "
            f"acknowledged={adapter.model_dump()}, current={current.model_dump()}"
        )
    return adapter


def _validate_adapter_publication(adapter: OptimizerAdapter) -> None:
    if "staging" in Path(adapter.identity).parts:
        raise RuntimeError(
            f"Optimizer pointers cannot reference a staging adapter: {adapter.identity}"
        )
    if read_adapter_publication(adapter.identity, step=adapter.step) != adapter:
        raise RuntimeError(
            f"Optimizer adapter publication is not acknowledged: {adapter.model_dump()}"
        )


def _hash_files(paths: tuple[Path, ...], *, relative_to: Path | None = None) -> str:
    digest = hashlib.sha256()
    for path in paths:
        name = str(
            path.relative_to(relative_to) if relative_to is not None else path.name
        )
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name.encode())
        with path.open("rb") as input_file:
            while chunk := input_file.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_model_atomic(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            output.write(json.dumps(model.model_dump(mode="json"), sort_keys=True))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_pointer(path: Path) -> OptimizerGenerationPointer | None:
    pointer_path = path / OPTIMIZER_POINTER
    if pointer_path.is_file():
        try:
            return OptimizerGenerationPointer.model_validate_json(
                pointer_path.read_text("utf-8")
            )
        except Exception as exc:
            raise RuntimeError(
                f"Invalid optimizer generation pointer: {pointer_path}"
            ) from exc
    if pointer_path.exists():
        raise RuntimeError(
            f"Invalid optimizer generation pointer: {pointer_path} is not a file"
        )
    if not path.exists():
        return None
    legacy = sorted(
        entry.name
        for entry in path.iterdir()
        if entry.is_file()
        and (
            entry.name == OPTIMIZER_MANIFEST
            or entry.name.isdigit()
            or (entry.name.endswith(".pt") and "-of-" in entry.name)
        )
    )
    if legacy:
        raise RuntimeError(
            "Legacy optimizer checkpoint format is unsupported; expected an atomic "
            f"{OPTIMIZER_POINTER} pointer, found {legacy} in {path}"
        )
    return None


def read_committed_optimizer_pointer(
    optimizer_state_path: str,
) -> OptimizerGenerationPointer | None:
    return _read_pointer(Path(optimizer_state_path))


def read_committed_optimizer_step(optimizer_state_path: str) -> int | None:
    pointer = read_committed_optimizer_pointer(optimizer_state_path)
    return None if pointer is None else pointer.step


def read_committed_optimizer_adapter_step(optimizer_state_path: str) -> int | None:
    pointer = read_committed_optimizer_pointer(optimizer_state_path)
    return None if pointer is None else pointer.adapter.step


def _read_manifest(generation_path: Path) -> OptimizerGenerationManifest:
    manifest_path = generation_path / OPTIMIZER_MANIFEST
    try:
        return OptimizerGenerationManifest.model_validate_json(
            manifest_path.read_text("utf-8")
        )
    except Exception as exc:
        raise RuntimeError(
            f"Invalid optimizer generation manifest: {manifest_path}"
        ) from exc


def _ordered_manifest_shards(
    manifest: OptimizerGenerationManifest,
) -> tuple[OptimizerShard, ...]:
    topology = manifest.topology
    ordered = tuple(sorted(manifest.shards, key=lambda shard: shard.rank))
    expected_ranks = tuple(range(topology.world_size))
    actual_ranks = tuple(shard.rank for shard in ordered)
    if actual_ranks != expected_ranks:
        raise RuntimeError(
            "Optimizer manifest shard coverage mismatch: "
            f"expected_ranks={expected_ranks}, actual_ranks={actual_ranks}"
        )
    return ordered


def build_optimizer_manifest(
    *,
    generation: str,
    step: int,
    adapter: OptimizerAdapter,
    runtime_sha256: str,
    world_size: int,
    shards: list[OptimizerShard],
) -> OptimizerGenerationManifest:
    manifest = OptimizerGenerationManifest(
        generation=generation,
        step=step,
        adapter=adapter,
        runtime_sha256=runtime_sha256,
        topology=current_optimizer_topology(world_size),
        shards=tuple(shards),
    )
    _ordered_manifest_shards(manifest)
    return manifest


def _validate_pointer_manifest(
    pointer: OptimizerGenerationPointer,
    manifest: OptimizerGenerationManifest,
) -> None:
    if (
        manifest.generation,
        manifest.step,
        manifest.adapter,
    ) != (pointer.generation, pointer.step, pointer.adapter):
        raise RuntimeError(
            "Optimizer pointer/manifest identity mismatch: "
            f"pointer={pointer.model_dump()}, manifest={manifest.model_dump()}"
        )


def _validate_generation_files(
    generation_path: Path,
    manifest: OptimizerGenerationManifest,
    *,
    local_rank: int | None,
) -> tuple[OptimizerShard, ...]:
    ordered = _ordered_manifest_shards(manifest)
    names = tuple(
        optimizer_shard_name(shard.rank, manifest.topology.world_size)
        for shard in ordered
    )
    expected_entries = tuple(sorted((OPTIMIZER_MANIFEST, *names)))
    if not generation_path.is_dir():
        raise RuntimeError(
            f"Optimizer generation directory is missing: {generation_path}"
        )
    actual_entries = tuple(sorted(entry.name for entry in generation_path.iterdir()))
    if actual_entries != expected_entries:
        raise RuntimeError(
            "Optimizer generation shard coverage mismatch: "
            f"expected={expected_entries}, actual={actual_entries}"
        )
    for shard in ordered:
        name = optimizer_shard_name(shard.rank, manifest.topology.world_size)
        actual_size = (generation_path / name).stat().st_size
        if actual_size != shard.size_bytes:
            raise RuntimeError(
                f"Optimizer shard size mismatch for {name}: "
                f"expected={shard.size_bytes}, actual={actual_size}"
            )
    if local_rank is not None:
        if local_rank < 0 or local_rank >= len(ordered):
            raise RuntimeError(
                f"Invalid local optimizer rank {local_rank} for {len(ordered)} shards"
            )
        local_shard = ordered[local_rank]
        name = optimizer_shard_name(local_rank, manifest.topology.world_size)
        actual_sha256 = hash_optimizer_shard(generation_path / name)
        if actual_sha256 != local_shard.sha256:
            raise RuntimeError(
                f"Optimizer shard checksum mismatch for {name}: "
                f"expected={local_shard.sha256}, actual={actual_sha256}"
            )
    return ordered


@contextmanager
def _writer_lease(path: Path) -> Iterator[OptimizerGenerationPointer | None]:
    path.mkdir(parents=True, exist_ok=True)
    with (path / OPTIMIZER_WRITER_LOCK).open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield _read_pointer(path)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def commit_optimizer_generation(
    optimizer_state_path: str,
    manifest: OptimizerGenerationManifest,
    *,
    expected_pointer: OptimizerGenerationPointer | None,
) -> Path:
    path = Path(optimizer_state_path)
    pending = optimizer_pending_generation_path(
        optimizer_state_path, manifest.generation
    )
    committed = optimizer_generation_path(optimizer_state_path, manifest.generation)
    _write_model_atomic(pending / OPTIMIZER_MANIFEST, manifest)
    _validate_generation_files(pending, manifest, local_rank=None)
    with _writer_lease(path) as current_pointer:
        if current_pointer != expected_pointer:
            raise RuntimeError(
                "Stale optimizer writer: committed pointer changed before publication; "
                f"expected={expected_pointer.model_dump() if expected_pointer else None}, "
                f"current={current_pointer.model_dump() if current_pointer else None}"
            )
        if current_pointer is not None and manifest.step <= current_pointer.step:
            raise RuntimeError(
                "Optimizer generation step must advance monotonically: "
                f"current={current_pointer.step}, attempted={manifest.step}"
            )
        _validate_adapter_publication(manifest.adapter)
        if committed.exists():
            raise RuntimeError(f"Optimizer generation already exists: {committed}")
        os.replace(pending, committed)
        _fsync_directory(committed.parent)
        pointer = OptimizerGenerationPointer(
            generation=manifest.generation,
            step=manifest.step,
            adapter=manifest.adapter,
        )
        _write_model_atomic(path / OPTIMIZER_POINTER, pointer)
    return committed


def _validate_generation(
    optimizer_state_path: str,
    pointer: OptimizerGenerationPointer,
    world_size: int,
    local_rank: int | None,
) -> tuple[Path, OptimizerGenerationManifest, tuple[OptimizerShard, ...]]:
    path = optimizer_generation_path(optimizer_state_path, pointer.generation)
    manifest = _read_manifest(path)
    _validate_pointer_manifest(pointer, manifest)
    current = current_optimizer_topology(world_size)
    if manifest.topology != current:
        raise RuntimeError(
            "Optimizer checkpoint topology mismatch; optimizer state is topology-strict: "
            f"saved={manifest.topology.model_dump()} current={current.model_dump()}"
        )
    return (
        path,
        manifest,
        _validate_generation_files(path, manifest, local_rank=local_rank),
    )


def pin_optimizer_generation(
    optimizer_state_path: str,
    *,
    world_size: int,
    runtime_sha256: str,
    layout_sha256_by_rank: tuple[str, ...],
    adapter: OptimizerAdapter,
) -> OptimizerGenerationPointer | None:
    pointer = read_committed_optimizer_pointer(optimizer_state_path)
    if pointer is None:
        return None
    _, manifest, ordered = _validate_generation(
        optimizer_state_path, pointer, world_size, None
    )
    if manifest.runtime_sha256 != runtime_sha256:
        raise RuntimeError(
            "Optimizer checkpoint model-runtime mismatch: "
            f"saved={manifest.runtime_sha256}, current={runtime_sha256}"
        )
    if pointer.adapter != adapter:
        raise RuntimeError(
            "Optimizer checkpoint adapter mismatch: "
            f"saved={pointer.adapter.model_dump()}, current={adapter.model_dump()}"
        )
    _validate_adapter_publication(pointer.adapter)
    saved_layouts = tuple(shard.layout_sha256 for shard in ordered)
    if saved_layouts != layout_sha256_by_rank:
        raise RuntimeError(
            "Optimizer parameter ownership/layout mismatch: "
            f"saved={saved_layouts}, current={layout_sha256_by_rank}"
        )
    return pointer


def resolve_optimizer_shard(
    optimizer_state_path: str,
    *,
    rank: int,
    world_size: int,
    pointer: OptimizerGenerationPointer | None = None,
) -> Path | None:
    pointer = pointer or read_committed_optimizer_pointer(optimizer_state_path)
    if pointer is None:
        return None
    generation_path, _, ordered = _validate_generation(
        optimizer_state_path, pointer, world_size, rank
    )
    return generation_path / optimizer_shard_name(ordered[rank].rank, world_size)


def _type_identity(value: object) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _runtime_json_default(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, set):
        return sorted(value, key=repr)
    if callable(value):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", type(value).__qualname__)
        return f"{module}.{name}"
    return _type_identity(value)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        default=_runtime_json_default,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _public_fields(value: object) -> dict[str, Any]:
    return {
        key: item
        for key, item in sorted(vars(value).items())
        if not key.startswith("_")
    }


def _model_runtime_sha256(runtime: Any) -> str:
    return _json_sha256(
        {
            "model_support": runtime.model_support_spec,
            "provider": {
                "type": _type_identity(runtime.provider),
                "fields": _public_fields(runtime.provider),
            },
            "optimizer": _type_identity(runtime.optimizer),
            "optimizer_config": _public_fields(runtime.optimizer_config),
            "compile": runtime.transformer_layers_compiled,
            "topology": current_optimizer_topology(runtime.world_size),
            "torch": torch.__version__,
        }
    )


def _optimizer_layout_sha256(runtime: Any) -> str:
    names_by_parameter: dict[int, list[str]] = {}
    for chunk_index, chunk in enumerate(runtime.model):
        for name, parameter in chunk.named_parameters(remove_duplicate=False):
            qualified = f"chunk.{chunk_index}.{name}"
            names_by_parameter.setdefault(id(parameter), []).append(qualified)
            main_parameter = getattr(parameter, "main_param", None)
            if main_parameter is not None:
                names_by_parameter.setdefault(id(main_parameter), []).append(qualified)

    groups = []
    for group_index, group in enumerate(runtime.optimizer.param_groups):
        parameters = []
        for group_order, parameter in enumerate(group["params"]):
            names = tuple(sorted(set(names_by_parameter.get(id(parameter), ()))))
            if not names:
                raise RuntimeError(
                    "Optimizer parameter is not owned by a model chunk: "
                    f"group={group_index}, order={group_order}, "
                    f"shape={tuple(parameter.shape)}"
                )
            parameters.append(
                {
                    "names": names,
                    "shape": tuple(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "requires_grad": bool(parameter.requires_grad),
                }
            )
        groups.append(parameters)
    return _json_sha256(groups)


def _distributed_rank(runtime: Any) -> int:
    if torch.distributed.is_initialized():  # ty:ignore[possibly-missing-attribute]
        return int(torch.distributed.get_rank())  # ty:ignore[possibly-missing-attribute]
    if (runtime.rank, runtime.world_size) != (0, 1):
        raise RuntimeError(
            "Multi-rank optimizer durability requires an initialized process group: "
            f"rank={runtime.rank}, world_size={runtime.world_size}"
        )
    return 0


def _all_gather_objects(runtime: Any, value: Any) -> list[Any]:
    if not torch.distributed.is_initialized():  # ty:ignore[possibly-missing-attribute]
        _distributed_rank(runtime)
        return [value]
    gathered: list[Any] = [None] * int(
        torch.distributed.get_world_size()  # ty:ignore[possibly-missing-attribute]
    )
    torch.distributed.all_gather_object(  # ty:ignore[possibly-missing-attribute]
        gathered, value
    )
    return gathered


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _result_errors(results: list[Any], missing: str) -> list[str]:
    return [
        f"rank {rank}: {missing}"
        if result is None
        else f"rank {rank}: {result['error']}"
        for rank, result in enumerate(results)
        if result is None or "error" in result
    ]


def optimizer_group_decision(
    runtime: Any,
    decide: Callable[[], Any],
    *,
    operation: str,
) -> Any:
    box: list[dict[str, Any] | None] = [None]
    if _distributed_rank(runtime) == 0:
        try:
            box[0] = {"value": decide()}
        except Exception as exc:
            box[0] = {"error": _error_text(exc)}
    if torch.distributed.is_initialized():  # ty:ignore[possibly-missing-attribute]
        torch.distributed.broadcast_object_list(  # ty:ignore[possibly-missing-attribute]
            box, src=0
        )
    result = box[0]
    if result is None:
        raise RuntimeError(f"Rank 0 returned no {operation} decision")
    if "error" in result:
        raise RuntimeError(f"{operation} failed: {result['error']}")
    return result["value"]


def _raise_rank_errors(runtime: Any, results: list[Any], *, operation: str) -> None:
    def decide() -> None:
        errors = _result_errors(results, "missing result")
        if errors:
            raise RuntimeError("; ".join(errors))

    optimizer_group_decision(runtime, decide, operation=operation)


def _run_rank_operation(runtime: Any, operation: str, run: Callable[[], Any]) -> Any:
    value: Any = None
    try:
        value = run()
        local_result: dict[str, str] = {}
    except Exception as exc:
        local_result = {"error": _error_text(exc)}
    _raise_rank_errors(
        runtime, _all_gather_objects(runtime, local_result), operation=operation
    )
    return value


def _runtime_layout_record(runtime: Any) -> dict[str, Any]:
    try:
        return {
            "rank": runtime.rank,
            "runtime_sha256": _model_runtime_sha256(runtime),
            "layout_sha256": _optimizer_layout_sha256(runtime),
        }
    except Exception as exc:
        return {"rank": runtime.rank, "error": _error_text(exc)}


def _validated_runtime_layouts(
    runtime: Any, records: list[Any]
) -> tuple[str, tuple[str, ...]]:
    errors = _result_errors(records, "missing runtime metadata")
    if errors:
        raise RuntimeError("; ".join(errors))
    ranks = tuple(record["rank"] for record in records)
    expected_ranks = tuple(range(len(records)))
    if ranks != expected_ranks or len(records) != runtime.world_size:
        raise RuntimeError(
            "Optimizer rank metadata mismatch: "
            f"expected={expected_ranks}, actual={ranks}, "
            f"runtime_world={runtime.world_size}"
        )
    runtime_digests = {record["runtime_sha256"] for record in records}
    if len(runtime_digests) != 1:
        raise RuntimeError(
            f"Trainer ranks disagree on model-runtime digest: {sorted(runtime_digests)}"
        )
    return runtime_digests.pop(), tuple(record["layout_sha256"] for record in records)


def _loaded_adapter(adapter_path: str, step: int) -> OptimizerAdapter:
    path = Path(adapter_path).absolute()
    canonical = _canonical_adapter_path(path, step)
    adapter = optimizer_adapter(canonical, step)
    loaded_digest = hash_adapter_checkpoint(path)
    if loaded_digest != adapter.sha256:
        raise RuntimeError(
            "Loaded adapter differs from its canonical checkpoint: "
            f"loaded={loaded_digest}, canonical={adapter.sha256}"
        )
    return adapter


def await_adapter_publication_group(
    runtime: Any,
    staging_path: str,
    *,
    step: int,
    ready: Callable[[], None],
) -> OptimizerAdapter:
    def wait_for_acknowledgment() -> dict[str, Any]:
        ready()
        deadline = time.monotonic() + float(
            os.environ.get(ADAPTER_PUBLICATION_TIMEOUT_ENV, "300")
        )
        while (adapter := read_adapter_publication(staging_path, step=step)) is None:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for canonical adapter publication "
                    f"acknowledgment for step {step}"
                )
            time.sleep(0.05)
        return adapter.model_dump(mode="json")

    return OptimizerAdapter.model_validate(
        optimizer_group_decision(
            runtime,
            wait_for_acknowledgment,
            operation="canonical adapter publication acknowledgment",
        )
    )


def _write_optimizer_shard(
    runtime: Any, generation_path: Path, *, layout_sha256: str
) -> OptimizerShard:
    shard_path = optimizer_shard_path(
        generation_path,
        rank=runtime.rank,
        world_size=runtime.world_size,
    )
    temporary = shard_path.with_name(f".{shard_path.name}.{os.getpid()}.tmp")
    generation_path.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("wb") as output:
            torch.save(runtime.optimizer.state_dict(), output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, shard_path)
    finally:
        temporary.unlink(missing_ok=True)
    return OptimizerShard(
        rank=runtime.rank,
        size_bytes=shard_path.stat().st_size,
        sha256=hash_optimizer_shard(shard_path),
        layout_sha256=layout_sha256,
    )


def save_optimizer_state(
    runtime: Any,
    *,
    optimizer_state_path: str,
    step: int,
    adapter: OptimizerAdapter,
) -> None:
    records = _all_gather_objects(runtime, _runtime_layout_record(runtime))

    def select_generation() -> tuple[str, str, tuple[str, ...], dict[str, Any] | None]:
        runtime_sha256, layouts = _validated_runtime_layouts(runtime, records)
        expected = read_committed_optimizer_pointer(optimizer_state_path)
        if expected is not None and step <= expected.step:
            raise RuntimeError(
                "Optimizer save step must advance the committed pointer: "
                f"current={expected.step}, attempted={step}"
            )
        return (
            new_optimizer_generation(step),
            runtime_sha256,
            layouts,
            None if expected is None else expected.model_dump(mode="json"),
        )

    generation, runtime_sha256, layouts, expected_data = cast(
        tuple[str, str, tuple[str, ...], dict[str, Any] | None],
        optimizer_group_decision(
            runtime, select_generation, operation="optimizer generation selection"
        ),
    )
    expected = (
        None
        if expected_data is None
        else OptimizerGenerationPointer.model_validate(expected_data)
    )
    pending = optimizer_pending_generation_path(optimizer_state_path, generation)
    try:
        shard = _write_optimizer_shard(
            runtime, pending, layout_sha256=layouts[runtime.rank]
        )
        local_result: dict[str, Any] = {"shard": shard.model_dump(mode="json")}
    except Exception as exc:
        local_result = {"rank": runtime.rank, "error": _error_text(exc)}
    gathered = _all_gather_objects(runtime, local_result)

    def publish_generation() -> None:
        errors = _result_errors(gathered, "missing shard metadata")
        if errors:
            raise RuntimeError("; ".join(errors))
        manifest = build_optimizer_manifest(
            generation=generation,
            step=step,
            adapter=adapter,
            runtime_sha256=runtime_sha256,
            world_size=runtime.world_size,
            shards=[
                OptimizerShard.model_validate(result["shard"]) for result in gathered
            ],
        )
        commit_optimizer_generation(
            optimizer_state_path, manifest, expected_pointer=expected
        )

    optimizer_group_decision(
        runtime, publish_generation, operation="optimizer generation publication"
    )


def load_optimizer_state(
    runtime: Any,
    *,
    optimizer_state_path: str,
    adapter_path: str,
    adapter_step: int,
    allow_missing: bool,
    initialize: Callable[[Any], None],
) -> Path | None:
    records = _all_gather_objects(runtime, _runtime_layout_record(runtime))

    def select_generation() -> dict[str, Any] | None:
        runtime_sha256, layouts = _validated_runtime_layouts(runtime, records)
        adapter = _loaded_adapter(adapter_path, adapter_step)
        pointer = pin_optimizer_generation(
            optimizer_state_path,
            world_size=runtime.world_size,
            runtime_sha256=runtime_sha256,
            layout_sha256_by_rank=layouts,
            adapter=adapter,
        )
        if pointer is None and not allow_missing:
            raise RuntimeError(
                "No optimizer generation is paired with canonical adapter "
                f"step {adapter_step}"
            )
        return None if pointer is None else pointer.model_dump(mode="json")

    pointer_data = optimizer_group_decision(
        runtime, select_generation, operation="optimizer load selection"
    )
    if pointer_data is None:
        _run_rank_operation(
            runtime, "optimizer reset", lambda: initialize(runtime.optimizer)
        )
        return None

    pointer = OptimizerGenerationPointer.model_validate(pointer_data)

    def load_shard() -> tuple[Path, Any]:
        shard_path = resolve_optimizer_shard(
            optimizer_state_path,
            rank=runtime.rank,
            world_size=runtime.world_size,
            pointer=pointer,
        )
        assert shard_path is not None
        return shard_path, torch.load(shard_path)

    shard_path, loaded_state = cast(
        tuple[Path, Any],
        _run_rank_operation(runtime, "optimizer shard load", load_shard),
    )
    try:
        _run_rank_operation(
            runtime,
            "optimizer state apply",
            lambda: runtime.optimizer.load_state_dict(loaded_state),
        )
    finally:
        del loaded_state
    return shard_path


def _allow_unpaired_resume() -> bool:
    return os.environ.get(ALLOW_UNPAIRED_MEGATRON_RESUME_ENV, "").lower() in {
        "1",
        "true",
        "yes",
    }


def resolve_megatron_resume_step(
    *,
    output_dir: str,
    optimizer_state_path: str,
) -> MegatronResumeStep:
    latest_lora_step = get_step_from_dir(output_dir)
    pointer = read_committed_optimizer_pointer(optimizer_state_path)
    optimizer_step = None if pointer is None else pointer.step
    if pointer is not None:
        generation_path = optimizer_generation_path(
            optimizer_state_path, pointer.generation
        )
        _validate_pointer_manifest(pointer, _read_manifest(generation_path))
        expected_path = Path(
            get_step_checkpoint_dir(output_dir, pointer.adapter.step)
        ).absolute()
        if pointer.adapter.identity != str(expected_path):
            raise RuntimeError(
                "Optimizer pointer does not identify the canonical adapter path: "
                f"saved={pointer.adapter.identity}, expected={expected_path}"
            )
        _validate_adapter_publication(pointer.adapter)
        return MegatronResumeStep(
            step=pointer.step,
            latest_lora_step=latest_lora_step,
            optimizer_step=pointer.step,
        )
    if latest_lora_step == 0:
        return MegatronResumeStep(
            step=0,
            latest_lora_step=latest_lora_step,
            optimizer_step=None,
        )
    if _allow_unpaired_resume():
        return MegatronResumeStep(
            step=latest_lora_step,
            latest_lora_step=latest_lora_step,
            optimizer_step=None,
            used_unpaired_override=True,
        )
    raise RuntimeError(
        "Cannot resume Megatron training from an unpaired LoRA/optimizer state: "
        f"latest LoRA checkpoint is {latest_lora_step:04d}, no optimizer pointer. "
        f"Set {ALLOW_UNPAIRED_MEGATRON_RESUME_ENV}=1 to override."
    )


def prepare_megatron_resume_state(
    *,
    output_dir: str,
    optimizer_state_path: str,
) -> MegatronResumeStep:
    info = resolve_megatron_resume_step(
        output_dir=output_dir,
        optimizer_state_path=optimizer_state_path,
    )
    if info.used_unpaired_override or info.latest_lora_step <= info.step:
        return info

    checkpoints_dir = Path(output_dir) / "checkpoints"
    quarantine_dir = (
        Path(output_dir)
        / "unpaired_checkpoints"
        / f"resume_from_{info.step:04d}_{int(time.time())}_{os.getpid()}"
    )
    moved_steps: list[int] = []
    for checkpoint_dir in sorted(checkpoints_dir.iterdir()):
        if not checkpoint_dir.is_dir() or not checkpoint_dir.name.isdigit():
            continue
        step = int(checkpoint_dir.name)
        if step <= info.step:
            continue
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.rename(quarantine_dir / checkpoint_dir.name)
        moved_steps.append(step)
    return info.model_copy(update={"quarantined_lora_steps": tuple(moved_steps)})


def format_megatron_resume_message(info: MegatronResumeStep) -> str:
    if info.used_unpaired_override:
        return (
            "Resuming Megatron from unpaired LoRA checkpoint "
            f"{info.step} because {ALLOW_UNPAIRED_MEGATRON_RESUME_ENV} is set"
        )
    if info.step != info.latest_lora_step:
        suffix = ""
        if info.quarantined_lora_steps:
            moved = ", ".join(f"{step:04d}" for step in info.quarantined_lora_steps)
            suffix = f"; quarantined unpaired LoRA checkpoint(s): {moved}"
        return (
            "Resuming Megatron from paired LoRA/optimizer checkpoint "
            f"{info.step} instead of latest LoRA checkpoint "
            f"{info.latest_lora_step}{suffix}"
        )
    return f"Resuming Megatron from checkpoint {info.step}"
