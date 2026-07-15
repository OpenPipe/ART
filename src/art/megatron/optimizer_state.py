from __future__ import annotations

from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
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
OPTIMIZER_MODEL_LOCK = ".optimizer.lock"
OPTIMIZER_WRITER_LOCK = ".writer.lock"
OPTIMIZER_GENERATION_LEASE_PREFIX = ".lease-"
OPTIMIZER_TRASH_PREFIX = ".trash-"
OPTIMIZER_ORPHAN_GRACE_S = 3600.0
ADAPTER_PUBLICATION_ACK = ".optimizer-published.json"
ADAPTER_PUBLICATION_TIMEOUT_ENV = "ART_ADAPTER_PUBLICATION_TIMEOUT_S"
_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
_GENERATION_PATTERN = r"step-\d{8,}-[0-9a-f]{32}"
_GENERATION_RE = re.compile(f"^{_GENERATION_PATTERN}$")
_TRASH_RE = re.compile(f"^\\.trash-({_GENERATION_PATTERN})-[0-9a-f]{{32}}$")
_POINTER_TEMP_RE = re.compile(r"^\.committed\.json\.\d+\.[0-9a-f]{32}\.tmp$")
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_POINTER_UNSET = object()


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


def _generation_lease_path(path: Path, generation: str) -> Path:
    _validate_generation_name(generation)
    return (
        path
        / OPTIMIZER_GENERATIONS_DIR
        / f"{OPTIMIZER_GENERATION_LEASE_PREFIX}{generation}"
    )


def _generation_step(generation: str) -> int:
    _validate_generation_name(generation)
    return int(generation.split("-", 2)[1])


@contextmanager
def optimizer_model_lease(optimizer_state_path: str | Path) -> Iterator[None]:
    model_root = Path(optimizer_state_path).absolute().parent
    model_root.mkdir(parents=True, exist_ok=True)
    with (model_root / OPTIMIZER_MODEL_LOCK).open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def optimizer_shard_path(generation_path: Path, *, rank: int, world_size: int) -> Path:
    return generation_path / optimizer_shard_name(rank, world_size)


def hash_optimizer_shard(path: Path) -> str:
    return _hash_files((path,))


def _adapter_checkpoint_files(path: str | Path) -> tuple[Path, tuple[Path, ...]]:
    adapter_path = Path(path)
    files = tuple(adapter_path / name for name in _ADAPTER_FILES)
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        raise RuntimeError(f"Adapter checkpoint is incomplete; missing {missing}")
    return adapter_path, files


def hash_adapter_checkpoint(path: str | Path) -> str:
    adapter_path, files = _adapter_checkpoint_files(path)
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
    _, files = _adapter_checkpoint_files(staging)
    for path in files:
        with path.open("rb") as adapter_file:
            os.fsync(adapter_file.fileno())
    _fsync_directory(staging)
    adapter = OptimizerAdapter(
        identity=str(canonical),
        step=step,
        sha256=hash_adapter_checkpoint(staging),
    )
    _write_model_atomic(staging / ADAPTER_PUBLICATION_ACK, adapter)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, canonical)
    _fsync_directory(canonical.parent)
    return adapter


def read_adapter_publication(
    adapter_path: str | Path,
    *,
    step: int,
    verify_files: bool = True,
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
    expected_identity = str(canonical)
    if (
        adapter.identity != expected_identity
        or adapter.step != step
        or "staging" in Path(adapter.identity).parts
    ):
        raise RuntimeError(
            "Adapter publication acknowledgment does not identify the canonical "
            f"adapter: acknowledged={adapter.model_dump()}, "
            f"expected_identity={expected_identity!r}, expected_step={step}"
        )
    if verify_files:
        current_digest = hash_adapter_checkpoint(canonical)
        if adapter.sha256 != current_digest:
            raise RuntimeError(
                "Adapter publication acknowledgment digest does not match the "
                f"canonical adapter: acknowledged={adapter.sha256}, "
                f"current={current_digest}"
            )
    return adapter


def _validate_adapter_publication(
    adapter: OptimizerAdapter, *, verify_files: bool = False
) -> None:
    if "staging" in Path(adapter.identity).parts:
        raise RuntimeError(
            f"Optimizer pointers cannot reference a staging adapter: {adapter.identity}"
        )
    if (
        read_adapter_publication(
            adapter.identity,
            step=adapter.step,
            verify_files=verify_files,
        )
        != adapter
    ):
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
def _root_lease(
    path: Path, operation: int
) -> Iterator[OptimizerGenerationPointer | None]:
    path.mkdir(parents=True, exist_ok=True)
    with (path / OPTIMIZER_WRITER_LOCK).open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), operation)
        try:
            yield _read_pointer(path)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _writer_lease(path: Path) -> Iterator[OptimizerGenerationPointer | None]:
    with _root_lease(path, fcntl.LOCK_EX) as pointer:
        yield pointer


@contextmanager
def _generation_lease(
    path: Path,
    generation: str,
    *,
    exclusive: bool,
    nonblocking: bool = False,
) -> Iterator[bool]:
    lease_path = _generation_lease_path(path, generation)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    with lease_path.open("a+b") as lease_file:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if nonblocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(lease_file.fileno(), operation)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lease_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _committed_generation_lease(
    path: Path,
) -> Iterator[OptimizerGenerationPointer | None]:
    stack = ExitStack()
    with _root_lease(path, fcntl.LOCK_SH) as pointer:
        if pointer is not None and not stack.enter_context(
            _generation_lease(path, pointer.generation, exclusive=False)
        ):
            raise RuntimeError(
                f"Could not lease optimizer generation {pointer.generation}"
            )
    try:
        yield pointer
    finally:
        stack.close()


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


def _prune_optimizer_generations_locked(
    optimizer_state_path: str,
    *,
    retain_adapter_steps: set[int],
    orphan_grace_s: float = OPTIMIZER_ORPHAN_GRACE_S,
) -> set[int]:
    """Reclaim unretained generations and return adapter steps still in use."""
    if orphan_grace_s < 0:
        raise ValueError("orphan_grace_s must be non-negative")
    path = Path(optimizer_state_path)
    generations = path / OPTIMIZER_GENERATIONS_DIR
    if not path.exists():
        return set()

    protected_steps: set[int] = set()
    trash: list[Path] = []
    now = time.time()
    with _writer_lease(path) as pointer:
        if not generations.exists():
            if pointer is not None:
                raise RuntimeError(
                    "Optimizer pointer exists without a generations directory: "
                    f"{generations}"
                )
            return set()
        if not generations.is_dir():
            raise RuntimeError(
                f"Optimizer generations path is not a directory: {generations}"
            )

        pointer_temps, candidates = _scan_optimizer_transactions(path)
        if len(pointer_temps) > 1:
            raise RuntimeError(
                "Cannot collect optimizer generations with multiple interrupted "
                "pointers"
            )
        records: list[tuple[Path, str, int, bool, bool]] = []
        manifests: dict[str, OptimizerGenerationManifest] = {}
        current_found = False
        for entry, generation, pending in candidates:
            step = _generation_step(generation)
            manifest = None if pending else _read_manifest(entry)
            if manifest is not None and (
                manifest.generation != generation or manifest.step != step
            ):
                raise RuntimeError(
                    f"Optimizer generation directory/manifest mismatch: {entry}"
                )
            if manifest is not None:
                manifests[generation] = manifest
            adapter_step = step if manifest is None else manifest.adapter.step
            current = pointer is not None and pointer.generation == generation
            young = now - entry.stat().st_mtime < orphan_grace_s
            if current:
                assert manifest is not None
                _validate_pointer_manifest(pointer, manifest)
                _validate_adapter_publication(pointer.adapter)
                current_found = True
            records.append((entry, generation, adapter_step, current, young))

        if pointer is not None and not current_found:
            raise RuntimeError(
                f"Optimizer pointer generation is missing: {pointer.generation}"
            )
        interrupted_generation: str | None = None
        if pointer_temps:
            temporary_pointer = pointer_temps[0][1]
            interrupted_generation = temporary_pointer.generation
            manifest = manifests.get(interrupted_generation)
            if manifest is None:
                raise RuntimeError(
                    "Interrupted optimizer pointer has no committed generation"
                )
            _validate_pointer_manifest(temporary_pointer, manifest)
            if pointer is not None and temporary_pointer.step <= pointer.step:
                raise RuntimeError(
                    "Interrupted optimizer pointer does not advance the committed "
                    "generation"
                )
        trash.extend(
            entry
            for entry in generations.iterdir()
            if entry.name.startswith(OPTIMIZER_TRASH_PREFIX)
            and now - entry.stat().st_mtime >= orphan_grace_s
        )

        for entry, generation, adapter_step, current, young in records:
            if current or adapter_step in retain_adapter_steps or young:
                protected_steps.add(adapter_step)
                continue

            with _generation_lease(
                path,
                generation,
                exclusive=True,
                nonblocking=True,
            ) as acquired:
                if not acquired:
                    protected_steps.add(adapter_step)
                    continue
                destination = generations / (
                    f"{OPTIMIZER_TRASH_PREFIX}{generation}-{uuid4().hex}"
                )
                if interrupted_generation == generation:
                    pointer_temps[0][0].unlink()
                    _fsync_directory(path)
                os.replace(entry, destination)
                os.utime(destination)
                trash.append(destination)
            _generation_lease_path(path, generation).unlink(missing_ok=True)

        live = {generation for entry, generation, *_ in records if entry.exists()}
        for lease in generations.iterdir():
            if not lease.name.startswith(OPTIMIZER_GENERATION_LEASE_PREFIX):
                continue
            generation = lease.name.removeprefix(OPTIMIZER_GENERATION_LEASE_PREFIX)
            _validate_generation_name(generation)
            if generation in live:
                continue
            with _generation_lease(
                path,
                generation,
                exclusive=True,
                nonblocking=True,
            ) as acquired:
                if acquired:
                    lease.unlink()
        if trash:
            _fsync_directory(generations)

    for entry in trash:
        shutil.rmtree(entry)
    return protected_steps


def prune_optimizer_generations(
    optimizer_state_path: str,
    *,
    retain_adapter_steps: set[int],
    orphan_grace_s: float = OPTIMIZER_ORPHAN_GRACE_S,
) -> set[int]:
    with optimizer_model_lease(optimizer_state_path):
        return _prune_optimizer_generations_locked(
            optimizer_state_path,
            retain_adapter_steps=retain_adapter_steps,
            orphan_grace_s=orphan_grace_s,
        )


@contextmanager
def optimizer_retention_lease(
    output_dir: str, retain_adapter_steps: set[int]
) -> Iterator[set[int]]:
    paths = tuple(
        f"{output_dir}/optimizer_states_{job_type}" for job_type in ("rl", "sft")
    )
    with optimizer_model_lease(paths[0]):
        protected = set(retain_adapter_steps)
        for path in paths:
            protected.update(
                _prune_optimizer_generations_locked(
                    path, retain_adapter_steps=protected
                )
            )
        yield protected


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
    pointer: OptimizerGenerationPointer | None | object = _POINTER_UNSET,
    verify_adapter_files: bool = True,
) -> OptimizerGenerationPointer | None:
    if pointer is _POINTER_UNSET:
        pointer = read_committed_optimizer_pointer(optimizer_state_path)
    if pointer is None:
        return None
    pointer = cast(OptimizerGenerationPointer, pointer)
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
    _validate_adapter_publication(pointer.adapter, verify_files=verify_adapter_files)
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
    adapter = read_adapter_publication(canonical, step=step, verify_files=True)
    if adapter is None:
        adapter = optimizer_adapter(canonical, step)
    if path == canonical:
        return adapter
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
        while (
            adapter := read_adapter_publication(
                staging_path,
                step=step,
                verify_files=False,
            )
        ) is None:
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


def _save_optimizer_state_locked(
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


def save_optimizer_state(
    runtime: Any,
    *,
    optimizer_state_path: str,
    step: int,
    adapter: OptimizerAdapter,
    model_lock_held: bool = False,
) -> None:
    with ExitStack() as leases:
        if not model_lock_held:
            optimizer_group_decision(
                runtime,
                lambda: leases.enter_context(
                    optimizer_model_lease(optimizer_state_path)
                ),
                operation="optimizer model lease acquisition",
            )
        _save_optimizer_state_locked(
            runtime,
            optimizer_state_path=optimizer_state_path,
            step=step,
            adapter=adapter,
        )


def _sibling_optimizer_owns_adapter(
    optimizer_state_path: str, adapter: OptimizerAdapter
) -> bool:
    current = Path(optimizer_state_path).absolute()
    for sibling in (
        current.parent / "optimizer_states_rl",
        current.parent / "optimizer_states_sft",
    ):
        if sibling == current or not sibling.exists():
            continue
        with _committed_generation_lease(sibling) as pointer:
            if pointer is None:
                continue
            manifest = _read_manifest(
                optimizer_generation_path(str(sibling), pointer.generation)
            )
            _validate_pointer_manifest(pointer, manifest)
            if pointer.adapter == adapter:
                return True
    return False


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
    with ExitStack() as leases:

        def select_generation() -> dict[str, Any] | None:
            runtime_sha256, layouts = _validated_runtime_layouts(runtime, records)
            adapter = _loaded_adapter(adapter_path, adapter_step)
            leased_pointer = leases.enter_context(
                _committed_generation_lease(Path(optimizer_state_path))
            )
            lineage_switch = (
                leased_pointer is None or leased_pointer.adapter != adapter
            ) and _sibling_optimizer_owns_adapter(optimizer_state_path, adapter)
            if lineage_switch:
                return None
            pointer = pin_optimizer_generation(
                optimizer_state_path,
                world_size=runtime.world_size,
                runtime_sha256=runtime_sha256,
                layout_sha256_by_rank=layouts,
                adapter=adapter,
                pointer=leased_pointer,
                verify_adapter_files=False,
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


def _scan_optimizer_transactions(
    path: Path,
) -> tuple[
    list[tuple[Path, OptimizerGenerationPointer]],
    list[tuple[Path, str, bool]],
]:
    pointer_temps: list[tuple[Path, OptimizerGenerationPointer]] = []
    allowed = {
        OPTIMIZER_POINTER,
        OPTIMIZER_WRITER_LOCK,
        OPTIMIZER_GENERATIONS_DIR,
        "uncommitted_generations",
    }
    for entry in sorted(path.iterdir()):
        if entry.name in allowed:
            continue
        if _POINTER_TEMP_RE.fullmatch(entry.name) is None or not entry.is_file():
            raise RuntimeError(
                "Ambiguous interrupted optimizer transaction; unexpected root "
                f"entry {entry}"
            )
        try:
            pointer = OptimizerGenerationPointer.model_validate_json(
                entry.read_text("utf-8")
            )
        except Exception as exc:
            raise RuntimeError(
                f"Invalid interrupted optimizer pointer: {entry}"
            ) from exc
        pointer_temps.append((entry, pointer))

    candidates: list[tuple[Path, str, bool]] = []
    generations = path / OPTIMIZER_GENERATIONS_DIR
    if not generations.exists():
        return pointer_temps, candidates
    if not generations.is_dir():
        raise RuntimeError(
            f"Optimizer generations path is not a directory: {generations}"
        )
    for entry in sorted(generations.iterdir()):
        name = entry.name
        if name.startswith(OPTIMIZER_GENERATION_LEASE_PREFIX):
            _validate_generation_name(
                name.removeprefix(OPTIMIZER_GENERATION_LEASE_PREFIX)
            )
            if not entry.is_file():
                raise RuntimeError(f"Invalid optimizer generation lease: {entry}")
            continue
        if name.startswith(OPTIMIZER_TRASH_PREFIX):
            if _TRASH_RE.fullmatch(name) is None or not entry.is_dir():
                raise RuntimeError(f"Invalid optimizer generation trash: {entry}")
            continue
        pending = name.startswith(".pending-")
        generation = name.removeprefix(".pending-") if pending else name
        if _GENERATION_RE.fullmatch(generation) is None or not entry.is_dir():
            raise RuntimeError(
                "Ambiguous interrupted optimizer transaction; unexpected generation "
                f"entry {entry}"
            )
        candidates.append((entry, generation, pending))
    return pointer_temps, candidates


def _optimizer_state_paths(output_dir: str, current: str) -> tuple[Path, ...]:
    selected = Path(current).absolute()
    candidates = {selected} | {
        (Path(output_dir) / f"optimizer_states_{kind}").absolute()
        for kind in ("rl", "sft")
    }
    return tuple(
        sorted(path for path in candidates if path == selected or path.exists())
    )


def _recover_uncommitted_initial_transaction(
    *,
    output_dir: str,
    optimizer_state_path: str,
) -> tuple[int, ...]:
    if get_step_from_dir(output_dir) != 1:
        return ()
    path = Path(optimizer_state_path).absolute()
    checkpoint = Path(get_step_checkpoint_dir(output_dir, 1)).absolute()
    roots = _optimizer_state_paths(output_dir, optimizer_state_path)
    with ExitStack() as locks:
        pointers = {root: locks.enter_context(_writer_lease(root)) for root in roots}
        pointer = pointers[path]
        if pointer is not None:
            return ()
        adapter = read_adapter_publication(checkpoint, step=1, verify_files=True)
        if adapter is None:
            return ()

        pointer_temps, candidates = _scan_optimizer_transactions(path)
        for sibling in roots:
            if sibling == path:
                continue
            sibling_temps, sibling_candidates = _scan_optimizer_transactions(sibling)
            sibling_pointer = pointers[sibling]
            if sibling_pointer is not None and sibling_pointer.adapter.step == 1:
                return ()
            if any(pointer.adapter.step == 1 for _, pointer in sibling_temps) or any(
                _generation_step(generation) == 1
                for _, generation, _ in sibling_candidates
            ):
                raise RuntimeError(
                    "Cannot recover interrupted initial optimizer transaction; "
                    f"sibling optimizer state may own checkpoint 0001: {sibling}"
                )
        if len(candidates) > 1:
            raise RuntimeError(
                "Ambiguous interrupted initial optimizer transaction; found "
                f"{len(candidates)} candidate generations"
            )
        manifest: OptimizerGenerationManifest | None = None
        if candidates:
            entry, generation, pending = candidates[0]
            if _generation_step(generation) != 1:
                raise RuntimeError(
                    "Ambiguous interrupted initial optimizer transaction; "
                    f"unexpected generation {generation}"
                )
            manifest_path = entry / OPTIMIZER_MANIFEST
            if not pending or manifest_path.exists():
                manifest = _read_manifest(entry)
                if (
                    manifest.generation != generation
                    or manifest.step != 1
                    or manifest.adapter != adapter
                ):
                    raise RuntimeError(
                        "Interrupted initial optimizer generation does not match "
                        f"the published adapter: {entry}"
                    )
        if len(pointer_temps) > 1:
            raise RuntimeError(
                "Ambiguous interrupted initial optimizer transaction; found "
                f"{len(pointer_temps)} temporary pointers"
            )
        if pointer_temps:
            if not candidates or candidates[0][2] or manifest is None:
                raise RuntimeError(
                    "Interrupted optimizer pointer has no committed generation"
                )
            expected = OptimizerGenerationPointer(
                generation=manifest.generation,
                step=manifest.step,
                adapter=manifest.adapter,
            )
            if pointer_temps[0][1] != expected:
                raise RuntimeError(
                    "Interrupted optimizer pointer does not match its generation"
                )
        if candidates and not locks.enter_context(
            _generation_lease(
                path,
                candidates[0][1],
                exclusive=True,
                nonblocking=True,
            )
        ):
            raise RuntimeError(
                "Interrupted initial optimizer generation is still in use: "
                f"{candidates[0][1]}"
            )
        if _allow_unpaired_resume() and (pointer_temps or candidates):
            raise RuntimeError(
                f"{ALLOW_UNPAIRED_MEGATRON_RESUME_ENV} cannot bypass an interrupted "
                "optimizer transaction"
            )
        if _allow_unpaired_resume():
            return ()

        tag = f"initial_step_0001_{adapter.sha256[:16]}"
        previous = Path(output_dir) / "unpaired_checkpoints" / tag / checkpoint.name
        if previous.exists():
            tag = f"{tag}_{uuid4().hex}"
        quarantine = path / "uncommitted_generations" / tag
        quarantine.mkdir(parents=True, exist_ok=True)
        if pointer_temps:
            pointer_temp = pointer_temps[0][0]
            destination = quarantine / pointer_temp.name
            if destination.exists():
                raise RuntimeError(f"Optimizer quarantine entry exists: {destination}")
            os.replace(pointer_temp, destination)
            _fsync_directory(path)
        if candidates:
            entry = candidates[0][0]
            destination = quarantine / entry.name
            if destination.exists():
                raise RuntimeError(f"Optimizer quarantine entry exists: {destination}")
            os.replace(entry, destination)
            _fsync_directory(quarantine)
            _fsync_directory(entry.parent)

        checkpoint_quarantine = Path(output_dir) / "unpaired_checkpoints" / tag
        checkpoint_quarantine.mkdir(parents=True, exist_ok=True)
        destination = checkpoint_quarantine / checkpoint.name
        if destination.exists():
            raise RuntimeError(f"Checkpoint quarantine entry exists: {destination}")
        os.replace(checkpoint, destination)
        _fsync_directory(checkpoint_quarantine)
        _fsync_directory(checkpoint.parent)
    return (1,)


def resolve_megatron_resume_step(
    *,
    output_dir: str,
    optimizer_state_path: str,
) -> MegatronResumeStep:
    latest_lora_step = get_step_from_dir(output_dir)
    with _committed_generation_lease(Path(optimizer_state_path)) as pointer:
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
            _validate_adapter_publication(pointer.adapter, verify_files=True)
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


def _resolve_model_resume_step(
    *, output_dir: str, optimizer_state_path: str
) -> MegatronResumeStep:
    paired = []
    for path in _optimizer_state_paths(output_dir, optimizer_state_path):
        if read_committed_optimizer_pointer(str(path)) is not None:
            paired.append(
                resolve_megatron_resume_step(
                    output_dir=output_dir,
                    optimizer_state_path=str(path),
                )
            )
    if paired:
        return max(paired, key=lambda info: info.step)
    return resolve_megatron_resume_step(
        output_dir=output_dir,
        optimizer_state_path=optimizer_state_path,
    )


def _prepare_megatron_resume_state_locked(
    *,
    output_dir: str,
    optimizer_state_path: str,
) -> MegatronResumeStep:
    recovered_steps = _recover_uncommitted_initial_transaction(
        output_dir=output_dir,
        optimizer_state_path=optimizer_state_path,
    )
    info = _resolve_model_resume_step(
        output_dir=output_dir,
        optimizer_state_path=optimizer_state_path,
    )
    if recovered_steps:
        info = info.model_copy(update={"quarantined_lora_steps": recovered_steps})
    if info.used_unpaired_override or info.latest_lora_step <= info.step:
        return info

    checkpoints_dir = Path(output_dir) / "checkpoints"
    quarantine_dir = (
        Path(output_dir)
        / "unpaired_checkpoints"
        / f"resume_from_{info.step:04d}_{int(time.time())}_{os.getpid()}"
    )
    to_move = [
        checkpoint_dir
        for checkpoint_dir in sorted(checkpoints_dir.iterdir())
        if checkpoint_dir.is_dir()
        and checkpoint_dir.name.isdigit()
        and int(checkpoint_dir.name) > info.step
    ]
    moved_steps: list[int] = []
    for checkpoint_dir in to_move:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        os.replace(checkpoint_dir, quarantine_dir / checkpoint_dir.name)
        moved_steps.append(int(checkpoint_dir.name))
    if moved_steps:
        _fsync_directory(checkpoints_dir)
        _fsync_directory(quarantine_dir)
    return info.model_copy(update={"quarantined_lora_steps": tuple(moved_steps)})


def prepare_megatron_resume_state(
    *,
    output_dir: str,
    optimizer_state_path: str,
) -> MegatronResumeStep:
    with optimizer_model_lease(optimizer_state_path):
        return _prepare_megatron_resume_state_locked(
            output_dir=output_dir,
            optimizer_state_path=optimizer_state_path,
        )


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
