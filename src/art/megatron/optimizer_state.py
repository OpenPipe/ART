from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import ExitStack, asynccontextmanager, contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterator, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from art.utils.safetensors import FileIdentity

if TYPE_CHECKING:
    from art.megatron.optimizer_archive import PreparedOptimizerArchive
    from art.megatron.portable_optimizer_archive import (
        PreparedPortableOptimizerArchive,
    )

from ..utils.get_model_step import get_step_from_dir
from ..utils.output_dirs import get_step_checkpoint_dir
from .tensor_snapshot import (
    PendingCpuSnapshot,
    PinnedCpuSnapshotBuilder,
    PinnedCpuSnapshotStager,
)
from .weights.rank_sharded_lora import (
    RANK_SHARDED_LORA_MANIFEST,
    CheckpointFile,
    detect_adapter_checkpoint_format,
    encode_rank_sharded_lora_manifest,
    rank_sharded_lora_checkpoint_files,
    read_rank_sharded_lora_checkpoint,
)

ALLOW_UNPAIRED_MEGATRON_RESUME_ENV = "ART_ALLOW_UNPAIRED_MEGATRON_RESUME"
OPTIMIZER_GENERATIONS_DIR = "generations"
OPTIMIZER_MANIFEST = "manifest.json"
OPTIMIZER_POINTER = "committed.json"
OPTIMIZER_POLICY_POINTER = "policy.json"
OPTIMIZER_MODEL_LOCK = ".optimizer.lock"
OPTIMIZER_WRITER_LOCK = ".writer.lock"
OPTIMIZER_GENERATION_LEASE_PREFIX = ".lease-"
OPTIMIZER_TRASH_PREFIX = ".trash-"
OPTIMIZER_ORPHAN_GRACE_S = 3600.0
ADAPTER_PUBLICATION_ACK = ".optimizer-published.json"
ADAPTER_LATEST_POINTER = "latest-adapter.json"
_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
_GENERATION_PATTERN = r"step-\d{8,}-[0-9a-f]{32}"
_GENERATION_RE = re.compile(f"^{_GENERATION_PATTERN}$")
_TRASH_RE = re.compile(f"^\\.trash-({_GENERATION_PATTERN})-[0-9a-f]{{32}}$")
_POINTER_TEMP_RE = re.compile(r"^\.committed\.json\.\d+\.[0-9a-f]{32}\.tmp$")
_POLICY_TEMP_RE = re.compile(r"^\.policy\.json\.\d+\.[0-9a-f]{32}\.tmp$")
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_POINTER_UNSET = object()
_STUB_OPTIMIZER_STATE = {"art_stub_optimizer": 1}
_SCHEDULE_PROVIDER_FIELDS = {
    "batch_p2p_comm",
    "batch_p2p_sync",
    "finalize_model_grads_func",
    "microbatch_group_size_per_vp_stage",
    "overlap_p2p_comm",
    "variable_seq_lengths",
}
# OptimStepRequest mutates these and the optimizer state persists their values.
_STEP_OPTIMIZER_CONFIG_FIELDS = {
    "adam_beta1",
    "adam_beta2",
    "adam_eps",
    "lr",
    "weight_decay",
}
OptimizerShardSerialization = Literal[
    "torch_pickle_v1",
    "art_safetensors_v1",
    "art_logical_safetensors_v1",
]


class _OptimizerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MegatronResumeStep(_OptimizerRecord):
    step: int
    latest_lora_step: int
    optimizer_step: int | None
    used_unpaired_override: bool = False
    quarantined_lora_steps: tuple[int, ...] = ()


def adapter_checkpoint_file_format(
    files: tuple[CheckpointFile, ...],
) -> Literal["canonical", "rank_sharded"]:
    names = tuple(file.name for file in files)
    if names == _ADAPTER_FILES:
        return "canonical"
    if (
        len(files) >= 3
        and names[:2] == ("adapter_config.json", RANK_SHARDED_LORA_MANIFEST)
        and names[2:]
        == tuple(f"shards/{index:08d}.bin" for index in range(len(files) - 2))
        and all(file.sha256 is not None for file in files)
    ):
        return "rank_sharded"
    raise ValueError("adapter checkpoint files do not identify a supported format")


class OptimizerAdapter(_OptimizerRecord):
    identity: str = Field(min_length=1)
    training_session_id: str = Field(min_length=1)
    step: int = Field(ge=0)
    generation_id: str = Field(pattern=f"^{_GENERATION_PATTERN}$")
    files: tuple[CheckpointFile, ...]

    @model_validator(mode="after")
    def _validate_files(self) -> "OptimizerAdapter":
        if _generation_step(self.generation_id) != self.step:
            raise ValueError("adapter generation ID and policy step must match")
        try:
            adapter_checkpoint_file_format(self.files)
        except ValueError as error:
            raise ValueError(
                "adapter manifest must cover every payload file once"
            ) from error
        return self


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
    layout_sha256: str = Field(pattern=_SHA256_PATTERN)
    sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    serialization: OptimizerShardSerialization = "torch_pickle_v1"
    logical_keys: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_logical_keys(self) -> "OptimizerShard":
        portable = self.serialization == "art_logical_safetensors_v1"
        if not portable and self.logical_keys:
            raise ValueError("only logical optimizer shards may declare logical keys")
        if self.logical_keys != tuple(sorted(set(self.logical_keys))):
            raise ValueError("optimizer shard logical keys must be sorted and unique")
        return self


class _PairedOptimizerRecord(_OptimizerRecord):
    step: int = Field(ge=0)
    adapter: OptimizerAdapter

    @model_validator(mode="after")
    def _validate_adapter_step(self) -> "_PairedOptimizerRecord":
        if self.step != self.adapter.step:
            raise ValueError("optimizer and adapter steps must match")
        return self


class _OptimizerGenerationRecord(_PairedOptimizerRecord):
    generation: str = Field(pattern=f"^{_GENERATION_PATTERN}$")

    @model_validator(mode="after")
    def _validate_generation_step(self) -> "_OptimizerGenerationRecord":
        if int(self.generation.split("-", 2)[1]) != self.step:
            raise ValueError("optimizer generation name and step must match")
        if self.generation != self.adapter.generation_id:
            raise ValueError("optimizer and adapter generation IDs must match")
        return self


class OptimizerGenerationManifest(_OptimizerGenerationRecord):
    format_version: Literal[3] = 3
    runtime_sha256: str = Field(pattern=_SHA256_PATTERN)
    optimizer_semantic_sha256: str | None = Field(
        default=None, pattern=_SHA256_PATTERN
    )
    topology: OptimizerTopology
    shards: tuple[OptimizerShard, ...]

    @model_validator(mode="after")
    def _validate_portable_semantics(self) -> "OptimizerGenerationManifest":
        portable = {
            shard.serialization == "art_logical_safetensors_v1"
            for shard in self.shards
        }
        if len(portable) != 1:
            raise ValueError("optimizer generation cannot mix physical formats")
        if portable.pop() != (self.optimizer_semantic_sha256 is not None):
            raise ValueError(
                "logical optimizer generations require a semantic fingerprint"
            )
        return self


class VerifiedOptimizerGeneration(_OptimizerRecord):
    """Exact optimizer generation authenticated before distributed loading."""

    generation: str = Field(pattern=f"^{_GENERATION_PATTERN}$")
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)


class OptimizerGenerationPointer(_OptimizerGenerationRecord):
    format_version: Literal[3] = 3


class OptimizerPolicyPointer(_OptimizerRecord):
    format_version: Literal[2] = 2
    policy_adapter: OptimizerAdapter
    optimizer_anchor: OptimizerGenerationPointer | None

    @model_validator(mode="after")
    def _validate_policy_alias(self) -> "OptimizerPolicyPointer":
        if self.policy_adapter.step == 0:
            raise ValueError("policy alias must advance beyond checkpoint 0")
        if self.optimizer_anchor is not None and (
            self.policy_adapter.step <= self.optimizer_anchor.step
        ):
            raise ValueError("policy alias must be newer than its optimizer anchor")
        return self


class CommittedOptimizerPolicy(_OptimizerRecord):
    policy_adapter: OptimizerAdapter
    state_adapter: OptimizerAdapter | None
    optimizer_anchor: OptimizerGenerationPointer | None


class OptimizerStateSnapshot(_OptimizerRecord):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    generation_id: str = Field(pattern=f"^{_GENERATION_PATTERN}$")
    step: int = Field(ge=1)
    rank: int = Field(ge=0)
    world_size: int = Field(gt=0)
    runtime_sha256: str = Field(pattern=_SHA256_PATTERN)
    optimizer_semantic_sha256: str | None = Field(
        default=None, pattern=_SHA256_PATTERN
    )
    layout_sha256: str = Field(pattern=_SHA256_PATTERN)
    topology: OptimizerTopology
    state_dict: Any

    @model_validator(mode="after")
    def _validate_identity(self) -> "OptimizerStateSnapshot":
        if _generation_step(self.generation_id) != self.step:
            raise ValueError("optimizer snapshot generation and step must match")
        if self.rank >= self.world_size or self.topology.world_size != self.world_size:
            raise ValueError("optimizer snapshot rank/topology mismatch")
        return self


def optimizer_shard_name(
    rank: int,
    world_size: int,
    serialization: OptimizerShardSerialization = "torch_pickle_v1",
) -> str:
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(
            f"Invalid optimizer shard rank {rank} for world size {world_size}"
        )
    suffix = "pt" if serialization == "torch_pickle_v1" else "safetensors"
    return f"{rank + 1:02d}-of-{world_size:02d}.{suffix}"


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


def _adapter_generation_lease_path(output_dir: str | Path, generation: str) -> Path:
    _validate_generation_name(generation)
    return Path(output_dir).absolute() / "megatron_runtime" / "leases" / generation


@contextmanager
def adapter_generation_lease(adapter: OptimizerAdapter) -> Iterator[None]:
    path = _adapter_generation_lease_path(
        Path(adapter.identity).absolute().parent.parent,
        adapter.generation_id,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lease_file:
        fcntl.flock(lease_file.fileno(), fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(lease_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _adapter_retention_leases(
    output_dir: str, protected_steps: set[int]
) -> Iterator[set[int]]:
    checkpoints = Path(output_dir) / "checkpoints"
    with ExitStack() as leases:
        if checkpoints.is_dir():
            for checkpoint in checkpoints.iterdir():
                if (
                    not checkpoint.is_dir()
                    or not checkpoint.name.isdigit()
                    or (step := int(checkpoint.name)) in protected_steps
                ):
                    continue
                publication = read_adapter_publication(
                    checkpoint, step=step, verify_files=False
                )
                generation = (
                    publication.generation_id
                    if publication is not None
                    else _initial_generation_id(checkpoint, step)
                )
                path = _adapter_generation_lease_path(output_dir, generation)
                path.parent.mkdir(parents=True, exist_ok=True)
                lease = leases.enter_context(path.open("a+b"))
                try:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    protected_steps.add(step)
                else:
                    leases.callback(fcntl.flock, lease.fileno(), fcntl.LOCK_UN)
        yield protected_steps


@contextmanager
def optimizer_model_lease(optimizer_state_path: str | Path) -> Iterator[None]:
    with _optimizer_model_lock_path(optimizer_state_path).open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@asynccontextmanager
async def async_optimizer_model_lease(
    optimizer_state_path: str | Path,
) -> AsyncIterator[None]:
    with _optimizer_model_lock_path(optimizer_state_path).open("a+b") as lock_file:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _optimizer_model_lock_path(optimizer_state_path: str | Path) -> Path:
    model_root = Path(optimizer_state_path).absolute().parent
    model_root.mkdir(parents=True, exist_ok=True)
    return model_root / OPTIMIZER_MODEL_LOCK


def optimizer_shard_path(
    generation_path: Path,
    *,
    rank: int,
    world_size: int,
    serialization: OptimizerShardSerialization = "torch_pickle_v1",
) -> Path:
    return generation_path / optimizer_shard_name(rank, world_size, serialization)


def _adapter_checkpoint_files(path: str | Path) -> tuple[Path, tuple[Path, ...]]:
    adapter_path = Path(path)
    format = detect_adapter_checkpoint_format(adapter_path)
    names = (
        _ADAPTER_FILES
        if format == "canonical"
        else tuple(
            file.name for file in read_rank_sharded_lora_checkpoint(adapter_path).files
        )
    )
    files = tuple(adapter_path / name for name in names)
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        raise RuntimeError(f"Adapter checkpoint is incomplete; missing {missing}")
    return adapter_path, files


def _adapter_file_records(path: str | Path) -> tuple[CheckpointFile, ...]:
    adapter_path, files = _adapter_checkpoint_files(path)
    if detect_adapter_checkpoint_format(adapter_path) == "rank_sharded":
        return read_rank_sharded_lora_checkpoint(adapter_path, verify_files=True).files
    return tuple(
        CheckpointFile(
            name=file.relative_to(adapter_path).as_posix(),
            size_bytes=file.stat().st_size,
            sha256=_file_sha256(file),
        )
        for file in files
    )


def validate_adapter_checkpoint(
    path: str | Path,
    adapter: OptimizerAdapter,
    *,
    verify_files: bool,
) -> None:
    root = Path(path).absolute()
    physical_format = detect_adapter_checkpoint_format(root)
    if adapter_checkpoint_file_format(adapter.files) != physical_format:
        raise RuntimeError("adapter manifest names another physical format")
    if physical_format == "rank_sharded":
        checkpoint = read_rank_sharded_lora_checkpoint(root, verify_files=verify_files)
        manifest = checkpoint.manifest
        if (
            manifest.training_session_id,
            manifest.generation_id,
            manifest.step,
        ) != (
            adapter.training_session_id,
            adapter.generation_id,
            adapter.step,
        ):
            raise RuntimeError("rank-sharded adapter manifest names another generation")
        if checkpoint.files != adapter.files:
            raise RuntimeError("rank-sharded adapter physical coverage changed")
        return
    _adapter_root, payloads = _adapter_checkpoint_files(root)
    if tuple(payload.relative_to(root).as_posix() for payload in payloads) != tuple(
        record.name for record in adapter.files
    ) or any(
        payload.stat().st_size != record.size_bytes
        for payload, record in zip(payloads, adapter.files, strict=True)
    ):
        raise RuntimeError(
            "canonical adapter physical coverage is incomplete or changed"
        )
    if verify_files and any(
        record.sha256 is not None and _file_sha256(payload) != record.sha256
        for payload, record in zip(payloads, adapter.files, strict=True)
    ):
        raise RuntimeError("canonical adapter file digest changed")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as value:
        while chunk := value.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _initial_generation_id(path: str | Path, step: int) -> str:
    suffix = hashlib.sha256(str(Path(path).absolute()).encode()).hexdigest()[:32]
    return f"step-{step:08d}-{suffix}"


def optimizer_adapter(
    path: str | Path,
    step: int,
    *,
    training_session_id: str = "legacy",
    generation_id: str | None = None,
) -> OptimizerAdapter:
    if step < 0:
        raise ValueError(f"Adapter step must be non-negative, got {step}")
    identity = str(Path(path).absolute())
    return OptimizerAdapter(
        identity=identity,
        training_session_id=training_session_id,
        step=step,
        generation_id=generation_id or _initial_generation_id(identity, step),
        files=_adapter_file_records(identity),
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
    staging_path: str | Path,
    *,
    step: int,
    files: tuple[CheckpointFile, ...],
    training_session_id: str = "legacy",
    generation_id: str | None = None,
) -> OptimizerAdapter:
    staging = Path(staging_path).absolute()
    canonical = canonical_adapter_path(staging, step)
    if canonical.exists():
        raise RuntimeError(f"Refusing to replace canonical adapter {canonical}")
    records = files
    _adapter_path, paths = _adapter_checkpoint_files(staging)
    if tuple(path.relative_to(staging).as_posix() for path in paths) != tuple(
        record.name for record in records
    ):
        raise RuntimeError("adapter file identity has invalid coverage")
    adapter = OptimizerAdapter(
        identity=str(canonical),
        training_session_id=training_session_id,
        step=step,
        generation_id=generation_id or _initial_generation_id(canonical, step),
        files=records,
    )
    validate_adapter_checkpoint(staging, adapter, verify_files=False)
    for path in paths:
        with path.open("rb") as adapter_file:
            os.fsync(adapter_file.fileno())
    for parent in sorted(
        {path.parent for path in paths}, key=lambda path: len(path.parts)
    ):
        if parent != staging:
            _fsync_directory(parent)
    _fsync_directory(staging)
    _write_model_atomic(staging / ADAPTER_PUBLICATION_ACK, adapter)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, canonical)
    _fsync_directory(canonical.parent)
    _write_model_atomic(
        canonical.parent.parent / "megatron_runtime" / ADAPTER_LATEST_POINTER,
        adapter,
    )
    return adapter


def acknowledge_materialized_adapter(
    adapter_path: str | Path,
    *,
    step: int,
    training_session_id: str,
    generation_id: str,
    files: tuple[CheckpointFile, ...],
) -> OptimizerAdapter:
    """Attach exact learner identity to an already verified adapter tree."""
    path = Path(adapter_path).absolute()
    adapter = OptimizerAdapter(
        identity=str(path),
        training_session_id=training_session_id,
        step=step,
        generation_id=generation_id,
        files=files,
    )
    validate_adapter_checkpoint(path, adapter, verify_files=False)
    acknowledgment = path / ADAPTER_PUBLICATION_ACK
    if acknowledgment.exists():
        if read_adapter_publication(path, step=step, verify_files=False) != adapter:
            raise RuntimeError("materialized adapter acknowledgement changed identity")
    else:
        _write_model_atomic(acknowledgment, adapter)
    return adapter


def read_latest_adapter_pointer(output_dir: str | Path) -> OptimizerAdapter | None:
    pointer = Path(output_dir) / "megatron_runtime" / ADAPTER_LATEST_POINTER
    if not pointer.exists():
        return None
    try:
        adapter = OptimizerAdapter.model_validate_json(pointer.read_text("utf-8"))
    except Exception as error:
        raise RuntimeError(f"Invalid adapter generation pointer: {pointer}") from error
    _validate_adapter_publication(adapter, verify_files=True)
    return adapter


def read_adapter_publication(
    adapter_path: str | Path,
    *,
    step: int,
    verify_files: bool = True,
) -> OptimizerAdapter | None:
    canonical = _canonical_adapter_path(adapter_path, step)
    acknowledgment = canonical / ADAPTER_PUBLICATION_ACK
    try:
        payload = acknowledgment.read_text("utf-8")
    except FileNotFoundError:
        return None
    try:
        adapter = OptimizerAdapter.model_validate_json(payload)
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
        validate_adapter_checkpoint(canonical, adapter, verify_files=True)
    else:
        validate_adapter_checkpoint(canonical, adapter, verify_files=False)
    return adapter


def validate_adapter_manifest(adapter: OptimizerAdapter) -> None:
    """Validate an immutable adapter without rescanning payload bytes."""
    if (
        read_adapter_publication(
            adapter.identity,
            step=adapter.step,
            verify_files=False,
        )
        != adapter
    ):
        raise RuntimeError("adapter manifest is not the acknowledged publication")
    validate_adapter_checkpoint(adapter.identity, adapter, verify_files=False)


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


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _remove_abandoned_adapter_staging(staging: Path) -> None:
    if staging.is_symlink() or (staging.exists() and not staging.is_dir()):
        raise RuntimeError(f"Adapter staging path is not a directory: {staging}")
    if staging.exists():
        shutil.rmtree(staging)
        _fsync_directory(staging.parent)


def discard_uncommitted_adapter_staging(
    staging_path: str | Path,
    *,
    step: int,
    training_session_id: str,
    generation_id: str,
) -> bool:
    """Remove only one fenced staging generation; preserve an exact commit."""
    staging = Path(staging_path).absolute()
    canonical = canonical_adapter_path(staging, step)
    if staging.name != generation_id:
        raise RuntimeError("Adapter staging path identifies another generation")
    staging.parent.mkdir(parents=True, exist_ok=True)
    lock_path = staging.with_name(f".{staging.name}.lock")
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        existing = read_adapter_publication(canonical, step=step)
        if existing is not None:
            if (
                existing.training_session_id,
                existing.generation_id,
            ) != (training_session_id, generation_id):
                raise RuntimeError("Committed adapter generation changed identity")
            return False
        existed = staging.exists()
        _remove_abandoned_adapter_staging(staging)
        return existed


@contextmanager
def adapter_publication_transaction(
    staging_path: str | Path,
    *,
    step: int,
    training_session_id: str,
    generation_id: str,
) -> Iterator[tuple[Path, OptimizerAdapter | None]]:
    """Serialize one adapter generation and rebuild abandoned staging state."""
    staging = Path(staging_path).absolute()
    canonical = canonical_adapter_path(staging, step)
    if staging.name != generation_id:
        raise RuntimeError("Adapter staging path identifies another generation")
    staging.parent.mkdir(parents=True, exist_ok=True)
    lock_path = staging.with_name(f".{staging.name}.lock")
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        existing = read_adapter_publication(canonical, step=step)
        if existing is not None:
            if (
                existing.training_session_id,
                existing.step,
                existing.generation_id,
            ) != (training_session_id, step, generation_id):
                raise RuntimeError("Committed adapter generation changed identity")
            yield staging, existing
            return
        _remove_abandoned_adapter_staging(staging)
        try:
            yield staging, None
            published = read_adapter_publication(canonical, step=step)
            if published is None or (
                published.training_session_id,
                published.step,
                published.generation_id,
            ) != (training_session_id, step, generation_id):
                raise RuntimeError("Adapter transaction did not publish its generation")
        except BaseException as error:
            try:
                _remove_abandoned_adapter_staging(staging)
            except BaseException as cleanup_error:
                error.add_note(
                    "adapter staging cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise


def link_adapter_generation(
    source_path: str | Path,
    *,
    source_step: int,
    staging_path: str | Path,
    step: int,
    training_session_id: str,
    generation_id: str,
) -> OptimizerAdapter:
    """Publish immutable adapter bytes under a new learner identity without copying."""
    source = read_adapter_publication(source_path, step=source_step)
    if source is None:
        raise RuntimeError("loaded checkpoint has no immutable adapter publication")
    with adapter_publication_transaction(
        staging_path,
        step=step,
        training_session_id=training_session_id,
        generation_id=generation_id,
    ) as (staging, existing):
        if existing is not None:
            return existing
        staging.mkdir(parents=True)
        source_root = Path(source.identity)
        if adapter_checkpoint_file_format(source.files) == "rank_sharded":
            source_checkpoint = read_rank_sharded_lora_checkpoint(
                source_root, verify_files=True
            )
            manifest = type(source_checkpoint.manifest).model_validate(
                {
                    **source_checkpoint.manifest.model_dump(mode="json"),
                    "training_session_id": training_session_id,
                    "generation_id": generation_id,
                    "step": step,
                }
            )
            files = rank_sharded_lora_checkpoint_files(manifest)
            for record in files:
                target = staging / record.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if record.name == RANK_SHARDED_LORA_MANIFEST:
                    target.write_bytes(encode_rank_sharded_lora_manifest(manifest))
                else:
                    os.link(source_root / record.name, target)
        else:
            files = source.files
            for record in files:
                os.link(source_root / record.name, staging / record.name)
        adapter = publish_adapter_checkpoint(
            staging,
            step=step,
            files=files,
            training_session_id=training_session_id,
            generation_id=generation_id,
        )
        if adapter.files != files:
            raise RuntimeError("cloned adapter file coverage changed")
        return adapter


def _serialized_model_json(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), sort_keys=True) + "\n"


def optimizer_generation_nbytes(manifest: OptimizerGenerationManifest) -> int:
    return sum(shard.size_bytes for shard in manifest.shards) + len(
        _serialized_model_json(manifest).encode("utf-8")
    )


def _write_model_atomic(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            output.write(_serialized_model_json(model))
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


def _read_policy_pointer(path: Path) -> OptimizerPolicyPointer | None:
    policy_path = path / OPTIMIZER_POLICY_POINTER
    if policy_path.is_file():
        try:
            return OptimizerPolicyPointer.model_validate_json(
                policy_path.read_text("utf-8")
            )
        except Exception as exc:
            raise RuntimeError(
                f"Invalid optimizer policy pointer: {policy_path}"
            ) from exc
    if policy_path.exists():
        raise RuntimeError(
            f"Invalid optimizer policy pointer: {policy_path} is not a file"
        )
    return None


def _resolve_policy_pointer(
    path: Path,
    pointer: OptimizerGenerationPointer | None,
) -> OptimizerPolicyPointer | None:
    policy = _read_policy_pointer(path)
    if policy is None:
        return None
    if policy.optimizer_anchor != pointer:
        if pointer is not None and pointer.step >= policy.policy_adapter.step:
            return None
        raise RuntimeError(
            "Optimizer policy pointer lost or changed its optimizer anchor: "
            f"policy={policy.model_dump()}, "
            f"current={pointer.model_dump() if pointer else None}"
        )
    _validate_adapter_publication(policy.policy_adapter, verify_files=True)
    if pointer is not None and any(
        not os.path.samefile(
            Path(policy.policy_adapter.identity) / name,
            Path(pointer.adapter.identity) / name,
        )
        for name in _ADAPTER_FILES
    ):
        raise RuntimeError("Optimizer policy alias does not reuse its anchor payload")
    expected = Path(
        get_step_checkpoint_dir(str(path.absolute().parent), policy.policy_adapter.step)
    ).absolute()
    if policy.policy_adapter.identity != str(expected):
        raise RuntimeError("Optimizer policy does not identify a canonical checkpoint")
    return policy


def _committed_policy(
    path: Path,
    pointer: OptimizerGenerationPointer | None,
    *,
    initial_adapter_path: str,
) -> CommittedOptimizerPolicy:
    if policy := _resolve_policy_pointer(path, pointer):
        return CommittedOptimizerPolicy(
            policy_adapter=policy.policy_adapter,
            state_adapter=None if pointer is None else pointer.adapter,
            optimizer_anchor=pointer,
        )
    if pointer is not None:
        return CommittedOptimizerPolicy(
            policy_adapter=pointer.adapter,
            state_adapter=pointer.adapter,
            optimizer_anchor=pointer,
        )
    if (
        Path(initial_adapter_path).absolute()
        != Path(get_step_checkpoint_dir(str(path.absolute().parent), 0)).absolute()
    ):
        raise RuntimeError("Initial optimizer policy must use canonical checkpoint 0")
    initial = optimizer_adapter(initial_adapter_path, 0)
    return CommittedOptimizerPolicy(
        policy_adapter=initial,
        state_adapter=None,
        optimizer_anchor=None,
    )


def resolve_committed_optimizer_policy(
    optimizer_state_path: str,
    *,
    initial_adapter_path: str,
) -> CommittedOptimizerPolicy:
    path = Path(optimizer_state_path)
    with _committed_generation_lease(path) as pointer:
        if pointer is not None:
            generation_path = optimizer_generation_path(
                optimizer_state_path, pointer.generation
            )
            manifest = _read_manifest(generation_path)
            _validate_pointer_manifest(pointer, manifest)
            _validate_generation_files(generation_path, manifest, local_rank=None)
            _validate_adapter_publication(pointer.adapter, verify_files=True)
        return _committed_policy(
            path,
            pointer,
            initial_adapter_path=initial_adapter_path,
        )


def commit_optimizer_policy_advance(
    optimizer_state_path: str,
    *,
    initial_adapter_path: str,
    expected_step: int,
    adapter: OptimizerAdapter,
) -> OptimizerPolicyPointer:
    path = Path(optimizer_state_path)
    with _writer_lease(path) as pointer:
        current = _committed_policy(
            path,
            pointer,
            initial_adapter_path=initial_adapter_path,
        )
        if current.policy_adapter.step != expected_step:
            raise RuntimeError(
                "Stale no-op policy writer: "
                f"expected={expected_step}, current={current.policy_adapter.step}"
            )
        if adapter.step != expected_step + 1:
            raise RuntimeError("Policy checkpoint must advance exactly one step")
        if any(
            not os.path.samefile(
                Path(current.policy_adapter.identity) / name,
                Path(adapter.identity) / name,
            )
            for name in _ADAPTER_FILES
        ):
            raise RuntimeError(
                "No-op policy checkpoint must reuse immutable adapter payloads"
            )
        _validate_adapter_publication(adapter, verify_files=True)
        policy = OptimizerPolicyPointer(
            policy_adapter=adapter,
            optimizer_anchor=pointer,
        )
        _write_model_atomic(path / OPTIMIZER_POLICY_POINTER, policy)
        return policy


def read_committed_optimizer_pointer(
    optimizer_state_path: str,
) -> OptimizerGenerationPointer | None:
    return _read_pointer(Path(optimizer_state_path))


def read_optimizer_generation_manifest(
    optimizer_state_path: str, generation: str
) -> OptimizerGenerationManifest:
    return _read_manifest(optimizer_generation_path(optimizer_state_path, generation))


def verify_optimizer_generation(
    optimizer_state_path: str, generation: str
) -> VerifiedOptimizerGeneration:
    """Authenticate one immutable generation once before rank-local reads."""
    with optimizer_generation_lease(optimizer_state_path, generation) as pointer:
        generation_path = optimizer_generation_path(optimizer_state_path, generation)
        manifest, manifest_sha256 = _read_manifest_identity(generation_path)
        _validate_pointer_manifest(pointer, manifest)
        verify_optimizer_generation_manifest(optimizer_state_path, manifest)
    return VerifiedOptimizerGeneration(
        generation=generation,
        manifest_sha256=manifest_sha256,
    )


def verify_optimizer_generation_manifest(
    optimizer_state_path: str, manifest: OptimizerGenerationManifest
) -> None:
    """Verify every byte before adopting a generation after an ambiguous write."""
    generation_path = optimizer_generation_path(
        optimizer_state_path, manifest.generation
    )
    for shard in _validate_generation_files(generation_path, manifest, local_rank=None):
        if shard.sha256 is None:
            raise RuntimeError("Optimizer shard has no immutable digest")
        path = optimizer_shard_path(
            generation_path,
            rank=shard.rank,
            world_size=manifest.topology.world_size,
            serialization=shard.serialization,
        )
        if _file_sha256(path) != shard.sha256:
            raise RuntimeError(f"Optimizer shard digest mismatch: {path.name}")


def read_committed_optimizer_step(optimizer_state_path: str) -> int | None:
    pointer = read_committed_optimizer_pointer(optimizer_state_path)
    return None if pointer is None else pointer.step


def read_committed_optimizer_adapter_step(optimizer_state_path: str) -> int | None:
    pointer = read_committed_optimizer_pointer(optimizer_state_path)
    return None if pointer is None else pointer.adapter.step


def _read_manifest(generation_path: Path) -> OptimizerGenerationManifest:
    return _read_manifest_identity(generation_path)[0]


def _read_manifest_identity(
    generation_path: Path,
) -> tuple[OptimizerGenerationManifest, str]:
    manifest_path = generation_path / OPTIMIZER_MANIFEST
    try:
        encoded = manifest_path.read_bytes()
        manifest = OptimizerGenerationManifest.model_validate_json(encoded)
    except Exception as exc:
        raise RuntimeError(
            f"Invalid optimizer generation manifest: {manifest_path}"
        ) from exc
    return manifest, hashlib.sha256(encoded).hexdigest()


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
    optimizer_semantic_sha256: str | None = None,
    world_size: int,
    shards: list[OptimizerShard],
    topology: OptimizerTopology | None = None,
) -> OptimizerGenerationManifest:
    manifest = OptimizerGenerationManifest(
        generation=generation,
        step=step,
        adapter=adapter,
        runtime_sha256=runtime_sha256,
        optimizer_semantic_sha256=optimizer_semantic_sha256,
        topology=topology or current_optimizer_topology(world_size),
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
        optimizer_shard_name(
            shard.rank, manifest.topology.world_size, shard.serialization
        )
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
        name = optimizer_shard_name(
            shard.rank, manifest.topology.world_size, shard.serialization
        )
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
        yield _recover_optimizer_pointer_locked(path, pointer)


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


@contextmanager
def optimizer_generation_lease(
    optimizer_state_path: str, generation: str
) -> Iterator[OptimizerGenerationPointer]:
    path = Path(optimizer_state_path)
    with _generation_lease(path, generation, exclusive=False) as acquired:
        if not acquired:
            raise RuntimeError(f"Could not lease optimizer generation {generation}")
        manifest = _read_manifest(
            optimizer_generation_path(optimizer_state_path, generation)
        )
        if manifest.generation != generation:
            raise RuntimeError("Optimizer generation manifest identity mismatch")
        yield OptimizerGenerationPointer(
            generation=manifest.generation,
            step=manifest.step,
            adapter=manifest.adapter,
        )


def commit_optimizer_generation(
    optimizer_state_path: str,
    manifest: OptimizerGenerationManifest,
    *,
    expected_pointer: OptimizerGenerationPointer | None,
    expected_policy_step: int | None = None,
    initial_adapter_path: str | None = None,
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
        if expected_policy_step is not None:
            if initial_adapter_path is None:
                raise ValueError("initial_adapter_path is required for lineage checks")
            current_policy = _committed_policy(
                path,
                current_pointer,
                initial_adapter_path=initial_adapter_path,
            )
            if current_policy.policy_adapter.step != expected_policy_step:
                raise RuntimeError(
                    "Stale optimizer writer: policy lineage changed before publication; "
                    f"expected={expected_policy_step}, "
                    f"current={current_policy.policy_adapter.step}"
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
        policy_path = path / OPTIMIZER_POLICY_POINTER
        if policy_path.exists():
            policy_path.unlink()
            _fsync_directory(path)
    return committed


def discard_uncommitted_optimizer_generation(
    optimizer_state_path: str, generation: str
) -> bool:
    """Remove one fenced pending generation without touching a committed one."""
    path = Path(optimizer_state_path)
    pending = optimizer_pending_generation_path(optimizer_state_path, generation)
    committed = optimizer_generation_path(optimizer_state_path, generation)
    trash: Path | None = None
    with _root_lease(path, fcntl.LOCK_EX):
        if committed.exists():
            return False
        if not pending.exists():
            return False
        if pending.is_symlink() or not pending.is_dir():
            raise RuntimeError(f"Optimizer pending path is not a directory: {pending}")
        trash = pending.with_name(f"{OPTIMIZER_TRASH_PREFIX}{generation}-{uuid4().hex}")
        os.replace(pending, trash)
        _fsync_directory(pending.parent)
    shutil.rmtree(trash)
    return True


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
        pointer_temps, candidates = _scan_optimizer_transactions(path)
        policy = _resolve_policy_pointer(path, pointer)
        if policy is not None:
            protected_steps.add(policy.policy_adapter.step)
        if not generations.exists():
            if pointer is not None:
                raise RuntimeError(
                    "Optimizer pointer exists without a generations directory: "
                    f"{generations}"
                )
            if pointer_temps:
                raise RuntimeError(
                    "Interrupted optimizer pointer has no committed generation"
                )
            return protected_steps
        if not generations.is_dir():
            raise RuntimeError(
                f"Optimizer generations path is not a directory: {generations}"
            )

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


def delete_optimizer_generation(
    optimizer_state_path: str,
    generation: str,
    *,
    replacement_generation: str | None,
) -> None:
    """Delete one immutable generation without leaving a dangling root pointer."""
    _validate_generation_name(generation)
    if replacement_generation == generation:
        raise ValueError("optimizer replacement must differ from deleted generation")
    if replacement_generation is not None:
        _validate_generation_name(replacement_generation)
    path = Path(optimizer_state_path)
    if not path.exists():
        return
    trash: list[Path] = []
    with optimizer_model_lease(path), _writer_lease(path) as current:
        target = optimizer_generation_path(optimizer_state_path, generation)
        replacement = (
            _generation_pointer(optimizer_state_path, replacement_generation)
            if replacement_generation is not None
            else None
        )
        if current is not None and current.generation == generation:
            pointer_path = path / OPTIMIZER_POINTER
            if replacement is None:
                pointer_path.unlink()
                _fsync_directory(path)
            else:
                _write_model_atomic(pointer_path, replacement)
        policy = _read_policy_pointer(path)
        if policy is not None and (
            policy.policy_adapter.generation_id == generation
            or (
                policy.optimizer_anchor is not None
                and policy.optimizer_anchor.generation == generation
            )
        ):
            (path / OPTIMIZER_POLICY_POINTER).unlink()
            _fsync_directory(path)
        if target.exists():
            with _generation_lease(path, generation, exclusive=True) as acquired:
                if not acquired:
                    raise RuntimeError(
                        f"Could not exclusively lease optimizer generation {generation}"
                    )
                destination = target.parent / (
                    f"{OPTIMIZER_TRASH_PREFIX}{generation}-{uuid4().hex}"
                )
                os.replace(target, destination)
                trash.append(destination)
                _fsync_directory(target.parent)
            _generation_lease_path(path, generation).unlink(missing_ok=True)
        trash.extend(
            entry
            for entry in (path / OPTIMIZER_GENERATIONS_DIR).glob(
                f"{OPTIMIZER_TRASH_PREFIX}{generation}-*"
            )
            if entry.is_dir() and entry not in trash
        )
    for entry in trash:
        shutil.rmtree(entry)


def delete_adapter_generation(
    adapter_path: str,
    *,
    step: int,
    generation: str,
    replacement_adapter_path: str | None,
    replacement_step: int | None,
) -> None:
    """Delete one canonical adapter after all logical and physical leases end."""
    if (replacement_adapter_path is None) != (replacement_step is None):
        raise ValueError("replacement adapter path and step must be paired")
    adapter = read_adapter_publication(adapter_path, step=step)
    path = Path(adapter_path).absolute()
    if adapter is None:
        if path.exists():
            raise RuntimeError(f"Unrecognized adapter generation path: {path}")
        for trash in path.parent.glob(f".trash-{path.name}-*"):
            if trash.is_dir():
                shutil.rmtree(trash)
        return
    if adapter.generation_id != generation:
        raise RuntimeError("adapter generation identity changed before deletion")
    replacement = (
        read_adapter_publication(replacement_adapter_path, step=replacement_step)
        if replacement_adapter_path is not None and replacement_step is not None
        else None
    )
    if replacement_adapter_path is not None and replacement is None:
        raise RuntimeError("replacement adapter is not durably published")
    lease_path = _adapter_generation_lease_path(path.parent.parent, generation)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    with lease_path.open("a+b") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
        latest_path = path.parent.parent / "megatron_runtime" / ADAPTER_LATEST_POINTER
        latest = read_latest_adapter_pointer(path.parent.parent)
        if latest is not None and latest.generation_id == generation:
            if replacement is None:
                latest_path.unlink()
                _fsync_directory(latest_path.parent)
            else:
                _write_model_atomic(latest_path, replacement)
        trash = path.with_name(f".trash-{path.name}-{uuid4().hex}")
        os.replace(path, trash)
        _fsync_directory(path.parent)
    lease_path.unlink(missing_ok=True)
    shutil.rmtree(trash)


def _generation_pointer(
    optimizer_state_path: str, generation: str
) -> OptimizerGenerationPointer:
    manifest = _read_manifest(
        optimizer_generation_path(optimizer_state_path, generation)
    )
    if manifest.generation != generation:
        raise RuntimeError("Optimizer generation manifest identity mismatch")
    return OptimizerGenerationPointer(
        generation=manifest.generation,
        step=manifest.step,
        adapter=manifest.adapter,
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
        with _adapter_retention_leases(output_dir, protected):
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
    shard = ordered[rank]
    return generation_path / optimizer_shard_name(
        shard.rank, world_size, shard.serialization
    )


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


def _canonical_runtime_json(value: Any) -> Any:
    if isinstance(value, dict):
        keys = tuple(value)
        try:
            supported = all(
                key is None or isinstance(key, (str, int, float, bool)) for key in keys
            )
            if supported:
                sorted(keys)
        except TypeError:
            supported = False
        if supported:
            return {key: _canonical_runtime_json(item) for key, item in value.items()}
        return [
            "__art_typed_mapping__",
            [
                [_type_identity(key), repr(key), _canonical_runtime_json(item)]
                for key, item in sorted(
                    value.items(),
                    key=lambda pair: (_type_identity(pair[0]), repr(pair[0])),
                )
            ],
        ]
    if isinstance(value, (list, tuple)):
        return [_canonical_runtime_json(item) for item in value]
    if isinstance(value, set):
        return sorted(
            (_canonical_runtime_json(item) for item in value),
            key=repr,
        )
    if isinstance(value, BaseModel):
        return _canonical_runtime_json(value.model_dump(mode="json"))
    return value


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _canonical_runtime_json(value),
        default=_runtime_json_default,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _public_fields(value: object, *, exclude: set[str] | None = None) -> dict[str, Any]:
    exclude = exclude or set()
    return {
        key: item
        for key, item in sorted(vars(value).items())
        if not key.startswith("_") and key not in exclude
    }


def _model_runtime_sha256(runtime: Any) -> str:
    if runtime.optimizer_runtime_sha256 is not None:
        return runtime.optimizer_runtime_sha256
    runtime.optimizer_runtime_sha256 = _json_sha256(
        {
            "model_support": runtime.model_support_spec,
            "provider": {
                "type": _type_identity(runtime.provider),
                "fields": _public_fields(
                    runtime.provider,
                    exclude=_SCHEDULE_PROVIDER_FIELDS,
                ),
            },
            "optimizer": _type_identity(runtime.optimizer),
            "optimizer_config": _public_fields(
                runtime.optimizer_config,
                exclude=_STEP_OPTIMIZER_CONFIG_FIELDS,
            ),
            "compile": runtime.transformer_layers_compiled,
            "topology": current_optimizer_topology(runtime.world_size),
            "torch": torch.__version__,
        }
    )
    return runtime.optimizer_runtime_sha256


def trainer_rank_optimizer_semantic_sha256(runtime: Any) -> str:
    """Return the topology-neutral logical runtime identity for exact resume."""
    value = getattr(runtime, "optimizer_semantic_sha256", None)
    if not isinstance(value, str) or re.fullmatch(_SHA256_PATTERN, value) is None:
        raise RuntimeError(
            "TrainerRank optimizer durability requires a logical runtime fingerprint"
        )
    return value


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


def _stage_optimizer_value(value: Any, stager: PinnedCpuSnapshotBuilder) -> Any:
    if isinstance(value, torch.Tensor):
        return stager.stage(value)
    if isinstance(value, dict):
        return {
            _stage_optimizer_value(key, stager): _stage_optimizer_value(item, stager)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_stage_optimizer_value(item, stager) for item in value]
    if isinstance(value, tuple):
        return (
            type(value)(*(_stage_optimizer_value(item, stager) for item in value))
            if hasattr(value, "_fields")
            else tuple(_stage_optimizer_value(item, stager) for item in value)
        )
    return copy.deepcopy(value)


def _optimizer_state_dict(optimizer: Any) -> Any:
    children = getattr(optimizer, "chained_optimizers", None)
    if children is not None:
        states = [_optimizer_state_dict(child) for child in children]
        return states[0] if len(states) == 1 else states
    if getattr(optimizer, "is_stub_optimizer", False):
        return _STUB_OPTIMIZER_STATE.copy()
    return optimizer.state_dict()


def _is_stub_optimizer_state(state: Any) -> bool:
    return (
        isinstance(state, dict)
        and len(state) == 1
        and type(state.get("art_stub_optimizer")) is int
        and state["art_stub_optimizer"] == 1
    )


def _load_optimizer_state_dict(optimizer: Any, state: Any) -> None:
    children = getattr(optimizer, "chained_optimizers", None)
    if children is not None:
        states = [state] if len(children) == 1 else state
        if not isinstance(states, list) or len(states) != len(children):
            raise RuntimeError(
                "Optimizer checkpoint chain layout does not match runtime"
            )
        for child, child_state in zip(children, states, strict=True):
            _load_optimizer_state_dict(child, child_state)
        return
    is_stub = getattr(optimizer, "is_stub_optimizer", False)
    if is_stub != _is_stub_optimizer_state(state):
        raise RuntimeError("Optimizer checkpoint stub layout does not match runtime")
    if not is_stub:
        optimizer.load_state_dict(state)


def snapshot_optimizer_state(
    runtime: Any,
    *,
    generation_id: str,
    step: int,
) -> OptimizerStateSnapshot:
    return stage_optimizer_state_snapshot(
        runtime,
        generation_id=generation_id,
        step=step,
        stager=PinnedCpuSnapshotStager(),
    ).resolve()


def stage_optimizer_state_snapshot(
    runtime: Any,
    *,
    generation_id: str,
    step: int,
    stager: PinnedCpuSnapshotStager,
) -> PendingCpuSnapshot[OptimizerStateSnapshot]:
    if runtime.optimizer is None:
        raise RuntimeError("Cannot snapshot an uninitialized optimizer")
    # Cache eviction makes restaging rank-local. Cross-rank validation belongs
    # at publication commit, where every rank record is already required.
    runtime_sha256 = _model_runtime_sha256(runtime)
    layout_sha256 = _optimizer_layout_sha256(runtime)
    builder = stager.begin()
    return builder.finish(
        OptimizerStateSnapshot(
            generation_id=generation_id,
            step=step,
            rank=runtime.rank,
            world_size=runtime.world_size,
            runtime_sha256=runtime_sha256,
            layout_sha256=layout_sha256,
            topology=current_optimizer_topology(runtime.world_size),
            state_dict=_stage_optimizer_value(
                _optimizer_state_dict(runtime.optimizer), builder
            ),
        )
    )


def stage_trainer_rank_optimizer_state_snapshot(
    runtime: Any,
    state: Any,
    *,
    generation_id: str,
    step: int,
    stager: PinnedCpuSnapshotStager,
) -> PendingCpuSnapshot[OptimizerStateSnapshot]:
    layout = state.get("layout")
    if layout is None:
        raise RuntimeError("TrainerRank optimizer state has no topology layout")
    # This path may run on only the rank that evicted a cached generation.
    builder = stager.begin()
    return builder.finish(
        OptimizerStateSnapshot(
            generation_id=generation_id,
            step=step,
            rank=runtime.rank,
            world_size=runtime.world_size,
            runtime_sha256=_model_runtime_sha256(runtime),
            optimizer_semantic_sha256=trainer_rank_optimizer_semantic_sha256(runtime),
            layout_sha256=_json_sha256(layout),
            topology=current_optimizer_topology(runtime.world_size),
            state_dict=_stage_optimizer_value(state, builder),
        )
    )


def trainer_rank_optimizer_snapshot_from_cpu(
    runtime: Any,
    state: Any,
    *,
    generation_id: str,
    step: int,
) -> OptimizerStateSnapshot:
    """Wrap an immutable residency image without another CPU or GPU copy."""
    layout = state.get("layout")
    if layout is None:
        raise RuntimeError("TrainerRank optimizer state has no topology layout")
    tensors = tuple(_optimizer_value_tensors(state))
    if not tensors or any(tensor.device.type != "cpu" for tensor in tensors):
        raise RuntimeError("resident optimizer snapshot must be entirely CPU resident")
    return OptimizerStateSnapshot(
        generation_id=generation_id,
        step=step,
        rank=runtime.rank,
        world_size=runtime.world_size,
        runtime_sha256=_model_runtime_sha256(runtime),
        optimizer_semantic_sha256=trainer_rank_optimizer_semantic_sha256(runtime),
        layout_sha256=_json_sha256(layout),
        topology=current_optimizer_topology(runtime.world_size),
        state_dict=state,
    )


def _optimizer_value_tensors(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _optimizer_value_tensors(key)
            yield from _optimizer_value_tensors(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _optimizer_value_tensors(item)


def write_optimizer_snapshot_shard(
    snapshot: OptimizerStateSnapshot,
    *,
    optimizer_state_path: str,
    prepared: PreparedOptimizerArchive,
    identity: FileIdentity,
) -> OptimizerShard:
    pending = optimizer_pending_generation_path(
        optimizer_state_path, snapshot.generation_id
    )
    if prepared.nbytes != identity.size_bytes:
        raise RuntimeError("optimizer archive size changed after planning")
    shard_path = optimizer_shard_path(
        pending,
        rank=snapshot.rank,
        world_size=snapshot.world_size,
        serialization="art_safetensors_v1",
    )
    pending.mkdir(parents=True, exist_ok=True)
    written = prepared.write(shard_path, identity=identity)
    if written != identity:
        raise RuntimeError("optimizer archive identity changed after planning")
    return OptimizerShard(
        rank=snapshot.rank,
        size_bytes=identity.size_bytes,
        layout_sha256=snapshot.layout_sha256,
        sha256=identity.sha256,
        serialization="art_safetensors_v1",
    )


def write_portable_optimizer_snapshot_shard(
    snapshot: OptimizerStateSnapshot,
    *,
    optimizer_state_path: str,
    prepared: PreparedPortableOptimizerArchive,
    identity: FileIdentity,
) -> OptimizerShard:
    """Write one source-owned logical shard for topology-neutral restoration."""
    from art.megatron.portable_optimizer_archive import (
        write_portable_optimizer_archive,
    )

    pending = optimizer_pending_generation_path(
        optimizer_state_path, snapshot.generation_id
    )
    shard_path = optimizer_shard_path(
        pending,
        rank=snapshot.rank,
        world_size=snapshot.world_size,
        serialization="art_logical_safetensors_v1",
    )
    pending.mkdir(parents=True, exist_ok=True)
    written = write_portable_optimizer_archive(prepared, shard_path, identity=identity)
    if written != identity:
        raise RuntimeError("portable optimizer archive identity changed after planning")
    return OptimizerShard(
        rank=snapshot.rank,
        size_bytes=identity.size_bytes,
        layout_sha256=snapshot.layout_sha256,
        sha256=identity.sha256,
        serialization="art_logical_safetensors_v1",
        logical_keys=prepared.metadata.logical_keys,
    )


def _loaded_adapter(adapter_path: str, step: int) -> OptimizerAdapter:
    path = Path(adapter_path).absolute()
    canonical = _canonical_adapter_path(path, step)
    adapter = read_adapter_publication(canonical, step=step, verify_files=True)
    if adapter is None:
        adapter = optimizer_adapter(canonical, step)
    if path != canonical:
        raise RuntimeError("Optimizer state must load an immutable canonical adapter")
    return adapter


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
            torch.save(_optimizer_state_dict(runtime.optimizer), output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, shard_path)
    finally:
        temporary.unlink(missing_ok=True)
    return OptimizerShard(
        rank=runtime.rank,
        size_bytes=shard_path.stat().st_size,
        layout_sha256=layout_sha256,
        sha256=_file_sha256(shard_path),
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
        path = Path(optimizer_state_path)
        with _writer_lease(path) as expected:
            if expected is not None and step <= expected.step:
                raise RuntimeError(
                    "Optimizer save step must advance the committed pointer: "
                    f"current={expected.step}, attempted={step}"
                )
            expected_data = (
                None if expected is None else expected.model_dump(mode="json")
            )
        return (
            adapter.generation_id,
            runtime_sha256,
            layouts,
            expected_data,
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
) -> None:
    with ExitStack() as leases:
        optimizer_group_decision(
            runtime,
            lambda: leases.enter_context(optimizer_model_lease(optimizer_state_path)),
            operation="optimizer model lease acquisition",
        )
        save_optimizer_state_under_model_lease(
            runtime,
            optimizer_state_path=optimizer_state_path,
            step=step,
            adapter=adapter,
        )


def save_optimizer_state_under_model_lease(
    runtime: Any,
    *,
    optimizer_state_path: str,
    step: int,
    adapter: OptimizerAdapter,
) -> None:
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
            policy_pointer = _read_policy_pointer(sibling)
            if pointer is None and policy_pointer is None:
                continue
            if pointer is not None:
                manifest = _read_manifest(
                    optimizer_generation_path(str(sibling), pointer.generation)
                )
                _validate_pointer_manifest(pointer, manifest)
            policy = _committed_policy(
                sibling,
                pointer,
                initial_adapter_path=get_step_checkpoint_dir(str(sibling.parent), 0),
            )
            if policy.policy_adapter == adapter:
                return True
    return False


def load_optimizer_state(
    runtime: Any,
    *,
    optimizer_state_path: str,
    adapter_path: str,
    adapter_step: int,
    optimizer_generation_id: str | None,
    allow_missing: bool,
    initialize: Callable[[Any], None],
) -> Path | None:
    records = _all_gather_objects(runtime, _runtime_layout_record(runtime))
    with ExitStack() as leases:

        def select_generation() -> dict[str, Any] | None:
            runtime_sha256, layouts = _validated_runtime_layouts(runtime, records)
            adapter = _loaded_adapter(adapter_path, adapter_step)
            path = Path(optimizer_state_path)
            leased_pointer = leases.enter_context(
                optimizer_generation_lease(
                    optimizer_state_path, optimizer_generation_id
                )
                if optimizer_generation_id is not None
                else _committed_generation_lease(path)
            )
            if optimizer_generation_id is not None:
                pinned_adapter = adapter
            else:
                policy = _committed_policy(
                    path,
                    leased_pointer,
                    initial_adapter_path=get_step_checkpoint_dir(
                        str(path.absolute().parent), 0
                    ),
                )
                if policy.policy_adapter == adapter:
                    if policy.optimizer_anchor is None:
                        return None
                    assert policy.state_adapter is not None
                    pinned_adapter = policy.state_adapter
                else:
                    pinned_adapter = adapter
                lineage_switch = (
                    leased_pointer is None or leased_pointer.adapter != pinned_adapter
                ) and _sibling_optimizer_owns_adapter(optimizer_state_path, adapter)
                if lineage_switch:
                    return None
            pointer = pin_optimizer_generation(
                optimizer_state_path,
                world_size=runtime.world_size,
                runtime_sha256=runtime_sha256,
                layout_sha256_by_rank=layouts,
                adapter=pinned_adapter,
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
                lambda: _load_optimizer_state_dict(runtime.optimizer, loaded_state),
            )
        finally:
            del loaded_state
        return shard_path


def load_trainer_rank_optimizer_state(
    *,
    optimizer_state_path: str,
    adapter_path: str,
    adapter_step: int,
    optimizer_generation_id: str,
    verification: VerifiedOptimizerGeneration,
    expected_optimizer_semantic_sha256: str,
    layout: Any,
    sites: Sequence[tuple[Any, Any]],
    expected_keys: Sequence[str],
) -> Any:
    """Repartition one logical optimizer generation for this destination rank."""
    if verification.generation != optimizer_generation_id:
        raise RuntimeError("Optimizer verification names another generation")
    with optimizer_generation_lease(
        optimizer_state_path, optimizer_generation_id
    ) as pointer:
        generation_path = optimizer_generation_path(
            optimizer_state_path, pointer.generation
        )
        manifest, manifest_sha256 = _read_manifest_identity(generation_path)
        if manifest_sha256 != verification.manifest_sha256:
            raise RuntimeError("Optimizer generation manifest verification mismatch")
        _validate_pointer_manifest(pointer, manifest)
        if (
            manifest.optimizer_semantic_sha256
            != expected_optimizer_semantic_sha256
        ):
            raise RuntimeError(
                "Optimizer checkpoint logical runtime mismatch: "
                f"saved={manifest.optimizer_semantic_sha256}, "
                f"current={expected_optimizer_semantic_sha256}"
            )
        adapter = _loaded_adapter(adapter_path, adapter_step)
        shards = _validate_generation_files(generation_path, manifest, local_rank=None)
        if pointer.adapter.model_copy(update={"identity": adapter.identity}) != adapter:
            raise RuntimeError(
                "Optimizer checkpoint adapter mismatch: "
                f"saved={pointer.adapter.model_dump()}, current={adapter.model_dump()}"
            )
        _validate_adapter_publication(adapter, verify_files=True)
        if any(shard.serialization != "art_logical_safetensors_v1" for shard in shards):
            raise RuntimeError(
                "TrainerRank optimizer checkpoints require logical safetensors shards"
            )
        from art.megatron.portable_optimizer_archive import (
            portable_optimizer_logical_keys_for_sites,
            read_portable_optimizer_archive,
            reconstruct_portable_optimizer_components,
            reconstruct_trainer_rank_optimizer_state,
        )

        required = portable_optimizer_logical_keys_for_sites(sites)
        expected = set(expected_keys)
        owners: dict[str, OptimizerShard] = {}
        for shard in shards:
            for key in shard.logical_keys:
                if key in owners:
                    raise RuntimeError(
                        f"Optimizer logical key has multiple source owners: {key}"
                    )
                owners[key] = shard
        if set(owners) != expected:
            raise RuntimeError(
                "Optimizer checkpoint logical keys differ from the destination: "
                f"missing={sorted(expected.difference(owners))}, "
                f"unexpected={sorted(set(owners).difference(expected))}"
            )
        if missing := set(required).difference(expected):
            raise RuntimeError(
                "Destination rank requires optimizer keys outside the prepared "
                f"checkpoint: {sorted(missing)}"
            )
        archives = []
        for shard in shards:
            selected = set(shard.logical_keys).intersection(required)
            if not selected:
                continue
            path = optimizer_shard_path(
                generation_path,
                rank=shard.rank,
                world_size=manifest.topology.world_size,
                serialization=shard.serialization,
            )
            archives.append(
                read_portable_optimizer_archive(path, logical_keys=selected)
            )
        components = reconstruct_portable_optimizer_components(archives)
        return reconstruct_trainer_rank_optimizer_state(components, sites, layout)


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
        OPTIMIZER_POLICY_POINTER,
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


def _quarantine_pointer_temp(path: Path, temporary: Path) -> None:
    quarantine = (
        path
        / "uncommitted_generations"
        / f"invalid_pointer_{int(time.time())}_{uuid4().hex}"
    )
    quarantine.mkdir(parents=True)
    os.replace(temporary, quarantine / temporary.name)
    _fsync_directory(quarantine)
    _fsync_directory(path)


def _validate_committed_generation(
    path: Path, pointer: OptimizerGenerationPointer
) -> None:
    generation_path = optimizer_generation_path(str(path), pointer.generation)
    manifest = _read_manifest(generation_path)
    _validate_pointer_manifest(pointer, manifest)
    _validate_generation_files(generation_path, manifest, local_rank=None)
    _validate_adapter_publication(pointer.adapter, verify_files=True)


def _recover_optimizer_pointer_locked(
    path: Path, current: OptimizerGenerationPointer | None
) -> OptimizerGenerationPointer | None:
    policy_temps = tuple(
        entry
        for entry in sorted(path.iterdir())
        if _POLICY_TEMP_RE.fullmatch(entry.name) is not None and entry.is_file()
    )
    for temporary in policy_temps:
        temporary.unlink()
    if policy_temps:
        _fsync_directory(path)
    temporary_paths = tuple(
        entry
        for entry in sorted(path.iterdir())
        if _POINTER_TEMP_RE.fullmatch(entry.name) is not None and entry.is_file()
    )
    if len(temporary_paths) > 1:
        raise RuntimeError(
            "Ambiguous interrupted optimizer transaction; found multiple temporary "
            "pointers"
        )
    if temporary_paths:
        try:
            temporary = OptimizerGenerationPointer.model_validate_json(
                temporary_paths[0].read_text("utf-8")
            )
        except Exception:
            _quarantine_pointer_temp(path, temporary_paths[0])
            temporary = None
    else:
        temporary = None

    _, candidates = _scan_optimizer_transactions(path)
    current_step = -1 if current is None else current.step
    advancing = tuple(
        (entry, generation)
        for entry, generation, pending in candidates
        if not pending
        and generation != (None if current is None else current.generation)
        and _generation_step(generation) > current_step
    )
    if temporary is not None and temporary.step <= current_step:
        if temporary == current:
            temporary_paths[0].unlink()
            _fsync_directory(path)
        else:
            _quarantine_pointer_temp(path, temporary_paths[0])
        temporary = None
    if temporary is not None:
        if len(advancing) != 1 or advancing[0][1] != temporary.generation:
            raise RuntimeError(
                "Interrupted optimizer pointer does not uniquely identify an "
                "advancing committed generation"
            )
        pointer = temporary
    elif not advancing:
        return current
    elif len(advancing) == 1:
        manifest = _read_manifest(advancing[0][0])
        pointer = OptimizerGenerationPointer(
            generation=manifest.generation,
            step=manifest.step,
            adapter=manifest.adapter,
        )
    else:
        raise RuntimeError(
            "Ambiguous interrupted optimizer transaction; found multiple advancing "
            "committed generations"
        )

    _validate_committed_generation(path, pointer)
    if temporary is None:
        _write_model_atomic(path / OPTIMIZER_POINTER, pointer)
    else:
        os.replace(temporary_paths[0], path / OPTIMIZER_POINTER)
        _fsync_directory(path)
    return pointer


def _optimizer_state_paths(output_dir: str, current: str) -> tuple[Path, ...]:
    selected = Path(current).absolute()
    candidates = {selected} | {
        (Path(output_dir) / f"optimizer_states_{kind}").absolute()
        for kind in ("rl", "sft")
    }
    return tuple(
        sorted(path for path in candidates if path == selected or path.exists())
    )


def _recover_optimizer_transactions(output_dir: str, current: str) -> None:
    for path in _optimizer_state_paths(output_dir, current):
        with _writer_lease(path):
            pass


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
        policy = _resolve_policy_pointer(path, pointer)
        if policy is not None and policy.policy_adapter.step == 1:
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

        tag = f"initial_step_0001_{adapter.generation_id.rsplit('-', 1)[-1][:16]}"
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
            _validate_committed_generation(Path(optimizer_state_path), pointer)
            expected_path = Path(
                get_step_checkpoint_dir(output_dir, pointer.adapter.step)
            ).absolute()
            if pointer.adapter.identity != str(expected_path):
                raise RuntimeError(
                    "Optimizer pointer does not identify the canonical adapter path: "
                    f"saved={pointer.adapter.identity}, expected={expected_path}"
                )
        policy = _resolve_policy_pointer(Path(optimizer_state_path), pointer)
        if policy is not None:
            expected_path = Path(
                get_step_checkpoint_dir(output_dir, policy.policy_adapter.step)
            ).absolute()
            if policy.policy_adapter.identity != str(expected_path):
                raise RuntimeError(
                    "Optimizer policy pointer does not identify the canonical "
                    f"adapter path: saved={policy.policy_adapter.identity}, "
                    f"expected={expected_path}"
                )
            return MegatronResumeStep(
                step=policy.policy_adapter.step,
                latest_lora_step=latest_lora_step,
                optimizer_step=None if pointer is None else pointer.step,
            )
        if pointer is not None:
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
        if (
            read_committed_optimizer_pointer(str(path)) is not None
            or _read_policy_pointer(path) is not None
        ):
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
    _recover_optimizer_transactions(output_dir, optimizer_state_path)
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
        info = _prepare_megatron_resume_state_locked(
            output_dir=output_dir,
            optimizer_state_path=optimizer_state_path,
        )
        latest = Path(output_dir) / "megatron_runtime" / ADAPTER_LATEST_POINTER
        if info.step == 0:
            if latest.exists():
                latest.unlink()
                _fsync_directory(latest.parent)
        else:
            policy = resolve_committed_optimizer_policy(
                optimizer_state_path,
                initial_adapter_path=get_step_checkpoint_dir(output_dir, 0),
            )
            _write_model_atomic(latest, policy.policy_adapter)
        return info


def format_megatron_resume_message(info: MegatronResumeStep) -> str:
    if info.used_unpaired_override:
        return (
            "Resuming Megatron from unpaired LoRA checkpoint "
            f"{info.step} because {ALLOW_UNPAIRED_MEGATRON_RESUME_ENV} is set"
        )
    suffix = ""
    if info.quarantined_lora_steps:
        moved = ", ".join(f"{step:04d}" for step in info.quarantined_lora_steps)
        suffix = f"; quarantined unpaired LoRA checkpoint(s): {moved}"
    if info.step > 0 and info.optimizer_step != info.step:
        optimizer = (
            "an uninitialized optimizer"
            if info.optimizer_step is None
            else f"optimizer state {info.optimizer_step}"
        )
        latest = (
            ""
            if info.step == info.latest_lora_step
            else f" instead of latest LoRA checkpoint {info.latest_lora_step}"
        )
        return (
            f"Resuming no-op policy checkpoint {info.step} with {optimizer}"
            f"{latest}{suffix}"
        )
    if info.step != info.latest_lora_step:
        return (
            "Resuming Megatron from paired LoRA/optimizer checkpoint "
            f"{info.step} instead of latest LoRA checkpoint "
            f"{info.latest_lora_step}{suffix}"
        )
    return f"Resuming Megatron from checkpoint {info.step}"
