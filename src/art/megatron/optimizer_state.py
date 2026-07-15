from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ..utils.get_model_step import get_step_from_dir
from ..utils.output_dirs import get_step_checkpoint_dir

ALLOW_UNPAIRED_MEGATRON_RESUME_ENV = "ART_ALLOW_UNPAIRED_MEGATRON_RESUME"
OPTIMIZER_GENERATIONS_DIR = "generations"
OPTIMIZER_MANIFEST = "manifest.json"
OPTIMIZER_POINTER = "committed.json"
_GENERATION_PATTERN = r"step-\d{8,}-[0-9a-f]{32}"
_GENERATION_RE = re.compile(f"^{_GENERATION_PATTERN}$")


class _OptimizerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MegatronResumeStep(_OptimizerRecord):
    step: int
    latest_lora_step: int
    optimizer_step: int | None
    used_unpaired_override: bool = False
    quarantined_lora_steps: tuple[int, ...] = ()


class OptimizerTopology(_OptimizerRecord):
    world_size: int = Field(gt=0)
    tp: int = Field(gt=0)
    cp: int = Field(gt=0)
    ep: int = Field(gt=0)
    etp: int = Field(gt=0)
    pp: int = Field(gt=0)
    vpp: int = Field(gt=0)
    expected_shards: tuple[str, ...]


class OptimizerShard(_OptimizerRecord):
    rank: int = Field(ge=0)
    name: str
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class OptimizerGenerationManifest(_OptimizerRecord):
    format_version: Literal[1] = 1
    generation: str = Field(pattern=f"^{_GENERATION_PATTERN}$")
    step: int = Field(ge=0)
    topology: OptimizerTopology
    shards: tuple[OptimizerShard, ...]


class OptimizerGenerationPointer(_OptimizerRecord):
    format_version: Literal[1] = 1
    generation: str = Field(pattern=f"^{_GENERATION_PATTERN}$")
    step: int = Field(ge=0)


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
        expected_shards=tuple(
            optimizer_shard_name(rank, world_size) for rank in range(world_size)
        ),
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
    digest = hashlib.sha256()
    with path.open("rb") as shard_file:
        while chunk := shard_file.read(8 * 1024 * 1024):
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


def read_committed_optimizer_step(optimizer_state_path: str) -> int | None:
    pointer = _read_pointer(Path(optimizer_state_path))
    return None if pointer is None else pointer.step


def _read_manifest(
    generation_path: Path,
) -> OptimizerGenerationManifest:
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
    expected_names = tuple(
        optimizer_shard_name(rank, topology.world_size)
        for rank in range(topology.world_size)
    )
    if topology.expected_shards != expected_names:
        raise RuntimeError(
            "Optimizer manifest expected-shard identity mismatch: "
            f"expected={expected_names}, manifest={topology.expected_shards}"
        )
    ordered = tuple(sorted(manifest.shards, key=lambda shard: shard.rank))
    actual_ranks = tuple(shard.rank for shard in ordered)
    actual_names = tuple(shard.name for shard in ordered)
    if (
        actual_ranks != tuple(range(topology.world_size))
        or actual_names != expected_names
    ):
        raise RuntimeError(
            "Optimizer manifest shard coverage mismatch: "
            f"expected_ranks={tuple(range(topology.world_size))}, "
            f"actual_ranks={actual_ranks}, expected_names={expected_names}, "
            f"actual_names={actual_names}"
        )
    return ordered


def build_optimizer_manifest(
    *,
    generation: str,
    step: int,
    world_size: int,
    shards: list[OptimizerShard],
) -> OptimizerGenerationManifest:
    manifest = OptimizerGenerationManifest(
        generation=generation,
        step=step,
        topology=current_optimizer_topology(world_size),
        shards=tuple(shards),
    )
    _ordered_manifest_shards(manifest)
    return manifest


def _validate_generation_files(
    generation_path: Path,
    manifest: OptimizerGenerationManifest,
    *,
    local_rank: int | None,
) -> tuple[OptimizerShard, ...]:
    ordered = _ordered_manifest_shards(manifest)
    expected_entries = tuple(
        sorted((OPTIMIZER_MANIFEST, *manifest.topology.expected_shards))
    )
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
        actual_size = (generation_path / shard.name).stat().st_size
        if actual_size != shard.size_bytes:
            raise RuntimeError(
                f"Optimizer shard size mismatch for {shard.name}: "
                f"expected={shard.size_bytes}, actual={actual_size}"
            )
    if local_rank is not None:
        if local_rank < 0 or local_rank >= len(ordered):
            raise RuntimeError(
                f"Invalid local optimizer rank {local_rank} for {len(ordered)} shards"
            )
        local_shard = ordered[local_rank]
        actual_sha256 = hash_optimizer_shard(generation_path / local_shard.name)
        if actual_sha256 != local_shard.sha256:
            raise RuntimeError(
                f"Optimizer shard checksum mismatch for {local_shard.name}: "
                f"expected={local_shard.sha256}, actual={actual_sha256}"
            )
    return ordered


def _cleanup_old_generations(path: Path, committed: Path) -> None:
    generations_path = path / OPTIMIZER_GENERATIONS_DIR
    removed = False
    for candidate in generations_path.iterdir():
        generation = candidate.name.removeprefix(".pending-")
        if candidate == committed or not candidate.is_dir():
            continue
        if _GENERATION_RE.fullmatch(generation) is not None:
            shutil.rmtree(candidate)
            removed = True
    if removed:
        _fsync_directory(generations_path)


def commit_optimizer_generation(
    optimizer_state_path: str, manifest: OptimizerGenerationManifest
) -> Path:
    path = Path(optimizer_state_path)
    _read_pointer(path)
    pending = optimizer_pending_generation_path(
        optimizer_state_path, manifest.generation
    )
    committed = optimizer_generation_path(optimizer_state_path, manifest.generation)
    _write_model_atomic(pending / OPTIMIZER_MANIFEST, manifest)
    _validate_generation_files(pending, manifest, local_rank=None)
    os.replace(pending, committed)
    _fsync_directory(committed.parent)
    _write_model_atomic(
        path / OPTIMIZER_POINTER,
        OptimizerGenerationPointer(
            generation=manifest.generation,
            step=manifest.step,
        ),
    )
    _cleanup_old_generations(path, committed)
    return committed


def resolve_optimizer_shard(
    optimizer_state_path: str, *, rank: int, world_size: int
) -> Path | None:
    path = Path(optimizer_state_path)
    pointer = _read_pointer(path)
    if pointer is None:
        return None
    generation_path = optimizer_generation_path(
        optimizer_state_path, pointer.generation
    )
    manifest = _read_manifest(generation_path)
    if (manifest.generation, manifest.step) != (pointer.generation, pointer.step):
        raise RuntimeError(
            "Optimizer pointer/manifest identity mismatch: "
            f"pointer={pointer.model_dump()}, "
            f"manifest_generation={manifest.generation!r}, manifest_step={manifest.step}"
        )
    current = current_optimizer_topology(world_size)
    if manifest.topology != current:
        raise RuntimeError(
            "Optimizer checkpoint topology mismatch; optimizer state is topology-strict: "
            f"saved={manifest.topology.model_dump()} current={current.model_dump()}"
        )
    ordered = _validate_generation_files(generation_path, manifest, local_rank=rank)
    return generation_path / ordered[rank].name


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
    optimizer_step = read_committed_optimizer_step(optimizer_state_path)
    if latest_lora_step == 0:
        return MegatronResumeStep(
            step=0,
            latest_lora_step=latest_lora_step,
            optimizer_step=optimizer_step,
        )
    if optimizer_step is not None and os.path.isdir(
        get_step_checkpoint_dir(output_dir, optimizer_step)
    ):
        return MegatronResumeStep(
            step=optimizer_step,
            latest_lora_step=latest_lora_step,
            optimizer_step=optimizer_step,
        )
    if _allow_unpaired_resume():
        return MegatronResumeStep(
            step=latest_lora_step,
            latest_lora_step=latest_lora_step,
            optimizer_step=optimizer_step,
            used_unpaired_override=True,
        )
    marker = (
        "no optimizer step marker"
        if optimizer_step is None
        else f"optimizer marker step {optimizer_step:04d} has no matching LoRA checkpoint"
    )
    raise RuntimeError(
        "Cannot resume Megatron training from an unpaired LoRA/optimizer state: "
        f"latest LoRA checkpoint is {latest_lora_step:04d}, {marker}. "
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
