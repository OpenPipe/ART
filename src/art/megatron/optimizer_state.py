from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..utils.get_model_step import get_step_from_dir
from ..utils.output_dirs import get_step_checkpoint_dir

ALLOW_UNPAIRED_MEGATRON_RESUME_ENV = "ART_ALLOW_UNPAIRED_MEGATRON_RESUME"
OPTIMIZER_MANIFEST = "CURRENT.json"
_GENERATION_SHARD_RE = re.compile(
    r"^step-(?P<step>\d+)-(?P<rank>\d+)-of-(?P<world>\d+)\.pt$"
)


class OptimizerCommit(BaseModel):
    schema_version: Literal[1] = 1
    step: int = Field(ge=0)
    world_size: int = Field(ge=1)
    files: tuple[str, ...]

    @model_validator(mode="after")
    def validate_files(self) -> "OptimizerCommit":
        if self.files != optimizer_generation_files(self.step, self.world_size):
            raise ValueError(
                "optimizer manifest files do not match its step/world size"
            )
        return self


class MegatronResumeStep(BaseModel):
    step: int
    latest_lora_step: int
    optimizer_step: int | None
    used_unpaired_override: bool = False
    quarantined_lora_steps: tuple[int, ...] = ()


def optimizer_generation_files(step: int, world_size: int) -> tuple[str, ...]:
    return tuple(
        f"step-{step:08d}-{rank:02d}-of-{world_size:02d}.pt"
        for rank in range(1, world_size + 1)
    )


def read_optimizer_commit(optimizer_state_path: str) -> OptimizerCommit | None:
    path = Path(optimizer_state_path)
    manifest_path = path / OPTIMIZER_MANIFEST
    if not manifest_path.exists():
        return None
    commit = OptimizerCommit.model_validate_json(manifest_path.read_text())
    missing = [name for name in commit.files if not (path / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Optimizer manifest {manifest_path} references missing shard(s): {missing}"
        )
    return commit


def resolve_optimizer_shard_path(
    optimizer_state_path: str,
    *,
    rank: int,
    world_size: int,
    expected_step: int,
) -> Path | None:
    if not 0 <= rank < world_size:
        raise ValueError(f"optimizer rank {rank} is outside world size {world_size}")
    path = Path(optimizer_state_path)
    commit = read_optimizer_commit(optimizer_state_path)
    if commit is not None:
        if commit.world_size != world_size:
            raise RuntimeError(
                "Optimizer world size does not match the active Megatron runtime: "
                f"{commit.world_size} != {world_size}"
            )
        if commit.step != expected_step:
            raise RuntimeError(
                "Optimizer state does not match the source policy checkpoint: "
                f"{commit.step} != {expected_step}"
            )
        return path / commit.files[rank]
    return None


def commit_optimizer_generation(
    optimizer_state_path: str,
    *,
    step: int,
    world_size: int,
    files: tuple[str, ...],
) -> None:
    path = Path(optimizer_state_path)
    path.mkdir(parents=True, exist_ok=True)
    previous = read_optimizer_commit(optimizer_state_path)
    missing = [name for name in files if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"Cannot commit missing optimizer shard(s): {missing}")
    commit = OptimizerCommit(step=step, world_size=world_size, files=files)
    _atomic_write(path / OPTIMIZER_MANIFEST, commit.model_dump_json())

    retained = set(files) | {OPTIMIZER_MANIFEST}
    obsolete = set(previous.files if previous is not None else ())
    obsolete.update(
        item.name
        for item in path.iterdir()
        if item.is_file()
        and (_GENERATION_SHARD_RE.fullmatch(item.name) or item.name.isdigit())
    )
    for name in obsolete - retained:
        candidate = path / name
        if candidate.exists():
            candidate.unlink()


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


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
    commit = read_optimizer_commit(optimizer_state_path)
    optimizer_step = commit.step if commit is not None else None
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
