from __future__ import annotations

import json
import os
from pathlib import Path
import time

from pydantic import BaseModel

OPTIMIZER_MANIFEST = "manifest.json"

from ..utils.get_model_step import get_step_from_dir
from ..utils.output_dirs import get_step_checkpoint_dir

ALLOW_UNPAIRED_MEGATRON_RESUME_ENV = "ART_ALLOW_UNPAIRED_MEGATRON_RESUME"


class MegatronResumeStep(BaseModel):
    step: int
    latest_lora_step: int
    optimizer_step: int | None
    used_unpaired_override: bool = False
    quarantined_lora_steps: tuple[int, ...] = ()


class OptimizerTopology(BaseModel):
    world_size: int
    tp: int
    cp: int
    ep: int
    etp: int
    pp: int
    vpp: int
    expected_shards: tuple[str, ...]


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
            f"{rank + 1:02d}-of-{world_size:02d}.pt" for rank in range(world_size)
        ),
    )


def write_optimizer_manifest(optimizer_state_path: str, *, world_size: int) -> None:
    path = Path(optimizer_state_path)
    path.mkdir(parents=True, exist_ok=True)
    manifest = current_optimizer_topology(world_size)
    missing = [name for name in manifest.expected_shards if not (path / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Cannot publish optimizer manifest with missing shards: {missing}"
        )
    temporary = path / f".{OPTIMIZER_MANIFEST}.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(manifest.model_dump(mode="json"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path / OPTIMIZER_MANIFEST)


def validate_optimizer_manifest(optimizer_state_path: str, *, world_size: int) -> bool:
    path = Path(optimizer_state_path)
    shard_paths = sorted(path.glob("*-of-*.pt")) if path.exists() else []
    manifest_path = path / OPTIMIZER_MANIFEST
    if not manifest_path.exists():
        if shard_paths:
            raise RuntimeError(
                "Optimizer shards exist without a topology manifest; refusing to reset "
                f"or load ambiguous state from {path}"
            )
        return False
    try:
        saved = OptimizerTopology.model_validate_json(manifest_path.read_text("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Invalid optimizer topology manifest: {manifest_path}"
        ) from exc
    current = current_optimizer_topology(world_size)
    if saved != current:
        raise RuntimeError(
            "Optimizer checkpoint topology mismatch; optimizer state is topology-strict: "
            f"saved={saved.model_dump()} current={current.model_dump()}"
        )
    actual = tuple(sorted(shard.name for shard in shard_paths))
    if actual != saved.expected_shards:
        raise RuntimeError(
            "Optimizer checkpoint shard coverage mismatch: "
            f"expected={saved.expected_shards}, actual={actual}"
        )
    return True


def write_optimizer_step_marker(optimizer_state_path: str, step: int) -> None:
    path = Path(optimizer_state_path)
    path.mkdir(parents=True, exist_ok=True)
    for marker in path.iterdir():
        if marker.is_file() and marker.name.isdigit():
            marker.unlink()
    (path / f"{step:04d}").touch()


def read_optimizer_step_marker(optimizer_state_path: str) -> int | None:
    path = Path(optimizer_state_path)
    if not path.exists():
        return None
    return max(
        (
            int(marker.name)
            for marker in path.iterdir()
            if marker.is_file() and marker.name.isdigit()
        ),
        default=None,
    )


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
    optimizer_step = read_optimizer_step_marker(optimizer_state_path)
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
