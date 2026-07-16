from __future__ import annotations

import os
from pathlib import Path
import re
import warnings

from ..utils.get_model_step import get_step_from_dir
from .optimizer_state import (
    commit_optimizer_generation,
    optimizer_generation_files,
    read_optimizer_commit,
)

_LEGACY_SHARD_RE = re.compile(r"^(?P<rank>\d+)-of-(?P<world>\d+)\.pt$")


def optimizer_state_path(output_dir: str) -> str:
    return str(Path(output_dir) / "optimizer_states")


def _legacy_shards(path: Path) -> tuple[Path, ...] | None:
    if not path.exists():
        return None
    if not path.is_dir():
        raise RuntimeError(f"Legacy optimizer path is not a directory: {path}")
    entries = list(path.iterdir())
    if not entries:
        return None
    matches = [
        (item, match)
        for item in entries
        if item.is_file() and (match := _LEGACY_SHARD_RE.fullmatch(item.name))
    ]
    if len(matches) != len(entries):
        unknown = sorted(
            item.name for item in entries if item not in {m[0] for m in matches}
        )
        raise RuntimeError(
            f"Legacy optimizer state at {path} contains unsupported entries: {unknown}"
        )
    worlds = {int(match.group("world")) for _, match in matches}
    if len(worlds) != 1:
        raise RuntimeError(f"Legacy optimizer shards at {path} mix world sizes")
    world_size = worlds.pop()
    by_rank = {int(match.group("rank")): item for item, match in matches}
    if set(by_rank) != set(range(1, world_size + 1)):
        raise RuntimeError(f"Legacy optimizer shards at {path} are incomplete")
    return tuple(by_rank[rank] for rank in range(1, world_size + 1))


def apply_megatron_migrations(output_dir: str) -> str:
    """Apply all durable Megatron state migrations for one training run."""
    # Keep future Megatron migrations centralized behind this call.
    destination = Path(optimizer_state_path(output_dir))
    if read_optimizer_commit(str(destination)) is not None:
        return str(destination)

    candidates = {
        mode: shards
        for mode in ("rl", "sft")
        if (shards := _legacy_shards(Path(output_dir) / f"optimizer_states_{mode}"))
        is not None
    }
    if len(candidates) > 1:
        raise RuntimeError(
            "Both legacy RL and SFT optimizer states exist. ART cannot infer which "
            "state belongs to the latest checkpoint. Keep only the intended "
            "optimizer_states_rl or optimizer_states_sft directory, or remove both "
            "to explicitly reset the optimizer."
        )
    if not candidates:
        return str(destination)

    mode, shards = next(iter(candidates.items()))
    step = get_step_from_dir(output_dir)
    files = optimizer_generation_files(step, len(shards))
    destination.mkdir(parents=True, exist_ok=True)
    for source, name in zip(shards, files, strict=True):
        target = destination / name
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        if temporary.exists():
            temporary.unlink()
        os.link(source, temporary)
        os.replace(temporary, target)
    commit_optimizer_generation(
        str(destination), step=step, world_size=len(shards), files=files
    )
    for source in shards:
        source.unlink()
    legacy_dir = Path(output_dir) / f"optimizer_states_{mode}"
    legacy_dir.rmdir()
    warnings.warn(
        f"Migrated legacy {mode.upper()} optimizer state to the run-level optimizer "
        f"commit at step {step}.",
        stacklevel=2,
    )
    return str(destination)
