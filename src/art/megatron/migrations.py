from __future__ import annotations

import os
from pathlib import Path
import warnings

from .optimizer_state import read_committed_optimizer_pointer

_IGNORED_ROOT_ENTRIES = {".writer.lock"}


def optimizer_state_path(output_dir: str) -> str:
    return str(Path(output_dir) / "optimizer_states")


def _contains_optimizer_state(path: Path) -> bool:
    return path.is_dir() and any(
        entry.name not in _IGNORED_ROOT_ENTRIES for entry in path.iterdir()
    )


def apply_megatron_migrations(output_dir: str) -> str:
    """Move one immutable split optimizer root to the unified run root."""
    destination = Path(optimizer_state_path(output_dir))
    split = tuple(
        path
        for mode in ("rl", "sft")
        if _contains_optimizer_state(
            path := Path(output_dir) / f"optimizer_states_{mode}"
        )
    )
    if destination.exists():
        if split:
            raise RuntimeError(
                "Unified and split Megatron optimizer states both exist; ART "
                "cannot infer which lineage to keep"
            )
        return str(destination)
    if len(split) > 1:
        raise RuntimeError(
            "Both legacy RL and SFT optimizer states exist. ART cannot infer "
            "which lineage to keep"
        )
    if not split:
        return str(destination)

    source = split[0]
    # This validates the generation format and deliberately rejects loose shards.
    read_committed_optimizer_pointer(str(source))
    os.replace(source, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    warnings.warn(
        f"Migrated split Megatron optimizer state {source.name} to {destination.name}.",
        stacklevel=2,
    )
    return str(destination)
