from __future__ import annotations

from itertools import chain
from pathlib import Path
import shutil

RUNTIME_ARTIFACT_DIR_NAMES = frozenset(
    {
        "checkpoints",
        "megatron_runtime",
        "optimizer_states",
        "production_width_model",
        "trajectories",
    }
)


def _allocated_bytes(path: Path) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for item in chain((path,), path.rglob("*")):
        try:
            stat = item.lstat()
        except FileNotFoundError:
            continue
        identity = (stat.st_dev, stat.st_ino)
        if identity not in seen:
            seen.add(identity)
            total += stat.st_blocks * 512
    return total


def prune_runtime_artifacts(stage_dir: Path) -> dict[str, int]:
    candidates = sorted(
        (
            path
            for path in stage_dir.rglob("*")
            if path.is_dir() and path.name in RUNTIME_ARTIFACT_DIR_NAMES
        ),
        key=lambda path: len(path.parts),
    )
    paths: list[Path] = []
    for path in candidates:
        if not any(parent in path.parents for parent in paths):
            paths.append(path)

    before = shutil.disk_usage(stage_dir).free
    allocated = 0
    for path in paths:
        for log_path in path.rglob("vllm-runtime.log"):
            retained_path = (
                stage_dir / "retained_runtime_logs" / log_path.relative_to(stage_dir)
            )
            retained_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log_path, retained_path)
        allocated += _allocated_bytes(path)
        shutil.rmtree(path)
    after = shutil.disk_usage(stage_dir).free
    return {
        "workflow_pruned_runtime_artifact_dirs": len(paths),
        "workflow_pruned_runtime_artifact_bytes": allocated,
        "workflow_pruned_runtime_filesystem_bytes": after - before,
    }
