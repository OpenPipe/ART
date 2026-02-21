from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import wandb


def record_provenance(run: wandb.Run, provenance: str) -> None:
    """Record provenance on the latest artifact version's metadata."""
    import wandb as wandb_module

    api = wandb_module.Api()
    artifact_path = f"{run.entity}/{run.project}/{run.name}:latest"
    try:
        artifact = api.artifact(artifact_path, type="lora")
    except wandb_module.errors.CommError:
        return  # No artifact exists yet

    existing = artifact.metadata.get("wandb.provenance")
    if existing is not None:
        existing = list(existing)
        if existing[-1] != provenance:
            existing.append(provenance)
        artifact.metadata["wandb.provenance"] = existing
    else:
        artifact.metadata["wandb.provenance"] = [provenance]
    artifact.save()
