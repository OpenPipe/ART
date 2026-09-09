from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

from . import wandb_sdk


def record_provenance(run: Run, provenance: str) -> None:
    """Record provenance on the latest artifact version's metadata."""
    record_provenance_for_artifact(
        entity=str(run.entity),
        project=str(run.project),
        name=str(run.name),
        provenance=provenance,
    )


def record_provenance_for_artifact(
    *,
    entity: str,
    project: str,
    name: str,
    provenance: str,
) -> None:
    """Record provenance on the latest artifact version's metadata."""
    api = wandb_sdk.api()
    artifact_path = f"{entity}/{project}/{name}:latest"
    try:
        artifact = api.artifact(artifact_path, type="lora")
    except wandb_sdk.comm_error_type():
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
