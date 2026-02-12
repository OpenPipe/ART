from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import wandb


def record_provenance(run: wandb.Run, provenance: str) -> None:
    """Record provenance in run metadata, ensuring it's the last value in the array."""
    if "provenance" in run.config:
        existing = list(run.config["provenance"])
        if existing[-1] != provenance:
            existing.append(provenance)
        run.config.update({"provenance": existing})
    else:
        run.config.update({"provenance": [provenance]})
