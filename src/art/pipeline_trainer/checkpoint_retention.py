from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime

from pydantic import BaseModel, Field

CHECKPOINT_CREATED_AT_METRIC = "checkpoint/created_at_unix"
CHECKPOINT_EVAL_COMPLETED_METRIC = "checkpoint/eval_completed"
CHECKPOINT_SAVED_METRIC = "checkpoint/saved"


class CheckpointInfo(BaseModel):
    step: int
    path: str
    created_at: datetime
    is_eval_step: bool = False
    metrics: dict[str, float] = Field(default_factory=dict)


class CheckpointRetentionContext(BaseModel):
    current_step: int
    checkpoints: list[CheckpointInfo] = Field(default_factory=list)


# Strategies receive only checkpoints that ART has determined are safe to delete
# and return the subset of those checkpoint steps to remove.
CheckpointRetentionStrategy = Callable[[CheckpointRetentionContext], Iterable[int]]


def keep_recent_and_top(
    *,
    recent: int = 5,
    top: int = 2,
    metric: str = "val/reward",
) -> CheckpointRetentionStrategy:
    """Delete eligible checkpoints except the most recent and top metric steps."""
    if recent < 0:
        raise ValueError("recent must be >= 0")
    if top < 0:
        raise ValueError("top must be >= 0")

    def strategy(context: CheckpointRetentionContext) -> set[int]:
        eligible_steps = {checkpoint.step for checkpoint in context.checkpoints}
        keep_steps: set[int] = set()
        if recent > 0:
            keep_steps.update(
                checkpoint.step
                for checkpoint in sorted(
                    context.checkpoints, key=lambda item: item.step
                )[-recent:]
            )
        ranked = [
            checkpoint
            for checkpoint in context.checkpoints
            if metric in checkpoint.metrics
        ]
        ranked.sort(key=lambda item: (item.metrics[metric], item.step), reverse=True)
        keep_steps.update(checkpoint.step for checkpoint in ranked[:top])
        return eligible_steps - keep_steps

    return strategy


__all__ = [
    "CHECKPOINT_CREATED_AT_METRIC",
    "CHECKPOINT_EVAL_COMPLETED_METRIC",
    "CHECKPOINT_SAVED_METRIC",
    "CheckpointInfo",
    "CheckpointRetentionContext",
    "CheckpointRetentionStrategy",
    "keep_recent_and_top",
]
