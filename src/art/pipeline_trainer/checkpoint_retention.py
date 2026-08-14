from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel, Field

CHECKPOINT_CREATED_AT_METRIC = "checkpoint/created_at_unix"
CHECKPOINT_EVAL_COMPLETED_METRIC = "checkpoint/eval_completed"
CHECKPOINT_SAVED_METRIC = "checkpoint/saved"


class CheckpointInfo(BaseModel):
    step: int
    path: str | None = None
    created_at: datetime
    deletion_eligible: bool = False
    is_eval_step: bool = False
    metrics: dict[str, float] = Field(default_factory=dict)


class CheckpointRetentionContext(BaseModel):
    current_step: int
    checkpoints: list[CheckpointInfo] = Field(default_factory=list)


class CheckpointRetentionPlan(BaseModel):
    observed_steps: set[int] = Field(default_factory=set)
    retain_steps: set[int] = Field(default_factory=set)
    archive_steps: set[int] = Field(default_factory=set)


# Strategies see every checkpoint. Only deletion-eligible checkpoints may be
# removed, while archive selection is explicit and independent of retention.
CheckpointRetentionStrategy = Callable[
    [CheckpointRetentionContext], CheckpointRetentionPlan
]


def plan_checkpoint_retention(
    *,
    current_step: int,
    checkpoints: list[CheckpointInfo],
    protected_steps: set[int],
    strategy: CheckpointRetentionStrategy,
) -> CheckpointRetentionPlan | None:
    known_steps = {checkpoint.step for checkpoint in checkpoints}
    eligible_steps = known_steps - protected_steps
    plan = strategy(
        CheckpointRetentionContext(
            current_step=current_step,
            checkpoints=[
                checkpoint.model_copy(
                    update={"deletion_eligible": checkpoint.step in eligible_steps}
                )
                for checkpoint in checkpoints
            ],
        )
    )
    unknown = (plan.retain_steps | plan.archive_steps) - known_steps
    if unknown:
        raise ValueError(
            f"checkpoint retention selected unknown steps: {sorted(unknown)}"
        )
    delete_steps = eligible_steps - (plan.retain_steps & eligible_steps)
    if not delete_steps and not plan.archive_steps:
        return None
    return plan.model_copy(
        update={
            "observed_steps": known_steps,
            "retain_steps": known_steps - delete_steps,
        }
    )


def keep_recent_and_top(
    *,
    recent: int = 5,
    top: int = 2,
    metric: str = "reward/val",
) -> CheckpointRetentionStrategy:
    """Keep the most recent eligible checkpoints and top metric checkpoints."""
    if recent < 0:
        raise ValueError("recent must be >= 0")
    if top < 0:
        raise ValueError("top must be >= 0")

    def strategy(context: CheckpointRetentionContext) -> CheckpointRetentionPlan:
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
        return CheckpointRetentionPlan(retain_steps=keep_steps)

    return strategy


def keep_recent_and_periodic(
    *, recent: int = 5, archive_interval: int = 25, archives: int = 2
) -> CheckpointRetentionStrategy:
    if recent < 0 or archives < 0:
        raise ValueError("recent and archives must be >= 0")
    if archive_interval < 1:
        raise ValueError("archive_interval must be >= 1")

    def strategy(context: CheckpointRetentionContext) -> CheckpointRetentionPlan:
        ordered = sorted(context.checkpoints, key=lambda item: item.step)
        periodic = [
            item.step
            for item in ordered
            if item.step > 0 and item.step % archive_interval == 0
        ]
        return CheckpointRetentionPlan(
            retain_steps={item.step for item in ordered[-recent:]} if recent else set(),
            archive_steps=set(periodic[-archives:]) if archives else set(),
        )

    return strategy


__all__ = [
    "CHECKPOINT_CREATED_AT_METRIC",
    "CHECKPOINT_EVAL_COMPLETED_METRIC",
    "CHECKPOINT_SAVED_METRIC",
    "CheckpointInfo",
    "CheckpointRetentionContext",
    "CheckpointRetentionPlan",
    "CheckpointRetentionStrategy",
    "keep_recent_and_periodic",
    "keep_recent_and_top",
    "plan_checkpoint_retention",
]
