"""Canonical optimizer semantics shared by TrainerRank persistence paths."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import cast

import torch


def optimizer_step(value: object) -> int:
    """Validate and return one nonnegative optimizer iteration."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("optimizer step tensor must contain one scalar")
        value = value.item()
    if type(value) not in {int, float}:
        raise ValueError("optimizer step must be numeric")
    numeric_step = float(cast(int | float, value))
    if (
        not math.isfinite(numeric_step)
        or numeric_step < 0
        or not numeric_step.is_integer()
    ):
        raise ValueError("optimizer step must be a finite nonnegative integer")
    return int(numeric_step)


def shared_optimizer_step(
    param_group: Mapping[str, object],
    parameter_states: Iterable[Mapping[str, object]],
) -> int:
    """Return the one step owned by TrainerRank's single FusedAdam group.

    Current TE FusedAdam stores the counter on the parameter group. Logical ART
    archives duplicate that counter per logical parameter so it survives topology
    changes. A per-parameter representation is accepted only when it is complete
    and uniform, which is the exact inverse of that logical projection.
    """
    states = tuple(parameter_states)
    state_steps = tuple(
        optimizer_step(state["step"]) for state in states if "step" in state
    )
    if "step" in param_group:
        group_step = optimizer_step(param_group["step"])
        if any(step != group_step for step in state_steps):
            raise ValueError(
                "optimizer parameter step differs from the shared parameter-group step"
            )
        return group_step
    if not state_steps:
        return 0
    if len(state_steps) != len(states):
        raise ValueError(
            "optimizer parameter steps are incomplete without a shared group step"
        )
    first = state_steps[0]
    if any(step != first for step in state_steps[1:]):
        raise ValueError("optimizer parameter steps differ within one FusedAdam group")
    return first


def require_uniform_optimizer_steps(steps: Iterable[object]) -> int:
    """Collapse ART's logical per-parameter steps into TE's shared group step."""
    values = tuple(optimizer_step(value) for value in steps)
    if not values:
        return 0
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError("logical optimizer steps differ within one FusedAdam group")
    return first
