"""Optimizer-step representation shared by TrainerRank persistence paths."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import cast

import torch


def optimizer_iteration(value: object) -> int:
    """Return one exact nonnegative optimizer iteration."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("optimizer iteration tensor must contain one scalar")
        value = value.item()
    if type(value) not in {int, float}:
        raise ValueError("optimizer iteration must be numeric")
    numeric = float(cast(int | float, value))
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError("optimizer iteration must be a finite nonnegative integer")
    return int(numeric)


def shared_optimizer_iteration(
    param_group: Mapping[str, object],
    parameter_states: Iterable[Mapping[str, object]],
) -> int:
    """Read TE FusedAdam's one iteration counter without accepting ambiguity."""
    states = tuple(parameter_states)
    if "step" not in param_group:
        raise ValueError("optimizer parameter group is missing its shared iteration")
    if any("step" in state for state in states):
        raise ValueError("optimizer parameter state must not contain iteration counters")
    return optimizer_iteration(param_group["step"])


def require_uniform_optimizer_iterations(values: Iterable[object]) -> int:
    """Collapse logical per-parameter copies into TE's shared group counter."""
    iterations = tuple(optimizer_iteration(value) for value in values)
    if not iterations:
        return 0
    first = iterations[0]
    if any(value != first for value in iterations[1:]):
        raise ValueError("logical optimizer iterations differ within one group")
    return first
