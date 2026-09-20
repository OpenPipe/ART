"""Immutable forward policy shared by native and remote trainers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from enum import Enum
import math
from typing import Literal


class _Unset(Enum):
    VALUE = "Unset"

    def __repr__(self) -> str:
        return "Unset"


Unset = _Unset.VALUE


@dataclass(frozen=True, kw_only=True)
class ImportanceSamplingGradientCorrection:
    """Weight selected-token cotangents by clipped current/original probability.

    This is a sampling-distribution correction for score-function estimators,
    not an exact correction for arbitrary losses or stale Jacobians. Clipping
    introduces bias; token-local and top-k ratios do not correct a full sequence
    distribution. Top-k correction requires current probabilities of the
    original token IDs, without renormalizing over the selected tokens.

    ``when_available`` never adds a forward solely to obtain current logprobs.
    ``always`` requires them, and fails if the runtime cannot supply them.
    Only stale, active logprob cotangents are eligible. Hidden states, logits,
    and custom-head gradients remain bounded and uncorrected under both policies.
    """

    clip_low: float = 0.0
    clip_high: float = 5.0
    policy: Literal["when_available", "always"] = "when_available"

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.clip_low)
            and math.isfinite(self.clip_high)
            and 0 <= self.clip_low <= self.clip_high
        ):
            raise ValueError(
                "correction clipping bounds must be finite and 0 <= low <= high"
            )
        if self.policy not in ("when_available", "always"):
            raise ValueError(f"unknown correction policy: {self.policy!r}")


@dataclass(frozen=True, kw_only=True)
class ForwardOptions:
    """Per-field overrides; ``Unset`` inherits input > method > constructor.

    An empty correction collection disables inherited corrections. Collections
    are snapshotted to tuples at construction. Existing checkpoint, grad mode,
    and output selectors remain separate forward arguments.

    Staleness counts completed checkpoint updates since the original forward,
    measured before the update consuming its gradient; replay does not reset it.
    Zero requires current weights. ``backward_state`` forces a cache placement
    unless set to ``auto``. ``allow_cpu_offload`` controls saved backward state;
    output placement is independent. ``output_device="model"`` retains native
    model-device outputs; ``auto`` permits planner placement, and ``cpu`` forces
    CPU outputs. Remote drivers transport these outputs as CPU tensor proxies.
    """

    max_gradient_staleness: int | _Unset = Unset
    stale_gradient_corrections: (
        Sequence[ImportanceSamplingGradientCorrection] | _Unset
    ) = Unset
    backward_state: Literal["auto", "gpu", "cpu", "replay"] | _Unset = Unset
    allow_cpu_offload: bool | _Unset = Unset
    allow_replay: bool | _Unset = Unset
    output_device: Literal["auto", "model", "cpu"] | _Unset = Unset

    def __post_init__(self) -> None:
        corrections = self.stale_gradient_corrections
        if corrections is not Unset:
            object.__setattr__(self, "stale_gradient_corrections", tuple(corrections))
        _validate_options(self)


@dataclass(frozen=True, kw_only=True)
class ResolvedForwardOptions:
    """Concrete policy captured at submission, never read from mutable defaults."""

    max_gradient_staleness: int = 2
    stale_gradient_corrections: tuple[ImportanceSamplingGradientCorrection, ...] = (
        ImportanceSamplingGradientCorrection(),
    )
    backward_state: Literal["auto", "gpu", "cpu", "replay"] = "auto"
    allow_cpu_offload: bool = True
    allow_replay: bool = True
    output_device: Literal["auto", "model", "cpu"] = "model"

    def __post_init__(self) -> None:
        if any(getattr(self, field.name) is Unset for field in fields(self)):
            raise ValueError("resolved options cannot contain Unset")
        object.__setattr__(
            self, "stale_gradient_corrections", tuple(self.stale_gradient_corrections)
        )
        _validate_options(self)
        if self.backward_state == "cpu" and not self.allow_cpu_offload:
            raise ValueError("backward_state='cpu' requires allow_cpu_offload=True")
        if self.backward_state == "replay" and not self.allow_replay:
            raise ValueError("backward_state='replay' requires allow_replay=True")


def _validate_options(options: ForwardOptions | ResolvedForwardOptions) -> None:
    age = options.max_gradient_staleness
    if age is not Unset and (type(age) is not int or age < 0):
        raise ValueError("max_gradient_staleness must be a nonnegative integer")
    for name in ("allow_cpu_offload", "allow_replay"):
        value = getattr(options, name)
        if value is not Unset and type(value) is not bool:
            raise ValueError(f"{name} must be a bool")
    for name, choices in (
        ("backward_state", ("auto", "gpu", "cpu", "replay")),
        ("output_device", ("auto", "model", "cpu")),
    ):
        value = getattr(options, name)
        if value is not Unset and value not in choices:
            raise ValueError(f"unknown {name}: {value!r}")
    corrections = options.stale_gradient_corrections
    if corrections is not Unset:
        if any(
            not isinstance(correction, ImportanceSamplingGradientCorrection)
            for correction in corrections
        ):
            raise TypeError("unsupported stale gradient correction")
        if len(corrections) > 1:
            raise ValueError("only one importance sampling correction may be specified")


def resolve_forward_options(
    constructor: ForwardOptions | None = None,
    method: ForwardOptions | None = None,
    input: ForwardOptions | None = None,
) -> ResolvedForwardOptions:
    """Resolve each field independently and validate the resulting policy."""
    values = {}
    for options in (constructor, method, input):
        if options is not None:
            if not isinstance(options, ForwardOptions):
                raise TypeError("options must be ForwardOptions or None")
            values.update(
                (field.name, value)
                for field in fields(options)
                if (value := getattr(options, field.name)) is not Unset
            )
    return ResolvedForwardOptions(**values)
