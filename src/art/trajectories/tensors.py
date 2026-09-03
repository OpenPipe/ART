"""Lazy Torch-backed trajectory views.

Importing :mod:`art.trajectories` does not import this module. Tensorization and
tensorized compact validation load it only when requested.
"""

from __future__ import annotations

import math
from typing import Generic, Self, TypeVar, cast

import pydantic
import torch

from . import (
    CompactDumpable,
    CompactTrajectoryPayload,
    MetadataValue,
    TokenFlag,
    TokenizedHistory,
    TokenizedMultiHistoryTrajectory,
    TokenizedTrajectory,
    TokenizedTrajectoryGroup,
    Trajectory,
    TrajectoryGroup,
    TrajectoryHistory,
    _FirstOccurrenceTrie,
    _StringInterningModel,
)
from ._serialization import (
    _rebind_history_sources,
    serialize_history,
    validate_history,
)

type Device = torch.device | str | None


def _tensor(value: object, *, dtype: torch.dtype, label: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value.to(dtype=dtype)
    else:
        if label == "logprobs" and isinstance(value, (list, tuple)):
            value = [math.nan if item == "NaN" else item for item in value]
        try:
            result = torch.tensor(value, dtype=dtype)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"{label} must be a one-dimensional tensor") from exc
    if result.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional tensor")
    return result.contiguous()


def _json_tensor(value: torch.Tensor) -> list[int] | list[float | str]:
    items = value.detach().cpu().tolist()
    if value.dtype.is_floating_point:
        return [
            "NaN" if isinstance(item, float) and math.isnan(item) else float(item)
            for item in items
        ]
    return [int(item) for item in items]


class TensorizedHistory(_StringInterningModel):
    """One tokenizable history represented by canonical one-dimensional tensors."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    history: TrajectoryHistory
    model: str
    tokens: torch.Tensor
    logprobs: torch.Tensor
    flags: torch.Tensor

    @pydantic.field_serializer("history")
    def serialize_source_history(
        self, value: TrajectoryHistory
    ) -> dict[str, pydantic.JsonValue]:
        return serialize_history(value)

    @pydantic.field_validator("history", mode="before")
    @classmethod
    def validate_source_history(cls, value: object) -> object:
        return validate_history(value)

    @pydantic.field_validator("tokens", mode="before")
    @classmethod
    def validate_tokens(cls, value: object) -> torch.Tensor:
        return _tensor(value, dtype=torch.int64, label="tokens")

    @pydantic.field_validator("logprobs", mode="before")
    @classmethod
    def validate_logprobs(cls, value: object) -> torch.Tensor:
        return _tensor(value, dtype=torch.float32, label="logprobs")

    @pydantic.field_validator("flags", mode="before")
    @classmethod
    def validate_flags(cls, value: object) -> torch.Tensor:
        return _tensor(value, dtype=torch.int32, label="flags")

    @pydantic.field_serializer("tokens", "logprobs", "flags", when_used="json")
    def serialize_tensor(self, value: torch.Tensor) -> list[int] | list[float | str]:
        return _json_tensor(value)

    @pydantic.model_validator(mode="after")
    def validate_tokenwise_lengths(self) -> Self:
        if not (len(self.tokens) == len(self.logprobs) == len(self.flags)):
            raise ValueError("Tensorized history fields differ in length")
        sampled = self.flags.bitwise_and(int(TokenFlag.SAMPLED)).bool()
        self.flags = self.flags.bitwise_or(
            sampled.to(self.flags.dtype) * int(TokenFlag.OUTPUT)
        )
        exact = self.flags.bitwise_and(int(TokenFlag.EXACT)).bool()
        if bool((sampled & ~exact).any().item()):
            raise ValueError(
                "SAMPLED tokens must also be EXACT; regenerate this tokenization"
            )
        return self

    def to(self, device: torch.device | str) -> Self:
        """Move owned tensors to ``device`` in place and return this view."""

        self.tokens = self.tokens.to(device=device)
        self.logprobs = self.logprobs.to(device=device)
        self.flags = self.flags.to(device=device)
        return self

    def compact_dump(self) -> CompactTrajectoryPayload:
        from ._compact import dump

        return dump(self)


class TensorizedTrajectory(TensorizedHistory):
    trajectory: Trajectory

    @property
    def reward(self) -> float:
        return self.trajectory.reward

    @reward.setter
    def reward(self, value: float) -> None:
        self.trajectory.reward = value

    @property
    def metrics(self) -> dict[str, float | int | bool]:
        return self.trajectory.metrics

    @metrics.setter
    def metrics(self, value: dict[str, float | int | bool]) -> None:
        self.trajectory.metrics = value

    @property
    def metadata(self) -> dict[str, MetadataValue]:
        return self.trajectory.metadata

    @metadata.setter
    def metadata(self, value: dict[str, MetadataValue]) -> None:
        self.trajectory.metadata = value

    @pydantic.model_validator(mode="after")
    def bind_source_trajectory(self) -> Self:
        _rebind_history_sources(self.history, self.trajectory)
        return self


class TensorizedMultiHistoryTrajectory(_StringInterningModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    trajectory: Trajectory
    histories: list[TensorizedHistory]

    @property
    def reward(self) -> float:
        return self.trajectory.reward

    @reward.setter
    def reward(self, value: float) -> None:
        self.trajectory.reward = value

    @property
    def metrics(self) -> dict[str, float | int | bool]:
        return self.trajectory.metrics

    @metrics.setter
    def metrics(self, value: dict[str, float | int | bool]) -> None:
        self.trajectory.metrics = value

    @property
    def metadata(self) -> dict[str, MetadataValue]:
        return self.trajectory.metadata

    @metadata.setter
    def metadata(self, value: dict[str, MetadataValue]) -> None:
        self.trajectory.metadata = value

    @pydantic.model_validator(mode="after")
    def bind_source_trajectory(self) -> Self:
        for history in self.histories:
            _rebind_history_sources(history.history, self.trajectory)
        return self

    def to(self, device: torch.device | str) -> Self:
        for history in self.histories:
            history.to(device)
        return self

    def first_occurrence_masks(
        self, *, where: TokenFlag | None = None
    ) -> list[torch.Tensor]:
        """Select the first eligible occurrence of each model-visible prefix."""

        return first_occurrence_masks(self.histories, where=where)

    def compact_dump(self) -> CompactTrajectoryPayload:
        from ._compact import dump

        return dump(self)


def first_occurrence_masks(
    histories: list[TensorizedHistory],
    *,
    where: TokenFlag | None = None,
) -> list[torch.Tensor]:
    """Tensor implementation using one bulk device transfer per source tensor."""

    trie = _FirstOccurrenceTrie()
    result: list[torch.Tensor] = []
    for history in histories:
        tokens, flags = (
            torch.stack((history.tokens, history.flags)).detach().cpu().tolist()
        )
        mask = trie.mask(history.model, tokens, flags, where=where)
        result.append(
            torch.tensor(mask, dtype=torch.bool, device=history.tokens.device)
        )
    return result


TensorizedTrajectoryT = TypeVar(
    "TensorizedTrajectoryT",
    TensorizedTrajectory,
    TensorizedMultiHistoryTrajectory,
)


class TensorizedTrajectoryGroup(_StringInterningModel, Generic[TensorizedTrajectoryT]):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    trajectory_group: TrajectoryGroup
    trajectories: list[TensorizedTrajectoryT]

    @property
    def metrics(self) -> dict[str, float | int | bool]:
        return self.trajectory_group.metrics

    @metrics.setter
    def metrics(self, value: dict[str, float | int | bool]) -> None:
        self.trajectory_group.metrics = value

    @property
    def metadata(self) -> dict[str, MetadataValue]:
        return self.trajectory_group.metadata

    @metadata.setter
    def metadata(self, value: dict[str, MetadataValue]) -> None:
        self.trajectory_group.metadata = value

    @pydantic.model_validator(mode="after")
    def bind_source_group(self) -> Self:
        if len(self.trajectories) != len(self.trajectory_group.trajectories):
            raise ValueError("Tensorized group differs in length from its source group")
        for tensorized, trajectory in zip(
            self.trajectories, self.trajectory_group.trajectories, strict=True
        ):
            if (
                tensorized.trajectory is not trajectory
                and tensorized.trajectory.model_dump() != trajectory.model_dump()
            ):
                raise ValueError(
                    "Tensorized trajectory does not match its source group"
                )
            tensorized.trajectory = trajectory
            if isinstance(tensorized, TensorizedTrajectory):
                _rebind_history_sources(tensorized.history, trajectory)
            else:
                for history in tensorized.histories:
                    _rebind_history_sources(history.history, trajectory)
        return self

    def to(self, device: torch.device | str) -> Self:
        for trajectory in self.trajectories:
            if isinstance(trajectory, TensorizedTrajectory):
                trajectory.to(device)
            else:
                trajectory.to(device)
        return self

    def compact_dump(self) -> CompactTrajectoryPayload:
        from ._compact import dump

        return dump(cast(CompactDumpable, self))


def tensorize_history(
    value: TokenizedHistory, *, device: Device = None
) -> TensorizedHistory:
    return TensorizedHistory(
        history=value.history,
        model=value.model,
        tokens=torch.tensor(value.tokens, dtype=torch.int64, device=device or "cpu"),
        logprobs=torch.tensor(
            value.logprobs, dtype=torch.float32, device=device or "cpu"
        ),
        flags=torch.tensor(value.flags, dtype=torch.int32, device=device or "cpu"),
    )


def tensorize_trajectory(
    value: TokenizedTrajectory, *, device: Device = None
) -> TensorizedTrajectory:
    history = tensorize_history(value, device=device)
    return TensorizedTrajectory(
        history=history.history,
        trajectory=value.trajectory,
        model=history.model,
        tokens=history.tokens,
        logprobs=history.logprobs,
        flags=history.flags,
    )


def tensorize_multi_history_trajectory(
    value: TokenizedMultiHistoryTrajectory,
    *,
    device: Device = None,
) -> TensorizedMultiHistoryTrajectory:
    return TensorizedMultiHistoryTrajectory(
        trajectory=value.trajectory,
        histories=[
            tensorize_history(history, device=device) for history in value.histories
        ],
    )


def tensorize_group(
    value: TokenizedTrajectoryGroup,
    *,
    device: Device = None,
) -> (
    TensorizedTrajectoryGroup[TensorizedTrajectory]
    | TensorizedTrajectoryGroup[TensorizedMultiHistoryTrajectory]
):
    kinds = {
        isinstance(item, TokenizedMultiHistoryTrajectory) for item in value.trajectories
    }
    if len(kinds) > 1:
        raise ValueError("Tokenized group mixes trajectory types")
    arguments = value.__class__.__pydantic_generic_metadata__.get("args", ())
    multi = kinds == {True} or (
        not kinds and arguments == (TokenizedMultiHistoryTrajectory,)
    )
    if multi:
        items = [
            item
            for item in value.trajectories
            if isinstance(item, TokenizedMultiHistoryTrajectory)
        ]
        if len(items) != len(value.trajectories):
            raise ValueError("Tokenized group contains a non-multi-history trajectory")
        return TensorizedTrajectoryGroup[TensorizedMultiHistoryTrajectory](
            trajectory_group=value.trajectory_group,
            trajectories=[
                tensorize_multi_history_trajectory(item, device=device)
                for item in items
            ],
        )
    items = [
        item for item in value.trajectories if isinstance(item, TokenizedTrajectory)
    ]
    if len(items) != len(value.trajectories):
        raise ValueError("Tokenized group contains a non-single-history trajectory")
    return TensorizedTrajectoryGroup[TensorizedTrajectory](
        trajectory_group=value.trajectory_group,
        trajectories=[tensorize_trajectory(item, device=device) for item in items],
    )


__all__ = [
    "TensorizedHistory",
    "TensorizedMultiHistoryTrajectory",
    "TensorizedTrajectory",
    "TensorizedTrajectoryGroup",
]
