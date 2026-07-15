from __future__ import annotations

from typing import Any

from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel, ConfigDict, Field

from art.pipeline_tuner.config import PackedGroupShape
from art.trajectories import (
    MetadataValue,
    PydanticException,
    Trajectory,
    TrajectoryGroup,
)

from .data_plane import PackedBatchRef
from .rollout import RolloutModelSpec


class TrajectoryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    payload: dict[str, Any]
    choice_positions: tuple[int, ...] = ()
    additional_history_choice_positions: tuple[tuple[int, ...], ...] = ()

    @classmethod
    def from_trajectory(cls, trajectory: Trajectory) -> "TrajectoryPayload":
        return cls(
            payload=trajectory.model_dump(mode="json"),
            choice_positions=tuple(
                index
                for index, item in enumerate(trajectory.messages_and_choices)
                if isinstance(item, Choice)
            ),
            additional_history_choice_positions=tuple(
                tuple(
                    index
                    for index, item in enumerate(history.messages_and_choices)
                    if isinstance(item, Choice)
                )
                for history in trajectory.additional_histories
            ),
        )

    def build(self) -> Trajectory:
        payload = dict(self.payload)
        messages = list(payload["messages_and_choices"])
        for index in self.choice_positions:
            messages[index] = Choice.model_validate(messages[index])
        payload["messages_and_choices"] = messages
        histories = [dict(history) for history in payload["additional_histories"]]
        for history, positions in zip(
            histories, self.additional_history_choice_positions, strict=True
        ):
            messages = list(history["messages_and_choices"])
            for index in positions:
                messages[index] = Choice.model_validate(messages[index])
            history["messages_and_choices"] = messages
        payload["additional_histories"] = histories
        return Trajectory.model_validate(payload)


class TrajectoryGroupPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trajectories: tuple[TrajectoryPayload, ...]
    exceptions: tuple[dict[str, str], ...] = ()
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
    metrics: dict[str, float | int | bool] = Field(default_factory=dict)
    logs: tuple[str, ...] = ()
    collect_packing_shape: bool = False

    @classmethod
    def from_group(cls, group: TrajectoryGroup) -> "TrajectoryGroupPayload":
        return cls(
            trajectories=tuple(
                TrajectoryPayload.from_trajectory(trajectory)
                for trajectory in group.trajectories
            ),
            exceptions=tuple(
                exception.model_dump(mode="json") for exception in group.exceptions
            ),
            metadata=group.metadata,
            metrics=group.metrics,
            logs=tuple(group.logs),
            collect_packing_shape=group._collect_packing_shape,
        )

    def build(self) -> TrajectoryGroup:
        group = TrajectoryGroup(
            (payload.build() for payload in self.trajectories),
            metadata=self.metadata,
            metrics=self.metrics,
            logs=list(self.logs),
        )
        group.exceptions = [
            PydanticException.model_validate(payload) for payload in self.exceptions
        ]
        group._collect_packing_shape = self.collect_packing_shape
        return group


class PackingRequest(BaseModel):
    """Current ART packing inputs; generalized loss programs are intentionally absent."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    model: RolloutModelSpec
    trajectory_groups: tuple[TrajectoryGroupPayload, ...]
    advantage_balance: float = 0.0
    allow_training_without_logprobs: bool = False
    scale_rewards: bool = True
    plot_tensors: bool = False
    packed_sequence_length: int = Field(ge=1)
    logprob_calculation_chunk_size: int = Field(default=1024, ge=1)
    include_moe_routing: bool = False
    group_ids: tuple[str, ...] = ()
    record_ids: tuple[str, ...] = ()
    min_source_version: int = Field(default=0, ge=0)
    max_source_version: int = Field(default=0, ge=0)


class PackingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    ref: PackedBatchRef | None
    packed_group_shapes: tuple[PackedGroupShape | None, ...]
