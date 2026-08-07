from __future__ import annotations

from collections.abc import Iterable
import secrets
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel, ConfigDict, Field, model_validator

from art.pipeline_tuner.config import PackedGroupShape
from art.preprocessing.moe_routing import (
    ART_MOE_ROUTING_METADATA_KEY,
    NUM_EXPERTS_KEY,
    ROUTED_EXPERTS_KEY,
    MoeRouteArray,
    moe_route_dtype,
)
from art.trajectories import (
    MetadataValue,
    PydanticException,
    Trajectory,
    TrajectoryGroup,
)

from .data_plane import PackedBatchRef
from .rollout import RolloutModelSpec
from .trajectory_store import (
    TrajectoryBatchTransfer,
    TrajectoryGroupBundle,
    TrajectoryQueueItem,
)

if TYPE_CHECKING:
    from art.model import TrainableModel


class _ChoiceRoutingPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata: dict[str, Any]
    dtype: Literal["uint8", "uint16"]
    shape: tuple[int, int, int]
    data: bytes

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "_ChoiceRoutingPayload":
        routes = metadata[ROUTED_EXPERTS_KEY]
        if not isinstance(routes, np.ndarray) or routes.dtype not in {
            np.dtype(np.uint8),
            np.dtype(np.uint16),
        }:
            raise RuntimeError("routed experts must be a uint8 or uint16 array")
        if routes.ndim != 3:
            raise RuntimeError(f"routed experts must have rank 3, got {routes.shape}")
        num_experts = int(metadata.get(NUM_EXPERTS_KEY, 0))
        if routes.dtype != moe_route_dtype(num_experts):
            raise RuntimeError("routed experts do not match exact expert count")
        dtype: Literal["uint8", "uint16"] = (
            "uint8" if routes.dtype == np.dtype(np.uint8) else "uint16"
        )
        return cls(
            metadata={
                key: value
                for key, value in metadata.items()
                if key != ROUTED_EXPERTS_KEY
            },
            dtype=dtype,
            shape=routes.shape,
            data=routes.tobytes(),
        )

    def build(self) -> dict[str, Any]:
        num_experts = int(self.metadata[NUM_EXPERTS_KEY])
        routes = MoeRouteArray(
            np.frombuffer(self.data, dtype=self.dtype).reshape(self.shape),
            num_experts=num_experts,
        )
        return {**self.metadata, ROUTED_EXPERTS_KEY: routes}


class TrajectoryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    payload: dict[str, Any]
    choice_positions: tuple[int, ...] = ()
    additional_history_choice_positions: tuple[tuple[int, ...], ...] = ()
    choice_routing_metadata: dict[int, _ChoiceRoutingPayload] = Field(
        default_factory=dict
    )
    additional_history_choice_routing_metadata: tuple[
        dict[int, _ChoiceRoutingPayload], ...
    ] = ()
    exchange_choice_routing_metadata: tuple[dict[int, _ChoiceRoutingPayload], ...] = ()

    @classmethod
    def from_trajectory(cls, trajectory: Trajectory) -> "TrajectoryPayload":
        choice_routing = _choice_routing_metadata(trajectory.messages_and_choices)
        history_routing = tuple(
            _choice_routing_metadata(history.messages_and_choices)
            for history in trajectory.additional_histories
        )
        exchange_routing = tuple(
            _choice_routing_metadata(exchange.response.choices)
            for exchange in trajectory.exchanges.chat_completions
        )
        exclude: dict[str, Any] = {
            "messages_and_choices": _routing_exclude(choice_routing),
            "additional_histories": {
                index: {
                    "messages_and_choices": _routing_exclude(routing),
                }
                for index, routing in enumerate(history_routing)
            },
        }
        return cls(
            payload=trajectory.model_dump(mode="json", exclude=exclude),
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
            choice_routing_metadata=choice_routing,
            additional_history_choice_routing_metadata=history_routing,
            exchange_choice_routing_metadata=exchange_routing,
        )

    def build(self) -> Trajectory:
        payload = dict(self.payload)
        messages = list(payload.get("messages_and_choices", []))
        for index in self.choice_positions:
            messages[index] = _build_choice(
                messages[index], self.choice_routing_metadata.get(index)
            )
        payload["messages_and_choices"] = messages
        histories = [
            dict(history) for history in payload.get("additional_histories", [])
        ]
        for history, positions, routing in zip(
            histories,
            self.additional_history_choice_positions,
            self.additional_history_choice_routing_metadata,
            strict=True,
        ):
            messages = list(history["messages_and_choices"])
            for index in positions:
                messages[index] = _build_choice(messages[index], routing.get(index))
            history["messages_and_choices"] = messages
        payload["additional_histories"] = histories
        exchanges = dict(payload.get("exchanges", {}))
        chat_exchanges = [
            dict(exchange) for exchange in exchanges.get("chat_completions", [])
        ]
        for exchange, routing in zip(
            chat_exchanges,
            self.exchange_choice_routing_metadata,
            strict=True,
        ):
            response = dict(exchange["response"])
            choices = list(response["choices"])
            for index, metadata in routing.items():
                choices[index] = _build_choice(choices[index], metadata)
            response["choices"] = choices
            exchange["response"] = response
        exchanges["chat_completions"] = chat_exchanges
        payload["exchanges"] = exchanges
        return Trajectory.model_validate(payload)


def _choice_routing_metadata(items: list[Any]) -> dict[int, _ChoiceRoutingPayload]:
    return {
        index: _ChoiceRoutingPayload.from_metadata(metadata)
        for index, item in enumerate(items)
        if isinstance(item, Choice)
        and isinstance(
            metadata := (item.model_extra or {}).get(ART_MOE_ROUTING_METADATA_KEY),
            dict,
        )
    }


def _routing_exclude(
    routing: dict[int, _ChoiceRoutingPayload],
) -> dict[int, set[str]]:
    return {index: {ART_MOE_ROUTING_METADATA_KEY} for index in routing}


def _build_choice(payload: Any, routing: _ChoiceRoutingPayload | None) -> Choice:
    choice = Choice.model_validate(payload)
    if routing is not None:
        if choice.model_extra is None:
            raise RuntimeError("OpenAI Choice.model_extra is unavailable")
        choice.model_extra[ART_MOE_ROUTING_METADATA_KEY] = routing.build()
    return choice


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
    generation_id: str = Field(min_length=1)
    trajectory_groups: tuple[TrajectoryGroupBundle, ...] = ()
    trajectory_transfer: TrajectoryBatchTransfer | None = None
    trajectory_sources: tuple[TrajectoryQueueItem, ...] = ()
    trajectory_log_path: str | None = None
    advantage_balance: float = 0.0
    allow_training_without_logprobs: bool = False
    scale_rewards: bool = True
    plot_tensors: bool = False
    packed_sequence_length: int = Field(ge=1)
    logprob_calculation_chunk_size: int = Field(default=1024, ge=1)
    include_moe_routing: bool = False
    collect_packing_shapes: bool = False
    group_ids: tuple[str, ...] = ()
    record_ids: tuple[str, ...] = ()
    min_source_version: int = Field(default=0, ge=0)
    max_source_version: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_trajectory_input(self) -> "PackingRequest":
        inputs = (
            bool(self.trajectory_groups),
            self.trajectory_transfer is not None,
            bool(self.trajectory_sources),
        )
        if sum(inputs) != 1:
            raise ValueError("packing requires exactly one trajectory input")
        return self

    @classmethod
    def from_groups(
        cls,
        model: TrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        *,
        packed_sequence_length: int,
        advantage_balance: float = 0.0,
        allow_training_without_logprobs: bool = False,
        scale_rewards: bool = True,
        plot_tensors: bool = False,
        logprob_calculation_chunk_size: int = 1024,
        include_moe_routing: bool = False,
        group_ids: tuple[str, ...] = (),
        record_ids: tuple[str, ...] = (),
        min_source_version: int = 0,
        max_source_version: int = 0,
    ) -> "PackingRequest":
        """Build a serializable packing request from public ART objects."""

        return cls(
            model=RolloutModelSpec.from_model(model),
            generation_id=secrets.token_hex(16),
            trajectory_groups=tuple(
                TrajectoryGroupBundle.from_group(group) for group in trajectory_groups
            ),
            advantage_balance=advantage_balance,
            allow_training_without_logprobs=allow_training_without_logprobs,
            scale_rewards=scale_rewards,
            plot_tensors=plot_tensors,
            packed_sequence_length=packed_sequence_length,
            logprob_calculation_chunk_size=logprob_calculation_chunk_size,
            include_moe_routing=include_moe_routing,
            collect_packing_shapes=any(
                group._collect_packing_shape for group in trajectory_groups
            ),
            group_ids=group_ids,
            record_ids=record_ids,
            min_source_version=min_source_version,
            max_source_version=max_source_version,
        )


class PackingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    ref: PackedBatchRef | None
    packed_group_shapes: tuple[PackedGroupShape | None, ...]
    trainable_assistant_tokens: int = Field(default=0, ge=0)
    loss_bearing_tokens: int = Field(default=0, ge=0)
    non_padding_tokens: int = Field(default=0, ge=0)
    trajectory_log_path: str | None = None
    trajectory_fetch_s: float = Field(default=0.0, ge=0)
    packing_core_s: float = Field(default=0.0, ge=0)
    trajectory_log_wait_s: float = Field(default=0.0, ge=0)
    packed_batch_finalize_s: float = Field(default=0.0, ge=0)
    generation_id: str = Field(min_length=1)
