from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from msgspec import msgpack
import numpy as np
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from art.pipeline_tuner.config import PackedGroupShape
from art.preprocessing.moe_routing import (
    ART_MOE_ROUTING_METADATA_KEY,
    NUM_EXPERTS_KEY,
    ROUTED_EXPERTS_KEY,
    MoeRouteArray,
    MoeRouteSegments,
    moe_route_dtype,
)
from art.preprocessing.pack import DEFAULT_MIN_PREFIX_TREE_SHARED_SEGMENT_LENGTH
from art.training.token_matrix import (
    NamedLossRequest,
    RetainedTokenRoutes,
    TokenMatrixBatch,
    validate_token_matrix_batch,
)
from art.trajectories import (
    MetadataValue,
    PydanticException,
    Trajectory,
    TrajectoryGroup,
)
from art.vllm_route_transport import (
    RETAINED_ROUTE_BUNDLE_KEY,
    RetainedRouteBundleRef,
    RouteBundleBatchTransfer,
    RouteBundleLayout,
    retained_route_bundles_from_groups,
    unique_retained_route_bundles,
)

from .data_plane import ByteStreamTransfer, PackedBatchRef, receive_byte_stream

_TOKEN_MATRIX_BATCH_ADAPTER = TypeAdapter(TokenMatrixBatch)


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
            "messages_and_choices": _routing_exclude(trajectory.messages_and_choices),
            "additional_histories": {
                index: {
                    "messages_and_choices": _routing_exclude(
                        history.messages_and_choices
                    ),
                }
                for index, history in enumerate(trajectory.additional_histories)
            },
            "exchanges": {
                "chat_completions": {
                    index: {
                        "response": {
                            "choices": _routing_exclude(exchange.response.choices)
                        }
                    }
                    for index, exchange in enumerate(
                        trajectory.exchanges.chat_completions
                    )
                }
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
    routing = {}
    for index, item in enumerate(items):
        if not isinstance(item, Choice):
            continue
        metadata = (item.model_extra or {}).get(ART_MOE_ROUTING_METADATA_KEY)
        if not isinstance(metadata, dict):
            continue
        if RETAINED_ROUTE_BUNDLE_KEY in metadata:
            continue
        routing[index] = _ChoiceRoutingPayload.from_metadata(metadata)
    return routing


def _routing_exclude(
    items: list[Any],
) -> dict[int, set[str]]:
    return {
        index: {ART_MOE_ROUTING_METADATA_KEY}
        for index, item in enumerate(items)
        if isinstance(item, Choice)
        and ART_MOE_ROUTING_METADATA_KEY in (item.model_extra or {})
    }


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
    retained_route_bundles: tuple[RetainedRouteBundleRef, ...] = ()

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
            retained_route_bundles=retained_route_bundles_from_groups((group,)),
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


class TokenMatrixBatchTransfer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stream: ByteStreamTransfer

    async def receive(self, *, timeout_s: float) -> TokenMatrixBatch:
        payload = await receive_byte_stream(self.stream, timeout_s=timeout_s)
        return _TOKEN_MATRIX_BATCH_ADAPTER.validate_python(msgpack.decode(payload))


def encode_token_matrix_batch(batch: TokenMatrixBatch) -> bytes:
    return msgpack.encode(batch.model_dump(mode="python"))


class PackingRequest(BaseModel):
    """One canonical TokenMatrix request before its controller-to-host transfer."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    batch: TokenMatrixBatch
    loss: NamedLossRequest
    return_token_logprobs: bool = True
    generation_id: str = Field(min_length=1)
    retained_route_bundles: tuple[RetainedRouteBundleRef, ...] = ()
    packed_sequence_length: int = Field(ge=1)
    collect_packing_shapes: bool = False
    min_prefix_tree_shared_segment_length: int = Field(
        default=DEFAULT_MIN_PREFIX_TREE_SHARED_SEGMENT_LENGTH,
        ge=0,
    )
    compute_content_sha256: bool = False

    @model_validator(mode="after")
    def _validate_request(self) -> "PackingRequest":
        validate_token_matrix_batch(
            self.batch,
            self.loss,
            output_rows=("learner_logprobs",) if self.return_token_logprobs else (),
        )
        refs = unique_retained_route_bundles(self.retained_route_bundles)
        if len(refs) != len(self.retained_route_bundles):
            raise ValueError("packing repeats a retained route bundle")
        referenced = retained_route_bundles_from_token_matrix_batch(self.batch)
        if {ref.layout.bundle_id: ref for ref in refs} != {
            ref.layout.bundle_id: ref for ref in referenced
        }:
            raise ValueError(
                "retained route sidecar must exactly match TokenMatrix references"
            )
        return self


class PackingTransferRequest(BaseModel):
    """Transport-only request received by the packing host."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    batch_transfer: TokenMatrixBatchTransfer
    loss: NamedLossRequest
    return_token_logprobs: bool
    generation_id: str = Field(min_length=1)
    route_bundle_transfer: RouteBundleBatchTransfer | None = None
    packed_sequence_length: int = Field(ge=1)
    collect_packing_shapes: bool = False
    min_prefix_tree_shared_segment_length: int = Field(
        default=DEFAULT_MIN_PREFIX_TREE_SHARED_SEGMENT_LENGTH,
        ge=0,
    )
    compute_content_sha256: bool = False

    @classmethod
    def from_request(
        cls,
        request: PackingRequest,
        *,
        batch_transfer: TokenMatrixBatchTransfer,
        route_bundle_transfer: RouteBundleBatchTransfer | None,
    ) -> "PackingTransferRequest":
        return cls(
            batch_transfer=batch_transfer,
            loss=request.loss,
            return_token_logprobs=request.return_token_logprobs,
            generation_id=request.generation_id,
            route_bundle_transfer=route_bundle_transfer,
            packed_sequence_length=request.packed_sequence_length,
            collect_packing_shapes=request.collect_packing_shapes,
            min_prefix_tree_shared_segment_length=(
                request.min_prefix_tree_shared_segment_length
            ),
            compute_content_sha256=request.compute_content_sha256,
        )


def retained_route_bundles_from_token_matrix_batch(
    batch: TokenMatrixBatch,
) -> tuple[RetainedRouteBundleRef, ...]:
    return unique_retained_route_bundles(
        RetainedRouteBundleRef.model_validate(route.bundle)
        for route in batch.routes
        if isinstance(route, RetainedTokenRoutes)
    )


def resolve_retained_token_matrix_routes(
    batch: TokenMatrixBatch,
    layouts: tuple[RouteBundleLayout, ...],
    payload: bytes | bytearray | memoryview,
) -> dict[str, MoeRouteArray | MoeRouteSegments]:
    """Validate transferred retained bundles and select matrix-aligned routes."""

    retained = tuple(
        route for route in batch.routes if isinstance(route, RetainedTokenRoutes)
    )
    refs = retained_route_bundles_from_token_matrix_batch(batch)
    refs_by_id = {ref.layout.bundle_id: ref for ref in refs}
    layouts_by_id = {layout.bundle_id: layout for layout in layouts}
    if len(layouts_by_id) != len(layouts):
        raise ValueError("retained route transfer repeats a bundle")
    if set(layouts_by_id) != set(refs_by_id):
        raise ValueError("retained route transfer differs from TokenMatrix references")
    for bundle_id, layout in layouts_by_id.items():
        if refs_by_id[bundle_id].layout != layout:
            raise ValueError("retained route transfer changed a referenced layout")

    selected: dict[str, list[RetainedTokenRoutes]] = {}
    for route in retained:
        ref = RetainedRouteBundleRef.model_validate(route.bundle)
        selected.setdefault(ref.layout.bundle_id, []).append(route)

    view = memoryview(payload).cast("B").toreadonly()
    offset = 0
    resolved: dict[str, MoeRouteArray | MoeRouteSegments] = {}
    for layout in layouts:
        end = offset + layout.byte_count
        if end > len(view):
            raise RuntimeError("retained route transfer ended before its layout")
        chunk = view[offset:end]
        if hashlib.sha256(chunk).hexdigest() != layout.sha256:
            raise RuntimeError("retained route bundle changed its exact digest")
        choices = {choice.choice_index: choice for choice in layout.choices}
        for route in selected[layout.bundle_id]:
            choice = choices.get(route.choice_index)
            if choice is None:
                raise ValueError("retained TokenMatrix route selects an unknown choice")
            matrix = batch.matrix(route.matrix_id)
            token_ids = tuple(
                int(value) for value in matrix.row("token_ids").dense_values()
            )
            token_sha256 = hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode()
            ).hexdigest()
            if choice.shape[0] != matrix.token_count or token_sha256 != (
                choice.token_ids_sha256
            ):
                raise RuntimeError("retained routes do not match TokenMatrix tokens")
            dtype = np.dtype("u1" if choice.dtype == "uint8" else "<u2")
            resolved[route.matrix_id] = MoeRouteArray(
                np.frombuffer(
                    chunk[choice.offset : choice.offset + choice.byte_count],
                    dtype=dtype,
                ).reshape(choice.shape),
                num_experts=layout.num_experts,
            )
        offset = end
    if offset != len(view):
        raise RuntimeError("retained route transfer exceeded its layouts")
    return resolved


class PackingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    ref: PackedBatchRef
    packed_group_shapes: tuple[PackedGroupShape, ...]
    batch_fetch_s: float = Field(default=0.0, ge=0)
    route_fetch_s: float = Field(default=0.0, ge=0)
    route_transfer_backend: Literal["stream", "local", "nixl"] | None = None
    packing_core_s: float = Field(default=0.0, ge=0)
    packed_batch_finalize_s: float = Field(default=0.0, ge=0)
    generation_id: str = Field(min_length=1)
