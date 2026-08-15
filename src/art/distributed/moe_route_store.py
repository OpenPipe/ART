from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .data_plane import (
    ByteStreamPublisher,
    ByteStreamServerLoop,
    ByteStreamTransfer,
    receive_byte_stream,
)

if TYPE_CHECKING:
    from .packing import TrajectoryGroupPayload, _ChoiceRoutingPayload

RouteScope = Literal["messages", "additional_history", "exchange"]
RouteKey = tuple[int, RouteScope, int, int]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MoeRouteSlice(_Contract):
    trajectory_index: int = Field(ge=0)
    scope: RouteScope
    scope_index: int = Field(ge=0)
    choice_index: int = Field(ge=0)
    offset: int = Field(ge=0)
    byte_count: int = Field(ge=1)


class MoeRouteObjectPayload(_Contract):
    data: bytes = Field(min_length=1)
    slices: tuple[MoeRouteSlice, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_slices(self) -> "MoeRouteObjectPayload":
        _validate_route_slices(self.slices, len(self.data))
        return self


class MoeRouteGroupPayload(_Contract):
    objects: tuple[MoeRouteObjectPayload, ...] = ()

    @model_validator(mode="after")
    def _validate_positions(self) -> "MoeRouteGroupPayload":
        positions = [
            (
                item.trajectory_index,
                item.scope,
                item.scope_index,
                item.choice_index,
            )
            for value in self.objects
            for item in value.slices
        ]
        if len(positions) != len(set(positions)):
            raise ValueError("route group contains duplicate trajectory positions")
        return self


class MoeRouteObjectLayout(_Contract):
    byte_count: int = Field(ge=1)
    slices: tuple[MoeRouteSlice, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_slices(self) -> "MoeRouteObjectLayout":
        _validate_route_slices(self.slices, self.byte_count)
        return self


class MoeRouteBatchTransfer(_Contract):
    stream: ByteStreamTransfer
    groups: tuple[tuple[MoeRouteObjectLayout, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_layout(self) -> "MoeRouteBatchTransfer":
        byte_count = sum(value.byte_count for group in self.groups for value in group)
        if byte_count != self.stream.byte_count:
            raise ValueError("route stream layout does not match its payload")
        return self

    async def receive_groups(
        self, *, timeout_s: float
    ) -> tuple[MoeRouteGroupPayload, ...]:
        payload = await receive_byte_stream(self.stream, timeout_s=timeout_s)
        return await asyncio.to_thread(self._build_groups, payload)

    def _build_groups(self, payload: bytearray) -> tuple[MoeRouteGroupPayload, ...]:
        view = memoryview(payload)
        offset = 0
        groups = []
        try:
            for group in self.groups:
                objects = []
                for layout in group:
                    end = offset + layout.byte_count
                    objects.append(
                        MoeRouteObjectPayload(
                            data=bytes(view[offset:end]), slices=layout.slices
                        )
                    )
                    offset = end
                groups.append(MoeRouteGroupPayload(objects=tuple(objects)))
        finally:
            view.release()
        return tuple(groups)


async def publish_moe_route_groups(
    groups: tuple[MoeRouteGroupPayload, ...],
    *,
    stream_id: str,
    advertise_host: str,
    on_sent: Callable[[], None] | None = None,
    server_loop: ByteStreamServerLoop | None = None,
) -> tuple[MoeRouteBatchTransfer, ByteStreamPublisher]:
    if not any(group.objects for group in groups):
        raise ValueError("route publication requires at least one route object")
    publisher = await ByteStreamPublisher.create(
        stream_id,
        tuple(value.data for group in groups for value in group.objects),
        advertise_host=advertise_host,
        on_sent=on_sent,
        server_loop=server_loop,
    )
    try:
        transfer = MoeRouteBatchTransfer(
            stream=publisher.transfer,
            groups=tuple(
                tuple(
                    MoeRouteObjectLayout(
                        byte_count=len(value.data), slices=value.slices
                    )
                    for value in group.objects
                )
                for group in groups
            ),
        )
    except BaseException:
        await publisher.close()
        raise
    return transfer, publisher


def hydrate_trajectory_group_routes(
    payload: TrajectoryGroupPayload, routes: MoeRouteGroupPayload
) -> TrajectoryGroupPayload:
    routed: dict[RouteKey, bytes] = {
        (
            item.trajectory_index,
            item.scope,
            item.scope_index,
            item.choice_index,
        ): value.data[item.offset : item.offset + item.byte_count]
        for value in routes.objects
        for item in value.slices
    }
    trajectories = []
    for trajectory_index, trajectory in enumerate(payload.trajectories):
        histories = tuple(
            _hydrate_route_map(
                values,
                trajectory_index=trajectory_index,
                scope="additional_history",
                scope_index=index,
                routed=routed,
            )
            for index, values in enumerate(
                trajectory.additional_history_choice_routing_metadata
            )
        )
        exchanges = tuple(
            _hydrate_route_map(
                values,
                trajectory_index=trajectory_index,
                scope="exchange",
                scope_index=index,
                routed=routed,
            )
            for index, values in enumerate(trajectory.exchange_choice_routing_metadata)
        )
        trajectories.append(
            trajectory.model_copy(
                update={
                    "choice_routing_metadata": _hydrate_route_map(
                        trajectory.choice_routing_metadata,
                        trajectory_index=trajectory_index,
                        scope="messages",
                        scope_index=0,
                        routed=routed,
                    ),
                    "additional_history_choice_routing_metadata": histories,
                    "exchange_choice_routing_metadata": exchanges,
                }
            )
        )
    if routed:
        raise RuntimeError("route object contains unbound trajectory slices")
    return payload.model_copy(update={"trajectories": tuple(trajectories)})


def _hydrate_route_map(
    values: Mapping[int, _ChoiceRoutingPayload],
    *,
    trajectory_index: int,
    scope: RouteScope,
    scope_index: int,
    routed: dict[RouteKey, bytes],
) -> dict[int, _ChoiceRoutingPayload]:
    hydrated = {}
    for choice_index, route in values.items():
        key = (trajectory_index, scope, scope_index, choice_index)
        if route.data:
            if key in routed:
                raise RuntimeError("inline routed experts also name a route object")
            hydrated[choice_index] = route
            continue
        try:
            data = routed.pop(key)
        except KeyError:
            raise RuntimeError(
                "routed experts are missing their route object slice"
            ) from None
        itemsize = 1 if route.dtype == "uint8" else 2
        expected = itemsize
        for extent in route.shape:
            expected *= extent
        if len(data) != expected:
            raise RuntimeError("route object slice does not match routed-expert shape")
        hydrated[choice_index] = route.model_copy(update={"data": data})
    return hydrated


def _validate_route_slices(slices: tuple[MoeRouteSlice, ...], byte_count: int) -> None:
    cursor = 0
    for value in sorted(slices, key=lambda item: item.offset):
        if value.offset != cursor:
            raise ValueError("route slices must exactly partition their object")
        cursor += value.byte_count
    if cursor != byte_count:
        raise ValueError("route slices must exactly partition their object")
