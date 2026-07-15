from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .data_plane import PackedBatchRef
from .packing import PackingRequest, PackingResult
from .rollout import (
    RolloutHostEndpoint,
    RolloutInvocation,
    RolloutResult,
)
from .vllm_replica import HostMemberLaunchRequest, HostMemberState


class RemoteCallError(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["cancelled", "capacity", "input", "lease", "serving", "internal"]
    error_type: str
    message: str
    traceback: str


class RemoteCallResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    value: Any = None
    error: RemoteCallError | None = None


def unwrap_remote_call(result: RemoteCallResult) -> Any:
    if result.error is None:
        return result.value
    error = result.error
    message = f"remote {error.error_type}: {error.message}\n{error.traceback}"
    if error.kind == "cancelled":
        raise asyncio.CancelledError(message)
    if error.kind == "serving":
        from art.errors import LocalServingUnavailableError

        raise LocalServingUnavailableError(message)
    if error.kind == "capacity":
        from .data_plane import PackedBatchCapacityError

        raise PackedBatchCapacityError(message)
    if error.kind == "lease":
        from .data_plane import PackedBatchLeaseError

        raise PackedBatchLeaseError(message)
    if error.kind == "input":
        raise ValueError(message)
    raise RuntimeError(message)


async def call_remote(endpoint: Any, *args: Any) -> Any:
    return unwrap_remote_call(await endpoint.call_one(*args))


class MonarchRolloutHostEndpoint(RolloutHostEndpoint):
    def __init__(self, actor: Any, *, owns_actor: bool = False) -> None:
        self.actor = actor
        self.owns_actor = owns_actor

    async def run(self, invocation: RolloutInvocation) -> RolloutResult:
        await self.actor.initialized
        return await call_remote(self.actor.run, invocation)

    async def close(self) -> None:
        if self.owns_actor:
            await call_remote(self.actor.close)


class MonarchVllmHostLauncher:
    def __init__(self, actor: Any) -> None:
        self.actor = actor

    async def start_member(self, request: HostMemberLaunchRequest) -> HostMemberState:
        return await call_remote(self.actor.start_vllm_member, request)

    async def member_state(
        self, replica_id: str, member_id: str, generation: int
    ) -> HostMemberState:
        return await call_remote(
            self.actor.vllm_member_state, replica_id, member_id, generation
        )

    async def stop_member(
        self, replica_id: str, member_id: str, generation: int
    ) -> None:
        await call_remote(
            self.actor.stop_vllm_member, replica_id, member_id, generation
        )


class MonarchPackedBatchInbox:
    def __init__(self, actor: Any) -> None:
        self.actor = actor

    async def receive_rdma(
        self, ref: PackedBatchRef, rdma_buffer: Any, *, timeout_s: float
    ) -> PackedBatchRef:
        return await call_remote(
            self.actor.receive_rdma_batch, ref, rdma_buffer, timeout_s
        )

    async def drop(self, ref: PackedBatchRef) -> None:
        await call_remote(self.actor.drop_batch_ref, ref)


class MonarchPackedBatchSource:
    def __init__(self, actor: Any) -> None:
        self.actor = actor

    async def publish(self, ref: PackedBatchRef) -> Any:
        return await call_remote(self.actor.publish_batch, ref)

    async def drop(self, batch_id: str) -> None:
        await call_remote(self.actor.drop_batch, batch_id)

    async def note_transmitted(self, byte_count: int) -> None:
        await call_remote(self.actor.note_batch_transmitted, byte_count)


class MonarchPackingEndpoint:
    def __init__(self, actor: Any) -> None:
        self.actor = actor

    async def pack(self, request: PackingRequest) -> PackingResult:
        return await call_remote(self.actor.pack_batch, request)
