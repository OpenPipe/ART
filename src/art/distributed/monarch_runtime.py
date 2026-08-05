from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from art.trajectories import TrajectoryGroup

from .data_plane import PackedBatchRef, PackedBatchTransfer
from .packing import PackingRequest, PackingResult
from .rollout import (
    RolloutHostEndpoint,
    RolloutInvocation,
    RolloutResult,
)
from .trajectory_store import (
    TrajectoryEnqueueResult,
    TrajectoryGroupBundle,
    TrajectoryGroupRef,
    TrajectoryQueueItem,
    TrajectoryQueueResize,
    TrajectoryQueueSnapshot,
    TrajectoryQueueTake,
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

    async def materialize(self, ref: TrajectoryGroupRef) -> TrajectoryGroup:
        bundle: TrajectoryGroupBundle = await call_remote(
            self.actor.materialize_result, ref
        )
        return bundle.build()

    async def drop(self, ref: TrajectoryGroupRef) -> None:
        await call_remote(self.actor.drop_result, ref)

    async def close(self) -> None:
        if self.owns_actor:
            await call_remote(self.actor.close)


class MonarchTrajectoryQueueEndpoint:
    def __init__(self, actor: Any) -> None:
        self.actor = actor

    async def create(
        self,
        queue_id: str,
        max_ready_groups: int,
        capacity_records: int,
        capacity_bytes: int,
    ) -> None:
        await call_remote(
            self.actor.create_trajectory_queue,
            queue_id,
            max_ready_groups,
            capacity_records,
            capacity_bytes,
        )

    async def enqueue(
        self, queue_id: str, item: TrajectoryQueueItem
    ) -> TrajectoryEnqueueResult:
        return await call_remote(self.actor.enqueue_trajectory, queue_id, item)

    async def resize(self, operation: TrajectoryQueueResize) -> None:
        await call_remote(self.actor.resize_trajectory_queue, operation)

    async def take(self, queue_id: str, consumer_id: str) -> TrajectoryQueueTake:
        return await call_remote(self.actor.take_trajectory, queue_id, consumer_id)

    async def acknowledge(
        self, queue_id: str, result_id: str, consumer_id: str
    ) -> None:
        await call_remote(
            self.actor.acknowledge_trajectory, queue_id, result_id, consumer_id
        )

    async def finish(self, queue_id: str) -> None:
        await call_remote(self.actor.finish_trajectory_queue, queue_id)

    async def snapshot(self, queue_id: str) -> TrajectoryQueueSnapshot:
        return await call_remote(self.actor.trajectory_queue_snapshot, queue_id)

    async def close(self, queue_id: str) -> tuple[TrajectoryGroupRef, ...]:
        return await call_remote(self.actor.close_trajectory_queue, queue_id)


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

    async def receive(
        self, ref: PackedBatchRef, transfer: PackedBatchTransfer, *, timeout_s: float
    ) -> PackedBatchRef:
        return await call_remote(self.actor.receive_batch, ref, transfer, timeout_s)

    async def drop(self, ref: PackedBatchRef) -> None:
        await call_remote(self.actor.drop_batch_ref, ref)

    async def reclaim(self, batch_id: str, *, fence: bool) -> bool:
        return await call_remote(self.actor.reclaim_batch, batch_id, fence)


class MonarchPackedBatchSource:
    def __init__(self, actor: Any) -> None:
        self.actor = actor

    async def publish(self, ref: PackedBatchRef) -> PackedBatchTransfer:
        return await call_remote(self.actor.publish_batch, ref)

    async def drop(self, batch_id: str) -> None:
        await call_remote(self.actor.drop_batch, batch_id)

    async def note_transmitted(self, byte_count: int) -> None:
        await call_remote(self.actor.note_batch_transmitted, byte_count)


class MonarchPackingEndpoint:
    def __init__(self, actor: Any) -> None:
        self.actor = actor

    async def pack(self, request: PackingRequest, batch_id: str) -> PackingResult:
        return await call_remote(self.actor.pack_batch, request, batch_id)
