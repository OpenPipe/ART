from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from .data_plane import BatchReservation, PackedBatchRef
from .monarch_bootstrap import activate_child_virtualenv, monarch_identifier
from .packing import PackingRequest, PackingResult
from .rollout import (
    DistributedRolloutExecutor,
    InstalledAsyncCallable,
    RolloutHostEndpoint,
    RolloutInvocation,
    RolloutResult,
)
from .specs import RuntimeTopology
from .vllm_replica import HostMemberLaunchRequest, HostMemberState


class RemoteCallError(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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
    raise RuntimeError(f"remote {error.error_type}: {error.message}\n{error.traceback}")


async def call_remote(endpoint: Any, *args: Any) -> Any:
    return unwrap_remote_call(await endpoint.call_one(*args))


class MonarchRolloutHostEndpoint(RolloutHostEndpoint):
    def __init__(self, actor: Any, *, owns_actor: bool = False) -> None:
        self.actor = actor
        self.owns_actor = owns_actor

    async def run(self, invocation: RolloutInvocation) -> RolloutResult:
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

    async def allocate_port(self) -> int:
        return int(await call_remote(self.actor.allocate_port))


class MonarchPackedBatchInbox:
    def __init__(self, actor: Any) -> None:
        self.actor = actor

    async def reserve(self, ref: PackedBatchRef) -> BatchReservation:
        return await call_remote(self.actor.reserve_batch, ref)

    async def put(
        self, reservation: BatchReservation, ref: PackedBatchRef, payload: bytes
    ) -> PackedBatchRef:
        return await call_remote(self.actor.put_batch, reservation, ref, payload)

    async def receive_rdma(
        self, ref: PackedBatchRef, rdma_buffer: Any, *, timeout_s: float
    ) -> PackedBatchRef:
        return await call_remote(
            self.actor.receive_rdma_batch, ref, rdma_buffer, timeout_s
        )

    async def abort(self, reservation_id: str) -> None:
        await call_remote(self.actor.abort_batch, reservation_id)

    async def release(self, lease_id: str) -> None:
        await call_remote(self.actor.release_batch, lease_id)

    async def unlink(self, batch_id: str) -> None:
        await call_remote(self.actor.unlink_batch, batch_id)


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


async def create_rollout_executor(
    *,
    host_mesh: Any,
    topology: RuntimeTopology,
    rollout_callable: InstalledAsyncCallable,
    target_workers: int,
    packed_batch_capacity_bytes: int,
) -> tuple[DistributedRolloutExecutor, tuple[Any, ...]]:
    """Spawn one optional-Monarch actor per rollout host and return its executor."""

    # Lazy import keeps `import art` and local PipelineTrainer use Monarch-free.
    from .monarch_actor import RolloutHostService

    host_by_id = {host.host_id: host for host in topology.cluster.hosts}
    indices = {
        host.host_id: index
        for index, host in enumerate(topology.cluster.hosts)
        if host.host_id in topology.rollout_host_ids
    }
    procs = []
    endpoints: dict[str, RolloutHostEndpoint] = {}
    for host_id, index in indices.items():
        proc = host_mesh.slice(hosts=index).spawn_procs(
            per_host={"rollout": 1},
            bootstrap=activate_child_virtualenv,
            name=monarch_identifier(f"art_rollout_{host_id}"),
        )
        actor = proc.spawn(
            monarch_identifier(f"rollout_host_service_{host_id}"),
            RolloutHostService,
            host_id,
            packed_batch_capacity_bytes,
        )
        await actor.initialized
        procs.append(proc)
        endpoints[host_id] = MonarchRolloutHostEndpoint(actor, owns_actor=True)
    executor = DistributedRolloutExecutor(
        callable=rollout_callable,
        hosts=endpoints,
        host_slots={host_id: host_by_id[host_id].cpu_slots for host_id in endpoints},
        target_workers=target_workers,
    )
    return executor, tuple(procs)
