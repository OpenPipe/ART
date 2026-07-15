from __future__ import annotations

from typing import Any

from .data_plane import BatchReservation, PackedBatchRef
from .monarch_bootstrap import monarch_identifier
from .rollout import (
    DistributedRolloutExecutor,
    InstalledAsyncCallable,
    RolloutHostEndpoint,
    RolloutInvocation,
    RolloutResult,
)
from .specs import RuntimeTopology


class _MonarchRolloutHostEndpoint(RolloutHostEndpoint):
    def __init__(self, actor: Any) -> None:
        self.actor = actor

    async def run(self, invocation: RolloutInvocation) -> RolloutResult:
        return await self.actor.run.call_one(invocation)

    async def close(self) -> None:
        await self.actor.close.call_one()


class MonarchPackedBatchInbox:
    def __init__(self, actor: Any) -> None:
        self.actor = actor

    async def reserve(self, ref: PackedBatchRef) -> BatchReservation:
        return await self.actor.reserve_batch.call_one(ref)

    async def put(
        self, reservation: BatchReservation, ref: PackedBatchRef, payload: bytes
    ) -> PackedBatchRef:
        return await self.actor.put_batch.call_one(reservation, ref, payload)

    async def abort(self, reservation_id: str) -> None:
        await self.actor.abort_batch.call_one(reservation_id)

    async def release(self, lease_id: str) -> None:
        await self.actor.release_batch.call_one(lease_id)

    async def unlink(self, batch_id: str) -> None:
        await self.actor.unlink_batch.call_one(batch_id)


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
        endpoints[host_id] = _MonarchRolloutHostEndpoint(actor)
    executor = DistributedRolloutExecutor(
        callable=rollout_callable,
        hosts=endpoints,
        host_slots={host_id: host_by_id[host_id].cpu_slots for host_id in endpoints},
        target_workers=target_workers,
    )
    return executor, tuple(procs)
