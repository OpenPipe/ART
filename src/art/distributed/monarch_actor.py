from __future__ import annotations

# This module is imported only by explicit distributed runtime construction.
from monarch.actor import Actor, endpoint  # ty: ignore[unresolved-import]

from .data_plane import BatchReservation, PackedBatchInbox, PackedBatchRef
from .rollout import RolloutInvocation


class RolloutHostService(Actor):
    """One coarse CPU rollout and packed-batch inbox actor per host."""

    def __init__(self, host_id: str, packed_batch_capacity_bytes: int) -> None:
        self.inbox = PackedBatchInbox(
            host_id=host_id, capacity_bytes=packed_batch_capacity_bytes
        )

    @endpoint
    async def run(self, invocation: RolloutInvocation):
        function = invocation.callable.resolve()
        return await function(invocation.model, invocation.scenario, invocation.config)

    @endpoint
    async def reserve_batch(self, ref: PackedBatchRef) -> BatchReservation:
        return await self.inbox.reserve(ref)

    @endpoint
    async def put_batch(
        self, reservation: BatchReservation, ref: PackedBatchRef, payload: bytes
    ) -> PackedBatchRef:
        return await self.inbox.put(reservation, ref, payload)

    @endpoint
    async def abort_batch(self, reservation_id: str) -> None:
        await self.inbox.abort(reservation_id)

    @endpoint
    async def release_batch(self, lease_id: str) -> None:
        await self.inbox.release(lease_id)

    @endpoint
    async def unlink_batch(self, batch_id: str) -> None:
        await self.inbox.unlink(batch_id)

    @endpoint
    async def stats(self):
        return self.inbox.store.stats()
