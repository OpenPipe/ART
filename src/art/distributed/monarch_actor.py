from __future__ import annotations

from collections import OrderedDict

# This module is imported only by explicit distributed runtime construction.
from monarch.actor import Actor, endpoint  # ty: ignore[unresolved-import]

from .data_plane import BatchReservation, PackedBatchInbox, PackedBatchRef
from .rollout import RolloutInvocation, RolloutResult


class RolloutHostService(Actor):
    """One coarse CPU rollout and packed-batch inbox actor per host."""

    def __init__(self, host_id: str, packed_batch_capacity_bytes: int) -> None:
        self.inbox = PackedBatchInbox(
            host_id=host_id, capacity_bytes=packed_batch_capacity_bytes
        )
        self._models = OrderedDict()

    @endpoint
    async def run(self, invocation: RolloutInvocation):
        from art.metrics import MetricsBuilder

        key = invocation.model.cache_key
        model = self._models.get(key)
        if model is None:
            model = invocation.model.build()
            self._models[key] = model
            if len(self._models) > 16:
                _, evicted = self._models.popitem(last=False)
                await evicted._reset_inference_runtime()
        else:
            self._models.move_to_end(key)
        function = invocation.callable.resolve()
        builder = MetricsBuilder(cost_context="train")
        token = builder.activate()
        try:
            value = await function(model, invocation.scenario, invocation.config)
        finally:
            token.var.reset(token)
        return RolloutResult(value=value, metrics=await builder.drain_pending())

    @endpoint
    async def close(self) -> None:
        for model in self._models.values():
            await model._reset_inference_runtime()
        self._models.clear()

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
