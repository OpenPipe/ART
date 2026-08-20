import asyncio
from types import SimpleNamespace

import pytest

from art.megatron.training.slot import MegatronTrainingSlot


@pytest.mark.asyncio
async def test_packed_batch_release_tasks_are_bounded() -> None:
    gates = [asyncio.Event() for _ in range(3)]
    active = 0
    max_active = 0

    async def release_batch(batch) -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await gates[batch.packing_generation_id].wait()
        finally:
            active -= 1

    slot = MegatronTrainingSlot.__new__(MegatronTrainingSlot)
    slot.runtime = SimpleNamespace(release_batch=release_batch)
    slot._batch_releases = set()
    slot._batch_release_failures = []
    batches = tuple(
        SimpleNamespace(packing_generation_id=index) for index in range(3)
    )

    await slot._release_batch_soon(batches[0])
    await slot._release_batch_soon(batches[1])
    third = asyncio.create_task(slot._release_batch_soon(batches[2]))
    await asyncio.sleep(0)
    assert not third.done()
    assert active == 2

    gates[0].set()
    await third
    await asyncio.sleep(0)
    assert max_active == 2

    gates[1].set()
    gates[2].set()
    await asyncio.gather(*tuple(slot._batch_releases))
