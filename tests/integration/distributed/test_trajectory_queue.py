import asyncio
from collections.abc import Callable
from unittest.mock import AsyncMock

import pytest

from art.distributed.rollout import (
    DistributedTrajectoryQueue,
    DistributedTrajectorySelection,
    _InProcessTrajectoryQueueEndpoint,
)
from art.distributed.trajectory_store import (
    TrajectoryCapacityError,
    TrajectoryEnqueueResult,
    TrajectoryGroupAnnotations,
    TrajectoryGroupDescriptor,
    TrajectoryGroupRef,
    TrajectoryQueueItem,
    TrajectoryRecordRef,
)


def _item(
    result_id: str, *, records: int = 1, byte_count: int = 1
) -> TrajectoryQueueItem:
    return TrajectoryQueueItem(
        ref=TrajectoryGroupRef(
            result_id=result_id,
            owner_actor_id="owner",
            lease_id=f"lease-{result_id}",
            records=tuple(
                TrajectoryRecordRef(
                    record_id=f"{result_id}-{index}",
                    owner_actor_id="owner",
                    byte_count=1,
                )
                for index in range(records)
            ),
            descriptor=TrajectoryGroupDescriptor(
                grouping_key=result_id,
                trajectory_count=records,
                exception_count=0,
                rewards=(0.0,) * records,
                initial_policy_versions=(0,) * records,
                completion_tokens=(1.0,) * records,
                policy_token_counts={},
                trajectory_initial_policy_versions=(0,) * records,
                trajectory_final_policy_versions=(0,) * records,
                trajectory_policy_token_counts=({},) * records,
                trajectory_metrics=({},) * records,
                trajectory_metadata=({},) * records,
                group_metadata={},
                group_metrics={},
                exceptions=(),
                byte_count=byte_count,
            ),
        ),
        annotations=TrajectoryGroupAnnotations(
            initial_policy_version=0,
            final_policy_version=0,
        ),
    )


async def _put(queue: DistributedTrajectoryQueue, item: TrajectoryQueueItem) -> bool:
    accepted, _ = await queue.put(
        item.ref,
        metadata={},
        initial_policy_version=0,
        final_policy_version=0,
        rollout_wall_s=0.0,
        actor_idle_s=0.0,
    )
    return accepted


async def _wait_until(condition: Callable[[], bool]) -> None:
    for _ in range(100):
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not reached")


class _ObservedQueueEndpoint(_InProcessTrajectoryQueueEndpoint):
    def __init__(self) -> None:
        super().__init__()
        self.enqueue_results: list[TrajectoryEnqueueResult] = []

    async def enqueue(
        self, queue_id: str, item: TrajectoryQueueItem
    ) -> TrajectoryEnqueueResult:
        result = await super().enqueue(queue_id, item)
        self.enqueue_results.append(result)
        return result


@pytest.mark.asyncio
async def test_packing_occupancy_backpressures_until_lease_release() -> None:
    endpoint = _ObservedQueueEndpoint()
    queue = DistributedTrajectoryQueue(
        endpoint=endpoint,
        owner_endpoints={"owner": AsyncMock()},
        maxsize=6,
        capacity_records=8,
        capacity_bytes=8,
    )
    await queue.start()
    for index in range(6):
        assert await _put(queue, _item(f"initial-{index}"))

    groups, closed = await queue.get_many(6, wait=True)
    assert len(groups) == 6
    assert not closed
    snapshot = await queue.snapshot()
    assert (
        snapshot.ready_groups,
        snapshot.packing_groups,
        snapshot.packed_groups,
        len(snapshot.items),
        snapshot.max_ready_groups,
    ) == (0, 6, 0, 6, 6)

    pending_take = asyncio.create_task(queue.get_many(2, wait=True))
    await _wait_until(lambda: queue._minimum_take_size == 2)
    blocked_put = asyncio.create_task(_put(queue, _item("blocked")))
    await _wait_until(lambda: len(endpoint.enqueue_results) == 7)
    assert endpoint.enqueue_results[-1].status == "full"
    await asyncio.sleep(0)
    assert not blocked_put.done()

    selections = []
    for group in groups:
        selection = group._distributed_lease
        assert isinstance(selection, DistributedTrajectorySelection)
        selections.append(selection)
    await queue.mark_packed(selections, "generation")
    await queue.release_selections(
        selections,
        disposition="consumed",
        generation_id="generation",
    )
    assert await blocked_put
    assert await _put(queue, _item("unblocks-minimum"))

    acquired, closed = await pending_take
    assert len(acquired) == 2
    assert not closed
    for group in acquired:
        await queue.discard_group(group)
    await queue.close()


@pytest.mark.parametrize(
    ("capacity_records", "capacity_bytes", "blocker"),
    ((1, 8, "record capacity"), (8, 1, "byte capacity")),
)
@pytest.mark.asyncio
async def test_ready_occupancy_makes_limit_failure_sticky(
    capacity_records: int, capacity_bytes: int, blocker: str
) -> None:
    queue = DistributedTrajectoryQueue(
        endpoint=_InProcessTrajectoryQueueEndpoint(),
        owner_endpoints={"owner": AsyncMock()},
        maxsize=6,
        capacity_records=capacity_records,
        capacity_bytes=capacity_bytes,
    )
    await queue.start()
    assert await _put(queue, _item("ready"))
    pending_take = asyncio.create_task(queue.get_many(2, wait=True))
    await _wait_until(lambda: queue._minimum_take_size == 2)

    with pytest.raises(TrajectoryCapacityError) as blocked_error:
        await _put(queue, _item("blocked"))
    with pytest.raises(TrajectoryCapacityError) as take_error:
        await pending_take
    with pytest.raises(TrajectoryCapacityError) as sticky_error:
        await _put(queue, _item("later"))
    assert blocker in str(blocked_error.value)
    assert str(take_error.value) == str(blocked_error.value)
    assert str(sticky_error.value) == str(blocked_error.value)
    await queue.close()


@pytest.mark.asyncio
async def test_minimum_larger_than_group_capacity_fails_promptly() -> None:
    queue = DistributedTrajectoryQueue(
        endpoint=_InProcessTrajectoryQueueEndpoint(),
        owner_endpoints={},
        maxsize=6,
        capacity_records=8,
        capacity_bytes=8,
    )
    await queue.start()
    with pytest.raises(
        TrajectoryCapacityError,
        match="minimum acquisition requires 7 trajectory groups",
    ):
        await queue.get_many(7, wait=True)
    await queue.close()


@pytest.mark.asyncio
async def test_pending_minimum_defers_shrink_until_cancelled() -> None:
    queue = DistributedTrajectoryQueue(
        endpoint=_InProcessTrajectoryQueueEndpoint(),
        owner_endpoints={},
        maxsize=6,
        capacity_records=8,
        capacity_bytes=8,
    )
    await queue.start()
    pending_take = asyncio.create_task(queue.get_many(2, wait=True))
    await _wait_until(lambda: queue._minimum_take_size == 2)

    queue.set_maxsize(1)
    assert (await queue.snapshot()).max_ready_groups == 2
    pending_take.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending_take
    assert (await queue.snapshot()).max_ready_groups == 1
    await queue.close()
