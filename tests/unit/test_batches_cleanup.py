import asyncio
import gc
from typing import Any

import pytest

import art


async def completed_group(identity: int) -> art.TrajectoryGroup:
    return art.TrajectoryGroup(
        [art.Trajectory(reward=float(identity), metadata={"identity": identity})]
    )


class BlockedRollout:
    """A rollout that stays in flight until released and records how it ended."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.finished = False

    async def __call__(self) -> art.TrajectoryGroup:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.finished = True
        return await completed_group(-1)


async def assert_not_resumed(rollout: BlockedRollout) -> None:
    # Releasing a leaked rollout would let it finish after its consumer stopped.
    rollout.release.set()
    for _ in range(5):
        await asyncio.sleep(0)
    assert rollout.cancelled
    assert not rollout.finished


async def test_closing_iterator_cancels_started_rollouts() -> None:
    blocked = BlockedRollout()

    async def first_group() -> art.TrajectoryGroup:
        await blocked.started.wait()
        return await completed_group(0)

    batches = art.trajectory_group_batches(
        [first_group(), blocked()],
        batch_size=1,
        max_concurrent_batches=2,
        pbar_desc=None,
    )
    assert len(await anext(batches)) == 1
    await batches.aclose()

    await assert_not_resumed(blocked)


async def test_rollout_failure_cancels_started_siblings() -> None:
    blocked = BlockedRollout()

    async def failed_group() -> art.TrajectoryGroup:
        await blocked.started.wait()
        raise RuntimeError("rollout failed")

    batches = art.trajectory_group_batches(
        [failed_group(), blocked()], batch_size=2, pbar_desc=None
    )
    with pytest.raises(RuntimeError, match="rollout failed"):
        await anext(batches)

    await assert_not_resumed(blocked)


async def test_cancelled_consumer_cancels_started_rollouts() -> None:
    blocked = BlockedRollout()
    batches = art.trajectory_group_batches([blocked()], batch_size=1, pbar_desc=None)
    consumer = asyncio.ensure_future(anext(batches))
    await asyncio.wait_for(blocked.started.wait(), timeout=5)

    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    await assert_not_resumed(blocked)


async def test_failure_retrieves_sibling_exceptions() -> None:
    unretrieved: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _, context: unretrieved.append(context))

    async def failed_group(message: str) -> art.TrajectoryGroup:
        raise RuntimeError(message)

    try:
        batches = art.trajectory_group_batches(
            [failed_group("first"), failed_group("second")],
            batch_size=2,
            pbar_desc=None,
        )
        with pytest.raises(RuntimeError):
            await anext(batches)
        del batches
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert unretrieved == []
