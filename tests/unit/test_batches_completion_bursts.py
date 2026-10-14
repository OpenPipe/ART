import asyncio
from collections.abc import Coroutine, Sequence
from typing import Any

import pytest

import art


async def completed_group(identity: int) -> art.TrajectoryGroup:
    return art.TrajectoryGroup(
        [art.Trajectory(reward=float(identity), metadata={"identity": identity})]
    )


async def collect_batches(
    groups: Sequence[Coroutine[Any, Any, art.TrajectoryGroup]],
    *,
    max_concurrent_batches: int = 4,
    skip_batches: int = 0,
    max_batch_exceptions: int = 0,
) -> list[list[art.TrajectoryGroup]]:
    try:
        return [
            batch
            async for batch in art.trajectory_group_batches(
                groups,
                batch_size=2,
                max_concurrent_batches=max_concurrent_batches,
                skip_batches=skip_batches,
                max_batch_exceptions=max_batch_exceptions,
                pbar_desc=None,
            )
        ]
    finally:
        # Skipped inputs are never awaited by the generator and remain caller-owned.
        for group in groups:
            group.close()


def identities(batches: list[list[art.TrajectoryGroup]]) -> list[int]:
    return sorted(
        group.trajectories[0].metadata["identity"]
        for batch in batches
        for group in batch
    )


@pytest.mark.parametrize(
    ("count", "concurrency", "skip", "expected_sizes"),
    [
        (0, 4, 0, []),
        (1, 4, 0, [1]),
        (3, 4, 0, [2, 1]),
        (7, 4, 0, [2, 2, 2, 1]),
        (8, 4, 0, [2, 2, 2, 2]),
        (12, 4, 0, [2, 2, 2, 2, 2, 2]),
        (8, 1, 0, [2, 2, 2, 2]),
        (8, 4, 1, [2, 2, 2]),
    ],
)
async def test_completion_bursts_keep_batches_bounded(
    count: int, concurrency: int, skip: int, expected_sizes: list[int]
) -> None:
    batches = await collect_batches(
        [completed_group(identity) for identity in range(count)],
        max_concurrent_batches=concurrency,
        skip_batches=skip,
    )

    assert [len(batch) for batch in batches] == expected_sizes
    assert identities(batches) == list(range(skip * 2, count))


async def failed_group() -> art.TrajectoryGroup:
    raise RuntimeError("rollout failed")


async def test_completion_burst_preserves_tolerated_exception() -> None:
    # All inputs finish in the same wait, within the original exception budget.
    groups = [completed_group(identity) for identity in range(7)]
    groups.append(failed_group())
    batches = await collect_batches(groups, max_batch_exceptions=1)

    assert [len(batch) for batch in batches] == [2, 2, 2, 1]
    assert identities(batches) == list(range(7))


async def test_completion_burst_propagates_exception_over_budget() -> None:
    groups = [completed_group(identity) for identity in range(7)]
    groups.append(failed_group())

    with pytest.raises(RuntimeError, match="rollout failed"):
        await collect_batches(groups)


async def test_buffered_batch_checks_completed_pending_failure() -> None:
    release_failure = asyncio.Event()
    failure_finished = asyncio.Event()

    async def delayed_failure() -> art.TrajectoryGroup:
        await release_failure.wait()
        failure_finished.set()
        raise RuntimeError("pending rollout failed")

    groups = [completed_group(identity) for identity in range(4)]
    groups.append(delayed_failure())
    batches = art.trajectory_group_batches(groups, batch_size=2, pbar_desc=None)
    try:
        assert len(await anext(batches)) == 2
        release_failure.set()
        await asyncio.wait_for(failure_finished.wait(), timeout=5)
        with pytest.raises(RuntimeError, match="pending rollout failed"):
            await anext(batches)
    finally:
        release_failure.set()
        try:
            async for _ in batches:
                pass
        except RuntimeError:
            pass
        for group in groups:
            group.close()


async def test_buffered_batch_refills_rollout_slots() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def next_group(identity: int) -> art.TrajectoryGroup:
        started.set()
        await release.wait()
        return await completed_group(identity)

    groups = [completed_group(identity) for identity in range(8)]
    groups.extend(next_group(identity) for identity in range(8, 10))
    batches = art.trajectory_group_batches(groups, batch_size=2, pbar_desc=None)
    observed = []
    second = None
    try:
        observed.append(await anext(batches))
        second = asyncio.ensure_future(anext(batches))
        # A barrier checks scheduling while the consumer handles a buffered batch.
        await asyncio.wait_for(started.wait(), timeout=5)
        release.set()
        observed.append(await second)
        observed.extend([batch async for batch in batches])
        assert identities(observed) == list(range(10))
    finally:
        release.set()
        if second is not None:
            await second
        async for _ in batches:
            pass
        for group in groups:
            group.close()
