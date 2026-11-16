import asyncio

import pytest

import art


class BlockedRollout:
    """A rollout that stays in flight until released and records how it ended."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.finished = False

    async def wait(self) -> None:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.finished = True


async def assert_not_resumed(rollout: BlockedRollout) -> None:
    # Releasing a leaked rollout would let it finish after gather raised.
    rollout.release.set()
    for _ in range(5):
        await asyncio.sleep(0)
    assert rollout.cancelled
    assert not rollout.finished


async def test_group_failure_cancels_sibling_rollouts() -> None:
    blocked = BlockedRollout()

    async def slow_group() -> art.TrajectoryGroup:
        await blocked.wait()
        return art.TrajectoryGroup([art.Trajectory(reward=1)])

    async def failed_group() -> art.TrajectoryGroup:
        await blocked.started.wait()
        raise RuntimeError("rollout failed")

    with pytest.raises(RuntimeError, match="rollout failed"):
        await art.gather_trajectory_groups(
            [failed_group(), slow_group()], pbar_desc=None
        )

    await assert_not_resumed(blocked)


async def test_trajectory_failure_cancels_sibling_rollouts() -> None:
    blocked = BlockedRollout()

    async def slow_trajectory() -> art.Trajectory:
        await blocked.wait()
        return art.Trajectory(reward=1)

    async def failed_trajectory() -> art.Trajectory:
        await blocked.started.wait()
        raise RuntimeError("rollout failed")

    with pytest.raises(RuntimeError, match="rollout failed"):
        await art.gather_trajectories(
            [failed_trajectory(), slow_trajectory()], pbar_desc=None
        )

    await assert_not_resumed(blocked)


async def test_tolerated_failure_still_gathers_every_group() -> None:
    async def group(reward: float) -> art.TrajectoryGroup:
        return art.TrajectoryGroup([art.Trajectory(reward=reward)])

    async def failed_group() -> art.TrajectoryGroup:
        raise RuntimeError("rollout failed")

    groups = await art.gather_trajectory_groups(
        [group(0.0), failed_group(), group(1.0)], pbar_desc=None, max_exceptions=1
    )

    assert [g.trajectories[0].reward for g in groups] == [0.0, 1.0]
