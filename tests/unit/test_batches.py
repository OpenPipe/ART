"""Tests for trajectory_group_batches."""

import art
from art import Trajectory, TrajectoryGroup


def _group(reward: float) -> TrajectoryGroup:
    return TrajectoryGroup(
        [
            Trajectory(
                messages_and_choices=[{"role": "user", "content": "hi"}],
                reward=reward,
            )
        ]
    )


async def _ok(reward: float) -> TrajectoryGroup:
    return _group(reward)


async def _fail() -> TrajectoryGroup:
    raise RuntimeError("rollout failed")


async def test_max_batch_exceptions_applies_to_every_batch() -> None:
    batches = [
        [trajectory.reward for group in batch for trajectory in group]
        async for batch in art.trajectory_group_batches(
            [_ok(1.0), _fail(), _ok(2.0)],
            batch_size=1,
            max_batch_exceptions=1,
            max_concurrent_batches=1,
            pbar_desc=None,
        )
    ]

    assert batches == [[1.0], [2.0]]
