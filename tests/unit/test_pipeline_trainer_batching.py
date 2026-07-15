import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from art import TrainableModel, Trajectory, TrajectoryGroup
from art.pipeline_trainer.trainer import PipelineTrainer


async def _noop_rollout(*_args: object, **_kwargs: object) -> TrajectoryGroup:
    return TrajectoryGroup([])


def _make_group() -> TrajectoryGroup:
    return TrajectoryGroup(
        [
            Trajectory(
                reward=reward,
                initial_policy_version=0,
                messages_and_choices=[
                    {"role": "user", "content": f"prompt-{idx}"},
                    {"role": "assistant", "content": f"answer-{idx}"},
                ],
            )
            for idx, reward in enumerate([0.0, 1.0])
        ]
    )


@pytest.mark.asyncio
async def test_collect_batch_respects_max_batch_size(tmp_path: Path) -> None:
    model = TrainableModel(
        name="pipeline-max-batch-size-test",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    trainer = PipelineTrainer(
        model=model,
        backend=MagicMock(),  # type: ignore[arg-type]
        rollout_fn=_noop_rollout,
        scenarios=[],
        config={},
        num_rollout_workers=1,
        min_batch_size=1,
        max_batch_size=2,
        max_steps=1,
        eval_fn=None,
    )
    trainer._output_queue = asyncio.Queue()

    first = _make_group()
    second = _make_group()
    third = _make_group()
    await trainer._output_queue.put(first)
    await trainer._output_queue.put(second)
    await trainer._output_queue.put(third)
    await trainer._output_queue.put(None)

    batch, discarded, saw_sentinel = await trainer._collect_batch(current_step=0)

    assert batch == [first, second]
    assert discarded == 0
    assert not saw_sentinel

    batch, discarded, saw_sentinel = await trainer._collect_batch(current_step=0)

    assert batch == [third]
    assert discarded == 0
    assert saw_sentinel
