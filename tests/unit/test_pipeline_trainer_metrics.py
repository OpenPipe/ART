import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from art import PipelineRuntimeConfig, TrainableModel, Trajectory, TrajectoryGroup
from art.pipeline_trainer.trainer import PipelineTrainer


async def _noop_rollout(*_args: object, **_kwargs: object) -> TrajectoryGroup:
    return TrajectoryGroup([])


def _group(rewards: list[float], policy: int) -> TrajectoryGroup:
    return TrajectoryGroup(
        [
            Trajectory(
                reward=reward,
                initial_policy_version=policy,
                metrics={"completion_tokens": 1},
                messages_and_choices=[
                    {"role": "user", "content": f"prompt-{index}"},
                    {"role": "assistant", "content": f"answer-{index}"},
                ],
            )
            for index, reward in enumerate(rewards)
        ]
    )


@pytest.mark.asyncio
async def test_training_records_stale_and_zero_variance_discards(
    tmp_path: Path,
) -> None:
    model = TrainableModel(
        name="pipeline-discard-metrics-test",
        project="pipeline-discard-metrics-test",
        base_model="test-model",
        base_path=str(tmp_path),
        report_metrics=[],
    )
    backend = MagicMock()
    backend.train = AsyncMock(return_value=SimpleNamespace(step=1, metrics={}))
    trainer = PipelineTrainer(
        model=model,
        backend=backend,
        rollout_fn=_noop_rollout,
        scenarios=[],
        config={},
        pipeline=PipelineRuntimeConfig(
            num_rollout_workers=1,
            min_batch_size=1,
            max_batch_size=1,
            max_steps_off_policy=0,
        ),
        eval_fn=None,
        max_steps=1,
    )
    trainer._output_queue = asyncio.Queue()
    for group in (
        _group([0.25, 0.75], -1),
        _group([1.0, 1.0], 0),
        _group([0.0, 1.0], 0),
    ):
        await trainer._output_queue.put(group)
    await trainer._output_queue.put(None)

    await trainer._training_stage()

    history = Path(model._get_output_dir()) / "history.jsonl"
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    train_row = next(row for row in rows if "reward/train" in row)
    zero_variance_row = next(row for row in rows if "task/discarded/reward" in row)
    assert "discarded/cum/stale_groups" in train_row
    assert "discarded/step/stale_groups" in train_row
    assert "task/discarded/reward" in zero_variance_row
