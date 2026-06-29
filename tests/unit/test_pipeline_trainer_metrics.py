import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from art import PipelineRuntimeConfig, TrainableModel, Trajectory, TrajectoryGroup
from art.pipeline_trainer.trainer import PipelineTrainer


def _make_group(
    rewards: list[float], *, initial_policy_version: int | None
) -> TrajectoryGroup:
    return TrajectoryGroup(
        [
            Trajectory(
                reward=reward,
                initial_policy_version=initial_policy_version,
                messages_and_choices=[
                    {"role": "user", "content": f"prompt-{idx}"},
                    {"role": "assistant", "content": f"answer-{idx}"},
                ],
            )
            for idx, reward in enumerate(rewards)
        ]
    )


def test_pipeline_trainer_score_uses_start_policy_age_and_tau(tmp_path: Path) -> None:
    model = TrainableModel(
        name="pipeline-score-metrics-test",
        project="pipeline-score-metrics-test",
        base_model="test-model",
        base_path=str(tmp_path),
        report_metrics=[],
    )
    trainer = PipelineTrainer(
        model=model,
        backend=MagicMock(),
        rollout_fn=lambda *_args, **_kwargs: asyncio.sleep(0),
        scenarios=[],
        config={},
        pipeline=PipelineRuntimeConfig(
            num_rollout_workers=1,
            min_batch_size=1,
            max_batch_size=1,
            score_reference_groups_per_step=1,
        ),
        eval_fn=None,
        max_steps=1,
    )
    group = _make_group([0.25, 0.75], initial_policy_version=8)

    metrics = trainer._score_metrics(
        10,
        [group],
        step_seconds=2.0,
        result_metrics={"data/step_trainer_assistant_tokens": 100.0},
    )

    assert metrics["offpolicy/token_weighted_policy_age_steps"] == 2.0
    assert metrics["sample_efficiency/freshness_discount"] == pytest.approx(
        1.0 / (1.0 + 2.0 / 4.0)
    )
    assert metrics["objective/score_default"] == pytest.approx(
        50.0 * (1.0 / (1.0 + 2.0 / 4.0))
    )


@pytest.mark.asyncio
async def test_pipeline_trainer_logs_explicit_stale_and_zero_variance_metrics(
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
        rollout_fn=lambda *_args, **_kwargs: asyncio.sleep(0),
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

    await trainer._output_queue.put(
        _make_group([0.25, 0.75], initial_policy_version=-1)
    )
    await trainer._output_queue.put(_make_group([1.0, 1.0], initial_policy_version=0))
    await trainer._output_queue.put(_make_group([0.0, 1.0], initial_policy_version=0))
    await trainer._output_queue.put(None)

    await trainer._training_stage()

    history_path = (
        tmp_path
        / "pipeline-discard-metrics-test"
        / "models"
        / "pipeline-discard-metrics-test"
        / "history.jsonl"
    )
    with open(history_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    train_row = next(row for row in rows if "train/reward" in row)
    zero_variance_row = next(
        row for row in rows if any(key.startswith("discarded/") for key in row)
    )

    assert "train/discarded_stale_groups" in train_row
    assert "train/discarded_stale_samples" not in train_row
    assert "discarded/reward" in zero_variance_row
