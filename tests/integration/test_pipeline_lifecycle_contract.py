import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from art import TrainableModel, Trajectory, TrajectoryGroup
from art.errors import LocalServingUnavailableError
from art.gather import GatherContext, record_metrics
from art.model import _OpenAIChatCompletionsProxy
from art.pipeline_trainer import PipelineRuntimeConfig, PipelineTrainer


def _group(scenario: int) -> TrajectoryGroup:
    return TrajectoryGroup(
        [
            Trajectory(
                reward=float(index),
                metrics={"completion_tokens": 1},
                messages_and_choices=[
                    {"role": "user", "content": str(scenario)},
                    {"role": "assistant", "content": str(index)},
                ],
            )
            for index in range(2)
        ]
    )


def _trainer(
    tmp_path: Path,
    rollout_fn,
    *,
    scenarios=range(5),
) -> PipelineTrainer:
    return PipelineTrainer(
        model=TrainableModel(
            run_name="pipeline-lifecycle",
            name="pipeline-lifecycle",
            project="pipeline-tests",
            base_model="test-model",
            base_path=str(tmp_path),
        ),
        backend=MagicMock(),  # type: ignore[arg-type]
        rollout_fn=rollout_fn,
        scenarios=scenarios,
        config={},
        pipeline=PipelineRuntimeConfig(
            num_rollout_workers=4,
            min_batch_size=1,
            max_batch_size=1,
        ),
        eval_fn=None,
    )


@pytest.mark.asyncio
async def test_finite_source_drains_every_assigned_rollout(tmp_path: Path) -> None:
    seen: list[int] = []

    async def rollout(_model, scenario: int, _config) -> TrajectoryGroup:
        seen.append(scenario)
        return _group(scenario)

    trainer = _trainer(tmp_path, rollout)
    trainer._output_queue = asyncio.Queue()
    await trainer._rollout_stage()

    groups = []
    while (item := await trainer._output_queue.get()) is not None:
        groups.append(item)
    assert sorted(seen) == list(range(5))
    assert len(groups) == 5


@pytest.mark.asyncio
async def test_fatal_rollout_worker_failure_is_propagated(tmp_path: Path) -> None:
    async def rollout(_model, _scenario, _config) -> TrajectoryGroup:
        raise LocalServingUnavailableError("runtime exited")

    trainer = _trainer(tmp_path, rollout, scenarios=[0])
    trainer._output_queue = asyncio.Queue()
    with pytest.raises(LocalServingUnavailableError, match="runtime exited"):
        await trainer._rollout_stage()


def test_deprecated_pipeline_kwargs_preserve_previous_defaults(tmp_path: Path) -> None:
    async def rollout(_model, scenario: int, _config) -> TrajectoryGroup:
        return _group(scenario)

    with pytest.warns(DeprecationWarning):
        trainer = PipelineTrainer(
            model=TrainableModel(
                run_name="pipeline-aliases",
                name="pipeline-aliases",
                project="pipeline-tests",
                base_model="test-model",
                base_path=str(tmp_path),
            ),
            backend=MagicMock(),  # type: ignore[arg-type]
            rollout_fn=rollout,
            scenarios=[],
            config={},
            min_batch_size=3,
            eval_fn=None,
        )
    assert trainer.max_batch_size == 30


def test_gather_preserves_caller_completion_count_without_exact_choices() -> None:
    trajectory = Trajectory(metrics={"completion_tokens": 17})
    record_metrics(GatherContext(), trajectory)
    assert trajectory.metrics["completion_tokens"] == 17


@pytest.mark.asyncio
async def test_policy_tracking_rejects_streaming_before_dispatch() -> None:
    completions = MagicMock()
    proxy = _OpenAIChatCompletionsProxy(
        completions,
        lambda _response: None,
        policy_span_mode="require",
    )
    with pytest.raises(ValueError, match="Streaming completions"):
        await proxy.create(model="model@1", stream=True)
    completions.create.assert_not_called()
