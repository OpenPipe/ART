import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from art import TrainableModel, Trajectory, TrajectoryGroup
from art.pipeline_trainer.trainer import PipelineTrainer


@pytest.fixture(autouse=True)
def _skip_backend_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PipelineTrainer, "_validate_backend_support", lambda _self: None)


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


def _make_group_with_rewards(rewards: list[float]) -> TrajectoryGroup:
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
            for idx, reward in enumerate(rewards)
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
        rollout_fn=lambda *_args, **_kwargs: asyncio.sleep(0),
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


@pytest.mark.asyncio
async def test_pipeline_trainer_forwards_kl_window_reference_step(
    tmp_path: Path,
) -> None:
    model = TrainableModel(
        name="pipeline-kl-window",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    backend = MagicMock()
    backend.train = AsyncMock(return_value=SimpleNamespace(step=801, metrics={}))

    trainer = PipelineTrainer(
        model=model,
        backend=backend,  # type: ignore[arg-type]
        rollout_fn=lambda *_args, **_kwargs: asyncio.sleep(0),
        scenarios=[],
        config={},
        num_rollout_workers=1,
        min_batch_size=1,
        max_batch_size=1,
        max_steps_off_policy=1000,
        max_steps=1,
        eval_every_n_steps=10,
        kl_penalty_coef=1.0,
        kl_window_size=50,
        kl_window_base_step=686,
    )
    trainer.state.next_training_step = 800
    trainer._output_queue = asyncio.Queue()
    await trainer._output_queue.put(_make_group())
    await trainer._output_queue.put(None)

    await trainer._training_stage()

    assert backend.train.await_args.kwargs["kl_penalty_coef"] == 1.0
    assert backend.train.await_args.kwargs["kl_penalty_reference_step"] == 746


@pytest.mark.asyncio
async def test_pipeline_trainer_kl_window_zero_uses_base_adapter_path(
    tmp_path: Path,
) -> None:
    model = TrainableModel(
        name="pipeline-kl-window-zero",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    backend = MagicMock()
    backend.train = AsyncMock(return_value=SimpleNamespace(step=1, metrics={}))
    adapter_path = str(tmp_path / "initial-checkpoint")

    trainer = PipelineTrainer(
        model=model,
        backend=backend,  # type: ignore[arg-type]
        rollout_fn=lambda *_args, **_kwargs: asyncio.sleep(0),
        scenarios=[],
        config={},
        num_rollout_workers=1,
        min_batch_size=1,
        max_batch_size=1,
        max_steps=1,
        kl_penalty_coef=0.5,
        kl_window_size=0,
        kl_window_base_adapter_path=adapter_path,
    )
    trainer._output_queue = asyncio.Queue()
    await trainer._output_queue.put(_make_group())
    await trainer._output_queue.put(None)

    await trainer._training_stage()

    assert backend.train.await_args.kwargs["kl_penalty_coef"] == 0.5
    assert backend.train.await_args.kwargs["kl_ref_adapter_path"] == adapter_path


@pytest.mark.asyncio
async def test_pipeline_trainer_rollout_worker_accepts_multiple_groups(
    tmp_path: Path,
) -> None:
    model = TrainableModel(
        name="pipeline-multi-group-rollout",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )

    group_a = _make_group_with_rewards([0.0, 1.0])
    group_b = _make_group_with_rewards([0.25, 0.75])

    async def rollout_fn(*_args: object) -> list[TrajectoryGroup]:
        return [group_a, group_b]

    trainer = PipelineTrainer(
        model=model,
        backend=MagicMock(),  # type: ignore[arg-type]
        rollout_fn=rollout_fn,
        scenarios=[{"id": "scenario-1"}],
        config={},
        num_rollout_workers=1,
        min_batch_size=1,
        max_steps=1,
    )
    trainer._output_queue = asyncio.Queue()

    await trainer._rollout_worker(worker_id=0)

    assert await trainer._output_queue.get() is group_a
    assert await trainer._output_queue.get() is group_b
    assert group_a.metadata["_art_rollout_wall_s"] >= 0
    assert group_b.metadata["_art_actor_idle_s"] >= 0


@pytest.mark.asyncio
async def test_pipeline_trainer_skips_groups_marked_skip_training(
    tmp_path: Path,
) -> None:
    model = TrainableModel(
        name="pipeline-skip-training-group",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    trainer = PipelineTrainer(
        model=model,
        backend=MagicMock(),  # type: ignore[arg-type]
        rollout_fn=lambda *_args, **_kwargs: asyncio.sleep(0),
        scenarios=[],
        config={},
        num_rollout_workers=1,
        min_batch_size=1,
        max_steps=1,
    )
    trainer._output_queue = asyncio.Queue()
    group = _make_group_with_rewards([0.0, 0.0])
    group.metadata["skip_training"] = True

    queue_wait_s = await trainer._put_output_group(group)

    assert queue_wait_s == 0.0
    assert trainer._output_queue.empty()
