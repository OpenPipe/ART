import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
import pytest

from art import (
    PipelineAutotuneConfig,
    PipelineRuntimeConfig,
    TrainableModel,
    Trajectory,
    TrajectoryGroup,
)
from art.pipeline_trainer.trainer import PipelineTrainer


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


def _make_trainer(
    tmp_path: Path,
    *,
    max_steps_off_policy: int | None = 4,
    limit_mean_steps_off_policy: float | None = None,
) -> PipelineTrainer:
    return PipelineTrainer(
        model=TrainableModel(
            name="pipeline-freshness-test",
            project="pipeline-tests",
            base_model="test-model",
            base_path=str(tmp_path),
        ),
        backend=MagicMock(),  # type: ignore[arg-type]
        rollout_fn=lambda *_args, **_kwargs: asyncio.sleep(0),
        scenarios=[],
        config={},
        pipeline=PipelineRuntimeConfig(
            num_rollout_workers=1,
            min_batch_size=1,
            max_batch_size=2,
            max_steps_off_policy=max_steps_off_policy,
            limit_mean_steps_off_policy=limit_mean_steps_off_policy,
        ),
        max_steps=1,
        eval_fn=None,
    )


def _choice_with_policy_spans(
    *spans: tuple[int, int, int],
) -> Choice:
    choice = Choice(
        index=0,
        finish_reason="stop",
        message=ChatCompletionMessage(role="assistant", content="answer"),
    )
    extra = cast(dict[str, Any], choice.model_extra)
    extra["policy_token_spans"] = [
        {
            "start_token": start,
            "end_token": end,
            "policy_version": policy_version,
            "lora_slot": "slot",
            "update_seq": idx + 1,
        }
        for idx, (start, end, policy_version) in enumerate(spans)
    ]
    return choice


def test_unset_max_batch_size_defaults_to_min_batch_size(tmp_path: Path) -> None:
    trainer = PipelineTrainer(
        model=TrainableModel(
            name="pipeline-default-batch-size-test",
            project="pipeline-tests",
            base_model="test-model",
            base_path=str(tmp_path),
        ),
        backend=MagicMock(),  # type: ignore[arg-type]
        rollout_fn=lambda *_args, **_kwargs: asyncio.sleep(0),
        scenarios=[],
        config={},
        pipeline=PipelineRuntimeConfig(
            num_rollout_workers=1,
            min_batch_size=4,
        ),
        max_steps=1,
        eval_fn=None,
    )

    assert trainer.max_batch_size == 4


def test_autotune_rejects_pipeline_runtime_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="autotuning is enabled"):
        PipelineTrainer(
            model=TrainableModel(
                name="pipeline-autotune-config-test",
                project="pipeline-tests",
                base_model="test-model",
                base_path=str(tmp_path),
            ),
            backend=MagicMock(),  # type: ignore[arg-type]
            rollout_fn=lambda *_args, **_kwargs: asyncio.sleep(0),
            scenarios=[],
            config={},
            pipeline=PipelineRuntimeConfig(num_rollout_workers=1),
            autotune=PipelineAutotuneConfig(mode="online"),
            max_steps=1,
            eval_fn=None,
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
        pipeline=PipelineRuntimeConfig(
            num_rollout_workers=1,
            min_batch_size=1,
            max_batch_size=2,
        ),
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


def test_mean_steps_off_policy_uses_policy_token_spans(tmp_path: Path) -> None:
    trainer = _make_trainer(
        tmp_path,
        max_steps_off_policy=None,
        limit_mean_steps_off_policy=1.0,
    )
    group = TrajectoryGroup(
        [
            Trajectory(
                reward=0.0,
                initial_policy_version=4,
                messages_and_choices=[
                    {"role": "user", "content": "prompt"},
                    _choice_with_policy_spans((0, 8, 7)),
                ],
            ),
            Trajectory(
                reward=1.0,
                initial_policy_version=4,
                messages_and_choices=[
                    {"role": "user", "content": "prompt"},
                    _choice_with_policy_spans((0, 4, 6), (4, 8, 7)),
                ],
            ),
        ]
    )

    assert trainer._group_mean_steps_off_policy(7, group) == 0.25
    assert not trainer._is_group_stale(group, 7)


def test_mean_steps_off_policy_can_replace_hard_max(tmp_path: Path) -> None:
    group = TrajectoryGroup(
        [
            Trajectory(
                reward=0.0,
                initial_policy_version=4,
                messages_and_choices=[
                    {"role": "user", "content": "prompt"},
                    _choice_with_policy_spans((0, 8, 7)),
                ],
            ),
            Trajectory(
                reward=1.0,
                initial_policy_version=4,
                messages_and_choices=[
                    {"role": "user", "content": "prompt"},
                    _choice_with_policy_spans((0, 8, 7)),
                ],
            ),
        ]
    )

    mean_only = _make_trainer(
        tmp_path,
        max_steps_off_policy=None,
        limit_mean_steps_off_policy=1.0,
    )
    both_limits = _make_trainer(
        tmp_path,
        max_steps_off_policy=2,
        limit_mean_steps_off_policy=1.0,
    )

    assert not mean_only._is_group_stale(group, 7)
    assert both_limits._is_group_stale(group, 7)


def test_mean_steps_off_policy_falls_back_to_initial_policy(tmp_path: Path) -> None:
    trainer = _make_trainer(
        tmp_path,
        max_steps_off_policy=None,
        limit_mean_steps_off_policy=2.5,
    )
    fresh = TrajectoryGroup(
        [
            Trajectory(reward=0.0, initial_policy_version=5),
            Trajectory(reward=1.0, initial_policy_version=5),
        ]
    )
    stale = TrajectoryGroup(
        [
            Trajectory(reward=0.0, initial_policy_version=4),
            Trajectory(reward=1.0, initial_policy_version=4),
        ]
    )

    assert trainer._group_mean_steps_off_policy(7, fresh) == 2.0
    assert not trainer._is_group_stale(fresh, 7)
    assert trainer._group_mean_steps_off_policy(7, stale) == 3.0
    assert trainer._is_group_stale(stale, 7)
