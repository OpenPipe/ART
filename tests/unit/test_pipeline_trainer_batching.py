import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
import pytest

from art import PipelineRuntimeConfig, TrainableModel, Trajectory, TrajectoryGroup
from art.pipeline_trainer.trainer import PipelineTrainer


async def _noop_rollout(*_args: object, **_kwargs: object) -> TrajectoryGroup:
    return TrajectoryGroup([])


def _group() -> TrajectoryGroup:
    return TrajectoryGroup(
        [
            Trajectory(
                reward=reward,
                initial_policy_version=0,
                messages_and_choices=[
                    {"role": "user", "content": f"prompt-{index}"},
                    {"role": "assistant", "content": f"answer-{index}"},
                ],
            )
            for index, reward in enumerate([0.0, 1.0])
        ]
    )


def test_eval_rejects_tokens_from_another_policy() -> None:
    choice = Choice(
        index=0,
        finish_reason="stop",
        message=ChatCompletionMessage(role="assistant", content="answer"),
    )
    cast(dict[str, Any], choice.model_extra)["policy_token_spans"] = [
        {
            "start_token": 0,
            "end_token": 4,
            "policy_version": 6,
            "lora_slot": "slot",
            "update_seq": 1,
        }
    ]
    trajectory = Trajectory(
        messages_and_choices=[{"role": "user", "content": "prompt"}, choice]
    )

    with pytest.raises(RuntimeError, match="step 7 returned policy-6 tokens"):
        PipelineTrainer._validate_eval_policy_spans(7, [trajectory])


@pytest.mark.asyncio
async def test_collect_batch_respects_max_batch_size(tmp_path: Path) -> None:
    trainer = PipelineTrainer(
        model=TrainableModel(
            name="pipeline-max-batch-size-test",
            project="pipeline-tests",
            base_model="test-model",
            base_path=str(tmp_path),
        ),
        backend=MagicMock(),  # type: ignore[arg-type]
        rollout_fn=_noop_rollout,
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
    groups = [_group() for _ in range(3)]
    for group in groups:
        await trainer._output_queue.put(group)
    await trainer._output_queue.put(None)

    batch, discarded, saw_sentinel = await trainer._collect_batch(current_step=0)
    assert (batch, discarded, saw_sentinel) == (groups[:2], 0, False)

    batch, discarded, saw_sentinel = await trainer._collect_batch(current_step=0)
    assert (batch, discarded, saw_sentinel) == (groups[2:], 0, True)
