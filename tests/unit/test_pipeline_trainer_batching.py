import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
import pytest

from art import PipelineRuntimeConfig, TrainableModel, Trajectory, TrajectoryGroup
from art.pipeline_trainer.trainer import PipelineTrainer, _PreparedPipelineItem


async def _noop_rollout(*_args: object, **_kwargs: object) -> TrajectoryGroup:
    return TrajectoryGroup([])


def _group() -> TrajectoryGroup:
    return TrajectoryGroup(
        [
            Trajectory(
                reward=reward,
                initial_policy_version=0,
                metrics={"completion_tokens": 1},
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
            "generation_id": "generation-6",
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
            run_name="pipeline-max-batch-size-test",
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


@pytest.mark.asyncio
async def test_pre_next_dispatch_hook_blocks_packed_lookahead(tmp_path: Path) -> None:
    model = TrainableModel(
        run_name="pipeline-dispatch-hook-test",
        name="pipeline-dispatch-hook-test",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
        report_metrics=[],
    )
    backend = MagicMock()

    async def train(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(step=backend.train.await_count, metrics={})

    backend.train = AsyncMock(side_effect=train)
    trainer = PipelineTrainer(
        model=model,
        backend=backend,  # type: ignore[arg-type]
        rollout_fn=_noop_rollout,
        scenarios=[],
        config={},
        pipeline=PipelineRuntimeConfig(
            num_rollout_workers=1,
            min_batch_size=1,
            max_batch_size=1,
        ),
        max_steps=2,
        eval_fn=None,
    )
    trainer._output_queue = asyncio.Queue()
    trainer._packed_queue = asyncio.Queue(maxsize=1)
    hook_started = asyncio.Event()
    release_hook = asyncio.Event()
    second_prepared = asyncio.Event()

    async def freeze_after_step(step: int) -> None:
        assert step == 1
        hook_started.set()
        await release_hook.wait()

    trainer.add_pre_next_dispatch_hook(1, freeze_after_step)
    first = _PreparedPipelineItem(
        batch=[_group()],
        discarded=0,
        zero_variance_discarded=0,
        saw_sentinel=False,
        packing_policy_step=0,
        selection_s=0.0,
        preparation_s=0.0,
        preparation_metrics={},
    )
    second = first.model_copy(
        update={"batch": [_group()], "saw_sentinel": True, "handoff": asyncio.Event()}
    )

    async def prepare() -> None:
        await trainer._packed_queue.put(first)
        await first.handoff.wait()
        second_prepared.set()
        await trainer._packed_queue.put(second)

    producer = asyncio.create_task(prepare())
    training = asyncio.create_task(trainer._training_stage())
    await asyncio.wait_for(hook_started.wait(), timeout=2.0)
    assert backend.train.await_count == 1
    assert not second_prepared.is_set()

    release_hook.set()
    await asyncio.wait_for(asyncio.gather(producer, training), timeout=2.0)
    assert backend.train.await_count == 2
