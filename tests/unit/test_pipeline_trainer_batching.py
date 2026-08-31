import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
import pytest

from art import PipelineRuntimeConfig, TrainableModel, Trajectory, TrajectoryGroup
from art.distributed.rollout import (
    DistributedTrajectoryQueue,
    _InProcessTrajectoryQueueEndpoint,
)
from art.pipeline_trainer.trainer import PipelineTrainer, _PreparedPipelineItem
from art.pipeline_tuner.config import PipelineTuneSettings


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


def _packed_trainer(
    tmp_path: Path, backend: Any, *, run_name: str
) -> PipelineTrainer[Any, Any]:
    trainer = PipelineTrainer(
        model=TrainableModel(
            run_name=run_name,
            name=run_name,
            project="pipeline-tests",
            base_model="test-model",
            base_path=str(tmp_path),
            report_metrics=[],
        ),
        backend=backend,
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
    return trainer


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
async def test_async_packing_prepares_beyond_ready_batch(tmp_path: Path) -> None:
    backend = MagicMock()
    second_prepared = asyncio.Event()

    async def prepare(
        _model: object, batch: list[TrajectoryGroup], **_kwargs: object
    ) -> dict[str, float]:
        if backend.prepare_pipeline_batch.await_count == 2:
            second_prepared.set()
        return {}

    backend.prepare_pipeline_batch = AsyncMock(side_effect=prepare)
    backend.prepare_pipeline_commands = None
    trainer = _packed_trainer(tmp_path, backend, run_name="pipeline-ready-ahead-test")
    groups = [_group(), _group()]
    for group in groups:
        await trainer._output_queue.put(group)
    await trainer._output_queue.put(None)

    packing = asyncio.create_task(trainer._packing_stage())
    await asyncio.wait_for(second_prepared.wait(), timeout=2.0)

    first = await trainer._packed_queue.get()
    second = await trainer._packed_queue.get()
    terminal = await trainer._packed_queue.get()
    await asyncio.wait_for(packing, timeout=2.0)

    assert first is not None and first.batch == groups[:1]
    assert not first.handoff.is_set()
    assert second is not None and second.batch == groups[1:]
    assert terminal is None


def test_async_packing_queue_reserves_ready_batch_across_target_shrink(
    tmp_path: Path,
) -> None:
    trainer = _packed_trainer(
        tmp_path, MagicMock(), run_name="pipeline-queue-reserve-test"
    )
    trainer.max_batch_size = 33
    output_queue = DistributedTrajectoryQueue(
        endpoint=_InProcessTrajectoryQueueEndpoint(),
        owner_endpoints={},
        maxsize=66,
        capacity_records=128,
        capacity_bytes=128,
    )
    trainer._output_queue = output_queue

    trainer.apply_pipeline_settings(
        PipelineTuneSettings(
            num_rollout_workers=1,
            min_batch_size=31,
            max_batch_size=31,
            target_groups_per_step=31,
            queue_maxsize=31,
        )
    )

    assert trainer.queue_maxsize == 31
    assert output_queue.maxsize == 97


@pytest.mark.asyncio
async def test_command_preparation_is_bounded_to_one_ready_batch(
    tmp_path: Path,
) -> None:
    backend = MagicMock()
    contexts: list[SimpleNamespace] = []

    async def prepare_commands(*_args: object, **_kwargs: object) -> SimpleNamespace:
        context = SimpleNamespace(preparation_metrics={}, abort=AsyncMock())
        contexts.append(context)
        return context

    backend.prepare_pipeline_commands = AsyncMock(side_effect=prepare_commands)
    backend.prepare_pipeline_batch = None
    trainer = _packed_trainer(tmp_path, backend, run_name="pipeline-command-ahead")
    for group in (_group(), _group()):
        await trainer._output_queue.put(group)
    await trainer._output_queue.put(None)

    packing = asyncio.create_task(trainer._packing_stage())
    first = await asyncio.wait_for(trainer._packed_queue.get(), timeout=2.0)
    assert first is not None
    assert len(contexts) == 1
    await asyncio.sleep(0)
    assert len(contexts) == 1

    first.handoff.set()
    second = await asyncio.wait_for(trainer._packed_queue.get(), timeout=2.0)
    assert second is not None
    assert len(contexts) == 2
    second.handoff.set()
    assert await asyncio.wait_for(trainer._packed_queue.get(), timeout=2.0) is None
    await asyncio.wait_for(packing, timeout=2.0)


@pytest.mark.asyncio
async def test_pre_next_dispatch_hook_blocks_packed_lookahead(tmp_path: Path) -> None:
    backend = MagicMock()

    async def train(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(step=backend.train.await_count, metrics={})

    backend.train = AsyncMock(side_effect=train)
    trainer = _packed_trainer(tmp_path, backend, run_name="pipeline-dispatch-hook-test")
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


@pytest.mark.asyncio
async def test_command_lookahead_overlaps_current_command_execution(
    tmp_path: Path,
) -> None:
    backend = MagicMock()
    backend.discard_pipeline_batch = AsyncMock()
    trainer = _packed_trainer(tmp_path, backend, run_name="pipeline-lookahead-test")
    first_started = asyncio.Event()
    first_admitted = asyncio.Event()
    finish_first = asyncio.Event()
    second_queued = asyncio.Event()

    class CommandContext:
        def __init__(
            self,
            step: int,
            *,
            started: asyncio.Event,
            admitted: asyncio.Event,
            finished: asyncio.Event,
        ) -> None:
            self.step = step
            self.started = started
            self.admitted = admitted
            self.finished = finished

        async def complete(
            self,
            next_train_dispatched: asyncio.Event | None,
            next_batch_handoff: asyncio.Event | None,
        ) -> SimpleNamespace:
            self.started.set()
            if next_train_dispatched is not None:
                next_train_dispatched.set()
            if next_batch_handoff is not None:
                next_batch_handoff.set()
            await self.admitted.wait()
            await self.finished.wait()
            return SimpleNamespace(step=self.step, metrics={})

        async def abort(self) -> None:
            raise AssertionError("command context was not consumed")

    async def block_settlement(
        _item: object, next_train_dispatched: asyncio.Event
    ) -> None:
        await next_train_dispatched.wait()

    trainer._finalize_post_train = block_settlement  # type: ignore[method-assign]
    ready = asyncio.Event()
    ready.set()
    first = _PreparedPipelineItem(
        batch=[_group()],
        discarded=0,
        zero_variance_discarded=0,
        saw_sentinel=False,
        packing_policy_step=0,
        selection_s=0.0,
        preparation_s=0.0,
        preparation_metrics={},
        command_context=CommandContext(
            1,
            started=first_started,
            admitted=first_admitted,
            finished=finish_first,
        ),
    )
    second = first.model_copy(
        update={
            "batch": [_group()],
            "saw_sentinel": True,
            "handoff": asyncio.Event(),
            "command_context": CommandContext(
                2,
                started=asyncio.Event(),
                admitted=ready,
                finished=ready,
            ),
        }
    )

    async def prepare() -> None:
        await trainer._packed_queue.put(first)
        await first.handoff.wait()
        await trainer._packed_queue.put(second)
        second_queued.set()
        await second.handoff.wait()

    producer = asyncio.create_task(prepare())
    training = asyncio.create_task(trainer._training_stage())
    await asyncio.wait_for(first_started.wait(), timeout=2.0)
    await asyncio.wait_for(second_queued.wait(), timeout=2.0)
    assert first.handoff.is_set()
    assert not first_admitted.is_set()

    first_admitted.set()
    assert not finish_first.is_set()
    finish_first.set()
    await asyncio.wait_for(asyncio.gather(producer, training), timeout=2.0)
