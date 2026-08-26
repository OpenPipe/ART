from __future__ import annotations

import asyncio
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

from openai.types.chat.chat_completion import Choice
import pytest

from art import TrainableModel, Trajectory, TrajectoryGroup
from art.distributed.rollout import (
    DistributedTrajectoryQueue,
    DistributedTrajectorySelection,
)
from art.distributed.trajectory_store import (
    TrajectoryGroupAnnotations,
    TrajectoryGroupBundle,
    TrajectoryGroupDescriptor,
    TrajectoryGroupRef,
    TrajectoryQueueItem,
    TrajectoryQueueLease,
)
from art.metrics_taxonomy import TRAIN_GRADIENT_STEPS_KEY
from art.pipeline_trainer.checkpoint_retention import CheckpointRetentionPlan
from art.pipeline_trainer.trainer import PipelineTrainer
import art.serverless.backend as serverless_backend_module
from art.serverless.backend import ServerlessBackend
from art.serverless.data_plane import (
    decode_trajectory_group,
    encode_trajectory_group,
    prepare_training_batch,
)
from art.training.contracts import (
    CheckpointRef,
    ForwardBackwardResult,
    OperationRef,
    OptimStepResult,
    PackingOutcome,
    PolicyTokenCount,
    SamplerWeightsResult,
    SaveStateResult,
)
from art.types import ServerlessTrainResult, TrainSFTConfig


class _Sampler:
    def __init__(self) -> None:
        self.published = asyncio.Event()
        self.active_started = asyncio.Event()
        self.active_gate: asyncio.Event | None = None
        self.fail_active = False
        self.publications: list[str] = []
        self.active_versions: list[int] = []

    async def publish(self, model, weights, publication):
        del model
        self.publications.append(publication.mode)
        self.published.set()
        if publication.mode == "in_flight_lora":
            self.active_versions.append(weights.checkpoint.learner_version)
            self.active_started.set()
            if self.active_gate is not None:
                await self.active_gate.wait()
            if self.fail_active:
                raise RuntimeError("active publication failed")
        return {
            "publication/sampler": 1.0,
            "publication/sampler_step": float(weights.checkpoint.learner_version),
        }

    async def remove(self, model, publication) -> None:
        del model
        self.publications.append(f"remove:{publication.mode}")

    async def close(self) -> None:
        pass


class _Operation:
    def __init__(self, ref: OperationRef) -> None:
        self.ref = ref
        self.result_future = asyncio.get_running_loop().create_future()
        self.preparation_future = (
            asyncio.get_running_loop().create_future()
            if ref.kind == "forward_backward"
            else None
        )
        self.cancelled = False

    async def result(self):
        return await asyncio.shield(self.result_future)

    async def cancel(self) -> None:
        self.cancelled = True

    async def gradient_disposition(self):
        if self.preparation_future is None:
            raise TypeError("operation is not F/B")
        return await asyncio.shield(self.preparation_future)


class _Client:
    def __init__(self, *, fail_optimizer: bool = False) -> None:
        self.run_id = "run"
        self.next_sequence_id = 0
        self.projected_learner_version = 3
        self.operations: list[_Operation] = []
        self.fail_optimizer = fail_optimizer
        self.gate_sampler = False
        self.sampler_operations: list[tuple[_Operation, Any, int]] = []
        self.service: _Service | None = None
        self.shutdown_calls = 0
        self.abort_calls = 0
        self.encoded_batches = []

    async def stage_rl_group(self, value) -> None:
        assert self.service is not None
        assert not value.routes
        await self.service.put_training_data(
            self.run_id, value.data.ref, value.data.payload
        )

    async def ensure_route_objects(self, refs) -> None:
        assert not refs

    def forget_route_objects(self, refs) -> None:
        assert not refs

    def _operation(self, request, kind, *, transition: bool = False) -> _Operation:
        assert request.sequence_id == self.next_sequence_id
        parent = self.projected_learner_version
        reserved = parent + 1 if transition else None
        operation = _Operation(
            OperationRef(
                run_id=self.run_id,
                operation_id=f"{kind}-{request.sequence_id}",
                sequence_id=request.sequence_id,
                learner_parent_version=parent,
                reserved_output_learner_version=reserved,
                kind=kind,
            )
        )
        self.next_sequence_id += 1
        if reserved is not None:
            self.projected_learner_version = reserved
        self.operations.append(operation)
        return operation

    async def forward_backward_refs(
        self, *, request_id, batch, loss, collect_packing_shapes
    ):
        del request_id, loss
        self.batch = batch
        self.collect_packing_shapes = collect_packing_shapes
        return self._operation(
            SimpleNamespace(sequence_id=self.next_sequence_id), "forward_backward"
        )

    async def forward_backward(self, request, *, encoded_batch=None):
        self.batch = request.batch
        self.collect_packing_shapes = request.collect_packing_shapes
        self.encoded_batches.append(encoded_batch)
        return self._operation(request, "forward_backward")

    async def optim_step(self, request):
        if self.fail_optimizer:
            raise RuntimeError("optimizer admission failed")
        operation = self._operation(request, "optim_step", transition=True)
        operation.result_future.set_result(
            OptimStepResult(
                operation_id=operation.ref.operation_id,
                contributing_forward_backward_operation_ids=(
                    self.operations[-2].ref.operation_id,
                ),
                metrics={"time/optimizer_step_s": 0.1},
            )
        )
        return operation

    async def save_weights_for_sampler(self, request):
        assert request.publication.mode == "none"
        operation = self._operation(request, "save_sampler")
        self.sampler_operations.append(
            (operation, request, self.projected_learner_version)
        )
        if not self.gate_sampler:
            self.complete_sampler(len(self.sampler_operations) - 1)
        return operation

    def complete_sampler(self, index: int = 0) -> None:
        operation, request, learner_version = self.sampler_operations[index]
        operation.result_future.set_result(
            SamplerWeightsResult(
                operation_id=operation.ref.operation_id,
                checkpoint=CheckpointRef(
                    run_id=self.run_id,
                    learner_version=learner_version,
                    checkpoint_id=request.checkpoint_name,
                ),
                lora="lora",
                training_session_id="session",
                generation_id=f"generation-{learner_version}",
                lora_bytes=1,
                publication_metrics={"publication/materialization": 1.0},
            )
        )

    async def save_state(self, request):
        operation = self._operation(request, "save_state")
        operation.result_future.set_result(
            SaveStateResult(
                operation_id=operation.ref.operation_id,
                checkpoint=CheckpointRef(
                    run_id=self.run_id,
                    learner_version=self.projected_learner_version,
                    checkpoint_id=request.checkpoint_name,
                ),
                lora="lora",
                training_session_id="session",
                generation_id="generation",
                lora_bytes=1,
                optimizer_state="optimizer",
                optimizer_bytes=1,
            )
        )
        return operation

    async def shutdown(self) -> None:
        self.shutdown_calls += 1

    async def abort_result_waiters(self) -> None:
        self.abort_calls += 1


class _Queue:
    def __init__(
        self, materialized: TrajectoryGroup, *, fail_materialize: bool = False
    ):
        self.materialized = materialized
        self.fail_materialize = fail_materialize
        self.marked: list[tuple[tuple[Any, ...], str]] = []
        self.released: list[tuple[tuple[Any, ...], str, str | None]] = []
        self.release_gate: asyncio.Event | None = None

    async def receive_bundle(self, ref: Any) -> TrajectoryGroupBundle:
        del ref
        if self.fail_materialize:
            raise RuntimeError("materialization failed")
        return TrajectoryGroupBundle.from_group(self.materialized)

    async def mark_packed(
        self, selections: tuple[Any, ...], generation_id: str
    ) -> None:
        self.marked.append((selections, generation_id))

    async def release_selections(
        self,
        selections: tuple[Any, ...],
        *,
        disposition: str,
        generation_id: str | None,
    ) -> None:
        if self.release_gate is not None:
            await self.release_gate.wait()
        self.released.append((selections, disposition, generation_id))


class _Service:
    def __init__(self) -> None:
        self.data = {}
        self.deleted = []
        self.close_calls = 0

    async def put_training_data(self, run_id, ref, payload) -> None:
        assert run_id == "run"
        self.data[ref.object_id] = (ref, payload)

    async def delete_training_data(self, run_id, ref) -> None:
        assert run_id == "run"
        self.deleted.append(ref)
        self.data.pop(ref.object_id, None)

    async def close(self) -> None:
        self.close_calls += 1


class _RetentionService(_Service):
    def __init__(self, *checkpoints: Any, protected: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.checkpoints = checkpoints
        self.protected = protected
        self.apply_started = asyncio.Event()
        self.apply_gate = asyncio.Event()
        self.retention_request = None

    async def iter_checkpoint_pages(self, run_id):
        assert run_id == "run"
        yield SimpleNamespace(
            checkpoints=self.checkpoints,
            current_checkpoint_id=None,
            protected_checkpoint_ids=self.protected,
        )

    async def apply_checkpoint_retention(self, run_id, request) -> None:
        assert run_id == "run"
        self.retention_request = request
        self.apply_started.set()
        await self.apply_gate.wait()


class _DiscardBackend:
    def __init__(self) -> None:
        self.called = False

    async def discard_pipeline_group(self, *_args) -> None:
        self.called = True


def _backend(
    *, fail_optimizer: bool = False
) -> tuple[ServerlessBackend, TrainableModel, _Client, _Sampler]:
    sampler = _Sampler()
    backend = ServerlessBackend(
        training_base_url="http://training.invalid/v1",
        inference_base_url="http://inference.invalid/v1",
        sampler_manager=sampler,
        api_key="test",
        enable_expert_replay=False,
    )
    model = TrainableModel(
        name="model",
        run_name="run",
        project="scratch",
        base_model="Qwen/Qwen3.5-35B-A3B",
    )
    model._backend = backend
    client = _Client(fail_optimizer=fail_optimizer)
    backend._clients[backend._model_key(model)] = client
    backend._service = _Service()
    client.service = backend._service
    return backend, model, client, sampler


def _train_kwargs() -> dict[str, Any]:
    return {
        "learning_rate": 1e-5,
        "loss_fn": "cispo",
        "loss_fn_config": None,
        "normalize_advantages": True,
        "save_checkpoint": False,
        "adam_params": None,
        "optimizer_save_interval": 5,
    }


async def _prepare_context(
    backend, model, groups, *, parent: int = 3, train_kwargs=None
):
    return await backend.prepare_pipeline_commands(
        model,
        groups,
        train_kwargs=(train_kwargs if train_kwargs is not None else _train_kwargs()),
        learner_parent_version=parent,
    )


async def _admit_forward(context):
    return await context.forward_backward(context.client.next_sequence_id)


async def _admit_commands(context, forward):
    optimizer = sampler = None
    try:
        optimizer = await context.client.optim_step(
            context.optimizer_request(context.client.next_sequence_id)
        )
        step = optimizer.ref.reserved_output_learner_version
        if step is None:
            raise RuntimeError("optimizer did not reserve a learner version")
        sampler = await context.client.save_weights_for_sampler(
            await context.sampler_request(step, context.client.next_sequence_id)
        )
        state_request = context.state_request(step, context.client.next_sequence_id)
        state = (
            await context.client.save_state(state_request)
            if state_request is not None
            else None
        )
        await context.commands_admitted(
            forward=forward,
            optimizer=optimizer,
            sampler=sampler,
            state=state,
        )
    except BaseException:
        await context.abort(
            forward,
            optimizer,
            sampler,
            optimizer_admitted=optimizer is not None,
        )
        raise
    return step, optimizer


def _complete_commands(context, forward, optimizer, step):
    return asyncio.create_task(
        context.complete(
            step=step,
            forward=forward,
            optimizer=optimizer,
            forward_submit_s=0.0,
        )
    )


def _full_group() -> TrajectoryGroup:
    choice = Choice.model_validate(
        {
            "index": 0,
            "finish_reason": "length",
            "message": {"role": "assistant", "content": "answer"},
            "prompt_token_ids": [10, 11],
            "token_ids": [12, 13],
            "logprobs": {
                "content": [
                    {"token": "a", "logprob": -0.1, "bytes": [97], "top_logprobs": []},
                    {"token": "b", "logprob": -0.2, "bytes": [98], "top_logprobs": []},
                ]
            },
            "policy_token_spans": [
                {"start_token": 0, "end_token": 2, "policy_version": 3}
            ],
        }
    )
    return TrajectoryGroup(
        [
            Trajectory(
                messages_and_choices=[
                    {"role": "user", "content": "question"},
                    choice,
                ],
                reward=1.0,
                initial_policy_version=3,
                final_policy_version=3,
            )
        ]
    )


def _summary(queue: _Queue) -> tuple[TrajectoryGroup, TrajectoryGroupRef]:
    full = queue.materialized
    bundle = TrajectoryGroupBundle.from_group(full)
    ref = TrajectoryGroupRef(
        result_id="result",
        owner_actor_id="owner",
        lease_id="lease",
        records=(),
        descriptor=TrajectoryGroupDescriptor(
            grouping_key="group",
            trajectory_count=1,
            exception_count=0,
            rewards=(1.0,),
            initial_policy_versions=(3,),
            completion_tokens=(2.0,),
            policy_token_counts={3: 2},
            trajectory_initial_policy_versions=(3,),
            trajectory_final_policy_versions=(3,),
            trajectory_policy_token_counts=({3: 2},),
            trajectory_metrics=({},),
            trajectory_metadata=({},),
            group_metadata={},
            group_metrics={},
            exceptions=(),
            byte_count=len(bundle.header) + sum(map(len, bundle.records)),
        ),
    )
    annotations = TrajectoryGroupAnnotations(
        initial_policy_version=3,
        final_policy_version=3,
        rollout_wall_s=1.0,
        actor_idle_s=0.1,
        queue_wait_s=0.2,
    )
    lease = TrajectoryQueueLease(
        claim_id="claim",
        consumer_id="consumer",
        generation=1,
        item=TrajectoryQueueItem(ref=ref, annotations=annotations),
    )
    group = TrajectoryGroup(
        [Trajectory(reward=1.0, initial_policy_version=3, final_policy_version=3)]
    )
    group._distributed_lease = DistributedTrajectorySelection(queue, lease)
    return group, ref


def _distributed_group() -> tuple[_Queue, TrajectoryGroup]:
    queue = _Queue(_full_group())
    group, _ = _summary(queue)
    return queue, group


async def _start_commands(backend, model, *, parent=3, train_kwargs=None):
    queue, group = _distributed_group()
    context = await _prepare_context(
        backend,
        model,
        [group],
        parent=parent,
        train_kwargs=train_kwargs,
    )
    forward = await _admit_forward(context)
    step, optimizer = await _admit_commands(context, forward)
    return SimpleNamespace(
        queue=queue,
        context=context,
        forward=forward,
        optimizer=optimizer,
        step=step,
        completion=_complete_commands(context, forward, optimizer, step),
    )


@pytest.mark.asyncio
async def test_bundle_failure_releases_marked_selection_once() -> None:
    backend, model, _, _ = _backend()
    queue = _Queue(_full_group(), fail_materialize=True)
    summary, _ = _summary(queue)
    with pytest.raises(RuntimeError, match="materialization failed"):
        await _prepare_context(backend, model, [summary])
    selections, generation_id = queue.marked[0]
    assert summary._distributed_lease is None
    assert len(selections) == 1
    assert queue.released == [(selections, "discarded", generation_id)]


@pytest.mark.asyncio
async def test_pipeline_encodes_remote_batch_before_forward_admission() -> None:
    backend, model, client, _ = _backend()
    queue, group = _distributed_group()

    context = await _prepare_context(backend, model, [group])
    assert context.encoded_batch.batch is context.forward_request.batch
    assert client.encoded_batches == []

    await _admit_forward(context)
    assert client.encoded_batches == [context.encoded_batch]
    assert queue.released == []


@pytest.mark.asyncio
async def test_pipeline_encoding_failure_releases_selection(monkeypatch) -> None:
    backend, model, _, _ = _backend()
    queue, group = _distributed_group()

    def fail(_batch, *, object_namespace):
        del object_namespace
        raise RuntimeError("encoding failed")

    monkeypatch.setattr(serverless_backend_module, "prepare_training_batch", fail)
    with pytest.raises(RuntimeError, match="encoding failed"):
        await _prepare_context(backend, model, [group])

    selections, generation_id = queue.marked[0]
    assert queue.released == [(selections, "discarded", generation_id)]


def _forward_result(operation_id: str) -> ForwardBackwardResult:
    return ForwardBackwardResult(
        operation_id=operation_id,
        packing=PackingOutcome(
            packed_sequence_length=16,
            packed_sequences=1,
            target_packed_sequences=1,
            nominal_capacity_tokens=16,
            physical_tokens=4,
            non_padding_tokens=4,
            loss_bearing_tokens=1,
            trainable_assistant_tokens=1,
            policy_token_counts=(
                PolicyTokenCount(policy_version=3, trainable_assistant_tokens=1),
            ),
            group_shapes=(),
        ),
        loss_fn_outputs=(),
        metrics={
            "time/forward_backward_s": 0.2,
            "data/gradient_step_nonpadding_logical_tokens": 4.0,
            "data/gradient_step_loss_bearing_tokens": 1.0,
            "data/gradient_step_executed_token_equivalents": 4.0,
            "data/gradient_step_nominal_schedule_capacity_tokens": 16.0,
            "data/gradient_step_dummy_executed_token_equivalents": 0.0,
            "data/gradient_step_dummy_schedule_capacity_tokens": 0.0,
            "pipeline/gradient_step_real_microbatches": 1.0,
            "pipeline/gradient_step_dummy_microbatches": 0.0,
        },
        produced_gradient=True,
    )


def _zero_gradient_sft_result(operation_id: str) -> ForwardBackwardResult:
    return ForwardBackwardResult(
        operation_id=operation_id,
        packing=PackingOutcome(
            packed_sequence_length=16,
            packed_sequences=0,
            target_packed_sequences=1,
            nominal_capacity_tokens=16,
            physical_tokens=0,
            non_padding_tokens=0,
            loss_bearing_tokens=0,
            trainable_assistant_tokens=0,
            policy_token_counts=None,
            group_shapes=(),
        ),
        loss_fn_outputs=(),
        produced_gradient=False,
    )


async def _wait_for_operations(client: _Client, count: int) -> None:
    async with asyncio.timeout(1):
        while len(client.operations) < count:
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_sft_admits_optimizer_before_forward_result_transport() -> None:
    backend, model, client, _ = _backend()
    stream = backend._train_sft(
        model,
        [Trajectory()],
        TrainSFTConfig(batch_size=1),
        {},
    )
    result = asyncio.create_task(anext(stream))

    await _wait_for_operations(client, 1)
    forward = client.operations[0]
    assert forward.preparation_future is not None
    assert not forward.preparation_future.done()
    assert len(client.operations) == 1

    forward.preparation_future.set_result("contributes")
    await _wait_for_operations(client, 2)
    optimizer = client.operations[1]
    assert forward.ref.kind == "forward_backward"
    assert optimizer.ref.kind == "optim_step"
    assert not forward.result_future.done()

    forward.result_future.set_result(_forward_result(forward.ref.operation_id))
    row = await asyncio.wait_for(result, 1)
    assert row[TRAIN_GRADIENT_STEPS_KEY] == 1
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


@pytest.mark.asyncio
async def test_sft_optimizer_admission_failure_cancels_forward() -> None:
    backend, model, client, _ = _backend(fail_optimizer=True)
    stream = backend._train_sft(
        model,
        [Trajectory()],
        TrainSFTConfig(batch_size=1),
        {},
    )

    result = asyncio.create_task(anext(stream))
    await _wait_for_operations(client, 1)
    forward = client.operations[0]
    assert forward.preparation_future is not None
    forward.preparation_future.set_result("contributes")
    with pytest.raises(RuntimeError, match="optimizer admission failed"):
        await result
    assert len(client.operations) == 1
    assert client.operations[0].cancelled


@pytest.mark.asyncio
async def test_zero_gradient_sft_does_not_admit_optimizer_or_publish() -> None:
    backend, model, client, sampler = _backend()
    stream = backend._train_sft(
        model,
        [Trajectory()],
        TrainSFTConfig(batch_size=1),
        {},
    )
    result = asyncio.create_task(anext(stream))

    await _wait_for_operations(client, 1)
    forward = client.operations[0]
    assert forward.preparation_future is not None
    forward.preparation_future.set_result("empty")
    forward.result_future.set_result(
        _zero_gradient_sft_result(forward.ref.operation_id)
    )
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(result, 1)
    assert len(client.operations) == 1
    assert sampler.publications == []


def _sampler_result(step: int) -> SamplerWeightsResult:
    return SamplerWeightsResult(
        operation_id=f"save-{step}",
        checkpoint=CheckpointRef(
            run_id="run",
            learner_version=step,
            checkpoint_id=f"step-{step}",
        ),
        lora="lora",
        training_session_id="session",
        generation_id=f"generation-{step}",
        lora_bytes=1,
    )


async def _materialize_sampler_result(
    backend: ServerlessBackend, model: TrainableModel, step: int
) -> None:
    pending = await backend._reserve_sampler_publication(model, step)
    await backend._resolve_sampler_result(model, step, pending, _sampler_result(step))
    await backend._finish_sampler_publication(model, step, pending)


def _checkpoint(step: int) -> SimpleNamespace:
    return SimpleNamespace(
        learner_version=step,
        checkpoint_id=f"step-{step}",
        revision=1,
        state="ready",
    )


@pytest.mark.asyncio
async def test_remote_prepare_materializes_distributed_group_and_discards_once() -> (
    None
):
    backend, model, client, _ = _backend()
    queue = _Queue(_full_group())
    summary, _ = _summary(queue)
    summary._collect_packing_shape = True

    context = await _prepare_context(backend, model, [summary])
    forward = await _admit_forward(context)
    encoded = prepare_training_batch(
        client.batch, object_namespace="test-remote-prepare"
    )
    group_ref = encoded.remote.groups[0]
    payload = b"".join(encoded.objects[0].wire_chunks())
    decoded = decode_trajectory_group(group_ref, payload, route_payloads={})
    replay_payload = b"".join(
        encode_trajectory_group(decoded.bundle, object_id="0" * 64).data.wire_chunks()
    )
    rebuilt = group_ref.annotations.apply(decoded.bundle.payload().build())

    assert context.preparation_metrics["time/step_prepare_remote_batch_s"] >= 0
    assert len(rebuilt.trajectories[0].messages_and_choices) == 2
    assert rebuilt.trajectories[0].messages_and_choices[-1].logprobs is not None
    assert client.collect_packing_shapes
    assert replay_payload == payload
    assert summary._distributed_lease is None
    assert queue.marked == [(context.selections, context.generation_id)]

    await context.abort(forward, None, None, optimizer_admitted=False)
    assert forward.cancelled
    assert queue.released == [(context.selections, "discarded", context.generation_id)]


@pytest.mark.asyncio
async def test_remote_optimizer_admission_failure_discards_prepared_batch() -> None:
    backend, model, _, _ = _backend(fail_optimizer=True)
    queue = _Queue(_full_group())
    summary, _ = _summary(queue)

    context = await _prepare_context(backend, model, [summary])
    forward = await _admit_forward(context)
    with pytest.raises(RuntimeError, match="optimizer admission failed"):
        await _admit_commands(context, forward)

    assert forward.cancelled
    assert queue.released == [(context.selections, "discarded", context.generation_id)]


@pytest.mark.asyncio
async def test_admitted_commands_consume_marked_selection_once() -> None:
    backend, model, _, _ = _backend()
    queue = _Queue(_full_group())
    summary, _ = _summary(queue)

    context = await _prepare_context(backend, model, [summary])
    forward = await _admit_forward(context)
    await _admit_commands(context, forward)
    await backend._drain_background()

    assert queue.released == [(context.selections, "consumed", context.generation_id)]


@pytest.mark.asyncio
async def test_next_forward_is_admitted_before_prior_forward_completes() -> None:
    backend, model, client, _ = _backend()
    _, first = _distributed_group()
    first_context = await _prepare_context(backend, model, [first])
    first_forward = await _admit_forward(first_context)

    first_step, first_optimizer = await _admit_commands(first_context, first_forward)
    completion = _complete_commands(
        first_context, first_forward, first_optimizer, first_step
    )
    assert first_step == 4
    assert [operation.ref.kind for operation in client.operations] == [
        "forward_backward",
        "optim_step",
        "save_sampler",
    ]
    assert not completion.done()

    _, second = _distributed_group()
    second_context = await _prepare_context(backend, model, [second], parent=4)
    second_forward = await _admit_forward(second_context)
    assert [operation.ref.sequence_id for operation in client.operations] == [
        0,
        1,
        2,
        3,
    ]
    assert client.operations[-1].ref.kind == "forward_backward"
    assert client.operations[-1].ref.learner_parent_version == 4

    first_forward.result_future.set_result(
        _forward_result(first_forward.ref.operation_id)
    )
    result = await completion
    assert result.step == 4
    assert result.packed_policy_token_counts == ((3, 1),)
    await second_context.abort(second_forward, None, None, optimizer_admitted=False)
    await backend._drain_background()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parent", "final_step", "save_checkpoint", "expected"),
    (
        (0, None, False, True),
        (4, None, False, True),
        (3, 4, False, True),
        (3, None, False, False),
        (3, None, True, False),
    ),
)
async def test_remote_optimizer_state_save_policy(
    parent: int, final_step: int | None, save_checkpoint: bool, expected: bool
) -> None:
    backend, model, client, _ = _backend()
    client.projected_learner_version = parent
    queue = _Queue(_full_group())
    group, _ = _summary(queue)

    kwargs = _train_kwargs()
    kwargs["final_training_step"] = final_step
    kwargs["save_checkpoint"] = save_checkpoint
    context = await _prepare_context(
        backend, model, [group], parent=parent, train_kwargs=kwargs
    )
    forward = await _admit_forward(context)
    step, optimizer = await _admit_commands(context, forward)

    assert (client.operations[-1].ref.kind == "save_state") is expected
    forward.result_future.set_result(_forward_result(forward.ref.operation_id))
    await _complete_commands(context, forward, optimizer, step)
    await backend._drain_background()


@pytest.mark.asyncio
async def test_group_discard_skips_backend_after_selection_ownership_transfer() -> None:
    trainer = object.__new__(PipelineTrainer)
    trainer._output_queue = DistributedTrajectoryQueue(
        endpoint=object(),
        owner_endpoints={},
        maxsize=1,
        capacity_records=1,
        capacity_bytes=1,
    )
    trainer.backend = _DiscardBackend()
    trainer.model = object()
    group = TrajectoryGroup()

    await trainer._discard_collected_group(group)

    assert not trainer.backend.called


@pytest.mark.asyncio
async def test_committed_train_does_not_wait_for_batch_release() -> None:
    backend, model, _, sampler = _backend()
    queue = _Queue(_full_group())
    queue.release_gate = asyncio.Event()
    group, _ = _summary(queue)

    context = await _prepare_context(backend, model, [group])
    forward = await _admit_forward(context)
    step, optimizer = await _admit_commands(context, forward)
    completion = _complete_commands(context, forward, optimizer, step)
    forward.result_future.set_result(_forward_result(forward.ref.operation_id))

    await asyncio.wait_for(sampler.published.wait(), timeout=1)
    assert (await asyncio.wait_for(completion, timeout=1)).step == 4
    assert queue.released == []

    queue.release_gate.set()
    await backend._drain_background()
    assert len(queue.released) == 1


@pytest.mark.asyncio
async def test_train_and_next_optimizer_do_not_wait_for_sampler_publication() -> None:
    backend, model, client, sampler = _backend()
    client.gate_sampler = True
    sampler.active_gate = asyncio.Event()

    _, first_group = _distributed_group()
    first_context = await _prepare_context(backend, model, [first_group])
    first_forward = await _admit_forward(first_context)
    first_step, first_optimizer = await _admit_commands(first_context, first_forward)
    first_completion = _complete_commands(
        first_context, first_forward, first_optimizer, first_step
    )
    first_forward.result_future.set_result(
        _forward_result(first_forward.ref.operation_id)
    )
    first_result = await asyncio.wait_for(first_completion, timeout=1)

    assert first_result.step == 4
    assert first_result.checkpoint_id is None
    first_ready = first_result.checkpoint_ready
    assert isinstance(first_ready, asyncio.Future)
    assert not first_ready.done()
    assert "publication/materialization" not in first_result.metrics

    _, second_group = _distributed_group()
    second_context = await _prepare_context(
        backend, model, [second_group], parent=first_step
    )
    second_forward = await _admit_forward(second_context)
    second_step, second_optimizer = await _admit_commands(
        second_context, second_forward
    )
    assert [operation.ref.kind for operation in client.operations].count(
        "optim_step"
    ) == 2
    second_completion = _complete_commands(
        second_context, second_forward, second_optimizer, second_step
    )
    second_forward.result_future.set_result(
        _forward_result(second_forward.ref.operation_id)
    )
    second_result = await asyncio.wait_for(second_completion, timeout=1)

    client.complete_sampler(0)
    await asyncio.wait_for(sampler.active_started.wait(), timeout=1)
    while first_result.checkpoint_id is None:
        await asyncio.sleep(0)
    assert not first_ready.done()
    with pytest.raises(RuntimeError, match="pending or active exact adapter"):
        await backend._validate_sampler_retention(model, set())

    client.complete_sampler(1)
    sampler.active_gate.set()
    second_ready = second_result.checkpoint_ready
    assert isinstance(second_ready, asyncio.Future)
    await asyncio.gather(first_ready, second_ready)
    assert first_result.publication_metrics_ready is not None
    assert second_result.publication_metrics_ready is not None
    first_metrics, second_metrics = await asyncio.gather(
        first_result.publication_metrics_ready,
        second_result.publication_metrics_ready,
    )
    assert first_metrics == {
        "publication/materialization": 1.0,
        "publication/sampler": 1.0,
        "publication/sampler_step": 4.0,
    }
    assert second_metrics == {
        "publication/materialization": 1.0,
        "publication/sampler": 1.0,
        "publication/sampler_step": 5.0,
    }
    await backend._drain_background()


@pytest.mark.asyncio
async def test_sampler_save_materializes_before_external_activation() -> None:
    backend, model, client, sampler = _backend()
    step = client.projected_learner_version

    operation, pending = await backend._start_sampler_publication(
        model, client, step, client.next_sequence_id
    )

    assert client.sampler_operations[-1][1].publication.mode == "none"
    await backend._complete_sampler_publication(model, step, operation, pending)
    assert sampler.publications == ["in_flight_lora"]


@pytest.mark.asyncio
async def test_public_train_waits_for_sampler_checkpoint_and_active_alias() -> None:
    backend, model, client, sampler = _backend()
    client.gate_sampler = True
    sampler.active_gate = asyncio.Event()

    training = asyncio.create_task(
        backend.train(model, [_full_group()], **_train_kwargs())
    )
    await _wait_for_operations(client, 1)
    forward = client.operations[0]
    assert forward.preparation_future is not None
    forward.preparation_future.set_result("contributes")
    await _wait_for_operations(client, 3)
    forward.result_future.set_result(_forward_result(forward.ref.operation_id))
    await asyncio.sleep(0)
    assert not training.done()

    client.complete_sampler()
    await asyncio.wait_for(sampler.active_started.wait(), timeout=1)
    assert not training.done()
    sampler.active_gate.set()

    result = await asyncio.wait_for(training, timeout=1)
    assert result.checkpoint_id == "step-4"
    assert result.checkpoint_ready is None
    assert result.publication_metrics_ready is None
    assert result.metrics["publication/materialization"] == 1.0
    assert result.metrics["publication/sampler_step"] == 4.0
    await backend._drain_background()


@pytest.mark.asyncio
async def test_active_sampler_publications_follow_learner_order() -> None:
    backend, model, client, sampler = _backend()
    client.gate_sampler = True
    sampler.active_gate = asyncio.Event()

    _, first_group = _distributed_group()
    first_context = await _prepare_context(backend, model, [first_group])
    first_forward = await _admit_forward(first_context)
    first_step, first_optimizer = await _admit_commands(first_context, first_forward)
    first_forward.result_future.set_result(
        _forward_result(first_forward.ref.operation_id)
    )
    first_result = await _complete_commands(
        first_context, first_forward, first_optimizer, first_step
    )
    _, second_group = _distributed_group()
    second_context = await _prepare_context(
        backend, model, [second_group], parent=first_step
    )
    second_forward = await _admit_forward(second_context)
    second_step, second_optimizer = await _admit_commands(
        second_context, second_forward
    )
    second_forward.result_future.set_result(
        _forward_result(second_forward.ref.operation_id)
    )
    second_result = await _complete_commands(
        second_context, second_forward, second_optimizer, second_step
    )

    second_pending = backend._pending_sampler_publications[
        backend._sampler_key(model, second_result.step)
    ]
    client.complete_sampler(1)
    assert (await second_pending.materialized).checkpoint.learner_version == 5
    await asyncio.sleep(0)
    assert sampler.active_versions == []

    client.complete_sampler(0)
    await asyncio.wait_for(sampler.active_started.wait(), timeout=1)
    assert sampler.active_versions == [4]
    sampler.active_gate.set()
    assert first_result.checkpoint_ready is not None
    assert second_result.checkpoint_ready is not None
    await asyncio.gather(
        first_result.checkpoint_ready,
        second_result.checkpoint_ready,
    )
    assert sampler.active_versions == [4, 5]
    assert first_result.publication_metrics_ready is not None
    assert second_result.publication_metrics_ready is not None
    first_metrics, second_metrics = await asyncio.gather(
        first_result.publication_metrics_ready,
        second_result.publication_metrics_ready,
    )
    assert first_metrics["publication/sampler_step"] == 4.0
    assert second_metrics["publication/sampler_step"] == 5.0
    await backend._drain_background()


@pytest.mark.asyncio
async def test_retention_apply_does_not_block_or_forget_new_publication() -> None:
    backend, model, _, _ = _backend()
    service = _RetentionService(_checkpoint(3))
    backend._service = service
    await _materialize_sampler_result(backend, model, 3)

    retention = asyncio.create_task(
        backend._apply_checkpoint_retention(
            model, CheckpointRetentionPlan(observed_steps={3})
        )
    )
    await asyncio.wait_for(service.apply_started.wait(), timeout=1)

    run = await asyncio.wait_for(_start_commands(backend, model), timeout=1)
    run.forward.result_future.set_result(_forward_result(run.forward.ref.operation_id))
    result = await asyncio.wait_for(run.completion, timeout=1)
    assert result.checkpoint_ready is not None
    await asyncio.wait_for(result.checkpoint_ready, timeout=1)

    service.apply_gate.set()
    await asyncio.wait_for(retention, timeout=1)
    assert backend._sampler_key(model, 3) not in backend._sampler_results
    assert backend._sampler_key(model, 4) in backend._sampler_results
    await backend._drain_background()


@pytest.mark.asyncio
async def test_retention_ignores_checkpoint_committed_after_planning() -> None:
    backend, model, _, _ = _backend()
    service = _RetentionService(
        _checkpoint(3),
        _checkpoint(4),
        protected=("step-4",),
    )
    service.apply_gate.set()
    backend._service = service

    await backend._apply_checkpoint_retention(
        model,
        CheckpointRetentionPlan(observed_steps={3}),
    )

    assert tuple(item.checkpoint_id for item in service.retention_request.observed) == (
        "step-3",
    )
    assert service.retention_request.retain_checkpoint_ids == ()
    assert service.retention_request.archive_checkpoint_ids == ()


@pytest.mark.asyncio
async def test_retention_rejects_checkpoint_removed_after_planning() -> None:
    backend, model, _, _ = _backend()
    service = _RetentionService(_checkpoint(4))
    backend._service = service

    with pytest.raises(
        RuntimeError, match="remote checkpoint catalog changed during retention"
    ):
        await backend._apply_checkpoint_retention(
            model,
            CheckpointRetentionPlan(observed_steps={3}),
        )


@pytest.mark.asyncio
async def test_retention_keeps_server_protected_recovery_generation() -> None:
    backend, model, _, _ = _backend()
    service = _RetentionService(
        _checkpoint(1),
        _checkpoint(6),
        protected=("step-1", "step-6"),
    )
    service.apply_gate.set()
    backend._service = service

    await backend._apply_checkpoint_retention(
        model,
        CheckpointRetentionPlan(
            observed_steps={1, 6},
            retain_steps={6},
        ),
    )

    assert service.retention_request.retain_checkpoint_ids == ("step-1", "step-6")


@pytest.mark.asyncio
async def test_pending_materialization_is_atomic_with_retention_reservation() -> None:
    backend, model, client, _ = _backend()
    client.gate_sampler = True
    service = _RetentionService(_checkpoint(4))
    backend._service = service
    run = await _start_commands(backend, model)
    run.forward.result_future.set_result(_forward_result(run.forward.ref.operation_id))
    result = await asyncio.wait_for(run.completion, timeout=1)

    reserve_started = asyncio.Event()
    reserve_gate = asyncio.Event()
    reserve = backend._reserve_sampler_forgetting

    async def gated_reserve(*args, **kwargs):
        reserve_started.set()
        await reserve_gate.wait()
        return await reserve(*args, **kwargs)

    backend._reserve_sampler_forgetting = gated_reserve
    retention = asyncio.create_task(
        backend._apply_checkpoint_retention(
            model, CheckpointRetentionPlan(observed_steps={4})
        )
    )
    await asyncio.wait_for(reserve_started.wait(), timeout=1)
    async with backend._sampler_state_lock:
        reserve_gate.set()
        await asyncio.sleep(0)
        client.complete_sampler()
        await asyncio.sleep(0)

    await asyncio.wait_for(service.apply_started.wait(), timeout=1)
    assert service.retention_request.retain_checkpoint_ids == ("step-4",)
    assert result.checkpoint_ready is not None
    await asyncio.wait_for(result.checkpoint_ready, timeout=1)
    service.apply_gate.set()
    await asyncio.wait_for(retention, timeout=1)
    assert backend._sampler_key(model, 4) in backend._sampler_results
    await backend._drain_background()


@pytest.mark.asyncio
async def test_exact_eval_waits_for_its_pending_sampler_then_publishes_version() -> (
    None
):
    backend, model, client, sampler = _backend()
    client.gate_sampler = True
    sampler.active_gate = asyncio.Event()
    run = await _start_commands(backend, model)
    run.forward.result_future.set_result(_forward_result(run.forward.ref.operation_id))
    result = await run.completion
    entered = asyncio.Event()
    leave = asyncio.Event()

    async def evaluate() -> None:
        async with backend.exact_adapter_lease(model, result.step):
            assert model.get_inference_name() == f"model@{result.step}"
            entered.set()
            await leave.wait()

    evaluation = asyncio.create_task(evaluate())
    await asyncio.sleep(0)
    assert not entered.is_set()
    with pytest.raises(RuntimeError, match="pending or active exact adapter"):
        await backend._forget_sampler_results(model, set())

    client.complete_sampler()
    await asyncio.wait_for(sampler.active_started.wait(), timeout=1)
    await asyncio.wait_for(entered.wait(), timeout=1)
    assert sampler.publications[:2] == ["in_flight_lora", "versioned_lora"]
    ready = result.checkpoint_ready
    assert isinstance(ready, asyncio.Future)
    assert not ready.done()

    leave.set()
    await evaluation
    sampler.active_gate.set()
    await ready
    assert result.checkpoint_id == "step-4"
    await backend._drain_background()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["next_train", "session_finalization"])
async def test_publication_failure_does_not_relabel_committed_train(
    boundary: str,
) -> None:
    backend, model, _, sampler = _backend()
    sampler.fail_active = True
    run = await _start_commands(backend, model)
    run.forward.result_future.set_result(_forward_result(run.forward.ref.operation_id))

    result = await asyncio.wait_for(run.completion, timeout=1)
    assert result.step == 4
    assert result.packed_policy_token_counts == ((3, 1),)
    assert result.checkpoint_ready is not None
    with pytest.raises(RuntimeError, match="active publication failed"):
        await result.checkpoint_ready
    assert result.checkpoint_id == "step-4"
    async with backend.exact_adapter_lease(model, result.step):
        assert model.get_inference_name() == "model@4"

    with pytest.raises(BaseExceptionGroup, match="remote background operations failed"):
        if boundary == "next_train":
            await _start_commands(backend, model, parent=result.step)
        else:
            await backend.finalize_training_session(model)


@pytest.mark.asyncio
async def test_close_drains_sampler_materialization_and_activation() -> None:
    backend, model, client, sampler = _backend()
    backend._close_timeout_s = 1.0
    client.gate_sampler = True
    sampler.active_gate = asyncio.Event()
    run = await _start_commands(backend, model)
    run.forward.result_future.set_result(_forward_result(run.forward.ref.operation_id))
    result = await run.completion

    close = asyncio.create_task(backend.close())
    await asyncio.sleep(0)
    assert not close.done()
    client.complete_sampler()
    await asyncio.wait_for(sampler.active_started.wait(), timeout=1)
    assert not close.done()
    sampler.active_gate.set()
    await asyncio.wait_for(close, timeout=1)

    assert client.shutdown_calls == 1
    assert client.service is not None
    assert client.service.close_calls == 1
    assert result.checkpoint_ready is not None
    await result.checkpoint_ready
    assert not backend._pending_sampler_publications
    assert not backend._sampler_results
    assert not backend._sampler_publication_tails


@pytest.mark.asyncio
async def test_close_bounds_stuck_sampler_publication() -> None:
    backend, model, client, _ = _backend()
    backend._close_timeout_s = 0.02
    client.gate_sampler = True
    run = await _start_commands(backend, model)
    run.forward.result_future.set_result(_forward_result(run.forward.ref.operation_id))
    result = await run.completion

    started = time.monotonic()
    with pytest.raises(BaseExceptionGroup, match="ServerlessBackend shutdown failed"):
        await backend.close()
    assert time.monotonic() - started < 0.2
    await asyncio.sleep(0)
    assert not backend._background
    assert not backend._pending_sampler_publications
    assert not backend._sampler_results
    assert not backend._sampler_publication_tails
    ready = result.checkpoint_ready
    assert isinstance(ready, asyncio.Future)
    assert ready.done()


@pytest.mark.asyncio
async def test_pipeline_logs_remote_checkpoint_only_after_readiness() -> None:
    trainer = object.__new__(PipelineTrainer)
    recorded: list[tuple[int, Path | None, bool]] = []
    trainer._record_checkpoint_saved = lambda step, path, remote=False: recorded.append(
        (step, path, remote)
    )
    result = ServerlessTrainResult(step=4)
    ready = asyncio.get_running_loop().create_future()
    logging = asyncio.create_task(
        trainer._log_checkpoint_when_ready(result, Path("/missing"), ready)
    )

    await asyncio.sleep(0)
    assert recorded == []
    result.checkpoint_id = "checkpoint-4"
    ready.set_result(None)
    await logging

    assert recorded == [(4, None, True)]


@pytest.mark.asyncio
async def test_pipeline_emits_publication_metrics_on_their_own_step() -> None:
    trainer = object.__new__(PipelineTrainer)
    trainer._publication_metric_tasks = set()
    trainer._publication_metric_failure = None
    emitted: list[tuple[int, dict[str, float]]] = []
    logged: list[tuple[int, dict[str, float]]] = []

    async def emit(metrics, *, step, **_kwargs) -> None:
        emitted.append((step, dict(metrics)))

    async def log(*_args, step, metrics, **_kwargs) -> None:
        logged.append((step, dict(metrics)))

    trainer._emit_pipeline_metrics = emit
    trainer.model = SimpleNamespace(log=log)
    trainer.request_stop = lambda: None
    ready = asyncio.get_running_loop().create_future()
    result = ServerlessTrainResult(step=7, publication_metrics_ready=ready)

    trainer._schedule_publication_metrics(result)
    tasks = tuple(trainer._publication_metric_tasks)
    assert len(tasks) == 1 and not tasks[0].done()
    ready.set_result({"publication/activation_s": 0.25})
    await asyncio.gather(*tasks)

    expected = (7, {"publication/activation_s": 0.25})
    assert emitted == [expected]
    assert logged == [expected]
    assert trainer._publication_metric_failure is None
