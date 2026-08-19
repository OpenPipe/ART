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
from art.pipeline_trainer.trainer import PipelineTrainer
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
from art.types import ServerlessTrainResult


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


class _Operation:
    def __init__(self, ref: OperationRef) -> None:
        self.ref = ref
        self.result_future = asyncio.get_running_loop().create_future()
        self.cancelled = False

    async def result(self):
        return await asyncio.shield(self.result_future)

    async def cancel(self) -> None:
        self.cancelled = True


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

    async def forward_backward(self, request):
        self.batch = request.batch
        self.collect_packing_shapes = request.collect_packing_shapes
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


async def _prepare(backend, model, groups, *, parent: int = 3):
    return await backend.prepare_pipeline_batch(
        model,
        groups,
        train_kwargs=_train_kwargs(),
        learner_parent_version=parent,
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


async def _stage_and_prepare(backend, model, queue, summary, ref, *, parent=3):
    del queue, ref
    return await _prepare(backend, model, [summary], parent=parent)


@pytest.mark.asyncio
async def test_bundle_failure_keeps_selection_owned_by_group() -> None:
    backend, model, _, _ = _backend()
    queue = _Queue(_full_group(), fail_materialize=True)
    summary, _ = _summary(queue)
    with pytest.raises(RuntimeError, match="materialization failed"):
        await _prepare(backend, model, [summary])
    assert summary._distributed_lease is not None
    assert queue.marked == []
    assert queue.released == []


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
    )


@pytest.mark.asyncio
async def test_remote_prepare_materializes_distributed_group_and_discards_once() -> (
    None
):
    backend, model, client, _ = _backend()
    queue = _Queue(_full_group())
    summary, ref = _summary(queue)
    summary._collect_packing_shape = True

    metrics = await _stage_and_prepare(backend, model, queue, summary, ref)
    prepared = summary._prepared_training_batch
    encoded = prepare_training_batch(client.batch)
    group_ref = encoded.remote.groups[0]
    payload = encoded.objects[0].payload
    decoded = decode_trajectory_group(group_ref, payload, route_payloads={})
    replay_payload = encode_trajectory_group(decoded.bundle).data.payload
    rebuilt = group_ref.annotations.apply(decoded.bundle.payload().build())

    assert metrics["time/step_prepare_remote_batch_s"] >= 0
    assert len(rebuilt.trajectories[0].messages_and_choices) == 2
    assert rebuilt.trajectories[0].messages_and_choices[-1].logprobs is not None
    assert client.collect_packing_shapes
    assert replay_payload == payload
    assert summary._distributed_lease is None
    assert queue.marked == [((prepared.selections[0],), prepared.generation_id)]

    await backend.discard_pipeline_batch([summary])
    assert summary._prepared_training_batch is None
    assert prepared.forward.cancelled
    assert queue.released == [
        ((prepared.selections[0],), "discarded", prepared.generation_id)
    ]


@pytest.mark.asyncio
async def test_remote_optimizer_admission_failure_discards_prepared_batch() -> None:
    backend, model, client, _ = _backend(fail_optimizer=True)
    queue = _Queue(_full_group())
    summary, ref = _summary(queue)

    await _stage_and_prepare(backend, model, queue, summary, ref)
    prepared = summary._prepared_training_batch
    with pytest.raises(RuntimeError, match="optimizer admission failed"):
        await backend.start_pipeline_train(model, [summary], **_train_kwargs())

    assert summary._prepared_training_batch is None
    assert prepared.forward.cancelled
    assert queue.released == [
        ((prepared.selections[0],), "discarded", prepared.generation_id)
    ]


@pytest.mark.asyncio
async def test_remote_prepared_upload_consumes_marked_selection_once() -> None:
    backend, model, _, _ = _backend()
    queue = _Queue(_full_group())
    summary, ref = _summary(queue)

    await _stage_and_prepare(backend, model, queue, summary, ref)
    prepared = summary._prepared_training_batch
    await backend._release_remote_pipeline_batch(prepared, disposition="consumed")

    assert queue.released == [
        ((prepared.selections[0],), "consumed", prepared.generation_id)
    ]


@pytest.mark.asyncio
async def test_next_forward_is_admitted_before_prior_forward_completes() -> None:
    backend, model, client, _ = _backend()
    first_queue = _Queue(_full_group())
    first, first_ref = _summary(first_queue)
    await _stage_and_prepare(backend, model, first_queue, first, first_ref)

    pending = await backend.start_pipeline_train(model, [first], **_train_kwargs())
    assert pending.step == 4
    assert [operation.ref.kind for operation in client.operations] == [
        "forward_backward",
        "optim_step",
        "save_sampler",
    ]
    assert not pending.completion.done()

    second_queue = _Queue(_full_group())
    second, second_ref = _summary(second_queue)
    await _stage_and_prepare(backend, model, second_queue, second, second_ref, parent=4)
    assert [operation.ref.sequence_id for operation in client.operations] == [
        0,
        1,
        2,
        3,
    ]
    assert client.operations[-1].ref.kind == "forward_backward"
    assert client.operations[-1].ref.learner_parent_version == 4

    client.operations[0].result_future.set_result(
        _forward_result(client.operations[0].ref.operation_id)
    )
    result = await pending.result()
    assert result.step == 4
    assert result.packed_policy_token_counts == ((3, 1),)
    await backend.discard_pipeline_batch([second])


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
    group, ref = _summary(queue)

    await _stage_and_prepare(backend, model, queue, group, ref, parent=parent)
    kwargs = _train_kwargs()
    kwargs["final_training_step"] = final_step
    kwargs["save_checkpoint"] = save_checkpoint
    pending = await backend.start_pipeline_train(model, [group], **kwargs)

    assert (client.operations[-1].ref.kind == "save_state") is expected
    client.operations[0].result_future.set_result(
        _forward_result(client.operations[0].ref.operation_id)
    )
    await pending.result()
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
    backend, model, client, sampler = _backend()
    queue = _Queue(_full_group())
    queue.release_gate = asyncio.Event()
    group, ref = _summary(queue)

    await _stage_and_prepare(backend, model, queue, group, ref)
    pending = await backend.start_pipeline_train(model, [group], **_train_kwargs())
    client.operations[0].result_future.set_result(
        _forward_result(client.operations[0].ref.operation_id)
    )

    await asyncio.wait_for(sampler.published.wait(), timeout=1)
    assert (await asyncio.wait_for(pending.result(), timeout=1)).step == 4
    assert queue.released == []

    queue.release_gate.set()
    await backend._drain_background()
    assert len(queue.released) == 1


@pytest.mark.asyncio
async def test_train_and_next_optimizer_do_not_wait_for_sampler_publication() -> None:
    backend, model, client, sampler = _backend()
    client.gate_sampler = True
    sampler.active_gate = asyncio.Event()

    first = await backend.start_pipeline_train(
        model, [_full_group()], **_train_kwargs()
    )
    client.operations[0].result_future.set_result(
        _forward_result(client.operations[0].ref.operation_id)
    )
    first_result = await asyncio.wait_for(first.result(), timeout=1)

    assert first_result.step == 4
    assert first_result.checkpoint_id is None
    first_ready = first_result.checkpoint_ready
    assert isinstance(first_ready, asyncio.Future)
    assert not first_ready.done()
    assert "publication/materialization" not in first_result.metrics

    second = await backend.start_pipeline_train(
        model, [_full_group()], **_train_kwargs()
    )
    assert [operation.ref.kind for operation in client.operations].count(
        "optim_step"
    ) == 2
    client.operations[3].result_future.set_result(
        _forward_result(client.operations[3].ref.operation_id)
    )
    second_result = await asyncio.wait_for(second.result(), timeout=1)

    client.complete_sampler(0)
    await asyncio.wait_for(sampler.active_started.wait(), timeout=1)
    while first_result.checkpoint_id is None:
        await asyncio.sleep(0)
    assert not first_ready.done()
    with pytest.raises(RuntimeError, match="pending or active exact adapter"):
        backend._validate_sampler_retention(model, set())

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
async def test_active_sampler_publications_follow_learner_order() -> None:
    backend, model, client, sampler = _backend()
    client.gate_sampler = True
    sampler.active_gate = asyncio.Event()

    first = await backend.start_pipeline_train(
        model, [_full_group()], **_train_kwargs()
    )
    client.operations[0].result_future.set_result(
        _forward_result(client.operations[0].ref.operation_id)
    )
    first_result = await first.result()
    second = await backend.start_pipeline_train(
        model, [_full_group()], **_train_kwargs()
    )
    client.operations[3].result_future.set_result(
        _forward_result(client.operations[3].ref.operation_id)
    )
    second_result = await second.result()

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
async def test_sampler_reservation_serializes_with_retention() -> None:
    backend, model, _, _ = _backend()

    async with backend._exact_adapter_lock:
        reservation = asyncio.create_task(
            backend._reserve_sampler_publication(model, 4)
        )
        await asyncio.sleep(0)
        assert not reservation.done()

    pending = await reservation
    assert backend._pending_sampler_steps(model) == {4}
    backend._fail_sampler_result(model, 4, pending, asyncio.CancelledError())
    backend._clear_sampler_state()


@pytest.mark.asyncio
async def test_exact_eval_waits_for_its_pending_sampler_then_publishes_version() -> (
    None
):
    backend, model, client, sampler = _backend()
    client.gate_sampler = True
    sampler.active_gate = asyncio.Event()
    pending = await backend.start_pipeline_train(
        model, [_full_group()], **_train_kwargs()
    )
    client.operations[0].result_future.set_result(
        _forward_result(client.operations[0].ref.operation_id)
    )
    result = await pending.result()
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
        backend._forget_sampler_results(model, set())

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
    backend, model, client, sampler = _backend()
    sampler.fail_active = True
    pending = await backend.start_pipeline_train(
        model, [_full_group()], **_train_kwargs()
    )
    client.operations[0].result_future.set_result(
        _forward_result(client.operations[0].ref.operation_id)
    )

    result = await asyncio.wait_for(pending.result(), timeout=1)
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
            await backend.start_pipeline_train(
                model, [_full_group()], **_train_kwargs()
            )
        else:
            await backend.finalize_training_session(model)


@pytest.mark.asyncio
async def test_close_drains_sampler_materialization_and_activation() -> None:
    backend, model, client, sampler = _backend()
    backend._close_timeout_s = 1.0
    client.gate_sampler = True
    sampler.active_gate = asyncio.Event()
    pending = await backend.start_pipeline_train(
        model, [_full_group()], **_train_kwargs()
    )
    client.operations[0].result_future.set_result(
        _forward_result(client.operations[0].ref.operation_id)
    )
    result = await pending.result()

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
    pending = await backend.start_pipeline_train(
        model, [_full_group()], **_train_kwargs()
    )
    client.operations[0].result_future.set_result(
        _forward_result(client.operations[0].ref.operation_id)
    )
    result = await pending.result()

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
