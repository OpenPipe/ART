from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from art.distributed.trajectory_store import TrajectoryGroupBundle
from art.megatron.runtime.monarch import MonarchTrainerRun
import art.megatron.training.client as client_module
from art.megatron.training.client import (
    LocalMegatronTrainingClient,
    _run_command_while_releasing,
)
from art.training.contracts import (
    AdamConfig,
    ForwardBackwardRequest,
    ForwardBackwardResult,
    LossConfig,
    LossFnOutput,
    OperationRef,
    OptimStepRequest,
    PackingOutcome,
    RlTrajectoryBatch,
    SamplerPublication,
    SaveWeightsForSamplerRequest,
    TokenLogprobs,
)


class _Service:
    def __init__(self) -> None:
        self.fail_retirement = False
        self.retired: list[str] = []

    async def consume_cancelled_command(self, _ref: OperationRef) -> None:
        return None

    async def optimizer_command(self, _ref, _optimizer, contributions):
        return {"metrics": {}}, SimpleNamespace(contributions=contributions)

    def retire_command_operation(self, operation_id: str) -> None:
        if self.fail_retirement:
            raise RuntimeError("retirement failed")
        self.retired.append(operation_id)


def _client(service: Any) -> LocalMegatronTrainingClient:
    return LocalMegatronTrainingClient(
        run_id="run",
        learner_version=0,
        backend=SimpleNamespace(),
        model=SimpleNamespace(),
        service=service,
    )


def _save(request_id: str, sequence_id: int) -> SaveWeightsForSamplerRequest:
    return SaveWeightsForSamplerRequest(
        run_id="run",
        request_id=request_id,
        sequence_id=sequence_id,
        checkpoint_name=request_id,
        publication=SamplerPublication(mode="none"),
    )


def _forward_backward(request_id: str, sequence_id: int) -> ForwardBackwardRequest:
    return ForwardBackwardRequest(
        run_id="run",
        request_id=request_id,
        sequence_id=sequence_id,
        batch=RlTrajectoryBatch(
            groups=(TrajectoryGroupBundle(header=b"header", records=()),),
            min_source_version=0,
            max_source_version=0,
        ),
        loss=LossConfig(name="cispo"),
    )


def _forward_backward_result(operation_id: str) -> ForwardBackwardResult:
    return ForwardBackwardResult(
        operation_id=operation_id,
        packing=PackingOutcome(
            packed_sequence_length=3,
            packed_sequences=1,
            target_packed_sequences=1,
            nominal_capacity_tokens=3,
            physical_tokens=3,
            non_padding_tokens=3,
            loss_bearing_tokens=3,
            trainable_assistant_tokens=3,
            policy_token_counts=None,
            group_shapes=(),
        ),
        loss_fn_outputs=(
            LossFnOutput(token_logprobs=TokenLogprobs(shape=(3,), data=b"x" * 12)),
        ),
    )


@pytest.mark.asyncio
async def test_source_release_overlaps_command_and_both_settle() -> None:
    release_started = asyncio.Event()
    finish_release = asyncio.Event()
    command_started = asyncio.Event()

    async def release() -> float:
        release_started.set()
        await finish_release.wait()
        return 0.25

    async def command() -> str:
        await release_started.wait()
        command_started.set()
        return "trained"

    execution = asyncio.create_task(
        _run_command_while_releasing(
            command(), release(), release_name="test-source-release"
        )
    )
    await command_started.wait()
    assert not execution.done()
    finish_release.set()
    assert await execution == ("trained", 0.25)


def test_completed_operation_heap_compaction_is_geometric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(_Service())
    total = 1024
    client._completed_operations = {str(index): 0 for index in range(total)}
    client._completed_operation_order = [
        (index, str(index)) for index in range(total)
    ]
    heapify_calls = 0
    heapify = client_module.heapq.heapify

    def counted_heapify(values):
        nonlocal heapify_calls
        heapify_calls += 1
        heapify(values)

    monkeypatch.setattr(client_module.heapq, "heapify", counted_heapify)
    for index in range(total):
        client._completed_operations.pop(str(index))
        client._compact_completed_operation_order()

    assert client._completed_operation_order == []
    assert heapify_calls <= 4


@pytest.mark.asyncio
async def test_trainer_consumes_only_the_exact_cancelled_sequence() -> None:
    run = object.__new__(MonarchTrainerRun)
    run.run_spec = SimpleNamespace(run_id="run")
    run._cancelled_operations = {}
    run._operation_sequence_ids = {}
    run._operations = {}
    run._snapshot_launches = {}
    run._next_operation_sequence = 0
    run._learner_version = 7
    run._open_forward_backward_ids = ["prior-gradient"]
    run._lock = asyncio.Lock()
    run._jobs = {}
    run._closed = False
    run._valid = True

    first = OperationRef(
        run_id="run",
        operation_id="cancelled-0",
        sequence_id=0,
        learner_parent_version=7,
        kind="forward_backward",
    )
    await run.consume_cancelled_operation(first)
    await run.consume_cancelled_operation(first)
    assert run._next_operation_sequence == 1
    assert run._open_forward_backward_ids == ["prior-gradient"]
    run._operations["prior-gradient"] = ("fingerprint", {})
    run._operation_sequence_ids["prior-gradient"] = 0
    run.retire_operation("prior-gradient")
    assert run._open_forward_backward_ids == ["prior-gradient"]

    with pytest.raises(RuntimeError, match="gapless"):
        await run.consume_cancelled_operation(
            first.model_copy(update={"operation_id": "cancelled-2", "sequence_id": 2})
        )
    await run.consume_cancelled_operation(
        first.model_copy(update={"operation_id": "cancelled-1", "sequence_id": 1})
    )
    assert run._next_operation_sequence == 2


@pytest.mark.asyncio
async def test_retirement_failure_keeps_heap_retryable_and_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client_module, "_MAX_RETAINED_COMPLETED_OPERATIONS", 1)
    service = _Service()
    client = _client(service)

    async def execute(admission, _own_task):
        return admission.ref.operation_id

    first = await client._submit(
        _save("first", 0), kind="save_sampler", execute=execute
    )
    assert await first.result() == first.ref.operation_id
    service.fail_retirement = True
    second = await client._submit(
        _save("second", 1), kind="save_sampler", execute=execute
    )
    assert await second.result() == second.ref.operation_id
    await asyncio.sleep(0)

    assert first.ref.operation_id in client._operations
    assert client._completed_operation_order[0][1] == first.ref.operation_id
    with pytest.raises(BaseExceptionGroup, match="lifecycle failed"):
        await client._submit(_save("third", 2), kind="save_sampler", execute=execute)
    assert client.next_sequence_id == 2

    service.fail_retirement = False
    third = await client._submit(
        _save("third", 2), kind="save_sampler", execute=execute
    )
    assert await third.result() == third.ref.operation_id
    assert first.ref.operation_id not in client._operations
    await client.close()


@pytest.mark.asyncio
async def test_open_fb_bytes_are_evicted_without_losing_gradient_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client_module, "_MAX_RETAINED_COMPLETED_RESULT_BYTES", 8)
    service = _Service()
    client = _client(service)
    executions = 0

    async def forward(admission, _own_task):
        nonlocal executions
        executions += 1
        return _forward_backward_result(admission.ref.operation_id)

    request = _forward_backward("fb", 0)
    contribution = await client._submit(
        request, kind="forward_backward", execute=forward
    )
    assert (await contribution.result()).operation_id == contribution.ref.operation_id
    await asyncio.sleep(0)

    assert contribution.ref.operation_id not in client._operations
    assert client._completed_result_bytes == 0
    assert client._ledger._open_forward_backward_ids == [contribution.ref.operation_id]
    assert tuple(client._ledger._records) == (request.request_id,)
    assert service.retired == [contribution.ref.operation_id]
    with pytest.raises(RuntimeError, match="result is no longer retained"):
        await client._submit(request, kind="forward_backward", execute=forward)
    assert executions == 1
    assert client.next_sequence_id == 1
    assert client._ledger._open_forward_backward_ids == [contribution.ref.operation_id]

    step = await client.optim_step(
        OptimStepRequest(
            run_id="run",
            request_id="optim",
            sequence_id=1,
            optimizer=AdamConfig(learning_rate=1e-4),
        )
    )
    assert (await step.result()).contributing_forward_backward_operation_ids == (
        contribution.ref.operation_id,
    )
    assert request.request_id not in client._ledger._records
    assert client._evicted_forward_backward_operations == {}
    await client.close()


@pytest.mark.asyncio
async def test_cancelled_sequence_failure_poisons_and_surfaces() -> None:
    failure = RuntimeError("cancel tombstone failed")

    class FailingService(_Service):
        async def consume_cancelled_command(self, _ref: OperationRef) -> None:
            raise failure

    client = _client(FailingService())

    async def must_not_execute(_admission, _own_task):
        raise AssertionError("cancelled command executed")

    operation = await client._submit(
        _save("cancelled", 0), kind="save_sampler", execute=must_not_execute
    )
    await operation.cancel()
    with pytest.raises(RuntimeError, match="cancel tombstone failed"):
        await operation._ordered
    with pytest.raises(asyncio.CancelledError):
        await operation.result()
    with pytest.raises(BaseExceptionGroup, match="lifecycle failed"):
        await client._submit(
            _save("successor", 1), kind="save_sampler", execute=must_not_execute
        )
    with pytest.raises(BaseExceptionGroup, match="close failed"):
        await client.close()
    await client.close()


@pytest.mark.asyncio
async def test_batch_release_obeys_the_single_close_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client_module, "_TASK_DRAIN_TIMEOUT_S", 0.03)
    client = _client(_Service())
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    client._batch_releases.add(task)
    task.add_done_callback(client._batch_releases.discard)
    started = asyncio.get_running_loop().time()

    with pytest.raises(BaseExceptionGroup, match="close failed") as raised:
        await client.close()
    elapsed = asyncio.get_running_loop().time() - started
    assert any(isinstance(error, TimeoutError) for error in raised.value.exceptions)
    assert elapsed < 0.15

    release.set()
    await task
    await client.close()
