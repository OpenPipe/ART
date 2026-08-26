from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from art.distributed.trajectory_store import (
    TrajectoryGroupBundle,
    TrajectoryGroupDataIdentity,
)
from art.megatron.distributed_service import DistributedMegatronService
from art.megatron.optimizer_state import CheckpointFile, OptimizerAdapter
from art.megatron.runtime.specs import ResolvedCheckpointState
import art.megatron.training.client as client_module
from art.megatron.training.client import (
    LocalMegatronTrainingClient,
    _DeferredResult,
)
from art.training.contracts import (
    AdamConfig,
    ForwardBackwardRequest,
    ForwardResult,
    LossConfig,
    LossFnOutput,
    OptimStepRequest,
    PackingOutcome,
    RlTrajectoryBatch,
    SamplerPublication,
    SaveWeightsForSamplerRequest,
    TokenLogprobs,
)


class _Service:
    def __init__(self) -> None:
        self.retired: list[str] = []
        self.optimizer_calls = 0

    def retire_command_operation(self, operation_id: str) -> None:
        self.retired.append(operation_id)

    async def optimizer_command(self, ref, optimizer, contributions):
        del optimizer
        self.optimizer_calls += 1
        return {
            "metrics": {},
            "contributing_forward_backward_operation_ids": contributions,
        }, SimpleNamespace(operation_id=ref.operation_id)


def _client(service: _Service) -> LocalMegatronTrainingClient:
    return LocalMegatronTrainingClient(
        run_id="run",
        learner_version=0,
        backend=SimpleNamespace(),
        model=SimpleNamespace(),
        service=service,
    )


def _checkpoint(step: int) -> ResolvedCheckpointState:
    return ResolvedCheckpointState(
        adapter=OptimizerAdapter(
            identity=f"/checkpoint/{step}",
            training_session_id="session",
            step=step,
            generation_id=(f"step-{step:08d}-0123456789abcdef0123456789abcdef"),
            files=(
                CheckpointFile(name="adapter_config.json", size_bytes=1),
                CheckpointFile(name="adapter_model.safetensors", size_bytes=1),
            ),
        ),
    )


def test_physical_retention_prunes_checkpoint_and_publication_indexes() -> None:
    client = _client(_Service())
    client._checkpoints = {
        "latest": _checkpoint(3),
        "retained-alias": _checkpoint(1),
        "deleted-alias": _checkpoint(2),
    }
    client.prune_checkpoints(retain_steps={1, 3})
    assert tuple(client._checkpoints) == ("latest", "retained-alias")

    service = object.__new__(DistributedMegatronService)
    service._latest_step = 3
    service._serving_step = 3
    service._exact_adapter_refcounts = {1: 1}
    service._published_adapters = {1: object(), 2: object(), 3: object()}
    service.prune_checkpoint_metadata_locked(retain_steps={1, 3})
    assert set(service._published_adapters) == {1, 3}

    with pytest.raises(RuntimeError, match="omitted an active generation"):
        service.prune_checkpoint_metadata_locked(retain_steps={3})


def _save(request_id: str, sequence_id: int) -> SaveWeightsForSamplerRequest:
    return SaveWeightsForSamplerRequest(
        run_id="run",
        request_id=request_id,
        sequence_id=sequence_id,
        checkpoint_name=request_id,
        publication=SamplerPublication(mode="none"),
    )


async def _submit_value(
    client: LocalMegatronTrainingClient,
    request: SaveWeightsForSamplerRequest,
    value: Any,
):
    async def execute(_admission, _own_task):
        return value

    return await client._submit(request, kind="save_sampler", execute=execute)


@pytest.mark.asyncio
async def test_terminal_retry_window_retires_every_cache_together(monkeypatch) -> None:
    monkeypatch.setattr(client_module, "_MAX_RETAINED_COMPLETED_OPERATIONS", 2)
    service = _Service()
    client = _client(service)
    requests = tuple(_save(f"request-{index}", index) for index in range(3))
    operations = tuple(
        [
            await _submit_value(client, request, request.request_id)
            for request in requests
        ]
    )

    assert await asyncio.gather(*(operation.result() for operation in operations)) == [
        request.request_id for request in requests
    ]
    await asyncio.sleep(0)

    assert tuple(client._operations) == tuple(
        operation.ref.operation_id for operation in operations[1:]
    )
    assert tuple(client._ledger._records) == tuple(
        request.request_id for request in requests[1:]
    )
    assert service.retired == [operations[0].ref.operation_id]
    assert await operations[0].result() == requests[0].request_id
    assert await _submit_value(client, requests[1], "must-not-run") is operations[1]
    with pytest.raises(RuntimeError, match="gapless"):
        await _submit_value(client, requests[0], "must-not-run")


@pytest.mark.asyncio
async def test_incomplete_result_is_not_retired(monkeypatch) -> None:
    monkeypatch.setattr(client_module, "_MAX_RETAINED_COMPLETED_OPERATIONS", 1)
    service = _Service()
    client = _client(service)
    release = asyncio.Event()

    async def deferred() -> str:
        await release.wait()
        return "pending"

    async def launch(_admission, _own_task):
        return _DeferredResult(asyncio.create_task(deferred()))

    pending = await client._submit(
        _save("pending", 0), kind="save_sampler", execute=launch
    )
    await pending._ordered
    first = await _submit_value(client, _save("first", 1), "first")
    second = await _submit_value(client, _save("second", 2), "second")
    assert await first.result() == "first"
    assert await second.result() == "second"
    await asyncio.sleep(0)

    assert tuple(client._operations) == (
        pending.ref.operation_id,
        second.ref.operation_id,
    )
    assert service.retired == [first.ref.operation_id]

    release.set()
    assert await pending.result() == "pending"
    await asyncio.sleep(0)
    assert tuple(client._operations) == (second.ref.operation_id,)
    assert service.retired == [first.ref.operation_id, pending.ref.operation_id]


@pytest.mark.asyncio
async def test_completed_result_bytes_bound_retention(monkeypatch) -> None:
    monkeypatch.setattr(client_module, "_MAX_RETAINED_COMPLETED_RESULT_BYTES", 16)
    service = _Service()
    client = _client(service)

    def result(operation_id: str) -> ForwardResult:
        return ForwardResult(
            operation_id=operation_id,
            packing=PackingOutcome(
                packed_sequence_length=1,
                packed_sequences=1,
                target_packed_sequences=1,
                nominal_capacity_tokens=1,
                physical_tokens=1,
                non_padding_tokens=1,
                loss_bearing_tokens=1,
                trainable_assistant_tokens=1,
                policy_token_counts=None,
                group_shapes=(),
            ),
            loss_fn_outputs=(
                LossFnOutput(token_logprobs=TokenLogprobs(shape=(3,), data=b"x" * 12)),
            ),
        )

    operations = []
    for index in range(2):
        request = _save(f"result-{index}", index)
        operations.append(
            await _submit_value(client, request, result(request.request_id))
        )
        await operations[-1].result()
    await asyncio.sleep(0)

    assert tuple(client._operations) == (operations[-1].ref.operation_id,)
    assert service.retired == [operations[0].ref.operation_id]
    assert await operations[0].result() == result("result-0")


@pytest.mark.asyncio
async def test_optimizer_retires_only_its_terminal_fb_contributions() -> None:
    service = _Service()
    client = _client(service)
    forward_request = ForwardBackwardRequest(
        run_id="run",
        request_id="forward",
        sequence_id=0,
        batch=RlTrajectoryBatch(
            groups=(
                TrajectoryGroupBundle(
                    header=b"header",
                    records=(b"record",),
                    route_free_identity=TrajectoryGroupDataIdentity(
                        sha256="0" * 64, byte_count=len(b"headerrecord")
                    ),
                ),
            ),
            min_source_version=0,
            max_source_version=0,
        ),
        loss=LossConfig(name="cispo"),
    )
    forward = await client._submit(
        forward_request,
        kind="forward_backward",
        execute=lambda _admission, _own_task: asyncio.sleep(0, result="forward"),
    )
    assert await forward.result() == "forward"
    assert forward.ref.operation_id in client._operations

    optimizer_request = OptimStepRequest(
        run_id="run",
        request_id="optimizer",
        sequence_id=1,
        optimizer=AdamConfig(learning_rate=1e-6),
    )
    optimizer = await client.optim_step(optimizer_request)
    result = await optimizer.result()
    await asyncio.sleep(0)

    assert result.contributing_forward_backward_operation_ids == (
        forward.ref.operation_id,
    )
    assert forward.ref.operation_id not in client._operations
    assert "forward" not in client._ledger._records
    assert service.retired == [forward.ref.operation_id]
    assert await forward.result() == "forward"
    assert await client.optim_step(optimizer_request) is optimizer
    assert service.optimizer_calls == 1
    with pytest.raises(RuntimeError, match="gapless"):
        await client._submit(
            forward_request,
            kind="forward_backward",
            execute=lambda _admission, _own_task: asyncio.sleep(0, result="replayed"),
        )
