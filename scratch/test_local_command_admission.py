from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from pydantic import ValidationError
import pytest

from art.distributed.trajectory_store import TrajectoryGroupBundle
from art.megatron.training.client import LocalMegatronTrainingClient
from art.training.contracts import (
    AdamConfig,
    ForwardBackwardRequest,
    LossConfig,
    OperationKind,
    OptimStepRequest,
    RlTrajectoryBatch,
    SamplerPublication,
    SaveWeightsForSamplerRequest,
)
from art.training.sequencing import CommandAdmissionPolicy


def _client(
    policy: CommandAdmissionPolicy | None = None,
) -> LocalMegatronTrainingClient:
    return LocalMegatronTrainingClient(
        run_id="run",
        learner_version=0,
        backend=SimpleNamespace(),
        model=SimpleNamespace(),
        service=SimpleNamespace(retire_command_operation=lambda _operation_id: None),
        admission_policy=policy,
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


def _save(request_id: str, sequence_id: int) -> SaveWeightsForSamplerRequest:
    return SaveWeightsForSamplerRequest(
        run_id="run",
        request_id=request_id,
        sequence_id=sequence_id,
        checkpoint_name=request_id,
        publication=SamplerPublication(mode="none"),
    )


async def _submit(
    client: LocalMegatronTrainingClient,
    request: Any,
    kind: OperationKind,
    execute: Any,
):
    return await client._submit(request, kind=kind, execute=execute)


def _admission_state(client: LocalMegatronTrainingClient) -> tuple[Any, ...]:
    ledger = client._ledger
    return (
        client.next_sequence_id,
        client.projected_learner_version,
        tuple(ledger._open_forward_backward_ids),
        tuple(ledger._records),
        frozenset(ledger._nonterminal_request_ids),
        tuple(client._operations),
        client._sequence_tail,
    )


@pytest.mark.asyncio
async def test_default_128_command_boundary_is_atomic_and_releases_cancelled() -> None:
    policy = CommandAdmissionPolicy()
    assert policy.model_dump() == {
        "max_command_depth": 128,
        "max_gradient_contributions": 64,
    }
    with pytest.raises(ValidationError):
        CommandAdmissionPolicy(max_command_depth=0)

    client = _client()
    release = asyncio.Event()

    async def blocked(_admission):
        await release.wait()

    first_request = _forward_backward("fb", 0)
    first = await _submit(client, first_request, "forward_backward", blocked)
    saves = [
        await _submit(
            client, _save(f"save-{index}", index + 1), "save_sampler", blocked
        )
        for index in range(127)
    ]
    assert len(client._ledger._nonterminal_request_ids) == 128
    assert await _submit(client, first_request, "forward_backward", blocked) is first

    before = _admission_state(client)
    optimizer = OptimStepRequest(
        run_id="run",
        request_id="optimizer",
        sequence_id=128,
        optimizer=AdamConfig(learning_rate=1e-4),
    )
    with pytest.raises(RuntimeError, match="command depth limit"):
        await _submit(client, optimizer, "optim_step", blocked)
    assert _admission_state(client) == before

    await saves[-1].cancel()
    await asyncio.sleep(0)
    assert len(client._ledger._nonterminal_request_ids) == 127
    admitted = await _submit(client, optimizer, "optim_step", blocked)
    assert admitted.ref.sequence_id == 128
    assert client.next_sequence_id == 129
    assert client.projected_learner_version == 1
    assert client._ledger._open_forward_backward_ids == []
    assert len(client._ledger._records) == 129
    await client.close()


@pytest.mark.asyncio
async def test_default_64_gradient_boundary_and_terminal_release_match_remote() -> None:
    client = _client()
    release = asyncio.Event()

    async def blocked(_admission):
        await release.wait()

    requests = tuple(_forward_backward(f"fb-{index}", index) for index in range(65))
    operations = [
        await _submit(client, request, "forward_backward", blocked)
        for request in requests[:64]
    ]
    assert len(client._ledger._open_forward_backward_ids) == 64
    assert (
        await _submit(client, requests[0], "forward_backward", blocked) is operations[0]
    )

    before = _admission_state(client)
    with pytest.raises(RuntimeError, match="gradient contribution limit"):
        await _submit(client, requests[64], "forward_backward", blocked)
    assert _admission_state(client) == before

    await operations[0].cancel()
    await asyncio.sleep(0)
    assert len(client._ledger._open_forward_backward_ids) == 63
    replacement = await _submit(
        client,
        requests[64].model_copy(update={"sequence_id": 64}),
        "forward_backward",
        blocked,
    )
    assert len(client._ledger._open_forward_backward_ids) == 64
    assert (
        await _submit(client, requests[0], "forward_backward", blocked) is operations[0]
    )

    optimizer = await _submit(
        client,
        OptimStepRequest(
            run_id="run",
            request_id="optimizer",
            sequence_id=65,
            optimizer=AdamConfig(learning_rate=1e-4),
        ),
        "optim_step",
        blocked,
    )
    assert len(optimizer._admission.contributing_forward_backward_operation_ids) == 64
    assert replacement.ref.operation_id in (
        optimizer._admission.contributing_forward_backward_operation_ids
    )
    assert operations[0].ref.operation_id not in (
        optimizer._admission.contributing_forward_backward_operation_ids
    )
    assert client._ledger._open_forward_backward_ids == []
    assert (
        await _submit(
            client,
            _forward_backward("after-optimizer", 66),
            "forward_backward",
            blocked,
        )
    ).ref.sequence_id == 66
    await client.close()


@pytest.mark.asyncio
async def test_configured_depth_releases_failure_without_losing_retry() -> None:
    client = _client(CommandAdmissionPolicy(max_command_depth=1))
    failure = RuntimeError("expected failure")
    executions = 0

    async def fail(_admission):
        nonlocal executions
        executions += 1
        raise failure

    request = _save("failed", 0)
    operation = await _submit(client, request, "save_sampler", fail)
    with pytest.raises(RuntimeError, match="expected failure") as raised:
        await operation.result()
    assert raised.value is failure
    await asyncio.sleep(0)
    assert client._ledger._nonterminal_request_ids == set()
    assert tuple(client._ledger._records) == (request.request_id,)
    assert await _submit(client, request, "save_sampler", fail) is operation
    assert executions == 1

    release = asyncio.Event()

    async def blocked(_admission):
        await release.wait()

    successor = await _submit(client, _save("successor", 1), "save_sampler", blocked)
    assert successor.ref.sequence_id == 1
    assert len(client._ledger._nonterminal_request_ids) == 1
    await client.close()

    mutating = _client(CommandAdmissionPolicy(max_gradient_contributions=1))
    failed_forward_backward = await _submit(
        mutating, _forward_backward("failed-fb", 0), "forward_backward", fail
    )
    with pytest.raises(RuntimeError, match="expected failure"):
        await failed_forward_backward.result()
    await asyncio.sleep(0)
    assert mutating._ledger._nonterminal_request_ids == set()
    assert mutating._ledger._open_forward_backward_ids == []
    await mutating.close()
