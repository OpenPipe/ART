from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import threading

import httpx
import pytest

import art.serverless.client as client_module
from art.serverless.client import RemoteTrainingClient
from art.serverless.contracts import AdapterSpec, TrainingRunSpec, TrainingRunView
import art.serverless.data_plane as data_plane
from art.training.contracts import (
    ForwardRequest,
    LossConfig,
    OperationRef,
    TokenizedTrainingBatch,
)
from art.training.tokenized import TokenizedDatum, TokenizedMoeRoutes


def _operation_id(run_id: str, request_id: str) -> str:
    return hashlib.sha256(f"{run_id}\0{request_id}".encode()).hexdigest()


def _batch() -> TokenizedTrainingBatch:
    return TokenizedTrainingBatch(
        datums=(
            TokenizedDatum(
                input_tokens=(1, 2),
                target_tokens=((2,), (3,)),
                weights=((0.0,), (1.0,)),
                moe_routes=TokenizedMoeRoutes(
                    num_experts=2,
                    dtype="uint8",
                    shape=(2, 1, 1),
                    data=(bytes((0, 1)),),
                ),
            ),
        )
    )


def _request(sequence_id: int, batch: TokenizedTrainingBatch) -> ForwardRequest:
    return ForwardRequest(
        run_id="run",
        request_id=f"request-{sequence_id}",
        sequence_id=sequence_id,
        batch=batch,
        loss=LossConfig(name="cross_entropy"),
    )


class _Events:
    def __init__(self) -> None:
        self._terminal: dict[str, asyncio.Future] = {}

    def reserve(self, operation_id: str) -> asyncio.Future:
        return self._terminal.setdefault(
            operation_id, asyncio.get_running_loop().create_future()
        )

    def claim(self, _operation_id: str, _future: asyncio.Future) -> None:
        pass


class _Service:
    def __init__(self, *, fail_once: bool = False) -> None:
        self.fail_once = fail_once
        self.attempted_sequences: list[int] = []

    async def submit_forward(self, kind, request, _payload) -> OperationRef:
        self.attempted_sequences.append(request.sequence_id)
        if self.fail_once:
            self.fail_once = False
            raise httpx.ConnectError("submission interrupted")
        return OperationRef(
            run_id=request.run_id,
            operation_id=_operation_id(request.run_id, request.request_id),
            sequence_id=request.sequence_id,
            learner_parent_version=0,
            kind=kind,
        )


def _client(service: _Service) -> RemoteTrainingClient:
    now = datetime.now(timezone.utc)
    client = RemoteTrainingClient(
        service,  # type: ignore[arg-type]
        TrainingRunView(
            run_id="run",
            spec=TrainingRunSpec(
                run_name="test",
                base_model="model",
                adapter=AdapterSpec(rank=1, target_modules=("q_proj",)),
            ),
            status="open",
            next_sequence_id=0,
            projected_learner_version=0,
            committed_learner_version=0,
            created_at=now,
            updated_at=now,
        ),
    )
    client._events = _Events()  # type: ignore[assignment]
    return client


async def _wait_for(event: threading.Event) -> None:
    async with asyncio.timeout(2):
        while not event.is_set():
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_exact_token_encoding_runs_outside_the_run_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(_Service())
    started, release = threading.Event(), threading.Event()
    prepare = client_module._prepare_forward_submission

    def gated_prepare(*args, **kwargs):
        assert not client._lock.locked()
        started.set()
        assert release.wait(2)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(client_module, "_prepare_forward_submission", gated_prepare)
    admission = asyncio.create_task(client.forward(_request(0, _batch())))
    await _wait_for(started)
    try:
        await asyncio.wait_for(client._lock.acquire(), 0.1)
        client._lock.release()
    finally:
        release.set()
    await admission


@pytest.mark.asyncio
async def test_concurrent_exact_token_admissions_keep_arrival_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _Service()
    client = _client(service)
    first_started = threading.Event()
    release_first = threading.Event()
    second_prepared = threading.Event()
    prepare = client_module._prepare_forward_submission

    def reordered_prepare(request, encoded_batch):
        if request.sequence_id == 0:
            first_started.set()
            assert release_first.wait(2)
        prepared = prepare(request, encoded_batch)
        if request.sequence_id == 1:
            second_prepared.set()
        return prepared

    monkeypatch.setattr(client_module, "_prepare_forward_submission", reordered_prepare)
    first = asyncio.create_task(client.forward(_request(0, _batch())))
    await _wait_for(first_started)
    second = asyncio.create_task(client.forward(_request(1, _batch())))
    try:
        await _wait_for(second_prepared)
        assert service.attempted_sequences == []
    finally:
        release_first.set()
    await asyncio.gather(first, second)

    assert service.attempted_sequences == [0, 1]
    assert client.next_sequence_id == 2


@pytest.mark.asyncio
async def test_exact_token_retry_reuses_the_remembered_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _Service(fail_once=True)
    client = _client(service)
    batch = _batch()
    request = _request(0, batch)
    encode = data_plane._encode_tokenized_wire_batch
    encoded_batches = 0

    def tracked_encode(*args, **kwargs):
        nonlocal encoded_batches
        encoded_batches += 1
        return encode(*args, **kwargs)

    monkeypatch.setattr(data_plane, "_encode_tokenized_wire_batch", tracked_encode)
    with pytest.raises(httpx.ConnectError, match="submission interrupted"):
        await client.forward(request)
    assert batch.encoded_payload() is not None

    await client.forward(request)

    assert encoded_batches == 1
    assert service.attempted_sequences == [0, 0]
