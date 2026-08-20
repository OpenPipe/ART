from __future__ import annotations

import asyncio
import threading
import time

import httpx
import pytest

from art.serverless import client as client_module
from art.serverless.client import RemoteTrainingClient, RemoteTrainingServiceClient
from art.serverless.contracts import (
    AdapterSpec,
    CreateTrainingRunRequest,
    TrainingRunSpec,
)
from art.training.contracts import ForwardBackwardRequest, LossConfig, RlTrajectoryBatch
from scratch.test_remote_training_client import FakeService
from scratch.test_serverless_distributed_pipeline_batch import (
    _backend,
    _full_group,
    _Queue,
    _summary,
    _train_kwargs,
)


async def _client() -> tuple[
    RemoteTrainingClient,
    RemoteTrainingServiceClient,
    FakeService,
    httpx.AsyncClient,
]:
    fake = FakeService()
    fake.fail_submit_once = False
    http = httpx.AsyncClient(
        base_url="http://test/v1/", transport=httpx.MockTransport(fake)
    )
    service = RemoteTrainingServiceClient(
        api_key="test",
        base_url="http://test/v1",
        control_http_client=http,
        transfer_http_client=http,
    )
    client = await RemoteTrainingClient.create(
        service,
        CreateTrainingRunRequest(
            spec=TrainingRunSpec(
                run_name="test",
                base_model="model",
                adapter=AdapterSpec(rank=1, alpha=32, target_modules=("q_proj",)),
            )
        ),
    )
    return client, service, fake, http


def _request(
    client: RemoteTrainingClient,
    batch: RlTrajectoryBatch,
    request_id: str,
    sequence: int,
) -> ForwardBackwardRequest:
    return ForwardBackwardRequest(
        run_id=client.run_id,
        request_id=request_id,
        sequence_id=sequence,
        batch=batch,
        loss=LossConfig(name="cispo"),
        return_token_logprobs=False,
    )


async def _close(
    client: RemoteTrainingClient,
    service: RemoteTrainingServiceClient,
    http: httpx.AsyncClient,
) -> None:
    await client.close_event_observer()
    await service.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_forward_preparation_is_one_off_loop_operation_and_order_checks_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, _, http = await _client()
    batch = RlTrajectoryBatch.from_groups([_full_group()], default_source_version=0)
    original_prepare = client_module.prepare_training_batch
    original_encode = client_module.encode_forward_submission
    original_fingerprint = client_module.remote_request_fingerprint
    threads: list[int] = []

    def prepare(value):
        threads.append(threading.get_ident())
        return original_prepare(value)

    def encode(request, value):
        threads.append(threading.get_ident())
        return original_encode(request, value)

    def fingerprint(request):
        threads.append(threading.get_ident())
        return original_fingerprint(request)

    monkeypatch.setattr(client_module, "prepare_training_batch", prepare)
    monkeypatch.setattr(client_module, "encode_forward_submission", encode)
    monkeypatch.setattr(client_module, "remote_request_fingerprint", fingerprint)
    with pytest.raises(ValueError, match="expected sequence 0, got 1"):
        await client.forward_backward(_request(client, batch, "future", 1))
    assert threads == []
    await client.forward_backward(_request(client, batch, "forward", 0))
    assert len(threads) == 3
    assert len(set(threads)) == 1
    assert threads[0] != threading.get_ident()
    await _close(client, service, http)


@pytest.mark.asyncio
async def test_cancelled_preparation_settles_without_admission_or_lock_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, fake, http = await _client()
    batch = RlTrajectoryBatch.from_groups([_full_group()], default_source_version=0)
    original = client_module.prepare_training_batch
    started, release = threading.Event(), threading.Event()

    def blocked(value):
        started.set()
        if not release.wait(2):
            raise TimeoutError("test did not release preparation")
        return original(value)

    monkeypatch.setattr(client_module, "prepare_training_batch", blocked)
    request = _request(client, batch, "forward", 0)
    task = asyncio.create_task(client.forward_backward(request))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    await asyncio.sleep(0.02)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake.submit_bodies == []
    assert client.next_sequence_id == 0
    monkeypatch.setattr(client_module, "prepare_training_batch", original)
    await client.forward_backward(request)
    await _close(client, service, http)


@pytest.mark.asyncio
async def test_same_run_forward_preparation_is_bounded_and_ordered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, _, http = await _client()
    batch = RlTrajectoryBatch.from_groups([_full_group()], default_source_version=0)
    original = client_module.prepare_training_batch
    started, release = threading.Event(), threading.Event()
    state = {"active": 0, "calls": 0, "max_active": 0}
    lock = threading.Lock()

    def blocked(value):
        with lock:
            state["active"] += 1
            state["calls"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            call = state["calls"]
        if call == 1:
            started.set()
            if not release.wait(2):
                raise TimeoutError("test did not release preparation")
        result = original(value)
        with lock:
            state["active"] -= 1
        return result

    monkeypatch.setattr(client_module, "prepare_training_batch", blocked)
    first = asyncio.create_task(
        client.forward_backward(_request(client, batch, "first", 0))
    )
    assert await asyncio.to_thread(started.wait, 1)
    second = asyncio.create_task(
        client.forward_backward(_request(client, batch, "second", 1))
    )
    await asyncio.sleep(0.02)
    assert state["calls"] == 1
    release.set()
    await asyncio.gather(first, second)
    assert state == {"active": 0, "calls": 2, "max_active": 1}
    await _close(client, service, http)


class _ConcurrentQueue(_Queue):
    def __init__(self, *, fail_receive: bool = False) -> None:
        super().__init__(_full_group())
        self.fail_receive = fail_receive
        self.receive_started = asyncio.Event()
        self.mark_started = asyncio.Event()

    async def receive_bundle(self, ref):
        self.receive_started.set()
        await asyncio.wait_for(self.mark_started.wait(), 1)
        if self.fail_receive:
            raise RuntimeError("materialization failed")
        return await super().receive_bundle(ref)

    async def mark_packed(self, selections, generation_id):
        self.mark_started.set()
        await asyncio.wait_for(self.receive_started.wait(), 1)
        await super().mark_packed(selections, generation_id)


@pytest.mark.asyncio
async def test_bundle_receive_overlaps_mark_and_failure_releases_exactly_once() -> None:
    backend, model, _, _ = _backend()
    queue = _ConcurrentQueue()
    group, _ = _summary(queue)
    started = time.perf_counter()
    context = await backend.prepare_pipeline_commands(
        model,
        [group],
        train_kwargs=_train_kwargs(),
        learner_parent_version=3,
    )
    assert time.perf_counter() - started < 1
    assert queue.marked == [((context.selections[0],), context.generation_id)]
    await context.abort(None, None, None, optimizer_admitted=False)
    assert queue.released == [
        ((context.selections[0],), "discarded", context.generation_id)
    ]

    queue = _ConcurrentQueue(fail_receive=True)
    group, _ = _summary(queue)
    with pytest.raises(RuntimeError, match="materialization failed"):
        await backend.prepare_pipeline_commands(
            model,
            [group],
            train_kwargs=_train_kwargs(),
            learner_parent_version=3,
        )
    assert group._distributed_lease is None
    assert len(queue.released) == 1
    assert queue.released[0][1:] == ("discarded", queue.marked[0][1])
