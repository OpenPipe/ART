from __future__ import annotations

from array import array
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import threading
from types import SimpleNamespace

import httpx
import pytest

from art import Trajectory, TrajectoryGroup
from art.distributed.trajectory_store import TrajectoryGroupBundle
from art.pipeline_tuner.config import PackedGroupShape, PackingLeafShape
import art.serverless.client as client_module
from art.serverless.client import (
    RemoteTrainingClient,
    RemoteTrainingError,
    RemoteTrainingOperation,
    RemoteTrainingOperationCancelled,
    RemoteTrainingServiceClient,
    _ByteBudget,
    _ResultAcknowledger,
)
from art.serverless.contracts import (
    FORWARD_BACKWARD_PREPARED_EVENT,
    MAX_OPERATION_RESULT_BYTES,
    AdapterSpec,
    CreateTrainingRunRequest,
    OperationResultRef,
    OperationView,
    TrainingRunSpec,
    TrainingRunView,
    remote_request_fingerprint,
)
from art.serverless.data_plane import (
    FORWARD_SUBMISSION_PREFIX_BYTES,
    EncodedRouteObject,
    EncodedTrainingObject,
    VerifiedOperationResultPayload,
    decode_forward_submission_manifest,
    decode_forward_submission_prefix,
    decode_operation_result,
    encode_operation_result,
)
from art.training.contracts import (
    AdamConfig,
    ForwardBackwardRequest,
    ForwardBackwardResult,
    LossConfig,
    LossFnOutput,
    OperationRef,
    OptimStepRequest,
    OptimStepResult,
    PackingOutcome,
    RlTrajectoryBatch,
    TokenizedTrainingBatch,
)
from art.training.tokenized import TokenizedDatum


@pytest.mark.asyncio
async def test_result_acknowledger_isolates_failed_results() -> None:
    attempts: list[str] = []

    async def acknowledge(_run_id: str, operation_id: str) -> None:
        attempts.append(operation_id)
        if operation_id == "failed":
            raise RuntimeError("failed acknowledgement")

    acknowledger = _ResultAcknowledger(acknowledge)
    await acknowledger.submit("run", "failed")
    await acknowledger.submit("run", "succeeded")
    with pytest.raises(RemoteTrainingError, match=r"failed for 1 result"):
        await acknowledger.close()
    assert attempts == ["failed", "succeeded"]


def _operation_id(run_id: str, request_id: str) -> str:
    return hashlib.sha256(f"{run_id}\0{request_id}".encode()).hexdigest()


class FakeService:
    def __init__(self) -> None:
        self.run = None
        self.operations = {}
        self.submit_bodies = []
        self.training_data = {}
        self.operation_results = {}
        self.acknowledged_results = []
        self.events = []
        self.nonempty_event_pages = 0
        self.operation_status_gets = 0
        self.fail_submit_once = True
        self.admit_before_submit_failure = False
        self.reclaim_failures = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = (
            None
            if path.endswith("/forward_backward")
            else json.loads(request.content)
            if request.content
            else None
        )
        now = datetime.now(timezone.utc).isoformat()
        if path == "/v1/training/runs":
            self.run = {
                "run_id": "run",
                "spec": body["spec"],
                "checkpoint": None,
                "restore_optimizer": False,
                "status": "open",
                "next_sequence_id": 0,
                "projected_learner_version": 0,
                "committed_learner_version": 0,
                "slot_id": None,
                "created_at": now,
                "updated_at": now,
            }
            return httpx.Response(200, json=self.run)
        if path == "/v1/training/runs/run/forward_backward":
            self.submit_bodies.append(request.content)
            if self.reclaim_failures:
                self.reclaim_failures -= 1
                return httpx.Response(
                    503,
                    json={
                        "detail": {
                            "code": "training_input_reclaiming",
                            "message": "training input object is being reclaimed",
                        }
                    },
                )
            fail = self.fail_submit_once
            self.fail_submit_once = False
            if fail and not self.admit_before_submit_failure:
                return httpx.Response(503, json={"detail": "retry"})
            submission = _decode_submission(request.content)
            for value in (*submission.objects, *submission.route_objects):
                self.training_data[value.ref.object_id] = value.payload
            ref = self._ref(submission.request, "forward_backward", None)
            result = ForwardBackwardResult(
                operation_id=ref["operation_id"],
                packing=PackingOutcome(
                    packed_sequence_length=8,
                    packed_sequences=1,
                    target_packed_sequences=1,
                    nominal_capacity_tokens=8,
                    physical_tokens=2,
                    non_padding_tokens=2,
                    loss_bearing_tokens=1,
                    trainable_assistant_tokens=1,
                    policy_token_counts=None,
                    group_shapes=(),
                ),
                loss_fn_outputs=(LossFnOutput(token_logprobs=(0.0,)),),
            )
            result_ref, payload = encode_operation_result(result)
            self.operation_results[ref["operation_id"]] = payload
            self.events.append(self._preparation_event(ref, now))
            terminal = self._event(ref, result_ref, now)
            self.events.append(terminal)
            self.operations[ref["operation_id"]] = self._operation(
                ref, result_ref, now, submission.request, terminal["cursor"]
            )
            if fail:
                return httpx.Response(503, json={"detail": "response lost"})
            return httpx.Response(200, json=ref)
        if path == "/v1/training/runs/run/optim_step":
            ref = self._ref(body, "optim_step", 1)
            result = OptimStepResult(
                operation_id=ref["operation_id"],
                contributing_forward_backward_operation_ids=(
                    _operation_id("run", "forward"),
                ),
            )
            terminal = self._event(ref, result, now)
            self.events.append(terminal)
            self.operations[ref["operation_id"]] = self._operation(
                ref,
                result,
                now,
                OptimStepRequest.model_validate(body),
                terminal["cursor"],
            )
            return httpx.Response(200, json=ref)
        if path == "/v1/training/runs/run/events":
            after = int(request.url.params.get("after", "0"))
            events = [event for event in self.events if event["cursor"] > after]
            self.nonempty_event_pages += bool(events)
            return httpx.Response(
                200,
                json={
                    "events": events,
                    "next_cursor": events[-1]["cursor"] if events else after,
                },
            )
        if path.endswith("/result"):
            operation_id = path.split("/")[-2]
            if request.method == "DELETE":
                self.acknowledged_results.append(operation_id)
                self.operation_results.pop(operation_id, None)
                return httpx.Response(204)
            return httpx.Response(200, content=self.operation_results[operation_id])
        if path.startswith("/v1/training/runs/run/operations/"):
            self.operation_status_gets += 1
            operation = self.operations.get(path.rsplit("/", 1)[1])
            return (
                httpx.Response(200, json=operation)
                if operation is not None
                else httpx.Response(404, json={"detail": "not found"})
            )
        if path == "/v1/training/runs/run:close":
            self.run["status"] = "closing"
            self.events.append(
                {
                    "cursor": len(self.events) + 1,
                    "run_id": "run",
                    "operation_id": None,
                    "event": "run_closed",
                    "payload": {},
                    "created_at": now,
                }
            )
            return httpx.Response(200, json=self.run)
        if path == "/v1/training/runs/run":
            self.run["status"] = "closed"
            return httpx.Response(200, json=self.run)
        raise AssertionError(path)

    @staticmethod
    def _ref(body, kind, output):
        sequence_id = (
            body["sequence_id"] if isinstance(body, dict) else body.sequence_id
        )
        run_id = body["run_id"] if isinstance(body, dict) else body.run_id
        request_id = body["request_id"] if isinstance(body, dict) else body.request_id
        return {
            "run_id": run_id,
            "operation_id": _operation_id(run_id, request_id),
            "sequence_id": sequence_id,
            "learner_parent_version": 0,
            "reserved_output_learner_version": output,
            "kind": kind,
        }

    @staticmethod
    def _operation(ref, result, now, request, event_cursor):
        return {
            "ref": ref,
            "request_id": request.request_id,
            "request_fingerprint": remote_request_fingerprint(request),
            "status": "succeeded",
            "gradient_disposition": (
                "contributes" if ref["kind"] == "forward_backward" else None
            ),
            "result": result.model_dump(mode="json"),
            "error": None,
            "event_cursor": event_cursor,
            "created_at": now,
            "updated_at": now,
        }

    def _preparation_event(self, ref, now):
        return {
            "cursor": len(self.events) + 1,
            "run_id": ref["run_id"],
            "operation_id": ref["operation_id"],
            "event": FORWARD_BACKWARD_PREPARED_EVENT,
            "payload": {"gradient_disposition": "contributes"},
            "created_at": now,
        }

    def _event(self, ref, result, now):
        cursor = len(self.events) + 1
        return {
            "cursor": cursor,
            "run_id": ref["run_id"],
            "operation_id": ref["operation_id"],
            "event": "operation_succeeded",
            "payload": result.model_dump(mode="json"),
            "created_at": now,
        }


class _ChunkedResult(httpx.AsyncByteStream):
    def __init__(self, payload: bytes, chunks: int = 3) -> None:
        self.payload = payload
        self.chunks = chunks
        self.reads = 0

    async def __aiter__(self):
        width = max(1, len(self.payload) // self.chunks)
        for start in range(0, len(self.payload), width):
            self.reads += 1
            yield self.payload[start : start + width]


def _decode_submission(payload: bytes):
    manifest_bytes = decode_forward_submission_prefix(
        payload[:FORWARD_SUBMISSION_PREFIX_BYTES]
    )
    offset = FORWARD_SUBMISSION_PREFIX_BYTES + manifest_bytes
    manifest = decode_forward_submission_manifest(
        payload[FORWARD_SUBMISSION_PREFIX_BYTES:offset]
    )
    objects = []
    for ref in manifest.objects:
        objects.append(
            EncodedTrainingObject(
                ref=ref, payload=payload[offset : offset + ref.byte_count]
            )
        )
        offset += ref.byte_count
    routes = []
    for ref in manifest.route_objects:
        route_payload = payload[offset : offset + ref.byte_count]
        routes.append(
            EncodedRouteObject(
                ref=ref,
                chunks=(memoryview(route_payload).toreadonly(),),
            )
        )
        offset += ref.byte_count
    assert offset == len(payload)
    return SimpleNamespace(
        request=manifest.request,
        objects=tuple(objects),
        route_objects=tuple(routes),
    )


def _forward_result() -> ForwardBackwardResult:
    return ForwardBackwardResult(
        operation_id="operation",
        packing=PackingOutcome(
            packed_sequence_length=8,
            packed_sequences=1,
            target_packed_sequences=1,
            nominal_capacity_tokens=8,
            physical_tokens=2,
            non_padding_tokens=2,
            loss_bearing_tokens=1,
            trainable_assistant_tokens=1,
            policy_token_counts=None,
            group_shapes=(),
        ),
        loss_fn_outputs=(LossFnOutput(token_logprobs=(-1.0,)),),
    )


def _verified_payload(
    ref: OperationResultRef, payload: bytes | bytearray
) -> VerifiedOperationResultPayload:
    return VerifiedOperationResultPayload(ref=ref, payload=payload)


def test_operation_result_sidecar_preserves_compact_packing_arrays():
    result = ForwardBackwardResult(
        operation_id="operation",
        packing=PackingOutcome(
            packed_sequence_length=8,
            packed_sequences=1,
            target_packed_sequences=1,
            nominal_capacity_tokens=8,
            physical_tokens=4,
            non_padding_tokens=4,
            loss_bearing_tokens=2,
            trainable_assistant_tokens=2,
            policy_token_counts=None,
            group_shapes=(
                PackedGroupShape(
                    leaves=(
                        PackingLeafShape(
                            token_ids=array("I", (1, 2, 2**32 - 1)),
                            shareable_length=2,
                        ),
                    )
                ),
            ),
        ),
        loss_fn_outputs=(LossFnOutput(token_logprobs=(-1.25, -2.5)),),
    )
    ref, payload = encode_operation_result(result)
    restored = decode_operation_result(ref, payload, ForwardBackwardResult)
    assert restored == result
    assert restored.packing.group_shapes[0].leaves[0].token_ids.typecode == "I"


def test_operation_result_sidecar_preserves_candidate_logprob_shape():
    result = ForwardBackwardResult(
        operation_id="operation",
        packing=PackingOutcome(
            packed_sequence_length=8,
            packed_sequences=1,
            target_packed_sequences=1,
            nominal_capacity_tokens=8,
            physical_tokens=2,
            non_padding_tokens=2,
            loss_bearing_tokens=2,
            trainable_assistant_tokens=2,
            policy_token_counts=None,
            group_shapes=(),
        ),
        loss_fn_outputs=(LossFnOutput(token_logprobs=((-1.0, -2.0), (-3.0, -4.0))),),
    )
    ref, payload = encode_operation_result(result)
    assert decode_operation_result(ref, payload, ForwardBackwardResult) == result


def test_operation_result_reference_rejects_unbounded_receive_allocation():
    with pytest.raises(ValueError, match="less than or equal"):
        OperationResultRef(
            object_id="0" * 64,
            byte_count=MAX_OPERATION_RESULT_BYTES + 1,
        )


@pytest.mark.asyncio
async def test_operation_result_streams_into_exact_receive_buffer():
    payload = b"result-payload"
    stream = _ChunkedResult(payload)
    ref = OperationResultRef(
        object_id=hashlib.sha256(payload).hexdigest(), byte_count=len(payload)
    )

    async def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    http = httpx.AsyncClient(
        base_url="http://test/v1/", transport=httpx.MockTransport(handle)
    )
    service = RemoteTrainingServiceClient(
        api_key="test",
        base_url="http://test/v1",
        control_http_client=http,
        transfer_http_client=http,
    )
    received = await service._receive_operation_result(
        "run",
        "operation",
        ref,
    )
    assert isinstance(received, VerifiedOperationResultPayload)
    assert received.ref == ref
    assert bytes(received.payload) == payload
    assert stream.reads > 1
    await http.aclose()


@pytest.mark.asyncio
async def test_operation_result_hashes_each_chunk_once_without_decode_rescan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _forward_result()
    ref, payload = encode_operation_result(result)
    stream = _ChunkedResult(payload)
    initial_sizes: list[int] = []
    update_sizes: list[int] = []
    sha256 = hashlib.sha256

    class TrackedSha256:
        def __init__(self, initial: bytes = b"") -> None:
            initial_sizes.append(len(initial))
            self._digest = sha256(initial)

        def update(self, chunk: bytes) -> None:
            update_sizes.append(len(chunk))
            self._digest.update(chunk)

        def hexdigest(self) -> str:
            return self._digest.hexdigest()

    async def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    http = httpx.AsyncClient(
        base_url="http://test/v1/", transport=httpx.MockTransport(handle)
    )
    service = RemoteTrainingServiceClient(
        api_key="test",
        base_url="http://test/v1",
        control_http_client=http,
        transfer_http_client=http,
    )
    monkeypatch.setattr(client_module.hashlib, "sha256", TrackedSha256)

    assert (
        await service.receive_operation_result(
            "run", "operation", ref, ForwardBackwardResult
        )
        == result
    )
    assert initial_sizes == [0]
    assert len(update_sizes) == stream.reads > 1
    assert sum(update_sizes) == len(payload)
    await http.aclose()


@pytest.mark.asyncio
async def test_operation_result_stream_rejects_corruption() -> None:
    ref, payload = encode_operation_result(_forward_result())
    corrupted = bytearray(payload)
    corrupted[-1] ^= 1

    async def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_ChunkedResult(bytes(corrupted)))

    http = httpx.AsyncClient(
        base_url="http://test/v1/", transport=httpx.MockTransport(handle)
    )
    service = RemoteTrainingServiceClient(
        api_key="test",
        base_url="http://test/v1",
        control_http_client=http,
        transfer_http_client=http,
    )
    with pytest.raises(RemoteTrainingError, match="result hash changed"):
        await service._receive_operation_result("run", "operation", ref)
    await http.aclose()


def test_unverified_operation_result_decoder_rejects_corruption() -> None:
    ref, payload = encode_operation_result(_forward_result())
    corrupted = bytearray(payload)
    corrupted[-1] ^= 1

    with pytest.raises(ValueError, match="hash differs"):
        decode_operation_result(ref, corrupted, ForwardBackwardResult)


@pytest.mark.asyncio
async def test_operation_result_receive_budget_serializes_allocations(monkeypatch):
    payload = b"12345678"
    release = asyncio.Event()
    first_started = asyncio.Event()
    calls = 0

    class BlockedResult(httpx.AsyncByteStream):
        async def __aiter__(self):
            await release.wait()
            yield payload

    async def handle(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        first_started.set()
        return httpx.Response(200, stream=BlockedResult())

    http = httpx.AsyncClient(
        base_url="http://test/v1/", transport=httpx.MockTransport(handle)
    )
    service = RemoteTrainingServiceClient(
        api_key="test",
        base_url="http://test/v1",
        control_http_client=http,
        transfer_http_client=http,
        max_result_bytes_in_flight=2 * len(payload),
    )
    ref = OperationResultRef(
        object_id=hashlib.sha256(payload).hexdigest(), byte_count=len(payload)
    )
    monkeypatch.setattr(
        client_module,
        "decode_verified_operation_result",
        lambda value, _type: bytes(value.payload),
    )
    monkeypatch.setattr(client_module, "_forward_result_buffer_bytes", len)
    first = asyncio.create_task(
        service.receive_operation_result("run", "first", ref, ForwardBackwardResult)
    )
    await first_started.wait()
    second = asyncio.create_task(
        service.receive_operation_result("run", "second", ref, ForwardBackwardResult)
    )
    await asyncio.sleep(0)
    assert calls == 1
    release.set()
    assert await asyncio.gather(first, second) == [payload, payload]
    assert calls == 2
    await http.aclose()


@pytest.mark.asyncio
async def test_operation_result_rejects_content_length_before_allocation():
    async def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Length": "9"}, content=b"1234")

    http = httpx.AsyncClient(
        base_url="http://test/v1/", transport=httpx.MockTransport(handle)
    )
    service = RemoteTrainingServiceClient(
        api_key="test",
        base_url="http://test/v1",
        control_http_client=http,
        transfer_http_client=http,
    )
    with pytest.raises(RemoteTrainingError, match="Content-Length changed"):
        await service.receive_operation_result(
            "run",
            "operation",
            OperationResultRef(object_id="0" * 64, byte_count=4),
            ForwardBackwardResult,
        )
    await http.aclose()


@pytest.mark.asyncio
async def test_operation_result_receive_budget_close_wakes_waiters(monkeypatch):
    payload = b"1234"
    release = asyncio.Event()
    first_started = asyncio.Event()

    class BlockedResult(httpx.AsyncByteStream):
        async def __aiter__(self):
            first_started.set()
            await release.wait()
            yield payload

    async def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=BlockedResult())

    http = httpx.AsyncClient(
        base_url="http://test/v1/", transport=httpx.MockTransport(handle)
    )
    service = RemoteTrainingServiceClient(
        api_key="test",
        base_url="http://test/v1",
        control_http_client=http,
        transfer_http_client=http,
        max_result_bytes_in_flight=2 * len(payload),
    )
    ref = OperationResultRef(
        object_id=hashlib.sha256(payload).hexdigest(), byte_count=len(payload)
    )
    monkeypatch.setattr(
        client_module,
        "decode_verified_operation_result",
        lambda value, _type: bytes(value.payload),
    )
    monkeypatch.setattr(client_module, "_forward_result_buffer_bytes", len)
    first = asyncio.create_task(
        service.receive_operation_result("run", "first", ref, ForwardBackwardResult)
    )
    await first_started.wait()
    waiting = asyncio.create_task(
        service.receive_operation_result("run", "waiting", ref, ForwardBackwardResult)
    )
    await asyncio.sleep(0)
    await service.close()
    with pytest.raises(RemoteTrainingError, match="budget is closed"):
        await waiting
    release.set()
    assert await first == payload
    await http.aclose()


def test_tokenized_request_rejects_oversized_logprob_result(monkeypatch):
    monkeypatch.setattr("art.training.contracts.MAX_TOKENIZED_LOGPROB_VALUES", 3)
    with pytest.raises(ValueError, match="configured value limit"):
        TokenizedTrainingBatch(
            datums=(
                TokenizedDatum(
                    input_tokens=(1, 2),
                    target_tokens=((3, 4), (5, 6)),
                    weights=((1.0, 1.0), (1.0, 1.0)),
                ),
            )
        )


@pytest.mark.asyncio
async def test_cancelled_forward_is_removed_from_open_accumulation_immediately():
    now = datetime.now(timezone.utc)
    ref = OperationRef(
        run_id="run",
        operation_id=_operation_id("run", "forward"),
        sequence_id=0,
        learner_parent_version=0,
        kind="forward_backward",
    )

    class CancellationService(RemoteTrainingServiceClient):
        async def cancel_operation(self, *_args):
            return OperationView(
                ref=ref,
                request_id="forward",
                request_fingerprint="0" * 64,
                status="cancelled",
                gradient_disposition="pending",
                event_cursor=1,
                created_at=now,
                updated_at=now,
            )

    service = CancellationService.__new__(CancellationService)
    client = RemoteTrainingClient(
        service,
        TrainingRunView(
            run_id="run",
            spec=TrainingRunSpec(
                run_name="test",
                base_model="model",
                adapter=AdapterSpec(rank=1, target_modules=("q_proj",)),
            ),
            status="open",
            next_sequence_id=1,
            projected_learner_version=0,
            committed_learner_version=0,
            created_at=now,
            updated_at=now,
        ),
    )
    terminal = asyncio.get_running_loop().create_future()
    preparation = asyncio.get_running_loop().create_future()
    operation = RemoteTrainingOperation(
        ref,
        service,
        terminal,
        ForwardBackwardResult,
        preparation=preparation,
        on_cancelled=lambda: client._forget_forward_backward(ref.operation_id),
    )
    client._open_forward_backward.append(operation)

    await operation.cancel()

    assert client._open_forward_backward == []
    with pytest.raises(RemoteTrainingOperationCancelled):
        await operation.gradient_disposition()
    with pytest.raises(RemoteTrainingOperationCancelled):
        await operation.result()


@pytest.mark.parametrize(
    ("admit_before_failure", "reclaim_failures"),
    ((False, 0), (True, 0), (False, 4)),
)
@pytest.mark.asyncio
async def test_remote_client_retries_and_preserves_command_order(
    admit_before_failure: bool,
    reclaim_failures: int,
    monkeypatch: pytest.MonkeyPatch,
):
    fake = FakeService()
    fake.admit_before_submit_failure = admit_before_failure
    fake.fail_submit_once = reclaim_failures == 0
    fake.reclaim_failures = reclaim_failures
    monkeypatch.setattr(client_module, "_retry_delay", lambda _attempt: 0.001)
    paths: dict[str, list[str]] = {"control": [], "transfer": []}

    def transport(plane: str):
        def handle(request: httpx.Request) -> httpx.Response:
            paths[plane].append(request.url.path)
            return fake(request)

        return httpx.MockTransport(handle)

    control_http = httpx.AsyncClient(
        base_url="http://test/v1/", transport=transport("control")
    )
    transfer_http = httpx.AsyncClient(
        base_url="http://test/v1/", transport=transport("transfer")
    )
    service = RemoteTrainingServiceClient(
        api_key="test",
        base_url="http://test/v1",
        control_http_client=control_http,
        transfer_http_client=transfer_http,
        max_retries=0 if reclaim_failures else 1,
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
    batch = RlTrajectoryBatch(
        groups=(TrajectoryGroupBundle.from_group(TrajectoryGroup([Trajectory()])),),
        min_source_version=0,
        max_source_version=0,
    )
    request = ForwardBackwardRequest(
        run_id=client.run_id,
        request_id="forward",
        sequence_id=0,
        batch=batch,
        loss=LossConfig(name="cispo"),
    )
    forward = await client.forward_backward(request)
    assert await forward.gradient_disposition() == "contributes"
    assert client._open_forward_backward == [forward]
    assert await client.forward_backward(request) is forward
    assert len(fake.submit_bodies) == (
        reclaim_failures + 1 if reclaim_failures else 1 if admit_before_failure else 2
    )
    assert len(set(fake.submit_bodies)) == 1
    submitted = _decode_submission(fake.submit_bodies[0])
    batch_ref = submitted.request.batch.model_dump(mode="python")
    assert set(batch_ref) == {
        "kind",
        "groups",
        "min_source_version",
        "max_source_version",
    }
    data = batch_ref["groups"][0]["data"]
    assert set(data) == {"object_id", "sha256", "byte_count", "format"}
    assert len(fake.training_data) == 1
    assert data["object_id"] in fake.training_data
    optimizer = await client.optim_step(
        OptimStepRequest(
            run_id=client.run_id,
            request_id="optimizer",
            sequence_id=1,
            optimizer=AdamConfig(learning_rate=1e-6),
        )
    )
    assert optimizer.ref.reserved_output_learner_version == 1
    assert client._open_forward_backward == []
    assert client.next_sequence_id == 2
    assert client.projected_learner_version == 1
    decode_started, decode_release = threading.Event(), threading.Event()
    verified_decoder = client_module.decode_verified_operation_result

    def gated_decode(*args, **kwargs):
        decode_started.set()
        assert decode_release.wait(1), "result decode blocked the event loop"
        return verified_decoder(*args, **kwargs)

    async def release_decode():
        while not decode_started.is_set():
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        decode_release.set()

    monkeypatch.setattr(client_module, "decode_verified_operation_result", gated_decode)
    forward_result, optimizer_result, _ = await asyncio.gather(
        forward.result(), optimizer.result(), release_decode()
    )
    forward_operation_id = _operation_id("run", "forward")
    assert forward_result.operation_id == forward_operation_id
    assert optimizer_result.operation_id == _operation_id("run", "optimizer")
    assert optimizer_result.contributing_forward_backward_operation_ids == (
        forward_operation_id,
    )
    assert fake.operation_status_gets == max(1, reclaim_failures)
    assert 1 <= fake.nonempty_event_pages <= 2
    result_path = f"/v1/training/runs/run/operations/{forward_operation_id}/result"
    assert paths["control"].count(result_path) == 1
    assert paths["transfer"].count(result_path) == 1
    assert all(
        "/training/operations/" not in path
        for values in paths.values()
        for path in values
    )
    assert all(
        path.endswith("/forward_backward") or path.endswith("/result")
        for path in paths["transfer"]
    )
    await client.close()
    assert client._run.status == "closing"
    await client.wait_closed()
    await client.close_event_observer()
    assert client._run.status == "closed"
    await service.close()
    assert fake.acknowledged_results == [forward_operation_id]
    await asyncio.gather(control_http.aclose(), transfer_http.aclose())


def _budgeted_result_service(
    byte_count: int,
) -> tuple[RemoteTrainingServiceClient, httpx.AsyncClient]:
    http = httpx.AsyncClient(
        base_url="http://test/v1/",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(500, json={"detail": "unexpected"})
        ),
    )
    return (
        RemoteTrainingServiceClient(
            api_key="test",
            base_url="http://test/v1",
            control_http_client=http,
            transfer_http_client=http,
            max_result_bytes_in_flight=2 * byte_count,
        ),
        http,
    )


async def _wait_for_thread_event(event: threading.Event) -> None:
    async with asyncio.timeout(1):
        while not event.is_set():
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_result_budget_remains_reserved_through_cancelled_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    byte_count = 1 << 20
    service, http = _budgeted_result_service(byte_count)
    ref = OperationResultRef(object_id="a" * 64, byte_count=byte_count)
    downloads: list[str] = []
    decode_started, decode_release = threading.Event(), threading.Event()
    decode_calls = 0

    async def download(_run_id, operation_id, result_ref):
        assert result_ref is ref
        downloads.append(operation_id)
        return _verified_payload(ref, bytearray(byte_count))

    def decode(_payload, _result_type):
        nonlocal decode_calls
        decode_calls += 1
        if decode_calls == 1:
            decode_started.set()
            assert decode_release.wait(1)
        return SimpleNamespace(operation_id=f"result-{decode_calls}")

    monkeypatch.setattr(service, "_download_operation_result", download)
    monkeypatch.setattr(client_module, "decode_verified_operation_result", decode)
    monkeypatch.setattr(
        client_module, "_forward_result_buffer_bytes", lambda _: byte_count
    )
    first = asyncio.create_task(
        service.receive_operation_result("run", "first", ref, ForwardBackwardResult)
    )
    await _wait_for_thread_event(decode_started)
    second = asyncio.create_task(
        service.receive_operation_result("run", "second", ref, ForwardBackwardResult)
    )
    await asyncio.sleep(0)
    first.cancel()
    await asyncio.sleep(0)
    assert downloads == ["first"]
    assert service._result_budget._used == 2 * byte_count

    decode_release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert (await asyncio.wait_for(second, 1)).operation_id == "result-2"
    assert downloads == ["first", "second"]
    assert service._result_budget._used == 0
    await service.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_result_budget_releases_after_decode_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    byte_count = 1 << 20
    service, http = _budgeted_result_service(byte_count)
    ref = OperationResultRef(object_id="b" * 64, byte_count=byte_count)
    downloads: list[str] = []
    decode_started, decode_fail = threading.Event(), threading.Event()
    decode_calls = 0

    async def download(_run_id, operation_id, _result_ref):
        downloads.append(operation_id)
        return _verified_payload(ref, bytearray(byte_count))

    def decode(_payload, _result_type):
        nonlocal decode_calls
        decode_calls += 1
        if decode_calls == 1:
            decode_started.set()
            assert decode_fail.wait(1)
            raise ValueError("decode failed")
        return SimpleNamespace(operation_id="result-2")

    monkeypatch.setattr(service, "_download_operation_result", download)
    monkeypatch.setattr(client_module, "decode_verified_operation_result", decode)
    monkeypatch.setattr(
        client_module, "_forward_result_buffer_bytes", lambda _: byte_count
    )
    first = asyncio.create_task(
        service.receive_operation_result("run", "first", ref, ForwardBackwardResult)
    )
    await _wait_for_thread_event(decode_started)
    second = asyncio.create_task(
        service.receive_operation_result("run", "second", ref, ForwardBackwardResult)
    )
    await asyncio.sleep(0)
    assert downloads == ["first"]

    decode_fail.set()
    with pytest.raises(ValueError, match="decode failed"):
        await first
    assert (await asyncio.wait_for(second, 1)).operation_id == "result-2"
    assert downloads == ["first", "second"]
    assert service._result_budget._used == 0
    await service.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_result_receive_budget_tracks_encoded_and_decoded_peak() -> None:
    encoded_byte_count = 32 << 20
    budget = _ByteBudget(2 * encoded_byte_count)

    async with budget.reserve(
        encoded_byte_count=encoded_byte_count,
        decoded_headroom_byte_count=encoded_byte_count,
    ) as reservation:
        assert budget._used == 64 << 20
        await reservation.transfer_to_decoded(31 << 20)
        assert budget._used == 31 << 20

    assert budget._used == 0


@pytest.mark.asyncio
async def test_impossible_receive_peak_is_rejected_before_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded_byte_count = 32 << 20
    service, http = _budgeted_result_service(encoded_byte_count)
    service._result_budget = _ByteBudget((2 * encoded_byte_count) - 1)
    downloaded = False

    async def download(*_args):
        nonlocal downloaded
        downloaded = True
        raise AssertionError("an impossible result reached the transport")

    monkeypatch.setattr(service, "_download_operation_result", download)
    ref = OperationResultRef(
        object_id="c" * 64,
        byte_count=encoded_byte_count,
    )

    with pytest.raises(RemoteTrainingError, match="required=67108864"):
        await service.receive_operation_result(
            "run", "operation", ref, ForwardBackwardResult
        )

    assert not downloaded
    assert service._result_budget._used == 0
    await service.close()
    await http.aclose()
