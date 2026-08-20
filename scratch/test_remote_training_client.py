from __future__ import annotations

from array import array
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from art import Trajectory, TrajectoryGroup
from art.distributed.trajectory_store import TrajectoryGroupBundle
from art.pipeline_tuner.config import PackedGroupShape, PackingLeafShape
from art.serverless.client import (
    RemoteTrainingClient,
    RemoteTrainingError,
    RemoteTrainingServiceClient,
)
from art.serverless.contracts import (
    MAX_OPERATION_RESULT_BYTES,
    AdapterSpec,
    CreateTrainingRunRequest,
    OperationResultRef,
    TrainingRunSpec,
    remote_request_fingerprint,
)
from art.serverless.data_plane import (
    FORWARD_SUBMISSION_PREFIX_BYTES,
    EncodedRouteObject,
    EncodedTrainingObject,
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
    OptimStepRequest,
    OptimStepResult,
    PackingOutcome,
    RlTrajectoryBatch,
    TokenizedTrainingBatch,
)
from art.training.tokenized import TokenizedDatum


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
            if self.fail_submit_once:
                self.fail_submit_once = False
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
            self.operations[ref["operation_id"]] = self._operation(
                ref, result_ref, now, submission.request
            )
            self.events.append(self._event(ref, result_ref, now))
            return httpx.Response(200, json=ref)
        if path == "/v1/training/runs/run/optim_step":
            ref = self._ref(body, "optim_step", 1)
            result = OptimStepResult(
                operation_id=ref["operation_id"],
                contributing_forward_backward_operation_ids=(
                    _operation_id("run", "forward"),
                ),
            )
            self.operations[ref["operation_id"]] = self._operation(
                ref, result, now, OptimStepRequest.model_validate(body)
            )
            self.events.append(self._event(ref, result, now))
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
        if path.startswith("/v1/training/operations/"):
            self.operation_status_gets += 1
            return httpx.Response(200, json=self.operations[path.rsplit("/", 1)[1]])
        if path == "/v1/training/runs/run:close":
            self.run["status"] = "closing"
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
    def _operation(ref, result, now, request):
        return {
            "ref": ref,
            "request_id": request.request_id,
            "request_fingerprint": remote_request_fingerprint(request),
            "status": "succeeded",
            "result": result.model_dump(mode="json"),
            "error": None,
            "event_cursor": ref["sequence_id"] + 1,
            "created_at": now,
            "updated_at": now,
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
    received = await service.get_operation_result(
        "operation",
        OperationResultRef(
            object_id="0" * 64,
            byte_count=len(payload),
        ),
    )
    assert isinstance(received, bytearray)
    assert received == payload
    assert stream.reads > 1
    await http.aclose()


@pytest.mark.asyncio
async def test_operation_result_receive_budget_serializes_allocations():
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
        max_result_bytes_in_flight=len(payload),
    )
    ref = OperationResultRef(object_id="0" * 64, byte_count=len(payload))
    first = asyncio.create_task(service.get_operation_result("first", ref))
    await first_started.wait()
    second = asyncio.create_task(service.get_operation_result("second", ref))
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
        await service.get_operation_result(
            "operation", OperationResultRef(object_id="0" * 64, byte_count=4)
        )
    await http.aclose()


@pytest.mark.asyncio
async def test_operation_result_receive_budget_close_wakes_waiters():
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
        max_result_bytes_in_flight=len(payload),
    )
    ref = OperationResultRef(object_id="0" * 64, byte_count=len(payload))
    first = asyncio.create_task(service.get_operation_result("first", ref))
    await first_started.wait()
    waiting = asyncio.create_task(service.get_operation_result("waiting", ref))
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
async def test_remote_client_retries_and_preserves_command_order():
    fake = FakeService()
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
        max_retries=1,
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
        poll_interval_s=0.001,
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
    assert await client.forward_backward(request) is forward
    assert fake.submit_bodies[0] == fake.submit_bodies[1]
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
    assert client.next_sequence_id == 2
    assert client.projected_learner_version == 1
    forward_result, optimizer_result = await asyncio.gather(
        forward.result(), optimizer.result()
    )
    forward_operation_id = _operation_id("run", "forward")
    assert forward_result.operation_id == forward_operation_id
    assert optimizer_result.operation_id == _operation_id("run", "optimizer")
    assert optimizer_result.contributing_forward_backward_operation_ids == (
        forward_operation_id,
    )
    assert fake.operation_status_gets == 0
    assert 1 <= fake.nonempty_event_pages <= 2
    assert sum(path.endswith("/result") for path in paths["control"]) == 1
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
