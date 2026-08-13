from __future__ import annotations

import asyncio
from collections.abc import Mapping
import hashlib
from typing import Any, Generic, TypeVar, cast
import uuid

import httpx

from art.training.client import TrainingOperation
from art.training.contracts import (
    Contract,
    ForwardBackwardRequest,
    ForwardBackwardResult,
    ForwardRequest,
    ForwardResult,
    LoadStateRequest,
    LoadStateResult,
    OperationKind,
    OperationRef,
    OperationResult,
    OptimStepRequest,
    OptimStepResult,
    RunCommand,
    SamplerWeightsResult,
    SaveStateRequest,
    SaveStateResult,
    SaveWeightsForSamplerRequest,
)

from .contracts import (
    ApplyCheckpointRetentionRequest,
    CancelOperationRequest,
    CheckpointPage,
    CheckpointView,
    CloseRunRequest,
    CreateTrainingRunRequest,
    DeleteCheckpointResult,
    OperationResultRef,
    OperationView,
    RemoteForwardRequest,
    SetCheckpointTtlRequest,
    TrainingCapabilities,
    TrainingDataRef,
    TrainingRunView,
)
from .data_plane import decode_operation_result, encode_training_batch

ResultT = TypeVar("ResultT", bound=OperationResult)
ResponseT = TypeVar("ResponseT", bound=Contract)


class RemoteTrainingError(RuntimeError):
    pass


class RemoteTrainingHttpError(RemoteTrainingError):
    def __init__(self, status_code: int, detail: object) -> None:
        super().__init__(f"remote training HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class RemoteTrainingOperationError(RemoteTrainingError):
    def __init__(self, operation_id: str, error: Mapping[str, Any]) -> None:
        super().__init__(
            f"remote training operation {operation_id} failed: {dict(error)}"
        )
        self.operation_id = operation_id
        self.error = dict(error)


class RemoteTrainingOperationCancelled(RemoteTrainingError):
    pass


class RemoteTrainingServiceClient:
    """Typed HTTP transport for the Remote Training operation API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        http_client: httpx.AsyncClient | None = None,
        request_timeout_s: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError("remote training api_key must not be empty")
        if request_timeout_s <= 0 or max_retries < 0:
            raise ValueError("remote training timeout and retry count are invalid")
        url = httpx.URL(base_url)
        if url.path.rstrip("/").split("/")[-1] != "v1":
            raise ValueError("remote training base_url must end in /v1")
        self._client = http_client or httpx.AsyncClient(
            base_url=str(url).rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(
                request_timeout_s, connect=min(request_timeout_s, 10)
            ),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
        )
        self._owns_client = http_client is None
        self._max_retries = max_retries

    async def create_run(self, request: CreateTrainingRunRequest) -> TrainingRunView:
        return await self._request(
            "POST", "training/runs", TrainingRunView, body=request
        )

    async def resolve_run(self, request: CreateTrainingRunRequest) -> TrainingRunView:
        return await self._request(
            "POST", "training/runs:resolve", TrainingRunView, body=request
        )

    async def get_run(self, run_id: str) -> TrainingRunView:
        return await self._request("GET", f"training/runs/{run_id}", TrainingRunView)

    async def submit(self, kind: OperationKind, request: RunCommand) -> OperationRef:
        endpoint = {
            "forward": "forward",
            "forward_backward": "forward_backward",
            "optim_step": "optim_step",
            "save_sampler": "save_weights_for_sampler",
            "save_state": "save_state",
            "load_state": (
                "load_state_with_optimizer"
                if isinstance(request, LoadStateRequest) and request.restore_optimizer
                else "load_state"
            ),
        }[kind]
        return await self._request(
            "POST",
            f"training/runs/{request.run_id}/{endpoint}",
            OperationRef,
            body=request,
        )

    async def put_training_data(
        self, run_id: str, ref: TrainingDataRef, payload: bytes
    ) -> None:
        response = await self._send(
            "PUT",
            f"training/runs/{run_id}/data/{ref.object_id}",
            content=payload,
            headers={
                "Content-Type": "application/vnd.art.training-batch+msgpack",
                "X-Art-Training-Data-Format": ref.format,
                "X-Art-Training-Batch-Kind": ref.batch_kind,
                "X-Art-Training-Byte-Count": str(ref.byte_count),
            },
        )
        received = TrainingDataRef.model_validate(response.json())
        if received != ref:
            raise RemoteTrainingError("remote training data identity changed")

    async def get_operation(self, operation_id: str) -> OperationView:
        return await self._request(
            "GET", f"training/operations/{operation_id}", OperationView
        )

    async def get_operation_result(
        self, operation_id: str, ref: OperationResultRef
    ) -> bytes:
        response = await self._send(
            "GET",
            f"training/operations/{operation_id}/result",
            content=None,
            headers=None,
        )
        payload = response.content
        if len(payload) != ref.byte_count:
            raise RemoteTrainingError("remote operation result byte count changed")
        return payload

    async def cancel_operation(
        self, operation_id: str, request_id: str
    ) -> OperationView:
        return await self._request(
            "POST",
            f"training/operations/{operation_id}:cancel",
            OperationView,
            body=CancelOperationRequest(request_id=request_id),
        )

    async def close_run(self, run_id: str, request_id: str) -> TrainingRunView:
        return await self._request(
            "POST",
            f"training/runs/{run_id}:close",
            TrainingRunView,
            body=CloseRunRequest(request_id=request_id),
        )

    async def capabilities(self) -> TrainingCapabilities:
        return await self._request("GET", "training/capabilities", TrainingCapabilities)

    async def list_checkpoints(self, run_id: str) -> CheckpointPage:
        return await self._request(
            "GET", f"training/runs/{run_id}/checkpoints", CheckpointPage
        )

    async def apply_checkpoint_retention(
        self, run_id: str, plan: ApplyCheckpointRetentionRequest
    ) -> CheckpointPage:
        return await self._request(
            "POST",
            f"training/runs/{run_id}/checkpoints:apply_retention",
            CheckpointPage,
            body=plan,
        )

    async def set_checkpoint_ttl(
        self, run_id: str, checkpoint_id: str, ttl_seconds: int | None
    ) -> CheckpointView:
        return await self._request(
            "POST",
            f"training/runs/{run_id}/checkpoints/{checkpoint_id}:set_ttl",
            CheckpointView,
            body=SetCheckpointTtlRequest(ttl_seconds=ttl_seconds),
        )

    async def archive_checkpoint(
        self, run_id: str, checkpoint_id: str
    ) -> CheckpointView:
        return await self._request(
            "POST",
            f"training/runs/{run_id}/checkpoints/{checkpoint_id}:archive",
            CheckpointView,
        )

    async def evict_checkpoint(self, run_id: str, checkpoint_id: str) -> CheckpointView:
        return await self._request(
            "POST",
            f"training/runs/{run_id}/checkpoints/{checkpoint_id}:evict_local",
            CheckpointView,
        )

    async def delete_checkpoint(
        self, run_id: str, checkpoint_id: str
    ) -> DeleteCheckpointResult:
        return await self._request(
            "DELETE",
            f"training/runs/{run_id}/checkpoints/{checkpoint_id}",
            DeleteCheckpointResult,
        )

    async def _request(
        self,
        method: str,
        path: str,
        response_type: type[ResponseT],
        *,
        body: Contract | None = None,
    ) -> ResponseT:
        content = None if body is None else body.model_dump_json()
        response = await self._send(
            method,
            path,
            content=content,
            headers=None if body is None else {"Content-Type": "application/json"},
        )
        return response_type.model_validate(response.json())

    async def _send(
        self,
        method: str,
        path: str,
        *,
        content: str | bytes | None,
        headers: dict[str, str] | None,
    ) -> httpx.Response:
        response: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.request(
                    method,
                    path,
                    content=content,
                    headers=headers,
                )
            except httpx.TransportError:
                if attempt == self._max_retries:
                    raise
            else:
                if response.status_code not in {429, 502, 503, 504}:
                    break
                if attempt == self._max_retries:
                    break
            await asyncio.sleep(min(0.1 * 2**attempt, 1.0))
        if response is None:
            raise RuntimeError("remote training request produced no response")
        if response.is_error:
            try:
                value = response.json()
            except ValueError:
                value = response.text
            detail = value.get("detail", value) if isinstance(value, dict) else value
            raise RemoteTrainingHttpError(response.status_code, detail)
        return response

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class RemoteTrainingOperation(Generic[ResultT]):
    def __init__(
        self,
        ref: OperationRef,
        service: RemoteTrainingServiceClient,
        result_type: type[ResultT],
        *,
        poll_interval_s: float,
    ) -> None:
        self._ref = ref
        self._service = service
        self._result_type = result_type
        self._poll_interval_s = poll_interval_s
        self._completion: asyncio.Task[ResultT] | None = None
        self._cancel_request_id = uuid.uuid4().hex

    @property
    def ref(self) -> OperationRef:
        return self._ref

    async def result(self) -> ResultT:
        if self._completion is None:
            self._completion = asyncio.create_task(
                self._wait(), name=f"remote-training-result-{self._ref.operation_id}"
            )
        return await asyncio.shield(self._completion)

    async def _wait(self) -> ResultT:
        while True:
            operation = await self._service.get_operation(self._ref.operation_id)
            if operation.ref != self._ref:
                raise RemoteTrainingError("remote operation identity changed")
            if operation.status == "succeeded":
                if operation.result is None:
                    raise RemoteTrainingError("successful operation has no result")
                remote = operation.result
                if self._ref.kind in {"forward", "forward_backward"}:
                    if not isinstance(remote, OperationResultRef):
                        raise RemoteTrainingError(
                            "remote forward result has no binary sidecar"
                        )
                    payload = await self._service.get_operation_result(
                        self._ref.operation_id, remote
                    )
                    result = decode_operation_result(remote, payload, self._result_type)
                else:
                    if isinstance(remote, OperationResultRef):
                        raise RemoteTrainingError(
                            "remote control result unexpectedly uses a sidecar"
                        )
                    result = self._result_type.model_validate(remote)
                if result.operation_id != self._ref.operation_id:
                    raise RemoteTrainingError("operation result identity changed")
                return result
            if operation.status == "failed":
                raise RemoteTrainingOperationError(
                    self._ref.operation_id, operation.error or {}
                )
            if operation.status == "cancelled":
                raise RemoteTrainingOperationCancelled(
                    f"remote training operation {self._ref.operation_id} was cancelled"
                )
            await asyncio.sleep(self._poll_interval_s)

    async def cancel(self) -> None:
        await self._service.cancel_operation(
            self._ref.operation_id, self._cancel_request_id
        )


class RemoteTrainingClient:
    """Run-scoped implementation of ART's canonical training command API."""

    def __init__(
        self,
        service: RemoteTrainingServiceClient,
        run: TrainingRunView,
        *,
        poll_interval_s: float = 0.1,
        close_timeout_s: float = 20.0,
    ) -> None:
        if run.status != "open":
            raise ValueError(f"remote training run is {run.status}, not open")
        if poll_interval_s <= 0 or close_timeout_s <= 0:
            raise ValueError(
                "remote training polling and close timeouts must be positive"
            )
        self._service = service
        self._run = run
        self._next_sequence_id = run.next_sequence_id
        self._projected_learner_version = run.projected_learner_version
        self._poll_interval_s = poll_interval_s
        self._close_timeout_s = close_timeout_s
        self._lock = asyncio.Lock()
        self._operations: dict[str, tuple[str, RemoteTrainingOperation[Any]]] = {}
        self._close_request_id = uuid.uuid4().hex
        self._closed = False

    @classmethod
    async def create(
        cls,
        service: RemoteTrainingServiceClient,
        request: CreateTrainingRunRequest,
        **kwargs: Any,
    ) -> "RemoteTrainingClient":
        return cls(service, await service.create_run(request), **kwargs)

    @property
    def run_id(self) -> str:
        return self._run.run_id

    @property
    def next_sequence_id(self) -> int:
        return self._next_sequence_id

    @property
    def projected_learner_version(self) -> int:
        return self._projected_learner_version

    async def _submit(
        self,
        request: RunCommand,
        *,
        kind: OperationKind,
        result_type: type[ResultT],
    ) -> TrainingOperation[ResultT]:
        fingerprint = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        async with self._lock:
            if self._closed:
                raise RuntimeError("remote training client is closed")
            if request.run_id != self.run_id:
                raise ValueError("command run_id does not match the remote client")
            if existing := self._operations.get(request.request_id):
                if existing[0] != fingerprint:
                    raise ValueError("request_id was reused with different content")
                return cast(TrainingOperation[ResultT], existing[1])
            if request.sequence_id != self._next_sequence_id:
                raise ValueError(
                    f"expected sequence {self._next_sequence_id}, "
                    f"got {request.sequence_id}"
                )
            wire_request: RunCommand = request
            if isinstance(request, (ForwardRequest, ForwardBackwardRequest)):
                batch_ref, payload = await asyncio.to_thread(
                    encode_training_batch, request.batch
                )
                await self._service.put_training_data(
                    request.run_id, batch_ref, payload
                )
                wire_request = RemoteForwardRequest.from_command(request, batch_ref)
            ref = await self._service.submit(kind, wire_request)
            if (
                ref.run_id != self.run_id
                or ref.sequence_id != request.sequence_id
                or ref.kind != kind
                or ref.learner_parent_version != self._projected_learner_version
            ):
                raise RemoteTrainingError("server admitted a divergent command")
            operation = RemoteTrainingOperation(
                ref,
                self._service,
                result_type,
                poll_interval_s=self._poll_interval_s,
            )
            self._operations[request.request_id] = (fingerprint, operation)
            self._next_sequence_id += 1
            if ref.reserved_output_learner_version is not None:
                self._projected_learner_version = ref.reserved_output_learner_version
            return operation

    async def forward(
        self, request: ForwardRequest
    ) -> TrainingOperation[ForwardResult]:
        return await self._submit(request, kind="forward", result_type=ForwardResult)

    async def forward_backward(
        self, request: ForwardBackwardRequest
    ) -> TrainingOperation[ForwardBackwardResult]:
        return await self._submit(
            request,
            kind="forward_backward",
            result_type=ForwardBackwardResult,
        )

    async def optim_step(
        self, request: OptimStepRequest
    ) -> TrainingOperation[OptimStepResult]:
        return await self._submit(
            request, kind="optim_step", result_type=OptimStepResult
        )

    async def save_weights_for_sampler(
        self, request: SaveWeightsForSamplerRequest
    ) -> TrainingOperation[SamplerWeightsResult]:
        return await self._submit(
            request,
            kind="save_sampler",
            result_type=SamplerWeightsResult,
        )

    async def save_state(
        self, request: SaveStateRequest
    ) -> TrainingOperation[SaveStateResult]:
        return await self._submit(
            request, kind="save_state", result_type=SaveStateResult
        )

    async def load_state(
        self, request: LoadStateRequest
    ) -> TrainingOperation[LoadStateResult]:
        return await self._submit(
            request.model_copy(update={"restore_optimizer": False}),
            kind="load_state",
            result_type=LoadStateResult,
        )

    async def load_state_with_optimizer(
        self, request: LoadStateRequest
    ) -> TrainingOperation[LoadStateResult]:
        return await self._submit(
            request.model_copy(update={"restore_optimizer": True}),
            kind="load_state",
            result_type=LoadStateResult,
        )

    async def close(self) -> None:
        async with self._lock:
            if self._run.status == "closed":
                return
            self._closed = True
        await self._service.close_run(self.run_id, self._close_request_id)
        async with asyncio.timeout(self._close_timeout_s):
            while True:
                run = await self._service.get_run(self.run_id)
                if run.status == "closed":
                    self._run = run
                    return
                if run.status == "failed":
                    raise RemoteTrainingError("remote training run failed during close")
                await asyncio.sleep(self._poll_interval_s)

    async def abort_result_waiters(self) -> None:
        tasks = tuple(
            operation._completion
            for _, operation in self._operations.values()
            if operation._completion is not None and not operation._completion.done()
        )
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
