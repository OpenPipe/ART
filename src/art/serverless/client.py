from __future__ import annotations

import asyncio
from collections.abc import Mapping
import hashlib
from typing import Any, Callable, Generic, TypeVar, cast
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
from art.utils.lifecycle import process_shutdown_timeout

from .contracts import (
    ApplyCheckpointRetentionRequest,
    CancelOperationRequest,
    CheckpointPage,
    CheckpointView,
    CloseRunRequest,
    CreateTrainingRunRequest,
    DeleteCheckpointResult,
    EventPage,
    OperationResultRef,
    OperationView,
    RemoteForwardRequest,
    RunEvent,
    SetCheckpointTtlRequest,
    TrainingCapabilities,
    TrainingRunView,
)
from .data_plane import (
    EncodedTrainingBatch,
    decode_operation_result,
    encode_forward_submission,
    prepare_training_batch,
)

ResultT = TypeVar("ResultT", bound=OperationResult)
ResponseT = TypeVar("ResponseT", bound=Contract)
_CREATE_RUN_RESOLVE_STATUSES = frozenset({409, 429, 500, 502, 503, 504})
_MAX_RETAINED_COMPLETED_OPERATIONS = 1024


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
        control_http_client: httpx.AsyncClient | None = None,
        transfer_http_client: httpx.AsyncClient | None = None,
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
        owns_clients = control_http_client is None
        if owns_clients != (transfer_http_client is None):
            raise ValueError("remote training HTTP clients must be provided together")
        if control_http_client is None:

            def client(max_connections: int) -> httpx.AsyncClient:
                return httpx.AsyncClient(
                    base_url=str(url).rstrip("/") + "/",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=httpx.Timeout(
                        request_timeout_s, connect=min(request_timeout_s, 10)
                    ),
                    limits=httpx.Limits(
                        max_connections=max_connections,
                        max_keepalive_connections=16,
                    ),
                )

            control_http_client, transfer_http_client = client(32), client(64)
        assert transfer_http_client is not None
        self._control_client = control_http_client
        self._transfer_client = transfer_http_client
        self._owns_clients = owns_clients
        self._max_retries = max_retries

    async def create_run(self, request: CreateTrainingRunRequest) -> TrainingRunView:
        try:
            return await self._request(
                "POST",
                "training/runs",
                TrainingRunView,
                body=request,
                max_retries=0,
            )
        except (httpx.TransportError, RemoteTrainingHttpError) as error:
            if (
                isinstance(error, RemoteTrainingHttpError)
                and error.status_code not in _CREATE_RUN_RESOLVE_STATUSES
            ):
                raise
            try:
                return await self.resolve_run(request)
            except BaseException as resolve_error:
                resolve_error.add_note(
                    f"create_run failed before run-name resolve: {error!r}"
                )
                raise

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

    async def submit_forward(
        self,
        kind: OperationKind,
        request: RemoteForwardRequest,
        batch: EncodedTrainingBatch,
    ) -> OperationRef:
        if kind not in {"forward", "forward_backward"}:
            raise ValueError("forward submission requires a forward operation")
        payload = await asyncio.to_thread(encode_forward_submission, request, batch)
        response = await self._send(
            "POST",
            f"training/runs/{request.run_id}/{kind}",
            content=payload,
            headers={"Content-Type": "application/vnd.art.forward+msgpack"},
            transfer=True,
        )
        return OperationRef.model_validate(response.json())

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
            transfer=True,
        )
        payload = response.content
        if len(payload) != ref.byte_count:
            raise RemoteTrainingError("remote operation result byte count changed")
        return payload

    async def get_events(self, run_id: str, *, after: int) -> EventPage:
        return await self._request(
            "GET",
            f"training/runs/{run_id}/events?after={after}&limit={_EVENT_PAGE_LIMIT}",
            EventPage,
        )

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
        max_retries: int | None = None,
    ) -> ResponseT:
        content = None if body is None else body.model_dump_json()
        response = await self._send(
            method,
            path,
            content=content,
            headers=None if body is None else {"Content-Type": "application/json"},
            max_retries=max_retries,
        )
        return response_type.model_validate(response.json())

    async def _send(
        self,
        method: str,
        path: str,
        *,
        content: str | bytes | None,
        headers: dict[str, str] | None,
        transfer: bool = False,
        max_retries: int | None = None,
    ) -> httpx.Response:
        client = self._transfer_client if transfer else self._control_client
        response: httpx.Response | None = None
        retries = self._max_retries if max_retries is None else max_retries
        for attempt in range(retries + 1):
            try:
                response = await client.request(
                    method,
                    path,
                    content=content,
                    headers=headers,
                )
            except httpx.TransportError:
                if attempt == retries:
                    raise
            else:
                if response.status_code not in {429, 502, 503, 504}:
                    break
                if attempt == retries:
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
        if self._owns_clients:
            await asyncio.gather(
                self._control_client.aclose(), self._transfer_client.aclose()
            )


_EVENT_PAGE_LIMIT = 1000
_TERMINAL_EVENTS = frozenset(
    {"operation_succeeded", "operation_failed", "operation_cancelled"}
)


class _RunEventObserver:
    def __init__(
        self,
        service: RemoteTrainingServiceClient,
        run_id: str,
        poll_interval_s: float,
    ) -> None:
        self._service = service
        self._run_id = run_id
        self._poll_interval_s = poll_interval_s
        self._cursor = 0
        self._terminal: dict[str, RunEvent] = {}
        self._condition = asyncio.Condition()
        self._task: asyncio.Task[None] | None = None
        self._error: BaseException | None = None
        self._closed = False

    async def wait(self, operation_id: str) -> RunEvent:
        async with self._condition:
            if self._task is None:
                self._task = asyncio.create_task(
                    self._observe(), name=f"remote-training-events-{self._run_id}"
                )
            await self._condition.wait_for(
                lambda: (
                    operation_id in self._terminal
                    or self._error is not None
                    or self._closed
                )
            )
            if self._error is not None:
                raise self._error
            if self._closed:
                raise RemoteTrainingError("remote training event observer is closed")
            return self._terminal.pop(operation_id)

    async def close(self, *, timeout_s: float | None = None) -> None:
        async with self._condition:
            if self._closed:
                return
            self._closed = True
            task, self._task = self._task, None
            self._condition.notify_all()
        if task is not None:
            task.cancel()
            if timeout_s is None:
                await asyncio.gather(task, return_exceptions=True)
            else:
                _, pending = await asyncio.wait((task,), timeout=timeout_s)
                if pending:
                    raise TimeoutError(
                        f"event observer for run {self._run_id} did not stop"
                    )

    async def _observe(self) -> None:
        try:
            while True:
                page = await self._service.get_events(self._run_id, after=self._cursor)
                async with self._condition:
                    if page.next_cursor < self._cursor:
                        raise RemoteTrainingError("remote event cursor moved backwards")
                    self._cursor = page.next_cursor
                    for event in page.events:
                        if event.event in _TERMINAL_EVENTS:
                            if event.operation_id is None:
                                raise RemoteTrainingError(
                                    "terminal operation event has no operation_id"
                                )
                            self._terminal[event.operation_id] = event
                    self._condition.notify_all()
                if len(page.events) < _EVENT_PAGE_LIMIT:
                    await asyncio.sleep(self._poll_interval_s)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            async with self._condition:
                self._error = error
                self._condition.notify_all()


class RemoteTrainingOperation(Generic[ResultT]):
    def __init__(
        self,
        ref: OperationRef,
        service: RemoteTrainingServiceClient,
        events: _RunEventObserver,
        result_type: type[ResultT],
        on_completion: Callable[[], None] | None = None,
    ) -> None:
        self._ref = ref
        self._service = service
        self._events = events
        self._result_type = result_type
        self._completion: asyncio.Task[ResultT] | None = None
        self._cancel_request_id = uuid.uuid4().hex
        self._on_completion = on_completion

    @property
    def ref(self) -> OperationRef:
        return self._ref

    async def result(self) -> ResultT:
        if self._completion is None:
            self._completion = asyncio.create_task(
                self._wait(), name=f"remote-training-result-{self._ref.operation_id}"
            )
            on_completion = self._on_completion
            if on_completion is not None:
                self._completion.add_done_callback(lambda _: on_completion())
        return await asyncio.shield(self._completion)

    async def _wait(self) -> ResultT:
        event = await self._events.wait(self._ref.operation_id)
        if event.event == "operation_failed":
            raise RemoteTrainingOperationError(self._ref.operation_id, event.payload)
        if event.event == "operation_cancelled":
            raise RemoteTrainingOperationCancelled(
                f"remote training operation {self._ref.operation_id} was cancelled"
            )
        if event.event != "operation_succeeded":
            raise RemoteTrainingError(f"unexpected terminal event {event.event!r}")
        if self._ref.kind in {"forward", "forward_backward"}:
            remote = OperationResultRef.model_validate(event.payload)
            payload = await self._service.get_operation_result(
                self._ref.operation_id, remote
            )
            result = decode_operation_result(remote, payload, self._result_type)
        else:
            result = self._result_type.model_validate(event.payload)
        if result.operation_id != self._ref.operation_id:
            raise RemoteTrainingError("operation result identity changed")
        return result

    async def cancel(self) -> None:
        try:
            await self._service.cancel_operation(
                self._ref.operation_id, self._cancel_request_id
            )
        except RemoteTrainingHttpError as error:
            if error.status_code != 409 or (
                await self._service.get_operation(self._ref.operation_id)
            ).status not in {"running", "succeeded"}:
                raise
            await self.result()


class RemoteTrainingClient:
    """Run-scoped implementation of ART's canonical training command API."""

    def __init__(
        self,
        service: RemoteTrainingServiceClient,
        run: TrainingRunView,
        *,
        poll_interval_s: float = 0.1,
        close_timeout_s: float = process_shutdown_timeout(2),
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
        self._events = _RunEventObserver(service, run.run_id, poll_interval_s)
        self._close_timeout_s = close_timeout_s
        self._lock = asyncio.Lock()
        self._operations: dict[str, tuple[str, RemoteTrainingOperation[Any]]] = {}
        self._close_request_id = uuid.uuid4().hex
        self._closed = False
        self._close_accepted = False

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
        encoded_batch: EncodedTrainingBatch | None = None,
    ) -> TrainingOperation[ResultT]:
        wire_request: RunCommand = request
        prepared_batch: EncodedTrainingBatch | None = None
        if isinstance(request, (ForwardRequest, ForwardBackwardRequest)):
            if encoded_batch is not None:
                if request.batch is not encoded_batch.batch:
                    raise ValueError("prepared batch does not belong to the request")
                prepared_batch = encoded_batch
            else:
                prepared_batch = await asyncio.to_thread(
                    prepare_training_batch,
                    request.batch,
                    identity=f"{request.run_id}:{request.request_id}",
                )
            wire_request = RemoteForwardRequest.from_command(
                request, prepared_batch.remote
            )
        elif encoded_batch is not None:
            raise TypeError("only forward commands accept a prepared batch")
        return await self._admit(
            request,
            wire_request,
            kind=kind,
            result_type=result_type,
            prepared_batch=prepared_batch,
        )

    async def _admit(
        self,
        request: RunCommand,
        wire_request: RunCommand,
        *,
        kind: OperationKind,
        result_type: type[ResultT],
        prepared_batch: EncodedTrainingBatch | None = None,
    ) -> TrainingOperation[ResultT]:
        fingerprint = hashlib.sha256(
            wire_request.model_dump_json().encode()
        ).hexdigest()
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
            ref = (
                await self._service.submit_forward(
                    kind, cast(RemoteForwardRequest, wire_request), prepared_batch
                )
                if prepared_batch is not None
                else await self._service.submit(kind, wire_request)
            )
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
                self._events,
                result_type,
                on_completion=self._bound_operation_cache,
            )
            self._operations[request.request_id] = (fingerprint, operation)
            self._bound_operation_cache()
            self._next_sequence_id += 1
            if ref.reserved_output_learner_version is not None:
                self._projected_learner_version = ref.reserved_output_learner_version
            return operation

    def _bound_operation_cache(self) -> None:
        overflow = len(self._operations) - _MAX_RETAINED_COMPLETED_OPERATIONS
        if overflow <= 0:
            return
        for request_id, (_fingerprint, operation) in tuple(self._operations.items()):
            if overflow <= 0:
                return
            completion = operation._completion
            if completion is not None and completion.done():
                self._operations.pop(request_id, None)
                overflow -= 1

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
            if self._close_accepted:
                return
            self._closed = True
            self._run = await self._service.close_run(
                self.run_id, self._close_request_id
            )
            self._close_accepted = True

    async def wait_closed(self, *, timeout_s: float | None = None) -> None:
        if not self._close_accepted:
            raise RuntimeError("remote training run closure has not been accepted")
        async with asyncio.timeout(timeout_s or self._close_timeout_s):
            while True:
                run = await self._service.get_run(self.run_id)
                if run.status == "closed":
                    self._run = run
                    return
                if run.status == "failed":
                    raise RemoteTrainingError("remote training run failed during close")
                await asyncio.sleep(self._poll_interval_s)

    async def abort_result_waiters(self) -> None:
        failures: list[BaseException] = []
        tasks = tuple(
            operation._completion
            for _, operation in self._operations.values()
            if operation._completion is not None and not operation._completion.done()
        )
        for task in tasks:
            task.cancel()
        cleanup_timeout = self._close_timeout_s * 0.2
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=cleanup_timeout)
            for task in done:
                try:
                    task.exception()
                except asyncio.CancelledError:
                    pass
            if pending:
                failures.append(
                    TimeoutError(f"{len(pending)} remote result waiters did not stop")
                )
        try:
            await self._events.close(timeout_s=cleanup_timeout)
        except BaseException as error:
            failures.append(error)
        if failures:
            raise BaseExceptionGroup(
                "remote training result waiter cleanup failed", failures
            )

    async def close_event_observer(self, *, timeout_s: float | None = None) -> None:
        await self._events.close(timeout_s=timeout_s)

    async def shutdown(self) -> None:
        """Request run closure, await settlement, and release client resources."""
        failures: list[BaseException] = []
        drain_timeout = self._close_timeout_s * 0.8
        try:
            async with asyncio.timeout(drain_timeout):
                await self.close()
                await self.wait_closed(timeout_s=drain_timeout)
        except BaseException as error:
            failures.append(error)
        try:
            await self.close_event_observer(
                timeout_s=self._close_timeout_s - drain_timeout
            )
        except BaseException as error:
            failures.append(error)
        if failures:
            raise BaseExceptionGroup("remote training client shutdown failed", failures)
