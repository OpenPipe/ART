from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
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
from art.utils.lifecycle import complete_to_thread, process_shutdown_timeout

from .contracts import (
    DEFAULT_CHECKPOINT_ALIAS_PAGE_LIMIT,
    DEFAULT_CHECKPOINT_PAGE_LIMIT,
    EVENT_LONG_POLL_TIMEOUT_S,
    FORWARD_BACKWARD_PREPARED_EVENT,
    MAX_CHECKPOINT_ALIAS_PAGE_LIMIT,
    MAX_CHECKPOINT_PAGE_LIMIT,
    MAX_EVENT_PAGE_LIMIT,
    MAX_OPERATION_RESULT_BYTES,
    ApplyCheckpointRetentionRequest,
    CancelOperationRequest,
    CheckpointAliasPage,
    CheckpointCursor,
    CheckpointPage,
    CheckpointView,
    CloseRunRequest,
    CreateTrainingRunRequest,
    DeleteCheckpointResult,
    EventPage,
    ForwardBackwardPreparation,
    OperationResultRef,
    OperationView,
    PreparedGradientDisposition,
    RemoteForwardRequest,
    RunEvent,
    SetCheckpointTtlRequest,
    TrainingRunView,
    remote_request_fingerprint,
)
from .data_plane import (
    FORWARD_SUBMISSION_MEDIA_TYPE,
    EncodedForwardSubmission,
    EncodedTrainingBatch,
    decode_operation_result,
    encode_forward_submission,
    prepare_training_batch,
)

ResultT = TypeVar("ResultT", bound=OperationResult)
ResponseT = TypeVar("ResponseT", bound=Contract)
_CREATE_RUN_RESOLVE_STATUSES = frozenset({409, 429, 500, 502, 503, 504})
_TRANSIENT_HTTP_STATUSES = frozenset({429, 502, 503, 504})
_MAX_RETAINED_COMPLETED_OPERATIONS = 1024


class _ByteBudget:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("byte budget must be positive")
        self._capacity = capacity
        self._used = 0
        self._condition = asyncio.Condition()
        self._closed = False

    @asynccontextmanager
    async def reserve(self, byte_count: int):
        if not 0 < byte_count <= self._capacity:
            raise RemoteTrainingError("remote result exceeds the client receive budget")
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._closed or self._used + byte_count <= self._capacity
            )
            if self._closed:
                raise RemoteTrainingError("remote result receive budget is closed")
            self._used += byte_count
        try:
            yield
        finally:
            async with self._condition:
                self._used -= byte_count
                self._condition.notify_all()

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()


class _ResultAcknowledger:
    def __init__(
        self,
        acknowledge: Callable[[str, str], Awaitable[None]],
        *,
        max_pending: int = 256,
    ) -> None:
        self._acknowledge = acknowledge
        self._queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue(max_pending)
        self._worker: asyncio.Task[None] | None = None
        self._failure: BaseException | None = None
        self._closed = False

    async def submit(self, run_id: str, operation_id: str) -> None:
        if self._failure is not None:
            raise RemoteTrainingError("remote result acknowledgement failed") from (
                self._failure
            )
        if self._closed:
            raise RemoteTrainingError("remote result acknowledger is closed")
        if self._worker is None:
            self._worker = asyncio.create_task(
                self._run(), name="remote-training-result-acknowledger"
            )
        await self._queue.put((run_id, operation_id))

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._worker is not None:
                await self._queue.put(None)
                await self._worker
        if self._failure is not None:
            raise RemoteTrainingError("remote result acknowledgement failed") from (
                self._failure
            )

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                await self._acknowledge(*item)
            except BaseException as error:
                self._failure = error
                while not self._queue.empty():
                    self._queue.get_nowait()
                    self._queue.task_done()
                return
            finally:
                self._queue.task_done()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _operation_id(run_id: str, request_id: str) -> str:
    return _sha256(f"{run_id}\0{request_id}".encode())


def _prepare_forward_submission(
    request: ForwardRequest | ForwardBackwardRequest,
    encoded_batch: EncodedTrainingBatch | None,
) -> tuple[RemoteForwardRequest, EncodedForwardSubmission, str]:
    prepared = (
        encoded_batch
        if encoded_batch is not None
        else prepare_training_batch(request.batch)
    )
    if prepared.batch is not request.batch:
        raise ValueError("prepared batch does not belong to the request")
    remote = RemoteForwardRequest.from_command(request, prepared.remote)
    return (
        remote,
        encode_forward_submission(remote, prepared),
        remote_request_fingerprint(remote),
    )


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


def _is_transient_failure(error: BaseException) -> bool:
    return isinstance(
        error,
        (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError),
    ) or (
        isinstance(error, RemoteTrainingHttpError)
        and error.status_code in _TRANSIENT_HTTP_STATUSES
    )


def _retry_delay(attempt: int) -> float:
    return min(0.1 * 2 ** min(attempt, 4), 1.0)


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
        max_result_bytes_in_flight: int = MAX_OPERATION_RESULT_BYTES,
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
        self._result_budget = _ByteBudget(max_result_bytes_in_flight)
        self._result_acknowledger = _ResultAcknowledger(
            self._acknowledge_operation_result
        )
        self._closed = False

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
                isinstance(error, httpx.TransportError)
                and not _is_transient_failure(error)
            ) or (
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
        payload: EncodedForwardSubmission,
    ) -> OperationRef:
        if kind not in {"forward", "forward_backward"}:
            raise ValueError("forward submission requires a forward operation")
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._send(
                    "POST",
                    f"training/runs/{request.run_id}/{kind}",
                    content=payload.stream(),
                    headers={
                        "Content-Type": FORWARD_SUBMISSION_MEDIA_TYPE,
                        "Content-Length": str(payload.byte_count),
                    },
                    transfer=True,
                    max_retries=0,
                )
            except BaseException as error:
                if not _is_transient_failure(error):
                    raise
                try:
                    view = await self.get_operation(
                        request.run_id,
                        _operation_id(request.run_id, request.request_id),
                    )
                except RemoteTrainingHttpError as probe_error:
                    if probe_error.status_code != 404:
                        probe_error.add_note(
                            f"forward submission was ambiguous: {error!r}"
                        )
                        raise
                    if attempt == self._max_retries:
                        raise error
                    await asyncio.sleep(_retry_delay(attempt))
                    continue
                return self._validated_forward_admission(view, kind, request)
            return OperationRef.model_validate(response.json())
        raise AssertionError("forward admission retry loop did not terminate")

    @staticmethod
    def _validated_forward_admission(
        view: OperationView,
        kind: OperationKind,
        request: RemoteForwardRequest,
    ) -> OperationRef:
        if (
            view.ref.operation_id != _operation_id(request.run_id, request.request_id)
            or view.ref.run_id != request.run_id
            or view.ref.sequence_id != request.sequence_id
            or view.ref.kind != kind
            or view.request_id != request.request_id
            or view.request_fingerprint != remote_request_fingerprint(request)
        ):
            raise RemoteTrainingError(
                "ambiguous forward resolved to a different operation"
            )
        return view.ref

    async def get_operation(self, run_id: str, operation_id: str) -> OperationView:
        return await self._request(
            "GET",
            f"training/runs/{run_id}/operations/{operation_id}",
            OperationView,
        )

    async def receive_operation_result(
        self,
        run_id: str,
        operation_id: str,
        ref: OperationResultRef,
        result_type: type[ResultT],
    ) -> ResultT:
        async with self._result_budget.reserve(ref.byte_count):
            payload = await self._receive_operation_result(run_id, operation_id, ref)
            try:
                result, cancelled = await complete_to_thread(
                    lambda: decode_operation_result(ref, payload, result_type)
                )
            finally:
                del payload
            if cancelled is not None:
                raise cancelled
            return result

    async def _receive_operation_result(
        self, run_id: str, operation_id: str, ref: OperationResultRef
    ) -> bytearray:
        failure_count = 0
        while True:
            if self._closed:
                raise RemoteTrainingError("remote training service client is closed")
            try:
                return await self._download_operation_result(run_id, operation_id, ref)
            except BaseException as error:
                if not _is_transient_failure(error):
                    raise
            if self._closed:
                raise RemoteTrainingError("remote training service client is closed")
            await asyncio.sleep(_retry_delay(failure_count))
            failure_count = min(failure_count + 1, 4)

    async def _download_operation_result(
        self, run_id: str, operation_id: str, ref: OperationResultRef
    ) -> bytearray:
        response = await self._send(
            "GET",
            f"training/runs/{run_id}/operations/{operation_id}/result",
            content=None,
            headers=None,
            transfer=True,
            stream=True,
        )
        try:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and content_length != str(ref.byte_count):
                raise RemoteTrainingError(
                    "remote operation result Content-Length changed"
                )
            payload = bytearray(ref.byte_count)
            offset = 0
            async for chunk in response.aiter_bytes():
                end = offset + len(chunk)
                if end > len(payload):
                    raise RemoteTrainingError("remote operation result grew in transit")
                payload[offset:end] = chunk
                offset = end
        finally:
            await response.aclose()
        if offset != ref.byte_count:
            raise RemoteTrainingError("remote operation result byte count changed")
        return payload

    async def acknowledge_operation_result(
        self, run_id: str, operation_id: str
    ) -> None:
        await self._result_acknowledger.submit(run_id, operation_id)

    async def _acknowledge_operation_result(
        self, run_id: str, operation_id: str
    ) -> None:
        await self._send(
            "DELETE",
            f"training/runs/{run_id}/operations/{operation_id}/result",
            content=None,
            headers=None,
        )

    async def get_events(self, run_id: str, *, after: int) -> EventPage:
        return await self._request(
            "GET",
            f"training/runs/{run_id}/events?after={after}&limit={MAX_EVENT_PAGE_LIMIT}",
            EventPage,
            timeout_s=EVENT_LONG_POLL_TIMEOUT_S + 5.0,
        )

    async def cancel_operation(
        self, run_id: str, operation_id: str, request_id: str
    ) -> OperationView:
        return await self._request(
            "POST",
            f"training/runs/{run_id}/operations/{operation_id}:cancel",
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

    async def list_checkpoint_page(
        self,
        run_id: str,
        *,
        cursor: CheckpointCursor | None = None,
        limit: int = DEFAULT_CHECKPOINT_PAGE_LIMIT,
    ) -> CheckpointPage:
        if not 1 <= limit <= MAX_CHECKPOINT_PAGE_LIMIT:
            raise ValueError("checkpoint page limit is invalid")
        query = httpx.QueryParams(
            {"limit": str(limit), **({"cursor": cursor} if cursor is not None else {})}
        )
        return await self._request(
            "GET", f"training/runs/{run_id}/checkpoints?{query}", CheckpointPage
        )

    async def iter_checkpoint_pages(self, run_id: str) -> AsyncIterator[CheckpointPage]:
        cursor: CheckpointCursor | None = None
        current_checkpoint_id: str | None = None
        first_page = True
        while True:
            page = await self.list_checkpoint_page(run_id, cursor=cursor)
            if first_page:
                current_checkpoint_id = page.current_checkpoint_id
                first_page = False
            elif page.current_checkpoint_id != current_checkpoint_id:
                raise RemoteTrainingError(
                    "current checkpoint changed while listing checkpoints"
                )
            yield page
            next_cursor = page.next_cursor
            if next_cursor is None:
                return
            if next_cursor == cursor:
                raise RemoteTrainingError("remote checkpoint cursor did not advance")
            cursor = next_cursor

    async def list_checkpoint_alias_page(
        self,
        run_id: str,
        checkpoint_id: str,
        *,
        cursor: CheckpointCursor | None = None,
        limit: int = DEFAULT_CHECKPOINT_ALIAS_PAGE_LIMIT,
    ) -> CheckpointAliasPage:
        if not 1 <= limit <= MAX_CHECKPOINT_ALIAS_PAGE_LIMIT:
            raise ValueError("checkpoint alias page limit is invalid")
        query = httpx.QueryParams(
            {"limit": str(limit), **({"cursor": cursor} if cursor is not None else {})}
        )
        return await self._request(
            "GET",
            f"training/runs/{run_id}/checkpoints/{checkpoint_id}/aliases?{query}",
            CheckpointAliasPage,
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
        timeout_s: float | None = None,
    ) -> ResponseT:
        content = None if body is None else body.model_dump_json()
        response = await self._send(
            method,
            path,
            content=content,
            headers=None if body is None else {"Content-Type": "application/json"},
            max_retries=max_retries,
            timeout_s=timeout_s,
        )
        return response_type.model_validate(response.json())

    async def _send(
        self,
        method: str,
        path: str,
        *,
        content: str | bytes | AsyncIterable[bytes] | None,
        headers: dict[str, str] | None,
        transfer: bool = False,
        stream: bool = False,
        max_retries: int | None = None,
        timeout_s: float | None = None,
    ) -> httpx.Response:
        client = self._transfer_client if transfer else self._control_client
        response: httpx.Response | None = None
        retries = self._max_retries if max_retries is None else max_retries
        for attempt in range(retries + 1):
            try:
                response = await client.send(
                    client.build_request(
                        method,
                        path,
                        content=content,
                        headers=headers,
                        timeout=(
                            httpx.USE_CLIENT_DEFAULT if timeout_s is None else timeout_s
                        ),
                    ),
                    stream=stream,
                )
            except httpx.TransportError as error:
                if attempt == retries or not _is_transient_failure(error):
                    raise
            else:
                if response.status_code not in _TRANSIENT_HTTP_STATUSES:
                    break
                if attempt == retries:
                    break
                await response.aclose()
            await asyncio.sleep(_retry_delay(attempt))
        if response is None:
            raise RuntimeError("remote training request produced no response")
        if response.is_error:
            if stream:
                try:
                    await response.aread()
                except httpx.TransportError:
                    value: object = response.reason_phrase
                else:
                    try:
                        value = response.json()
                    except ValueError:
                        value = response.text
            else:
                try:
                    value = response.json()
                except ValueError:
                    value = response.text
            detail = value.get("detail", value) if isinstance(value, dict) else value
            if stream:
                await response.aclose()
            raise RemoteTrainingHttpError(response.status_code, detail)
        return response

    async def close(self) -> None:
        if self._closed:
            return
        failures: list[BaseException] = []
        try:
            await self._result_acknowledger.close()
        except BaseException as error:
            failures.append(error)
        self._closed = True
        await self._result_budget.close()
        if self._owns_clients:
            results = await asyncio.gather(
                self._control_client.aclose(),
                self._transfer_client.aclose(),
                return_exceptions=True,
            )
            failures.extend(
                result for result in results if isinstance(result, BaseException)
            )
        if failures:
            raise BaseExceptionGroup("remote training service close failed", failures)


_TERMINAL_EVENTS = frozenset(
    {"operation_succeeded", "operation_failed", "operation_cancelled"}
)
_RUN_TERMINAL_EVENTS = frozenset({"run_closed", "run_failed_released"})


class _RunEventObserver:
    def __init__(
        self,
        service: RemoteTrainingServiceClient,
        run_id: str,
    ) -> None:
        self._service = service
        self._run_id = run_id
        self._cursor = 0
        self._pending: dict[str, asyncio.Future[RunEvent]] = {}
        self._preparations: dict[str, asyncio.Future[RunEvent]] = {}
        self._run_terminal: asyncio.Future[RunEvent] | None = None
        self._run_terminal_event: RunEvent | None = None
        self._task: asyncio.Task[None] | None = None
        self._error: BaseException | None = None
        self._closed = False

    def reserve(self, operation_id: str) -> asyncio.Future[RunEvent]:
        future = self._pending.get(operation_id)
        if future is not None:
            return future
        future = asyncio.get_running_loop().create_future()
        future.add_done_callback(_consume_event_future)
        if self._error is not None:
            future.set_exception(self._error)
        elif self._closed:
            future.set_exception(
                RemoteTrainingError("remote training event observer is closed")
            )
        else:
            self._pending[operation_id] = future
            if self._task is None:
                self._task = asyncio.create_task(
                    self._observe(), name=f"remote-training-events-{self._run_id}"
                )
        return future

    def reserve_run_terminal(self) -> asyncio.Future[RunEvent]:
        if self._run_terminal is not None:
            return self._run_terminal
        future = asyncio.get_running_loop().create_future()
        future.add_done_callback(_consume_event_future)
        if self._run_terminal_event is not None:
            future.set_result(self._run_terminal_event)
        elif self._error is not None:
            future.set_exception(self._error)
        elif self._closed:
            future.set_exception(
                RemoteTrainingError("remote training event observer is closed")
            )
        else:
            self._run_terminal = future
            if self._task is None:
                self._task = asyncio.create_task(
                    self._observe(), name=f"remote-training-events-{self._run_id}"
                )
        return future

    def reserve_preparation(self, operation_id: str) -> asyncio.Future[RunEvent]:
        future = self._preparations.get(operation_id)
        if future is not None:
            return future
        future = asyncio.get_running_loop().create_future()
        future.add_done_callback(_consume_event_future)
        if self._error is not None:
            future.set_exception(self._error)
        elif self._closed:
            future.set_exception(
                RemoteTrainingError("remote training event observer is closed")
            )
        else:
            self._preparations[operation_id] = future
            if self._task is None:
                self._task = asyncio.create_task(
                    self._observe(), name=f"remote-training-events-{self._run_id}"
                )
        return future

    def claim(self, operation_id: str, future: asyncio.Future[RunEvent]) -> None:
        future.add_done_callback(lambda _: self._release(operation_id, future))

    def claim_preparation(
        self, operation_id: str, future: asyncio.Future[RunEvent]
    ) -> None:
        future.add_done_callback(
            lambda _: self._release_preparation(operation_id, future)
        )

    def abandon(
        self,
        operation_id: str,
        future: asyncio.Future[RunEvent],
        error: BaseException,
    ) -> None:
        self._release(operation_id, future)
        if not future.done():
            future.set_exception(error)

    def abandon_preparation(
        self,
        operation_id: str,
        future: asyncio.Future[RunEvent],
        error: BaseException,
    ) -> None:
        self._release_preparation(operation_id, future)
        if not future.done():
            future.set_exception(error)

    def _release(self, operation_id: str, future: asyncio.Future[RunEvent]) -> None:
        if self._pending.get(operation_id) is future:
            self._pending.pop(operation_id)

    def _release_preparation(
        self, operation_id: str, future: asyncio.Future[RunEvent]
    ) -> None:
        if self._preparations.get(operation_id) is future:
            self._preparations.pop(operation_id)

    async def close(self, *, timeout_s: float | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        task, self._task = self._task, None
        pending, self._pending = tuple(self._pending.values()), {}
        preparations, self._preparations = tuple(self._preparations.values()), {}
        if self._run_terminal is not None and not self._run_terminal.done():
            self._run_terminal.set_exception(
                RemoteTrainingError("remote training event observer is closed")
            )
        for future in (*pending, *preparations):
            if not future.done():
                future.set_exception(
                    RemoteTrainingError("remote training event observer is closed")
                )
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
            failure_count = 0
            while True:
                try:
                    page = await self._service.get_events(
                        self._run_id, after=self._cursor
                    )
                except BaseException as error:
                    if not _is_transient_failure(error):
                        raise
                    await asyncio.sleep(_retry_delay(failure_count))
                    failure_count = min(failure_count + 1, 4)
                    continue
                failure_count = 0
                page_cursor = self._cursor
                for event in page.events:
                    if event.run_id != self._run_id:
                        raise RemoteTrainingError("remote event run identity changed")
                    if event.cursor <= page_cursor:
                        raise RemoteTrainingError(
                            "remote event cursor is not increasing"
                        )
                    page_cursor = event.cursor
                if page.next_cursor != page_cursor:
                    raise RemoteTrainingError("remote event page cursor changed")
                self._cursor = page.next_cursor
                for event in page.events:
                    if event.event in _RUN_TERMINAL_EVENTS:
                        self._run_terminal_event = event
                        if (
                            self._run_terminal is not None
                            and not self._run_terminal.done()
                        ):
                            self._run_terminal.set_result(event)
                    if event.event == FORWARD_BACKWARD_PREPARED_EVENT:
                        if event.operation_id is None:
                            raise RemoteTrainingError(
                                "F/B preparation event has no operation_id"
                            )
                        future = self._preparations.get(event.operation_id)
                        if future is not None and not future.done():
                            future.set_result(event)
                        continue
                    if event.event not in _TERMINAL_EVENTS:
                        continue
                    if event.operation_id is None:
                        raise RemoteTrainingError(
                            "terminal operation event has no operation_id"
                        )
                    future = self._pending.get(event.operation_id)
                    if future is not None and not future.done():
                        future.set_result(event)
                    preparation = self._preparations.get(event.operation_id)
                    if preparation is not None and not preparation.done():
                        preparation.set_result(event)
                await asyncio.sleep(0)
                if (
                    not self._pending
                    and not self._preparations
                    and (self._run_terminal is None or self._run_terminal.done())
                ):
                    return
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self._error = error
            pending, self._pending = tuple(self._pending.values()), {}
            preparations, self._preparations = (
                tuple(self._preparations.values()),
                {},
            )
            if self._run_terminal is not None and not self._run_terminal.done():
                self._run_terminal.set_exception(error)
            for future in (*pending, *preparations):
                if not future.done():
                    future.set_exception(error)
        finally:
            if self._task is asyncio.current_task():
                self._task = None


def _terminal_event(view: OperationView) -> RunEvent | None:
    event = {
        "succeeded": "operation_succeeded",
        "failed": "operation_failed",
        "cancelled": "operation_cancelled",
    }.get(view.status)
    if event is None:
        return None
    payload = view.result if view.status == "succeeded" else view.error
    if isinstance(payload, Contract):
        payload = payload.model_dump(mode="json")
    return RunEvent(
        cursor=view.event_cursor,
        run_id=view.ref.run_id,
        operation_id=view.ref.operation_id,
        event=event,
        payload=payload or {},
        created_at=view.updated_at,
    )


def _preparation_event(view: OperationView) -> RunEvent | None:
    disposition = view.gradient_disposition
    if view.ref.kind != "forward_backward" or disposition == "pending":
        return None
    if disposition is None:
        raise RemoteTrainingError("F/B operation omitted gradient disposition")
    return RunEvent(
        cursor=view.event_cursor,
        run_id=view.ref.run_id,
        operation_id=view.ref.operation_id,
        event=FORWARD_BACKWARD_PREPARED_EVENT,
        payload=ForwardBackwardPreparation(gradient_disposition=disposition).model_dump(
            mode="json"
        ),
        created_at=view.updated_at,
    )


class RemoteTrainingOperation(Generic[ResultT]):
    def __init__(
        self,
        ref: OperationRef,
        service: RemoteTrainingServiceClient,
        terminal: asyncio.Future[RunEvent],
        result_type: type[ResultT],
        preparation: asyncio.Future[RunEvent] | None = None,
        on_completion: Callable[[], None] | None = None,
    ) -> None:
        if (ref.kind == "forward_backward") != (preparation is not None):
            raise ValueError("preparation future is required exclusively for F/B")
        self._ref = ref
        self._service = service
        self._result_type = result_type
        self._completion: asyncio.Task[ResultT] | None = None
        self._terminal = terminal
        self._preparation = preparation
        self._cancel_request_id = uuid.uuid4().hex
        if on_completion is not None:
            self._terminal.add_done_callback(lambda _: on_completion())

    @property
    def ref(self) -> OperationRef:
        return self._ref

    async def result(self) -> ResultT:
        if self._completion is None:
            self._completion = asyncio.create_task(
                self._wait(), name=f"remote-training-result-{self._ref.operation_id}"
            )
        return await asyncio.shield(self._completion)

    async def gradient_disposition(self) -> PreparedGradientDisposition:
        if self._preparation is None:
            raise TypeError("gradient disposition is only available for F/B")
        event = await asyncio.shield(self._preparation)
        if event.event == "operation_failed":
            raise RemoteTrainingOperationError(self._ref.operation_id, event.payload)
        if event.event == "operation_cancelled":
            raise RemoteTrainingOperationCancelled(
                f"remote training operation {self._ref.operation_id} was cancelled"
            )
        if event.event != FORWARD_BACKWARD_PREPARED_EVENT:
            raise RemoteTrainingError("F/B terminated without durable preparation")
        return ForwardBackwardPreparation.model_validate(
            event.payload
        ).gradient_disposition

    async def _wait(self) -> ResultT:
        event = await asyncio.shield(self._terminal)
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
            result = await self._service.receive_operation_result(
                self._ref.run_id,
                self._ref.operation_id,
                remote,
                self._result_type,
            )
        else:
            result = self._result_type.model_validate(event.payload)
        if result.operation_id != self._ref.operation_id:
            raise RemoteTrainingError("operation result identity changed")
        if self._ref.kind in {"forward", "forward_backward"}:
            await self._service.acknowledge_operation_result(
                self._ref.run_id, self._ref.operation_id
            )
        return result

    async def cancel(self) -> None:
        try:
            await self._service.cancel_operation(
                self._ref.run_id,
                self._ref.operation_id,
                self._cancel_request_id,
            )
        except RemoteTrainingHttpError as error:
            if error.status_code != 409 or (
                await self._service.get_operation(
                    self._ref.run_id, self._ref.operation_id
                )
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
        close_timeout_s: float = process_shutdown_timeout(2),
    ) -> None:
        if run.status != "open":
            raise ValueError(f"remote training run is {run.status}, not open")
        if close_timeout_s <= 0:
            raise ValueError("remote training close timeout must be positive")
        self._service = service
        self._run = run
        self._next_sequence_id = run.next_sequence_id
        self._projected_learner_version = run.projected_learner_version
        self._events = _RunEventObserver(service, run.run_id)
        self._close_timeout_s = close_timeout_s
        self._lock = asyncio.Lock()
        self._operations: dict[str, tuple[str, RemoteTrainingOperation[Any]]] = {}
        self._open_forward_backward: list[
            RemoteTrainingOperation[ForwardBackwardResult]
        ] = []
        self._reserved_admission: tuple[str, str, OperationKind] | None = None
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
        if encoded_batch is not None and not isinstance(
            request, (ForwardRequest, ForwardBackwardRequest)
        ):
            raise TypeError("only forward commands accept a prepared batch")
        return await self._admit(
            request,
            kind=kind,
            result_type=result_type,
            encoded_batch=encoded_batch,
        )

    async def _admit(
        self,
        request: RunCommand,
        *,
        kind: OperationKind,
        result_type: type[ResultT],
        encoded_batch: EncodedTrainingBatch | None = None,
    ) -> TrainingOperation[ResultT]:
        async with self._lock:
            if self._closed:
                raise RuntimeError("remote training client is closed")
            if request.run_id != self.run_id:
                raise ValueError("command run_id does not match the remote client")
            reserved = self._reserved_admission
            if reserved is not None:
                request_id, _, reserved_kind = reserved
                if request_id != request.request_id:
                    raise RuntimeError("remote command admission remains unresolved")
                if reserved_kind != kind:
                    raise ValueError("request_id was reused with different content")
            if request.sequence_id > self._next_sequence_id:
                raise ValueError(
                    f"expected sequence {self._next_sequence_id}, "
                    f"got {request.sequence_id}"
                )
            if isinstance(request, (ForwardRequest, ForwardBackwardRequest)):
                prepared, cancelled = await complete_to_thread(
                    lambda: _prepare_forward_submission(request, encoded_batch)
                )
                if cancelled is not None:
                    raise cancelled
                wire_request, submission, fingerprint = prepared
            else:
                wire_request, submission = request, None
                fingerprint = remote_request_fingerprint(request)
            if existing := self._operations.get(request.request_id):
                if existing[0] != fingerprint:
                    raise ValueError("request_id was reused with different content")
                return cast(TrainingOperation[ResultT], existing[1])
            if request.sequence_id < self._next_sequence_id:
                return await self._resolve_replay(
                    request,
                    kind=kind,
                    fingerprint=fingerprint,
                    result_type=result_type,
                )
            if request.sequence_id != self._next_sequence_id:
                raise ValueError(
                    f"expected sequence {self._next_sequence_id}, "
                    f"got {request.sequence_id}"
                )
            if reserved is not None:
                request_id, reserved_fingerprint, reserved_kind = reserved
                if reserved_fingerprint != fingerprint or reserved_kind != kind:
                    raise ValueError("request_id was reused with different content")
            if kind == "optim_step":
                await asyncio.gather(
                    *(
                        operation.gradient_disposition()
                        for operation in self._open_forward_backward
                    )
                )
            operation_id = _operation_id(request.run_id, request.request_id)
            terminal = self._events.reserve(operation_id)
            preparation = (
                self._events.reserve_preparation(operation_id)
                if kind == "forward_backward"
                else None
            )
            self._reserved_admission = (request.request_id, fingerprint, kind)
            try:
                ref = (
                    await self._service.submit_forward(
                        kind, cast(RemoteForwardRequest, wire_request), submission
                    )
                    if submission is not None
                    else await self._service.submit(kind, wire_request)
                )
                if ref.operation_id != operation_id:
                    raise RemoteTrainingError(
                        "server returned a divergent operation_id"
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
                    terminal,
                    result_type,
                    preparation=preparation,
                    on_completion=self._bound_operation_cache,
                )
            except (asyncio.CancelledError, httpx.TransportError):
                raise
            except BaseException as error:
                self._events.abandon(operation_id, terminal, error)
                if preparation is not None:
                    self._events.abandon_preparation(operation_id, preparation, error)
                self._reserved_admission = None
                raise
            self._operations[request.request_id] = (fingerprint, operation)
            self._events.claim(operation_id, terminal)
            if preparation is not None:
                self._events.claim_preparation(operation_id, preparation)
            self._reserved_admission = None
            self._bound_operation_cache()
            self._next_sequence_id += 1
            if ref.reserved_output_learner_version is not None:
                self._projected_learner_version = ref.reserved_output_learner_version
            if kind == "forward_backward":
                forward = cast(
                    RemoteTrainingOperation[ForwardBackwardResult], operation
                )
                assert preparation is not None
                self._track_forward_backward(forward, preparation)
            elif kind == "optim_step":
                self._open_forward_backward.clear()
            return operation

    def _forget_empty_forward_backward(
        self,
        operation: RemoteTrainingOperation[ForwardBackwardResult],
        future: asyncio.Future[RunEvent],
    ) -> None:
        if future.cancelled():
            return
        try:
            event = future.result()
        except BaseException:
            return
        if event.event != FORWARD_BACKWARD_PREPARED_EVENT or (
            ForwardBackwardPreparation.model_validate(event.payload)
            .gradient_disposition
            != "empty"
        ):
            return
        try:
            self._open_forward_backward.remove(operation)
        except ValueError:
            pass

    async def _resolve_replay(
        self,
        request: RunCommand,
        *,
        kind: OperationKind,
        fingerprint: str,
        result_type: type[ResultT],
    ) -> TrainingOperation[ResultT]:
        operation_id = _operation_id(request.run_id, request.request_id)
        view = await self._service.get_operation(request.run_id, operation_id)
        self._validate_replay(view, request, kind, fingerprint)
        terminal_event = _terminal_event(view)
        preparation_event = _preparation_event(view)
        terminal_reserved = terminal_event is None
        if terminal_reserved:
            terminal = self._events.reserve(operation_id)
        else:
            terminal = asyncio.get_running_loop().create_future()
            terminal.set_result(terminal_event)

        preparation_reserved = False
        preparation: asyncio.Future[RunEvent] | None = None
        if kind == "forward_backward":
            preparation_event = preparation_event or terminal_event
            if preparation_event is None:
                preparation = self._events.reserve_preparation(operation_id)
                preparation_reserved = True
            else:
                preparation = asyncio.get_running_loop().create_future()
                preparation.set_result(preparation_event)

        if terminal_reserved or preparation_reserved:
            view = await self._service.get_operation(request.run_id, operation_id)
            self._validate_replay(view, request, kind, fingerprint)
            terminal_event = _terminal_event(view)
            preparation_event = _preparation_event(view) or terminal_event
            if terminal_event is not None and not terminal.done():
                terminal.set_result(terminal_event)
            if (
                preparation is not None
                and preparation_event is not None
                and not preparation.done()
            ):
                preparation.set_result(preparation_event)
            if terminal_reserved:
                self._events.claim(operation_id, terminal)
            if preparation_reserved:
                assert preparation is not None
                self._events.claim_preparation(operation_id, preparation)
        operation = RemoteTrainingOperation(
            view.ref,
            self._service,
            terminal,
            result_type,
            preparation=preparation,
            on_completion=self._bound_operation_cache,
        )
        self._operations[request.request_id] = (fingerprint, operation)
        if kind == "forward_backward":
            assert preparation is not None
            self._track_forward_backward(
                cast(RemoteTrainingOperation[ForwardBackwardResult], operation),
                preparation,
            )
        self._bound_operation_cache()
        return operation

    def _track_forward_backward(
        self,
        operation: RemoteTrainingOperation[ForwardBackwardResult],
        preparation: asyncio.Future[RunEvent],
    ) -> None:
        self._open_forward_backward.append(operation)
        preparation.add_done_callback(
            lambda future: self._forget_empty_forward_backward(operation, future)
        )

    def _validate_replay(
        self,
        view: OperationView,
        request: RunCommand,
        kind: OperationKind,
        fingerprint: str,
    ) -> None:
        if (
            view.ref.operation_id != _operation_id(request.run_id, request.request_id)
            or view.ref.run_id != self.run_id
            or view.ref.sequence_id != request.sequence_id
            or view.ref.kind != kind
            or view.request_id != request.request_id
            or view.request_fingerprint != fingerprint
        ):
            raise ValueError("replayed request differs from the persisted operation")

    def _bound_operation_cache(self) -> None:
        overflow = len(self._operations) - _MAX_RETAINED_COMPLETED_OPERATIONS
        if overflow <= 0:
            return
        for request_id, (_fingerprint, operation) in tuple(self._operations.items()):
            if overflow <= 0:
                return
            if operation._terminal.done():
                self._operations.pop(request_id, None)
                overflow -= 1

    async def forward(
        self, request: ForwardRequest
    ) -> TrainingOperation[ForwardResult]:
        return await self._submit(request, kind="forward", result_type=ForwardResult)

    async def forward_backward(
        self, request: ForwardBackwardRequest
    ) -> RemoteTrainingOperation[ForwardBackwardResult]:
        return cast(
            RemoteTrainingOperation[ForwardBackwardResult],
            await self._submit(
                request,
                kind="forward_backward",
                result_type=ForwardBackwardResult,
            ),
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
            if self._run.status not in {"closed", "failed"}:
                await asyncio.shield(self._events.reserve_run_terminal())
            self._run = await self._service.get_run(self.run_id)
            if self._run.status == "failed":
                raise RemoteTrainingError("remote training run failed during close")
            if self._run.status != "closed":
                raise RemoteTrainingError(
                    "terminal run event did not produce a terminal run state"
                )

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


def _consume_event_future(future: asyncio.Future[RunEvent]) -> None:
    try:
        future.exception()
    except asyncio.CancelledError:
        pass
