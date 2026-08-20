from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import hashlib

import httpx
import pytest

from art.serverless.client import (
    RemoteTrainingClient,
    RemoteTrainingError,
    RemoteTrainingServiceClient,
)
from art.serverless.contracts import (
    AdapterSpec,
    EventPage,
    RunEvent,
    TrainingRunSpec,
    TrainingRunView,
)
from art.training.contracts import (
    AdamConfig,
    OperationKind,
    OperationRef,
    OptimStepRequest,
    OptimStepResult,
    RunCommand,
)


def _operation_id(run_id: str, request_id: str) -> str:
    return hashlib.sha256(f"{run_id}\0{request_id}".encode()).hexdigest()


class _OrderedService(RemoteTrainingServiceClient):
    def __init__(self) -> None:
        self.events: list[RunEvent] = []
        self.delivered: dict[str, asyncio.Event] = {}
        self.attempts: dict[str, int] = {}

    async def submit(self, kind: OperationKind, request: RunCommand) -> OperationRef:
        operation_id = _operation_id(request.run_id, request.request_id)
        ref = OperationRef(
            run_id=request.run_id,
            operation_id=operation_id,
            sequence_id=request.sequence_id,
            learner_parent_version=request.sequence_id,
            reserved_output_learner_version=request.sequence_id + 1,
            kind=kind,
        )
        if request.request_id == "divergent":
            return ref.model_copy(update={"operation_id": "wrong"})

        attempt = self.attempts.get(request.request_id, 0) + 1
        self.attempts[request.request_id] = attempt
        if attempt == 1:
            result = OptimStepResult(
                operation_id=operation_id,
                contributing_forward_backward_operation_ids=(
                    f"forward-{request.sequence_id}",
                ),
            )
            self.events.append(
                RunEvent(
                    cursor=len(self.events) + 1,
                    run_id=request.run_id,
                    operation_id=operation_id,
                    event="operation_succeeded",
                    payload=result.model_dump(mode="json"),
                    created_at=datetime.now(UTC),
                )
            )
            self.delivered[operation_id] = asyncio.Event()

        if request.request_id in {"race", "ambiguous"}:
            await self.delivered[operation_id].wait()
        if request.request_id == "ambiguous" and attempt == 1:
            raise httpx.ReadError(
                "response lost after admission",
                request=httpx.Request("POST", "http://test/submit"),
            )
        return ref

    async def get_events(self, run_id: str, *, after: int) -> EventPage:
        events = tuple(
            event
            for event in self.events
            if event.run_id == run_id and event.cursor > after
        )
        for event in events:
            assert event.operation_id is not None
            self.delivered[event.operation_id].set()
        return EventPage(
            events=events,
            next_cursor=events[-1].cursor if events else after,
        )


def _request(request_id: str, sequence_id: int) -> OptimStepRequest:
    return OptimStepRequest(
        run_id="run",
        request_id=request_id,
        sequence_id=sequence_id,
        optimizer=AdamConfig(learning_rate=1e-6),
    )


@pytest.mark.asyncio
async def test_terminal_event_before_submit_response_is_not_lost() -> None:
    now = datetime.now(UTC)
    service = _OrderedService()
    client = RemoteTrainingClient(
        service,
        TrainingRunView(
            run_id="run",
            spec=TrainingRunSpec(
                run_name="race",
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
    try:
        warm = await client.optim_step(_request("warm", 0))
        await asyncio.wait_for(warm.result(), 0.2)

        raced = await asyncio.wait_for(client.optim_step(_request("race", 1)), 0.2)
        raced_id = _operation_id("run", "race")
        assert service.delivered[raced_id].is_set()
        assert client._events._cursor >= 2
        assert (await asyncio.wait_for(raced.result(), 0.2)).operation_id == raced_id

        with pytest.raises(httpx.ReadError, match="response lost"):
            await client.optim_step(_request("ambiguous", 2))
        ambiguous_id = _operation_id("run", "ambiguous")
        reservation = client._events._pending[ambiguous_id]
        assert reservation.done()
        with pytest.raises(RuntimeError, match="admission remains unresolved"):
            await client.optim_step(_request("different", 2))
        retried = await client.optim_step(_request("ambiguous", 2))
        assert (await asyncio.wait_for(retried.result(), 0.2)).operation_id == (
            ambiguous_id
        )
        await asyncio.sleep(0)
        assert ambiguous_id not in client._events._pending

        divergent_id = _operation_id("run", "divergent")
        with pytest.raises(RemoteTrainingError, match="divergent operation_id"):
            await client.optim_step(_request("divergent", 3))
        assert divergent_id not in client._events._pending
        assert client._reserved_admission is None

        recovery = await client.optim_step(_request("recovery", 3))
        await asyncio.wait_for(recovery.result(), 0.2)
        await asyncio.sleep(0)
        assert not client._events._pending
    finally:
        await client.abort_result_waiters()
