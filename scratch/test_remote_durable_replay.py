from __future__ import annotations

from datetime import UTC, datetime
import hashlib

import pytest

import art.serverless.client as client_module
from art.serverless.client import RemoteTrainingClient, RemoteTrainingServiceClient
from art.serverless.contracts import (
    AdapterSpec,
    EventPage,
    OperationView,
    TrainingRunSpec,
    TrainingRunView,
    remote_request_fingerprint,
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


def _request(request_id: str, sequence_id: int, learning_rate: float = 1e-6):
    return OptimStepRequest(
        run_id="run",
        request_id=request_id,
        sequence_id=sequence_id,
        optimizer=AdamConfig(learning_rate=learning_rate),
    )


def _run(next_sequence_id: int, learner_version: int) -> TrainingRunView:
    now = datetime.now(UTC)
    return TrainingRunView(
        run_id="run",
        spec=TrainingRunSpec(
            run_name="replay",
            base_model="model",
            adapter=AdapterSpec(rank=1, target_modules=("q_proj",)),
        ),
        status="open",
        next_sequence_id=next_sequence_id,
        projected_learner_version=learner_version,
        committed_learner_version=learner_version,
        created_at=now,
        updated_at=now,
    )


class _ReplayService(RemoteTrainingServiceClient):
    def __init__(self) -> None:
        self.learner_version = 0
        self.views: dict[str, OperationView] = {}
        self.gets: dict[str, int] = {}
        self.finish_on_second_get: set[str] = set()

    async def submit(self, kind: OperationKind, request: RunCommand) -> OperationRef:
        ref = OperationRef(
            run_id=request.run_id,
            operation_id=_operation_id(request.run_id, request.request_id),
            sequence_id=request.sequence_id,
            learner_parent_version=self.learner_version,
            reserved_output_learner_version=self.learner_version + 1,
            kind=kind,
        )
        self.learner_version += 1
        self.views[ref.operation_id] = self._view(request, ref, "succeeded")
        return ref

    async def get_operation(self, run_id: str, operation_id: str) -> OperationView:
        assert run_id == "run"
        self.gets[operation_id] = self.gets.get(operation_id, 0) + 1
        view = self.views[operation_id]
        if operation_id in self.finish_on_second_get and self.gets[operation_id] == 2:
            view = view.model_copy(
                update={
                    "status": "succeeded",
                    "result": OptimStepResult(
                        operation_id=operation_id,
                        contributing_forward_backward_operation_ids=("fb",),
                    ).model_dump(mode="json"),
                }
            )
            self.views[operation_id] = view
        return view

    async def get_events(self, run_id: str, *, after: int) -> EventPage:
        del run_id
        return EventPage(events=(), next_cursor=after)

    @staticmethod
    def _view(
        request: OptimStepRequest,
        ref: OperationRef,
        status: str,
    ) -> OperationView:
        now = datetime.now(UTC)
        result = (
            OptimStepResult(
                operation_id=ref.operation_id,
                contributing_forward_backward_operation_ids=("fb",),
            ).model_dump(mode="json")
            if status == "succeeded"
            else None
        )
        return OperationView(
            ref=ref,
            request_id=request.request_id,
            request_fingerprint=remote_request_fingerprint(request),
            status=status,
            result=result,
            event_cursor=request.sequence_id + 1,
            created_at=now,
            updated_at=now,
        )


@pytest.mark.asyncio
async def test_evicted_operation_replays_without_advancing_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client_module, "_MAX_RETAINED_COMPLETED_OPERATIONS", 1)
    service = _ReplayService()
    requests = [_request(f"request-{index}", index) for index in range(3)]
    for request in requests:
        ref = OperationRef(
            run_id="run",
            operation_id=_operation_id("run", request.request_id),
            sequence_id=request.sequence_id,
            learner_parent_version=request.sequence_id,
            reserved_output_learner_version=request.sequence_id + 1,
            kind="optim_step",
        )
        service.views[ref.operation_id] = service._view(request, ref, "succeeded")
    client = RemoteTrainingClient(service, _run(3, 3))
    for request in requests:
        assert (await (await client.optim_step(request)).result()).operation_id == (
            _operation_id("run", request.request_id)
        )
    assert requests[0].request_id not in client._operations

    before = client.next_sequence_id, client.projected_learner_version
    replay = await client.optim_step(requests[0])
    assert (await replay.result()).operation_id == _operation_id("run", "request-0")
    assert (client.next_sequence_id, client.projected_learner_version) == before
    await client.abort_result_waiters()


@pytest.mark.asyncio
async def test_reconstructed_client_resolves_terminal_and_resolution_race() -> None:
    service = _ReplayService()
    first = _request("finished", 0)
    first_ref = await service.submit("optim_step", first)
    client = RemoteTrainingClient(service, _run(1, 1))

    finished = await client.optim_step(first)
    assert (await finished.result()).operation_id == first_ref.operation_id
    assert client.next_sequence_id == 1

    live = _request("live", 0)
    live_ref = first_ref.model_copy(
        update={"operation_id": _operation_id("run", "live")}
    )
    service.views[live_ref.operation_id] = service._view(live, live_ref, "running")
    service.finish_on_second_get.add(live_ref.operation_id)
    raced = await client.optim_step(live)
    assert (await raced.result()).operation_id == live_ref.operation_id
    assert service.gets[live_ref.operation_id] == 2
    await client.abort_result_waiters()


@pytest.mark.asyncio
async def test_replay_rejects_changed_content() -> None:
    service = _ReplayService()
    request = _request("request", 0)
    await service.submit("optim_step", request)
    client = RemoteTrainingClient(service, _run(1, 1))
    with pytest.raises(ValueError, match="differs from the persisted operation"):
        await client.optim_step(_request("request", 0, learning_rate=2e-6))
    await client.abort_result_waiters()
