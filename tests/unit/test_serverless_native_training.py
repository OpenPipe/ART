from datetime import UTC, datetime
from typing import Any, cast

import pytest

from art.serverless.client import (
    NativeOperationStatus,
    NativeTrainingOperation,
    NativeTrainingResult,
    NativeTrainingResultRelease,
    NativeTrainingRun,
    TrainingRuns,
)
from art.serverless.native_training import RemoteTrainingClient, RemoteTrainingError
from art.training import (
    AdapterSpec,
    ForwardRequest,
    LossConfig,
    PackedInputCaptureRef,
    PackingOutcome,
    RunInitialState,
    ServiceCheckpointSource,
    TrainingRunSpec,
    WandbArtifactCheckpointSource,
)


class _TrainingRuns:
    def __init__(self) -> None:
        self.submissions = 0
        self.operations: dict[str, NativeTrainingOperation] = {}
        self.requests: dict[str, ForwardRequest] = {}
        self.result_calls = 0
        self.released: tuple[str, str] | None = None
        self.run: NativeTrainingRun | None = None
        self.resolve_kwargs: dict[str, Any] | None = None

    async def resolve(self, **kwargs: Any) -> NativeTrainingRun:
        self.resolve_kwargs = kwargs
        spec = kwargs["spec"]
        self.run = NativeTrainingRun(
            run_id="run-native",
            run_name=kwargs["run_name"],
            spec={
                "base_model": spec.base_model,
                "dtype": spec.dtype,
                "lora_rank": spec.adapter.rank,
                "lora_target_modules": list(spec.adapter.target_modules),
                "seed": spec.seed,
            },
            status="open",
            next_sequence_id=0,
            projected_learner_version=0,
            committed_learner_version=0,
        )
        return self.run

    async def submit(self, request: ForwardRequest) -> NativeTrainingOperation:
        self.submissions += 1
        prior = self.operations.get(request.request_id)
        if prior is not None:
            if self.requests[prior.operation_id] != request:
                raise RemoteTrainingError("server rejected divergent command")
            return prior
        operation_id = (
            "operation-native"
            if request.request_id == "request-native"
            else f"operation-{request.sequence_id}"
        )
        admitted = _operation(request, operation_id=operation_id, status="admitted")
        self.operations[request.request_id] = admitted
        self.requests[operation_id] = request
        return admitted

    async def operation(
        self, run_id: str, operation_id: str
    ) -> NativeTrainingOperation:
        assert run_id == "run-native"
        admitted = next(
            operation
            for operation in self.operations.values()
            if operation.operation_id == operation_id
        )
        request = self.requests[operation_id]
        return _operation(
            request,
            operation_id=operation_id,
            status="succeeded",
            result_available=True,
        )

    async def result(self, run_id: str, operation_id: str) -> NativeTrainingResult:
        assert run_id == "run-native"
        self.result_calls += 1
        return NativeTrainingResult(
            operation_id=operation_id,
            kind="forward",
            result={
                "kind": "forward",
                "operation_id": operation_id,
                "packing": PackingOutcome(
                    packed_sequence_length=8,
                    packed_sequences=1,
                    target_packed_sequences=1,
                    physical_tokens=8,
                    non_padding_tokens=7,
                    loss_bearing_tokens=4,
                    trainable_assistant_tokens=4,
                ).model_dump(mode="json"),
            },
        )

    async def release_result(
        self, run_id: str, operation_id: str, *, request_id: str
    ) -> NativeTrainingResultRelease:
        assert run_id == "run-native"
        self.released = (operation_id, request_id)
        return NativeTrainingResultRelease(
            operation_id=operation_id,
            request_id=request_id,
            released=True,
        )

    async def cancel(self, run_id: str, operation_id: str) -> NativeTrainingOperation:
        raise AssertionError((run_id, operation_id))

    async def close(self, run_id: str) -> NativeTrainingRun:
        assert self.run is not None and run_id == self.run.run_id
        return self.run.model_copy(update={"status": "closing"})


class _ResolveTransport(TrainingRuns):
    def __init__(self, *, change_seed: bool = False) -> None:
        self.change_seed = change_seed
        self.path: str | None = None
        self.body: dict[str, Any] | None = None

    async def _post(self, path, *, cast_to, body):
        assert cast_to is NativeTrainingRun
        self.path = path
        self.body = body
        spec = dict(body["spec"])
        if self.change_seed:
            spec["seed"] = 999
        return NativeTrainingRun(
            run_id="run-native",
            run_name=body["run_name"],
            spec=spec,
            status="open",
            next_sequence_id=4,
            projected_learner_version=2,
            committed_learner_version=2,
        )


def _request(
    *, request_id: str = "request-native", sequence_id: int = 0
) -> ForwardRequest:
    return ForwardRequest(
        run_id="run-native",
        request_id=request_id,
        sequence_id=sequence_id,
        batch=PackedInputCaptureRef(
            run_id="run-native",
            capture_id="capture-native",
            manifest_sha256="a" * 64,
            input_kind="rl",
        ),
        loss=LossConfig(name="cispo"),
    )


def _operation(
    request: ForwardRequest,
    *,
    operation_id: str = "operation-native",
    status: NativeOperationStatus,
    result_available: bool = False,
) -> NativeTrainingOperation:
    return NativeTrainingOperation(
        operation_id=operation_id,
        run_id=request.run_id,
        request_id=request.request_id,
        sequence_id=request.sequence_id,
        kind="forward",
        status=status,
        learner_parent_version=0,
        reserved_output_learner_version=None,
        admitted_at=datetime(2026, 8, 29, tzinfo=UTC),
        execution_started_at=datetime(2026, 8, 29, tzinfo=UTC),
        execution_ended_at=(
            datetime(2026, 8, 29, tzinfo=UTC)
            if status in {"succeeded", "failed", "cancelled"}
            else None
        ),
        cancel_requested=False,
        latest_event_cursor=1,
        result_available=result_available,
        error=None,
    )


@pytest.mark.asyncio
async def test_native_client_retains_run_and_terminal_operation_identity() -> None:
    service = _TrainingRuns()
    client = await RemoteTrainingClient.resolve(
        cast(TrainingRuns, service),
        request_id="resolve-native",
        run_name="gate-2-run",
        spec=TrainingRunSpec(
            base_model="Qwen/Qwen3-30B-A3B",
            adapter=AdapterSpec(rank=32, target_modules=("linear_qkv",)),
        ),
        poll_interval_s=0.001,
    )

    operation = await client.forward(_request())
    replay = await client.forward(_request())
    result = await operation.result()

    assert client.run_id == "run-native"
    assert client.operation_ids == ("operation-native",)
    assert replay is operation
    assert operation.ref.operation_id == result.operation_id == "operation-native"
    evidence = await client.operation_evidence(operation.ref.operation_id)
    assert evidence.result is not None
    assert evidence.result["operation_id"] == result.operation_id
    assert evidence.result["packing"] == result.packing.model_dump(mode="json")
    assert evidence.execution_ended_at is not None
    released = await operation.release_result(request_id="release-native")
    assert released.released is True
    assert service.released == ("operation-native", "release-native")
    assert service.result_calls == 2
    assert service.submissions == 1
    assert service.resolve_kwargs is not None
    assert service.resolve_kwargs["initial_state"] is None


@pytest.mark.asyncio
async def test_native_client_resolve_preserves_typed_initial_state() -> None:
    service = _TrainingRuns()
    spec = TrainingRunSpec(
        base_model="Qwen/Qwen3-30B-A3B",
        adapter=AdapterSpec(rank=8, target_modules=("linear_qkv",)),
        seed=17,
    )
    initial_state = RunInitialState(
        source=ServiceCheckpointSource(checkpoint_id="checkpoint-1"),
        restore_optimizer=True,
    )

    client = await RemoteTrainingClient.resolve(
        cast(TrainingRuns, service),
        request_id="resume-native",
        run_name="resumed-run",
        spec=spec,
        initial_state=initial_state,
    )

    assert client.run_id == "run-native"
    assert service.resolve_kwargs == {
        "request_id": "resume-native",
        "run_name": "resumed-run",
        "spec": spec,
        "initial_state": initial_state,
    }
    await client.close()


@pytest.mark.asyncio
async def test_training_runs_resolve_serializes_exact_initial_state() -> None:
    service = _ResolveTransport()
    spec = TrainingRunSpec(
        base_model="Qwen/Qwen3-30B-A3B",
        adapter=AdapterSpec(rank=8, target_modules=("linear_qkv",)),
        seed=17,
    )
    initial_state = RunInitialState(
        source=WandbArtifactCheckpointSource(artifact="entity/project/checkpoint:v3"),
    )

    run = await service.resolve(
        request_id="resume-native",
        run_name="resumed-run",
        spec=spec,
        initial_state=initial_state,
    )

    assert run.run_id == "run-native"
    assert service.path == "/training/runs:resolve"
    assert service.body == {
        "request_id": "resume-native",
        "run_name": "resumed-run",
        "spec": {
            "base_model": "Qwen/Qwen3-30B-A3B",
            "dtype": "bfloat16",
            "lora_rank": 8,
            "lora_target_modules": ["linear_qkv"],
            "seed": 17,
        },
        "initial_state": {
            "source": {
                "kind": "wandb_artifact",
                "artifact": "entity/project/checkpoint:v3",
            },
            "restore_optimizer": False,
        },
    }


@pytest.mark.asyncio
async def test_training_runs_resolve_rejects_changed_seed() -> None:
    service = _ResolveTransport(change_seed=True)

    with pytest.raises(RuntimeError, match="run identity changed"):
        await service.resolve(
            request_id="resume-native",
            run_name="resumed-run",
            spec=TrainingRunSpec(
                base_model="Qwen/Qwen3-30B-A3B",
                adapter=AdapterSpec(rank=8, target_modules=("linear_qkv",)),
                seed=17,
            ),
        )


@pytest.mark.asyncio
async def test_native_client_bounds_indexes_and_replays_evicted_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import art.serverless.native_training as native_training

    monkeypatch.setattr(native_training, "_MAX_RETAINED_OPERATIONS", 1)
    service = _TrainingRuns()
    client = await RemoteTrainingClient.resolve(
        cast(TrainingRuns, service),
        request_id="resolve-native",
        run_name="gate-2-run",
        spec=TrainingRunSpec(
            base_model="Qwen/Qwen3-30B-A3B",
            adapter=AdapterSpec(rank=32, target_modules=("linear_qkv",)),
        ),
    )
    first_request = _request()
    second_request = _request(request_id="request-second", sequence_id=1)

    first = await client.forward(first_request)
    second = await client.forward(second_request)

    assert client.operation_ids == (second.ref.operation_id,)
    with pytest.raises(native_training.RemoteTrainingError, match="divergent"):
        await client.forward(
            first_request.model_copy(update={"collect_packing_shapes": True})
        )

    replay = await client.forward(first_request)
    assert replay is not first
    assert replay.ref == first.ref
    assert client.next_sequence_id == 2
    assert client.operation_ids == (first.ref.operation_id,)
    assert service.submissions == 4

    monkeypatch.setattr(native_training, "_MAX_RETAINED_OPERATIONS", 8)
    monkeypatch.setattr(
        native_training,
        "_MAX_RETAINED_OPERATION_INDEX_BYTES",
        client._operation_index_bytes,
    )
    third = await client.forward(_request(request_id="request-third", sequence_id=2))

    assert client.operation_ids == (third.ref.operation_id,)
    assert (
        client._operation_index_bytes
        <= native_training._MAX_RETAINED_OPERATION_INDEX_BYTES
    )
    assert service.submissions == 5

    await client.close()
    assert client.operation_ids == ()
    assert client._operation_index_bytes == 0
