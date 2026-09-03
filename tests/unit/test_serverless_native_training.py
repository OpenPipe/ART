from datetime import UTC, datetime
from typing import Any, cast

import pytest

from art.serverless.client import (
    NativeOperationStatus,
    NativeTrainingOperation,
    NativeTrainingResult,
    NativeTrainingResultRelease,
    NativeTrainingRun,
    RemoteSamplerPublicationResult,
    TrainingRuns,
)
from art.serverless.native_training import RemoteTrainingClient, RemoteTrainingError
from art.training import (
    AdamConfig,
    AdapterSpec,
    ForwardBackwardRequest,
    ForwardBackwardResult,
    ForwardRequest,
    ForwardResult,
    LoadStateRequest,
    LoadStateResult,
    NamedLossRequest,
    OptimStepRequest,
    OptimStepResult,
    PackedInputCaptureRef,
    PackingOutcome,
    RunInitialState,
    SamplerPublication,
    SaveStateRequest,
    SaveStateResult,
    SaveWeightsForSamplerRequest,
    ServiceCheckpointSource,
    TokenMatrixBatch,
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
        self.close_request_id: str | None = None
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
                    logical_tokens=7,
                    physical_tokens=7,
                    packed_capacity_tokens=8,
                    padding_tokens=1,
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

    async def close(self, run_id: str, *, request_id: str) -> NativeTrainingRun:
        assert self.run is not None and run_id == self.run.run_id
        self.close_request_id = request_id
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
            run_name=body.get("run_name"),
            spec=spec,
            status="open",
            next_sequence_id=4,
            projected_learner_version=2,
            committed_learner_version=2,
        )


class _PublicWireTransport(TrainingRuns):
    def __init__(self, result_payloads: tuple[dict[str, Any], ...]) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.result_payloads = result_payloads
        self.operations: dict[str, tuple[dict[str, Any], str]] = {}

    async def _post(self, path, *, cast_to, body):
        self.posts.append((path, body))
        endpoint = path.rsplit("/", 1)[-1]
        operation_id = f"operation-{body['sequence_id']}"
        self.operations[operation_id] = (body, endpoint)
        return cast_to.model_validate(self._operation(operation_id, status="admitted"))

    async def _get(self, path, *, cast_to):
        operation_id = path.split("/operations/", 1)[1].split("/", 1)[0]
        if path.endswith("/result"):
            sequence_id = int(operation_id.removeprefix("operation-"))
            return cast_to.model_validate(self.result_payloads[sequence_id])
        return cast_to.model_validate(self._operation(operation_id, status="succeeded"))

    def _operation(self, operation_id: str, *, status: str) -> dict[str, Any]:
        body, endpoint = self.operations[operation_id]
        now = datetime(2026, 9, 2, tzinfo=UTC)
        transition = endpoint in {"optim_step", "load_state"}
        return {
            "operation_id": operation_id,
            "run_id": "run-native",
            "request_id": body["request_id"],
            "sequence_id": body["sequence_id"],
            "kind": endpoint,
            "status": status,
            "learner_parent_version": 0,
            "reserved_output_learner_version": 1 if transition else None,
            "admitted_at": now,
            "started_at": now if status == "succeeded" else None,
            "ended_at": now if status == "succeeded" else None,
            "cancel_requested": False,
            "latest_event_cursor": body["sequence_id"] + 1,
            "result_summary": None,
            "result_ref": (
                {
                    "result_id": operation_id,
                    "media_type": "application/json",
                    "size_bytes": 1,
                    "expires_at": datetime(2026, 9, 3, tzinfo=UTC),
                }
                if status == "succeeded"
                else None
            ),
            "error": None,
        }


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
        ),
        loss=NamedLossRequest(name="cispo"),
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
    assert service.close_request_id is not None
    assert service.close_request_id.startswith("close-")


@pytest.mark.asyncio
async def test_training_runs_resolve_serializes_exact_initial_state() -> None:
    service = _ResolveTransport()
    spec = TrainingRunSpec(
        base_model="Qwen/Qwen3-30B-A3B",
        adapter=AdapterSpec(
            rank=8,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        ),
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
            "lora_target_modules": ["k_proj", "o_proj", "q_proj", "v_proj"],
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
async def test_training_runs_translate_the_complete_public_wire() -> None:
    packed = {
        "packed_sequence_length": 8,
        "packed_sequences": 1,
        "target_packed_sequences": 1,
        "logical_tokens": 7,
        "physical_tokens": 7,
        "packed_capacity_tokens": 8,
        "padding_tokens": 1,
    }
    public_capture = {
        "capture_id": "capture-public",
        "manifest_sha256": "a" * 64,
        "content_sha256": "b" * 64,
    }
    public_checkpoint = {"checkpoint_id": "checkpoint-1", "learner_version": 1}
    result_payloads = (
        {
            "kind": "forward",
            "operation_id": "operation-0",
            "metrics": {"loss": 1.0},
            "packing": packed,
            "token_logprobs": [],
            "group_shapes": [
                {
                    "leaves": [
                        {
                            "matrix_id": "rollout-0",
                            "token_ids": [1, 2],
                            "shareable_length": 1,
                        }
                    ]
                }
            ],
            "packed_input": public_capture,
        },
        {
            "kind": "forward_backward",
            "operation_id": "operation-1",
            "metrics": {},
            "packing": packed,
            "training": {
                "accepted_trainable_tokens": 4,
                "policy_token_counts": [
                    {"policy_version": 0, "accepted_trainable_tokens": 4}
                ],
            },
            "loss": {
                "contract_id": "cispo_v1",
                "value": 0.25,
                "reduction": "mean_active_token",
            },
            "produced_gradient": True,
            "token_logprobs": [],
            "group_shapes": [],
            "packed_input": public_capture,
        },
        {
            "kind": "optim_step",
            "operation_id": "operation-2",
            "metrics": {},
            "contributing_operation_ids": ["operation-1"],
            "checkpoint": public_checkpoint,
            "learner_version": 1,
        },
        {
            "kind": "save_weights_for_sampler",
            "operation_id": "operation-3",
            "metrics": {},
            "target": "saved_generation",
            "model_alias": "sampler",
            "generation_id": "generation-1",
            "learner_version": 1,
            "checkpoint": public_checkpoint,
        },
        {
            "kind": "save_state",
            "operation_id": "operation-4",
            "metrics": {},
            "checkpoint": public_checkpoint,
            "archive": {
                **public_checkpoint,
                "components": ["weights", "optimizer"],
                "wandb_artifact": "entity/project/checkpoint:v1",
            },
        },
        {
            "kind": "load_state",
            "operation_id": "operation-5",
            "metrics": {},
            "checkpoint": {"checkpoint_id": "checkpoint-2", "learner_version": 2},
            "optimizer_restored": True,
        },
    )
    service = _PublicWireTransport(result_payloads)
    raw_batch = TokenMatrixBatch.model_validate(
        {
            "matrices": [
                {
                    "matrix_id": "rollout-0",
                    "rows": [
                        {
                            "name": "token_ids",
                            "dtype": "int64",
                            "shape": [2],
                            "values": {"encoding": "dense", "data": [1, 2]},
                        },
                        {
                            "name": "target_token_ids",
                            "dtype": "int64",
                            "shape": [2, 1],
                            "values": {"encoding": "dense", "data": [2, 3]},
                        },
                        {
                            "name": "behavior_logprobs",
                            "dtype": "float32",
                            "shape": [2, 1],
                            "values": {"encoding": "dense", "data": [-1.0, -1.0]},
                        },
                        {
                            "name": "advantages",
                            "dtype": "float32",
                            "shape": [2, 1],
                            "values": {"encoding": "dense", "data": [1.0, 1.0]},
                        },
                    ],
                }
            ]
        }
    )
    requests = (
        ForwardRequest(
            run_id="run-native",
            request_id="forward",
            sequence_id=0,
            batch=raw_batch,
            loss=NamedLossRequest(
                name="cispo",
                values={"clip_low_threshold": 0.2, "clip_high_threshold": 0.3},
            ),
            collect_packing_shapes=True,
            retain_packed_input=True,
        ),
        ForwardBackwardRequest(
            run_id="run-native",
            request_id="forward-backward",
            sequence_id=1,
            batch=raw_batch,
            loss=NamedLossRequest(name="cispo"),
            retain_packed_input=True,
        ),
        OptimStepRequest(
            run_id="run-native",
            request_id="optimizer",
            sequence_id=2,
            optimizer=AdamConfig(learning_rate=1e-5),
        ),
        SaveWeightsForSamplerRequest(
            run_id="run-native",
            request_id="publication",
            sequence_id=3,
            checkpoint_name="checkpoint-1",
            publication=SamplerPublication(
                mode="versioned_lora", model_alias="sampler"
            ),
        ),
        SaveStateRequest(
            run_id="run-native",
            request_id="save",
            sequence_id=4,
            checkpoint_name="checkpoint-1",
        ),
        LoadStateRequest(
            run_id="run-native",
            request_id="load",
            sequence_id=5,
            checkpoint="service-checkpoint:checkpoint-2",
            restore_optimizer=True,
        ),
    )
    result_types = (
        ForwardResult,
        ForwardBackwardResult,
        OptimStepResult,
        RemoteSamplerPublicationResult,
        SaveStateResult,
        LoadStateResult,
    )

    translated = []
    for request, result_type in zip(requests, result_types, strict=True):
        admitted = await service.submit(request)
        assert admitted.execution_started_at is None
        assert admitted.execution_ended_at is None
        assert not admitted.result_available
        terminal = await service.operation("run-native", admitted.operation_id)
        assert terminal.execution_started_at == terminal.execution_ended_at
        assert terminal.result_available and terminal.result is None
        envelope = await service.result("run-native", admitted.operation_id)
        translated.append(result_type.model_validate(envelope.result))

    assert [path.rsplit("/", 1)[-1] for path, _ in service.posts] == [
        "forward",
        "forward_backward",
        "optim_step",
        "save_weights_for_sampler",
        "save_state",
        "load_state",
    ]
    assert service.posts[0][1]["loss"] == {
        "name": "cispo",
        "normalize_advantages": True,
        "clip_low_threshold": 0.2,
        "clip_high_threshold": 0.3,
    }
    assert service.posts[3][1]["publication"] == {
        "target": "saved_generation",
        "model_alias": "sampler",
    }
    assert service.posts[5][1]["source"] == {
        "kind": "service_checkpoint",
        "checkpoint_id": "checkpoint-2",
    }
    forward, forward_backward, optimizer, sampler, saved, loaded = translated
    leaf = forward.packing.group_shapes[0].leaves[0]
    assert leaf.matrix_id == "rollout-0"
    assert leaf.token_ids.tolist() == [1, 2]
    assert forward.packed_input_capture is not None
    assert forward.packed_input_capture.run_id == "run-native"
    assert forward_backward.produced_gradient
    assert forward_backward.training.accepted_trainable_tokens == 4
    assert forward_backward.loss.contract_id == "cispo_v1"
    assert optimizer.contributing_forward_backward_operation_ids == ("operation-1",)
    assert sampler.kind == "save_sampler"
    assert (
        sampler.target,
        sampler.model_alias,
        sampler.generation_id,
    ) == ("saved_generation", "sampler", "generation-1")
    assert all(
        result.checkpoint.run_id == "run-native"
        for result in (optimizer, sampler, saved, loaded)
    )
    assert saved.archive is not None and loaded.optimizer_restored

    with pytest.raises(ValueError, match="public raw training batch"):
        await service.submit(_request())
    service.result_payloads = (
        result_payloads[0] | {"physical_lora_locator": "private"},
        *result_payloads[1:],
    )
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        await service.result("run-native", "operation-0")


@pytest.mark.asyncio
async def test_training_runs_resolve_omits_implicit_default_name() -> None:
    service = _ResolveTransport()
    run = await service.resolve(
        request_id="resolve-default",
        spec=TrainingRunSpec(
            base_model="Qwen/Qwen3-30B-A3B",
            adapter=AdapterSpec(rank=8, target_modules=("linear_qkv",)),
        ),
    )

    assert run.run_name is None
    assert service.body is not None
    assert "run_name" not in service.body


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
