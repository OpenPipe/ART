from __future__ import annotations

import asyncio
import threading
from typing import Any

import numpy as np
from pydantic import BaseModel
import pytest
import tinker

from art.serverless.contracts import (
    CreateTrainingRunRequest,
    TrainingRunSpec,
)
import art.tinker_compat as compat
from art.tinker_compat._runtime import AsyncRuntime
import art.tinker_compat.client as compat_client
from art.training.contracts import (
    CheckpointRef,
    ForwardBackwardRequest,
    ForwardBackwardResult,
    ForwardRequest,
    ForwardResult,
    LoadStateRequest,
    LoadStateResult,
    LossFnOutput,
    OptimStepRequest,
    OptimStepResult,
    PackingOutcome,
    SamplerWeightsResult,
    SaveStateRequest,
    SaveStateResult,
    SaveWeightsForSamplerRequest,
)


def _packing() -> PackingOutcome:
    return PackingOutcome(
        packed_sequence_length=2,
        packed_sequences=1,
        target_packed_sequences=1,
        nominal_capacity_tokens=2,
        physical_tokens=2,
        non_padding_tokens=2,
        loss_bearing_tokens=2,
        trainable_assistant_tokens=2,
        policy_token_counts=None,
        group_shapes=(),
    )


def _checkpoint(run_id: str, checkpoint_id: str) -> CheckpointRef:
    return CheckpointRef(
        run_id=run_id,
        learner_version=1,
        checkpoint_id=checkpoint_id,
    )


def test_compat_runtime_stop_is_bounded() -> None:
    runtime = AsyncRuntime()
    started = threading.Event()
    release = threading.Event()

    async def cancellation_resistant() -> None:
        started.set()
        while not release.is_set():
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                pass

    runtime.submit_future(cancellation_resistant())
    assert started.wait(1.0)
    with pytest.raises(RuntimeError, match="did not stop in time"):
        runtime.stop(0.01)
    release.set()
    runtime.stop(1.0)


class FakeOperation:
    def __init__(self, value: Any) -> None:
        self.value = value

    async def result(self) -> Any:
        return self.value


class FakeRemoteTrainingClient:
    def __init__(self) -> None:
        self.run_id = "run-1"
        self.next_sequence_id = 0
        self.projected_learner_version = 0
        self.requests: list[Any] = []
        self.closed = False
        self.shutdown_calls = 0
        self.shutdown_failures = 0
        self.shutdown_delay_s = 0.0
        self.delay_next_forward_backward = False

    def _operation(self, request: Any, value: Any) -> FakeOperation:
        assert request.sequence_id == self.next_sequence_id
        self.next_sequence_id += 1
        self.requests.append(request)
        return FakeOperation(value)

    def _forward_result(
        self, request: ForwardRequest | ForwardBackwardRequest, *, backward: bool
    ) -> ForwardResult | ForwardBackwardResult:
        outputs = []
        for datum in request.batch.datums:
            if datum.candidate_count == 1:
                values: tuple[float, ...] | tuple[tuple[float, ...], ...] = tuple(
                    -0.1 * (index + 1) for index in range(len(datum.input_tokens))
                )
            else:
                values = tuple(
                    tuple(
                        -0.1 * (column + 1) for column in range(datum.candidate_count)
                    )
                    for _ in datum.input_tokens
                )
            outputs.append(LossFnOutput(token_logprobs=values))
        result_type = ForwardBackwardResult if backward else ForwardResult
        return result_type(
            operation_id=f"operation-{request.sequence_id}",
            packing=_packing(),
            loss_fn_outputs=tuple(outputs),
            metrics={"loss": 1.25},
        )

    async def forward(self, request: ForwardRequest) -> FakeOperation:
        return self._operation(request, self._forward_result(request, backward=False))

    async def forward_backward(self, request: ForwardBackwardRequest) -> FakeOperation:
        if self.delay_next_forward_backward:
            self.delay_next_forward_backward = False
            await asyncio.sleep(0.05)
        return self._operation(request, self._forward_result(request, backward=True))

    async def optim_step(self, request: OptimStepRequest) -> FakeOperation:
        return self._operation(
            request,
            OptimStepResult(
                operation_id=f"operation-{request.sequence_id}",
                contributing_forward_backward_operation_ids=("fb-1",),
                checkpoint=_checkpoint(self.run_id, f"operation-{request.sequence_id}"),
                metrics={"learning_rate": request.optimizer.learning_rate},
            ),
        )

    async def save_state(self, request: SaveStateRequest) -> FakeOperation:
        return self._operation(
            request,
            SaveStateResult(
                operation_id=f"operation-{request.sequence_id}",
                checkpoint=_checkpoint(self.run_id, request.checkpoint_name),
                lora="lora://state",
                training_session_id="session",
                generation_id="generation",
                lora_bytes=1,
                optimizer_state="optimizer://state",
                optimizer_bytes=1,
            ),
        )

    async def load_state(self, request: LoadStateRequest) -> FakeOperation:
        return self._load(request, optimizer=False)

    async def load_state_with_optimizer(
        self, request: LoadStateRequest
    ) -> FakeOperation:
        return self._load(request, optimizer=True)

    def _load(self, request: LoadStateRequest, *, optimizer: bool) -> FakeOperation:
        return self._operation(
            request,
            LoadStateResult(
                operation_id=f"operation-{request.sequence_id}",
                checkpoint=_checkpoint(self.run_id, request.checkpoint),
                lora="lora://state",
                training_session_id="session",
                generation_id="generation",
                lora_bytes=1,
                optimizer_restored=optimizer,
            ),
        )

    async def save_weights_for_sampler(
        self, request: SaveWeightsForSamplerRequest
    ) -> FakeOperation:
        return self._operation(
            request,
            SamplerWeightsResult(
                operation_id=f"operation-{request.sequence_id}",
                checkpoint=_checkpoint(self.run_id, request.checkpoint_name),
                lora="lora://sampler",
                training_session_id="session",
                generation_id="generation",
                lora_bytes=1,
            ),
        )

    async def close(self) -> None:
        self.closed = True

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        await asyncio.sleep(self.shutdown_delay_s)
        if self.shutdown_calls <= self.shutdown_failures:
            raise RuntimeError("injected remote shutdown failure")
        await self.close()


class FakeRun(BaseModel):
    spec: TrainingRunSpec


class FakeService:
    def __init__(self) -> None:
        self.created: CreateTrainingRunRequest | None = None
        self.created_requests: list[CreateTrainingRunRequest] = []
        self.closed = False

    async def get_run(self, run_id: str) -> FakeRun:
        assert run_id == "run-1"
        assert self.created_requests
        return FakeRun(spec=self.created_requests[0].spec)

    async def close(self) -> None:
        self.closed = True


class FakeSamplingProvider:
    def __init__(self) -> None:
        self.targets: list[compat.SamplingTarget] = []

    async def sample(
        self,
        *,
        target: compat.SamplingTarget,
        prompt: tinker.ModelInput,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
        include_prompt_logprobs: bool,
        topk_prompt_logprobs: int,
    ) -> tinker.SampleResponse:
        del prompt, sampling_params, include_prompt_logprobs, topk_prompt_logprobs
        self.targets.append(target)
        sequence = tinker.SampledSequence(
            stop_reason="length",
            tokens_np=np.asarray([7], dtype=np.int64),
            logprobs_np=np.asarray([-0.25], dtype=np.float32),
        )
        return tinker.SampleResponse(sequences=[sequence] * num_samples)

    async def compute_logprobs(
        self, *, target: compat.SamplingTarget, prompt: tinker.ModelInput
    ) -> list[float | None]:
        self.targets.append(target)
        return [None, *([-0.5] * (prompt.length - 1))]


class Clients:
    def __init__(self) -> None:
        self.service = FakeService()
        self.remote = FakeRemoteTrainingClient()
        self.sampling = FakeSamplingProvider()

        async def create(
            _service: Any, request: CreateTrainingRunRequest
        ) -> FakeRemoteTrainingClient:
            self.service.created = request
            self.service.created_requests.append(request)
            return self.remote

        self.facade = compat.ServiceClient(
            remote_service=self.service,
            training_client_factory=create,
            sampling_provider=self.sampling,
            tokenizer_factory=lambda model: f"tokenizer:{model}",
            target_modules_resolver=lambda _model: ("q_proj", "up_proj"),
            run_name_factory=lambda: "compat-run",
        )
        self.training = self.facade.create_lora_training_client(
            "Qwen/test",
            rank=8,
            seed=123,
            train_unembed=False,
        )

    def close(self) -> None:
        self.facade.close()


@pytest.fixture
def clients() -> Any:
    value = Clients()
    try:
        yield value
    finally:
        value.close()


def _tensor(
    data: list[int] | list[float], dtype: tinker.TensorDtype, shape: list[int]
) -> tinker.TensorData:
    return tinker.TensorData(data=data, dtype=dtype, shape=shape)


def _ce_datum(*, multi_target: bool = False) -> tinker.Datum:
    target = (
        _tensor([21, 22, 23, 24], "int64", [2, 2])
        if multi_target
        else _tensor([21, 22], "int64", [2])
    )
    weights = (
        _tensor([0.25, 1.5, 0.0, 2.0], "float32", [2, 2])
        if multi_target
        else _tensor([0.25, 1.5], "float32", [2])
    )
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([11, 12]),
        loss_fn_inputs={"target_tokens": target, "weights": weights},
    )


def _rl_datum() -> tinker.Datum:
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([11, 12]),
        loss_fn_inputs={
            "target_tokens": _tensor([21, 22], "int64", [2]),
            "logprobs": _tensor([-1.0, -2.0], "float32", [2]),
            "advantages": _tensor([0.5, -0.25], "float32", [2]),
        },
    )


def test_public_types_and_exact_tokenized_conversion() -> None:
    assert compat.Datum is tinker.Datum
    assert compat.TensorData is tinker.TensorData
    assert compat.types is tinker.types
    assert issubclass(compat.UnsupportedCapabilityError, tinker.TinkerError)

    converted = compat.to_tokenized_datum(_ce_datum(multi_target=True), "cross_entropy")
    assert converted.input_tokens == (11, 12)
    assert converted.target_tokens == ((21, 22), (23, 24))
    assert converted.weights == ((0.25, 1.5), (0.0, 2.0))
    assert converted.logprobs is None
    assert converted.advantages is None

    for loss in ("importance_sampling", "ppo", "cispo"):
        with pytest.raises(
            compat.UnsupportedCapabilityError,
            match="Tinker Datum cannot provide the complete ART policy spans",
        ):
            compat.to_tokenized_datum(_rl_datum(), loss)


def test_forward_loss_lowering_future_and_order(clients: Clients) -> None:
    forward = clients.training.forward([_ce_datum(multi_target=True)], "cross_entropy")
    assert isinstance(forward, tinker.APIFuture)
    output = forward.result(timeout=2)
    assert output.loss_fn_outputs[0]["logprobs"].shape == [2, 2]
    assert output.loss_fn_outputs[0]["logprobs"].data == pytest.approx(
        [-0.1, -0.2, -0.1, -0.2]
    )
    request = clients.remote.requests[-1]
    assert isinstance(request, ForwardRequest)
    assert request.loss.name == "cross_entropy"
    assert request.loss.normalize_advantages is False

    request_count = len(clients.remote.requests)
    for loss, config in (
        ("importance_sampling", None),
        ("ppo", {"clip_low_threshold": 0.7, "clip_high_threshold": 1.3}),
        ("cispo", {"clip_low_threshold": 0.1, "clip_high_threshold": 2.0}),
    ):
        with pytest.raises(
            compat.UnsupportedCapabilityError,
            match="Tinker Datum cannot provide the complete ART policy spans",
        ):
            clients.training.forward_backward([_rl_datum()], loss, config)
    assert len(clients.remote.requests) == request_count

    clients.remote.delay_next_forward_backward = True
    backward = clients.training.forward_backward([_ce_datum()], "cross_entropy")
    optimizer = clients.training.optim_step(
        tinker.AdamParams(
            learning_rate=2e-4,
            beta1=0.8,
            beta2=0.9,
            eps=1e-8,
            weight_decay=0.01,
        )
    )
    assert backward.result(timeout=2).metrics["loss"] == 1.25
    assert optimizer.result(timeout=2).metrics["learning_rate"] == 2e-4
    assert forward.future().done()
    sequence_ids = [request.sequence_id for request in clients.remote.requests]
    assert sequence_ids == list(range(len(sequence_ids)))


@pytest.mark.asyncio
async def test_async_future_access(clients: Clients) -> None:
    future = await clients.training.forward_async([_ce_datum()], "cross_entropy")
    output = await future.result_async(timeout=2)
    assert output.loss_fn_outputs[0]["logprobs"].shape == [2]
    path = "tinker://run-1/weights/checkpoint"
    await clients.facade.create_training_client_from_state_async(path)
    await clients.facade.create_training_client_from_state_with_optimizer_async(path)
    assert [
        request.restore_optimizer for request in clients.service.created_requests[-2:]
    ] == [False, True]


def test_optimizer_checkpoint_and_sampling_methods(clients: Clients) -> None:
    clients.training.optim_step(
        tinker.AdamParams(learning_rate=1e-4, grad_clip_norm=1.0)
    ).result()
    optimizer_request = clients.remote.requests[-1]
    assert isinstance(optimizer_request, OptimStepRequest)
    assert optimizer_request.optimizer.grad_clip_norm == 1.0

    saved_path = clients.training.save_state("checkpoint", 60, True).result().path
    assert saved_path == "tinker://run-1/weights/checkpoint"
    assert clients.training.load_state(saved_path).result().path == saved_path
    assert (
        clients.training.load_state_with_optimizer(saved_path).result().path
        == saved_path
    )
    load_requests = [
        request
        for request in clients.remote.requests
        if isinstance(request, LoadStateRequest)
    ]
    assert [request.restore_optimizer for request in load_requests] == [False, True]
    assert [request.checkpoint for request in load_requests] == ["checkpoint"] * 2

    clients.facade.create_training_client_from_state(
        saved_path, user_metadata={"resume": "weights"}
    )
    weights_request = clients.service.created_requests[-1]
    assert weights_request.checkpoint == "checkpoint"
    assert weights_request.restore_optimizer is False
    assert weights_request.spec.base_model == "Qwen/test"
    assert weights_request.spec.adapter.rank == 8
    assert weights_request.spec.metadata == {"resume": "weights"}

    clients.facade.create_training_client_from_state_with_optimizer(saved_path)
    optimizer_request = clients.service.created_requests[-1]
    assert optimizer_request.checkpoint == "checkpoint"
    assert optimizer_request.restore_optimizer is True

    sampler_path = clients.training.save_weights_for_sampler("sampler").result().path
    assert sampler_path == "tinker://run-1/sampler_weights/sampler"
    sampler = clients.training.create_sampling_client(sampler_path)
    assert sampler.get_base_model() == "Qwen/test"
    assert sampler.get_tokenizer() == "tokenizer:Qwen/test"
    response = sampler.sample(
        tinker.ModelInput.from_ints([1, 2]),
        2,
        tinker.SamplingParams(max_tokens=1),
    ).result(timeout=2)
    assert len(response.sequences) == 2
    assert sampler.compute_logprobs(tinker.ModelInput.from_ints([1, 2])).result() == [
        None,
        -0.5,
    ]
    assert clients.sampling.targets[-1].lora == "lora://sampler"


def test_explicit_rejections(clients: Clients) -> None:
    with pytest.raises(compat.UnsupportedCapabilityError, match="custom loss"):
        compat.to_tokenized_datum(_ce_datum(), lambda *_args: None)
    with pytest.raises(compat.UnsupportedCapabilityError, match="custom"):
        clients.training.forward_backward_custom(
            [_ce_datum()], lambda _data, _logprobs: (None, {})
        )
    with pytest.raises(compat.UnsupportedCapabilityError, match="DRO"):
        compat.to_tokenized_datum(_ce_datum(), "dro")

    image_input = tinker.ModelInput(
        chunks=[
            tinker.types.ImageAssetPointerChunk(
                format="png", location="https://example.invalid/image.png"
            )
        ]
    )
    image_datum = tinker.Datum(
        model_input=image_input,
        loss_fn_inputs=_ce_datum().loss_fn_inputs,
    )
    with pytest.raises(compat.UnsupportedCapabilityError, match="text-only"):
        compat.to_tokenized_datum(image_datum, "cross_entropy")

    with pytest.raises(compat.UnsupportedCapabilityError, match="unembedding"):
        clients.facade.create_lora_training_client("Qwen/test")
    with pytest.raises(compat.UnsupportedCapabilityError, match="Tinker auth"):
        clients.facade.create_training_client_from_state(
            "tinker://run-1/weights/checkpoint", weights_access_token="token"
        )
    with pytest.raises(ValueError, match="/weights/"):
        clients.facade.create_training_client_from_state(
            "tinker://run-1/sampler_weights/checkpoint"
        )
    with pytest.raises(compat.UnsupportedCapabilityError, match="cross-run"):
        clients.training.load_state("tinker://run-2/weights/checkpoint")


def test_remote_training_api_key_and_clean_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    native_service = FakeService()

    def create_native_service(**kwargs: Any) -> FakeService:
        captured.update(kwargs)
        return native_service

    monkeypatch.setattr(
        compat_client, "RemoteTrainingServiceClient", create_native_service
    )
    monkeypatch.delenv("REMOTE_TRAINING_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "ignored-wandb-key")
    with pytest.raises(ValueError, match="REMOTE_TRAINING_API_KEY"):
        compat.ServiceClient(base_url="https://training.example.test/v1")

    monkeypatch.setenv("REMOTE_TRAINING_API_KEY", "remote-training-key")
    facade = compat.ServiceClient(base_url="https://training.wandb.test/v1")
    facade.close()
    facade.close()
    assert captured["api_key"] == "remote-training-key"
    assert native_service.closed

    clients = Clients()
    clients.close()
    clients.close()
    assert clients.remote.closed
    assert clients.service.closed


def test_training_client_close_retries_remote_shutdown() -> None:
    clients = Clients()
    clients.remote.shutdown_failures = 1
    try:
        with pytest.raises(RuntimeError, match="injected remote shutdown failure"):
            clients.training.close()
        with pytest.raises(RuntimeError, match="TrainingClient is closed"):
            clients.training.optim_step(tinker.AdamParams(learning_rate=1e-4))
        clients.training.close()
        assert clients.remote.shutdown_calls == 2
        assert clients.remote.closed
    finally:
        clients.facade.close()


def test_service_client_close_retries_remote_shutdown() -> None:
    clients = Clients()
    clients.remote.shutdown_failures = 1
    with pytest.raises(BaseExceptionGroup, match="Tinker compatibility close failed"):
        clients.facade.close()
    with pytest.raises(RuntimeError, match="ServiceClient is closed"):
        clients.facade.create_lora_training_client("Qwen/test", train_unembed=False)
    assert not clients.service.closed
    clients.facade.close()
    assert clients.remote.shutdown_calls == 2
    assert clients.remote.closed
    assert clients.service.closed


def test_service_client_concurrent_close_shares_one_attempt() -> None:
    clients = Clients()
    clients.remote.shutdown_delay_s = 0.05
    failures: list[BaseException] = []

    def close() -> None:
        try:
            clients.facade.close()
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=close) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures
    assert clients.remote.shutdown_calls == 1
