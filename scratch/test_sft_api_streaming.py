from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal

import pytest

from art.megatron.backend import MegatronBackend
from art.metrics_taxonomy import TRAIN_GRADIENT_STEPS_KEY, average_metric_samples
from art.serverless.backend import ServerlessBackend
from art.training.contracts import (
    CheckpointRef,
    ForwardBackwardResult,
    OperationRef,
    OptimStepResult,
    PackingOutcome,
    SamplerPublication,
    SamplerWeightsResult,
    SaveStateResult,
    SaveWeightsForSamplerRequest,
)
from art.trajectories import Trajectory
from art.types import TrainSFTConfig


class _Operation:
    def __init__(self, ref: OperationRef, result: Any, disposition: str | None = None):
        self.ref = ref
        self._result = result
        self._disposition = disposition

    async def result(self):
        return self._result

    async def cancel(self) -> None:
        return None

    async def gradient_disposition(self):
        if self._disposition is None:
            raise TypeError("gradient disposition is only available for F/B")
        return self._disposition


class _Client:
    def __init__(self) -> None:
        self.run_id = "run"
        self.next_sequence_id = 0
        self.projected_learner_version = 0
        self._pending_forward_id: str | None = None

    def _ref(self, request, kind: str, *, transition: bool = False) -> OperationRef:
        assert request.sequence_id == self.next_sequence_id
        parent = self.projected_learner_version
        reserved = parent + 1 if transition else None
        ref = OperationRef(
            run_id=self.run_id,
            operation_id=f"{kind}-{request.sequence_id}",
            sequence_id=request.sequence_id,
            learner_parent_version=parent,
            reserved_output_learner_version=reserved,
            kind=kind,
        )
        self.next_sequence_id += 1
        if reserved is not None:
            self.projected_learner_version = reserved
        return ref

    async def forward_backward(self, request):
        ref = self._ref(request, "forward_backward")
        self._pending_forward_id = ref.operation_id
        count = len(request.batch.trajectories)
        return _Operation(
            ref,
            ForwardBackwardResult(
                operation_id=ref.operation_id,
                packing=PackingOutcome(
                    packed_sequence_length=16,
                    packed_sequences=1,
                    target_packed_sequences=1,
                    nominal_capacity_tokens=16,
                    physical_tokens=16,
                    non_padding_tokens=count,
                    loss_bearing_tokens=count,
                    trainable_assistant_tokens=count,
                    policy_token_counts=None,
                    group_shapes=(),
                ),
                loss_fn_outputs=(),
                metrics={
                    "time/forward_backward_s": 0.2,
                    TRAIN_GRADIENT_STEPS_KEY: 1.0,
                },
            ),
            "contributes",
        )

    async def optim_step(self, request):
        ref = self._ref(request, "optim_step", transition=True)
        assert self._pending_forward_id is not None
        result = OptimStepResult(
            operation_id=ref.operation_id,
            contributing_forward_backward_operation_ids=(self._pending_forward_id,),
            metrics={"time/optimizer_step_s": 0.1},
        )
        self._pending_forward_id = None
        return _Operation(ref, result)

    async def save_weights_for_sampler(self, request):
        ref = self._ref(request, "save_sampler")
        return _Operation(
            ref,
            SamplerWeightsResult(
                operation_id=ref.operation_id,
                checkpoint=CheckpointRef(
                    run_id=self.run_id,
                    learner_version=self.projected_learner_version,
                    checkpoint_id=request.checkpoint_name,
                ),
                lora="lora",
                training_session_id="session",
                generation_id=f"generation-{self.projected_learner_version}",
                lora_bytes=1,
                publication_metrics={"publication/fake": 1.0},
            ),
        )

    async def save_state(self, request):
        ref = self._ref(request, "save_state")
        return _Operation(
            ref,
            SaveStateResult(
                operation_id=ref.operation_id,
                checkpoint=CheckpointRef(
                    run_id=self.run_id,
                    learner_version=self.projected_learner_version,
                    checkpoint_id=request.checkpoint_name,
                ),
                lora="lora",
                training_session_id="session",
                generation_id=f"generation-{self.projected_learner_version}",
                lora_bytes=1,
                optimizer_state="optimizer",
                optimizer_bytes=1,
                metrics={"state/fake": 1.0},
            ),
        )


class _LocalBackend:
    def __init__(self) -> None:
        self.client = _Client()
        self.service = SimpleNamespace(rollout_weight_update_mode="in_flight_lora")

    def _default_sft_batch_size(self) -> int:
        return 2

    async def training_client(self, _model):
        return self.client

    async def _get_service(self, _model):
        return self.service


class _ServerlessBackend:
    def __init__(self) -> None:
        self.client = _Client()

    def _raise_background_failures(self) -> None:
        return None

    async def training_client(self, _model):
        return self.client

    async def _start_sampler_publication(self, model, client, step, sequence):
        operation = await client.save_weights_for_sampler(
            SaveWeightsForSamplerRequest(
                run_id=client.run_id,
                request_id=f"sampler-{sequence}",
                sequence_id=sequence,
                checkpoint_name=f"step-{step}",
                publication=SamplerPublication(mode="none"),
            )
        )
        return operation, SimpleNamespace(model=model, step=step)

    async def _complete_sampler_publication(self, _model, _step, operation, _pending):
        return (await operation.result()).publication_metrics

    async def _fail_sampler_result(self, *_args) -> None:
        raise AssertionError("sampler publication unexpectedly failed")


class _TrajectoryStream:
    def __init__(self, count: int) -> None:
        self.count = count
        self.consumed = 0

    def __iter__(self):
        for _ in range(self.count):
            self.consumed += 1
            yield Trajectory()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_kind", ["local", "serverless"])
async def test_sft_batches_and_rows_stream_with_bounded_state(
    backend_kind: Literal["local", "serverless"],
) -> None:
    source = _TrajectoryStream(7)
    model = SimpleNamespace(name="model")
    config = TrainSFTConfig(batch_size=2, learning_rate=[1e-3] * 4)
    if backend_kind == "local":
        backend = _LocalBackend()
        stream = MegatronBackend._train_sft(backend, model, source, config, {})
    else:
        backend = _ServerlessBackend()
        stream = ServerlessBackend._train_sft(backend, model, source, config, {})

    rows = [await anext(stream)]
    assert source.consumed == 4
    rows.extend([row async for row in stream])

    assert [row["data/step_num_trajectories"] for row in rows] == [2.0, 2.0, 2.0, 1.0]
    assert all(TRAIN_GRADIENT_STEPS_KEY not in row for row in rows[:-1])
    assert rows[-1][TRAIN_GRADIENT_STEPS_KEY] == 4.0
    assert rows[-1]["publication/fake"] == 1.0
    assert rows[-1]["state/fake"] == 1.0
    assert average_metric_samples(rows)[TRAIN_GRADIENT_STEPS_KEY] == 4.0
    assert source.consumed == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_kind", ["local", "serverless"])
async def test_sft_validates_sized_rate_schedule_before_admission(
    backend_kind: Literal["local", "serverless"],
) -> None:
    backend: Any = _LocalBackend() if backend_kind == "local" else _ServerlessBackend()
    method = (
        MegatronBackend._train_sft
        if backend_kind == "local"
        else ServerlessBackend._train_sft
    )
    stream = method(
        backend,
        SimpleNamespace(name="model"),
        [Trajectory()] * 3,
        TrainSFTConfig(batch_size=2, learning_rate=[1e-3]),
        {},
    )
    with pytest.raises(ValueError, match="schedule must match batch count"):
        await anext(stream)
    assert backend.client.next_sequence_id == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_kind", ["local", "serverless"])
async def test_empty_sft_stream_ignores_unused_rate_schedule(
    backend_kind: Literal["local", "serverless"],
) -> None:
    backend: Any = _LocalBackend() if backend_kind == "local" else _ServerlessBackend()
    method = (
        MegatronBackend._train_sft
        if backend_kind == "local"
        else ServerlessBackend._train_sft
    )
    stream = method(
        backend,
        SimpleNamespace(name="model"),
        [],
        TrainSFTConfig(batch_size=2, learning_rate=[1e-3]),
        {},
    )
    assert [row async for row in stream] == []
    assert backend.client.next_sequence_id == 0
