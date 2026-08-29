from typing import Any, cast

import pytest

from art.serverless.client import (
    NativeOperationStatus,
    NativeTrainingOperation,
    NativeTrainingRun,
    TrainingRuns,
)
from art.serverless.native_training import RemoteTrainingClient
from art.training import (
    AdapterSpec,
    ForwardRequest,
    LossConfig,
    PackedInputCaptureRef,
    PackingOutcome,
    TrainingRunSpec,
)


class _TrainingRuns:
    def __init__(self) -> None:
        self.submissions = 0

    async def resolve(self, **kwargs: Any) -> NativeTrainingRun:
        spec = kwargs["spec"]
        return NativeTrainingRun(
            run_id="run-native",
            run_name=kwargs["run_name"],
            spec={
                "base_model": spec.base_model,
                "dtype": spec.dtype,
                "lora_rank": spec.adapter.rank,
                "lora_target_modules": list(spec.adapter.target_modules),
                "optimizer": "adamw",
            },
            status="open",
            next_sequence_id=0,
            projected_learner_version=0,
            committed_learner_version=0,
        )

    async def submit(self, request: ForwardRequest) -> NativeTrainingOperation:
        self.submissions += 1
        return _operation(request, status="admitted")

    async def operation(
        self, run_id: str, operation_id: str
    ) -> NativeTrainingOperation:
        assert run_id == "run-native"
        assert operation_id == "operation-native"
        request = _request()
        return _operation(
            request,
            status="succeeded",
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

    async def cancel(self, run_id: str, operation_id: str) -> NativeTrainingOperation:
        raise AssertionError((run_id, operation_id))

    async def close(self, run_id: str) -> NativeTrainingRun:
        run = await self.resolve()
        assert run_id == run.run_id
        return run.model_copy(update={"status": "closing"})


def _request() -> ForwardRequest:
    return ForwardRequest(
        run_id="run-native",
        request_id="request-native",
        sequence_id=0,
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
    status: NativeOperationStatus,
    result: dict[str, Any] | None = None,
) -> NativeTrainingOperation:
    return NativeTrainingOperation(
        operation_id="operation-native",
        run_id=request.run_id,
        request_id=request.request_id,
        sequence_id=request.sequence_id,
        kind="forward",
        status=status,
        learner_parent_version=0,
        reserved_output_learner_version=None,
        contributing_operation_ids=(),
        cancel_requested=False,
        event_cursor=1,
        result=result,
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
    assert service.submissions == 1
