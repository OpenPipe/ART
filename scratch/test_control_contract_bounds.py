from pydantic import ValidationError
import pytest

from art.serverless.contracts import (
    AdapterSpec,
    ApplyCheckpointRetentionRequest,
    CheckpointRevision,
    CreateTrainingRunRequest,
    TrainingRunSpec,
)
from art.training.contracts import (
    AdamConfig,
    LoadStateRequest,
    OptimStepRequest,
    SamplerPublication,
    SaveStateRequest,
)


def spec(**updates: object) -> TrainingRunSpec:
    values = {
        "run_name": "run",
        "base_model": "Qwen/Qwen3.5-35B-A3B",
        "adapter": AdapterSpec(rank=8, target_modules=("q_proj",)),
    }
    values.update(updates)
    return TrainingRunSpec.model_validate(values)


def test_create_run_accepts_exact_boundaries() -> None:
    request = CreateTrainingRunRequest(
        spec=spec(
            base_model="m" * 512,
            adapter=AdapterSpec(
                rank=8,
                target_modules=tuple("x" * 255 for _ in range(256)),
            ),
            metadata={str(index): "v" * 4096 for index in range(64)},
        ),
        checkpoint="c" * 2048,
    )
    assert len(request.spec.adapter.target_modules) == 256


@pytest.mark.parametrize(
    "factory",
    [
        lambda: spec(base_model="m" * 513),
        lambda: spec(
            adapter=AdapterSpec(rank=8, target_modules=tuple("x" for _ in range(257)))
        ),
        lambda: spec(adapter=AdapterSpec(rank=8, target_modules=("x" * 256,))),
        lambda: spec(metadata={str(index): "" for index in range(65)}),
        lambda: spec(metadata={"k" * 129: ""}),
        lambda: spec(metadata={"key": "v" * 4097}),
        lambda: CreateTrainingRunRequest(spec=spec(), checkpoint="c" * 2049),
    ],
)
def test_create_run_rejects_oversized_fields(factory) -> None:
    with pytest.raises(ValidationError):
        factory()


def test_control_commands_reject_oversized_identifiers() -> None:
    with pytest.raises(ValidationError):
        OptimStepRequest(
            run_id="r" * 256,
            request_id="request",
            sequence_id=0,
            optimizer=AdamConfig(learning_rate=1e-5),
        )
    with pytest.raises(ValidationError):
        SaveStateRequest(
            run_id="run",
            request_id="request",
            sequence_id=0,
            checkpoint_name="c" * 256,
        )
    with pytest.raises(ValidationError):
        LoadStateRequest(
            run_id="run",
            request_id="request",
            sequence_id=0,
            checkpoint="c" * 2049,
        )
    with pytest.raises(ValidationError):
        SamplerPublication(mode="versioned_lora", model_alias="a" * 256)


def test_checkpoint_retention_cardinality_is_bounded() -> None:
    observed = tuple(
        CheckpointRevision(checkpoint_id=str(index), revision=1) for index in range(513)
    )
    with pytest.raises(ValidationError):
        ApplyCheckpointRetentionRequest(observed=observed)
