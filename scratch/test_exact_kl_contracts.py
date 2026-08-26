from pydantic import ValidationError
import pytest

from art.megatron.training.commands import (
    experimental_train_config,
    forward_backward_config,
)
from art.serverless.contracts import (
    TOKENIZED_DATA_FORMAT,
    RemoteForwardRequest,
    RemoteTokenizedBatchRef,
    TrainingDataRef,
)
from art.training.contracts import (
    MAX_CHECKPOINT_REFERENCE_LENGTH,
    MAX_CONTROL_IDENTIFIER_LENGTH,
    CheckpointRef,
    ForwardBackwardRequest,
)


def _datum() -> dict[str, object]:
    return {
        "input_tokens": (1,),
        "target_tokens": ((2,),),
        "logprobs": ((0.0,),),
        "advantages": ((1.0,),),
        "policy_spans": (
            {"start_position": 0, "end_position": 1, "policy_version": 0},
        ),
    }


def _loss(*, run_id: str = "run", coefficient: object = 0.25) -> dict[str, object]:
    return {
        "name": "ppo",
        "values": {
            "kl_penalty_coef": coefficient,
            "kl_penalty_source": "sample",
            "kl_ref_adapter_path": "/reference",
            "kl_ref_checkpoint_id": "checkpoint",
        },
        "reference_checkpoint": {
            "run_id": run_id,
            "learner_version": 3,
            "checkpoint_id": "checkpoint",
        },
    }


def _local_request(*, run_id: str = "run", coefficient: object = 0.25):
    return ForwardBackwardRequest.model_validate(
        {
            "run_id": "run",
            "request_id": "request",
            "sequence_id": 0,
            "batch": {"kind": "tokenized", "datums": (_datum(),)},
            "loss": _loss(run_id=run_id, coefficient=coefficient),
        }
    )


def _remote_request(*, run_id: str = "run", coefficient: object = 0.25):
    data = TrainingDataRef(
        object_id="0" * 64,
        sha256="1" * 64,
        byte_count=1,
        format=TOKENIZED_DATA_FORMAT,
    )
    return RemoteForwardRequest.model_validate(
        {
            "run_id": "run",
            "request_id": "request",
            "sequence_id": 0,
            "batch": RemoteTokenizedBatchRef(data=data),
            "loss": {
                **_loss(run_id=run_id, coefficient=coefficient),
                "values": {
                    "kl_penalty_coef": coefficient,
                    "kl_penalty_source": "sample",
                },
            },
        }
    )


def test_checkpoint_reference_enforces_public_string_bounds() -> None:
    with pytest.raises(ValidationError):
        CheckpointRef(
            run_id="r" * (MAX_CONTROL_IDENTIFIER_LENGTH + 1),
            learner_version=0,
            checkpoint_id="checkpoint",
        )
    with pytest.raises(ValidationError):
        CheckpointRef(
            run_id="run",
            learner_version=0,
            checkpoint_id="c" * (MAX_CHECKPOINT_REFERENCE_LENGTH + 1),
        )


@pytest.mark.parametrize("factory", [_local_request, _remote_request])
def test_reference_is_bound_to_the_command_run(factory) -> None:
    with pytest.raises(ValidationError, match="command run"):
        factory(run_id="other-run")


@pytest.mark.parametrize("factory", [_local_request, _remote_request])
def test_boolean_kl_coefficient_is_not_numeric(factory) -> None:
    with pytest.raises((TypeError, ValidationError), match="must be numeric"):
        factory(coefficient=True)


def test_tokenized_exact_kl_reaches_both_megatron_configs() -> None:
    request = _local_request()

    config = forward_backward_config(request)
    experimental = experimental_train_config(request)

    assert config.kl_penalty_coef == 0.25
    assert config.kl_penalty_source == "sample"
    assert experimental.kl_penalty_coef == 0.25
    assert experimental.kl_penalty_source == "sample"
    assert experimental.kl_ref_adapter_path == "/reference"
    assert experimental.kl_ref_checkpoint_id == "checkpoint"
