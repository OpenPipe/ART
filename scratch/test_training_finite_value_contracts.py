from collections.abc import Callable, Mapping
import math
import sys

from pydantic import BaseModel, ValidationError
import pytest

from art.distributed.trajectory_store import TrajectoryGroupBundle
from art.training.contracts import (
    AdamConfig,
    ForwardBackwardRequest,
    LossConfig,
    OptimStepRequest,
    RlTrajectoryBatch,
)
from art.training.tokenized import tokenized_clip_bounds

NONFINITE_VALUES = (float("nan"), float("inf"), float("-inf"))
ADAM_VALUES = {
    "learning_rate": 1e-3,
    "beta1": 0.9,
    "beta2": 0.99,
    "eps": 1e-13,
    "weight_decay": 0.1,
    "grad_clip_norm": 0.1,
}
LOSS_NUMERIC_FIELDS = (
    "advantage_balance",
    "epsilon",
    "epsilon_high",
    "grad_accumulation_sequences",
    "kimi_k2_tau",
    "kl_penalty_coef",
    "kl_penalty_reference_step",
    "kl_penalty_step_lag",
    "logprob_calculation_chunk_size",
    "max_negative_advantage_importance_sampling_weight",
    "num_trajectories_learning_rate_multiplier_power",
    "packed_sequence_length",
    "truncated_importance_sampling",
)


def _validate_then_execute(
    model: type[BaseModel],
    payload: object,
    executor: Callable[[BaseModel], None],
) -> None:
    executor(model.model_validate(payload))


def _rl_batch() -> RlTrajectoryBatch:
    return RlTrajectoryBatch(
        groups=(TrajectoryGroupBundle(header=b"\x81\xff", records=(b"\x00\xfe",)),),
        min_source_version=0,
        max_source_version=0,
    )


def _rl_request_payload(values: Mapping[str, object]) -> dict[str, object]:
    return {
        "run_id": "run",
        "request_id": "forward_backward",
        "sequence_id": 0,
        "batch": _rl_batch(),
        "loss": {"name": "cispo", "values": values},
    }


def _tokenized_request_payload(
    *,
    loss: str,
    datum: dict[str, object],
    values: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "run_id": "run",
        "request_id": "forward_backward",
        "sequence_id": 0,
        "batch": {"kind": "tokenized", "datums": (datum,)},
        "loss": {"name": loss, "values": values or {}},
    }


def _tokenized_rl_datum() -> dict[str, object]:
    return {
        "input_tokens": (1,),
        "target_tokens": ((2,),),
        "logprobs": ((0.0,),),
        "advantages": ((1.0,),),
        "policy_spans": (
            {"start_position": 0, "end_position": 1, "policy_version": 0},
        ),
    }


@pytest.mark.parametrize("field", ADAM_VALUES)
@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_nonfinite_optimizer_fields_fail_before_executor(
    field: str, value: float
) -> None:
    optimizer = {**ADAM_VALUES, field: value}
    executor_calls: list[BaseModel] = []
    with pytest.raises(ValidationError):
        _validate_then_execute(
            OptimStepRequest,
            {
                "run_id": "run",
                "request_id": "optim",
                "sequence_id": 0,
                "optimizer": optimizer,
            },
            executor_calls.append,
        )
    assert not executor_calls


@pytest.mark.parametrize("field", LOSS_NUMERIC_FIELDS)
@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_nonfinite_loss_fields_fail_before_executor(field: str, value: float) -> None:
    executor_calls: list[BaseModel] = []
    with pytest.raises(ValidationError):
        _validate_then_execute(
            ForwardBackwardRequest,
            _rl_request_payload({field: value}),
            executor_calls.append,
        )
    assert not executor_calls


@pytest.mark.parametrize("field", ("clip_low_threshold", "clip_high_threshold"))
@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_nonfinite_clip_fields_fail_before_executor(field: str, value: float) -> None:
    executor_calls: list[BaseModel] = []
    with pytest.raises(ValidationError):
        _validate_then_execute(
            ForwardBackwardRequest,
            _tokenized_request_payload(
                loss="ppo",
                datum=_tokenized_rl_datum(),
                values={field: value},
            ),
            executor_calls.append,
        )
    assert not executor_calls


@pytest.mark.parametrize("field", ("clip_low_threshold", "clip_high_threshold"))
@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_tokenized_clip_helper_rejects_nonfinite_values(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be finite"):
        tokenized_clip_bounds("ppo", {field: value})


@pytest.mark.parametrize(
    ("field", "loss"),
    (
        ("weights", "cross_entropy"),
        ("logprobs", "importance_sampling"),
        ("advantages", "importance_sampling"),
    ),
)
@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_nonfinite_tokenized_loss_tensors_fail_before_executor(
    field: str, loss: str, value: float
) -> None:
    datum = (
        {
            "input_tokens": (1,),
            "target_tokens": ((2,),),
            "weights": ((value,),),
        }
        if field == "weights"
        else _tokenized_rl_datum()
    )
    if field != "weights":
        datum[field] = ((value,),)
    executor_calls: list[BaseModel] = []
    with pytest.raises(ValidationError):
        _validate_then_execute(
            ForwardBackwardRequest,
            _tokenized_request_payload(loss=loss, datum=datum),
            executor_calls.append,
        )
    assert not executor_calls


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("learning_rate", 0.0),
        ("learning_rate", sys.float_info.max),
        ("beta1", 0.0),
        ("beta1", math.nextafter(1.0, 0.0)),
        ("beta2", 0.0),
        ("beta2", math.nextafter(1.0, 0.0)),
        ("eps", math.ulp(0.0)),
        ("eps", sys.float_info.max),
        ("weight_decay", 0.0),
        ("weight_decay", sys.float_info.max),
        ("grad_clip_norm", 0.0),
        ("grad_clip_norm", sys.float_info.max),
    ),
)
def test_finite_optimizer_edges_remain_valid(field: str, value: float) -> None:
    optimizer = AdamConfig.model_validate({**ADAM_VALUES, field: value})
    assert getattr(optimizer, field) == value


def test_optional_loss_bounds_preserve_none() -> None:
    values = {
        "epsilon": 0.0,
        "epsilon_high": None,
        "max_negative_advantage_importance_sampling_weight": None,
        "truncated_importance_sampling": None,
    }
    request = ForwardBackwardRequest.model_validate(_rl_request_payload(values))
    assert request.loss.values == values


def test_finite_tokenized_edges_remain_valid() -> None:
    request = ForwardBackwardRequest.model_validate(
        _tokenized_request_payload(
            loss="ppo",
            datum={
                **_tokenized_rl_datum(),
                "logprobs": ((-sys.float_info.max,),),
                "advantages": ((sys.float_info.max,),),
            },
            values={"clip_low_threshold": 0.0, "clip_high_threshold": 0.0},
        )
    )
    assert request.loss.values == {
        "clip_low_threshold": 0.0,
        "clip_high_threshold": 0.0,
    }
