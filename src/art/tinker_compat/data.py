from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import cast

import tinker

from art.training.contracts import (
    ForwardBackwardResult,
    ForwardResult,
    TokenizedTrainingBatch,
)
from art.training.tokenized import TokenizedDatum, TokenizedLossName

from .errors import UnsupportedCapabilityError

SUPPORTED_LOSSES = frozenset({"cross_entropy", "importance_sampling", "ppo", "cispo"})


def validate_loss(loss_fn: object) -> TokenizedLossName:
    if callable(loss_fn):
        raise UnsupportedCapabilityError(
            "custom loss functions are not supported; use a named server loss"
        )
    if loss_fn == "dro":
        raise UnsupportedCapabilityError("DRO is not supported by Remote Training")
    if loss_fn not in SUPPORTED_LOSSES:
        raise ValueError(
            f"unsupported loss {loss_fn!r}; expected one of {sorted(SUPPORTED_LOSSES)}"
        )
    return cast(TokenizedLossName, loss_fn)


def model_input_tokens(model_input: tinker.ModelInput) -> tuple[int, ...]:
    unsupported = [
        type(chunk).__name__
        for chunk in model_input.chunks
        if not isinstance(chunk, tinker.EncodedTextChunk)
    ]
    if unsupported:
        raise UnsupportedCapabilityError(
            "Remote Training's Tinker profile is text-only; unsupported input "
            f"chunks: {unsupported}"
        )
    return tuple(model_input.to_ints())


def to_tokenized_datum(datum: tinker.Datum, loss_fn: object) -> TokenizedDatum:
    loss = validate_loss(loss_fn)
    expected = (
        {"target_tokens", "weights"}
        if loss == "cross_entropy"
        else {"target_tokens", "logprobs", "advantages"}
    )
    found = set(datum.loss_fn_inputs)
    if found != expected:
        raise ValueError(
            f"{loss} requires exactly {sorted(expected)}, got {sorted(found)}"
        )
    target_tokens, _ = _matrix(datum.loss_fn_inputs["target_tokens"], integer=True)
    values: dict[str, object] = {
        "input_tokens": model_input_tokens(datum.model_input),
        "target_tokens": target_tokens,
    }
    for name in ("weights", "logprobs", "advantages"):
        if name in datum.loss_fn_inputs:
            values[name] = _matrix(datum.loss_fn_inputs[name], integer=False)[0]
    return TokenizedDatum.model_validate(values)


def to_tokenized_batch(
    data: Sequence[tinker.Datum], loss_fn: object
) -> tuple[TokenizedTrainingBatch, tuple[tuple[int, ...], ...]]:
    loss = validate_loss(loss_fn)
    if not data:
        raise ValueError("No data provided")
    converted = tuple(to_tokenized_datum(datum, loss) for datum in data)
    shapes = tuple(
        _matrix(datum.loss_fn_inputs["target_tokens"], integer=True)[1]
        for datum in data
    )
    return TokenizedTrainingBatch(datums=converted), shapes


def to_tinker_forward_output(
    result: ForwardResult | ForwardBackwardResult,
    target_shapes: Sequence[tuple[int, ...]],
) -> tinker.ForwardBackwardOutput:
    if len(result.loss_fn_outputs) != len(target_shapes):
        raise RuntimeError(
            "Remote Training changed datum output cardinality: "
            f"{len(result.loss_fn_outputs)} != {len(target_shapes)}"
        )
    outputs = []
    for output, shape in zip(result.loss_fn_outputs, target_shapes, strict=True):
        values = _flatten(output.token_logprobs)
        if len(values) != prod(shape):
            raise RuntimeError(
                "Remote Training changed selected-logprob shape: "
                f"{len(values)} values for {shape}"
            )
        outputs.append(
            {
                "logprobs": tinker.TensorData(
                    data=values,
                    dtype="float32",
                    shape=list(shape),
                )
            }
        )
    return tinker.ForwardBackwardOutput(
        loss_fn_output_type="ArrayRecord",
        loss_fn_outputs=outputs,
        metrics=dict(result.metrics),
    )


def _matrix(
    tensor: tinker.TensorData, *, integer: bool
) -> tuple[
    tuple[tuple[int, ...], ...] | tuple[tuple[float, ...], ...], tuple[int, ...]
]:
    expected_dtype = "int64" if integer else "float32"
    if tensor.dtype != expected_dtype:
        raise TypeError(f"expected {expected_dtype} TensorData, got {tensor.dtype!r}")
    array = tensor.to_numpy()
    shape = tuple(int(value) for value in array.shape)
    if array.ndim == 1:
        rows = tuple((value,) for value in array.tolist())
    elif array.ndim == 2:
        rows = tuple(tuple(row) for row in array.tolist())
    else:
        raise ValueError(f"named-loss tensors must be 1-D or 2-D, got shape {shape}")
    if integer:
        return tuple(tuple(int(value) for value in row) for row in rows), shape
    return tuple(tuple(float(value) for value in row) for row in rows), shape


def _flatten(values: object) -> list[float]:
    flattened: list[float] = []
    for value in cast(Sequence[object], values):
        if isinstance(value, (tuple, list)):
            flattened.extend(_as_float(item) for item in value)
        else:
            flattened.append(_as_float(value))
    return flattened


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected a numeric logprob, got {type(value).__name__}")
    return float(value)
