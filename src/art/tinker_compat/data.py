from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import cast

import tinker

from art.training import (
    ForwardBackwardRequest,
    ForwardBackwardResult,
    ForwardRequest,
    ForwardResult,
    LossConfig,
    TokenizedDatum,
    TokenizedTrainingBatch,
)
from art.training.tokenized import TokenizedLossName

from .errors import UnsupportedCapabilityError

SUPPORTED_LOSSES = frozenset({"cross_entropy", "importance_sampling", "cispo"})
_LOSS_CONFIG_KEYS = {
    "cross_entropy": frozenset(),
    "importance_sampling": frozenset(),
    "cispo": frozenset({"clip_low_threshold", "clip_high_threshold"}),
}


@dataclass(frozen=True, slots=True)
class TinkerForwardTranslation:
    batch: TokenizedTrainingBatch
    loss: LossConfig
    target_shapes: tuple[tuple[int, ...], ...]

    def request(
        self,
        *,
        run_id: str,
        request_id: str,
        sequence_id: int,
        backward: bool,
    ) -> ForwardRequest | ForwardBackwardRequest:
        request_type = ForwardBackwardRequest if backward else ForwardRequest
        return request_type(
            run_id=run_id,
            request_id=request_id,
            sequence_id=sequence_id,
            batch=self.batch,
            loss=self.loss,
        )


def translate_tinker_forward_input(
    forward_input: tinker.types.ForwardBackwardInput,
) -> TinkerForwardTranslation:
    """Lower pinned Tinker wire data without retokenizing or changing shapes."""

    loss = _validate_loss(forward_input.loss_fn)
    values = _validate_loss_config(loss, forward_input.loss_fn_config)
    if not forward_input.data:
        raise ValueError("No data provided")
    converted = tuple(_convert_datum(datum, loss) for datum in forward_input.data)
    return TinkerForwardTranslation(
        batch=TokenizedTrainingBatch(datums=tuple(item[0] for item in converted)),
        loss=LossConfig(name=loss, normalize_advantages=False, values=values),
        target_shapes=tuple(item[1] for item in converted),
    )


def to_tinker_forward_output(
    result: ForwardResult | ForwardBackwardResult,
    target_shapes: Sequence[tuple[int, ...]],
) -> tinker.ForwardBackwardOutput:
    if len(result.token_logprobs) != len(target_shapes):
        raise RuntimeError("ART changed Tinker datum output cardinality")
    outputs = []
    for logprobs, shape in zip(result.token_logprobs, target_shapes, strict=True):
        values = logprobs.to_values()
        if len(values) != prod(shape):
            raise RuntimeError(
                f"ART changed selected-logprob shape: {len(values)} values for {shape}"
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


def _validate_loss(loss: object) -> TokenizedLossName:
    if callable(loss):
        raise UnsupportedCapabilityError(
            "custom loss functions are not supported; use a named server loss"
        )
    if loss in {"dro", "ppo"}:
        raise UnsupportedCapabilityError(
            f"{loss} is not supported by the scoped Tinker profile"
        )
    if loss not in SUPPORTED_LOSSES:
        raise ValueError(
            f"unsupported loss {loss!r}; expected one of {sorted(SUPPORTED_LOSSES)}"
        )
    return cast(TokenizedLossName, loss)


def _validate_loss_config(
    loss: TokenizedLossName,
    config: dict[str, float] | None,
) -> dict[str, float]:
    values = dict(config or {})
    unsupported = set(values) - _LOSS_CONFIG_KEYS[loss]
    if unsupported:
        raise UnsupportedCapabilityError(
            f"{loss} settings are not supported: {sorted(unsupported)}"
        )
    return values


def _convert_datum(
    datum: tinker.Datum,
    loss: TokenizedLossName,
) -> tuple[TokenizedDatum, tuple[int, ...]]:
    unsupported = [
        type(chunk).__name__
        for chunk in datum.model_input.chunks
        if not isinstance(chunk, tinker.EncodedTextChunk)
    ]
    if unsupported:
        raise UnsupportedCapabilityError(
            f"the scoped Tinker profile is text-only; got {unsupported}"
        )
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

    target_tokens, shape = _matrix(datum.loss_fn_inputs["target_tokens"], integer=True)
    values: dict[str, object] = {
        "input_tokens": tuple(datum.model_input.to_ints()),
        "target_tokens": target_tokens,
    }
    for name in ("weights", "logprobs", "advantages"):
        if name in datum.loss_fn_inputs:
            values[name] = _matrix(datum.loss_fn_inputs[name], integer=False)[0]
    converted = TokenizedDatum.model_validate(values)
    converted.validate_for_loss(loss)
    return converted, shape


def _matrix(
    tensor: tinker.TensorData,
    *,
    integer: bool,
) -> tuple[
    tuple[tuple[int, ...], ...] | tuple[tuple[float, ...], ...],
    tuple[int, ...],
]:
    expected_dtype = "int64" if integer else "float32"
    if tensor.dtype != expected_dtype:
        raise TypeError(f"expected {expected_dtype} TensorData, got {tensor.dtype!r}")
    shape = (
        tuple(_axis(value) for value in tensor.shape)
        if tensor.shape is not None
        else (len(tensor.data),)
    )
    if len(shape) not in {1, 2}:
        raise ValueError(f"named-loss tensors must be 1-D or 2-D, got shape {shape}")
    values = _dense_values(tensor, shape)
    width = 1 if len(shape) == 1 else shape[1]
    rows = tuple(
        tuple(_number(value) for value in values[offset : offset + width])
        for offset in range(0, len(values), width)
    )
    if integer:
        return tuple(tuple(int(value) for value in row) for row in rows), shape
    return tuple(tuple(float(value) for value in row) for row in rows), shape


def _dense_values(tensor: tinker.TensorData, shape: tuple[int, ...]) -> list[object]:
    crow = tensor.sparse_crow_indices
    columns = tensor.sparse_col_indices
    if (crow is None) != (columns is None):
        raise ValueError("sparse row and column indices must be provided together")
    if crow is None:
        if prod(shape) != len(tensor.data):
            raise ValueError("dense TensorData shape does not match its values")
        return list(tensor.data)
    if len(shape) != 2:
        raise ValueError("sparse TensorData must be 2-D")
    assert columns is not None
    row_offsets = tuple(_index("row pointer", value) for value in crow)
    column_indices = tuple(_index("column index", value) for value in columns)
    rows, width = shape
    if len(row_offsets) != rows + 1 or not row_offsets or row_offsets[0] != 0:
        raise ValueError("sparse row pointers do not match the tensor rows")
    if any(left > right for left, right in zip(row_offsets, row_offsets[1:])):
        raise ValueError("sparse row pointers must be nondecreasing")
    if row_offsets[-1] != len(tensor.data) or len(column_indices) != len(tensor.data):
        raise ValueError("sparse indices and values differ in cardinality")
    dense: list[object] = [0] * prod(shape)
    for row in range(rows):
        start, end = row_offsets[row : row + 2]
        row_columns = column_indices[start:end]
        if any(column >= width for column in row_columns):
            raise ValueError("sparse column index exceeds the tensor width")
        if any(left >= right for left, right in zip(row_columns, row_columns[1:])):
            raise ValueError("sparse column indices must increase within each row")
        for index, column in enumerate(row_columns, start=start):
            dense[row * width + column] = tensor.data[index]
    return dense


def _axis(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("TensorData shape axes must be positive integers")
    return value


def _index(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"sparse {name} must be a nonnegative integer")
    return value


def _number(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("TensorData values must be numeric")
    return value
