from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

from art.training.token_matrix import (
    CapturedTokenRoutes,
    DenseRowValues,
    InlineTokenRoutes,
    NamedLossRequest,
    SpanRowValues,
    SpanValue,
    TextDatum,
    TokenMatrix,
    TokenMatrixBatch,
    TokenRow,
    dense_row,
    validate_token_matrix_batch,
)
from art.training.tokenized import TokenizedDatum

from .moe_routing import MoeRouteArray, MoeRouteSegments
from .tokenize import SFTBatch, TokenizedResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from art.model import TrainableModel
    from art.trajectories import Trajectory

    from .sft import SftBatchTokenizer


class _SftTokenizer(Protocol):
    def tokenize(
        self,
        model: TrainableModel,
        trajectories: Sequence[Trajectory],
        *,
        assistant_turns: Literal["all", "last"],
        learning_rate: float,
    ) -> SFTBatch: ...


@dataclass(frozen=True, slots=True)
class LoweredTokenMatrixBatch:
    """Canonical matrices plus zero-copy local route values, when already resolved."""

    batch: TokenMatrixBatch
    resolved_routes: dict[str, MoeRouteArray | MoeRouteSegments]


def token_matrix_from_tokenized_datum(
    datum: TokenizedDatum,
    *,
    matrix_id: str,
    packing_affinity_id: str | None = None,
) -> TokenMatrix:
    """Convert the temporary exact-token donor without changing any value."""

    candidate_count = datum.candidate_count
    token_count = len(datum.input_tokens)
    rows = [
        dense_row("token_ids", "int64", (token_count,), datum.input_tokens),
        dense_row(
            "target_token_ids",
            "int64",
            (token_count, candidate_count),
            _flatten(datum.target_tokens),
        ),
    ]
    if datum.weights is not None:
        rows.append(
            dense_row(
                "loss_weights",
                "float32",
                (token_count, candidate_count),
                _flatten(datum.weights),
            )
        )
    if datum.logprobs is not None:
        rows.append(
            dense_row(
                "behavior_logprobs",
                "float32",
                (token_count, candidate_count),
                _flatten(datum.logprobs),
            )
        )
    if datum.advantages is not None:
        rows.extend(
            (
                dense_row(
                    "loss_weights",
                    "float32",
                    (token_count, candidate_count),
                    (1.0,) * (token_count * candidate_count),
                ),
                dense_row(
                    "advantages",
                    "float32",
                    (token_count, candidate_count),
                    _flatten(datum.advantages),
                ),
            )
        )
    return TokenMatrix(
        matrix_id=matrix_id,
        packing_affinity_id=packing_affinity_id,
        rows=tuple(rows),
    )


def token_matrix_batch_from_tokenized_datums(
    datums: Sequence[TokenizedDatum],
    *,
    loss: NamedLossRequest,
    matrix_id_prefix: str = "datum",
) -> TokenMatrixBatch:
    batch = TokenMatrixBatch(
        matrices=tuple(
            token_matrix_from_tokenized_datum(
                datum,
                matrix_id=f"{matrix_id_prefix}-{index}",
            )
            for index, datum in enumerate(datums)
        )
    )
    validate_token_matrix_batch(batch, loss)
    return batch


def token_matrix_batch_from_sft(
    batch: SFTBatch,
    *,
    matrix_id_prefix: str = "sft",
) -> TokenMatrixBatch | None:
    """Lower the existing model-aware SFT oracle to causal TokenMatrix rows."""

    if batch.num_trajectories != len(batch.trajectory_tensors):
        raise ValueError("SFT trajectory count does not match its tensor payload")
    if batch.num_dropped_trajectories < 0:
        raise ValueError("SFT dropped trajectory count must be nonnegative")
    if not math.isfinite(batch.learning_rate):
        raise ValueError("SFT learning rate must be finite")

    matrices: list[TokenMatrix] = []
    num_tokens = 0
    source_trainable_tokens = 0
    required = {"input_ids", "attention_mask", "labels"}
    for index, tensors in enumerate(batch.trajectory_tensors):
        if set(tensors) != required:
            raise ValueError("SFT trajectory tensors must have the exact schema")
        for name in required:
            tensor = tensors[name]
            if (
                str(tensor.dtype) != "torch.int64"
                or tensor.device.type != "cpu"
                or not tensor.is_contiguous()
                or tensor.ndim != 2
                or tensor.shape[0] != 1
            ):
                raise ValueError("SFT tensors must be contiguous CPU int64 [1,T]")
        if len({tuple(tensors[name].shape) for name in required}) != 1:
            raise ValueError("SFT trajectory tensor shapes differ")
        if not bool((tensors["attention_mask"] == 1).all().item()):
            raise ValueError("SFT command tensors must be unpadded")

        tokens = tuple(int(value) for value in tensors["input_ids"][0].tolist())
        labels = tuple(int(value) for value in tensors["labels"][0].tolist())
        num_tokens += len(tokens)
        source_trainable_tokens += sum(label != -100 for label in labels)
        targets = (*labels[1:], -100)
        weights = tuple(1.0 if target != -100 else 0.0 for target in targets)
        if not any(weights):
            continue
        matrices.append(
            TokenMatrix(
                matrix_id=f"{matrix_id_prefix}-{index}",
                rows=(
                    dense_row("token_ids", "int64", (len(tokens),), tokens),
                    dense_row(
                        "target_token_ids",
                        "int64",
                        (len(tokens), 1),
                        tuple(target if target != -100 else 0 for target in targets),
                    ),
                    _coefficient_row("loss_weights", weights),
                ),
            )
        )
    if (batch.num_tokens, batch.num_trainable_tokens) != (
        num_tokens,
        source_trainable_tokens,
    ):
        raise ValueError("SFT token counts do not match the tensor payload")
    if not matrices:
        return None
    result = TokenMatrixBatch(matrices=tuple(matrices))
    validate_token_matrix_batch(result, NamedLossRequest(name="cross_entropy"))
    return result


def token_matrix_batch_from_text_datums(
    datums: Sequence[TextDatum],
    *,
    model: TrainableModel,
    tokenizer: SftBatchTokenizer | _SftTokenizer,
) -> TokenMatrixBatch:
    """Lower native text through the existing production SFT semantic oracle."""

    from art.trajectories import Trajectory

    matrices: list[TokenMatrix] = []
    for datum in datums:
        trajectory = Trajectory(
            messages_and_choices=list(datum.messages),
            tools=list(datum.tools) if datum.tools is not None else None,
        )
        tokenized = tokenizer.tokenize(
            model,
            (trajectory,),
            assistant_turns=datum.assistant_turns,
            learning_rate=0.0,
        )
        lowered = token_matrix_batch_from_sft(tokenized)
        if lowered is None or len(lowered.matrices) != 1:
            raise ValueError(f"text datum {datum.datum_id!r} has no trainable target")
        matrices.append(
            lowered.matrices[0].model_copy(
                update={
                    "matrix_id": datum.datum_id,
                    "packing_affinity_id": datum.packing_affinity_id,
                }
            )
        )
    if not matrices:
        raise ValueError("text lowering requires at least one datum")
    result = TokenMatrixBatch(matrices=tuple(matrices))
    validate_token_matrix_batch(
        result,
        NamedLossRequest(name="cross_entropy", normalize_advantages=False),
    )
    return result


def token_matrix_batch_from_art_rollouts(
    results: Sequence[TokenizedResult],
    *,
    loss: Literal["cispo", "importance_sampling"] = "cispo",
    normalize_advantages: bool = True,
    advantage_balance: float = 0.0,
    matrix_id_prefix: str = "rollout",
) -> LoweredTokenMatrixBatch:
    """Lower completed ART rollout semantics before the sole prefix-tree packer."""

    if not results:
        raise ValueError("ART rollout lowering requires at least one sequence")
    if not -1.0 <= advantage_balance <= 1.0 or not math.isfinite(advantage_balance):
        raise ValueError("advantage_balance must be finite and in [-1,1]")

    active_masks = [
        tuple(bool(value) for value in (*result.assistant_mask[1:], 0))
        for result in results
    ]
    active_weight_values = [
        float(result.weight)
        for result, mask in zip(results, active_masks, strict=True)
        for active in mask
        if active
    ]
    if not active_weight_values:
        raise ValueError("ART rollout batch has no loss-bearing target")
    weight_mean = sum(active_weight_values) / len(active_weight_values)
    if not math.isfinite(weight_mean) or weight_mean == 0.0:
        raise ValueError("ART rollout weights cannot be normalized")
    normalized_weights = [value / weight_mean for value in active_weight_values]

    balanced_advantages = []
    for result in results:
        value = float(result.advantage)
        if advantage_balance > 0.0 and value <= 0:
            value *= 1.0 - advantage_balance
        elif advantage_balance < 0.0 and value >= 0:
            value *= 1.0 + advantage_balance
        balanced_advantages.append(value)
    denominator_values = [
        abs(advantage) * weight
        for advantage, result, mask in zip(
            balanced_advantages, results, active_masks, strict=True
        )
        for weight, active in zip(
            (float(result.weight) / weight_mean,) * len(mask), mask, strict=True
        )
        if active
    ]
    advantage_scale = (
        sum(denominator_values) / len(denominator_values)
        if normalize_advantages
        else 1.0
    )
    if not math.isfinite(advantage_scale) or advantage_scale == 0.0:
        raise ValueError("ART rollout advantages cannot be normalized")

    matrices: list[TokenMatrix] = []
    routes: dict[str, MoeRouteArray | MoeRouteSegments] = {}
    captured_routes: list[CapturedTokenRoutes] = []
    weight_cursor = 0
    for index, (result, mask, advantage) in enumerate(
        zip(results, active_masks, balanced_advantages, strict=True)
    ):
        token_count = len(result.token_ids)
        if token_count < 1 or any(
            len(values) != token_count
            for values in (result.input_pos, result.assistant_mask, result.logprobs)
        ):
            raise ValueError("tokenized rollout arrays are not aligned")
        matrix_id = f"{matrix_id_prefix}-{index}"
        weights: list[float] = []
        for active in mask:
            if active:
                weights.append(normalized_weights[weight_cursor])
                weight_cursor += 1
            else:
                weights.append(0.0)
        behavior = (*result.logprobs[1:], 0.0)
        if any(
            active and not math.isfinite(float(value))
            for active, value in zip(mask, behavior, strict=True)
        ):
            raise ValueError("loss-bearing rollout tokens require behavior logprobs")
        rows: list[TokenRow] = [
            dense_row("token_ids", "int64", (token_count,), result.token_ids),
            dense_row(
                "target_token_ids",
                "int64",
                (token_count, 1),
                (*result.token_ids[1:], 0),
            ),
            _coefficient_row("loss_weights", tuple(weights)),
            _coefficient_row(
                "advantages",
                tuple(
                    advantage / advantage_scale if active else 0.0 for active in mask
                ),
            ),
            _coefficient_row(
                "behavior_logprobs",
                tuple(
                    float(value) if active else 0.0
                    for active, value in zip(mask, behavior, strict=True)
                ),
            ),
        ]
        if result.policy_versions is not None:
            shifted_versions = (*result.policy_versions[1:], -1)
            if all(
                version >= 0
                for version, active in zip(shifted_versions, mask, strict=True)
                if active
            ):
                rows.append(_policy_version_row(tuple(shifted_versions)))
        matrices.append(
            TokenMatrix(
                matrix_id=matrix_id,
                packing_affinity_id=f"prompt-{result.prompt_id}",
                rows=tuple(rows),
            )
        )
        if result.moe_routed_experts is not None:
            routes[matrix_id] = result.moe_routed_experts
        elif result.captured_route is not None:
            response_id, choice_index = result.captured_route
            captured_routes.append(
                CapturedTokenRoutes(
                    matrix_id=matrix_id,
                    response_id=response_id,
                    choice_index=choice_index,
                )
            )
    if weight_cursor != len(normalized_weights):
        raise AssertionError("ART rollout coefficient lowering lost logical values")
    request = NamedLossRequest(name=loss, normalize_advantages=False)
    batch = TokenMatrixBatch(
        matrices=tuple(matrices),
        routes=tuple(captured_routes),
    )
    validate_token_matrix_batch(batch, request)
    return LoweredTokenMatrixBatch(batch=batch, resolved_routes=routes)


def inline_routes_array(routes: InlineTokenRoutes) -> MoeRouteArray:
    dtype = np.uint8 if routes.num_experts <= 256 else np.uint16
    values = np.frombuffer(routes.expert_ids, dtype=dtype).reshape(routes.shape)
    return MoeRouteArray(values, num_experts=routes.num_experts)


def _coefficient_row(name: str, values: tuple[float, ...]) -> TokenRow:
    spans: list[SpanValue] = []
    start = 0
    while start < len(values):
        value = float(values[start])
        end = start + 1
        while end < len(values) and float(values[end]) == value:
            end += 1
        if value != 0.0:
            spans.append(SpanValue(start=start, end=end, value=(value,)))
        start = end
    span_encoding = SpanRowValues(default=(0.0,), spans=tuple(spans))
    dense_encoding = DenseRowValues(data=values)
    encoded = span_encoding if len(spans) * 3 + 1 < len(values) else dense_encoding
    return TokenRow(name=name, dtype="float32", shape=(len(values), 1), values=encoded)


def _policy_version_row(values: tuple[int, ...]) -> TokenRow:
    spans: list[SpanValue] = []
    start = 0
    while start < len(values):
        value = int(values[start])
        end = start + 1
        while end < len(values) and int(values[end]) == value:
            end += 1
        if value != -1:
            spans.append(SpanValue(start=start, end=end, value=value))
        start = end
    return TokenRow(
        name="policy_version",
        dtype="int64",
        shape=(len(values),),
        values=SpanRowValues(default=-1, spans=tuple(spans)),
    )


def _flatten(values: Sequence[Sequence[int | float]]) -> tuple[int | float, ...]:
    return tuple(value for row in values for value in row)
