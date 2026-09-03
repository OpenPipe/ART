from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import isfinite, prod
import re
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

MAX_MATRIX_IDENTIFIER_LENGTH = 255
MAX_MATRIX_ROWS = 32
MAX_MATRIX_TOKENS = 1 << 20
MAX_MATRIX_VALUES = 64 << 20
MAX_TARGET_CANDIDATES = 256
MAX_TEXT_MESSAGES = 4096
MAX_TEXT_TOOLS = 256

RowDType: TypeAlias = Literal["bool", "int32", "int64", "float32"]
RowScalar: TypeAlias = bool | int | float
RowElement: TypeAlias = RowScalar | tuple[RowScalar, ...]
LossName: TypeAlias = Literal["cross_entropy", "importance_sampling", "cispo", "dpo"]
LossContractId: TypeAlias = Literal[
    "cross_entropy_v1", "importance_sampling_v1", "cispo_v1", "dpo_v1"
]

_ROW_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_COMMON_ROWS = frozenset(
    {
        "token_ids",
        "target_token_ids",
        "loss_weights",
        "advantages",
        "behavior_logprobs",
        "policy_version",
    }
)
_VIRTUAL_ROWS = frozenset({"learner_logprobs"})


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DenseRowValues(_Contract):
    """Row-major logical values. Shape and dtype live on the owning TokenRow."""

    encoding: Literal["dense"] = "dense"
    data: tuple[RowScalar, ...]


class SpanValue(_Contract):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    value: RowElement

    @model_validator(mode="after")
    def _validate_range(self) -> SpanValue:
        if self.end <= self.start:
            raise ValueError("span end must be greater than start")
        return self


class SpanRowValues(_Contract):
    """Piecewise-constant values along the logical token axis."""

    encoding: Literal["spans"] = "spans"
    default: RowElement
    spans: tuple[SpanValue, ...] = ()

    @model_validator(mode="after")
    def _validate_order(self) -> SpanRowValues:
        previous_end = 0
        for span in self.spans:
            if span.start < previous_end:
                raise ValueError("row spans must be sorted and non-overlapping")
            previous_end = span.end
        return self


RowValues = Annotated[DenseRowValues | SpanRowValues, Field(discriminator="encoding")]


class TokenRow(_Contract):
    name: str = Field(min_length=1, max_length=64)
    dtype: RowDType
    shape: tuple[int, ...] = Field(min_length=1, max_length=2)
    values: RowValues

    @model_validator(mode="after")
    def _validate_row(self) -> TokenRow:
        if _ROW_NAME.fullmatch(self.name) is None:
            raise ValueError("row name must be lower snake case")
        if self.name in _VIRTUAL_ROWS:
            raise ValueError(f"{self.name} is server-produced and cannot be supplied")
        if any(dimension < 1 for dimension in self.shape):
            raise ValueError("row dimensions must be positive")
        if self.shape[0] > MAX_MATRIX_TOKENS:
            raise ValueError("row exceeds the logical token limit")
        if prod(self.shape) > MAX_MATRIX_VALUES:
            raise ValueError("row exceeds the logical value limit")
        width = prod(self.shape[1:])
        if isinstance(self.values, DenseRowValues):
            if len(self.values.data) != prod(self.shape):
                raise ValueError("dense row value count does not match shape")
            raw = self.values.data
        else:
            raw = (*_element_values(self.values.default),)
            for span in self.values.spans:
                if span.end > self.shape[0]:
                    raise ValueError("row span exceeds the logical token axis")
                raw += _element_values(span.value)
            if len(_element_values(self.values.default)) != width or any(
                len(_element_values(span.value)) != width for span in self.values.spans
            ):
                raise ValueError("span values do not match the row trailing shape")
        for value in raw:
            _validate_scalar(self.dtype, value)
        return self

    @property
    def token_count(self) -> int:
        return self.shape[0]

    @property
    def trailing_width(self) -> int:
        return prod(self.shape[1:])

    def dense_values(self) -> tuple[RowScalar, ...]:
        if isinstance(self.values, DenseRowValues):
            return self.values.data
        width = self.trailing_width
        default = _element_values(self.values.default)
        dense = [value for _ in range(self.token_count) for value in default]
        for span in self.values.spans:
            value = _element_values(span.value)
            for position in range(span.start, span.end):
                start = position * width
                dense[start : start + width] = value
        return tuple(dense)


class TokenMatrix(_Contract):
    kind: Literal["matrix"] = "matrix"
    matrix_id: str = Field(min_length=1, max_length=MAX_MATRIX_IDENTIFIER_LENGTH)
    rows: tuple[TokenRow, ...] = Field(min_length=1, max_length=MAX_MATRIX_ROWS)
    packing_affinity_id: str | None = Field(
        default=None, min_length=1, max_length=MAX_MATRIX_IDENTIFIER_LENGTH
    )

    @model_validator(mode="after")
    def _validate_matrix(self) -> TokenMatrix:
        by_name = {row.name: row for row in self.rows}
        if len(by_name) != len(self.rows):
            raise ValueError("TokenMatrix row names must be unique")
        tokens = by_name.get("token_ids")
        if tokens is None:
            raise ValueError("TokenMatrix requires one token_ids row")
        if tokens.dtype != "int64" or len(tokens.shape) != 1:
            raise ValueError("token_ids must be int64 with shape [T]")
        if any(row.token_count != tokens.token_count for row in self.rows):
            raise ValueError("all TokenMatrix rows must share the token_ids axis")
        if any(int(value) < 0 for value in tokens.dense_values()):
            raise ValueError("token_ids must be nonnegative")
        return self

    @property
    def token_count(self) -> int:
        return self.row("token_ids").token_count

    def row(self, name: str) -> TokenRow:
        try:
            return next(row for row in self.rows if row.name == name)
        except StopIteration as exc:
            raise KeyError(name) from exc

    def optional_row(self, name: str) -> TokenRow | None:
        return next((row for row in self.rows if row.name == name), None)


class TextDatum(_Contract):
    """Model-aware text ingress lowered to TokenMatrix before packing."""

    kind: Literal["text"] = "text"
    datum_id: str = Field(min_length=1, max_length=MAX_MATRIX_IDENTIFIER_LENGTH)
    messages: tuple[dict[str, JsonValue], ...] = Field(
        min_length=1, max_length=MAX_TEXT_MESSAGES
    )
    tools: tuple[dict[str, JsonValue], ...] | None = Field(
        default=None, max_length=MAX_TEXT_TOOLS
    )
    assistant_turns: Literal["all", "last"] = "all"
    packing_affinity_id: str | None = Field(
        default=None, min_length=1, max_length=MAX_MATRIX_IDENTIFIER_LENGTH
    )

    @model_validator(mode="after")
    def _validate_messages(self) -> TextDatum:
        supported_roles = {"system", "user", "assistant", "tool"}
        for message in self.messages:
            role = message.get("role")
            if role not in supported_roles:
                raise ValueError(f"unsupported chat role {role!r}")
            if role == "tool" and not message.get("tool_call_id"):
                raise ValueError("tool messages require tool_call_id")
            if (
                role == "assistant"
                and message.get("content") is None
                and not message.get("tool_calls")
            ):
                raise ValueError("assistant messages require content or tool_calls")
        if (
            self.assistant_turns == "last"
            and self.messages[-1].get("role") != "assistant"
        ):
            raise ValueError(
                "assistant_turns='last' requires a final assistant message"
            )
        return self


InputDatum = Annotated[TextDatum | TokenMatrix, Field(discriminator="kind")]


class InlineTokenRoutes(_Contract):
    """Dense [T,L,K] route IDs for one exact matrix token lineage."""

    kind: Literal["inline"] = "inline"
    matrix_id: str = Field(min_length=1, max_length=MAX_MATRIX_IDENTIFIER_LENGTH)
    num_experts: int = Field(ge=1, le=65_536)
    shape: tuple[int, int, int]
    expert_ids: bytes

    @model_validator(mode="after")
    def _validate_routes(self) -> InlineTokenRoutes:
        tokens, layers, topk = self.shape
        if min(tokens, layers, topk) < 1 or topk > self.num_experts:
            raise ValueError("inline route shape is invalid")
        item_size = 1 if self.num_experts <= 256 else 2
        if len(self.expert_ids) != tokens * layers * topk * item_size:
            raise ValueError("inline route byte count does not match its shape")
        return self


class CapturedTokenRoutes(_Contract):
    """Public inference response selector resolved by the training service."""

    kind: Literal["captured"] = "captured"
    matrix_id: str = Field(min_length=1, max_length=MAX_MATRIX_IDENTIFIER_LENGTH)
    response_id: str = Field(min_length=1, max_length=512)
    choice_index: int = Field(ge=0)


class RetainedTokenRoutes(_Contract):
    """One matrix choice inside an authenticated retained route bundle."""

    kind: Literal["retained"] = "retained"
    matrix_id: str = Field(min_length=1, max_length=MAX_MATRIX_IDENTIFIER_LENGTH)
    bundle: dict[str, JsonValue]
    choice_index: int = Field(ge=0)


TokenRoutes = Annotated[
    CapturedTokenRoutes | InlineTokenRoutes | RetainedTokenRoutes,
    Field(discriminator="kind"),
]


class TokenMatrixBatch(_Contract):
    kind: Literal["token_matrix"] = "token_matrix"
    matrices: tuple[TokenMatrix, ...] = Field(min_length=1)
    routes: tuple[TokenRoutes, ...] = ()

    @model_validator(mode="after")
    def _validate_batch(self) -> TokenMatrixBatch:
        matrix_ids = tuple(matrix.matrix_id for matrix in self.matrices)
        if len(set(matrix_ids)) != len(matrix_ids):
            raise ValueError("TokenMatrix matrix_id values must be unique")
        if sum(matrix.token_count for matrix in self.matrices) > MAX_MATRIX_TOKENS:
            raise ValueError("TokenMatrixBatch exceeds the logical token limit")
        route_ids = tuple(route.matrix_id for route in self.routes)
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("TokenMatrix routes must name each matrix at most once")
        if not set(route_ids) <= set(matrix_ids):
            raise ValueError("TokenMatrix routes reference an unknown matrix")
        for route in self.routes:
            if isinstance(route, InlineTokenRoutes):
                matrix = self.matrix(route.matrix_id)
                if route.shape[0] != matrix.token_count:
                    raise ValueError("inline routes do not match the matrix token axis")
        return self

    def matrix(self, matrix_id: str) -> TokenMatrix:
        try:
            return next(
                matrix for matrix in self.matrices if matrix.matrix_id == matrix_id
            )
        except StopIteration as exc:
            raise KeyError(matrix_id) from exc


class MatrixPair(_Contract):
    component_id: str = Field(min_length=1, max_length=MAX_MATRIX_IDENTIFIER_LENGTH)
    chosen_matrix_id: str = Field(min_length=1, max_length=MAX_MATRIX_IDENTIFIER_LENGTH)
    rejected_matrix_id: str = Field(
        min_length=1, max_length=MAX_MATRIX_IDENTIFIER_LENGTH
    )

    @model_validator(mode="after")
    def _validate_pair(self) -> MatrixPair:
        if self.chosen_matrix_id == self.rejected_matrix_id:
            raise ValueError("a matrix pair requires distinct matrices")
        return self

    @property
    def matrix_ids(self) -> tuple[str, str]:
        return self.chosen_matrix_id, self.rejected_matrix_id


class NamedLossRequest(_Contract):
    name: LossName
    normalize_advantages: bool = True
    values: dict[str, float | int | bool | str | None] = Field(default_factory=dict)
    matrix_pairs: tuple[MatrixPair, ...] = ()

    @model_validator(mode="after")
    def _validate_settings(self) -> NamedLossRequest:
        allowed = {
            "cross_entropy": frozenset(),
            "importance_sampling": frozenset(),
            "cispo": frozenset({"clip_low_threshold", "clip_high_threshold"}),
            "dpo": frozenset({"beta"}),
        }[self.name]
        unknown = set(self.values) - allowed
        if unknown:
            raise ValueError(f"unsupported {self.name} settings: {sorted(unknown)}")
        if self.name == "cispo":
            low = _finite_setting(self.values, "clip_low_threshold", 0.0)
            high = _finite_setting(self.values, "clip_high_threshold", 4.0)
            if low > high:
                raise ValueError(
                    "clip_low_threshold must not exceed clip_high_threshold"
                )
        if self.name == "dpo":
            if not self.matrix_pairs:
                raise ValueError(
                    "dpo requires at least one chosen/rejected matrix pair"
                )
            if _finite_setting(self.values, "beta", 0.1) <= 0:
                raise ValueError("dpo beta must be positive")
        elif self.matrix_pairs:
            raise ValueError(f"{self.name} does not accept matrix pairs")
        component_ids = [pair.component_id for pair in self.matrix_pairs]
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("matrix pair component IDs must be unique")
        return self

    def placement_components(self) -> tuple[tuple[str, ...], ...]:
        return tuple(pair.matrix_ids for pair in self.matrix_pairs)

    @property
    def contract_id(self) -> LossContractId:
        return {
            "cross_entropy": "cross_entropy_v1",
            "importance_sampling": "importance_sampling_v1",
            "cispo": "cispo_v1",
            "dpo": "dpo_v1",
        }[self.name]


def validate_token_matrix_batch(
    batch: TokenMatrixBatch,
    loss: NamedLossRequest | None,
    *,
    output_rows: Iterable[str] = ("learner_logprobs",),
) -> None:
    """Validate semantic row reads and loss placement before packing or GPU work."""

    requested_outputs = frozenset(output_rows)
    if not requested_outputs <= _VIRTUAL_ROWS:
        raise ValueError(
            f"unsupported output rows: {sorted(requested_outputs - _VIRTUAL_ROWS)}"
        )
    required, optional = _loss_rows(loss)
    permitted = frozenset({"token_ids"}) | required | optional
    policy_inputs: list[tuple[str, tuple[bool, ...], TokenRow | None]] = []
    for matrix in batch.matrices:
        rows = {row.name: row for row in matrix.rows}
        missing = required - rows.keys()
        unknown = rows.keys() - permitted
        if missing:
            raise ValueError(
                f"matrix {matrix.matrix_id!r} is missing rows {sorted(missing)}"
            )
        if unknown:
            raise ValueError(
                f"matrix {matrix.matrix_id!r} has unused rows {sorted(unknown)}"
            )
        targets = rows.get("target_token_ids")
        if targets is None:
            if requested_outputs:
                raise ValueError("learner_logprobs output requires target_token_ids")
            continue
        if targets.dtype != "int64" or len(targets.shape) != 2:
            raise ValueError("target_token_ids must be int64 with shape [T,K]")
        if targets.shape[1] > MAX_TARGET_CANDIDATES:
            raise ValueError("target_token_ids exceeds the candidate limit")
        if any(int(value) < 0 for value in targets.dense_values()):
            raise ValueError("target_token_ids must be nonnegative")
        if loss is not None and loss.name != "cross_entropy" and targets.shape[1] != 1:
            raise ValueError(f"{loss.name} requires one target per token position")
        for name in (required | optional) - {"target_token_ids", "policy_version"}:
            row = rows.get(name)
            if row is None:
                continue
            if row.dtype != "float32" or row.shape != targets.shape:
                raise ValueError(f"{name} must be float32 and match target_token_ids")
        policy = rows.get("policy_version")
        if policy is not None and (
            policy.dtype != "int64"
            or policy.shape not in {(matrix.token_count,), (matrix.token_count, 1)}
        ):
            raise ValueError("policy_version must be int64 with shape [T] or [T,1]")
        active_positions = active_loss_positions(matrix, loss)
        policy_inputs.append((matrix.matrix_id, active_positions, policy))
        if loss is not None and not any(active_positions):
            raise ValueError(f"matrix {matrix.matrix_id!r} has no loss-bearing target")
    if any(policy is not None for _, _, policy in policy_inputs):
        for matrix_id, active_positions, policy in policy_inputs:
            if any(active_positions) and policy is None:
                raise ValueError(
                    "policy_version must cover every accepted position or be omitted"
                )
            if policy is None:
                continue
            policy_values = policy.dense_values()
            if any(
                active and int(policy_values[position * policy.trailing_width]) < 0
                for position, active in enumerate(active_positions)
            ):
                raise ValueError(
                    f"matrix {matrix_id!r} has a negative policy_version at an "
                    "accepted position"
                )
    if loss is not None:
        matrix_ids = {matrix.matrix_id for matrix in batch.matrices}
        paired_ids = [
            matrix_id for pair in loss.matrix_pairs for matrix_id in pair.matrix_ids
        ]
        if not set(paired_ids) <= matrix_ids:
            raise ValueError("loss component references an unknown matrix")
        if len(set(paired_ids)) != len(paired_ids):
            raise ValueError("a matrix can belong to at most one hard loss component")
        if loss.name == "dpo" and set(paired_ids) != matrix_ids:
            raise ValueError("every dpo matrix must belong to one matrix pair")


def dense_row(
    name: str,
    dtype: RowDType,
    shape: Sequence[int],
    values: Sequence[RowScalar],
) -> TokenRow:
    return TokenRow(
        name=name,
        dtype=dtype,
        shape=tuple(int(value) for value in shape),
        values=DenseRowValues(data=tuple(values)),
    )


def span_row(
    name: str,
    dtype: RowDType,
    shape: Sequence[int],
    *,
    default: RowElement,
    spans: Sequence[SpanValue],
) -> TokenRow:
    return TokenRow(
        name=name,
        dtype=dtype,
        shape=tuple(int(value) for value in shape),
        values=SpanRowValues(default=default, spans=tuple(spans)),
    )


def _loss_rows(
    loss: NamedLossRequest | None,
) -> tuple[frozenset[str], frozenset[str]]:
    if loss is None:
        return frozenset({"target_token_ids"}), frozenset({"policy_version"})
    required = {
        "cross_entropy": frozenset({"target_token_ids", "loss_weights"}),
        "importance_sampling": frozenset(
            {"target_token_ids", "behavior_logprobs", "advantages"}
        ),
        "cispo": frozenset({"target_token_ids", "behavior_logprobs", "advantages"}),
        "dpo": frozenset({"target_token_ids", "loss_weights", "behavior_logprobs"}),
    }[loss.name]
    optional = {"policy_version"}
    if loss.name in {"importance_sampling", "cispo"}:
        optional.add("loss_weights")
    return required, frozenset(optional)


def active_loss_positions(
    matrix: TokenMatrix,
    loss: NamedLossRequest | None,
) -> tuple[bool, ...]:
    """Return logical positions selected by the named loss, independent of K."""

    if loss is None:
        return (False,) * matrix.token_count
    coefficient_name = (
        "advantages"
        if loss.name in {"importance_sampling", "cispo"}
        else "loss_weights"
    )
    coefficient = matrix.optional_row(coefficient_name)
    if coefficient is None:
        return (False,) * matrix.token_count
    values = coefficient.dense_values()
    width = coefficient.trailing_width
    return tuple(
        any(
            float(value) != 0.0
            for value in values[position * width : (position + 1) * width]
        )
        for position in range(matrix.token_count)
    )


def _element_values(value: RowElement) -> tuple[RowScalar, ...]:
    return value if isinstance(value, tuple) else (value,)


def _validate_scalar(dtype: RowDType, value: RowScalar) -> None:
    if dtype == "bool":
        if not isinstance(value, bool):
            raise TypeError("bool rows require boolean values")
        return
    if dtype in {"int32", "int64"}:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{dtype} rows require integer values")
        if dtype == "int32" and not -(1 << 31) <= value < (1 << 31):
            raise ValueError("int32 row value is out of range")
        if dtype == "int64" and not -(1 << 63) <= value < (1 << 63):
            raise ValueError("int64 row value is out of range")
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(value)
    ):
        raise TypeError("float32 rows require finite numeric values")


def _finite_setting(
    values: Mapping[str, float | int | bool | str | None],
    name: str,
    default: float,
) -> float:
    raw = values.get(name, default)
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise TypeError(f"{name} must be numeric")
    result = float(raw)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
