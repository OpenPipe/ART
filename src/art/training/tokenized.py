from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, model_validator

from art.preprocessing.moe_routing import (
    MoeRouteArray,
    MoeRouteSegments,
    moe_route_dtype,
)

TokenizedLossName = Literal[
    "cross_entropy",
    "importance_sampling",
    "ppo",
    "cispo",
]
MAX_TOKENIZED_LOGPROB_VALUES = 16 << 20
MAX_TOKENIZED_PHYSICAL_VALUES = 64 << 20
TokenizedMoeRouteBytes = Annotated[
    bytes | memoryview,
    PlainSerializer(bytes, return_type=bytes, when_used="json"),
]


class TokenizedPolicySpan(BaseModel):
    """Contiguous causal target positions scored by one policy version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_position: int = Field(ge=0)
    end_position: int = Field(gt=0)
    policy_version: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_order(self) -> "TokenizedPolicySpan":
        if self.end_position <= self.start_position:
            raise ValueError("policy span end must exceed its start")
        return self


class TokenizedMoeRoutes(BaseModel):
    """Exact per-input-token MoE routes, optionally split into shared segments."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    num_experts: int = Field(ge=1, le=65_536)
    dtype: Literal["uint8", "uint16"]
    shape: tuple[int, int, int]
    data: tuple[TokenizedMoeRouteBytes, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_layout(self) -> "TokenizedMoeRoutes":
        tokens, layers, topk = self.shape
        if min(tokens, layers, topk) <= 0:
            raise ValueError("MoE route shape axes must be positive")
        if topk > self.num_experts:
            raise ValueError("MoE route topk exceeds num_experts")
        if np.dtype(self.dtype) != moe_route_dtype(self.num_experts):
            raise ValueError("MoE route dtype does not match num_experts")
        if any(
            isinstance(segment, memoryview)
            and (
                not segment.readonly
                or not segment.c_contiguous
                or segment.ndim != 1
                or segment.format != "B"
            )
            for segment in self.data
        ):
            raise ValueError("MoE route memoryviews must be readonly contiguous bytes")
        for segment in self.data:
            if isinstance(segment, memoryview) and not isinstance(segment.obj, bytes):
                raise ValueError("MoE route memoryviews must have bytes backing")
        bytes_per_token = np.dtype(self.dtype).itemsize * layers * topk
        if any(not segment or len(segment) % bytes_per_token for segment in self.data):
            raise ValueError("MoE route segments must contain whole tokens")
        if sum(map(len, self.data)) != tokens * bytes_per_token:
            raise ValueError("MoE route segments do not match their declared shape")
        object.__setattr__(
            self,
            "data",
            tuple(
                memoryview(segment) if isinstance(segment, memoryview) else segment
                for segment in self.data
            ),
        )
        return self

    def build(self) -> MoeRouteArray | MoeRouteSegments:
        _, layers, topk = self.shape
        bytes_per_token = np.dtype(self.dtype).itemsize * layers * topk
        segments = tuple(
            MoeRouteArray(
                np.frombuffer(data, dtype=self.dtype).reshape(
                    len(data) // bytes_per_token, layers, topk
                ),
                num_experts=self.num_experts,
            )
            for data in self.data
        )
        return (
            segments[0] if len(segments) == 1 else MoeRouteSegments(segments=segments)
        )


def tokenized_result_value_count(datums: tuple["TokenizedDatum", ...]) -> int:
    return sum(len(datum.input_tokens) * datum.candidate_count for datum in datums)


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_tokenized_loss_values(
    loss: TokenizedLossName,
    values: Mapping[str, float | int | bool | str | None],
) -> None:
    numeric = {
        "clip_low_threshold",
        "clip_high_threshold",
        "grad_accumulation_sequences",
        "kl_penalty_coef",
    }
    strings = {"kl_penalty_source", "kl_ref_adapter_path", "kl_ref_checkpoint_id"}
    allowed = numeric | strings if loss in {"ppo", "cispo"} else set()
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unsupported {loss} loss settings: {sorted(unknown)}")
    for name in numeric & values.keys():
        value = values[name]
        if value is not None:
            _finite_float(name, value)
    if _finite_float("kl_penalty_coef", values.get("kl_penalty_coef", 0.0)) < 0:
        raise ValueError("kl_penalty_coef must be nonnegative")
    source = values.get("kl_penalty_source")
    if source is not None and source not in {"current_learner", "sample"}:
        raise ValueError("kl_penalty_source is invalid")
    for name in strings - {"kl_penalty_source"}:
        value = values.get(name)
        if value is not None and (not isinstance(value, str) or not value):
            raise TypeError(f"{name} must be a nonempty string")
    if (
        loss in {"ppo", "cispo"}
        and tokenized_clip_bounds(loss, values)[0]
        > tokenized_clip_bounds(loss, values)[1]
    ):
        raise ValueError("clip_low_threshold must not exceed clip_high_threshold")


def tokenized_clip_bounds(
    loss: TokenizedLossName,
    values: Mapping[str, float | int | bool | str | None],
) -> tuple[float, float]:
    if loss not in {"ppo", "cispo"}:
        raise ValueError(f"{loss} has no clipping bounds")
    defaults = (0.8, 1.2) if loss == "ppo" else (0.0, 4.0)
    low = values.get("clip_low_threshold", defaults[0])
    high = values.get("clip_high_threshold", defaults[1])
    return (
        _finite_float("clip_low_threshold", low),
        _finite_float("clip_high_threshold", high),
    )


class TokenizedDatum(BaseModel):
    """Exact model inputs and named-loss tensors supplied by a client."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: tuple[int, ...] = Field(min_length=1)
    target_tokens: tuple[tuple[int, ...], ...] = Field(min_length=1)
    weights: tuple[tuple[float, ...], ...] | None = None
    logprobs: tuple[tuple[float, ...], ...] | None = None
    advantages: tuple[tuple[float, ...], ...] | None = None
    packing_group_id: int | None = Field(default=None, ge=0)
    policy_spans: tuple[TokenizedPolicySpan, ...] = ()
    moe_routes: TokenizedMoeRoutes | None = None

    @model_validator(mode="after")
    def _validate_tensors(self) -> "TokenizedDatum":
        if any(token < 0 for token in self.input_tokens):
            raise ValueError("input_tokens must contain nonnegative token IDs")
        if len(self.target_tokens) != len(self.input_tokens):
            raise ValueError("target_tokens must have one row per input token")
        candidates = len(self.target_tokens[0])
        if candidates < 1 or any(len(row) != candidates for row in self.target_tokens):
            raise ValueError("target_tokens must be a nonempty rectangular matrix")
        if any(token < 0 for row in self.target_tokens for token in row):
            raise ValueError("target_tokens must contain nonnegative token IDs")
        for name in ("weights", "logprobs", "advantages"):
            values = getattr(self, name)
            if values is None:
                continue
            if len(values) != len(self.input_tokens) or any(
                len(row) != candidates for row in values
            ):
                raise ValueError(f"{name} must match target_tokens")
            if any(not math.isfinite(value) for row in values for value in row):
                raise ValueError(f"{name} must contain finite values")
        cursor = 0
        for span in self.policy_spans:
            if span.start_position != cursor:
                raise ValueError("policy spans must form a contiguous partition")
            cursor = span.end_position
        if self.policy_spans and cursor != len(self.input_tokens):
            raise ValueError("policy spans must cover every causal target position")
        if self.moe_routes is not None and self.moe_routes.shape[0] != len(
            self.input_tokens
        ):
            raise ValueError("MoE routes must cover every exact input token")
        return self

    @property
    def candidate_count(self) -> int:
        return len(self.target_tokens[0])

    def policy_versions(self) -> np.ndarray:
        versions = np.full(len(self.input_tokens), -1, dtype=np.int64)
        for span in self.policy_spans:
            versions[span.start_position : span.end_position] = span.policy_version
        return versions

    def validate_for_loss(self, loss: TokenizedLossName) -> None:
        required = (
            {"weights"} if loss == "cross_entropy" else {"logprobs", "advantages"}
        )
        present = {
            name
            for name in ("weights", "logprobs", "advantages")
            if getattr(self, name) is not None
        }
        if present != required:
            raise ValueError(
                f"{loss} requires exactly {sorted(required)}, got {sorted(present)}"
            )
        if loss != "cross_entropy" and self.candidate_count != 1:
            raise ValueError(f"{loss} requires one target token per input position")
        if loss != "cross_entropy" and not self.policy_spans:
            raise ValueError(f"{loss} requires complete policy spans")
        coefficients = self.weights if loss == "cross_entropy" else self.advantages
        assert coefficients is not None
        if not any(value != 0.0 for row in coefficients for value in row):
            raise ValueError(f"{loss} datum has no loss-bearing target")
