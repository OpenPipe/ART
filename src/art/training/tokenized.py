from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TokenizedLossName = Literal["cross_entropy", "importance_sampling", "cispo"]
MAX_TOKENIZED_PHYSICAL_VALUES = 64 << 20


def validate_tokenized_loss_values(
    loss: TokenizedLossName,
    values: Mapping[str, float | int | bool | str | None],
) -> None:
    allowed = (
        {"clip_low_threshold", "clip_high_threshold"} if loss == "cispo" else set()
    )
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unsupported {loss} loss settings: {sorted(unknown)}")
    if loss != "cispo":
        return
    low, high = tokenized_clip_bounds(loss, values)
    if low > high:
        raise ValueError("clip_low_threshold must not exceed clip_high_threshold")


def tokenized_clip_bounds(
    loss: Literal["ppo", "cispo"],
    values: Mapping[str, float | int | bool | str | None],
) -> tuple[float, float]:
    defaults = (0.8, 1.2) if loss == "ppo" else (0.0, 4.0)
    return (
        _finite_float(
            "clip_low_threshold", values.get("clip_low_threshold", defaults[0])
        ),
        _finite_float(
            "clip_high_threshold", values.get("clip_high_threshold", defaults[1])
        ),
    )


class TokenizedDatum(BaseModel):
    """Exact text tokens and named-loss tensors supplied by a client."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: tuple[int, ...] = Field(min_length=1)
    target_tokens: tuple[tuple[int, ...], ...] = Field(min_length=1)
    weights: tuple[tuple[float, ...], ...] | None = None
    logprobs: tuple[tuple[float, ...], ...] | None = None
    advantages: tuple[tuple[float, ...], ...] | None = None

    @model_validator(mode="after")
    def _validate_tensors(self) -> TokenizedDatum:
        if any(token < 0 for token in self.input_tokens):
            raise ValueError("input_tokens must contain nonnegative token IDs")
        if len(self.target_tokens) != len(self.input_tokens):
            raise ValueError("target_tokens must have one row per input token")
        width = len(self.target_tokens[0])
        if width < 1 or any(len(row) != width for row in self.target_tokens):
            raise ValueError("target_tokens must be a nonempty rectangular matrix")
        if any(token < 0 for row in self.target_tokens for token in row):
            raise ValueError("target_tokens must contain nonnegative token IDs")
        for name in ("weights", "logprobs", "advantages"):
            values = getattr(self, name)
            if values is None:
                continue
            if len(values) != len(self.input_tokens) or any(
                len(row) != width for row in values
            ):
                raise ValueError(f"{name} must match target_tokens")
            if any(not math.isfinite(value) for row in values for value in row):
                raise ValueError(f"{name} must contain finite values")
        return self

    @property
    def candidate_count(self) -> int:
        return len(self.target_tokens[0])

    def validate_for_loss(self, loss: TokenizedLossName) -> None:
        required = (
            {"weights"}
            if loss == "cross_entropy"
            else {
                "logprobs",
                "advantages",
            }
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
        coefficients = self.weights if loss == "cross_entropy" else self.advantages
        assert coefficients is not None
        if not any(value != 0.0 for row in coefficients for value in row):
            raise ValueError(f"{loss} datum has no loss-bearing target")


def tokenized_physical_value_count(datums: tuple[TokenizedDatum, ...]) -> int:
    return sum(len(datum.input_tokens) * datum.candidate_count for datum in datums)


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
