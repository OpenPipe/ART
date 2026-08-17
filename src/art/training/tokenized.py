from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TokenizedLossName = Literal[
    "cross_entropy",
    "importance_sampling",
    "ppo",
    "cispo",
]


class TokenizedDatum(BaseModel):
    """Exact model inputs and named-loss tensors supplied by a client."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: tuple[int, ...] = Field(min_length=1)
    target_tokens: tuple[tuple[int, ...], ...] = Field(min_length=1)
    weights: tuple[tuple[float, ...], ...] | None = None
    logprobs: tuple[tuple[float, ...], ...] | None = None
    advantages: tuple[tuple[float, ...], ...] | None = None

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
        return self

    @property
    def candidate_count(self) -> int:
        return len(self.target_tokens[0])

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
        coefficients = self.weights if loss == "cross_entropy" else self.advantages
        assert coefficients is not None
        if not any(value != 0.0 for row in coefficients for value in row):
            raise ValueError(f"{loss} datum has no loss-bearing target")
