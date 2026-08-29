from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import torch

GradientReduction = Literal["token_mean", "sum"]


@dataclass(frozen=True)
class AccumulatedGradientSums:
    gradients: tuple[torch.Tensor, ...]
    local_token_count: torch.Tensor
    expected_global_token_count: int | None
    reduction: GradientReduction


class GradientAccumulator:
    """Retain many F/B contributions in one parameter-sized gradient image."""

    def __init__(
        self,
        model_chunks: list[torch.nn.Module],
        *,
        max_contributions: int = 64,
        flush_gradients: Callable[[list[torch.nn.Module]], None] | None = None,
    ) -> None:
        if not 1 <= max_contributions <= 64:
            raise ValueError("max_contributions must be between 1 and 64")
        self.model_chunks = model_chunks
        self.max_contributions = max_contributions
        self._flush_gradients = flush_gradients
        self._operation_ids: list[str] = []
        self._saved_gradients: tuple[torch.Tensor, ...] | None = None
        self._saved_tokens: torch.Tensor | None = None
        self._resident_tokens: torch.Tensor | None = None
        self._expected_global_tokens: int | None = None
        self._expects_global_tokens: bool | None = None
        self._reduction: GradientReduction | None = None
        self._sealed: tuple[str, ...] | None = None
        self._layout: tuple[int, ...] = ()
        gradients = self._main_gradients()
        if not gradients:
            raise RuntimeError("gradient accumulator requires trainable parameters")
        self._layout = tuple(id(gradient) for gradient in gradients)

    @property
    def contribution_ids(self) -> tuple[str, ...]:
        return tuple(self._operation_ids)

    def residency_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            *((self._saved_tokens,) if self._saved_tokens is not None else ()),
            *(self._saved_gradients or ()),
        )

    def before_forward_backward(self) -> None:
        if self._sealed is not None:
            raise RuntimeError("cannot add gradients to a sealed accumulator")
        if self._resident_tokens is None:
            return
        if self._flush_gradients is not None:
            self._flush_gradients(self.model_chunks)
        gradients = self._main_gradients()
        if self._saved_gradients is None:
            self._saved_gradients = tuple(gradient.clone() for gradient in gradients)
            self._saved_tokens = self._resident_tokens.clone()
        else:
            assert self._saved_tokens is not None
            for saved, gradient in zip(self._saved_gradients, gradients, strict=True):
                saved.add_(gradient)
            self._saved_tokens.add_(self._resident_tokens)
        self._resident_tokens = None

    def record(
        self,
        operation_id: str,
        token_count: torch.Tensor,
        *,
        expected_global_token_count: int | None = None,
        reduction: GradientReduction = "token_mean",
    ) -> None:
        if self._sealed is not None:
            raise RuntimeError("cannot add gradients to a sealed accumulator")
        if len(self._operation_ids) >= self.max_contributions:
            raise RuntimeError("gradient contribution limit reached")
        if operation_id in self._operation_ids:
            raise RuntimeError(f"duplicate gradient contribution {operation_id!r}")
        if token_count.numel() != 1:
            raise ValueError("gradient contribution token_count must be scalar")
        if self._resident_tokens is not None:
            raise RuntimeError("resident gradients must be stashed before another F/B")
        if self._reduction is not None and self._reduction != reduction:
            raise RuntimeError("one gradient accumulator cannot mix reduction modes")
        expects_global_tokens = expected_global_token_count is not None
        if (
            self._expects_global_tokens is not None
            and self._expects_global_tokens != expects_global_tokens
        ):
            raise RuntimeError(
                "one gradient accumulator cannot mix checked and unchecked tokens"
            )
        self._operation_ids.append(operation_id)
        self._resident_tokens = token_count.detach().clone()
        self._expects_global_tokens = expects_global_tokens
        if expected_global_token_count is not None:
            self._expected_global_tokens = (
                self._expected_global_tokens or 0
            ) + expected_global_token_count
        self._reduction = reduction

    def seal(self, operation_ids: tuple[str, ...]) -> None:
        if not operation_ids:
            raise RuntimeError("optimizer requires at least one F/B contribution")
        if self._sealed is not None:
            raise RuntimeError("gradient accumulator is already sealed")
        if operation_ids != tuple(self._operation_ids):
            raise RuntimeError(
                "optimizer contribution order does not match the open accumulator"
            )
        self._sealed = operation_ids

    def prepare_optimizer(self) -> AccumulatedGradientSums:
        if self._sealed is None:
            raise RuntimeError("gradient accumulator must be sealed before optimizer")
        if self._resident_tokens is None:
            raise RuntimeError("last gradient contribution is not resident")
        if self._expected_global_tokens is None:
            raise RuntimeError("optimizer gradients lack global token provenance")
        if self._flush_gradients is not None:
            self._flush_gradients(self.model_chunks)
        gradients = self._main_gradients()
        local_tokens = self._resident_tokens
        if self._saved_gradients is not None:
            assert self._saved_tokens is not None
            for gradient, saved in zip(gradients, self._saved_gradients, strict=True):
                gradient.add_(saved)
            local_tokens = local_tokens + self._saved_tokens
        assert self._reduction is not None
        return AccumulatedGradientSums(
            gradients=gradients,
            local_token_count=local_tokens,
            expected_global_token_count=self._expected_global_tokens,
            reduction=self._reduction,
        )

    def consume(self) -> tuple[str, ...]:
        if self._sealed is None:
            raise RuntimeError("cannot consume an unsealed gradient accumulator")
        consumed = self._sealed
        self._clear()
        return consumed

    def discard(self) -> None:
        for chunk in self.model_chunks:
            zero_grad_buffer = getattr(chunk, "zero_grad_buffer", None)
            if not callable(zero_grad_buffer):
                raise TypeError(
                    f"{type(chunk).__name__} has no zero_grad_buffer method"
                )
            zero_grad_buffer()
        self._clear()

    def _clear(self) -> None:
        self._operation_ids.clear()
        self._saved_gradients = None
        self._saved_tokens = None
        self._resident_tokens = None
        self._expected_global_tokens = None
        self._expects_global_tokens = None
        self._reduction = None
        self._sealed = None

    def _main_gradients(self) -> tuple[torch.Tensor, ...]:
        seen: set[int] = set()
        gradients: list[torch.Tensor] = []
        for chunk in self.model_chunks:
            for parameter in chunk.parameters():
                if not parameter.requires_grad or id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                value = getattr(parameter, "main_grad", None)
                if value is None:
                    raise RuntimeError("trainable Megatron parameter has no main_grad")
                gradients.append(
                    cast(
                        torch.Tensor,
                        value._local_tensor
                        if hasattr(value, "_local_tensor")
                        else value,
                    )
                )
        if (
            self._layout
            and tuple(id(gradient) for gradient in gradients) != self._layout
        ):
            raise RuntimeError("trainable Megatron gradient layout changed")
        return tuple(gradients)
