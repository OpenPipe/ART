from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
import torch

from .model_chunks import ModelChunks


class GradientContribution(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    operation_id: str = Field(min_length=1)
    token_count: torch.Tensor
    gradients: tuple[torch.Tensor, ...] | None = None


class GradientAccumulator(BaseModel):
    """Own token-weighted, already-normalized Megatron gradient contributions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_chunks: ModelChunks
    _layout: tuple[int, ...] = PrivateAttr(default=())
    _contributions: list[GradientContribution] = PrivateAttr(default_factory=list)
    _sealed: tuple[str, ...] | None = PrivateAttr(default=None)

    def model_post_init(self, _context: Any) -> None:
        gradients = self._main_gradients()
        if not gradients:
            raise RuntimeError("gradient accumulator requires trainable parameters")
        self._layout = tuple(id(grad) for grad in gradients)

    @property
    def contribution_ids(self) -> tuple[str, ...]:
        return tuple(item.operation_id for item in self._contributions)

    def before_forward_backward(self) -> None:
        if self._sealed is not None:
            raise RuntimeError("cannot add gradients to a sealed accumulator")
        if not self._contributions:
            return
        resident = self._contributions[-1]
        if resident.gradients is not None:
            raise RuntimeError("gradient accumulator has no resident contribution")
        self._contributions[-1] = resident.model_copy(
            update={"gradients": tuple(grad.clone() for grad in self._main_gradients())}
        )

    def record(self, operation_id: str, token_count: torch.Tensor) -> None:
        if self._sealed is not None:
            raise RuntimeError("cannot add gradients to a sealed accumulator")
        if operation_id in self.contribution_ids:
            raise RuntimeError(f"duplicate gradient contribution {operation_id!r}")
        if token_count.numel() != 1:
            raise ValueError("gradient contribution token_count must be scalar")
        if self._contributions and self._contributions[-1].gradients is None:
            raise RuntimeError("resident gradients must be stashed before another F/B")
        self._contributions.append(
            GradientContribution(
                operation_id=operation_id,
                token_count=token_count.detach().clone(),
            )
        )

    def seal(self, operation_ids: tuple[str, ...]) -> None:
        if not operation_ids:
            raise RuntimeError("optimizer requires at least one F/B contribution")
        if self._sealed is not None:
            raise RuntimeError("gradient accumulator is already sealed")
        if operation_ids != self.contribution_ids:
            raise RuntimeError(
                "optimizer contribution order does not match the open accumulator"
            )
        self._sealed = operation_ids

    def prepare_optimizer(self) -> None:
        if self._sealed is None:
            raise RuntimeError("gradient accumulator must be sealed before optimizer")
        resident = self._contributions[-1]
        if resident.gradients is not None:
            raise RuntimeError("last gradient contribution must remain resident")
        if len(self._contributions) == 1:
            return

        main_grads = self._main_gradients()
        total_tokens = sum(
            (item.token_count for item in self._contributions),
            torch.zeros_like(resident.token_count),
        )
        for grad in main_grads:
            grad.mul_(resident.token_count)
        for item in self._contributions[:-1]:
            assert item.gradients is not None
            if len(item.gradients) != len(main_grads):
                raise RuntimeError("gradient contribution layout changed")
            for grad, saved in zip(main_grads, item.gradients, strict=True):
                grad.add_(saved.mul_(item.token_count))
        for grad in main_grads:
            grad.div_(total_tokens)

    def consume(self) -> tuple[str, ...]:
        if self._sealed is None:
            raise RuntimeError("cannot consume an unsealed gradient accumulator")
        consumed = self._sealed
        self._contributions.clear()
        self._sealed = None
        return consumed

    def discard(self) -> None:
        for chunk in self.model_chunks:
            zero_grad_buffer = getattr(chunk, "zero_grad_buffer", None)
            if not callable(zero_grad_buffer):
                raise TypeError(
                    f"{type(chunk).__name__} has no zero_grad_buffer method"
                )
            zero_grad_buffer()
        self._contributions.clear()
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
        if self._layout and tuple(id(grad) for grad in gradients) != self._layout:
            raise RuntimeError("trainable Megatron gradient layout changed")
        return tuple(gradients)
