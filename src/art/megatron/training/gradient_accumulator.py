from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
import torch

from .model_chunks import ModelChunks

GradientReduction = Literal["token_mean", "sum"]


class GradientAccumulator(BaseModel):
    """Fold Megatron F/B contributions into one O(parameter-bytes) buffer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_chunks: ModelChunks
    _layout: tuple[int, ...] = PrivateAttr(default=())
    _operation_ids: list[str] = PrivateAttr(default_factory=list)
    _saved_gradients: tuple[torch.Tensor, ...] | None = PrivateAttr(default=None)
    _saved_tokens: torch.Tensor | None = PrivateAttr(default=None)
    _resident_tokens: torch.Tensor | None = PrivateAttr(default=None)
    _reduction: GradientReduction | None = PrivateAttr(default=None)
    _sealed: tuple[str, ...] | None = PrivateAttr(default=None)

    def model_post_init(self, _context: Any) -> None:
        gradients = self._main_gradients()
        if not gradients:
            raise RuntimeError("gradient accumulator requires trainable parameters")
        self._layout = tuple(id(grad) for grad in gradients)

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
        gradients = self._main_gradients()
        tokens = self._resident_tokens
        if self._saved_gradients is None:
            saved = tuple(grad.clone() for grad in gradients)
            saved_tokens = tokens.clone()
            if self._reduction == "token_mean":
                for grad in saved:
                    grad.mul_(tokens)
            self._saved_gradients = saved
            self._saved_tokens = saved_tokens
        else:
            assert self._saved_tokens is not None
            for saved, grad in zip(self._saved_gradients, gradients, strict=True):
                if self._reduction == "token_mean":
                    grad.mul_(tokens)
                saved.add_(grad)
            self._saved_tokens.add_(tokens)
        self._resident_tokens = None

    def record(
        self,
        operation_id: str,
        token_count: torch.Tensor,
        *,
        reduction: GradientReduction = "token_mean",
    ) -> None:
        if self._sealed is not None:
            raise RuntimeError("cannot add gradients to a sealed accumulator")
        if operation_id in self._operation_ids:
            raise RuntimeError(f"duplicate gradient contribution {operation_id!r}")
        if token_count.numel() != 1:
            raise ValueError("gradient contribution token_count must be scalar")
        if self._resident_tokens is not None:
            raise RuntimeError("resident gradients must be stashed before another F/B")
        if self._reduction is not None and self._reduction != reduction:
            raise RuntimeError("one gradient accumulator cannot mix reduction modes")
        self._operation_ids.append(operation_id)
        self._resident_tokens = token_count.detach().clone()
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

    def prepare_optimizer(self) -> None:
        if self._sealed is None:
            raise RuntimeError("gradient accumulator must be sealed before optimizer")
        if self._resident_tokens is None:
            raise RuntimeError("last gradient contribution is not resident")
        if self._saved_gradients is None:
            return
        main_grads = self._main_gradients()
        if self._reduction == "sum":
            for grad, saved in zip(main_grads, self._saved_gradients, strict=True):
                grad.add_(saved)
            return
        assert self._saved_tokens is not None
        total_tokens = self._saved_tokens + self._resident_tokens
        for grad in main_grads:
            grad.mul_(self._resident_tokens)
        for grad, saved in zip(main_grads, self._saved_gradients, strict=True):
            grad.add_(saved)
            grad.div_(total_tokens)

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
        if self._layout and tuple(id(grad) for grad in gradients) != self._layout:
            raise RuntimeError("trainable Megatron gradient layout changed")
        return tuple(gradients)


class ParameterGradientAccumulator(BaseModel):
    """Fold dynamic-parameter gradients into one O(parameter-bytes) buffer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    parameters: tuple[torch.nn.Parameter, ...]
    _layout: tuple[tuple[int, ...], ...] = PrivateAttr(default=())
    _operation_ids: list[str] = PrivateAttr(default_factory=list)
    _gradients: tuple[torch.Tensor, ...] | None = PrivateAttr(default=None)
    _token_count: torch.Tensor | None = PrivateAttr(default=None)
    _reduction: GradientReduction | None = PrivateAttr(default=None)
    _prepared: bool = PrivateAttr(default=False)
    _sealed: tuple[str, ...] | None = PrivateAttr(default=None)

    def model_post_init(self, _context: Any) -> None:
        if not self.parameters:
            raise RuntimeError("gradient accumulator requires trainable parameters")
        self._layout = tuple(tuple(param.shape) for param in self.parameters)

    @property
    def contribution_ids(self) -> tuple[str, ...]:
        return tuple(self._operation_ids)

    def residency_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            *((self._token_count,) if self._token_count is not None else ()),
            *(self._gradients or ()),
        )

    def before_forward_backward(self) -> None:
        if self._sealed is not None:
            raise RuntimeError("cannot add gradients to a sealed accumulator")
        for param in self.parameters:
            param.grad = None

    def record(
        self,
        operation_id: str,
        token_count: torch.Tensor,
        gradients: tuple[torch.Tensor, ...],
        *,
        reduction: GradientReduction = "token_mean",
    ) -> None:
        if self._sealed is not None:
            raise RuntimeError("cannot add gradients to a sealed accumulator")
        if operation_id in self._operation_ids:
            raise RuntimeError(f"duplicate gradient contribution {operation_id!r}")
        if token_count.numel() != 1:
            raise ValueError("gradient contribution token_count must be scalar")
        if tuple(tuple(grad.shape) for grad in gradients) != self._layout:
            raise RuntimeError("gradient contribution layout changed")
        if self._reduction is not None and self._reduction != reduction:
            raise RuntimeError("one gradient accumulator cannot mix reduction modes")
        if self._prepared:
            raise RuntimeError("cannot add gradients after optimizer preparation")
        owned = tuple(grad.detach() for grad in gradients)
        if reduction == "token_mean":
            for grad in owned:
                grad.mul_(token_count)
        if self._gradients is None:
            self._gradients = owned
            self._token_count = token_count.detach().clone()
        else:
            assert self._token_count is not None
            for target, grad in zip(self._gradients, owned, strict=True):
                target.add_(grad)
            self._token_count.add_(token_count)
        self._operation_ids.append(operation_id)
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

    def prepare_optimizer(self) -> tuple[torch.Tensor, ...]:
        if self._sealed is None:
            raise RuntimeError("gradient accumulator must be sealed before optimizer")
        if self._gradients is None or self._token_count is None:
            raise RuntimeError("gradient accumulator is empty")
        if not self._prepared and self._reduction == "token_mean":
            for grad in self._gradients:
                grad.div_(self._token_count)
        self._prepared = True
        return self._gradients

    def consume(self) -> tuple[str, ...]:
        if self._sealed is None:
            raise RuntimeError("cannot consume an unsealed gradient accumulator")
        consumed = self._sealed
        self._clear()
        return consumed

    def discard(self) -> None:
        for param in self.parameters:
            param.grad = None
        self._clear()

    def _clear(self) -> None:
        self._operation_ids.clear()
        self._gradients = None
        self._token_count = None
        self._reduction = None
        self._prepared = False
        self._sealed = None
