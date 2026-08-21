from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
import torch

from .model_chunks import ModelChunks

GradientReduction = Literal["token_mean", "sum"]


class AccumulatedGradientSums(BaseModel):
    """One rank's unnormalized gradient and token numerators."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    gradients: tuple[torch.Tensor, ...]
    local_token_count: torch.Tensor
    expected_global_token_count: int | None
    reduction: GradientReduction


class GradientAccumulator(BaseModel):
    """Fold Megatron F/B contributions into one O(parameter-bytes) buffer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_chunks: ModelChunks
    _layout: tuple[int, ...] = PrivateAttr(default=())
    _operation_ids: list[str] = PrivateAttr(default_factory=list)
    _saved_gradients: tuple[torch.Tensor, ...] | None = PrivateAttr(default=None)
    _saved_tokens: torch.Tensor | None = PrivateAttr(default=None)
    _resident_tokens: torch.Tensor | None = PrivateAttr(default=None)
    _expected_global_tokens: int | None = PrivateAttr(default=None)
    _expects_global_tokens: bool | None = PrivateAttr(default=None)
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
        if self._saved_gradients is None:
            saved = tuple(grad.clone() for grad in gradients)
            self._saved_gradients = saved
            self._saved_tokens = self._resident_tokens.clone()
        else:
            assert self._saved_tokens is not None
            for saved, grad in zip(self._saved_gradients, gradients, strict=True):
                saved.add_(grad)
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
                "one gradient accumulator cannot mix observed-only and "
                "provenance-checked contributions"
            )
        self._operation_ids.append(operation_id)
        self._resident_tokens = token_count.detach().clone()
        self._expects_global_tokens = expects_global_tokens
        if expected_global_token_count is not None:
            self._expected_global_tokens = (
                (self._expected_global_tokens or 0) + expected_global_token_count
            )
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

    def prepare_local_sums(self) -> AccumulatedGradientSums:
        if self._sealed is None:
            raise RuntimeError("gradient accumulator must be sealed before optimizer")
        if self._resident_tokens is None:
            raise RuntimeError("last gradient contribution is not resident")
        main_grads = self._main_gradients()
        local_tokens = self._resident_tokens
        if self._saved_gradients is not None:
            assert self._saved_tokens is not None
            for grad, saved in zip(main_grads, self._saved_gradients, strict=True):
                grad.add_(saved)
            local_tokens = local_tokens + self._saved_tokens
        assert self._reduction is not None
        return AccumulatedGradientSums(
            gradients=main_grads,
            local_token_count=local_tokens,
            expected_global_token_count=self._expected_global_tokens,
            reduction=self._reduction,
        )

    def prepare_optimizer(self) -> AccumulatedGradientSums:
        prepared = self.prepare_local_sums()
        if prepared.expected_global_token_count is None:
            raise RuntimeError("optimizer gradients lack global token provenance")
        return prepared

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
    _expected_global_tokens: int | None = PrivateAttr(default=None)
    _expects_global_tokens: bool | None = PrivateAttr(default=None)
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
        expected_global_token_count: int | None = None,
        reduction: GradientReduction = "token_mean",
    ) -> None:
        if tuple(tuple(grad.shape) for grad in gradients) != self._layout:
            raise RuntimeError("gradient contribution layout changed")
        self._validate_record(
            operation_id,
            token_count,
            expected_global_token_count=expected_global_token_count,
            reduction=reduction,
        )
        owned = tuple(grad.detach() for grad in gradients)
        if self._gradients is None:
            self._gradients = owned
        else:
            for target, grad in zip(self._gradients, owned, strict=True):
                target.add_(grad)
        self._commit_record(
            operation_id,
            token_count,
            expected_global_token_count=expected_global_token_count,
            reduction=reduction,
        )

    def record_parameters(
        self,
        operation_id: str,
        token_count: torch.Tensor,
        *,
        expected_global_token_count: int | None = None,
        reduction: GradientReduction = "token_mean",
    ) -> None:
        """Accumulate resident parameter grads without per-step gradient copies."""
        self._validate_record(
            operation_id,
            token_count,
            expected_global_token_count=expected_global_token_count,
            reduction=reduction,
        )
        for param in self.parameters:
            if param.grad is not None and (
                param.grad.shape != param.shape or param.grad.device != param.device
            ):
                raise RuntimeError("resident parameter gradient layout changed")
        if self._gradients is None:
            self._gradients = tuple(
                torch.zeros_like(param, dtype=torch.float32)
                for param in self.parameters
            )
        for target, param in zip(self._gradients, self.parameters, strict=True):
            if param.grad is not None:
                target.add_(param.grad)
        self._commit_record(
            operation_id,
            token_count,
            expected_global_token_count=expected_global_token_count,
            reduction=reduction,
        )

    def _validate_record(
        self,
        operation_id: str,
        token_count: torch.Tensor,
        *,
        expected_global_token_count: int | None,
        reduction: GradientReduction,
    ) -> None:
        if self._sealed is not None:
            raise RuntimeError("cannot add gradients to a sealed accumulator")
        if operation_id in self._operation_ids:
            raise RuntimeError(f"duplicate gradient contribution {operation_id!r}")
        if token_count.numel() != 1:
            raise ValueError("gradient contribution token_count must be scalar")
        if self._reduction is not None and self._reduction != reduction:
            raise RuntimeError("one gradient accumulator cannot mix reduction modes")
        if self._prepared:
            raise RuntimeError("cannot add gradients after optimizer preparation")
        expects_global_tokens = expected_global_token_count is not None
        if (
            self._expects_global_tokens is not None
            and self._expects_global_tokens != expects_global_tokens
        ):
            raise RuntimeError(
                "one gradient accumulator cannot mix observed-only and "
                "provenance-checked contributions"
            )

    def _commit_record(
        self,
        operation_id: str,
        token_count: torch.Tensor,
        *,
        expected_global_token_count: int | None,
        reduction: GradientReduction,
    ) -> None:
        expects_global_tokens = expected_global_token_count is not None
        if self._token_count is None:
            self._token_count = token_count.detach().clone()
        else:
            assert self._token_count is not None
            self._token_count.add_(token_count)
        self._operation_ids.append(operation_id)
        self._expects_global_tokens = expects_global_tokens
        if expected_global_token_count is not None:
            self._expected_global_tokens = (
                (self._expected_global_tokens or 0) + expected_global_token_count
            )
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

    def prepare_local_sums(self) -> AccumulatedGradientSums:
        if self._sealed is None:
            raise RuntimeError("gradient accumulator must be sealed before optimizer")
        if self._gradients is None or self._token_count is None:
            raise RuntimeError("gradient accumulator is empty")
        assert self._reduction is not None
        self._prepared = True
        return AccumulatedGradientSums(
            gradients=self._gradients,
            local_token_count=self._token_count,
            expected_global_token_count=self._expected_global_tokens,
            reduction=self._reduction,
        )

    def prepare_optimizer(self) -> AccumulatedGradientSums:
        prepared = self.prepare_local_sums()
        if prepared.expected_global_token_count is None:
            raise RuntimeError("optimizer gradients lack global token provenance")
        return prepared

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
        self._expected_global_tokens = None
        self._expects_global_tokens = None
        self._reduction = None
        self._prepared = False
        self._sealed = None
