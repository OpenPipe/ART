from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Any, Iterator

from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from art.loss import AlignedLossInputs, LossInputs
from art.training.tokenized import TokenizedLossName

_ENABLE_ENV = "ART_MEGATRON_SELECTIVE_LM_HEAD"
_ROW_ALIGNMENT = 16


class LmHeadTokenSelection(BaseModel):
    """Rows projected by the LM head, derived from already-shifted labels."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    flat_indices: torch.Tensor
    full_shape: tuple[int, int]
    logical_row_count: int = Field(ge=0)
    alignment_padding_rows: int = Field(ge=0, le=_ROW_ALIGNMENT)

    @model_validator(mode="after")
    def _validate_rows(self) -> "LmHeadTokenSelection":
        if self.flat_indices.ndim != 1 or self.flat_indices.dtype != torch.long:
            raise ValueError("LM-head row indices must be a flat int64 tensor")
        if int(self.flat_indices.numel()) != self.logical_row_count:
            raise ValueError("LM-head logical row metadata does not match indices")
        if self.logical_row_count > self.full_shape[0] * self.full_shape[1]:
            raise ValueError("LM-head logical row count exceeds the label shape")
        if self.projected_row_count and self.projected_row_count % _ROW_ALIGNMENT:
            raise ValueError("LM-head projected rows must satisfy row alignment")
        return self

    @property
    def projected_row_count(self) -> int:
        return self.logical_row_count + self.alignment_padding_rows

    @classmethod
    def from_labels(
        cls,
        labels: torch.Tensor,
        *,
        target_device: torch.device | None = None,
    ) -> "LmHeadTokenSelection":
        if labels.ndim != 2:
            raise ValueError(
                f"LM-head labels must be [B, S], got {tuple(labels.shape)}"
            )
        flat_labels = labels.reshape(-1)
        indices = torch.nonzero(flat_labels != -100, as_tuple=False).reshape(-1)
        logical_row_count = int(indices.numel())
        pad_rows = (
            _ROW_ALIGNMENT
            if labels.numel() and not logical_row_count
            else -logical_row_count % _ROW_ALIGNMENT
        )
        if target_device is not None:
            indices = indices.to(device=target_device, non_blocking=True)
        return cls(
            flat_indices=indices.to(dtype=torch.long).contiguous(),
            full_shape=(int(labels.shape[0]), int(labels.shape[1])),
            logical_row_count=logical_row_count,
            alignment_padding_rows=pad_rows,
        )

    @classmethod
    def from_mask(
        cls,
        mask: torch.Tensor,
        *,
        target_device: torch.device | None = None,
    ) -> "LmHeadTokenSelection":
        if mask.ndim != 2 or mask.dtype != torch.bool:
            raise ValueError(
                f"LM-head projection mask must be a [B, S] bool tensor, got "
                f"{tuple(mask.shape)} {mask.dtype}"
            )
        return cls.from_labels(
            torch.where(mask, torch.zeros_like(mask, dtype=torch.long), -100),
            target_device=target_device,
        )

    def _select_flat(
        self,
        tensor: torch.Tensor,
        *,
        padding_value: int | float | bool,
    ) -> torch.Tensor:
        selected = tensor.index_select(0, self.flat_indices)
        if not self.alignment_padding_rows:
            return selected
        # Synthetic rows keep the projection shape stable without connecting
        # alignment-only work to any logical token's gradient.
        padding = tensor.new_full(
            (self.alignment_padding_rows, *tensor.shape[1:]), padding_value
        )
        return torch.cat((selected, padding), dim=0)

    def select(
        self,
        tensor: torch.Tensor,
        *,
        padding_value: int | float | bool = 0,
    ) -> torch.Tensor:
        expected = self.full_shape[0] * self.full_shape[1]
        if tensor.numel() != expected:
            raise ValueError(
                "selected token tensor must match the label shape: "
                f"tensor={tuple(tensor.shape)} labels={self.full_shape}"
            )
        return self._select_flat(
            tensor.reshape(-1), padding_value=padding_value
        ).unsqueeze(0)

    def select_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return self.select(labels, padding_value=-100)

    def select_optional(
        self,
        tensor: torch.Tensor | None,
        *,
        padding_value: int | float | bool = 0,
    ) -> torch.Tensor | None:
        return (
            None if tensor is None else self.select(tensor, padding_value=padding_value)
        )

    def select_rows(
        self,
        tensor: torch.Tensor,
        *,
        padding_value: int | float | bool = 0,
    ) -> torch.Tensor:
        if tuple(tensor.shape[:2]) != self.full_shape:
            raise ValueError(
                "selected row tensor must match the label shape: "
                f"tensor={tuple(tensor.shape)} labels={self.full_shape}"
            )
        return self._select_flat(
            tensor.reshape(-1, *tensor.shape[2:]), padding_value=padding_value
        )

    def restore(self, tensor: torch.Tensor, *, fill_value: float = 0.0) -> torch.Tensor:
        if tensor.numel() != self.projected_row_count:
            raise ValueError(
                "selected tensor length does not match LM-head selection: "
                f"tensor={tensor.numel()} selection={self.projected_row_count}"
            )
        restored = tensor.new_full(self.full_shape, fill_value)
        restored.reshape(-1).index_copy_(
            0,
            self.flat_indices,
            tensor.reshape(-1)[: self.logical_row_count],
        )
        return restored

    def restore_rows(
        self, tensor: torch.Tensor, *, fill_value: float = 0.0
    ) -> torch.Tensor:
        if int(tensor.shape[0]) != self.projected_row_count:
            raise ValueError(
                "selected tensor rows do not match LM-head selection: "
                f"tensor={tensor.shape[0]} selection={self.projected_row_count}"
            )
        restored = tensor.new_full((*self.full_shape, *tensor.shape[1:]), fill_value)
        restored.reshape(-1, *tensor.shape[1:]).index_copy_(
            0, self.flat_indices, tensor[: self.logical_row_count]
        )
        return restored

    def compact_loss_inputs(
        self,
        inputs: LossInputs | AlignedLossInputs,
    ) -> AlignedLossInputs:
        aligned = inputs.align_inputs()
        return aligned.model_copy(
            update={
                "assistant_mask": self.select(aligned.assistant_mask),
                "old_logprobs": self.select(aligned.old_logprobs),
                "advantages": self.select(aligned.advantages),
                "weights": self.select(aligned.weights),
                "group_ids": self.select(aligned.group_ids),
                "original_logprobs": self.select_optional(aligned.original_logprobs),
                "entropies_are_aligned": True,
            }
        )


def tokenized_lm_head_masks(
    *,
    target_tokens: torch.Tensor,
    loss_weights: torch.Tensor | None,
    token_advantages: torch.Tensor | None,
    loss_name: TokenizedLossName | None,
    return_token_logprobs: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return projection and loss-active masks for exact tokenized targets."""
    if target_tokens.ndim != 3:
        raise ValueError(
            f"tokenized targets must be [B, S, K], got {tuple(target_tokens.shape)}"
        )
    valid = target_tokens >= 0
    if loss_name is None:
        active = valid.any(dim=-1)
    else:
        coefficients = (
            loss_weights if loss_name == "cross_entropy" else token_advantages
        )
        if coefficients is None or coefficients.shape != target_tokens.shape:
            raise ValueError(f"{loss_name} coefficients must match target tokens")
        active = (valid & (coefficients != 0)).any(dim=-1)
    projected = active | (valid.any(dim=-1) if return_token_logprobs else False)
    return projected, active


class TokenLossOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    token_losses: torch.Tensor
    selection: LmHeadTokenSelection | None = None

    def select(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if self.selection is None else self.selection.select(tensor)

    def select_optional(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        return (
            tensor if self.selection is None else self.selection.select_optional(tensor)
        )

    def compact_loss_inputs(
        self,
        inputs: LossInputs | AlignedLossInputs,
    ) -> LossInputs | AlignedLossInputs:
        if self.selection is None:
            return inputs
        return self.selection.compact_loss_inputs(inputs)

    def restore(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if self.selection is None else self.selection.restore(tensor)

    def masked_sum(self, mask: torch.Tensor) -> torch.Tensor:
        selected_mask = self.select(mask).to(dtype=torch.bool)
        return self.token_losses[selected_mask].sum() + self.token_losses.sum() * 0.0


def selective_lm_head_enabled() -> bool:
    raw = os.environ.get(_ENABLE_ENV, "1").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{_ENABLE_ENV} must be a boolean, got {raw!r}")


def forward_token_losses(
    model: torch.nn.Module,
    *,
    labels: torch.Tensor,
    selection: LmHeadTokenSelection,
    forward_kwargs: dict[str, Any],
    enabled: bool | None = None,
) -> TokenLossOutput:
    """Run the normal model path while projecting only labeled token rows.

    Sequence-parallel hidden states are gathered before selection, matching the
    communication performed by Megatron's output linear. The output linear's
    own gather is disabled for this call; gather autograd performs the matching
    reduce-scatter in backward.
    """
    if "labels" in forward_kwargs:
        raise ValueError("forward_kwargs must not contain labels")
    if enabled is None:
        enabled = selective_lm_head_enabled()
    if not enabled:
        return TokenLossOutput(
            token_losses=model(**forward_kwargs, labels=labels),
        )
    if tuple(labels.shape) != selection.full_shape:
        raise ValueError(
            f"labels={tuple(labels.shape)} selection={selection.full_shape}"
        )

    language_model = _language_model(model)
    _validate_language_model(language_model)
    if not labels.numel():
        with _select_output_rows(language_model.output_layer, selection):
            logits = model(**forward_kwargs, labels=None)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"model must return logits, got {type(logits).__name__}")
        return TokenLossOutput(
            token_losses=_empty_token_losses(logits, labels),
            selection=selection,
        )
    compact_labels = selection.select_labels(labels)
    with _select_output_rows(language_model.output_layer, selection):
        with _restore_root_output(model, selection):
            token_losses = model(**forward_kwargs, labels=compact_labels)
    if not isinstance(token_losses, torch.Tensor):
        raise TypeError(
            f"model must return token losses, got {type(token_losses).__name__}"
        )
    return TokenLossOutput(
        token_losses=selection.select(token_losses),
        selection=selection,
    )


def forward_token_logits(
    model: torch.nn.Module,
    *,
    selection: LmHeadTokenSelection,
    forward_kwargs: dict[str, Any],
) -> torch.Tensor:
    """Project only selected token rows and return local vocabulary logits."""
    if "labels" in forward_kwargs:
        raise ValueError("forward_kwargs must not contain labels")
    language_model = _language_model(model)
    _validate_language_model(language_model)
    with _select_output_rows(language_model.output_layer, selection):
        logits = model(**forward_kwargs, labels=None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise TypeError("selected model output must be a [tokens, batch, vocab] tensor")
    selected_tokens = selection.projected_row_count
    if tuple(logits.shape[:2]) == (selected_tokens, 1):
        return logits[:, 0, :].contiguous()
    if tuple(logits.shape[:2]) == (1, selected_tokens):
        return logits[0, :, :].contiguous()
    raise ValueError(
        "selected logits do not match LM-head selection: "
        f"logits={tuple(logits.shape)} selected_tokens={selected_tokens}"
    )


def selected_target_logprobs(
    model: torch.nn.Module,
    local_logits: torch.Tensor,
    target_tokens: torch.Tensor,
) -> torch.Tensor:
    """Return exact selected target logprobs without dense FP32 normalization."""
    if local_logits.ndim != 2 or target_tokens.ndim != 2:
        raise ValueError("selected logprobs require [N,V] logits and [N,K] targets")
    if int(local_logits.shape[0]) != int(target_tokens.shape[0]):
        raise ValueError("selected logprob logits and targets do not align")
    if int(target_tokens.shape[1]) != 1:
        from art.megatron.multi_target_logprobs import (
            vocab_parallel_multi_target_logprobs,
        )

        return vocab_parallel_multi_target_logprobs(local_logits, target_tokens)

    language_model = _language_model(model)
    _validate_language_model(language_model)
    if getattr(language_model.config, "cross_entropy_fusion_impl", None) != "te":
        raise RuntimeError("single-target selected logprobs require TE cross entropy")
    valid = target_tokens[:, 0] >= 0
    labels = target_tokens[:, 0].masked_fill(~valid, 0).reshape(1, -1)
    losses = language_model.compute_language_model_loss(
        labels=labels,
        logits=local_logits.unsqueeze(1),
    )
    return (-losses.reshape(-1, 1)).masked_fill(~valid.unsqueeze(1), 0.0)


def _language_model(model: torch.nn.Module) -> Any:
    module: Any = model
    seen: set[int] = set()
    while id(module) not in seen:
        seen.add(id(module))
        if hasattr(module, "module"):
            module = module.module
            continue
        language_model = getattr(module, "language_model", None)
        if language_model is not None:
            module = language_model
            continue
        break
    if not hasattr(module, "output_layer") or not hasattr(
        module, "compute_language_model_loss"
    ):
        raise TypeError(
            "selective LM head requires a GPT-compatible language model with "
            "output_layer and compute_language_model_loss"
        )
    return module


def _validate_language_model(language_model: Any) -> None:
    if not bool(getattr(language_model, "post_process", False)):
        raise RuntimeError("selective LM head requires the post-process model stage")
    config = language_model.config
    if bool(getattr(language_model, "mtp_process", False)) or int(
        getattr(config, "mtp_num_layers", 0) or 0
    ):
        raise RuntimeError("selective LM head does not support MTP training")
    if bool(getattr(language_model.output_layer, "gather_output", False)):
        raise RuntimeError("selective LM head requires vocabulary-parallel logits")


@contextmanager
def _restore_root_output(
    model: torch.nn.Module,
    selection: LmHeadTokenSelection,
) -> Iterator[None]:
    def restore(
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        output: Any,
    ) -> torch.Tensor:
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"model must return token losses, got {type(output).__name__}"
            )
        return selection.restore(output)

    handle = model.register_forward_hook(restore, prepend=True)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def _select_output_rows(
    output_layer: torch.nn.Module,
    selection: LmHeadTokenSelection,
) -> Iterator[None]:
    sequence_parallel = bool(getattr(output_layer, "sequence_parallel", False))
    disable_grad_reduce = bool(getattr(output_layer, "disable_grad_reduce", False))
    calls = 0

    def select_rows(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        nonlocal calls
        calls += 1
        if calls != 1 or not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("selective LM head expects one positional output call")
        hidden_states = args[0]
        if sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                group=getattr(module, "tp_group"),
            )
            setattr(module, "sequence_parallel", False)
            setattr(module, "disable_grad_reduce", True)
        batch, sequence = selection.full_shape
        if tuple(hidden_states.shape[:2]) != (sequence, batch):
            raise ValueError(
                "LM-head hidden states do not match labels: "
                f"hidden={tuple(hidden_states.shape)} labels={selection.full_shape}"
            )
        selected = selection.select_rows(hidden_states.transpose(0, 1)).unsqueeze(1)
        return (selected, *args[1:]), kwargs

    handle = output_layer.register_forward_pre_hook(select_rows, with_kwargs=True)
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        handle.remove()
        if sequence_parallel:
            setattr(output_layer, "sequence_parallel", True)
            setattr(output_layer, "disable_grad_reduce", disable_grad_reduce)
    if succeeded and calls != 1:
        raise RuntimeError(f"selective LM head expected one output call, got {calls}")


def _empty_token_losses(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.numel() or logits.ndim != 3 or not logits.shape[-1]:
        raise ValueError(
            f"expected empty labels and [B, 0, V] logits, got {tuple(logits.shape)}"
        )
    losses = logits[..., 0]
    if tuple(losses.shape) == tuple(labels.shape):
        return losses
    losses = losses.transpose(0, 1).contiguous()
    if tuple(losses.shape) != tuple(labels.shape):
        raise ValueError(
            f"empty logits={tuple(logits.shape)} labels={tuple(labels.shape)}"
        )
    return losses
