from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
import functools
from typing import Any

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layout import ROOT_ALIGNMENT, NumericalPolicy, RootLayout, current_layout

RowContext = Callable[[int, int], AbstractContextManager[Any]]


def _join(values: list[Any], first_rows: int) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        if first.ndim >= 2 and first.shape[0] == first_rows:
            return torch.cat(values, dim=0)
        # TE returns a shared bias alongside row-shaped outputs.
        return first
    if isinstance(first, (tuple, list)):
        return type(first)(
            _join(list(parts), first_rows) for parts in zip(*values, strict=True)
        )
    if first is None:
        return None
    raise TypeError(f"unsupported canonical row output: {type(first)}")


def _row_forward(
    original: Callable[..., Any], *, owner: torch.nn.Module | None = None
) -> Callable[..., Any]:
    @functools.wraps(original)
    def forward(x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        layout = current_layout()
        if (
            layout is None
            or not isinstance(x, torch.Tensor)
            or x.ndim < 2
            or x.shape[0] != layout.rows
            or len(layout.lengths) == 1
        ):
            return original(x, *args, **kwargs)
        row_context = getattr(owner, "_art_canonical_row_context", None)
        pieces = []
        for part in layout.slices():
            # Segmented LoRA rebases adapter ownership alongside root rows.
            # The optional callback avoids a dependency on that gated feature.
            with row_context(part.start, part.stop) if row_context else nullcontext():
                pieces.append(original(x[part].contiguous(), *args, **kwargs))
        return _join(pieces, layout.padded_lengths[0])

    return forward


def _attention_forward(module: Any, original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Any,
        attn_mask_type: Any = None,
        attention_bias: Any = None,
        packed_seq_params: Any = None,
    ) -> torch.Tensor:
        layout = current_layout()
        if layout is None:
            return original(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type,
                attention_bias,
                packed_seq_params,
            )
        if packed_seq_params is not None:
            raise ValueError("canonical attention does not accept PackedSeqParams")
        if getattr(module, "softmax_offset", None) is not None:
            raise ValueError("canonical attention does not support attention sinks")
        if getattr(module, "art_sliding_window", None) is not None:
            raise ValueError("canonical attention does not support sliding windows")
        if any(
            x.shape[0] != layout.rows or x.shape[1] != 1 for x in (query, key, value)
        ):
            raise ValueError(
                "canonical attention requires complete roots, batch size 1"
            )
        outputs = []
        for part in layout.slices():
            q, k, v = [
                x[part].permute(1, 2, 0, 3).contiguous() for x in (query, key, value)
            ]
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=True,
                    scale=module.softmax_scale,
                    enable_gqa=q.shape[1] != k.shape[1],
                )
            outputs.append(out.permute(2, 0, 1, 3).contiguous())
        out = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        return out.reshape(out.shape[0], out.shape[1], -1)

    return forward


def install(
    model: torch.nn.Module,
    *,
    numerical_policy: NumericalPolicy = "canonical",
    row_context: RowContext | None = None,
) -> None:
    """Install context-aware row/attention wrappers once, after ART/LoRA hooks.

    ``numerical_policy`` is the trainer's only policy flag. Use native for MoE.
    The initial policy qualifies TP1/CP1 dense models with zero dropout and
    disabled full-layer compilation. Component kernel compilation is retained.
    Parameters and optimizer identities are preserved. A nonnull row_context
    rebases optional adapter metadata for each root call, and upgrades existing
    wrappers without wrapping them twice. None preserves the installed callback.
    """
    if numerical_policy == "native":
        return
    if numerical_policy != "canonical":
        raise ValueError(f"unknown numerical policy: {numerical_policy}")
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("canonical installation requires model.config")
    if getattr(config, "num_moe_experts", None):
        raise ValueError("MoE models must use native numerics")
    for name in ("tensor_model_parallel_size", "context_parallel_size"):
        if getattr(config, name, 1) != 1:
            raise ValueError(f"canonical v1 requires {name}=1")
    for name in ("hidden_dropout", "attention_dropout"):
        if getattr(config, name, 0):
            raise ValueError("canonical v1 requires zero dropout")
    for module in model.modules():
        if getattr(module, "_art_canonical_installed", False):
            if row_context is not None:
                setattr(module, "_art_canonical_row_context", row_context)
            continue
        kind = type(module).__name__
        if kind in ("FlexDotProductAttention", "ArtContextParallelCoreAttention"):
            wrapped = _attention_forward(module, module.forward)
        elif (
            (kind == "LoRA" and not getattr(module, "is_expert", False))
            or ("Norm" in kind and "Transformer" not in kind)
            or (kind.startswith("TE") and "Linear" in kind and "Grouped" not in kind)
        ):
            wrapped = _row_forward(module.forward, owner=module)
        else:
            continue
        module.forward = wrapped
        if row_context is not None:
            setattr(module, "_art_canonical_row_context", row_context)
        setattr(module, "_art_canonical_installed", True)


def target_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    layout: RootLayout,
) -> tuple[torch.Tensor, ...]:
    """FP32 head with fixed 128-row calls, independently restarted per root.

    Hidden rows/targets include padding; ignore labels with -100. The returned
    logical rows include zeros for ignored labels. Weight is normally frozen.
    """
    if hidden.ndim != 2 or hidden.shape[0] != layout.rows:
        raise ValueError("head hidden rows must match the canonical layout")
    if targets.shape != (layout.rows,):
        raise ValueError("head targets must match the canonical layout")
    fp32_weight = weight.float()

    def chunk_logprobs(h: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return -F.cross_entropy(
            F.linear(h.float(), fp32_weight),
            labels,
            reduction="none",
            ignore_index=-100,
        )

    outputs = []
    for part, length in zip(layout.slices(), layout.lengths, strict=True):
        root_hidden, root_targets = hidden[part], targets[part]
        chunks = []
        for start in range(0, len(root_hidden), ROOT_ALIGNMENT):
            h = root_hidden[start : start + ROOT_ALIGNMENT]
            labels = root_targets[start : start + ROOT_ALIGNMENT]
            # Do not retain a sequence x vocabulary FP32 softmax for backward.
            chunks.append(
                checkpoint(chunk_logprobs, h, labels, use_reentrant=False)
                if torch.is_grad_enabled() and (h.requires_grad or weight.requires_grad)
                else chunk_logprobs(h, labels)
            )
        outputs.append(torch.cat(chunks)[:length])
    return tuple(outputs)
