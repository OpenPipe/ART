from collections.abc import Callable
import functools
from typing import Any

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F

from ..segmented_lora import slice_lora_segments
from .layout import NumericalPolicy, current_layout


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
    original: Callable[..., Any],
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
        pieces = []
        for part in layout.slices():
            with slice_lora_segments(part.start, part.stop):
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


def _module_types() -> tuple[tuple[type, ...], tuple[type, ...], tuple[type, ...]]:
    # Keep optional Megatron/CUDA imports out of the layout-only CPU API.
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
    import transformer_engine.pytorch as te

    from art.megatron.context_parallel.core_attention import (
        ArtContextParallelCoreAttention,
    )
    from art.megatron.flex_attn.attention import FlexDotProductAttention
    from art.megatron.lora import LoRA, LoRASlot

    return (
        (
            LoRA,
            TEColumnParallelLinear,
            TELayerNormColumnParallelLinear,
            TERowParallelLinear,
            te.Linear,
            te.LayerNormLinear,
            te.LayerNorm,
            te.RMSNorm,
        ),
        (FlexDotProductAttention, ArtContextParallelCoreAttention),
        (GatedDeltaNet, VocabParallelEmbedding, torch.nn.Embedding, LoRASlot),
    )


def install(
    model: torch.nn.Module,
    *,
    numerical_policy: NumericalPolicy = "canonical",
) -> dict[str, str]:
    """Wrap supported dense row/attention modules; return {path: wrapper kind}.

    Validate the complete layout before mutating it. Reinstallation reports the
    same modules without wrapping twice. Native installation returns an empty
    report. Parameter and optimizer identities are preserved. The model's
    ``output_layer`` stays native: canonical callers compute their own head
    from hidden states.
    """
    if numerical_policy == "native":
        return {}
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
    row_types, attention_types, parameter_types = _module_types()
    gdn_convolutions = {
        id(module.conv1d)
        for module in model.modules()
        if isinstance(module, parameter_types) and hasattr(module, "conv1d")
    }
    head = getattr(model, "output_layer", None)
    report = {}
    unsupported = []
    for name, module in model.named_modules():
        if module is head:
            continue
        core_attention = getattr(module, "core_attention", None)
        if core_attention is not None and not isinstance(
            core_attention, attention_types
        ):
            unsupported.append(
                f"{name}.core_attention ({type(core_attention).__name__})"
            )
        if isinstance(module, attention_types):
            if (
                getattr(module, "softmax_offset", None) is not None
                or getattr(module, "art_sliding_window", None) is not None
            ):
                unsupported.append(f"{name} (attention sinks or sliding windows)")
            report[name] = "attention"
        elif isinstance(module, row_types) and not getattr(module, "is_expert", False):
            report[name] = "row"
        elif (
            any(True for _ in module.parameters(recurse=False))
            and not isinstance(module, parameter_types)
            and id(module) not in gdn_convolutions
        ):
            unsupported.append(f"{name or '<root>'} ({type(module).__name__})")
    if unsupported or not report:
        raise ValueError(
            "unsupported canonical layout: "
            + ", ".join(unsupported or ["no supported operators"])
        )
    for name, module in model.named_modules():
        if name not in report or getattr(module, "_art_canonical_installed", False):
            continue
        module.forward = (
            _attention_forward(module, module.forward)
            if report[name] == "attention"
            else _row_forward(module.forward)
        )
        setattr(module, "_art_canonical_installed", True)
    return report
