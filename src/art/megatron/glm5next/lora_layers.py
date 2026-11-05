"""Runtime bindings to M2's separate KDA projections and M3's GLM52 interface."""

from __future__ import annotations

from collections.abc import Sequence
import math
import re
from typing import Any

import torch

from art.megatron.lora import LoRA, LoraLayout

from .lora import ART_PREFIX, HEAD_PREFIX, LoraFlags


def _projection_hook(
    linear: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any
) -> Any:
    x = args[0] if args else kwargs["input"]
    residual = linear.lora(x)
    return (
        (output[0] + residual, output[1])
        if isinstance(output, tuple)
        else output + residual
    )


def _attach_projection(linear: Any, prefix: str, rank: int, alpha: int) -> None:
    # Keep q_proj.weight/output_layer.weight unchanged for the Bridge registry.
    # The registered ART LoRA child still participates in slots and publication.
    weight = linear.weight
    linear.add_module(
        "lora",
        LoRA(
            prefix,
            weight.shape[1],
            weight.shape[0],
            rank,
            alpha,
            weight.dtype,
            weight.device,
        ),
    )
    linear.register_forward_hook(_projection_hook, with_kwargs=True)


def install_glm5next_lora(
    model_chunks: Sequence[Any],
    provider: Any,
    *,
    rank: int,
    alpha: int,
    flags: LoraFlags,
) -> None:
    from art.megatron.glm52.lora import apply_glm52_attention_lora
    from art.megatron.lora import (
        wrap_dense_mlp,
        wrap_grouped_moe_experts,
        wrap_shared_experts_mlp,
    )

    if int(getattr(provider, "tensor_model_parallel_size", 1)) != 1:
        raise ValueError("GLM MVP projection LoRA requires TP1")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("GLM alpha must be finite and positive")
    for chunk in model_chunks:
        if any(isinstance(m, LoRA) for m in chunk.modules()):
            raise ValueError("GLM LoRA is already installed")
    if not any(vars(flags).values()):
        for chunk in model_chunks:
            for param in chunk.parameters():
                param.requires_grad_(False)
        return
    layers: list[Any] = []
    heads: list[tuple[Any, str]] = []
    for chunk in model_chunks:
        for name, module in list(chunk.named_modules()):
            # The attention also has layer_number; only decoder layers own mlp.
            if re.search(r"(?:^|\.)decoder\.layers\.\d+$", name) and hasattr(
                module, "mlp"
            ):
                layers.append(module)
            if hasattr(module, "output_layer") and module.output_layer is not None:
                heads.append((module, "output_layer"))
    if (flags.train_attn or flags.train_mlp) and not layers:
        raise ValueError("No GLM decoder layers found for LoRA")
    if flags.train_unembed and len(heads) != 1:
        raise ValueError("GLM LoRA requires one untied output_layer (PP1)")
    # Validate projection IFACEs before modifying any model module.
    for layer in layers:
        attention = layer.self_attention
        if flags.train_attn and not (
            all(hasattr(attention, n) for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
            or all(
                hasattr(attention, n)
                for n in ("linear_q_down_proj", "linear_kv_down_proj", "linear_proj")
            )
        ):
            raise ValueError(
                "GLM attention does not expose the M2/M3 projection interface"
            )
    for chunk in model_chunks:
        for param in chunk.parameters():
            param.requires_grad_(False)
    for layer in layers:
        prefix = f"{ART_PREFIX}{layer.layer_number - 1}"
        attention = layer.self_attention
        if flags.train_attn:
            if hasattr(attention, "q_proj"):
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    _attach_projection(
                        getattr(attention, name),
                        f"{prefix}.self_attn.{name}",
                        rank,
                        alpha,
                    )
            else:
                apply_glm52_attention_lora(
                    attention,
                    adapter_model_prefix=prefix,
                    provider=provider,
                    target_modules={"q_a_proj", "kv_a_proj_with_mqa", "o_proj"},
                    rank=rank,
                    alpha=alpha,
                )
        if flags.train_mlp:
            targets = {"gate_proj", "up_proj", "down_proj"}
            experts = getattr(layer.mlp, "experts", None)
            if experts is None:
                wrap_dense_mlp(
                    layer.mlp,
                    adapter_model_prefix=prefix,
                    provider=provider,
                    target_modules=targets,
                    rank=rank,
                    alpha=alpha,
                )
            else:
                wrap_grouped_moe_experts(
                    experts,
                    adapter_model_prefix=prefix,
                    target_modules=targets,
                    rank=rank,
                    alpha=alpha,
                    fused_gate_up=False,
                    lora_layout=LoraLayout(expert="shared_outer"),
                )
                wrap_shared_experts_mlp(
                    layer.mlp.shared_experts,
                    adapter_model_prefix=prefix,
                    provider=provider,
                    target_modules=targets,
                    rank=rank,
                    alpha=alpha,
                )
    if flags.train_unembed:
        parent, attr = heads[0]
        _attach_projection(getattr(parent, attr), HEAD_PREFIX, rank, alpha)
    # lora_publish records the first LoRA module's layout in the artifact.
    # Attention/head/dense modules must carry the same contract as experts.
    for chunk in model_chunks:
        for module in chunk.modules():
            if isinstance(module, LoRA):
                module.lora_layout = LoraLayout(expert="shared_outer")


def build_glm5next_adapter_weights_by_base(
    model_chunks: Sequence[Any],
) -> dict[str, list[Any]]:
    """M4: Bridge's merged-base exporter, including KDA and the untied head."""
    from art.megatron.glm52.lora import add_glm52_attention_adapter_weights
    from art.megatron.weights.adapter_export import (
        _simple_adapter_weight,
        add_dense_mlp_adapter_weights,
        add_grouped_moe_adapter_weights,
        add_shared_experts_adapter_weights,
        layer_base_prefix,
    )
    from art.megatron.weights.param_name_canonicalization import (
        canonical_art_param_name,
    )

    result: dict[str, list[Any]] = {}
    for chunk in model_chunks:
        for name, module in chunk.named_modules():
            if re.search(r"(?:^|\.)decoder\.layers\.\d+$", name) and hasattr(
                module, "mlp"
            ):
                prefix = layer_base_prefix(module, module_name=name)
                attention = module.self_attention
                if hasattr(attention, "q_proj"):
                    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                        lora = getattr(getattr(attention, projection), "lora", None)
                        if isinstance(lora, LoRA):
                            base = f"{prefix}.self_attention.{projection}"
                            result[base + ".weight"] = [
                                _simple_adapter_weight(base, lora)
                            ]
                else:
                    add_glm52_attention_adapter_weights(
                        result, layer_prefix=prefix, attention=attention
                    )
                experts = getattr(module.mlp, "experts", None)
                if experts is None:
                    add_dense_mlp_adapter_weights(
                        result, layer_prefix=prefix, mlp=module.mlp
                    )
                else:
                    add_grouped_moe_adapter_weights(
                        result, layer_prefix=prefix, experts=experts
                    )
                    add_shared_experts_adapter_weights(
                        result,
                        layer_prefix=prefix,
                        shared_experts=module.mlp.shared_experts,
                    )
            if name.endswith("output_layer") and isinstance(
                getattr(module, "lora", None), LoRA
            ):
                base = canonical_art_param_name(name)
                result[base + ".weight"] = [_simple_adapter_weight(base, module.lora)]
    return result
