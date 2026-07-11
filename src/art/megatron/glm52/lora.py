from __future__ import annotations

from typing import Any

import torch

from art.megatron.lora import (
    GRAD_SYNC_OP_NONE,
    LORA_ALPHA,
    TP_DEFAULT_GRAD_SYNC_DOMAIN,
    LoRA,
    LoRAParallelSpec,
    SelfAttentionLinearProjLoRA,
    _parallel_lora,
    _targets_include,
)


def _replicated_lora(
    linear: Any,
    *,
    adapter_model_prefix: str,
    rank: int,
    alpha: int,
) -> LoRA:
    weight = linear.weight
    parallel = LoRAParallelSpec(
        grad_sync_domain=TP_DEFAULT_GRAD_SYNC_DOMAIN,
        grad_sync_op=GRAD_SYNC_OP_NONE,
    )
    return LoRA(
        adapter_model_prefix=adapter_model_prefix,
        in_features=weight.shape[1],
        out_features=weight.shape[0],
        rank=rank,
        alpha=alpha,
        dtype=weight.dtype,
        device=weight.device,
        a_parallel_spec=parallel,
        b_parallel_spec=parallel,
        allreduce=True,
    )


def apply_glm52_attention_lora(
    attention: Any,
    *,
    adapter_model_prefix: str,
    provider: Any,
    target_modules: set[str],
    rank: int,
    alpha: int = LORA_ALPHA,
) -> None:
    prefix = f"{adapter_model_prefix}.self_attn"
    for target, attr, linear_attr in (
        ("q_a_proj", "q_a_lora", "linear_q_down_proj"),
        ("kv_a_proj_with_mqa", "kv_a_lora", "linear_kv_down_proj"),
    ):
        if _targets_include(target_modules, target):
            setattr(
                attention,
                attr,
                _replicated_lora(
                    getattr(attention, linear_attr),
                    adapter_model_prefix=f"{prefix}.{target}",
                    rank=rank,
                    alpha=alpha,
                ),
            )
    for target, attr, linear_attr in (
        ("q_b_proj", "q_b_lora", "linear_q_up_proj"),
        ("kv_b_proj", "kv_b_lora", "linear_kv_up_proj"),
    ):
        if _targets_include(target_modules, target):
            linear = getattr(attention, linear_attr)
            setattr(
                attention,
                attr,
                _parallel_lora(
                    adapter_model_prefix=f"{prefix}.{target}",
                    linear=linear,
                    out_features=linear.weight.shape[0],
                    rank=rank,
                    alpha=alpha,
                    layout="column",
                ),
            )
    if _targets_include(target_modules, "o_proj"):
        attention.linear_proj = SelfAttentionLinearProjLoRA(
            adapter_model_prefix=f"{prefix}.o_proj",
            linear_proj=attention.linear_proj,
            rank=rank,
            alpha=alpha,
            provider=provider,
        )


def add_glm52_attention_adapter_weights(
    adapter_weights_by_base: dict[str, list[Any]],
    *,
    layer_prefix: str,
    attention: Any,
) -> None:
    from art.megatron.weights.adapter_export import (
        _simple_adapter_weight,
        add_self_attention_adapter_weights,
    )

    add_self_attention_adapter_weights(
        adapter_weights_by_base,
        layer_prefix=layer_prefix,
        self_attention=attention,
    )
    prefix = f"{layer_prefix}.self_attention"
    for attr, base_name in (
        ("q_a_lora", "linear_q_down_proj"),
        ("q_b_lora", "linear_q_up_proj"),
        ("kv_a_lora", "linear_kv_down_proj"),
        ("kv_b_lora", "linear_kv_up_proj"),
    ):
        lora = getattr(attention, attr)
        if lora is not None:
            base_prefix = f"{prefix}.{base_name}"
            adapter_weights_by_base[f"{base_prefix}.weight"] = [
                _simple_adapter_weight(base_prefix, lora)
            ]
