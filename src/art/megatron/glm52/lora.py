from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
import torch

from art.megatron.kernels.cute_grouped_lora_quack import quack_grouped_lora_residual
from art.megatron.lora import (
    GRAD_SYNC_OP_SUM,
    LORA_ALPHA,
    TP_DEFAULT_GRAD_SYNC_DOMAIN,
    LoRA,
    LoRAParallelSpec,
    MLPExpertsLinearFC1LoRA,
    MLPExpertsLinearFC2LoRA,
    SelfAttentionLinearProjLoRA,
    _parallel_lora,
    _targets_include,
    _unwrap_attr,
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
        grad_sync_op=GRAD_SYNC_OP_SUM,
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
    if _targets_include(target_modules, "q_b_proj"):
        linear = attention.linear_q_up_proj
        attention.q_b_lora = _parallel_lora(
            adapter_model_prefix=f"{prefix}.q_b_proj",
            linear=linear,
            out_features=linear.weight.shape[0],
            rank=rank,
            alpha=alpha,
            layout="column",
        )
    if _targets_include(target_modules, "o_proj"):
        attention.linear_proj = SelfAttentionLinearProjLoRA(
            adapter_model_prefix=f"{prefix}.o_proj",
            linear_proj=attention.linear_proj,
            rank=rank,
            alpha=alpha,
            provider=provider,
        )


def _expert_lora_residual(
    base: torch.Tensor,
    x: torch.Tensor,
    lora: LoRA,
    tokens_per_expert: list[int] | torch.Tensor,
) -> torch.Tensor:
    active = lora.active_lora_tensors()
    if active is None or x.shape[0] == 0:
        return base
    a_t, b_t, scale = active
    return quack_grouped_lora_residual(
        base, x, a_t, b_t, tokens_per_expert, scale=scale
    )


def _grouped_linear(
    linear: Any, x: torch.Tensor, tokens_per_expert: list[int] | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return cast(Callable[..., tuple[torch.Tensor, torch.Tensor | None]], linear)(
        x, tokens_per_expert
    )


class Glm52MLPExpertsLinearFC1LoRA(MLPExpertsLinearFC1LoRA):
    def forward(
        self, x: torch.Tensor, tokens_per_expert: list[int] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        base, bias = _grouped_linear(self.linear_fc1, x, tokens_per_expert)
        return _expert_lora_residual(base, x, self.lora, tokens_per_expert), bias


class Glm52MLPExpertsLinearFC2LoRA(MLPExpertsLinearFC2LoRA):
    def forward(
        self, x: torch.Tensor, tokens_per_expert: list[int] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        base, bias = _grouped_linear(self.linear_fc2, x, tokens_per_expert)
        return _expert_lora_residual(base, x, self.lora, tokens_per_expert), bias


def wrap_glm52_grouped_moe_experts_3d(
    experts: Any,
    *,
    adapter_model_prefix: str,
    target_modules: set[str],
    rank: int,
    alpha: int,
) -> None:
    if not _targets_include(target_modules, "experts"):
        return
    experts.linear_fc1 = Glm52MLPExpertsLinearFC1LoRA(
        adapter_model_prefix=f"{adapter_model_prefix}.mlp.experts",
        linear_fc1=_unwrap_attr(
            experts.linear_fc1,
            "linear_fc1",
            TEColumnParallelGroupedLinear,  # type: ignore[arg-type]
        ),
        rank=rank,
        alpha=alpha,
        num_local_experts=experts.num_local_experts,
        fused_gate_up=True,
    )
    experts.linear_fc2 = Glm52MLPExpertsLinearFC2LoRA(
        adapter_model_prefix=f"{adapter_model_prefix}.mlp.experts",
        linear_fc2=_unwrap_attr(
            experts.linear_fc2,
            "linear_fc2",
            TERowParallelGroupedLinear,  # type: ignore[arg-type]
        ),
        rank=rank,
        alpha=alpha,
        num_local_experts=experts.num_local_experts,
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
    ):
        lora = getattr(attention, attr)
        if lora is not None:
            base_prefix = f"{prefix}.{base_name}"
            adapter_weights_by_base[f"{base_prefix}.weight"] = [
                _simple_adapter_weight(base_prefix, lora)
            ]
