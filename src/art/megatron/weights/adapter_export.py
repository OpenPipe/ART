from collections.abc import Callable, Sequence
import math
from typing import Any

from megatron.bridge.models.conversion.model_bridge import MegatronWeightTuple
from megatron.bridge.models.conversion.peft_bridge import AdapterWeight
from megatron.core.transformer.transformer_layer import TransformerLayer
import torch

from art.megatron.lora import (
    GatedDeltaNetInProjLoRA,
    LoRA,
    MLPExpertsLinearFC1LoRA,
    MLPExpertsLinearFC2LoRA,
    SelfAttentionLinearProjLoRA,
    SelfAttentionLinearQKVLoRA,
    SharedExpertsLinearFC1LoRA,
)
from art.megatron.weights.param_name_canonicalization import canonical_art_param_name


def _adapter_alpha_dim(lora: LoRA) -> tuple[int, int]:
    dim = int(lora.A_T.shape[-1])
    alpha = float(lora.scale) * dim
    rounded_alpha = round(alpha)
    assert math.isclose(alpha, rounded_alpha)
    return rounded_alpha, dim


def _adapter_tensors(
    lora: LoRA,
    expert_idx: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    a_t = lora.A_T if expert_idx is None else lora.A_T[expert_idx]
    b_t = lora.B_T if expert_idx is None else lora.B_T[expert_idx]
    return a_t.transpose(-1, -2).contiguous(), b_t.transpose(-1, -2).contiguous()


def _adapter_weight(
    *,
    base_prefix: str,
    adapter_key: str | None,
    alpha: int,
    dim: int,
    linear_in: torch.Tensor,
    linear_out: torch.Tensor,
) -> AdapterWeight:
    adapter_suffix = "" if adapter_key is None else f".{adapter_key}"
    param_prefix = f"{base_prefix}.adapter{adapter_suffix}"
    return AdapterWeight(
        global_base_prefix=base_prefix,
        adapter_key=adapter_key,
        alpha=alpha,
        dim=dim,
        linear_in_weight=MegatronWeightTuple(
            param_name=f"{param_prefix}.linear_in.weight",
            weight=linear_in,
            vp_stage=0,
        ),
        linear_out_weight=MegatronWeightTuple(
            param_name=f"{param_prefix}.linear_out.weight",
            weight=linear_out,
            vp_stage=0,
        ),
    )


def _simple_adapter_weight(
    base_prefix: str,
    lora: LoRA,
    *,
    adapter_key: str | None = None,
    expert_idx: int | None = None,
) -> AdapterWeight:
    alpha, dim = _adapter_alpha_dim(lora)
    linear_in, linear_out = _adapter_tensors(lora, expert_idx)
    return _adapter_weight(
        base_prefix=base_prefix,
        adapter_key=adapter_key,
        alpha=alpha,
        dim=dim,
        linear_in=linear_in,
        linear_out=linear_out,
    )


def _zero_adapter_weight(
    *,
    base_prefix: str,
    adapter_key: str,
    input_dim: int,
    output_dim: int,
    like: torch.Tensor,
) -> AdapterWeight:
    return _adapter_weight(
        base_prefix=base_prefix,
        adapter_key=adapter_key,
        alpha=1,
        dim=1,
        linear_in=like.new_zeros((1, input_dim)),
        linear_out=like.new_zeros((output_dim, 1)),
    )


def _fused_pair_adapter_weight(
    base_prefix: str,
    first_lora: LoRA,
    second_lora: LoRA,
    *,
    first_expert_idx: int | None = None,
    second_expert_idx: int | None = None,
) -> AdapterWeight:
    first_linear_in, first_linear_out = _adapter_tensors(first_lora, first_expert_idx)
    second_linear_in, second_linear_out = _adapter_tensors(
        second_lora,
        second_expert_idx,
    )
    assert math.isclose(float(first_lora.scale), float(second_lora.scale))
    total_dim = int(first_linear_in.shape[0] + second_linear_in.shape[0])
    alpha = round(float(first_lora.scale) * total_dim)

    first_rank = int(first_linear_in.shape[0])
    second_rank = int(second_linear_in.shape[0])
    first_out = int(first_linear_out.shape[0])
    second_out = int(second_linear_out.shape[0])

    first_padding = first_linear_out.new_zeros((first_out, second_rank))
    second_padding = second_linear_out.new_zeros((second_out, first_rank))
    return _adapter_weight(
        base_prefix=base_prefix,
        adapter_key=None,
        alpha=alpha,
        dim=total_dim,
        linear_in=torch.cat([first_linear_in, second_linear_in], dim=0),
        linear_out=torch.cat(
            [
                torch.cat([first_linear_out, first_padding], dim=1),
                torch.cat([second_padding, second_linear_out], dim=1),
            ],
            dim=0,
        ),
    )


def _set_adapter_weights(
    out: dict[str, list[Any]],
    base_prefix: str,
    *weights: AdapterWeight,
    weight_suffix: str = ".weight",
) -> None:
    out[f"{base_prefix}{weight_suffix}"] = list(weights)


def _set_expert_adapter_weights(
    out: dict[str, list[Any]],
    base_prefix: str,
    lora: LoRA,
    build_weight: Callable[[int], AdapterWeight],
) -> None:
    for local_expert_idx in range(lora.num_local_experts):
        global_expert_idx = local_expert_idx + lora._expert_offset
        _set_adapter_weights(
            out,
            base_prefix,
            build_weight(local_expert_idx),
            weight_suffix=f".weight{global_expert_idx}",
        )


def _set_lora_weights(
    out: dict[str, list[Any]],
    base_prefix: str,
    *items: tuple[LoRA, str | None],
) -> None:
    _set_adapter_weights(
        out,
        base_prefix,
        *(
            _simple_adapter_weight(base_prefix, lora, adapter_key=adapter_key)
            for lora, adapter_key in items
        ),
    )


def build_transformer_layer_adapter_weights(
    model_chunks: Sequence[Any],
    grouped_moe: bool = False,
    language_layers_only: bool = False,
) -> dict[str, list[Any]]:
    layer_filter = None
    if language_layers_only:
        from art.megatron.lora import (
            _is_language_transformer_layer_name as layer_filter,
        )

    add_mlp_adapter_weights = (
        _add_moe_mlp_adapter_weights_for_layer
        if grouped_moe
        else _add_dense_mlp_adapter_weights_for_layer
    )
    adapter_weights_by_base: dict[str, list[Any]] = {}
    for chunk in model_chunks:
        for module_name, module in chunk.named_modules():
            if not isinstance(module, TransformerLayer):
                continue
            if layer_filter is not None and not layer_filter(module_name):
                continue
            canonical_name = canonical_art_param_name(module_name)
            layer_prefix = (
                canonical_name
                if canonical_name.startswith(
                    ("decoder.layers.", "language_model.decoder.layers.")
                )
                else f"language_model.decoder.layers.{module.layer_number - 1}"
            )
            add_self_attention_adapter_weights(
                adapter_weights_by_base,
                layer_prefix=layer_prefix,
                self_attention=module.self_attention,
            )
            add_mlp_adapter_weights(adapter_weights_by_base, layer_prefix, module)
    return adapter_weights_by_base


def add_self_attention_adapter_weights(
    adapter_weights_by_base: dict[str, list[Any]],
    *,
    layer_prefix: str,
    self_attention: Any,
) -> None:
    for attr in ("linear_proj", "out_proj"):
        linear_proj = getattr(self_attention, attr, None)
        if isinstance(linear_proj, SelfAttentionLinearProjLoRA):
            base_prefix = f"{layer_prefix}.self_attention.{attr}"
            _set_lora_weights(
                adapter_weights_by_base,
                base_prefix,
                (linear_proj.lora, None),
            )

    linear_qkv = getattr(self_attention, "linear_qkv", None)
    if isinstance(linear_qkv, SelfAttentionLinearQKVLoRA):
        base_prefix = f"{layer_prefix}.self_attention.linear_qkv"
        _set_lora_weights(
            adapter_weights_by_base,
            base_prefix,
            (linear_qkv.q_proj_lora, "adapter_q"),
            (linear_qkv.k_proj_lora, "adapter_k"),
            (linear_qkv.v_proj_lora, "adapter_v"),
        )

    in_proj = getattr(self_attention, "in_proj", None)
    if isinstance(in_proj, GatedDeltaNetInProjLoRA):
        base_prefix = f"{layer_prefix}.self_attention.in_proj"
        input_dim = int(in_proj.qkv_lora.A_T.shape[-2])
        output_dim = int(in_proj.num_value_heads_per_partition)
        _set_adapter_weights(
            adapter_weights_by_base,
            base_prefix,
            _simple_adapter_weight(
                base_prefix, in_proj.qkv_lora, adapter_key="adapter_qkv"
            ),
            _simple_adapter_weight(
                base_prefix, in_proj.z_lora, adapter_key="adapter_z"
            ),
            *(
                _zero_adapter_weight(
                    base_prefix=base_prefix,
                    adapter_key=adapter_key,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    like=in_proj.qkv_lora.B_T,
                )
                for adapter_key in ("adapter_b", "adapter_a")
            ),
        )


def _add_dense_mlp_adapter_weights_for_layer(
    adapter_weights_by_base: dict[str, list[Any]],
    layer_prefix: str,
    module: Any,
) -> None:
    from art.megatron.model_support.handlers.default_dense import _require_dense_mlp

    _require_dense_mlp(module)
    add_split_mlp_adapter_weights(
        adapter_weights_by_base,
        f"{layer_prefix}.mlp",
        module.mlp,
    )


def _add_moe_mlp_adapter_weights_for_layer(
    adapter_weights_by_base: dict[str, list[Any]],
    layer_prefix: str,
    module: Any,
) -> None:
    from art.megatron.model_support.handlers.default_dense import _require_moe_experts

    add_grouped_moe_adapter_weights(
        adapter_weights_by_base,
        layer_prefix=layer_prefix,
        experts=_require_moe_experts(module),
    )
    shared_experts = getattr(module.mlp, "shared_experts", None)
    if shared_experts is not None:
        add_split_mlp_adapter_weights(
            adapter_weights_by_base,
            f"{layer_prefix}.mlp.shared_experts",
            shared_experts,
        )


def add_grouped_moe_adapter_weights(
    adapter_weights_by_base: dict[str, list[Any]],
    *,
    layer_prefix: str,
    experts: Any,
) -> None:
    linear_fc1 = getattr(experts, "linear_fc1", None)
    base_prefix = f"{layer_prefix}.mlp.experts.linear_fc1"
    if isinstance(linear_fc1, MLPExpertsLinearFC1LoRA):
        if linear_fc1.fused_gate_up:
            lora = linear_fc1.lora
            build_weight = lambda local_expert_idx: _simple_adapter_weight(
                base_prefix,
                linear_fc1.lora,
                expert_idx=local_expert_idx,
            )
        else:
            lora = linear_fc1.gate_lora
            build_weight = lambda local_expert_idx: _fused_pair_adapter_weight(
                base_prefix,
                linear_fc1.gate_lora,
                linear_fc1.up_lora,
                first_expert_idx=local_expert_idx,
                second_expert_idx=local_expert_idx,
            )
        _set_expert_adapter_weights(
            adapter_weights_by_base,
            base_prefix,
            lora,
            build_weight,
        )

    linear_fc2 = getattr(experts, "linear_fc2", None)
    if isinstance(linear_fc2, MLPExpertsLinearFC2LoRA):
        base_prefix = f"{layer_prefix}.mlp.experts.linear_fc2"
        _set_expert_adapter_weights(
            adapter_weights_by_base,
            base_prefix,
            linear_fc2.lora,
            lambda local_expert_idx: _simple_adapter_weight(
                base_prefix,
                linear_fc2.lora,
                expert_idx=local_expert_idx,
            ),
        )


def add_split_mlp_adapter_weights(
    adapter_weights_by_base: dict[str, list[Any]],
    base_prefix: str,
    mlp: Any,
) -> None:
    linear_fc1 = getattr(mlp, "linear_fc1", None)
    if isinstance(linear_fc1, SharedExpertsLinearFC1LoRA):
        fc1_prefix = f"{base_prefix}.linear_fc1"
        _set_lora_weights(
            adapter_weights_by_base,
            fc1_prefix,
            (linear_fc1.gate_lora, "adapter_gate"),
            (linear_fc1.up_lora, "adapter_up"),
        )

    linear_fc2 = getattr(mlp, "linear_fc2", None)
    if isinstance(linear_fc2, SelfAttentionLinearProjLoRA):
        fc2_prefix = f"{base_prefix}.linear_fc2"
        _set_adapter_weights(
            adapter_weights_by_base,
            fc2_prefix,
            _simple_adapter_weight(fc2_prefix, linear_fc2.lora),
        )
