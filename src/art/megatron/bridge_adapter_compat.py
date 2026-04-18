import math

from megatron.bridge.models.conversion.model_bridge import MegatronWeightTuple
from megatron.bridge.models.conversion.peft_bridge import AdapterWeight
from megatron.core.transformer.module import MegatronModule
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
    SharedExpertsLinearFC2LoRA,
)


def _is_language_transformer_layer_name(module_name: str) -> bool:
    while module_name.startswith("module."):
        module_name = module_name.removeprefix("module.")
    return module_name.startswith(("decoder.layers.", "language_model.decoder.layers."))


def _adapter_alpha_dim(lora: LoRA) -> tuple[int, int]:
    dim = int(lora.A_T.shape[-1])
    alpha = float(lora.scale) * dim
    rounded_alpha = round(alpha)
    assert math.isclose(alpha, rounded_alpha)
    return rounded_alpha, dim


def _adapter_tensors(
    lora: LoRA, expert_idx: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    a_t = lora.A_T if expert_idx is None else lora.A_T[expert_idx]
    b_t = lora.B_T if expert_idx is None else lora.B_T[expert_idx]
    return a_t.transpose(-1, -2).contiguous(), b_t.transpose(-1, -2).contiguous()


def _adapter_param_prefix(base_prefix: str, adapter_key: str | None) -> str:
    if adapter_key is None:
        return f"{base_prefix}.adapter"
    return f"{base_prefix}.adapter.{adapter_key}"


def _adapter_weight(
    *,
    base_prefix: str,
    adapter_key: str | None,
    alpha: int,
    dim: int,
    linear_in: torch.Tensor,
    linear_out: torch.Tensor,
) -> AdapterWeight:
    param_prefix = _adapter_param_prefix(base_prefix, adapter_key)
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


def _fused_gdn_adapter_weight(
    base_prefix: str,
    handler: GatedDeltaNetInProjLoRA,
) -> AdapterWeight:
    qkv_linear_in, qkv_linear_out = _adapter_tensors(handler.qkv_lora)
    z_linear_in, z_linear_out = _adapter_tensors(handler.z_lora)
    assert math.isclose(float(handler.qkv_lora.scale), float(handler.z_lora.scale))
    total_dim = int(qkv_linear_in.shape[0] + z_linear_in.shape[0])
    alpha = round(float(handler.qkv_lora.scale) * total_dim)

    qkv_rank = int(qkv_linear_in.shape[0])
    z_rank = int(z_linear_in.shape[0])
    qkv_out = int(qkv_linear_out.shape[0])
    z_out = int(z_linear_out.shape[0])
    beta_alpha_out = int(handler.num_value_heads_per_partition)

    qkv_padding = qkv_linear_out.new_zeros((qkv_out, z_rank))
    z_padding = z_linear_out.new_zeros((z_out, qkv_rank))
    zeros = qkv_linear_out.new_zeros((beta_alpha_out, total_dim))

    return _adapter_weight(
        base_prefix=base_prefix,
        adapter_key=None,
        alpha=alpha,
        dim=total_dim,
        linear_in=torch.cat([qkv_linear_in, z_linear_in], dim=0),
        linear_out=torch.cat(
            [
                torch.cat([qkv_linear_out, qkv_padding], dim=1),
                torch.cat([z_padding, z_linear_out], dim=1),
                zeros,
                zeros.clone(),
            ],
            dim=0,
        ),
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
        second_lora, second_expert_idx
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


def build_adapter_weights_by_base(
    model_chunks: list[MegatronModule],
) -> dict[str, list[AdapterWeight]]:
    adapter_weights_by_base: dict[str, list[AdapterWeight]] = {}
    for chunk in model_chunks:
        for module_name, module in chunk.named_modules():
            if not isinstance(module, TransformerLayer):
                continue
            if not _is_language_transformer_layer_name(module_name):
                continue

            layer_prefix = f"language_model.decoder.layers.{module.layer_number - 1}"
            self_attention = module.self_attention

            linear_proj = getattr(self_attention, "linear_proj", None)
            if isinstance(linear_proj, SelfAttentionLinearProjLoRA):
                base_prefix = f"{layer_prefix}.self_attention.linear_proj"
                adapter_weights_by_base[f"{base_prefix}.weight"] = [
                    _simple_adapter_weight(base_prefix, linear_proj.lora)
                ]

            linear_qkv = getattr(self_attention, "linear_qkv", None)
            if isinstance(linear_qkv, SelfAttentionLinearQKVLoRA):
                base_prefix = f"{layer_prefix}.self_attention.linear_qkv"
                adapter_weights_by_base[f"{base_prefix}.weight"] = [
                    _simple_adapter_weight(
                        base_prefix, linear_qkv.q_proj_lora, adapter_key="adapter_q"
                    ),
                    _simple_adapter_weight(
                        base_prefix, linear_qkv.k_proj_lora, adapter_key="adapter_k"
                    ),
                    _simple_adapter_weight(
                        base_prefix, linear_qkv.v_proj_lora, adapter_key="adapter_v"
                    ),
                ]

            out_proj = getattr(self_attention, "out_proj", None)
            if isinstance(out_proj, SelfAttentionLinearProjLoRA):
                base_prefix = f"{layer_prefix}.self_attention.out_proj"
                adapter_weights_by_base[f"{base_prefix}.weight"] = [
                    _simple_adapter_weight(base_prefix, out_proj.lora)
                ]

            in_proj = getattr(self_attention, "in_proj", None)
            if isinstance(in_proj, GatedDeltaNetInProjLoRA):
                base_prefix = f"{layer_prefix}.self_attention.in_proj"
                adapter_weights_by_base[f"{base_prefix}.weight"] = [
                    _fused_gdn_adapter_weight(base_prefix, in_proj)
                ]

            experts = getattr(module.mlp, "experts", None)
            if experts is not None:
                if isinstance(experts.linear_fc1, MLPExpertsLinearFC1LoRA):
                    base_prefix = f"{layer_prefix}.mlp.experts.linear_fc1"
                    for local_expert_idx in range(
                        experts.linear_fc1.gate_lora.num_local_experts
                    ):
                        global_expert_idx = (
                            local_expert_idx
                            + experts.linear_fc1.gate_lora._expert_offset
                        )
                        adapter_weights_by_base[
                            f"{base_prefix}.weight{global_expert_idx}"
                        ] = [
                            _fused_pair_adapter_weight(
                                base_prefix,
                                experts.linear_fc1.gate_lora,
                                experts.linear_fc1.up_lora,
                                first_expert_idx=local_expert_idx,
                                second_expert_idx=local_expert_idx,
                            )
                        ]
                if isinstance(experts.linear_fc2, MLPExpertsLinearFC2LoRA):
                    base_prefix = f"{layer_prefix}.mlp.experts.linear_fc2"
                    for local_expert_idx in range(
                        experts.linear_fc2.lora.num_local_experts
                    ):
                        global_expert_idx = (
                            local_expert_idx + experts.linear_fc2.lora._expert_offset
                        )
                        adapter_weights_by_base[
                            f"{base_prefix}.weight{global_expert_idx}"
                        ] = [
                            _simple_adapter_weight(
                                base_prefix,
                                experts.linear_fc2.lora,
                                expert_idx=local_expert_idx,
                            )
                        ]
            else:
                linear_fc1 = getattr(module.mlp, "linear_fc1", None)
                if isinstance(linear_fc1, SharedExpertsLinearFC1LoRA):
                    base_prefix = f"{layer_prefix}.mlp.linear_fc1"
                    adapter_weights_by_base[f"{base_prefix}.weight"] = [
                        _simple_adapter_weight(
                            base_prefix,
                            linear_fc1.gate_lora,
                            adapter_key="adapter_gate",
                        ),
                        _simple_adapter_weight(
                            base_prefix, linear_fc1.up_lora, adapter_key="adapter_up"
                        ),
                    ]
                linear_fc2 = getattr(module.mlp, "linear_fc2", None)
                if isinstance(linear_fc2, SharedExpertsLinearFC2LoRA):
                    base_prefix = f"{layer_prefix}.mlp.linear_fc2"
                    adapter_weights_by_base[f"{base_prefix}.weight"] = [
                        _simple_adapter_weight(
                            base_prefix, linear_fc2.row_parallel_lora.lora
                        )
                    ]

            shared_experts = getattr(module.mlp, "shared_experts", None)
            if shared_experts is not None:
                if isinstance(shared_experts.linear_fc1, SharedExpertsLinearFC1LoRA):
                    base_prefix = f"{layer_prefix}.mlp.shared_experts.linear_fc1"
                    adapter_weights_by_base[f"{base_prefix}.weight"] = [
                        _simple_adapter_weight(
                            base_prefix,
                            shared_experts.linear_fc1.gate_lora,
                            adapter_key="adapter_gate",
                        ),
                        _simple_adapter_weight(
                            base_prefix,
                            shared_experts.linear_fc1.up_lora,
                            adapter_key="adapter_up",
                        ),
                    ]
                if isinstance(shared_experts.linear_fc2, SharedExpertsLinearFC2LoRA):
                    base_prefix = f"{layer_prefix}.mlp.shared_experts.linear_fc2"
                    adapter_weights_by_base[f"{base_prefix}.weight"] = [
                        _simple_adapter_weight(
                            base_prefix,
                            shared_experts.linear_fc2.row_parallel_lora.lora,
                        )
                    ]
    return adapter_weights_by_base
