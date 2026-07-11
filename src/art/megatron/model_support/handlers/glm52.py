from __future__ import annotations

from typing import Any, Literal, Sequence

import torch

from art.megatron.model_support.handlers.default_dense import DefaultMoeHandler
from art.megatron.model_support.spec import (
    ExpertPackedLoraGroup,
    ExpertPackedLoraSlot,
    LayerFamilyInstance,
    PrefixTreeModelStateContext,
)


def _hf_config(bridge: Any) -> Any:
    pretrained = bridge.hf_pretrained
    return getattr(pretrained, "config", pretrained)


class Glm52Handler(DefaultMoeHandler):
    key = "glm52"
    is_moe = True
    cp_supported = False
    native_vllm_lora_status = "wip"

    def patch_provider(self, provider: Any, bridge: Any) -> None:
        from art.megatron.glm52.spec import get_glm52_decoder_block_spec

        config = _hf_config(bridge)
        required_dims = {
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
            "index_head_dim": 128,
        }
        for name, expected in required_dims.items():
            actual = int(getattr(config, name))
            if actual != expected:
                raise ValueError(f"GLM-5.2 requires {name}={expected}, got {actual}.")
        topk = int(config.index_topk)
        if topk % 32:
            raise ValueError(f"GLM-5.2 index_topk must be divisible by 32, got {topk}.")
        provider.transformer_layer_spec = get_glm52_decoder_block_spec
        provider.experimental_attention_variant = None
        provider.kv_channels = int(config.v_head_dim)
        provider.num_moe_experts = int(config.n_routed_experts)
        provider.num_query_groups = int(config.num_attention_heads)
        provider.rotary_interleaved = False
        provider.rope_type = "rope"
        provider.position_embedding_type = "rope"
        provider.rotary_base = float(config.rope_parameters["rope_theta"])
        provider.rotary_scaling_factor = 1.0
        provider.mscale = 1.0
        provider.mscale_all_dim = 1.0
        provider.mtp_num_layers = None
        provider.dsa_indexer_n_heads = int(config.index_n_heads)
        provider.dsa_indexer_head_dim = int(config.index_head_dim)
        provider.dsa_indexer_topk = topk
        provider.dsa_indexer_loss_coeff = 0.0
        provider.dsa_indexer_use_sparse_loss = False
        provider.glm52_indexer_types = tuple(config.indexer_types)
        provider.moe_layer_freq = [
            0 if layer_type == "dense" else 1 for layer_type in config.mlp_layer_types
        ]
        provider.moe_shared_expert_intermediate_size = int(
            config.moe_intermediate_size
        ) * int(config.n_shared_experts)
        provider.moe_router_bias_update_rate = 0.0
        provider.moe_aux_loss_coeff = 0.0
        provider.attention_softmax_in_fp32 = True

    def configure_provider_for_runtime(self, provider: Any) -> None:
        provider.mtp_num_layers = None
        provider.mtp_loss_scaling_factor = None

    def build_prefix_tree_model_state(
        self, context: PrefixTreeModelStateContext
    ) -> dict[str, Any]:
        if context.input_pos is None:
            raise RuntimeError("GLM-5.2 prefix-tree attention requires input_pos.")
        from art.megatron.glm52.state import build_glm52_prefix_tree_state

        return {
            "glm52": build_glm52_prefix_tree_state(
                position_ids=context.input_pos,
                group_ids=context.group_ids,
                parent_ids=context.parent_ids,
                device=context.device,
            )
        }

    def correctness_precision(self) -> Literal["bf16", "fp32"]:
        return "bf16"

    def correctness_use_fp32_lora_reference(self) -> bool:
        return False

    def correctness_phase_pass_fns(self, oracle_harness: Any) -> dict[str, Any]:
        nonzero = {"typical_abs_scale": 0.0, "candidate_abs_scale": 0.0}
        forward = oracle_harness.MetricThresholdRule(
            limits={"mean_abs_pct": 3.0}, minimums=nonzero
        )
        grad = oracle_harness.MetricThresholdRule(
            limits={"mean_abs_pct": 5.0}, minimums=nonzero
        )
        return {
            "forward": forward,
            "outputs": forward,
            "losses": oracle_harness.MetricThresholdRule(limits={"mean_abs_pct": 3.0}),
            "grads": grad,
            "deltas": grad,
            "router_scores": forward,
            "router_topk_ids": oracle_harness.MetricThresholdRule(
                limits={"topk_mismatch_fraction": 0.0, "top1_mismatch_fraction": 0.0}
            ),
        }

    def collect_layer_families(self, provider: Any) -> list[LayerFamilyInstance]:
        pattern = tuple(provider.glm52_indexer_types)
        shared = next(
            (index for index, value in enumerate(pattern) if value == "shared"),
            None,
        )
        sparse_mlp = next(
            (index for index, value in enumerate(provider.moe_layer_freq) if value),
            None,
        )
        families = [
            LayerFamilyInstance(key="glm52_full_index_attention", layer_index=0),
            LayerFamilyInstance(key="dense_mlp", layer_index=0),
        ]
        if shared is not None:
            families.append(
                LayerFamilyInstance(
                    key="glm52_shared_index_attention", layer_index=shared
                )
            )
        if sparse_mlp is not None:
            families.extend(
                (
                    LayerFamilyInstance(key="grouped_moe_mlp", layer_index=sparse_mlp),
                    LayerFamilyInstance(
                        key="shared_experts_mlp", layer_index=sparse_mlp
                    ),
                )
            )
        return families

    def identity_lora_target_parameters(
        self,
        model: Any,
        *,
        target_modules: list[str],
    ) -> list[str]:
        targets = set(target_modules)
        suffixes = tuple(f"{target}.weight" for target in targets - {"experts"})
        return [
            name
            for name, _ in model.named_parameters()
            if ".indexer." not in name
            and (
                name.endswith(suffixes)
                or ("experts" in targets and ".experts." in name)
            )
        ]

    def apply_lora_adapters(
        self,
        model_chunks: Sequence[Any],
        provider: Any,
        *,
        target_modules: list[str],
        rank: int,
        alpha: int,
    ) -> None:
        from megatron.core.transformer.transformer_layer import TransformerLayer

        from art.megatron.glm52.attention import Glm52SelfAttention
        from art.megatron.glm52.lora import apply_glm52_attention_lora
        from art.megatron.lora import (
            _adapter_model_prefix,
            _is_language_transformer_layer_name,
            wrap_dense_mlp,
            wrap_grouped_moe_experts_3d,
            wrap_shared_experts_mlp,
        )

        targets = set(target_modules)
        for chunk in model_chunks:
            for module_name, layer in chunk.named_modules():
                if not isinstance(layer, TransformerLayer) or not (
                    _is_language_transformer_layer_name(module_name)
                ):
                    continue
                if not isinstance(layer.self_attention, Glm52SelfAttention):
                    raise TypeError(
                        "GLM-5.2 layer has unsupported attention "
                        f"{type(layer.self_attention).__name__}."
                    )
                prefix = _adapter_model_prefix(layer)
                apply_glm52_attention_lora(
                    layer.self_attention,
                    adapter_model_prefix=prefix,
                    provider=provider,
                    target_modules=targets,
                    rank=rank,
                    alpha=alpha,
                )
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
                    continue
                wrap_grouped_moe_experts_3d(
                    experts,
                    adapter_model_prefix=prefix,
                    target_modules=targets,
                    rank=rank,
                    alpha=alpha,
                )
                shared_experts = getattr(layer.mlp, "shared_experts", None)
                if shared_experts is not None:
                    wrap_shared_experts_mlp(
                        shared_experts,
                        adapter_model_prefix=prefix,
                        provider=provider,
                        target_modules=targets,
                        rank=rank,
                        alpha=alpha,
                    )

    def build_adapter_weights_by_base(
        self, model_chunks: Sequence[Any]
    ) -> dict[str, list[Any]]:
        from megatron.core.transformer.transformer_layer import TransformerLayer

        from art.megatron.glm52.attention import Glm52SelfAttention
        from art.megatron.glm52.lora import add_glm52_attention_adapter_weights
        from art.megatron.weights.adapter_export import (
            add_dense_mlp_adapter_weights,
            add_grouped_moe_adapter_weights,
            add_shared_experts_adapter_weights,
            layer_base_prefix,
        )

        result: dict[str, list[Any]] = {}
        for chunk in model_chunks:
            for module_name, layer in chunk.named_modules():
                if not isinstance(layer, TransformerLayer) or not isinstance(
                    layer.self_attention, Glm52SelfAttention
                ):
                    continue
                prefix = layer_base_prefix(layer, module_name=module_name)
                add_glm52_attention_adapter_weights(
                    result,
                    layer_prefix=prefix,
                    attention=layer.self_attention,
                )
                experts = getattr(layer.mlp, "experts", None)
                if experts is None:
                    add_dense_mlp_adapter_weights(
                        result, layer_prefix=prefix, mlp=layer.mlp
                    )
                    continue
                add_grouped_moe_adapter_weights(
                    result, layer_prefix=prefix, experts=experts
                )
                shared_experts = getattr(layer.mlp, "shared_experts", None)
                if shared_experts is not None:
                    add_shared_experts_adapter_weights(
                        result,
                        layer_prefix=prefix,
                        shared_experts=shared_experts,
                    )
        return result

    def expert_packed_lora_groups(self) -> tuple[ExpertPackedLoraGroup, ...]:
        return (
            ExpertPackedLoraGroup(
                art_group_suffix=".mlp.experts",
                slots=(
                    ExpertPackedLoraSlot(
                        source_projection="gate_up_proj",
                        source_lora="lora_A",
                        output_suffix="base_layer.lora_A.weight",
                        pack_layout="expert_rows",
                    ),
                    ExpertPackedLoraSlot(
                        source_projection="gate_up_proj",
                        source_lora="lora_B",
                        output_suffix="base_layer.lora_B.weight",
                        pack_layout="rank_major_expert_cols",
                    ),
                    ExpertPackedLoraSlot(
                        source_projection="down_proj",
                        source_lora="lora_A",
                        output_suffix="lora_A.weight",
                        pack_layout="expert_rows",
                    ),
                    ExpertPackedLoraSlot(
                        source_projection="down_proj",
                        source_lora="lora_B",
                        output_suffix="lora_B.weight",
                        pack_layout="rank_major_expert_cols",
                    ),
                ),
            ),
        )


GLM52_HANDLER = Glm52Handler()
