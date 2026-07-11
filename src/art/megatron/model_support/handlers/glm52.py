from __future__ import annotations

from typing import Any, Literal, Sequence, cast

import torch

from art.megatron.model_support.handlers.default_dense import (
    DefaultMoeHandler,
    _compile_workaround_flags_for_provider,
)
from art.megatron.model_support.spec import (
    CompileWorkaroundConfig,
    ExpertPackedLoraGroup,
    ExpertPackedLoraSlot,
    LayerFamilyInstance,
    PrefixTreeModelStateContext,
)


def _hf_config(bridge: Any) -> Any:
    pretrained = bridge.hf_pretrained
    return getattr(pretrained, "config", pretrained)


def _from_vllm_expert_lora(
    tensors: dict[str, torch.Tensor], adapter_config: dict[str, Any]
) -> dict[str, torch.Tensor]:
    slots = (
        ("base_layer.lora_A.weight", "gate_up_proj", "lora_A", "rows"),
        ("base_layer.lora_B.weight", "gate_up_proj", "lora_B", "cols"),
        ("lora_A.weight", "down_proj", "lora_A", "rows"),
        ("lora_B.weight", "down_proj", "lora_B", "cols"),
    )
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    used: set[str] = set()
    for key, tensor in tensors.items():
        for suffix, _projection, _lora, _layout in slots:
            marker = f".{suffix}"
            if key.endswith(marker) and key[: -len(marker)].endswith(".mlp.experts"):
                grouped.setdefault(key[: -len(marker)], {})[suffix] = tensor
                used.add(key)
                break
    if not grouped:
        return tensors
    try:
        rank = int(adapter_config["r"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "GLM-5.2 fused expert LoRA requires adapter rank r."
        ) from exc
    if rank <= 0:
        raise RuntimeError(f"GLM-5.2 LoRA rank must be positive, got {rank}.")

    result = {key: tensor for key, tensor in tensors.items() if key not in used}
    for prefix, block in grouped.items():
        missing = [suffix for suffix, *_ in slots if suffix not in block]
        if missing:
            raise RuntimeError(
                f"Incomplete GLM-5.2 expert LoRA block {prefix}: {missing}"
            )
        gate_a = block[slots[0][0]]
        if gate_a.ndim != 2 or gate_a.shape[0] % rank:
            raise RuntimeError(
                f"{prefix}: invalid fused expert A shape {tuple(gate_a.shape)} for rank {rank}."
            )
        experts = gate_a.shape[0] // rank
        for suffix, projection, lora, layout in slots:
            tensor = block[suffix]
            packed = experts * rank
            if tensor.ndim != 2 or tensor.shape[0 if layout == "rows" else 1] != packed:
                raise RuntimeError(
                    f"{prefix}.{suffix}: shape {tuple(tensor.shape)} does not encode "
                    f"{experts} experts at rank {rank}."
                )
            unpacked = (
                tensor.reshape(experts, rank, tensor.shape[1])
                if layout == "rows"
                else tensor.reshape(tensor.shape[0], rank, experts).permute(2, 0, 1)
            )
            for expert, expert_tensor in enumerate(unpacked):
                key = f"{prefix}.{expert}.{projection}.{lora}.weight"
                if key in result:
                    raise RuntimeError(f"Duplicate GLM-5.2 expert LoRA tensor {key}.")
                result[key] = expert_tensor.clone().contiguous()
    return result


class Glm52Handler(DefaultMoeHandler):
    key = "glm52"
    is_moe = True
    cp_supported = True
    native_vllm_lora_status = "validated"

    def compile_workaround_config(self, provider: Any) -> CompileWorkaroundConfig:
        ep1_alltoall = (
            int(getattr(provider, "expert_model_parallel_size", 1) or 1) == 1
            and getattr(provider, "moe_token_dispatcher_type", None) == "alltoall"
        )
        flags = ("mlp_forward", "moe_forward")
        if ep1_alltoall:
            flags = (*flags, "moe_preprocess")
        return CompileWorkaroundConfig(
            flags=_compile_workaround_flags_for_provider(provider, flags),
            shared_expert_state=self._shared_expert_compile_state(provider),
        )

    def patch_provider(self, provider: Any, bridge: Any) -> None:
        from art.megatron.glm52.spec import (
            build_glm52_pipeline_layout,
            get_glm52_decoder_block_spec,
        )

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
        pp_size = int(provider.pipeline_model_parallel_size or 1)
        vp_size = int(provider.virtual_pipeline_model_parallel_size or 1)
        if pp_size * vp_size > 1 and provider.pipeline_model_parallel_layout is None:
            provider.pipeline_model_parallel_layout = build_glm52_pipeline_layout(
                provider.glm52_indexer_types,
                pp_size,
                vp_size,
            )
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
        provider.moe_shared_expert_overlap = False

    def install_preprocess_patch(self, model_chunks: Sequence[Any]) -> None:
        from megatron.core.models.gpt.gpt_model import GPTModel

        for chunk in model_chunks:
            module = chunk
            while hasattr(module, "module"):
                module = module.module
            gpt = module if isinstance(module, GPTModel) else module.language_model
            preprocess = gpt._preprocess

            def preprocess_hook(*args: Any, _preprocess=preprocess, **kwargs: Any):
                output = list(_preprocess(*args, **kwargs))
                decoder_input = cast(torch.Tensor, output[0])
                if decoder_input.is_leaf and not decoder_input.requires_grad:
                    decoder_input.requires_grad_(True)
                return tuple(output)

            gpt._preprocess = preprocess_hook

    def build_prefix_tree_model_state(
        self, context: PrefixTreeModelStateContext
    ) -> dict[str, Any]:
        if context.input_pos is None:
            raise RuntimeError("GLM-5.2 prefix-tree attention requires input_pos.")
        from art.megatron.glm52.state import build_glm52_prefix_tree_state

        if context.context_parallel_state is not None:
            from art.megatron.glm52.state import build_glm52_context_parallel_state

            return {
                "glm52": build_glm52_context_parallel_state(
                    position_ids=context.input_pos,
                    context_parallel_state=context.context_parallel_state,
                    device=context.device,
                )
            }

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
        from art.megatron.glm52.lora import (
            apply_glm52_attention_lora,
            wrap_glm52_grouped_moe_experts_3d,
        )
        from art.megatron.lora import (
            _adapter_model_prefix,
            _is_language_transformer_layer_name,
            wrap_dense_mlp,
            wrap_shared_experts_mlp,
        )

        targets = set(target_modules)
        if "kv_b_proj" in targets:
            raise ValueError(
                "GLM-5.2 does not support kv_b_proj LoRA because native vLLM "
                "sparse MLA executes statically absorbed W_K/W_V weights."
            )
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
                wrap_glm52_grouped_moe_experts_3d(
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

    def from_vllm_lora_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        adapter_config: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        return _from_vllm_expert_lora(tensors, adapter_config)


GLM52_HANDLER = Glm52Handler()
