from functools import lru_cache, partial
from typing import Any, cast

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.core.models.gpt.gpt_model import GPTModel
import torch

from art.megatron.dsv4.spec import get_dsv4_decoder_block_spec


def _resolve_dsv4_hf_param(hf_param: Any, captures: tuple[str, ...]) -> Any:
    def resolve_one(pattern: str) -> str:
        resolved = pattern
        capture_index = 0
        while "**" in resolved and capture_index < len(captures):
            resolved = resolved.replace("**", captures[capture_index], 1)
            capture_index += 1
        while "*" in resolved and capture_index < len(captures):
            resolved = resolved.replace("*", captures[capture_index], 1)
            capture_index += 1
        return resolved

    if isinstance(hf_param, str):
        return resolve_one(hf_param)
    return {key: resolve_one(value) for key, value in hf_param.items()}


class _Dsv4AutoMapping(AutoMapping):
    def __init__(
        self,
        megatron_param: str,
        hf_param: str,
        export_hf_param: str | None = None,
        permute_dims: tuple[int, ...] | None = None,
    ):
        super().__init__(megatron_param, hf_param, permute_dims)
        self.export_hf_param = export_hf_param or hf_param

    def megatron_to_hf(self, megatron_weights: Any, megatron_module: Any):
        converted = super().megatron_to_hf(megatron_weights, megatron_module)
        if not converted or self.export_hf_param == self.hf_param:
            return converted
        return {self.export_hf_param: next(iter(converted.values()))}

    def resolve(self, captures: tuple[str, ...]):
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(
            resolved_megatron_param,
            cast(str, resolved_hf_param),
            cast(str, _resolve_dsv4_hf_param(self.export_hf_param, captures)),
            self.permute_dims,
        )


class _Dsv4ReplicatedMapping(ReplicatedMapping):
    def __init__(
        self,
        megatron_param: str,
        hf_param: str,
        export_hf_param: str | None = None,
    ):
        super().__init__(megatron_param, hf_param)
        self.export_hf_param = export_hf_param or hf_param

    def megatron_to_hf(self, megatron_weights: Any, megatron_module: Any):
        converted = super().megatron_to_hf(megatron_weights, megatron_module)
        if not converted or self.export_hf_param == self.hf_param:
            return converted
        return {self.export_hf_param: next(iter(converted.values()))}

    def resolve(self, captures: tuple[str, ...]):
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(
            resolved_megatron_param,
            cast(str, resolved_hf_param),
            cast(str, _resolve_dsv4_hf_param(self.export_hf_param, captures)),
        )


class _Dsv4GatedMLPMapping(GatedMLPMapping):
    def __init__(
        self,
        megatron_param: str,
        gate: str,
        up: str,
        export_gate: str | None = None,
        export_up: str | None = None,
    ):
        super().__init__(megatron_param, gate, up)
        self.export_hf_param = {
            "gate": export_gate or gate,
            "up": export_up or up,
        }

    def megatron_to_hf(self, megatron_weights: Any, megatron_module: Any):
        converted = super().megatron_to_hf(megatron_weights, megatron_module)
        if not converted or self.export_hf_param == self.hf_param:
            return converted
        remapped: dict[str, torch.Tensor] = {}
        source_hf_param = cast(dict[str, str], self.hf_param)
        for source_key, target_key in zip(
            source_hf_param.values(), self.export_hf_param.values(), strict=True
        ):
            if source_key in converted:
                remapped[target_key] = converted[source_key]
        return remapped

    def resolve(self, captures: tuple[str, ...]):
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        resolved_hf_param = cast(dict[str, str], resolved_hf_param)
        resolved_export = cast(
            dict[str, str], _resolve_dsv4_hf_param(self.export_hf_param, captures)
        )
        return type(self)(
            resolved_megatron_param,
            resolved_hf_param["gate"],
            resolved_hf_param["up"],
            resolved_export["gate"],
            resolved_export["up"],
        )


@lru_cache(maxsize=1)
def _art_dsv4_expert_mapping_types() -> tuple[type[Any], type[Any]]:
    class _ArtDsv4ExpertGateUpMapping(GatedMLPMapping):
        is_grouped_export = True

        def __init__(
            self,
            megatron_param: str,
            gate: str,
            up: str,
            export_hf_param: str,
        ):
            super().__init__(megatron_param, gate, up)
            self.export_hf_param = export_hf_param

        @property
        def group_key(self) -> str:
            return self.export_hf_param

        def hf_to_megatron(
            self,
            hf_weights: Any,
            megatron_module: Any,
        ) -> torch.Tensor:
            from megatron.bridge.models.conversion.param_mapping import (
                _align_expert_weight_to_shape,
            )
            from megatron.bridge.models.conversion.utils import (
                get_module_and_param_from_name,
            )

            normalized_param = self._normalize_expert_param_name(self.megatron_param)
            target_param = get_module_and_param_from_name(
                megatron_module, normalized_param
            )[1]
            full_target_shape = (
                target_param.shape[0] * self.tp_size,
                target_param.shape[1],
            )
            if full_target_shape[0] % 2 != 0:
                raise ValueError(
                    "Expected even fused dim for "
                    f"{self.megatron_param}, got {full_target_shape}."
                )
            gate_target_shape = (full_target_shape[0] // 2, full_target_shape[1])
            gate = _align_expert_weight_to_shape(
                cast(torch.Tensor, hf_weights["gate"]),
                torch.Size(gate_target_shape),
                "gate",
                transpose_hint=False,
            )
            up = _align_expert_weight_to_shape(
                cast(torch.Tensor, hf_weights["up"]),
                torch.Size(gate_target_shape),
                "up",
                transpose_hint=False,
            )
            return super().hf_to_megatron({"gate": gate, "up": up}, megatron_module)

        def megatron_to_hf(self, megatron_weights: Any, megatron_module: Any):
            converted = super().megatron_to_hf(megatron_weights, megatron_module)
            if not converted:
                return {}
            hf_param = cast(dict[str, str], self.hf_param)
            gate_key, up_key = hf_param["gate"], hf_param["up"]
            if gate_key not in converted or up_key not in converted:
                return {}
            return {
                self.export_hf_param: torch.cat(
                    [converted[gate_key], converted[up_key]], dim=0
                )
            }

        def resolve(self, captures: tuple[str, ...]):
            resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
            resolved_hf_param = cast(dict[str, str], resolved_hf_param)
            return type(self)(
                resolved_megatron_param,
                resolved_hf_param["gate"],
                resolved_hf_param["up"],
                cast(str, _resolve_dsv4_hf_param(self.export_hf_param, captures)),
            )

    class _ArtDsv4ExpertDownMapping(AutoMapping):
        is_grouped_export = True

        def __init__(
            self,
            megatron_param: str,
            hf_param: str,
            export_hf_param: str,
        ):
            super().__init__(megatron_param, hf_param)
            self.export_hf_param = export_hf_param

        @property
        def group_key(self) -> str:
            return self.export_hf_param

        def hf_to_megatron(
            self,
            hf_weights: Any,
            megatron_module: Any,
        ) -> torch.Tensor:
            from megatron.bridge.models.conversion.param_mapping import (
                ColumnParallelMapping,
                RowParallelMapping,
                _align_expert_weight_to_shape,
            )
            from megatron.bridge.models.conversion.utils import (
                get_module_and_param_from_name,
            )

            normalized_param = self._normalize_expert_param_name(self.megatron_param)
            target_param = get_module_and_param_from_name(
                megatron_module, normalized_param
            )[1]
            if self._mapping is None:
                self._detected_type = self._detect_parallelism_type(megatron_module)
                self._mapping = self._get_or_create_mapping(self._detected_type)
            if isinstance(self._mapping, ColumnParallelMapping):
                full_target_shape = (
                    target_param.shape[0] * self.tp_size,
                    target_param.shape[1],
                )
            elif isinstance(self._mapping, RowParallelMapping):
                full_target_shape = (
                    target_param.shape[0],
                    target_param.shape[1] * self.tp_size,
                )
            else:
                full_target_shape = tuple(target_param.shape)
            aligned = _align_expert_weight_to_shape(
                cast(torch.Tensor, hf_weights),
                torch.Size(full_target_shape),
                "down_proj",
                transpose_hint=False,
            )
            return self._mapping.hf_to_megatron(aligned, megatron_module)

        def megatron_to_hf(self, megatron_weights: Any, megatron_module: Any):
            converted = super().megatron_to_hf(megatron_weights, megatron_module)
            if not converted:
                return {}
            return {self.export_hf_param: next(iter(converted.values()))}

        def resolve(self, captures: tuple[str, ...]):
            resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
            return type(self)(
                resolved_megatron_param,
                cast(str, resolved_hf_param),
                cast(str, _resolve_dsv4_hf_param(self.export_hf_param, captures)),
            )

    return _ArtDsv4ExpertGateUpMapping, _ArtDsv4ExpertDownMapping


def _register_dsv4_module_types() -> None:
    AutoMapping.register_module_type("DeepSeekV4Attention", "column")
    AutoMapping.register_module_type("DeepSeekV4Compressor", "replicated")
    AutoMapping.register_module_type("Dsv4FinalNorm", "replicated")
    AutoMapping.register_module_type("Dsv4Router", "replicated")
    AutoMapping.register_module_type("Dsv4TransformerLayer", "replicated")
    AutoMapping.register_module_type("HCHeadParams", "replicated")


def _dsv4_mapping_registry() -> MegatronMappingRegistry:
    _register_dsv4_module_types()
    expert_gate_up_mapping, expert_down_mapping = _art_dsv4_expert_mapping_types()
    mappings: list[Any] = [
        _Dsv4AutoMapping(
            "embedding.word_embeddings.weight",
            "embed.weight",
            "model.embed_tokens.weight",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.input_layernorm.weight",
            "layers.*.attn_norm.weight",
            "model.layers.*.input_layernorm.weight",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.pre_mlp_layernorm.weight",
            "layers.*.ffn_norm.weight",
            "model.layers.*.post_attention_layernorm.weight",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.mlp.router.weight",
            "layers.*.ffn.gate.weight",
            "model.layers.*.mlp.gate.weight",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.mlp.router.tid2eid",
            "layers.*.ffn.gate.tid2eid",
            "model.layers.*.mlp.gate.tid2eid",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.mlp.router.e_score_correction_bias",
            "layers.*.ffn.gate.bias",
            "model.layers.*.mlp.gate.e_score_correction_bias",
        ),
        expert_down_mapping(
            "decoder.layers.*.mlp.experts.linear_fc2.weight*",
            "layers.*.ffn.experts.*.w2.weight",
            "model.layers.*.mlp.experts.down_proj",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
            "layers.*.ffn.shared_experts.w2.weight",
            "model.layers.*.mlp.shared_experts.down_proj.weight",
        ),
        _Dsv4AutoMapping("decoder.final_layernorm.weight", "model.norm.weight"),
        _Dsv4AutoMapping(
            "decoder.final_layernorm.hc_head_params.hc_head_fn",
            "hc_head_fn",
            "model.hc_head.hc_fn",
        ),
        _Dsv4AutoMapping(
            "decoder.final_layernorm.hc_head_params.hc_head_base",
            "hc_head_base",
            "model.hc_head.hc_base",
        ),
        _Dsv4AutoMapping(
            "decoder.final_layernorm.hc_head_params.hc_head_scale",
            "hc_head_scale",
            "model.hc_head.hc_scale",
        ),
        _Dsv4AutoMapping("output_layer.weight", "head.weight", "lm_head.weight"),
        _Dsv4AutoMapping(
            "decoder.layers.*.hc_attn_fn",
            "layers.*.hc_attn_fn",
            "model.layers.*.attn_hc.fn",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.hc_attn_base",
            "layers.*.hc_attn_base",
            "model.layers.*.attn_hc.base",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.hc_attn_scale",
            "layers.*.hc_attn_scale",
            "model.layers.*.attn_hc.scale",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.hc_ffn_fn",
            "layers.*.hc_ffn_fn",
            "model.layers.*.ffn_hc.fn",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.hc_ffn_base",
            "layers.*.hc_ffn_base",
            "model.layers.*.ffn_hc.base",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.hc_ffn_scale",
            "layers.*.hc_ffn_scale",
            "model.layers.*.ffn_hc.scale",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.self_attention.wq_a.weight",
            "layers.*.attn.wq_a.weight",
            "model.layers.*.self_attn.q_a_proj.weight",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.self_attention.q_norm.weight",
            "layers.*.attn.q_norm.weight",
            "model.layers.*.self_attn.q_a_norm.weight",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.self_attention.wq_b.weight",
            "layers.*.attn.wq_b.weight",
            "model.layers.*.self_attn.q_b_proj.weight",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.self_attention.wkv.weight",
            "layers.*.attn.wkv.weight",
            "model.layers.*.self_attn.kv_proj.weight",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.self_attention.kv_norm.weight",
            "layers.*.attn.kv_norm.weight",
            "model.layers.*.self_attn.kv_norm.weight",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.self_attention.wo_a.weight",
            "layers.*.attn.wo_a.weight",
            "model.layers.*.self_attn.o_a_proj.weight",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.self_attention.wo_b.weight",
            "layers.*.attn.wo_b.weight",
            "model.layers.*.self_attn.o_b_proj.weight",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.self_attention.attn_sink",
            "layers.*.attn.attn_sink",
            "model.layers.*.self_attn.sinks",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.self_attention.compressor.ape",
            "layers.*.attn.compressor.ape",
            "model.layers.*.self_attn.compressor.position_bias",
        ),
        _Dsv4ReplicatedMapping(
            "decoder.layers.*.self_attention.compressor.wkv.weight",
            "layers.*.attn.compressor.wkv.weight",
            "model.layers.*.self_attn.compressor.kv_proj.weight",
        ),
        _Dsv4ReplicatedMapping(
            "decoder.layers.*.self_attention.compressor.wgate.weight",
            "layers.*.attn.compressor.wgate.weight",
            "model.layers.*.self_attn.compressor.gate_proj.weight",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.self_attention.compressor.norm.weight",
            "layers.*.attn.compressor.norm.weight",
            "model.layers.*.self_attn.compressor.kv_norm.weight",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.self_attention.indexer.linear_wq_b.weight",
            "layers.*.attn.indexer.wq_b.weight",
            "model.layers.*.self_attn.compressor.indexer.q_b_proj.weight",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.self_attention.indexer.linear_weights_proj.weight",
            "layers.*.attn.indexer.weights_proj.weight",
            "model.layers.*.self_attn.compressor.indexer.weights_proj.weight",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.self_attention.indexer.compressor.ape",
            "layers.*.attn.indexer.compressor.ape",
            "model.layers.*.self_attn.compressor.indexer.position_bias",
        ),
        _Dsv4ReplicatedMapping(
            "decoder.layers.*.self_attention.indexer.compressor.wkv.weight",
            "layers.*.attn.indexer.compressor.wkv.weight",
            "model.layers.*.self_attn.compressor.indexer.kv_proj.weight",
        ),
        _Dsv4ReplicatedMapping(
            "decoder.layers.*.self_attention.indexer.compressor.wgate.weight",
            "layers.*.attn.indexer.compressor.wgate.weight",
            "model.layers.*.self_attn.compressor.indexer.gate_proj.weight",
        ),
        _Dsv4AutoMapping(
            "decoder.layers.*.self_attention.indexer.compressor.norm.weight",
            "layers.*.attn.indexer.compressor.norm.weight",
            "model.layers.*.self_attn.compressor.indexer.kv_norm.weight",
        ),
        expert_gate_up_mapping(
            megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
            gate="layers.*.ffn.experts.*.w1.weight",
            up="layers.*.ffn.experts.*.w3.weight",
            export_hf_param="model.layers.*.mlp.experts.gate_up_proj",
        ),
        _Dsv4GatedMLPMapping(
            megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
            gate="layers.*.ffn.shared_experts.w1.weight",
            up="layers.*.ffn.shared_experts.w3.weight",
            export_gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
            export_up="model.layers.*.mlp.shared_experts.up_proj.weight",
        ),
    ]
    return MegatronMappingRegistry(*mappings)


class ArtDeepSeekV4Bridge(DeepSeekV3Bridge):
    def provider_bridge(self, hf_pretrained: Any):
        hf_config = hf_pretrained.config
        if not hasattr(hf_config, "first_k_dense_replace"):
            hf_config.first_k_dense_replace = 0
        provider = cast(Any, super().provider_bridge(hf_pretrained))
        provider.transformer_layer_spec = partial(get_dsv4_decoder_block_spec)
        provider.num_layers = hf_config.num_hidden_layers
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.share_embeddings_and_output_weights = False
        provider.multi_latent_attention = False
        provider.q_lora_rank = hf_config.q_lora_rank
        provider.kv_lora_rank = hf_config.head_dim
        provider.qk_pos_emb_head_dim = hf_config.qk_rope_head_dim
        provider.num_attention_heads = hf_config.num_attention_heads
        provider.num_query_groups = 1
        provider.kv_channels = hf_config.head_dim
        provider.num_moe_experts = hf_config.n_routed_experts
        provider.moe_router_topk = hf_config.num_experts_per_tok
        provider.moe_router_score_function = hf_config.scoring_func
        provider.moe_router_topk_scaling_factor = hf_config.routed_scaling_factor
        provider.moe_router_enable_expert_bias = False
        provider.moe_router_fusion = False
        provider.moe_layer_freq = [1] * hf_config.num_hidden_layers
        provider.moe_ffn_hidden_size = hf_config.moe_intermediate_size
        provider.ffn_hidden_size = hf_config.moe_intermediate_size
        provider.moe_shared_expert_intermediate_size = (
            hf_config.moe_intermediate_size * hf_config.n_shared_experts
        )
        provider.dsv4_hc_mult = getattr(hf_config, "hc_mult", 4)
        provider.dsv4_hc_sinkhorn_iters = getattr(hf_config, "hc_sinkhorn_iters", 20)
        provider.dsv4_hc_eps = getattr(hf_config, "hc_eps", 1e-6)
        provider.dsv4_compress_ratios = getattr(hf_config, "compress_ratios", None)
        provider.dsv4_compress_rope_theta = getattr(
            hf_config, "compress_rope_theta", 160000
        )
        rope_scaling = getattr(hf_config, "rope_scaling", None) or {}
        provider.rotary_scaling_factor = rope_scaling.get("factor", 16)
        provider.original_max_position_embeddings = rope_scaling.get(
            "original_max_position_embeddings", 65536
        )
        provider.beta_fast = rope_scaling.get("beta_fast", 32)
        provider.beta_slow = rope_scaling.get("beta_slow", 1)
        provider.dsv4_swiglu_limit = getattr(hf_config, "swiglu_limit", 0.0)
        provider.dsv4_o_groups = getattr(hf_config, "o_groups", 16)
        provider.dsv4_o_lora_rank = getattr(hf_config, "o_lora_rank", 1024)
        provider.dsv4_n_hash_layers = getattr(hf_config, "n_hash_layers", 3)
        provider.dsv4_window_size = getattr(hf_config, "sliding_window", 128)
        provider.dsa_indexer_n_heads = getattr(hf_config, "index_n_heads", 64)
        provider.dsa_indexer_head_dim = getattr(hf_config, "index_head_dim", 128)
        provider.dsa_indexer_topk = getattr(hf_config, "index_topk", 1024)
        if provider.dsv4_swiglu_limit > 0:
            provider.bias_activation_fusion = False
            provider.activation_func_clamp_value = provider.dsv4_swiglu_limit
        provider.mtp_num_layers = None
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        return _dsv4_mapping_registry()

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: dict[str, torch.Tensor],
        hf_state_dict: Any,
    ) -> dict[str, torch.Tensor]:
        del task, hf_state_dict
        return converted_weights_dict


_DSV4_BRIDGE_REGISTERED = False


def ensure_dsv4_bridge_registered() -> None:
    global _DSV4_BRIDGE_REGISTERED
    if _DSV4_BRIDGE_REGISTERED:
        return
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge

    from art.megatron.dsv4.hf_config import ensure_dsv4_hf_config_registered

    ensure_dsv4_hf_config_registered()
    MegatronModelBridge.register_bridge(
        source="DeepseekV4ForCausalLM",
        target=GPTModel,
        provider=MLAModelProvider,
        model_type="deepseek_v4",
    )(ArtDeepSeekV4Bridge)
    _DSV4_BRIDGE_REGISTERED = True
