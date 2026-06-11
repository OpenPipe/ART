from __future__ import annotations

import sys
from typing import Any

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

_COMPRESS_RATIO_TO_LAYER_TYPE = {
    0: "sliding_attention",
    4: "compressed_sparse_attention",
    128: "heavily_compressed_attention",
}
_LAYER_TYPE_TO_COMPRESS_RATIO = {
    value: key for key, value in _COMPRESS_RATIO_TO_LAYER_TYPE.items()
}


class DeepseekV4Config(PretrainedConfig):
    """Local HF config shim for checkpoints newer than the pinned Transformers."""

    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_local_experts": "n_routed_experts",
        "intermediate_size": "moe_intermediate_size",
    }

    def __init__(
        self,
        *,
        vocab_size: int = 129280,
        hidden_size: int = 4096,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 43,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        q_lora_rank: int = 1024,
        num_experts_per_tok: int = 6,
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        scoring_func: str = "sqrtsoftplus",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.5,
        max_position_embeddings: int = 1048576,
        rope_theta: float | int = 10000.0,
        layer_types: list[str] | None = None,
        compress_rates: dict[str, int] | None = None,
        compress_ratios: list[int] | None = None,
        compress_rate_csa: int | None = None,
        compress_rate_hca: int | None = None,
        compress_rope_theta: float | int = 160000.0,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1.0e-6,
        mlp_layer_types: list[str] | None = None,
        num_hash_layers: int | None = None,
        swiglu_limit: float = 10.0,
        sliding_window: int = 128,
        o_groups: int = 8,
        o_lora_rank: int = 1024,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 512,
        num_nextn_predict_layers: int = 1,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        router_jitter_noise: float = 0.0,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1.0e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 0,
        eos_token_id: int | list[int] | None = 1,
        tie_word_embeddings: bool = False,
        rope_parameters: dict[str, Any] | None = None,
        rope_scaling: dict[str, Any] | None = None,
        partial_rotary_factor: float | None = None,
        qk_rope_head_dim: int | None = None,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        attention_dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.compress_rates = self._compress_rates(
            compress_rates,
            compress_rate_csa=compress_rate_csa,
            compress_rate_hca=compress_rate_hca,
        )
        self.compress_rope_theta = compress_rope_theta
        self.layer_types = self._layer_types(
            num_hidden_layers=num_hidden_layers,
            layer_types=layer_types,
            compress_ratios=compress_ratios,
        )
        self.compress_ratios = self._compress_ratios(
            num_hidden_layers=num_hidden_layers,
            compress_ratios=compress_ratios,
            layer_types=self.layer_types,
        )
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.num_hash_layers = 3 if num_hash_layers is None else num_hash_layers
        self.mlp_layer_types = self._mlp_layer_types(
            num_hidden_layers=num_hidden_layers,
            mlp_layer_types=mlp_layer_types,
            num_hash_layers=self.num_hash_layers,
        )
        self.swiglu_limit = swiglu_limit
        self.sliding_window = sliding_window
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.partial_rotary_factor = self._partial_rotary_factor(
            partial_rotary_factor=partial_rotary_factor,
            qk_rope_head_dim=qk_rope_head_dim,
            head_dim=head_dim,
        )
        self.qk_rope_head_dim = int(head_dim * self.partial_rotary_factor)
        self.rope_scaling = rope_scaling
        self.rope_parameters = self._rope_parameters(
            rope_parameters=rope_parameters,
            rope_scaling=rope_scaling,
            rope_theta=rope_theta,
            compress_rope_theta=compress_rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
        )
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.attention_dropout = attention_dropout

    @staticmethod
    def _compress_rates(
        compress_rates: dict[str, int] | None,
        *,
        compress_rate_csa: int | None,
        compress_rate_hca: int | None,
    ) -> dict[str, int]:
        rates = {
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 128,
        }
        if compress_rates is not None:
            rates.update(compress_rates)
        if compress_rate_csa is not None:
            rates["compressed_sparse_attention"] = compress_rate_csa
        if compress_rate_hca is not None:
            rates["heavily_compressed_attention"] = compress_rate_hca
        return rates

    @staticmethod
    def _layer_types(
        *,
        num_hidden_layers: int,
        layer_types: list[str] | None,
        compress_ratios: list[int] | None,
    ) -> list[str]:
        if layer_types is not None:
            return list(layer_types[:num_hidden_layers])
        if compress_ratios is not None:
            return [
                _COMPRESS_RATIO_TO_LAYER_TYPE[int(ratio)]
                for ratio in compress_ratios[:num_hidden_layers]
            ]
        interleave = [
            "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
            for i in range(max(num_hidden_layers - 2, 0))
        ]
        return ["heavily_compressed_attention"] * min(num_hidden_layers, 2) + interleave

    @staticmethod
    def _compress_ratios(
        *,
        num_hidden_layers: int,
        compress_ratios: list[int] | None,
        layer_types: list[str],
    ) -> list[int]:
        if compress_ratios is not None:
            return [int(ratio) for ratio in compress_ratios[:num_hidden_layers]]
        return [_LAYER_TYPE_TO_COMPRESS_RATIO[layer_type] for layer_type in layer_types]

    @staticmethod
    def _mlp_layer_types(
        *,
        num_hidden_layers: int,
        mlp_layer_types: list[str] | None,
        num_hash_layers: int,
    ) -> list[str]:
        if mlp_layer_types is not None:
            return list(mlp_layer_types[:num_hidden_layers])
        return ["hash_moe"] * min(num_hidden_layers, num_hash_layers) + ["moe"] * max(
            0, num_hidden_layers - num_hash_layers
        )

    @staticmethod
    def _partial_rotary_factor(
        *,
        partial_rotary_factor: float | None,
        qk_rope_head_dim: int | None,
        head_dim: int,
    ) -> float:
        if partial_rotary_factor is not None:
            return partial_rotary_factor
        if qk_rope_head_dim is not None:
            return qk_rope_head_dim / head_dim
        return 64 / 512

    @staticmethod
    def _rope_parameters(
        *,
        rope_parameters: dict[str, Any] | None,
        rope_scaling: dict[str, Any] | None,
        rope_theta: float | int,
        compress_rope_theta: float | int,
        partial_rotary_factor: float,
    ) -> dict[str, dict[str, Any]]:
        if isinstance(rope_parameters, dict) and isinstance(
            rope_parameters.get("main"), dict
        ):
            return {
                "main": dict(rope_parameters["main"]),
                "compress": dict(rope_parameters["compress"]),
            }

        def build_params(theta: float | int) -> dict[str, Any]:
            params = dict(rope_scaling or {})
            if "type" in params and "rope_type" not in params:
                params["rope_type"] = params.pop("type")
            params.setdefault("rope_type", "default")
            params["rope_theta"] = theta
            params["partial_rotary_factor"] = partial_rotary_factor
            if params["rope_type"] == "yarn":
                params.setdefault("attention_factor", 1.0)
            return params

        return {
            "main": build_params(rope_theta),
            "compress": build_params(compress_rope_theta),
        }


class DeepseekV4ForCausalLM:
    """Bridge-dispatch marker; this is not a runnable HF model implementation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError(
            "Pinned Transformers does not provide a native DeepSeek-V4 HF model. "
            "This marker only lets Megatron Bridge resolve DSV4 architecture names."
        )


def _ensure_transformers_marker() -> None:
    _add_marker_to_transformers_module(transformers, DeepseekV4ForCausalLM)
    auto_bridge = sys.modules.get("megatron.bridge.models.conversion.auto_bridge")
    if auto_bridge is not None:
        _add_marker_to_transformers_module(
            getattr(auto_bridge, "transformers", None), DeepseekV4ForCausalLM
        )


def _add_marker_to_transformers_module(module: Any, model_class: type) -> None:
    if module is None:
        return
    objects = getattr(module, "_objects", None)
    if isinstance(objects, dict):
        objects["DeepseekV4ForCausalLM"] = model_class
    setattr(module, "DeepseekV4ForCausalLM", model_class)


_REGISTERED = False
_MODEL_REGISTERED = False
_TORCHVISION_LIB: Any | None = None


def ensure_dsv4_hf_config_registered() -> None:
    global _REGISTERED
    _ensure_transformers_marker()
    if _REGISTERED:
        return
    AutoConfig.register(DeepseekV4Config.model_type, DeepseekV4Config)
    _ensure_transformers_marker()
    _REGISTERED = True


def ensure_dsv4_hf_model_registered() -> None:
    global _MODEL_REGISTERED
    if _MODEL_REGISTERED:
        return
    ensure_dsv4_hf_config_registered()
    _ensure_torchvision_nms_schema()
    from art.megatron.dsv4.hf_modeling import (
        DeepseekV4ForCausalLM as HfDeepseekV4ForCausalLM,
    )

    AutoModelForCausalLM.register(
        DeepseekV4Config, HfDeepseekV4ForCausalLM, exist_ok=True
    )
    _add_marker_to_transformers_module(transformers, HfDeepseekV4ForCausalLM)
    auto_bridge = sys.modules.get("megatron.bridge.models.conversion.auto_bridge")
    if auto_bridge is not None:
        _add_marker_to_transformers_module(
            getattr(auto_bridge, "transformers", None), HfDeepseekV4ForCausalLM
        )
    _MODEL_REGISTERED = True


def _ensure_torchvision_nms_schema() -> None:
    global _TORCHVISION_LIB
    if _TORCHVISION_LIB is not None:
        return
    import torch

    try:
        _TORCHVISION_LIB = torch.library.Library("torchvision", "DEF")
        _TORCHVISION_LIB.define(
            "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
        )
    except RuntimeError as exc:
        if "Only a single TORCH_LIBRARY" not in str(exc) and "already" not in str(exc):
            raise
        _TORCHVISION_LIB = torch.library.Library("torchvision", "FRAGMENT")
        try:
            _TORCHVISION_LIB.define(
                "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
            )
        except RuntimeError as define_exc:
            if "already" not in str(define_exc):
                raise


__all__ = [
    "DeepseekV4Config",
    "DeepseekV4ForCausalLM",
    "ensure_dsv4_hf_config_registered",
    "ensure_dsv4_hf_model_registered",
]
