from __future__ import annotations

from typing import Any, Sequence, cast

import torch

from art.megatron.model_support.handlers.default_dense import (
    DefaultMoeHandler,
    _compile_workaround_flags_for_provider,
    _require_moe_experts,
)
from art.megatron.model_support.spec import CompileWorkaroundConfig, LayerFamilyInstance

_ORACLE_HIDDEN_SIZE = 512
_ORACLE_Q_LORA_RANK = 128
_ORACLE_NUM_ATTENTION_HEADS = 1
_ORACLE_NUM_EXPERTS = 2
_ORACLE_NUM_EXPERTS_PER_TOK = 1
_ORACLE_FFN_HIDDEN_SIZE = 128
_ORACLE_INDEX_HEADS = 1
_ORACLE_INDEX_TOPK = 8
_DSV4_MOE_COMPILE_WORKAROUND_FLAGS = (
    "alltoall_dtoh",
    "alltoall_dispatch_preprocess",
    "deepep_dispatch_combine",
    "deepep_permute_restore",
    "te_triton_permute_with_mask_map",
)


class Dsv4Handler(DefaultMoeHandler):
    key = "dsv4"
    is_moe = True
    native_vllm_lora_status = "disabled"

    def patch_provider(self, provider: Any, bridge: Any) -> None:
        del bridge
        from art.megatron.dsv4.spec import get_dsv4_decoder_block_spec

        provider.transformer_layer_spec = get_dsv4_decoder_block_spec
        if int(getattr(provider, "context_parallel_size", 1) or 1) != 1:
            raise RuntimeError(
                "DSV4 model support in this worktree does not implement context parallelism."
            )

    def configure_provider_for_runtime(self, provider: Any) -> None:
        provider.mtp_num_layers = None

    def install_preprocess_patch(self, model_chunks: Sequence[Any]) -> None:
        from megatron.core.models.gpt.gpt_model import GPTModel

        from art.megatron.dsv4.layer import Dsv4MoELayer
        from art.megatron.dsv4.rope import materialize_rope_cache

        for chunk in list(model_chunks):
            module: Any = chunk
            while hasattr(module, "module"):
                module = module.module
            gpt_module = (
                module
                if isinstance(module, GPTModel)
                else cast(GPTModel, getattr(module, "language_model"))
            )
            for child in gpt_module.modules():
                materialize_rope_cache(child)
            preprocess = gpt_module._preprocess

            def preprocess_hook(
                *args: Any, _preprocess=preprocess, _gpt=gpt_module, **kwargs: Any
            ):
                input_ids = kwargs.get("input_ids")
                for child in _gpt.decoder.modules():
                    if isinstance(child, Dsv4MoELayer):
                        child.set_input_ids(
                            input_ids if isinstance(input_ids, torch.Tensor) else None
                        )
                return _preprocess(*args, **kwargs)

            gpt_module._preprocess = preprocess_hook  # type: ignore[attr-defined]

    def collect_layer_families(self, provider: Any) -> list[LayerFamilyInstance]:
        ratios = list(getattr(provider, "dsv4_compress_ratios", ()) or ())

        def first_layer_index(ratio: int) -> int | None:
            try:
                return ratios.index(ratio)
            except ValueError:
                return None

        return [
            LayerFamilyInstance(
                key="dsv4_sliding_attention", layer_index=first_layer_index(0)
            ),
            LayerFamilyInstance(
                key="dsv4_csa_attention", layer_index=first_layer_index(4)
            ),
            LayerFamilyInstance(
                key="dsv4_hca_attention", layer_index=first_layer_index(128)
            ),
            LayerFamilyInstance(key="grouped_moe_mlp", layer_index=0),
            LayerFamilyInstance(key="shared_experts_mlp", layer_index=0),
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
        from art.megatron.dsv4.layer import Dsv4TransformerLayer
        from art.megatron.lora import (
            _adapter_model_prefix,
            wrap_grouped_moe_experts,
            wrap_shared_experts_mlp,
        )

        target_set = set(target_modules)
        for chunk in model_chunks:
            for module in chunk.modules():
                if not isinstance(module, Dsv4TransformerLayer):
                    continue
                adapter_model_prefix = _adapter_model_prefix(module)
                wrap_grouped_moe_experts(
                    _require_moe_experts(module),
                    adapter_model_prefix=adapter_model_prefix,
                    target_modules=target_set,
                    rank=rank,
                    alpha=alpha,
                )
                if getattr(module.mlp, "shared_experts", None) is not None:
                    wrap_shared_experts_mlp(
                        module.mlp.shared_experts,
                        adapter_model_prefix=adapter_model_prefix,
                        provider=provider,
                        target_modules=target_set,
                        rank=rank,
                        alpha=alpha,
                    )

    def build_adapter_weights_by_base(
        self, model_chunks: Sequence[Any]
    ) -> dict[str, list[Any]]:
        from art.megatron.dsv4.layer import Dsv4TransformerLayer
        from art.megatron.weights.adapter_export import (
            add_grouped_moe_adapter_weights,
            add_shared_experts_adapter_weights,
            layer_base_prefix,
        )

        adapter_weights_by_base: dict[str, list[Any]] = {}
        for chunk in model_chunks:
            for module_name, module in chunk.named_modules():
                if not isinstance(module, Dsv4TransformerLayer):
                    continue
                layer_prefix = layer_base_prefix(module, module_name=module_name)
                add_grouped_moe_adapter_weights(
                    adapter_weights_by_base,
                    layer_prefix=layer_prefix,
                    experts=_require_moe_experts(module),
                )
                if getattr(module.mlp, "shared_experts", None) is not None:
                    add_shared_experts_adapter_weights(
                        adapter_weights_by_base,
                        layer_prefix=layer_prefix,
                        shared_experts=module.mlp.shared_experts,
                    )
        return adapter_weights_by_base

    def compile_workaround_config(self, provider: Any) -> CompileWorkaroundConfig:
        return CompileWorkaroundConfig(
            flags=_compile_workaround_flags_for_provider(
                provider,
                _DSV4_MOE_COMPILE_WORKAROUND_FLAGS,
            ),
            shared_expert_state=self._shared_expert_compile_state(provider),
        )

    def ensure_hf_reference_registered(self) -> None:
        from art.megatron.dsv4.hf_config import ensure_dsv4_hf_model_registered

        ensure_dsv4_hf_model_registered()

    def prepare_hf_reference_config(self, config: Any) -> None:
        """Puts the HF parity oracle in eager training mode with reduced fit-only axes."""
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")
        config._experts_implementation = "eager"
        self._apply_oracle_shape_overrides(config)

    def hf_reference_from_pretrained_kwargs(
        self, *, config: Any, dtype: torch.dtype
    ) -> dict[str, Any]:
        del config, dtype
        return {"experts_implementation": "eager", "ignore_mismatched_sizes": True}

    def use_hf_reference_state_for_hf_parity(self) -> bool:
        """DSV4 parity seeds Megatron from the reduced canonical HF oracle state.

        The public checkpoint uses Miles/RadixArk source names and full model
        shapes, while the validation oracle uses canonical HF names and reduced
        fit-only axes. This hook is validation-only; production loading remains
        tied to the normal Bridge checkpoint source.
        """
        return True

    def configure_oracle_provider(self, provider: Any, *, case_config: Any) -> None:
        """Mirrors HF oracle reductions while keeping DSV4 hard kernel invariants."""
        del case_config
        hooks = list(getattr(provider, "_pre_wrap_hooks", []))
        kept = [hook for hook in hooks if not self._is_bridge_hf_load_hook(hook)]
        if len(kept) != len(hooks):
            provider._pre_wrap_hooks = kept
        self._apply_oracle_shape_overrides(provider)
        provider.kv_lora_rank = 512
        provider.kv_channels = 512
        provider.qk_pos_emb_head_dim = 64
        provider.num_query_groups = 1
        provider.num_moe_experts = _ORACLE_NUM_EXPERTS
        provider.moe_ffn_hidden_size = _ORACLE_FFN_HIDDEN_SIZE
        provider.ffn_hidden_size = _ORACLE_FFN_HIDDEN_SIZE
        provider.moe_shared_expert_intermediate_size = _ORACLE_FFN_HIDDEN_SIZE
        provider.moe_router_topk = _ORACLE_NUM_EXPERTS_PER_TOK
        provider.dsv4_o_groups = 1
        provider.dsv4_o_lora_rank = 1024
        provider.dsa_indexer_n_heads = _ORACLE_INDEX_HEADS
        provider.dsa_indexer_head_dim = 128
        provider.dsa_indexer_topk = _ORACLE_INDEX_TOPK
        provider.dsv4_oracle_freeze_attn_sink = True

    @staticmethod
    def _is_bridge_hf_load_hook(hook: Any) -> bool:
        fn = getattr(hook, "func", hook)
        name = getattr(fn, "__name__", "")
        qualname = getattr(fn, "__qualname__", "")
        return name in {
            "load_weights_hf_to_megatron",
            "_optimized_load_weights_hf_to_megatron",
        } or qualname.endswith(".load_weights_hf_to_megatron")

    def _apply_oracle_shape_overrides(self, config: Any) -> None:
        """Reduces memory-heavy axes only; head_dim/window/o-rank stay production-sized."""
        config.hidden_size = _ORACLE_HIDDEN_SIZE
        config.q_lora_rank = _ORACLE_Q_LORA_RANK
        config.num_attention_heads = _ORACLE_NUM_ATTENTION_HEADS
        config.n_routed_experts = _ORACLE_NUM_EXPERTS
        config.num_experts_per_tok = _ORACLE_NUM_EXPERTS_PER_TOK
        config.moe_intermediate_size = _ORACLE_FFN_HIDDEN_SIZE
        config.o_groups = 1
        config.index_n_heads = _ORACLE_INDEX_HEADS
        config.index_head_dim = 128
        config.index_topk = _ORACLE_INDEX_TOPK
        config.dsv4_oracle_freeze_attn_sink = True
        config.dsv4_oracle_source_aliases = True


def ensure_dsv4_bridge_registered() -> None:
    from art.megatron.dsv4.bridge import ensure_dsv4_bridge_registered as _ensure

    _ensure()


DSV4_HANDLER = Dsv4Handler()
