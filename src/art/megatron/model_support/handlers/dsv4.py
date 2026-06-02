from __future__ import annotations

from typing import Any, Sequence, cast

import torch

from art.megatron.model_support.handlers.default_dense import (
    DefaultMoeHandler,
    _compile_workaround_flags_for_provider,
    _require_moe_experts,
)
from art.megatron.model_support.spec import CompileWorkaroundConfig, LayerFamilyInstance


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

        for chunk in list(model_chunks):
            module: Any = chunk
            while hasattr(module, "module"):
                module = module.module
            gpt_module = (
                module
                if isinstance(module, GPTModel)
                else cast(GPTModel, getattr(module, "language_model"))
            )
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
            flags=_compile_workaround_flags_for_provider(provider),
            shared_expert_state=self._shared_expert_compile_state(provider),
        )

    def ensure_hf_reference_registered(self) -> None:
        from art.megatron.dsv4.hf_config import ensure_dsv4_hf_model_registered

        ensure_dsv4_hf_model_registered()

    def prepare_hf_reference_config(self, config: Any) -> None:
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")
        config._experts_implementation = "eager"

    def hf_reference_from_pretrained_kwargs(
        self, *, config: Any, dtype: torch.dtype
    ) -> dict[str, Any]:
        del config, dtype
        return {"experts_implementation": "eager"}


def ensure_dsv4_bridge_registered() -> None:
    from art.megatron.dsv4.bridge import ensure_dsv4_bridge_registered as _ensure

    _ensure()


DSV4_HANDLER = Dsv4Handler()
