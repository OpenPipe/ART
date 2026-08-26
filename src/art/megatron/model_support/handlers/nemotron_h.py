from __future__ import annotations

from typing import Any, Literal, Sequence, cast

import torch

from art.megatron.mamba.plan import build_mamba_execution_plan
from art.megatron.mamba.runtime import MAMBA_STATE_KEY, install_mamba_prefix_tree_hooks
from art.megatron.model_support.handlers.default_dense import DefaultMoeHandler
from art.megatron.model_support.internal_padding import (
    pad_dim_right,
    round_up_to_multiple,
    trim_dim_right,
)
from art.megatron.model_support.spec import (
    LayerFamilyInstance,
    PrefixTreeModelStateContext,
)
from art.megatron.recurrent import parse_recurrent_prefix_tree

_MOE_FFN_ALIGNMENT = 128
_LOGICAL_MOE_FFN_ATTR = "art_nemotron_h_logical_moe_ffn_hidden_size"
_EXPERT_MAPPING_AXES = {
    "decoder.layers.*.mlp.experts.linear_fc1.weight*": 0,
    "decoder.layers.*.mlp.experts.linear_fc2.weight*": 1,
    "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight": 0,
    "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight": 1,
}


def _configure_moe_padding(provider: Any) -> None:
    logical = int(
        getattr(
            provider,
            _LOGICAL_MOE_FFN_ATTR,
            getattr(provider, "moe_ffn_hidden_size", 0),
        )
        or 0
    )
    if logical <= 0:
        raise RuntimeError("Nemotron-H provider is missing moe_ffn_hidden_size")
    setattr(provider, _LOGICAL_MOE_FFN_ATTR, logical)
    provider.moe_ffn_hidden_size = round_up_to_multiple(logical, _MOE_FFN_ALIGNMENT)


def _padding_sizes_from_hf_config(config: Any) -> tuple[int, int]:
    logical = int(getattr(config, "moe_intermediate_size", 0) or 0)
    if logical <= 0:
        raise RuntimeError("Nemotron-H config is missing moe_intermediate_size")
    return logical, round_up_to_multiple(logical, _MOE_FFN_ALIGNMENT)


def _model_bridge_hf_config(model_bridge: Any) -> Any:
    config = getattr(model_bridge, "hf_config", None)
    if config is None:
        config = getattr(getattr(model_bridge, "hf_pretrained", None), "config", None)
    if config is None:
        raise RuntimeError("Nemotron-H Bridge is missing its HF config")
    return config


def _padded_mapping_registry(
    upstream: Any,
    *,
    logical: int,
    internal: int,
) -> Any:
    from megatron.bridge.models.conversion.mapping_registry import (
        MegatronMappingRegistry,
    )
    from megatron.bridge.models.conversion.param_mapping import AutoMapping

    class PaddedExpertMapping(AutoMapping):
        def __init__(
            self,
            megatron_param: str,
            hf_param: str,
            axis: int,
            permute_dims: tuple[int, ...] | None = None,
        ) -> None:
            super().__init__(megatron_param, hf_param, permute_dims)
            self.axis = axis

        def resolve(self, captures: tuple[str, ...]) -> Any:
            megatron_param, hf_param = self._resolve_names(captures)
            return type(self)(
                megatron_param,
                cast(str, hf_param),
                self.axis,
                self.permute_dims,
            )

        def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: Any):
            if int(hf_weights.shape[self.axis]) != logical:
                raise RuntimeError(
                    f"{self.hf_param}: expected expert width {logical}, got "
                    f"{tuple(hf_weights.shape)}"
                )
            return super().hf_to_megatron(
                pad_dim_right(hf_weights, dim=self.axis, size=internal),
                megatron_module,
            )

        def megatron_to_hf(self, megatron_weights: Any, megatron_module: Any):
            converted = super().megatron_to_hf(megatron_weights, megatron_module)
            return {
                key: trim_dim_right(value, dim=self.axis, size=logical)
                for key, value in converted.items()
            }

    mappings = []
    found = set()
    for mapping in upstream.mappings:
        name = str(getattr(mapping, "megatron_param", ""))
        axis = _EXPERT_MAPPING_AXES.get(name)
        if axis is None:
            mappings.append(mapping)
            continue
        mappings.append(
            PaddedExpertMapping(
                name,
                cast(str, mapping.hf_param),
                axis,
                getattr(mapping, "permute_dims", None),
            )
        )
        found.add(name)
    if found != set(_EXPERT_MAPPING_AXES):
        raise RuntimeError(f"Nemotron-H expert mappings changed: {sorted(found)}")
    return MegatronMappingRegistry(*mappings)


def _patch_bridge_padding(model_bridge: Any) -> None:
    bridge_type = type(model_bridge)
    mapping_registry = bridge_type.mapping_registry
    if not getattr(mapping_registry, "_art_nemotron_h_padding", False):

        def padded_mapping_registry(self: Any) -> Any:
            logical, internal = _padding_sizes_from_hf_config(
                _model_bridge_hf_config(self)
            )
            return _padded_mapping_registry(
                mapping_registry(self), logical=logical, internal=internal
            )

        setattr(padded_mapping_registry, "_art_nemotron_h_padding", True)
        bridge_type.mapping_registry = padded_mapping_registry

    config_export = bridge_type.megatron_to_hf_config
    if not getattr(config_export, "_art_nemotron_h_padding", False):

        def padded_config_export(cls: type[Any], provider: Any) -> dict[str, Any]:
            del cls
            config = dict(config_export(provider))
            config["moe_intermediate_size"] = int(
                getattr(provider, _LOGICAL_MOE_FFN_ATTR)
            )
            return config

        setattr(padded_config_export, "_art_nemotron_h_padding", True)
        bridge_type.megatron_to_hf_config = classmethod(padded_config_export)


class NemotronHHandler(DefaultMoeHandler):
    key = "nemotron_h_moe"
    native_vllm_lora_status = "wip"

    def identity_lora_model_config(self, base_config: Any) -> Any:
        return base_config

    def _identity_lora_parameter_suffixes(
        self, target_modules: list[str]
    ) -> tuple[str, ...]:
        suffixes = list(super()._identity_lora_parameter_suffixes(target_modules))
        targets = set(target_modules)
        if "in_proj" in targets:
            suffixes.append("mixer.in_proj.weight")
        if "out_proj" in targets:
            suffixes.append("mixer.out_proj.weight")
        return tuple(dict.fromkeys(suffixes))

    def patch_bridge(self, bridge: Any) -> None:
        model_bridge = getattr(bridge, "_model_bridge", None)
        if type(model_bridge).__name__ != "NemotronHBridge":
            raise TypeError(
                "Nemotron-H requires Megatron Bridge's native NemotronHBridge, got "
                f"{type(model_bridge).__name__}"
            )
        _patch_bridge_padding(model_bridge)

    def configure_provider_for_runtime(self, provider: Any) -> None:
        if type(provider).__name__ != "MambaModelProvider":
            raise TypeError(
                f"Nemotron-H requires MambaModelProvider, got {type(provider).__name__}"
            )
        if getattr(provider, "virtual_pipeline_model_parallel_size", None) is not None:
            raise ValueError("Nemotron-H does not support virtual pipeline parallelism")
        _configure_moe_padding(provider)
        provider.use_mamba_mem_eff_path = True

    def install_preprocess_patch(self, model_chunks: Sequence[Any]) -> None:
        install_mamba_prefix_tree_hooks(model_chunks)

    def build_prefix_tree_model_state(
        self,
        context: PrefixTreeModelStateContext,
    ) -> dict[str, Any]:
        tree = parse_recurrent_prefix_tree(context.group_ids, context.parent_ids)
        token_layout = context.attention_token_layout_index
        cp_size = 1 if token_layout is None else len(token_layout.token_counts_by_rank)
        cp_state = context.context_parallel_state
        cp_rank = 0 if cp_state is None else int(cp_state.rank_plan.rank)
        return {
            MAMBA_STATE_KEY: build_mamba_execution_plan(
                tree,
                device=torch.device(context.device),
                cp_rank=cp_rank,
                cp_size=cp_size,
                token_layout=token_layout,
            )
        }

    def get_forward_kwargs(self, model: Any, **kwargs: Any) -> dict[str, Any]:
        del model
        return {
            "attention_mask": kwargs["attention_bias"],
            "packed_seq_params": None,
        }

    def correctness_precision(self) -> Literal["bf16", "fp32"]:
        return "bf16"

    def collect_layer_families(self, provider: Any) -> list[LayerFamilyInstance]:
        pattern = str(provider.hybrid_layer_pattern).split("/", 1)[0]
        families = []
        for symbol, key in (
            ("M", "mamba_2"),
            ("*", "standard_attention"),
            ("E", "grouped_moe_mlp"),
        ):
            if symbol in pattern:
                families.append(
                    LayerFamilyInstance(
                        key=key,
                        count=pattern.count(symbol),
                        layer_index=pattern.index(symbol),
                    )
                )
        if int(getattr(provider, "moe_shared_expert_intermediate_size", 0) or 0):
            families.append(
                LayerFamilyInstance(
                    key="shared_experts_mlp",
                    count=pattern.count("E"),
                    layer_index=pattern.index("E"),
                )
            )
        return families


NEMOTRON_H_HANDLER = NemotronHHandler()
