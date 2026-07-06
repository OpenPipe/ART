from __future__ import annotations

import re
from typing import Any, Sequence, cast

import torch

from art.megatron.model_support.handlers.default_dense import (
    DefaultMoeHandler,
    _compile_workaround_flags_for_provider,
    _require_moe_experts,
)
from art.megatron.model_support.handlers.qwen3_common import (
    _context_parallel_world_size,
)
from art.megatron.model_support.spec import (
    CompileWorkaroundConfig,
    ExpertPackedLoraGroup,
    ExpertPackedLoraSlot,
    HfWeightSource,
    LayerFamilyInstance,
    RolloutWeightsMode,
)

_GPT_OSS_MOE_COMPILE_WORKAROUND_FLAGS = (
    "alltoall_dtoh",
    "alltoall_dispatch_preprocess",
    "deepep_dispatch_combine",
    "deepep_permute_restore",
    "flex_token_dispatch_combine",
    "te_triton_permute_with_mask_map",
)
_ART_MOE_EXPERT_KEY_RE = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts)\.(?P<expert>\d+)\."
    r"(?P<module>gate_up_proj|down_proj)\.(?P<lora>lora_[AB])\.weight$"
)
_VLLM_MOE_KEY_RE = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts)\."
    r"(?:(?P<base_layer>base_layer)\.)?(?P<lora>lora_[AB])\.weight$"
)
_GPT_OSS_MXFP4_EXPERT_WEIGHT_RE = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.(?:gate_up_proj|down_proj)$"
)


class GptOssMoeHandler(DefaultMoeHandler):
    key = "gpt_oss_moe"
    is_moe = True
    native_vllm_lora_status = "wip"

    def identity_lora_model_config(self, base_config: Any) -> Any:
        return getattr(base_config, "text_config", base_config)

    def _identity_lora_parameter_suffixes(
        self,
        target_modules: list[str],
    ) -> tuple[str, ...]:
        suffixes = list(super()._identity_lora_parameter_suffixes(target_modules))
        target_set = set(target_modules)
        if {"experts", "gate_proj", "up_proj"} & target_set:
            suffixes.append("experts.gate_up_proj")
        if {"experts", "down_proj"} & target_set:
            suffixes.append("experts.down_proj")
        return tuple(dict.fromkeys(suffixes))

    def configure_provider_for_runtime(self, provider: Any) -> None:
        _register_gpt_oss_attention_mapping_types()
        sliding_window = _gpt_oss_sliding_window(provider)
        provider.art_flex_core_attention_wrapper = _gpt_oss_flex_core_attention_wrapper
        provider.art_flex_sliding_windows = (sliding_window,)
        provider.moe_shared_expert_overlap = False
        provider.moe_router_dtype = None
        _install_weighted_bias_quick_geglu_patch()

    def patch_bridge(self, bridge: Any) -> None:
        def _hf_weight_source(
            hf_param: str,
            *,
            task: Any | None = None,
        ) -> HfWeightSource | None:
            return self.hf_weight_source(
                bridge,
                hf_param,
                task=task,
            )

        setattr(bridge, "_art_hf_weight_source", _hf_weight_source)
        model_bridge = getattr(bridge, "_model_bridge", None)
        if model_bridge is not None and model_bridge is not bridge:
            if type(model_bridge) is object:
                return
            if type(model_bridge).__module__.startswith("megatron.bridge."):
                setattr(
                    type(model_bridge),
                    "_art_hf_weight_source",
                    staticmethod(_hf_weight_source),
                )
                return
            setattr(model_bridge, "_art_hf_weight_source", _hf_weight_source)

    def hf_weight_source(
        self,
        bridge: Any,
        hf_param: str,
        *,
        task: Any | None = None,
    ) -> HfWeightSource | None:
        del bridge, task
        if _GPT_OSS_MXFP4_EXPERT_WEIGHT_RE.match(hf_param) is None:
            return None
        return HfWeightSource(
            logical_key=hf_param,
            physical_key_options=(
                (hf_param,),
                (f"{hf_param}_blocks", f"{hf_param}_scales"),
            ),
            kind="bridge_materialized",
        )

    def vllm_engine_args(
        self,
        *,
        rollout_weights_mode: RolloutWeightsMode,
    ) -> dict[str, object]:
        del rollout_weights_mode
        return {"moe_backend": "triton_unfused"}

    def vllm_server_args(self) -> dict[str, object]:
        return {"tool_call_parser": "openai"}

    def install_preprocess_patch(self, model_chunks: Sequence[Any]) -> None:
        _install_gpt_oss_preprocess_patch(model_chunks)

    def get_forward_kwargs(self, model: Any, **kwargs: Any) -> dict[str, Any]:
        return _gpt_oss_forward_kwargs(model, **kwargs)

    def collect_layer_families(self, provider: Any) -> list[LayerFamilyInstance]:
        if int(getattr(provider, "num_moe_experts", 0) or 0) <= 0:
            raise TypeError("GPT OSS MoE handler received a dense provider")
        families = [
            LayerFamilyInstance(key="gpt_oss_sliding_attention", layer_index=0),
            LayerFamilyInstance(key="grouped_moe_mlp", layer_index=0),
        ]
        if int(getattr(provider, "num_layers", 2) or 2) > 1:
            families.append(
                LayerFamilyInstance(key="gpt_oss_full_attention", layer_index=1)
            )
        return families

    def apply_lora_adapters(
        self,
        model_chunks: Sequence[Any],
        provider: Any,
        *,
        target_modules: list[str],
        rank: int,
        alpha: int,
    ) -> None:
        from megatron.core.transformer.attention import SelfAttention
        from megatron.core.transformer.transformer_layer import TransformerLayer

        from art.megatron.lora import (
            _adapter_model_prefix,
            _is_language_transformer_layer_name,
            wrap_grouped_moe_experts_3d,
            wrap_standard_self_attention,
        )

        target_set = set(target_modules)
        for chunk in model_chunks:
            for module_name, module in chunk.named_modules():
                if not isinstance(module, TransformerLayer):
                    continue
                if not _is_language_transformer_layer_name(module_name):
                    continue
                if not isinstance(module.self_attention, SelfAttention):
                    raise TypeError(
                        "GPT OSS expected a SelfAttention module, got "
                        f"{type(module.self_attention)}"
                    )
                adapter_model_prefix = _adapter_model_prefix(module)
                wrap_standard_self_attention(
                    module.self_attention,
                    adapter_model_prefix=adapter_model_prefix,
                    provider=provider,
                    target_modules=target_set,
                    rank=rank,
                    alpha=alpha,
                )
                wrap_grouped_moe_experts_3d(
                    _require_moe_experts(module),
                    adapter_model_prefix=adapter_model_prefix,
                    target_modules=target_set,
                    rank=rank,
                    alpha=alpha,
                )

    def build_adapter_weights_by_base(
        self,
        model_chunks: Sequence[Any],
    ) -> dict[str, list[Any]]:
        from megatron.core.transformer.transformer_layer import TransformerLayer

        from art.megatron.lora import _is_language_transformer_layer_name
        from art.megatron.weights.adapter_export import (
            add_grouped_moe_adapter_weights,
            add_standard_self_attention_adapter_weights,
            layer_base_prefix,
        )

        adapter_weights_by_base: dict[str, list[Any]] = {}
        for chunk in model_chunks:
            for module_name, module in chunk.named_modules():
                if not isinstance(module, TransformerLayer):
                    continue
                if not _is_language_transformer_layer_name(module_name):
                    continue
                layer_prefix = layer_base_prefix(module, module_name=module_name)
                add_standard_self_attention_adapter_weights(
                    adapter_weights_by_base,
                    layer_prefix=layer_prefix,
                    self_attention=module.self_attention,
                )
                add_grouped_moe_adapter_weights(
                    adapter_weights_by_base,
                    layer_prefix=layer_prefix,
                    experts=_require_moe_experts(module),
                )
        return adapter_weights_by_base

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
                        pack_layout="interleaved_gate_up_rank_major_expert_cols",
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

    def to_vllm_lora_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        adapter_config: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return _to_vllm_lora_tensors(tensors, adapter_config=adapter_config)

    def from_vllm_lora_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        adapter_config: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        return _from_vllm_lora_tensors(tensors, adapter_config=adapter_config)

    def compile_workaround_config(
        self,
        provider: Any,
    ) -> CompileWorkaroundConfig:
        return CompileWorkaroundConfig(
            flags=_compile_workaround_flags_for_provider(
                provider,
                _GPT_OSS_MOE_COMPILE_WORKAROUND_FLAGS,
            ),
            shared_expert_state="none",
            disable_compile=False,
        )


GPT_OSS_MOE_HANDLER = GptOssMoeHandler()


def _register_gpt_oss_attention_mapping_types() -> None:
    from megatron.bridge.models.conversion.param_mapping import AutoMapping

    AutoMapping.register_module_type("GptOssArtFlexCoreAttention", "column")


def _gpt_oss_sliding_window(provider: Any) -> int:
    window_size = getattr(provider, "window_size", None)
    if window_size is None:
        raise RuntimeError("GPT OSS provider is missing window_size")
    if isinstance(window_size, tuple | list):
        if len(window_size) != 2:
            raise RuntimeError(f"Unsupported GPT OSS window_size: {window_size}")
        left, right = (int(window_size[0]), int(window_size[1]))
        if right != 0:
            raise RuntimeError(f"Unsupported GPT OSS right window: {window_size}")
        return left + 1
    return int(window_size)


def _gpt_oss_sliding_window_for_layer(provider: Any, layer_number: int) -> int | None:
    layer_types = getattr(provider, "layer_types", None)
    layer_index = int(layer_number) - 1
    if layer_types is not None:
        return (
            _gpt_oss_sliding_window(provider)
            if layer_types[layer_index] == "sliding_attention"
            else None
        )
    skip_freq = int(getattr(provider, "window_attn_skip_freq", 0) or 0)
    if skip_freq <= 0:
        return None
    if layer_index % skip_freq != 0:
        return None
    return _gpt_oss_sliding_window(provider)


def _gpt_oss_flex_core_attention_wrapper(
    provider: Any,
    base_cls: type[Any],
) -> type[Any]:
    class GptOssArtFlexCoreAttention(base_cls):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            config: Any,
            layer_number: int,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            super().__init__(config, layer_number, *args, **kwargs)
            self.art_sliding_window = _gpt_oss_sliding_window_for_layer(
                provider,
                layer_number,
            )

    return GptOssArtFlexCoreAttention


def _gpt_oss_forward_kwargs(model: Any, **kwargs: Any) -> dict[str, Any]:
    attention_bias = kwargs.get("attention_bias")
    from art.megatron.context_parallel.types import ArtContextParallelState

    module = model
    while hasattr(module, "module"):
        module = module.module
    gpt_module = getattr(module, "language_model", module)
    if isinstance(attention_bias, ArtContextParallelState):
        setattr(
            gpt_module,
            "_art_gpt_oss_rotary_seq_len",
            int(attention_bias.rank_plan.original_seq_len),
        )
    else:
        setattr(gpt_module, "_art_gpt_oss_rotary_seq_len", None)
    return {"extra_block_kwargs": kwargs}


def _gather_absolute_rotary_pos_emb(
    table_source: torch.Tensor,
    *,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    embedding_dim = int(table_source.shape[-1])
    batch_size, sequence_length = position_ids.shape
    gathered = table_source.view(table_source.shape[0], embedding_dim).index_select(
        0,
        position_ids.reshape(-1),
    )
    return (
        gathered.view(batch_size, sequence_length, embedding_dim)
        .permute(1, 0, 2)
        .contiguous()
        .unsqueeze(2)
    )


def _gpt_oss_absolute_rotary_pos_emb(
    gpt_module: Any,
    *,
    seq_len: int,
) -> torch.Tensor:
    rotary_output = gpt_module.rotary_pos_emb(seq_len, packed_seq=True)
    rotary_pos_emb = (
        rotary_output[0] if isinstance(rotary_output, tuple) else rotary_output
    )
    if not torch.is_tensor(rotary_pos_emb):
        raise TypeError(
            "GPT OSS YaRN rotary embedding returned "
            f"{type(rotary_pos_emb).__name__}, expected Tensor"
        )
    return rotary_pos_emb


def _install_gpt_oss_preprocess_patch(model_chunks: Sequence[Any]) -> None:
    from megatron.core.models.gpt.gpt_model import GPTModel

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
            *args: Any,
            _gpt_module: Any = gpt_module,
            _preprocess: Any = preprocess,
            **kwargs: Any,
        ) -> tuple[Any, ...]:
            position_ids = kwargs.get("position_ids")
            cp_world_size = _context_parallel_world_size(
                getattr(_gpt_module, "config", None),
            )
            packed_seq_params = kwargs.get("packed_seq_params")
            rotary_module = getattr(_gpt_module, "rotary_pos_emb", None)
            rotary_cp_group = getattr(rotary_module, "cp_group", None)
            packed_cp_group = getattr(packed_seq_params, "cp_group", None)
            uses_local_cp_positions = (
                isinstance(position_ids, torch.Tensor)
                and position_ids.ndim == 2
                and cp_world_size > 1
                and (rotary_cp_group is not None or packed_cp_group is not None)
            )
            if uses_local_cp_positions:
                if rotary_cp_group is not None:
                    setattr(rotary_module, "cp_group", None)
                if packed_cp_group is not None:
                    setattr(packed_seq_params, "cp_group", None)
            try:
                preproc_output = list(_preprocess(*args, **kwargs))
            finally:
                if uses_local_cp_positions:
                    if rotary_cp_group is not None:
                        setattr(rotary_module, "cp_group", rotary_cp_group)
                    if packed_cp_group is not None:
                        setattr(packed_seq_params, "cp_group", packed_cp_group)
            decoder_input = cast(torch.Tensor, preproc_output[0])
            if not decoder_input.requires_grad and decoder_input.is_leaf:
                decoder_input.requires_grad_(True)
            rotary_pos_emb = preproc_output[1]
            if not isinstance(position_ids, torch.Tensor) or not torch.is_tensor(
                rotary_pos_emb,
            ):
                return tuple(preproc_output)
            if position_ids.ndim != 2:
                raise RuntimeError(
                    "GPT OSS expected 2D position_ids for YaRN rotary gathering, "
                    f"got shape {tuple(position_ids.shape)}"
                )
            rotary_seq_len = getattr(_gpt_module, "_art_gpt_oss_rotary_seq_len", None)
            if rotary_seq_len is None:
                rotary_seq_len = int(position_ids.shape[-1]) * max(cp_world_size, 1)
            table_source = _gpt_oss_absolute_rotary_pos_emb(
                _gpt_module,
                seq_len=int(rotary_seq_len),
            )
            preproc_output[1] = _gather_absolute_rotary_pos_emb(
                table_source,
                position_ids=position_ids,
            )
            return tuple(preproc_output)

        gpt_module._preprocess = preprocess_hook  # type: ignore[attr-defined]


def _install_weighted_bias_quick_geglu_patch() -> None:
    import megatron.core.fusions.fused_bias_geglu as fused_bias_geglu
    import megatron.core.transformer.moe.experts as moe_experts

    original = fused_bias_geglu.weighted_bias_quick_geglu_impl
    if getattr(original, "_art_gpt_oss_compile_safe", False):
        return

    def _weighted_bias_quick_geglu_impl(
        input: torch.Tensor,
        bias: torch.Tensor | None,
        weights: torch.Tensor,
        fp8_input_store: bool = False,
        linear_offset: float = 0.0,
        clamp_value: float | None = None,
    ) -> torch.Tensor:
        ori_shape = input.shape
        if len(ori_shape) not in {2, 3}:
            raise AssertionError(
                "weighted_bias_quick_geglu_impl expects 2D or 3D input"
            )
        input_dtype = input.dtype
        input = input.view(-1, ori_shape[-1])
        if bias is not None:
            input = input + bias
        gate, up = input.chunk(2, -1)
        if clamp_value is not None:
            gate = gate.clamp(min=None, max=clamp_value)
            up = up.clamp(min=-clamp_value, max=clamp_value)
        output = fused_bias_geglu.quick_gelu(gate) * (up + linear_offset)
        output = (output * weights).to(input_dtype)
        return (
            output
            if len(ori_shape) == 2
            else output.view(ori_shape[0], ori_shape[1], -1)
        )

    setattr(_weighted_bias_quick_geglu_impl, "_art_gpt_oss_compile_safe", True)
    setattr(
        fused_bias_geglu,
        "weighted_bias_quick_geglu_impl",
        _weighted_bias_quick_geglu_impl,
    )
    setattr(
        moe_experts,
        "weighted_bias_quick_geglu_impl",
        _weighted_bias_quick_geglu_impl,
    )


def _to_vllm_key(key: str) -> str:
    return key.replace(".self_attn.", ".attn.", 1)


def _from_vllm_key(key: str) -> str:
    return key.replace(".attn.", ".self_attn.", 1)


def _pack_vllm_3d_lora_b(blocks: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(blocks, dim=0)
    return stacked.permute(1, 2, 0).reshape(stacked.shape[1], -1).contiguous()


def _unpack_vllm_3d_lora_b(
    tensor: torch.Tensor,
    *,
    num_experts: int,
    rank: int,
) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], rank, num_experts).permute(2, 0, 1)


def _gate_up_b_to_vllm(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[0] % 2 != 0:
        raise RuntimeError(
            f"GPT OSS gate/up lora_B rows {tensor.shape[0]} are not even"
        )
    gate, up = tensor.split(tensor.shape[0] // 2, dim=0)
    return torch.stack((gate, up), dim=1).flatten(0, 1).contiguous()


def _gate_up_b_from_vllm(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[0] % 2 != 0:
        raise RuntimeError(
            f"GPT OSS gate/up lora_B rows {tensor.shape[0]} are not even"
        )
    return torch.cat((tensor[::2], tensor[1::2]), dim=0).contiguous()


def _group_art_moe_tensors(
    tensors: dict[str, torch.Tensor],
) -> dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]]:
    grouped: dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]] = {}
    for key, tensor in tensors.items():
        match = _ART_MOE_EXPERT_KEY_RE.match(key)
        if match is None:
            continue
        grouped.setdefault(match.group("prefix"), {}).setdefault(
            int(match.group("expert")),
            {},
        ).setdefault(match.group("module"), {})[match.group("lora")] = tensor
    return grouped


def _vllm_moe_config(adapter_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(adapter_config)
    target_modules = [
        module
        for module in list(config.get("target_modules") or [])
        if module not in {"gate_proj", "up_proj", "down_proj", "gate_up_proj"}
    ]
    if "experts" not in target_modules:
        target_modules.append("experts")
    config["target_modules"] = target_modules
    config["art_merged_lora_delta_unsupported_target_modules"] = ["experts"]
    return config


def _to_vllm_lora_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    adapter_config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    grouped = _group_art_moe_tensors(tensors)
    transformed: dict[str, torch.Tensor] = {}
    if not grouped:
        has_fused_experts = False
        for key, tensor in tensors.items():
            vllm_key = _to_vllm_key(key)
            if vllm_key in transformed:
                raise RuntimeError(
                    f"Duplicate GPT OSS LoRA tensor after conversion: {vllm_key}"
                )
            transformed[vllm_key] = tensor
            has_fused_experts = has_fused_experts or (
                _VLLM_MOE_KEY_RE.match(vllm_key) is not None
            )
        return (
            transformed,
            _vllm_moe_config(adapter_config) if has_fused_experts else adapter_config,
        )

    used_keys: set[str] = set()
    for prefix, experts in grouped.items():
        vllm_prefix = _to_vllm_key(prefix)
        gate_up_a: list[torch.Tensor] = []
        gate_up_b: list[torch.Tensor] = []
        down_a: list[torch.Tensor] = []
        down_b: list[torch.Tensor] = []
        for expert in sorted(experts):
            modules = experts[expert]
            try:
                gate_up_a_tensor = modules["gate_up_proj"]["lora_A"]
                gate_up_b_tensor = modules["gate_up_proj"]["lora_B"]
                down_a_tensor = modules["down_proj"]["lora_A"]
                down_b_tensor = modules["down_proj"]["lora_B"]
            except KeyError as exc:
                raise RuntimeError(
                    f"Incomplete GPT OSS MoE LoRA block for {prefix}.{expert}"
                ) from exc
            gate_up_a.append(gate_up_a_tensor.contiguous())
            gate_up_b.append(_gate_up_b_to_vllm(gate_up_b_tensor))
            down_a.append(down_a_tensor.contiguous())
            down_b.append(down_b_tensor.contiguous())
            for module_name in ("gate_up_proj", "down_proj"):
                for lora_name in ("lora_A", "lora_B"):
                    used_keys.add(f"{prefix}.{expert}.{module_name}.{lora_name}.weight")

        transformed[f"{vllm_prefix}.base_layer.lora_A.weight"] = torch.cat(
            gate_up_a,
            dim=0,
        ).contiguous()
        transformed[f"{vllm_prefix}.base_layer.lora_B.weight"] = _pack_vllm_3d_lora_b(
            gate_up_b
        )
        transformed[f"{vllm_prefix}.lora_A.weight"] = torch.cat(
            down_a,
            dim=0,
        ).contiguous()
        transformed[f"{vllm_prefix}.lora_B.weight"] = _pack_vllm_3d_lora_b(down_b)

    for key, tensor in tensors.items():
        if key in used_keys:
            continue
        vllm_key = _to_vllm_key(key)
        if vllm_key in transformed:
            raise RuntimeError(
                f"Duplicate GPT OSS LoRA tensor after conversion: {vllm_key}"
            )
        transformed[vllm_key] = tensor
    return transformed, _vllm_moe_config(adapter_config)


def _from_vllm_lora_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    adapter_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in tensors.items():
        match = _VLLM_MOE_KEY_RE.match(key)
        if match is None:
            continue
        slot = (
            f"{'base_layer.' if match.group('base_layer') else ''}{match.group('lora')}"
        )
        grouped.setdefault(match.group("prefix"), {})[slot] = tensor
    if not grouped:
        transformed = {_from_vllm_key(key): tensor for key, tensor in tensors.items()}
        if len(transformed) != len(tensors):
            raise RuntimeError("Duplicate GPT OSS LoRA tensor after vLLM conversion")
        return transformed

    rank = int(adapter_config["r"])
    transformed: dict[str, torch.Tensor] = {}
    used_keys: set[str] = set()
    for prefix, slots in grouped.items():
        try:
            gate_up_a = slots["base_layer.lora_A"]
            gate_up_b = slots["base_layer.lora_B"]
            down_a = slots["lora_A"]
            down_b = slots["lora_B"]
        except KeyError as exc:
            raise RuntimeError(
                f"Incomplete GPT OSS vLLM MoE LoRA block for {prefix}"
            ) from exc
        if gate_up_a.shape[0] % rank != 0:
            raise RuntimeError(
                f"{prefix}: gate/up lora_A shape {tuple(gate_up_a.shape)} "
                f"is not divisible by rank {rank}"
            )
        num_experts = gate_up_a.shape[0] // rank
        art_prefix = _from_vllm_key(prefix)
        gate_up_b_by_expert = _unpack_vllm_3d_lora_b(
            gate_up_b,
            num_experts=num_experts,
            rank=rank,
        )
        down_b_by_expert = _unpack_vllm_3d_lora_b(
            down_b,
            num_experts=num_experts,
            rank=rank,
        )
        for expert in range(num_experts):
            row = expert * rank
            transformed[f"{art_prefix}.{expert}.gate_up_proj.lora_A.weight"] = (
                gate_up_a[row : row + rank].contiguous()
            )
            transformed[f"{art_prefix}.{expert}.gate_up_proj.lora_B.weight"] = (
                _gate_up_b_from_vllm(gate_up_b_by_expert[expert])
            )
            transformed[f"{art_prefix}.{expert}.down_proj.lora_A.weight"] = down_a[
                row : row + rank
            ].contiguous()
            transformed[f"{art_prefix}.{expert}.down_proj.lora_B.weight"] = (
                down_b_by_expert[expert].contiguous()
            )
        used_keys.update(
            {
                f"{prefix}.base_layer.lora_A.weight",
                f"{prefix}.base_layer.lora_B.weight",
                f"{prefix}.lora_A.weight",
                f"{prefix}.lora_B.weight",
            }
        )

    for key, tensor in tensors.items():
        if key in used_keys:
            continue
        art_key = _from_vllm_key(key)
        if art_key in transformed:
            raise RuntimeError(
                f"Duplicate GPT OSS LoRA tensor after conversion: {art_key}"
            )
        transformed[art_key] = tensor
    return transformed
