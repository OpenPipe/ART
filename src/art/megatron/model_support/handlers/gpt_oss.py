from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
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
from art.megatron.model_support.internal_padding import (
    group_expert_lora_tensors,
)
from art.megatron.model_support.internal_padding import (
    pack_vllm_3d_lora_b as _pack_vllm_3d_lora_b,
)
from art.megatron.model_support.internal_padding import (
    pad_dim_right as _pad_dim_right,
)
from art.megatron.model_support.internal_padding import (
    round_up_to_multiple as _round_up_to_multiple,
)
from art.megatron.model_support.internal_padding import (
    trim_dim_right as _trim_dim_right,
)
from art.megatron.model_support.internal_padding import (
    unpack_vllm_3d_lora_b as _unpack_vllm_3d_lora_b,
)
from art.megatron.model_support.internal_padding import (
    zero_lora_padding as _zero_gpt_oss_lora_padding_tensor_set,
)
from art.megatron.model_support.internal_padding import (
    zero_ranges as _zero_ranges,
)
from art.megatron.model_support.spec import (
    CompileWorkaroundConfig,
    ExpertPackedLoraGroup,
    ExpertPackedLoraSlot,
    HfWeightSource,
    LayerFamilyInstance,
)

_GPT_OSS_MOE_COMPILE_WORKAROUND_FLAGS = (
    # Torch 2.11 can illegal-address in compiled GPT-OSS MoE under CP2 even
    # when routing replay is disabled. Narrower dispatcher/expert/postprocess
    # eager boundaries do not isolate it, so keep only the MoE layer eager while
    # the surrounding transformer layer and compiled flex attention stay active.
    "moe_forward",
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
_ART_PACKED_MOE_KEY_RE = re.compile(
    r"^.*\.mlp\.experts\.(?:base_layer\.)?lora_[AB]\.weight$"
)
_GPT_OSS_HIDDEN_ALIGNMENT = 128
# Keep the serialized expert shape topology-independent while ensuring each
# local FFN remains 128-aligned for the supported ETP sizes 1, 2, 4, and 8.
_GPT_OSS_MOE_FFN_ALIGNMENT = 128 * 8
_GPT_OSS_LOGICAL_HIDDEN_ATTR = "art_gpt_oss_logical_hidden_size"
_GPT_OSS_INTERNAL_HIDDEN_ATTR = "art_gpt_oss_internal_hidden_size"
_GPT_OSS_LOGICAL_MOE_FFN_ATTR = "art_gpt_oss_logical_moe_ffn_hidden_size"
_GPT_OSS_INTERNAL_MOE_FFN_ATTR = "art_gpt_oss_internal_moe_ffn_hidden_size"
_GPT_OSS_QUICK_GELU_ONE = 1.1429453389509778


def _pad_gpt_oss_hidden_with_bias_coordinate(
    tensor: torch.Tensor,
    *,
    logical: int,
    internal: int,
) -> torch.Tensor:
    if int(tensor.shape[-1]) != logical or internal <= logical:
        raise RuntimeError(
            "GPT OSS expert bias encoding requires a padded hidden dimension: "
            f"shape={tuple(tensor.shape)}, hidden={logical}->{internal}"
        )
    pad_shape = (*tensor.shape[:-1], internal - logical - 1)
    return torch.cat(
        [
            tensor,
            tensor.new_ones(*tensor.shape[:-1], 1),
            tensor.new_zeros(pad_shape),
        ],
        dim=-1,
    )


def _pad_gpt_oss_gate_up_dim0(
    tensor: torch.Tensor,
    *,
    logical: int,
    internal: int,
) -> torch.Tensor:
    if logical == internal:
        return tensor.contiguous()
    if int(tensor.shape[0]) != 2 * logical:
        raise RuntimeError(
            "Expected GPT OSS gate/up logical dim "
            f"{2 * logical}, got {tuple(tensor.shape)}"
        )
    gate, up = torch.split(tensor, logical, dim=0)
    return torch.cat(
        [
            _pad_dim_right(gate, dim=0, size=internal),
            _pad_dim_right(up, dim=0, size=internal),
        ],
        dim=0,
    ).contiguous()


def _trim_gpt_oss_gate_up_dim0(
    tensor: torch.Tensor,
    *,
    logical: int,
    internal: int,
) -> torch.Tensor:
    if logical == internal:
        return tensor.contiguous()
    if int(tensor.shape[0]) != 2 * internal:
        raise RuntimeError(
            "Expected GPT OSS gate/up internal dim "
            f"{2 * internal}, got {tuple(tensor.shape)}"
        )
    return torch.cat(
        [
            tensor.narrow(0, 0, logical),
            tensor.narrow(0, internal, logical),
        ],
        dim=0,
    ).contiguous()


def _gate_up_to_etp_shard_order(tensor: torch.Tensor, etp_size: int) -> torch.Tensor:
    if etp_size == 1:
        return tensor
    local_rows = tensor.shape[0] // (2 * etp_size)
    return (
        tensor.reshape(2, etp_size, local_rows, *tensor.shape[1:])
        .transpose(0, 1)
        .reshape(tensor.shape)
        .contiguous()
    )


def _gate_up_from_etp_shard_order(tensor: torch.Tensor, etp_size: int) -> torch.Tensor:
    if etp_size == 1:
        return tensor
    local_rows = tensor.shape[0] // (2 * etp_size)
    return (
        tensor.reshape(etp_size, 2, local_rows, *tensor.shape[1:])
        .transpose(0, 1)
        .reshape(tensor.shape)
        .contiguous()
    )


def _pad_gpt_oss_interleaved_gate_up_last(
    tensor: torch.Tensor,
    *,
    logical: int,
    internal: int,
) -> torch.Tensor:
    if logical == internal:
        return tensor.contiguous()
    if int(tensor.shape[-1]) != 2 * logical:
        raise RuntimeError(
            "Expected GPT OSS interleaved gate/up logical dim "
            f"{2 * logical}, got {tuple(tensor.shape)}"
        )
    gate = tensor[..., 0::2]
    up = tensor[..., 1::2]
    return (
        torch.stack(
            [
                _pad_dim_right(gate, dim=-1, size=internal),
                _pad_dim_right(up, dim=-1, size=internal),
            ],
            dim=-1,
        )
        .flatten(-2)
        .contiguous()
    )


def _trim_gpt_oss_interleaved_gate_up_last(
    tensor: torch.Tensor,
    *,
    logical: int,
    internal: int,
) -> torch.Tensor:
    if logical == internal:
        return tensor.contiguous()
    if int(tensor.shape[-1]) != 2 * internal:
        raise RuntimeError(
            "Expected GPT OSS interleaved gate/up internal dim "
            f"{2 * internal}, got {tuple(tensor.shape)}"
        )
    gate = tensor[..., 0::2].narrow(-1, 0, logical)
    up = tensor[..., 1::2].narrow(-1, 0, logical)
    return torch.stack([gate, up], dim=-1).flatten(-2).contiguous()


def _configure_gpt_oss_moe_internal_padding(provider: Any) -> None:
    if int(getattr(provider, "num_moe_experts", 0) or 0) <= 0:
        return
    logical_hidden = int(
        getattr(provider, _GPT_OSS_LOGICAL_HIDDEN_ATTR, provider.hidden_size) or 0
    )
    logical_ffn = int(
        getattr(
            provider,
            _GPT_OSS_LOGICAL_MOE_FFN_ATTR,
            getattr(provider, "moe_ffn_hidden_size", 0),
        )
        or 0
    )
    if logical_hidden <= 0 or logical_ffn <= 0:
        raise RuntimeError(
            "GPT OSS provider is missing hidden_size or moe_ffn_hidden_size"
        )
    internal_hidden = _round_up_to_multiple(logical_hidden, _GPT_OSS_HIDDEN_ALIGNMENT)
    internal_ffn = _round_up_to_multiple(logical_ffn, _GPT_OSS_MOE_FFN_ALIGNMENT)
    setattr(provider, _GPT_OSS_LOGICAL_HIDDEN_ATTR, logical_hidden)
    setattr(provider, _GPT_OSS_INTERNAL_HIDDEN_ATTR, internal_hidden)
    setattr(provider, _GPT_OSS_LOGICAL_MOE_FFN_ATTR, logical_ffn)
    setattr(provider, _GPT_OSS_INTERNAL_MOE_FFN_ATTR, internal_ffn)
    # The external GPT-OSS hidden/FFN sizes remain logical. The TE grouped-MLP
    # patch below builds only expert GEMMs with the internal padded sizes so
    # CUTLASS grouped GEMM stays off TE's routed-shape cuBLAS cache path.
    provider.art_moe_grouped_gemm_hidden_size = internal_hidden
    provider.art_moe_grouped_gemm_ffn_hidden_size = internal_ffn


def _gpt_oss_padding_sizes_from_provider(provider: Any) -> tuple[int, int, int, int]:
    logical_hidden = int(
        getattr(
            provider, _GPT_OSS_LOGICAL_HIDDEN_ATTR, getattr(provider, "hidden_size", 0)
        )
        or 0
    )
    internal_hidden = int(
        getattr(provider, _GPT_OSS_INTERNAL_HIDDEN_ATTR, logical_hidden) or 0
    )
    logical_ffn = int(
        getattr(
            provider,
            _GPT_OSS_LOGICAL_MOE_FFN_ATTR,
            getattr(provider, "moe_ffn_hidden_size", 0),
        )
        or 0
    )
    internal_ffn = int(
        getattr(provider, _GPT_OSS_INTERNAL_MOE_FFN_ATTR, logical_ffn) or 0
    )
    if (
        logical_hidden <= 0
        or internal_hidden < logical_hidden
        or logical_ffn <= 0
        or internal_ffn < logical_ffn
    ):
        raise RuntimeError(
            "Invalid GPT OSS MoE padding sizes: "
            f"hidden={logical_hidden}->{internal_hidden}, "
            f"ffn={logical_ffn}->{internal_ffn}"
        )
    return logical_hidden, internal_hidden, logical_ffn, internal_ffn


def _gpt_oss_padding_sizes_from_module(
    module: Any,
) -> tuple[int, int, int, int] | None:
    config = getattr(module, "config", None)
    if config is None:
        return None
    return _gpt_oss_padding_sizes_from_provider(config)


def _gpt_oss_padding_sizes_from_hf_config(
    hf_config: Any | None,
) -> tuple[int, int, int, int] | None:
    if hf_config is None:
        return None
    config = getattr(hf_config, "text_config", hf_config)
    hidden = int(getattr(config, "hidden_size", 0) or 0)
    ffn = int(getattr(config, "intermediate_size", 0) or 0)
    if hidden <= 0 or ffn <= 0:
        return None
    return (
        hidden,
        _round_up_to_multiple(hidden, _GPT_OSS_HIDDEN_ALIGNMENT),
        ffn,
        _round_up_to_multiple(ffn, _GPT_OSS_MOE_FFN_ALIGNMENT),
    )


@lru_cache(maxsize=8)
def _gpt_oss_config_dict(base_model_name_or_path: str) -> dict[str, Any]:
    config_path = Path(base_model_name_or_path) / "config.json"
    if not config_path.exists():
        from huggingface_hub import hf_hub_download

        config_path = Path(hf_hub_download(base_model_name_or_path, "config.json"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return dict(config.get("text_config") or config)


def _gpt_oss_padding_sizes_from_adapter_config(
    adapter_config: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    base_model = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model:
        raise RuntimeError("GPT OSS LoRA conversion requires base_model_name_or_path")
    config = _gpt_oss_config_dict(base_model)
    hidden = int(config.get("hidden_size", 0) or 0)
    ffn = int(config.get("intermediate_size", 0) or 0)
    if hidden <= 0 or ffn <= 0:
        raise RuntimeError(
            f"GPT OSS config is missing hidden_size or intermediate_size: {base_model}"
        )
    return (
        hidden,
        _round_up_to_multiple(hidden, _GPT_OSS_HIDDEN_ALIGNMENT),
        ffn,
        _round_up_to_multiple(ffn, _GPT_OSS_MOE_FFN_ALIGNMENT),
    )


def _gpt_oss_padding_sizes_from_model_chunks(
    model_chunks: Sequence[Any],
) -> tuple[int, int, int, int] | None:
    for chunk in model_chunks:
        config = getattr(chunk, "config", None)
        if config is None:
            config = getattr(getattr(chunk, "module", None), "config", None)
        if config is not None:
            return _gpt_oss_padding_sizes_from_provider(config)
    return None


def _install_gpt_oss_grouped_mlp_padding_patch() -> None:
    from megatron.core.transformer.moe.experts import TEGroupedMLP

    if getattr(TEGroupedMLP, "_art_gpt_oss_padding_patch", False):
        return
    original_init = TEGroupedMLP.__init__
    original_forward = TEGroupedMLP.forward

    def __init__(
        self: Any,
        num_local_experts: int,
        config: Any,
        submodules: Any,
        pg_collection: Any | None = None,
    ) -> None:
        if not hasattr(config, _GPT_OSS_INTERNAL_HIDDEN_ATTR):
            original_init(self, num_local_experts, config, submodules, pg_collection)
            return
        logical_hidden, internal_hidden, logical_ffn, internal_ffn = (
            _gpt_oss_padding_sizes_from_provider(config)
        )
        original_hidden = config.hidden_size
        original_ffn = config.moe_ffn_hidden_size
        original_add_bias = config.add_bias_linear
        config.hidden_size = internal_hidden
        config.moe_ffn_hidden_size = internal_ffn
        config.add_bias_linear = False
        try:
            original_init(self, num_local_experts, config, submodules, pg_collection)
        finally:
            config.hidden_size = original_hidden
            config.moe_ffn_hidden_size = original_ffn
            config.add_bias_linear = original_add_bias
        setattr(self, _GPT_OSS_LOGICAL_HIDDEN_ATTR, logical_hidden)
        setattr(self, _GPT_OSS_INTERNAL_HIDDEN_ATTR, internal_hidden)
        setattr(self, _GPT_OSS_LOGICAL_MOE_FFN_ATTR, logical_ffn)
        setattr(self, _GPT_OSS_INTERNAL_MOE_FFN_ATTR, internal_ffn)

    def forward(
        self: Any,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logical_hidden = int(getattr(self, _GPT_OSS_LOGICAL_HIDDEN_ATTR, 0) or 0)
        internal_hidden = int(getattr(self, _GPT_OSS_INTERNAL_HIDDEN_ATTR, 0) or 0)
        if internal_hidden <= 0 or internal_hidden == logical_hidden:
            return original_forward(
                self,
                permuted_local_hidden_states,
                tokens_per_expert,
                permuted_probs,
            )
        padded = _pad_gpt_oss_hidden_with_bias_coordinate(
            permuted_local_hidden_states,
            logical=logical_hidden,
            internal=internal_hidden,
        )
        output, output_bias = original_forward(
            self,
            padded,
            tokens_per_expert,
            permuted_probs,
        )
        return _trim_dim_right(output, dim=-1, size=logical_hidden), output_bias

    cast(Any, TEGroupedMLP).__init__ = __init__
    cast(Any, TEGroupedMLP).forward = forward
    setattr(TEGroupedMLP, "_art_gpt_oss_padding_patch", True)


def _patch_gpt_oss_mapping_registry(target: Any) -> None:
    bridge_type = type(target)
    original = getattr(bridge_type, "mapping_registry", None)
    if original is None or getattr(original, "_art_gpt_oss_padding_patch", False):
        return

    def mapping_registry(self: Any) -> Any:
        upstream = original(self)
        hf_config = getattr(self, "hf_config", None)
        if hf_config is None:
            hf_config = getattr(getattr(self, "hf_pretrained", None), "config", None)
        return _gpt_oss_padded_mapping_registry(
            upstream,
            padding_sizes=_gpt_oss_padding_sizes_from_hf_config(hf_config),
        )

    setattr(mapping_registry, "_art_gpt_oss_padding_patch", True)
    bridge_type.mapping_registry = mapping_registry


def _gpt_oss_padded_mapping_registry(
    upstream_registry: Any,
    *,
    padding_sizes: tuple[int, int, int, int] | None,
) -> Any:
    from megatron.bridge.models.conversion.mapping_registry import (
        MegatronMappingRegistry,
    )
    from megatron.bridge.models.conversion.param_mapping import AutoMapping
    from megatron.bridge.models.gpt_oss.gpt_oss_bridge import (
        GPTOSSMLPDownProjMapping,
        GPTOSSMLPGateUpProjMapping,
    )

    if padding_sizes is None:
        raise RuntimeError("GPT OSS padded mappings require model padding dimensions")
    logical_hidden, internal_hidden, logical_ffn, internal_ffn = padding_sizes
    if internal_hidden <= logical_hidden or internal_ffn <= logical_ffn:
        raise RuntimeError(
            "GPT OSS expert bias encoding requires hidden and FFN padding: "
            f"hidden={logical_hidden}->{internal_hidden}, "
            f"ffn={logical_ffn}->{internal_ffn}"
        )

    class _ArtGptOssMLPGateUpProjMapping(GPTOSSMLPGateUpProjMapping):
        def __init__(
            self,
            megatron_param: str,
            weight_hf_param: str,
            bias_hf_param: str,
        ) -> None:
            cast(Any, AutoMapping).__init__(
                self,
                megatron_param,
                {"weight": weight_hf_param, "bias": bias_hf_param},
            )
            self.allow_hf_name_mismatch = True

        @property
        def group_key(self) -> str:
            return cast(dict[str, str], self.hf_param)["weight"]

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
            from megatron.bridge.utils.common_utils import (
                extract_expert_number_from_param,
            )

            global_expert_number = extract_expert_number_from_param(self.megatron_param)
            expert_weight = hf_weights["weight"][global_expert_number]
            expert_bias = hf_weights["bias"][global_expert_number]
            normalized_param = self._normalize_expert_param_name(self.megatron_param)
            target_param = get_module_and_param_from_name(
                megatron_module,
                normalized_param,
            )[1]
            full_target_shape = (
                int(target_param.shape[0]) * int(self.tp_size),
                int(target_param.shape[1]),
            )
            if full_target_shape != (2 * internal_ffn, internal_hidden):
                raise RuntimeError(
                    f"Unexpected GPT OSS gate/up target shape {full_target_shape}; "
                    f"expected {(2 * internal_ffn, internal_hidden)}"
                )
            aligned = _align_expert_weight_to_shape(
                expert_weight,
                torch.Size((2 * logical_ffn, logical_hidden)),
                "gate_up_proj",
            )
            padded = _pad_dim_right(
                _pad_gpt_oss_gate_up_dim0(
                    self._interleave(aligned),
                    logical=logical_ffn,
                    internal=internal_ffn,
                ),
                dim=1,
                size=internal_hidden,
            )
            padded[:logical_ffn, logical_hidden] = expert_bias[::2]
            padded[
                internal_ffn : internal_ffn + logical_ffn,
                logical_hidden,
            ] = expert_bias[1::2]
            padded[logical_ffn, logical_hidden] = _GPT_OSS_QUICK_GELU_ONE
            return AutoMapping.hf_to_megatron(
                self,
                _gate_up_to_etp_shard_order(padded, self.tp_size),
                megatron_module,
            )

        def megatron_to_hf(
            self,
            megatron_weights: torch.Tensor | None,
            megatron_module: Any | None,
        ) -> dict[str, torch.Tensor]:
            converted = AutoMapping.megatron_to_hf(
                self,
                megatron_weights,
                megatron_module,
            )
            if not converted:
                return converted
            tensor = next(iter(converted.values()))
            if self.ep_size > 1:
                tensor = torch.stack(
                    [
                        _gate_up_from_etp_shard_order(expert, self.tp_size)
                        for expert in tensor
                    ]
                )
            else:
                tensor = _gate_up_from_etp_shard_order(tensor, self.tp_size)
            gate = tensor[..., :logical_ffn, :logical_hidden]
            up = tensor[..., internal_ffn : internal_ffn + logical_ffn, :logical_hidden]
            interleaved = torch.empty(
                *tensor.shape[:-2],
                2 * logical_ffn,
                logical_hidden,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            interleaved[..., 0::2, :] = gate
            interleaved[..., 1::2, :] = up
            names = cast(dict[str, str], self.hf_param)
            return {
                names["weight"]: interleaved.transpose(-1, -2).contiguous(),
                names["bias"]: torch.stack(
                    [
                        tensor[..., :logical_ffn, logical_hidden],
                        tensor[
                            ...,
                            internal_ffn : internal_ffn + logical_ffn,
                            logical_hidden,
                        ],
                    ],
                    dim=-1,
                )
                .flatten(-2)
                .contiguous(),
            }

        def resolve(self, captures: tuple[str, ...]) -> Any:
            megatron_param, hf_params = self._resolve_names(captures)
            names = cast(dict[str, str], hf_params)
            return type(self)(megatron_param, names["weight"], names["bias"])

    class _ArtGptOssMLPDownProjMapping(GPTOSSMLPDownProjMapping):
        def __init__(
            self,
            megatron_param: str,
            weight_hf_param: str,
            bias_hf_param: str,
        ) -> None:
            cast(Any, AutoMapping).__init__(
                self,
                megatron_param,
                {"weight": weight_hf_param, "bias": bias_hf_param},
            )
            self.allow_hf_name_mismatch = True

        @property
        def group_key(self) -> str:
            return cast(dict[str, str], self.hf_param)["weight"]

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
            from megatron.bridge.utils.common_utils import (
                extract_expert_number_from_param,
            )

            global_expert_number = extract_expert_number_from_param(self.megatron_param)
            # Index through ExpertTensorSlice so global EP metadata is preserved.
            expert_weight = hf_weights["weight"][global_expert_number]
            expert_bias = hf_weights["bias"][global_expert_number]
            normalized_param = self._normalize_expert_param_name(self.megatron_param)
            target_param = get_module_and_param_from_name(
                megatron_module,
                normalized_param,
            )[1]
            full_target_shape = (
                int(target_param.shape[0]),
                int(target_param.shape[1]) * int(self.tp_size),
            )
            if full_target_shape != (internal_hidden, internal_ffn):
                raise RuntimeError(
                    f"Unexpected GPT OSS down target shape {full_target_shape}; "
                    f"expected {(internal_hidden, internal_ffn)}"
                )
            aligned = _align_expert_weight_to_shape(
                expert_weight,
                torch.Size((logical_hidden, logical_ffn)),
                "down_proj",
            )
            padded = _pad_dim_right(
                _pad_dim_right(aligned, dim=0, size=internal_hidden),
                dim=1,
                size=internal_ffn,
            )
            padded[:logical_hidden, logical_ffn] = expert_bias
            return AutoMapping.hf_to_megatron(self, padded, megatron_module)

        def megatron_to_hf(
            self,
            megatron_weights: torch.Tensor | None,
            megatron_module: Any | None,
        ) -> dict[str, torch.Tensor]:
            converted = AutoMapping.megatron_to_hf(
                self,
                megatron_weights,
                megatron_module,
            )
            if not converted:
                return converted
            tensor = next(iter(converted.values()))
            names = cast(dict[str, str], self.hf_param)
            return {
                names["weight"]: tensor[..., :logical_hidden, :logical_ffn]
                .transpose(-1, -2)
                .contiguous(),
                names["bias"]: tensor[..., :logical_hidden, logical_ffn].contiguous(),
            }

        def resolve(self, captures: tuple[str, ...]) -> Any:
            megatron_param, hf_params = self._resolve_names(captures)
            names = cast(dict[str, str], hf_params)
            return type(self)(megatron_param, names["weight"], names["bias"])

    mappings = []
    for mapping in upstream_registry.mappings:
        if isinstance(mapping, GPTOSSMLPGateUpProjMapping):
            hf_param = cast(str, mapping.hf_param)
            if not hf_param.endswith("_bias"):
                mappings.append(
                    _ArtGptOssMLPGateUpProjMapping(
                        mapping.megatron_param,
                        hf_param,
                        f"{hf_param}_bias",
                    )
                )
        elif isinstance(mapping, GPTOSSMLPDownProjMapping):
            hf_param = cast(str, mapping.hf_param)
            if not hf_param.endswith("_bias"):
                mappings.append(
                    _ArtGptOssMLPDownProjMapping(
                        mapping.megatron_param,
                        hf_param,
                        f"{hf_param}_bias",
                    )
                )
        else:
            mappings.append(mapping)
    return MegatronMappingRegistry(*mappings)


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
        _configure_gpt_oss_moe_internal_padding(provider)
        _install_gpt_oss_grouped_mlp_padding_patch()
        sliding_window = _gpt_oss_sliding_window(provider)
        provider.art_flex_core_attention_wrapper = _gpt_oss_flex_core_attention_wrapper
        provider.art_flex_sliding_windows = (sliding_window,)
        provider.moe_shared_expert_overlap = False
        provider.art_moe_grouped_gemm_bias_encoded = True
        # Match GPT-OSS HF and vLLM routing precision. HybridEP converts the
        # selected route probabilities to fp32 at its communication boundary.
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
            _patch_gpt_oss_mapping_registry(model_bridge)
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

    def vllm_engine_args(self) -> dict[str, object]:
        return {"moe_backend": "triton_unfused"}

    def vllm_server_args(self) -> dict[str, object]:
        return {"tool_call_parser": "openai"}

    def install_preprocess_patch(self, model_chunks: Sequence[Any]) -> None:
        _install_gpt_oss_preprocess_patch(model_chunks)

    def zero_internal_padding_grads(self, model_chunks: Sequence[Any]) -> None:
        _zero_gpt_oss_moe_lora_padding(model_chunks, grads=True, params=False)

    def zero_internal_padding_params(self, model_chunks: Sequence[Any]) -> None:
        _zero_gpt_oss_moe_lora_padding(model_chunks, grads=False, params=True)

    def canonicalize_loaded_lora_state(
        self,
        state: dict[str, Any],
        model_chunks: Sequence[Any],
    ) -> dict[str, Any]:
        return _canonicalize_gpt_oss_loaded_lora_state(state, model_chunks)

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
                    moe_parameterization=provider._art_lora_moe_parameterization,
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
            decoder_input = cast(torch.Tensor | None, preproc_output[0])
            if (
                decoder_input is not None
                and not decoder_input.requires_grad
                and decoder_input.is_leaf
            ):
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

        setattr(gpt_module, "_preprocess", preprocess_hook)


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
    return config


def _trim_gpt_oss_lora_for_vllm(
    key: str,
    tensor: torch.Tensor,
    *,
    adapter_config: dict[str, Any],
) -> torch.Tensor:
    sizes = _gpt_oss_padding_sizes_from_adapter_config(adapter_config)
    if sizes is None:
        return tensor.contiguous()
    logical_hidden, internal_hidden, logical_ffn, internal_ffn = sizes
    match = _ART_MOE_EXPERT_KEY_RE.match(key)
    if match is not None:
        module = match.group("module")
        lora = match.group("lora")
        if module == "gate_up_proj" and lora == "lora_A":
            return _trim_dim_right(tensor, dim=-1, size=logical_hidden)
        if module == "gate_up_proj" and lora == "lora_B":
            if int(tensor.shape[0]) == 2 * logical_ffn:
                return tensor.contiguous()
            return _trim_gpt_oss_gate_up_dim0(
                tensor,
                logical=logical_ffn,
                internal=internal_ffn,
            )
        if module == "down_proj" and lora == "lora_A":
            return _trim_dim_right(tensor, dim=-1, size=logical_ffn)
        if module == "down_proj" and lora == "lora_B":
            return _trim_dim_right(tensor, dim=0, size=logical_hidden)
    if _ART_PACKED_MOE_KEY_RE.match(key):
        if key.endswith(".base_layer.lora_A.weight"):
            return _trim_dim_right(tensor, dim=-1, size=logical_hidden)
        if key.endswith(".base_layer.lora_B.weight"):
            if int(tensor.shape[0]) == 2 * logical_ffn:
                return tensor.contiguous()
            return _trim_gpt_oss_gate_up_dim0(
                tensor,
                logical=logical_ffn,
                internal=internal_ffn,
            )
        if key.endswith(".lora_A.weight"):
            return _trim_dim_right(tensor, dim=-1, size=logical_ffn)
        if key.endswith(".lora_B.weight"):
            return _trim_dim_right(tensor, dim=0, size=logical_hidden)
    return tensor.contiguous()


def _pad_gpt_oss_lora_from_vllm(
    key: str,
    tensor: torch.Tensor,
    *,
    adapter_config: dict[str, Any],
) -> torch.Tensor:
    sizes = _gpt_oss_padding_sizes_from_adapter_config(adapter_config)
    if sizes is None:
        return tensor.contiguous()
    _logical_hidden, internal_hidden, _logical_ffn, internal_ffn = sizes
    match = _ART_MOE_EXPERT_KEY_RE.match(key)
    if match is not None:
        module = match.group("module")
        lora = match.group("lora")
        if module == "gate_up_proj" and lora == "lora_A":
            return _pad_dim_right(tensor, dim=-1, size=internal_hidden)
        if module == "gate_up_proj" and lora == "lora_B":
            return _pad_gpt_oss_gate_up_dim0(
                tensor,
                logical=tensor.shape[0] // 2,
                internal=internal_ffn,
            )
        if module == "down_proj" and lora == "lora_A":
            return _pad_dim_right(tensor, dim=-1, size=internal_ffn)
        if module == "down_proj" and lora == "lora_B":
            return _pad_dim_right(tensor, dim=0, size=internal_hidden)
    if _ART_PACKED_MOE_KEY_RE.match(key):
        if key.endswith(".base_layer.lora_A.weight"):
            return _pad_dim_right(tensor, dim=-1, size=internal_hidden)
        if key.endswith(".base_layer.lora_B.weight"):
            return _pad_gpt_oss_gate_up_dim0(
                tensor,
                logical=tensor.shape[0] // 2,
                internal=internal_ffn,
            )
        if key.endswith(".lora_A.weight"):
            return _pad_dim_right(tensor, dim=-1, size=internal_ffn)
        if key.endswith(".lora_B.weight"):
            return _pad_dim_right(tensor, dim=0, size=internal_hidden)
    return tensor.contiguous()


def _zero_gpt_oss_moe_lora_padding(
    model_chunks: Sequence[Any],
    *,
    grads: bool,
    params: bool,
) -> None:
    if not grads and not params:
        return
    sizes = _gpt_oss_padding_sizes_from_model_chunks(model_chunks)
    if sizes is None:
        return
    logical_hidden, internal_hidden, logical_ffn, internal_ffn = sizes
    if logical_hidden == internal_hidden and logical_ffn == internal_ffn:
        return
    with torch.no_grad():
        for chunk in model_chunks:
            for module in chunk.modules():
                prefix = getattr(module, "adapter_model_prefix", None)
                if not isinstance(prefix, str) or ".mlp.experts." not in prefix:
                    continue
                if prefix.endswith(".gate_up_proj"):
                    if hasattr(module, "A_T"):
                        _zero_gpt_oss_lora_padding_tensor_set(
                            cast(torch.nn.Parameter, module.A_T),
                            dim=-2,
                            logical=logical_hidden,
                            internal=internal_hidden,
                            components=(internal_hidden,),
                            grads=grads,
                            params=params,
                        )
                    if hasattr(module, "B_T"):
                        _zero_gpt_oss_lora_padding_tensor_set(
                            cast(torch.nn.Parameter, module.B_T),
                            dim=-1,
                            logical=logical_ffn,
                            internal=internal_ffn,
                            components=(internal_ffn, internal_ffn),
                            grads=grads,
                            params=params,
                        )
                elif prefix.endswith(".down_proj"):
                    if hasattr(module, "A_T"):
                        _zero_gpt_oss_lora_padding_tensor_set(
                            cast(torch.nn.Parameter, module.A_T),
                            dim=-2,
                            logical=logical_ffn,
                            internal=internal_ffn,
                            components=(internal_ffn,),
                            grads=grads,
                            params=params,
                        )
                    if hasattr(module, "B_T"):
                        _zero_gpt_oss_lora_padding_tensor_set(
                            cast(torch.nn.Parameter, module.B_T),
                            dim=-1,
                            logical=logical_hidden,
                            internal=internal_hidden,
                            components=(internal_hidden,),
                            grads=grads,
                            params=params,
                        )


def _zero_gpt_oss_lora_padding_state_tensor(
    key: str,
    tensor: torch.Tensor,
    *,
    logical_hidden: int,
    internal_hidden: int,
    logical_ffn: int,
    internal_ffn: int,
) -> torch.Tensor:
    result = tensor.clone().contiguous()
    match = _ART_MOE_EXPERT_KEY_RE.match(key)
    if match is not None:
        module = match.group("module")
        lora = match.group("lora")
        if module == "gate_up_proj" and lora == "lora_A":
            _zero_ranges(result, dim=-1, ranges=((logical_hidden, internal_hidden),))
        elif module == "gate_up_proj" and lora == "lora_B":
            _zero_ranges(
                result,
                dim=0,
                ranges=(
                    (logical_ffn, internal_ffn),
                    (internal_ffn + logical_ffn, 2 * internal_ffn),
                ),
            )
        elif module == "down_proj" and lora == "lora_A":
            _zero_ranges(result, dim=-1, ranges=((logical_ffn, internal_ffn),))
        elif module == "down_proj" and lora == "lora_B":
            _zero_ranges(result, dim=0, ranges=((logical_hidden, internal_hidden),))
        return result
    if _ART_PACKED_MOE_KEY_RE.match(key):
        if key.endswith(".base_layer.lora_A.weight"):
            _zero_ranges(result, dim=-1, ranges=((logical_hidden, internal_hidden),))
        elif key.endswith(".base_layer.lora_B.weight"):
            _zero_ranges(
                result,
                dim=0,
                ranges=(
                    (logical_ffn, internal_ffn),
                    (internal_ffn + logical_ffn, 2 * internal_ffn),
                ),
            )
        elif key.endswith(".lora_A.weight"):
            _zero_ranges(result, dim=-1, ranges=((logical_ffn, internal_ffn),))
        elif key.endswith(".lora_B.weight"):
            _zero_ranges(result, dim=0, ranges=((logical_hidden, internal_hidden),))
    return result


def _canonicalize_gpt_oss_loaded_lora_state(
    state: dict[str, Any],
    model_chunks: Sequence[Any],
) -> dict[str, Any]:
    sizes = _gpt_oss_padding_sizes_from_model_chunks(model_chunks)
    if sizes is None:
        return state
    logical_hidden, internal_hidden, logical_ffn, internal_ffn = sizes
    if logical_hidden == internal_hidden and logical_ffn == internal_ffn:
        return state
    return {
        key: _zero_gpt_oss_lora_padding_state_tensor(
            key,
            value,
            logical_hidden=logical_hidden,
            internal_hidden=internal_hidden,
            logical_ffn=logical_ffn,
            internal_ffn=internal_ffn,
        )
        if torch.is_tensor(value)
        else value
        for key, value in state.items()
    }


def _to_vllm_lora_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    adapter_config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    grouped = group_expert_lora_tensors(tensors, _ART_MOE_EXPERT_KEY_RE)
    transformed: dict[str, torch.Tensor] = {}
    if not grouped:
        has_fused_experts = False
        for key, tensor in tensors.items():
            vllm_key = _to_vllm_key(key)
            if vllm_key in transformed:
                raise RuntimeError(
                    f"Duplicate GPT OSS LoRA tensor after conversion: {vllm_key}"
                )
            transformed[vllm_key] = _trim_gpt_oss_lora_for_vllm(
                key,
                tensor,
                adapter_config=adapter_config,
            )
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
            gate_up_a.append(
                _trim_gpt_oss_lora_for_vllm(
                    f"{prefix}.{expert}.gate_up_proj.lora_A.weight",
                    gate_up_a_tensor,
                    adapter_config=adapter_config,
                )
            )
            gate_up_b.append(
                _gate_up_b_to_vllm(
                    _trim_gpt_oss_lora_for_vllm(
                        f"{prefix}.{expert}.gate_up_proj.lora_B.weight",
                        gate_up_b_tensor,
                        adapter_config=adapter_config,
                    )
                )
            )
            down_a.append(
                _trim_gpt_oss_lora_for_vllm(
                    f"{prefix}.{expert}.down_proj.lora_A.weight",
                    down_a_tensor,
                    adapter_config=adapter_config,
                )
            )
            down_b.append(
                _trim_gpt_oss_lora_for_vllm(
                    f"{prefix}.{expert}.down_proj.lora_B.weight",
                    down_b_tensor,
                    adapter_config=adapter_config,
                )
            )
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
        transformed[vllm_key] = _trim_gpt_oss_lora_for_vllm(
            key,
            tensor,
            adapter_config=adapter_config,
        )
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
        transformed = {
            _from_vllm_key(key): _pad_gpt_oss_lora_from_vllm(
                _from_vllm_key(key),
                tensor,
                adapter_config=adapter_config,
            )
            for key, tensor in tensors.items()
        }
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
            gate_up_a_key = f"{art_prefix}.{expert}.gate_up_proj.lora_A.weight"
            gate_up_b_key = f"{art_prefix}.{expert}.gate_up_proj.lora_B.weight"
            down_a_key = f"{art_prefix}.{expert}.down_proj.lora_A.weight"
            down_b_key = f"{art_prefix}.{expert}.down_proj.lora_B.weight"
            transformed[gate_up_a_key] = _pad_gpt_oss_lora_from_vllm(
                gate_up_a_key,
                gate_up_a[row : row + rank],
                adapter_config=adapter_config,
            )
            transformed[gate_up_b_key] = _pad_gpt_oss_lora_from_vllm(
                gate_up_b_key,
                _gate_up_b_from_vllm(gate_up_b_by_expert[expert]),
                adapter_config=adapter_config,
            )
            transformed[down_a_key] = _pad_gpt_oss_lora_from_vllm(
                down_a_key,
                down_a[row : row + rank],
                adapter_config=adapter_config,
            )
            transformed[down_b_key] = _pad_gpt_oss_lora_from_vllm(
                down_b_key,
                down_b_by_expert[expert],
                adapter_config=adapter_config,
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
        transformed[art_key] = _pad_gpt_oss_lora_from_vllm(
            art_key,
            tensor,
            adapter_config=adapter_config,
        )
    return transformed
