from functools import lru_cache, partial
import re
from typing import Any, Iterable, Mapping, cast

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    ReplicatedMapping,
    RowParallelMapping,
    extract_expert_number_from_param,
)
from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.core.models.gpt.gpt_model import GPTModel
import torch

from art.megatron.dsv4.spec import get_dsv4_decoder_block_spec

_DSV4_FP4_TABLE = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

_DSV4_FUSED_EXPORT_SPECS = (
    (
        ".self_attn.q_a_proj.weight",
        ".self_attn.kv_proj.weight",
        ".attn.fused_wqa_wkv.weight",
    ),
    (
        ".self_attn.compressor.kv_proj.weight",
        ".self_attn.compressor.gate_proj.weight",
        ".attn.compressor.fused_wkv_wgate.weight",
    ),
    (
        ".self_attn.compressor.indexer.kv_proj.weight",
        ".self_attn.compressor.indexer.gate_proj.weight",
        ".attn.indexer.compressor.fused_wkv_wgate.weight",
    ),
    (
        ".mlp.shared_experts.gate_proj.weight",
        ".mlp.shared_experts.up_proj.weight",
        ".ffn.shared_experts.gate_up_proj.weight",
    ),
)

_DSV4_RENAMED_EXPORT_SUFFIXES = (
    ("lm_head.weight", "head.weight"),
    ("model.hc_head.hc_fn", "model.hc_head_fn"),
    ("model.hc_head.hc_base", "model.hc_head_base"),
    ("model.hc_head.hc_scale", "model.hc_head_scale"),
    (".input_layernorm.weight", ".attn_norm.weight"),
    (".post_attention_layernorm.weight", ".ffn_norm.weight"),
    (".attn_hc.fn", ".hc_attn_fn"),
    (".attn_hc.base", ".hc_attn_base"),
    (".attn_hc.scale", ".hc_attn_scale"),
    (".ffn_hc.fn", ".hc_ffn_fn"),
    (".ffn_hc.base", ".hc_ffn_base"),
    (".ffn_hc.scale", ".hc_ffn_scale"),
    (".mlp.gate.weight", ".ffn.gate.weight"),
    (".mlp.gate.tid2eid", ".ffn.gate.tid2eid"),
    (".mlp.gate.e_score_correction_bias", ".ffn.gate.bias"),
    (".mlp.shared_experts.down_proj.weight", ".ffn.shared_experts.w2.weight"),
    (".self_attn.q_a_norm.weight", ".attn.q_norm.weight"),
    (".self_attn.q_b_proj.weight", ".attn.wq_b.weight"),
    (".self_attn.kv_norm.weight", ".attn.kv_norm.weight"),
    (".self_attn.o_a_proj.weight", ".attn.wo_a.weight"),
    (".self_attn.o_b_proj.weight", ".attn.wo_b.weight"),
    (".self_attn.sinks", ".attn.attn_sink"),
    (".self_attn.compressor.position_bias", ".attn.compressor.ape"),
    (".self_attn.compressor.kv_norm.weight", ".attn.compressor.norm.weight"),
    (
        ".self_attn.compressor.indexer.q_b_proj.weight",
        ".attn.indexer.wq_b.weight",
    ),
    (
        ".self_attn.compressor.indexer.scorer.weights_proj.weight",
        ".attn.indexer.weights_proj.weight",
    ),
    (
        ".self_attn.compressor.indexer.position_bias",
        ".attn.indexer.compressor.ape",
    ),
    (
        ".self_attn.compressor.indexer.kv_norm.weight",
        ".attn.indexer.compressor.norm.weight",
    ),
)
_DSV4_LAYER_TYPE_TO_COMPRESS_RATIO = {
    "sliding_attention": 0,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}


def _dsv4_compress_ratios_from_hf_config(hf_config: Any) -> list[int] | None:
    ratios = getattr(hf_config, "compress_ratios", None)
    if ratios is not None:
        return [int(ratio) for ratio in ratios]
    layer_types = getattr(hf_config, "layer_types", None)
    if layer_types is None:
        return None
    compress_rates = getattr(hf_config, "compress_rates", None) or {}
    return [
        int(
            compress_rates.get(
                layer_type,
                _DSV4_LAYER_TYPE_TO_COMPRESS_RATIO[layer_type],
            )
        )
        for layer_type in layer_types
    ]


def _dequant_dsv4_mxfp4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.dtype not in (torch.int8, torch.uint8):
        raise ValueError(
            f"Expected MXFP4 packed int8/uint8 weight, got {weight.dtype}."
        )
    if scale.dtype != torch.float8_e8m0fnu:
        raise ValueError(f"Expected MXFP4 E8M0 scale, got {scale.dtype}.")
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            f"Expected 2-D MXFP4 weight and scale, got {weight.shape=} {scale.shape=}."
        )

    out_dim, in_bytes = weight.shape
    in_dim = in_bytes * 2
    if in_dim % 32 != 0 or tuple(scale.shape) != (out_dim, in_dim // 32):
        raise ValueError(
            "Unexpected MXFP4 scale shape: "
            f"weight={tuple(weight.shape)} scale={tuple(scale.shape)}."
        )

    if torch.cuda.is_available():
        from art.megatron.dsv4.dequant import dequant_mxfp4_cuda

        return dequant_mxfp4_cuda(weight, scale)

    table = torch.tensor(_DSV4_FP4_TABLE, dtype=torch.float32, device=weight.device)
    packed = weight.contiguous().view(torch.uint8)
    low = (packed & 0x0F).long()
    high = ((packed >> 4) & 0x0F).long()
    decoded = torch.stack((table[low], table[high]), dim=-1).reshape(out_dim, in_dim)
    expanded_scale = scale.float().repeat_interleave(32, dim=1)
    return (decoded * expanded_scale).to(torch.bfloat16)


def _dequant_dsv4_block_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"Expected block-FP8 E4M3 weight, got {weight.dtype}.")
    if scale.dtype != torch.float8_e8m0fnu:
        raise ValueError(f"Expected block-FP8 E8M0 scale, got {scale.dtype}.")
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            f"Expected 2-D block-FP8 weight and scale, got {weight.shape=} {scale.shape=}."
        )

    block = 128
    out_dim, in_dim = weight.shape
    expected_scale_shape = (
        (out_dim + block - 1) // block,
        (in_dim + block - 1) // block,
    )
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            "Unexpected block-FP8 scale shape: "
            f"weight={tuple(weight.shape)} scale={tuple(scale.shape)} "
            f"expected={expected_scale_shape}."
        )

    if torch.cuda.is_available():
        from art.megatron.dsv4.dequant import dequant_block_fp8_cuda

        return dequant_block_fp8_cuda(weight, scale)

    expanded_scale = (
        scale.float().repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    )
    expanded_scale = expanded_scale[:out_dim, :in_dim]
    return (weight.float() * expanded_scale).to(torch.bfloat16)


def _dequant_dsv4_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.dtype in (torch.int8, torch.uint8):
        return _dequant_dsv4_mxfp4(weight, scale)
    if weight.dtype == torch.float8_e4m3fn:
        return _dequant_dsv4_block_fp8(weight, scale)
    return weight


def _quant_dsv4_mxfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2 or weight.shape[1] % 32 != 0:
        raise ValueError(
            f"Expected 2-D MXFP4 weight with K % 32 == 0, got {weight.shape}."
        )
    if weight.device.type == "cuda":
        from art.megatron.dsv4.dequant import quant_mxfp4_cuda

        return quant_mxfp4_cuda(weight)

    out_dim, in_dim = weight.shape
    blocks = weight.float().contiguous().view(out_dim, in_dim // 32, 32)
    amax = blocks.abs().amax(dim=-1).clamp_min(1e-4)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0)))
    scale_e8m0 = scale.to(torch.float8_e8m0fnu)

    scaled = blocks / scale_e8m0.float().unsqueeze(-1)
    thresholds = torch.tensor(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
        device=weight.device,
        dtype=torch.float32,
    )
    codes = torch.bucketize(scaled.abs(), thresholds).to(torch.uint8)
    codes = codes + ((scaled < 0) & (codes != 0)).to(torch.uint8) * 8
    low = codes[..., 0::2]
    high = codes[..., 1::2] << 4
    return (low | high).reshape(out_dim, in_dim // 2), scale_e8m0


def _is_dsv4_routed_expert_weight(name: str) -> bool:
    return ".ffn.experts." in name and name.endswith(
        (".w1.weight", ".w2.weight", ".w3.weight")
    )


def _dsv4_canonical_expert_source_name(name: str) -> str | None:
    replacements = {
        ".gate_proj.weight": ".w1.weight",
        ".up_proj.weight": ".w3.weight",
        ".down_proj.weight": ".w2.weight",
    }
    if ".mlp.experts." not in name:
        return None
    for canonical, source in replacements.items():
        if name.endswith(canonical):
            return (
                name.replace(".mlp.experts.", ".ffn.experts.").removesuffix(canonical)
                + source
            )
    return None


def _is_dsv4_hash_router_table(name: str) -> bool:
    return name.endswith(".ffn.gate.tid2eid")


def _dsv4_expert_id(name: str) -> int:
    match = re.search(r"\.experts\.(\d+)\.", name)
    if match is None:
        raise ValueError(f"Expected DSV4 expert id in weight name: {name}.")
    return int(match.group(1))


def _set_dsv4_expert_id(name: str, expert_id: int) -> str:
    return re.sub(r"\.experts\.\d+\.", f".experts.{expert_id}.", name, count=1)


def _export_dsv4_mxfp4_weight(
    name: str, weight: torch.Tensor
) -> dict[str, torch.Tensor]:
    packed, scale = _quant_dsv4_mxfp4(weight)
    return {
        name: packed.contiguous(),
        f"{name.removesuffix('.weight')}.scale": scale.contiguous(),
    }


def _dsv4_fused_export_key(name: str) -> tuple[str, int] | None:
    for first, second, target in _DSV4_FUSED_EXPORT_SPECS:
        if name.endswith(first):
            return f"{name.removesuffix(first)}{target}", 0
        if name.endswith(second):
            return f"{name.removesuffix(second)}{target}", 1
    return None


def _concat_dsv4_fused_export_parts(
    target: str, parts: dict[int, torch.Tensor]
) -> torch.Tensor:
    first, second = parts[0], parts[1]
    if first.ndim != second.ndim or first.shape[1:] != second.shape[1:]:
        raise ValueError(
            f"Cannot fuse DSV4 export parts for {target}: "
            f"{tuple(first.shape)} vs {tuple(second.shape)}."
        )
    return torch.cat((first, second), dim=0).contiguous()


def _dsv4_source_export_name(name: str) -> str:
    for canonical, source in _DSV4_RENAMED_EXPORT_SUFFIXES:
        if name.endswith(canonical):
            return f"{name.removesuffix(canonical)}{source}"
    return name


def _dsv4_full_parallel_shape(task: WeightConversionTask) -> list[int]:
    param_weight = task.param_weight
    if param_weight is None:
        raise RuntimeError(f"Missing DSV4 export param for {task.global_param_name}")
    shape = list(param_weight.shape)
    if not bool(getattr(param_weight, "tensor_model_parallel", False)):
        tp_size = int(getattr(task.mapping, "tp_size", 1) or 1)
        if task.global_param_name in {
            "embedding.word_embeddings.weight",
            "output_layer.weight",
        } or task.global_param_name.endswith(".self_attention.attn_sink"):
            shape[0] *= tp_size
        elif task.global_param_name.endswith(
            (
                ".self_attention.wq_b.weight",
                ".self_attention.wo_a.weight",
                ".self_attention.indexer.linear_wq_b.weight",
            )
        ):
            shape[0] *= tp_size
        elif task.global_param_name.endswith(
            (
                ".self_attention.wo_b.weight",
                ".mlp.shared_experts.linear_fc2.weight",
            )
        ):
            shape[1] *= tp_size
        return shape
    partition_dim = int(getattr(param_weight, "partition_dim", 0) or 0)
    shape[partition_dim] *= int(getattr(task.mapping, "tp_size", 1) or 1)
    return shape


def _dsv4_gated_shape(task: WeightConversionTask) -> list[int]:
    param_weight = task.param_weight
    if param_weight is None:
        raise RuntimeError(f"Missing DSV4 export param for {task.global_param_name}")
    shape = list(param_weight.shape)
    shape[0] *= int(getattr(task.mapping, "tp_size", 1) or 1)
    if shape[0] % 2 != 0:
        raise ValueError(
            f"Expected even DSV4 gated export dim for {task.global_param_name}: {shape}"
        )
    shape[0] //= 2
    return shape


def _dsv4_expert_down_shape(task: WeightConversionTask) -> list[int]:
    param_weight = task.param_weight
    if param_weight is None:
        raise RuntimeError(f"Missing DSV4 export param for {task.global_param_name}")
    shape = list(param_weight.shape)
    if len(shape) > 1:
        shape[1] *= int(getattr(task.mapping, "tp_size", 1) or 1)
    return shape


def _dsv4_expert_names(
    *,
    task: WeightConversionTask,
    export_name: str,
) -> list[str]:
    config = getattr(task.megatron_module, "config", None)
    num_experts = int(getattr(config, "num_moe_experts", 0) or 0)
    ep_size = int(getattr(task.mapping, "ep_size", 1) or 1)
    if num_experts <= 0 or num_experts % ep_size != 0:
        raise ValueError(
            f"Cannot infer DSV4 expert metadata for {task.global_param_name}: "
            f"num_experts={num_experts}, ep_size={ep_size}."
        )
    experts_per_rank = num_experts // ep_size
    local_expert = (
        extract_expert_number_from_param(task.mapping.megatron_param) % experts_per_rank
    )
    return [
        _set_dsv4_expert_id(export_name, local_expert + experts_per_rank * ep_rank)
        for ep_rank in range(ep_size)
    ]


def _dsv4_quantized_expert_metadata(
    name: str,
    shape: list[int],
) -> list[tuple[str, torch.dtype, list[int]]]:
    if len(shape) != 2 or shape[1] % 32 != 0:
        raise ValueError(f"Expected 2-D K%32 DSV4 expert weight for {name}: {shape}")
    return [
        (name, torch.uint8, [shape[0], shape[1] // 2]),
        (
            f"{name.removesuffix('.weight')}.scale",
            torch.float8_e8m0fnu,
            [shape[0], shape[1] // 32],
        ),
    ]


def _dsv4_modified_metadata(
    pending_fused: dict[str, dict[int, tuple[torch.dtype, list[int]]]],
    name: str,
    dtype: torch.dtype,
    shape: list[int],
) -> list[tuple[str, torch.dtype, list[int]]]:
    fused_key = _dsv4_fused_export_key(name)
    if fused_key is not None:
        target, part_index = fused_key
        parts = pending_fused.setdefault(target, {})
        if part_index in parts:
            raise ValueError(
                f"Duplicate DSV4 fused metadata part {part_index}: {name}."
            )
        parts[part_index] = (dtype, shape)
        if len(parts) < 2:
            return []
        pending_fused.pop(target)
        first_dtype, first_shape = parts[0]
        second_dtype, second_shape = parts[1]
        if first_dtype != second_dtype or first_shape[1:] != second_shape[1:]:
            raise ValueError(
                f"Cannot fuse DSV4 metadata parts for {target}: "
                f"{first_dtype}/{first_shape} vs {second_dtype}/{second_shape}."
            )
        return [
            (target, first_dtype, [first_shape[0] + second_shape[0], *first_shape[1:]])
        ]

    source_expert = _dsv4_canonical_expert_source_name(name)
    source_name = (
        source_expert if source_expert is not None else _dsv4_source_export_name(name)
    )
    if _is_dsv4_routed_expert_weight(source_name):
        return _dsv4_quantized_expert_metadata(source_name, shape)
    if _is_dsv4_hash_router_table(source_name):
        return [(source_name, torch.int32, shape)]
    return [(source_name, dtype, shape)]


def _load_dsv4_hf_tensor(
    hf_param: str, hf_state_dict: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    weight = hf_state_dict[hf_param]
    if not hf_param.endswith(".weight") or weight.dtype not in (
        torch.int8,
        torch.uint8,
        torch.float8_e4m3fn,
    ):
        return weight

    scale_param = f"{hf_param.removesuffix('.weight')}.scale"
    try:
        scale = hf_state_dict[scale_param]
    except KeyError as exc:
        raise ValueError(
            f"Quantized DSV4 weight {hf_param} requires scale {scale_param}."
        ) from exc
    return _dequant_dsv4_weight(weight, scale)


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


class _Dsv4AliasStateSource:
    def __init__(self, source: Any, aliases: Mapping[str, str]):
        self.source = source
        self.aliases = dict(aliases)
        self._all_keys_cache: list[str] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)

    def get_all_keys(self) -> list[str]:
        if self._all_keys_cache is not None:
            return self._all_keys_cache
        keys = set(self.source.get_all_keys())
        keys.update(alias for alias, target in self.aliases.items() if target in keys)
        self._all_keys_cache = sorted(keys)
        return self._all_keys_cache

    def load_tensors(self, keys: list[str]) -> dict[str, torch.Tensor]:
        source_keys = [self.aliases.get(key, key) for key in keys]
        loaded = self.source.load_tensors(source_keys)
        return {key: loaded[self.aliases.get(key, key)] for key in keys}

    def has_glob(self, pattern: str) -> bool:
        import fnmatch

        return any(fnmatch.fnmatch(key, pattern) for key in self.get_all_keys())


def _install_dsv4_source_aliases(hf_pretrained: Any) -> None:
    state = hf_pretrained.state
    source = getattr(state, "source", None)
    if source is None or isinstance(source, _Dsv4AliasStateSource):
        return
    keys = set(source.get_all_keys())
    aliases = {"model.norm.weight": "norm.weight"}
    active_aliases = {
        alias: target
        for alias, target in aliases.items()
        if alias not in keys and target in keys
    }
    if active_aliases:
        state.source = _Dsv4AliasStateSource(source, active_aliases)


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


class _Dsv4RowParallelMapping(RowParallelMapping):
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
        is_grouped_export = False

        def __init__(
            self,
            megatron_param: str,
            gate: str,
            up: str,
            export_gate: str,
            export_up: str,
        ):
            super().__init__(megatron_param, gate, up)
            self.export_hf_param = {
                "gate": export_gate,
                "up": export_up,
            }

        @property
        def group_key(self) -> str:
            return cast(dict[str, str], self.export_hf_param)["gate"]

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
            gate_suffix = hf_param["gate"].rpartition(".experts.")[2].split(".", 1)[1]
            remapped: dict[str, torch.Tensor] = {}
            export = cast(dict[str, str], self.export_hf_param)
            for gate_key, gate in converted.items():
                if not gate_key.endswith(gate_suffix):
                    continue
                expert_id = _dsv4_expert_id(gate_key)
                up_key = _set_dsv4_expert_id(hf_param["up"], expert_id)
                up = converted.get(up_key)
                if up is None:
                    raise ValueError(
                        f"Missing DSV4 gathered expert up weight {up_key} "
                        f"for gate weight {gate_key}."
                    )
                remapped[_set_dsv4_expert_id(export["gate"], expert_id)] = gate
                remapped[_set_dsv4_expert_id(export["up"], expert_id)] = up
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

    class _ArtDsv4ExpertDownMapping(AutoMapping):
        is_grouped_export = False

        def __init__(
            self,
            megatron_param: str,
            hf_param: str,
            export_hf_param: str,
        ):
            super().__init__(megatron_param, hf_param)
            self.weight_hf_param = hf_param
            self.hf_param = {"weight": hf_param}
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
                hf_param = self.hf_param
                self.hf_param = self.weight_hf_param
                try:
                    self._mapping = self._get_or_create_mapping(self._detected_type)
                finally:
                    self.hf_param = hf_param
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
            hf_weight = (
                cast(dict[str, torch.Tensor], hf_weights)["weight"]
                if isinstance(hf_weights, dict)
                else cast(torch.Tensor, hf_weights)
            )
            aligned = _align_expert_weight_to_shape(
                hf_weight,
                torch.Size(full_target_shape),
                "down_proj",
                transpose_hint=False,
            )
            return self._mapping.hf_to_megatron(aligned, megatron_module)

        def megatron_to_hf(self, megatron_weights: Any, megatron_module: Any):
            hf_param = self.hf_param
            self.hf_param = self.weight_hf_param
            try:
                converted = super().megatron_to_hf(megatron_weights, megatron_module)
            finally:
                self.hf_param = hf_param
            if not converted:
                return {}
            return {
                _set_dsv4_expert_id(
                    self.export_hf_param,
                    _dsv4_expert_id(name),
                ): weight
                for name, weight in converted.items()
            }

        def resolve(self, captures: tuple[str, ...]):
            resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
            resolved_hf_param = cast(dict[str, str], resolved_hf_param)
            return type(self)(
                resolved_megatron_param,
                resolved_hf_param["weight"],
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
            "model.layers.*.mlp.experts.*.down_proj.weight",
        ),
        _Dsv4RowParallelMapping(
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
        _Dsv4RowParallelMapping(
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
            "model.layers.*.self_attn.compressor.indexer.scorer.weights_proj.weight",
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
            export_gate="model.layers.*.mlp.experts.*.gate_proj.weight",
            export_up="model.layers.*.mlp.experts.*.up_proj.weight",
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
    def _maybe_collect_fused_export(
        self, name: str, weight: torch.Tensor
    ) -> dict[str, torch.Tensor] | None:
        key = _dsv4_fused_export_key(name)
        if key is None:
            return None
        target, part_index = key
        pending = getattr(self, "_dsv4_fused_export_parts", None)
        if pending is None:
            pending = {}
            self._dsv4_fused_export_parts = pending
        parts = pending.setdefault(target, {})
        if part_index in parts:
            raise ValueError(f"Duplicate DSV4 fused export part {part_index}: {name}.")
        parts[part_index] = weight
        if len(parts) < 2:
            return {}
        pending.pop(target)
        return {target: _concat_dsv4_fused_export_parts(target, parts)}

    def provider_bridge(self, hf_pretrained: Any):
        _install_dsv4_source_aliases(hf_pretrained)
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
        provider.dsv4_compress_ratios = _dsv4_compress_ratios_from_hf_config(hf_config)
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

    def art_extra_hf_prefetch_keys(
        self,
        keys: Iterable[str],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> list[str]:
        all_keys = set(hf_state_dict.keys())
        scale_keys: list[str] = []
        for key in keys:
            if not key.endswith(".weight"):
                continue
            scale_key = f"{key.removesuffix('.weight')}.scale"
            if scale_key in all_keys:
                scale_keys.append(scale_key)
        return scale_keys

    def maybe_modify_loaded_hf_weight(
        self,
        hf_param: str | dict[str, str],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(hf_param, str):
            return _load_dsv4_hf_tensor(hf_param, hf_state_dict)
        return cast(
            torch.Tensor,
            {
                name: _load_dsv4_hf_tensor(param, hf_state_dict)
                for name, param in hf_param.items()
            },
        )

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: dict[str, torch.Tensor],
        hf_state_dict: Any,
    ) -> dict[str, torch.Tensor]:
        del task
        if isinstance(hf_state_dict, dict):
            return converted_weights_dict
        remapped: dict[str, torch.Tensor] = {}
        for name, weight in converted_weights_dict.items():
            fused = self._maybe_collect_fused_export(name, weight)
            if fused is not None:
                remapped.update(fused)
            else:
                source_expert = _dsv4_canonical_expert_source_name(name)
                source_name = (
                    source_expert
                    if source_expert is not None
                    else _dsv4_source_export_name(name)
                )
                if _is_dsv4_routed_expert_weight(source_name):
                    remapped.update(_export_dsv4_mxfp4_weight(source_name, weight))
                elif _is_dsv4_hash_router_table(source_name):
                    remapped[source_name] = weight.to(torch.int32).contiguous()
                else:
                    remapped[source_name] = weight
        return remapped

    def iter_merged_vllm_weight_metadata(
        self,
        weight_export: Any,
    ) -> Iterable[tuple[str, torch.dtype, list[int]]]:
        pending_fused: dict[str, dict[int, tuple[torch.dtype, list[int]]]] = {}
        for task in weight_export.conversion_tasks:
            mapping_name = type(task.mapping).__name__
            dtype = task.param_weight.dtype
            if mapping_name == "_ArtDsv4ExpertGateUpMapping":
                shape = _dsv4_gated_shape(task)
                export = cast(dict[str, str], task.mapping.export_hf_param)
                for gate_name, up_name in zip(
                    _dsv4_expert_names(task=task, export_name=export["gate"]),
                    _dsv4_expert_names(task=task, export_name=export["up"]),
                    strict=True,
                ):
                    yield from _dsv4_modified_metadata(
                        pending_fused, gate_name, dtype, shape
                    )
                    yield from _dsv4_modified_metadata(
                        pending_fused, up_name, dtype, shape
                    )
                continue

            if mapping_name == "_ArtDsv4ExpertDownMapping":
                shape = _dsv4_expert_down_shape(task)
                export_name = cast(str, task.mapping.export_hf_param)
                for name in _dsv4_expert_names(task=task, export_name=export_name):
                    yield from _dsv4_modified_metadata(
                        pending_fused, name, dtype, shape
                    )
                continue

            if isinstance(task.mapping.export_hf_param, dict):
                shape = _dsv4_gated_shape(task)
                export = cast(dict[str, str], task.mapping.export_hf_param)
                yield from _dsv4_modified_metadata(
                    pending_fused, export["gate"], dtype, shape
                )
                yield from _dsv4_modified_metadata(
                    pending_fused, export["up"], dtype, shape
                )
                continue

            yield from _dsv4_modified_metadata(
                pending_fused,
                cast(str, task.mapping.export_hf_param),
                dtype,
                _dsv4_full_parallel_shape(task),
            )
        if pending_fused:
            raise ValueError(
                f"Incomplete DSV4 fused metadata parts: {sorted(pending_fused)}"
            )


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
