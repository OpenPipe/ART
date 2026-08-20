from __future__ import annotations

from functools import lru_cache
import inspect
import json
from pathlib import Path
import re
from typing import Any, Literal, Sequence, cast

import torch

from art.megatron.model_support.handlers.default_dense import (
    DefaultMoeHandler,
    _compile_workaround_flags_for_provider,
)
from art.megatron.model_support.internal_padding import (
    pad_dim_right,
    round_up_to_multiple,
    trim_dim_right,
    zero_lora_padding,
    zero_ranges,
)
from art.megatron.model_support.spec import (
    CompileWorkaroundConfig,
    ExpertPackedLoraGroup,
    ExpertPackedLoraSlot,
    LayerFamilyInstance,
)
from art.megatron.recurrent.contract import (
    HeadShardedFullTreePlannerConfig,
    LinearRecurrentContract,
    ProjectedStreamSpec,
    RecurrentStateSpec,
)

_NANO_PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
_NANO_MIN_PATTERN = _NANO_PATTERN[:13]
_MOE_FFN_ALIGNMENT = 128
_LOGICAL_MOE_FFN_ATTR = "art_nemotron_h_logical_moe_ffn_hidden_size"
_CONVOLUTION_WIDTH = 4
_LOCAL_CHUNK_SIZE = 128
_MAMBA_KERNEL_ID = (
    "mamba_ssm_2_3_2_post1_e9594ce1.chunk_scan_combined.chunk128.conv4.fp32_state.v1"
)
_MAMBA_LAYOUT_KEY = "mamba2.z_x_b_c_dt.head_group.v1"
_NEMOTRON_H_MOE_COMPILE_WORKAROUND_FLAGS = ("te_triton_permute_with_mask_map",)
_HF_SEMANTICS = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "chunk_size": _LOCAL_CHUNK_SIZE,
    "conv_kernel": _CONVOLUTION_WIDTH,
    "expand": 2,
    "hidden_dropout": 0.0,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-5,
    "mamba_hidden_act": "silu",
    "mamba_proj_bias": False,
    "mamba_ssm_cache_dtype": "float32",
    "mlp_bias": False,
    "mlp_hidden_act": "relu2",
    "n_group": 1,
    "n_shared_experts": 1,
    "norm_eps": 1e-5,
    "norm_topk_prob": True,
    "partial_rotary_factor": 1.0,
    "rescale_prenorm_residual": True,
    "residual_in_fp32": False,
    "rope_theta": 10_000,
    "routed_scaling_factor": 2.5,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "time_step_floor": 1e-4,
    "time_step_limit": (0.0, float("inf")),
    "time_step_max": 0.1,
    "time_step_min": 0.001,
    "topk_group": 1,
    "torch_dtype": "bfloat16",
    "use_bias": False,
    "use_cache": True,
    "use_conv_bias": True,
    "use_mamba_kernels": True,
}
_PROVIDER_SEMANTICS = {
    "add_bias_linear": False,
    "add_qkv_bias": False,
    "apply_query_key_layer_scaling": False,
    "attention_dropout": 0.0,
    "attention_softmax_in_fp32": False,
    "bf16": True,
    "fp16": False,
    "fp32_residual_connection": False,
    "hidden_dropout": 0.0,
    "init_method_std": 0.02,
    "is_hybrid_model": True,
    "layernorm_epsilon": 1e-5,
    "moe_aux_loss_coeff": 0.0,
    "moe_expert_capacity_factor": None,
    "moe_grouped_gemm": True,
    "moe_input_jitter_eps": None,
    "moe_pad_expert_input_to_capacity": False,
    "moe_permute_fusion": True,
    "moe_router_bias_update_rate": 0.0,
    "moe_router_dtype": "fp32",
    "moe_router_enable_expert_bias": True,
    "moe_router_force_load_balancing": False,
    "moe_router_group_topk": 1,
    "moe_router_load_balancing_type": "none",
    "moe_router_num_groups": 1,
    "moe_router_pre_softmax": False,
    "moe_router_score_function": "sigmoid",
    "moe_router_topk_scaling_factor": 2.5,
    "moe_z_loss_coeff": None,
    "moe_shared_expert_gate": False,
    "moe_shared_expert_overlap": False,
    "mtp_hybrid_override_pattern": None,
    "normalization": "RMSNorm",
    "params_dtype": torch.bfloat16,
    "position_embedding_type": "none",
    "rotary_base": 10_000,
    "rotary_percent": 1.0,
    "share_embeddings_and_output_weights": False,
    "use_mamba_mem_eff_path": True,
}
_MAMBA_SOURCE_DEFAULTS = {
    "d_conv": _CONVOLUTION_WIDTH,
    "expand": 2,
    "A_init_range": (1, 16),
    "D_has_hdim": False,
    "rmsnorm": True,
    "norm_before_gate": False,
    "dt_min": 0.001,
    "dt_max": 0.1,
    "dt_init": "random",
    "dt_scale": 1.0,
    "dt_init_floor": 1e-4,
    "bias": False,
    "conv_bias": True,
    "chunk_size": _LOCAL_CHUNK_SIZE,
}
_PROVIDER_PROFILES = {
    "production": {
        "hidden_size": 2688,
        "mamba_num_heads": 64,
        "mamba_head_dim": 64,
        "mamba_state_dim": 128,
        "mamba_num_groups": 8,
        "num_attention_heads": 32,
        "num_query_groups": 2,
        "kv_channels": 128,
        "num_moe_experts": 128,
        "moe_ffn_hidden_size": 1856,
        "moe_shared_expert_intermediate_size": 3712,
        "moe_router_topk": 6,
    },
    "compact": {
        "hidden_size": 256,
        "mamba_num_heads": 8,
        "mamba_head_dim": 32,
        "mamba_state_dim": 32,
        "mamba_num_groups": 2,
        "num_attention_heads": 8,
        "num_query_groups": 2,
        "kv_channels": 32,
        "num_moe_experts": 4,
        "moe_ffn_hidden_size": 256,
        "moe_shared_expert_intermediate_size": 512,
        "moe_router_topk": 2,
    },
}
_HF_PROFILE_FIELDS = {
    "hidden_size": ("hidden_size",),
    "mamba_num_heads": ("mamba_num_heads",),
    "mamba_head_dim": ("mamba_head_dim",),
    "mamba_state_dim": ("ssm_state_size", "mamba_state_dim"),
    "mamba_num_groups": ("n_groups", "mamba_n_groups"),
    "num_attention_heads": ("num_attention_heads",),
    "num_query_groups": ("num_key_value_heads",),
    "kv_channels": ("head_dim",),
    "num_moe_experts": ("n_routed_experts", "num_experts"),
    "moe_ffn_hidden_size": ("moe_intermediate_size",),
    "moe_shared_expert_intermediate_size": ("moe_shared_expert_intermediate_size",),
    "moe_router_topk": ("num_experts_per_tok",),
}
_ROUTED_LORA_RE = re.compile(
    r"^(?P<prefix>.*\.mixer\.experts)\.(?P<expert>\d+)\."
    r"(?P<projection>up_proj|down_proj)\.(?P<lora>lora_[AB])\.weight$"
)
_ROUTED_PACKED_LORA_RE = re.compile(
    r"^(?P<prefix>.*\.mixer\.experts)\."
    r"(?P<projection>up_proj|down_proj)\.(?P<lora>lora_[AB])\.weight$"
)
_ROUTED_LORA_PREFIX_RE = re.compile(
    r"^.*\.mixer\.experts\.(?:\{expert\}|\d+)\."
    r"(?P<projection>up_proj|down_proj)$"
)
_LAYER_LORA_NAMESPACE_RE = re.compile(
    r"^base_model\.model\.(?P<namespace>backbone|model)\.layers\."
    r"(?P<layer>\d+)\.(?P<suffix>mixer\..*\.lora_[AB]\.weight)$"
)
_EXPERT_MAPPING_AXES = {
    "decoder.layers.*.mlp.experts.linear_fc1.weight*": 0,
    "decoder.layers.*.mlp.experts.linear_fc2.weight*": 1,
    "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight": 0,
    "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight": 1,
}


def _main_pattern(provider: Any) -> str:
    raw = getattr(provider, "hybrid_layer_pattern", None) or getattr(
        provider, "hybrid_override_pattern", None
    )
    if not isinstance(raw, str) or not raw:
        raise RuntimeError("Nemotron-H provider is missing hybrid_layer_pattern")
    if "/" in raw:
        raise RuntimeError("ART Nemotron-H training does not support MTP patterns")
    pattern = raw.replace("|", "")
    if pattern not in {_NANO_MIN_PATTERN, _NANO_PATTERN}:
        raise RuntimeError(
            "Nemotron-H requires the production hybrid pattern or its complete "
            f"13-layer validation prefix; got {pattern!r}"
        )
    if int(getattr(provider, "num_layers", 0) or 0) != len(pattern):
        raise RuntimeError(
            "Nemotron-H hybrid pattern length does not match num_layers: "
            f"{len(pattern)} != {getattr(provider, 'num_layers', None)}"
        )
    return pattern


def _logical_moe_width(provider: Any) -> int:
    return int(
        getattr(
            provider,
            _LOGICAL_MOE_FFN_ATTR,
            getattr(provider, "moe_ffn_hidden_size", 0),
        )
        or 0
    )


def _profile_for_provider(provider: Any) -> tuple[str, dict[str, int]]:
    hidden_size = int(getattr(provider, "hidden_size", 0) or 0)
    matching = [
        (name, expected)
        for name, expected in _PROVIDER_PROFILES.items()
        if expected["hidden_size"] == hidden_size
    ]
    if len(matching) != 1:
        raise RuntimeError(
            "Unsupported Nemotron-H hidden size "
            f"{hidden_size}; expected one of "
            f"{sorted(profile['hidden_size'] for profile in _PROVIDER_PROFILES.values())}"
        )
    profile_name, expected = matching[0]
    observed = {
        name: (
            _logical_moe_width(provider)
            if name == "moe_ffn_hidden_size"
            else getattr(provider, name, None)
        )
        for name in expected
    }
    mismatches = {
        name: observed[name]
        for name, value in expected.items()
        if observed[name] != value
    }
    if mismatches:
        details = ", ".join(
            f"{name}={value!r} (expected {expected[name]!r})"
            for name, value in mismatches.items()
        )
        raise RuntimeError(f"Unsupported Nemotron-H {profile_name} geometry: {details}")
    pattern = _main_pattern(provider)
    if profile_name == "compact" and pattern != _NANO_MIN_PATTERN:
        raise RuntimeError("Compact Nemotron-H requires the exact 13-layer pattern")
    return profile_name, expected


def _require_nano_geometry(provider: Any) -> tuple[str, dict[str, int]]:
    profile = _profile_for_provider(provider)
    if getattr(provider, "gated_linear_unit", None) is not False:
        raise RuntimeError("Nemotron-H experts must use non-gated up/down projections")
    activation = getattr(provider, "activation_func", None)
    probe = torch.tensor((-2.0, -0.5, 0.0, 2.0))
    if not callable(activation) or not torch.equal(
        activation(probe), torch.nn.functional.relu(probe).square()
    ):
        raise RuntimeError("Nemotron-H experts require squared-ReLU activation")
    if int(getattr(provider, "mtp_num_layers", 0) or 0):
        raise RuntimeError("ART Nemotron-H training does not support MTP")
    return profile


def _configure_provider_semantics(provider: Any) -> None:
    for name, value in _PROVIDER_SEMANTICS.items():
        setattr(provider, name, value)


def _validate_no_virtual_pipeline(provider: Any) -> None:
    vpp = getattr(provider, "virtual_pipeline_model_parallel_size", None)
    if vpp is not None:
        raise RuntimeError(
            "Nemotron-H does not support virtual pipeline model parallelism; "
            f"virtual_pipeline_model_parallel_size={vpp!r}"
        )


def _validate_mamba_source_defaults() -> None:
    from megatron.core.ssm.mamba_mixer import MambaMixer

    parameters = inspect.signature(MambaMixer.__init__).parameters
    changed = {
        name: parameters[name].default if name in parameters else "<missing>"
        for name, expected in _MAMBA_SOURCE_DEFAULTS.items()
        if name not in parameters or parameters[name].default != expected
    }
    if changed:
        raise RuntimeError(
            "Pinned MCore MambaMixer defaults changed; Nemotron-H cannot preserve "
            f"its convolution/timestep semantics: {changed}"
        )


def _validate_provider_semantics(provider: Any) -> None:
    _require_nano_geometry(provider)
    invalid = {
        name: getattr(provider, name, "<missing>")
        for name, expected in _PROVIDER_SEMANTICS.items()
        if getattr(provider, name, "<missing>") != expected
    }
    if invalid:
        raise RuntimeError(f"Unsupported Nemotron-H provider semantics: {invalid}")
    _validate_mamba_source_defaults()


def _configure_internal_padding(provider: Any, expected: dict[str, int]) -> None:
    logical = _logical_moe_width(provider)
    internal = round_up_to_multiple(logical, _MOE_FFN_ALIGNMENT)
    if logical != expected["moe_ffn_hidden_size"]:
        raise RuntimeError(f"Unexpected Nemotron-H logical expert width {logical}")
    setattr(provider, _LOGICAL_MOE_FFN_ATTR, logical)
    provider.moe_ffn_hidden_size = internal


def _padding_sizes_from_provider(provider: Any) -> tuple[int, int]:
    _, expected = _require_nano_geometry(provider)
    logical = _logical_moe_width(provider)
    internal = int(getattr(provider, "moe_ffn_hidden_size", 0) or 0)
    expected_internal = round_up_to_multiple(logical, _MOE_FFN_ALIGNMENT)
    if (logical, internal) != (
        expected["moe_ffn_hidden_size"],
        expected_internal,
    ):
        raise RuntimeError(
            "Invalid Nemotron-H routed expert padding: "
            f"{logical}->{internal}; expected "
            f"{expected['moe_ffn_hidden_size']}->{expected_internal}"
        )
    return logical, internal


def _model_config(model_chunks: Sequence[Any]) -> Any | None:
    for chunk in model_chunks:
        config = getattr(chunk, "config", None)
        if config is None:
            config = getattr(getattr(chunk, "module", None), "config", None)
        if config is not None:
            return config
    return None


def _padding_sizes_from_model_chunks(
    model_chunks: Sequence[Any],
) -> tuple[int, int] | None:
    config = _model_config(model_chunks)
    return _padding_sizes_from_provider(config) if config is not None else None


def _config_value(config: Any, *names: str) -> Any:
    values = []
    for name in names:
        if isinstance(config, dict):
            if name in config:
                value = config[name]
                if value is not None or len(names) == 1:
                    values.append(value)
        elif hasattr(config, name):
            value = getattr(config, name)
            if value is not None or len(names) == 1:
                values.append(value)
    if not values:
        raise RuntimeError(f"Nemotron-H config is missing required field {names[0]!r}")
    if any(value != values[0] for value in values[1:]):
        raise RuntimeError(
            f"Conflicting Nemotron-H config aliases {dict(zip(names, values))}"
        )
    return values[0]


def _normalized_semantic_value(name: str, value: Any) -> Any:
    if name == "time_step_limit" and isinstance(value, (list, tuple)):
        return tuple(value)
    if name == "torch_dtype":
        return str(value).removeprefix("torch.")
    return value


def _profile_for_hf_config(config: Any | None) -> tuple[str, dict[str, int]]:
    if config is None or _config_value(config, "model_type") != "nemotron_h":
        raise RuntimeError("Nemotron-H conversion requires model_type='nemotron_h'")
    hidden_size = int(_config_value(config, "hidden_size") or 0)
    matching = [
        (name, expected)
        for name, expected in _PROVIDER_PROFILES.items()
        if expected["hidden_size"] == hidden_size
    ]
    if len(matching) != 1:
        raise RuntimeError(
            f"Unsupported Nemotron-H checkpoint hidden size {hidden_size}"
        )
    profile_name, expected = matching[0]
    observed = {
        name: _config_value(config, *_HF_PROFILE_FIELDS[name]) for name in expected
    }
    mismatches = {
        name: observed[name]
        for name, value in expected.items()
        if observed[name] != value
    }
    if mismatches:
        details = ", ".join(
            f"{name}={value!r} (expected {expected[name]!r})"
            for name, value in mismatches.items()
        )
        raise RuntimeError(
            f"Unsupported Nemotron-H {profile_name} checkpoint geometry: {details}"
        )
    pattern = _config_value(config, "hybrid_override_pattern", "hybrid_layer_pattern")
    depth = int(_config_value(config, "num_hidden_layers", "num_layers") or 0)
    allowed_patterns = (
        {(_NANO_PATTERN, len(_NANO_PATTERN)), (_NANO_MIN_PATTERN, 13)}
        if profile_name == "production"
        else {(_NANO_MIN_PATTERN, 13)}
    )
    if (pattern, depth) not in allowed_patterns:
        raise RuntimeError(
            f"Unsupported Nemotron-H {profile_name} checkpoint pattern/depth: "
            f"{pattern!r}/{depth}"
        )
    invalid = {}
    for name, expected_value in _HF_SEMANTICS.items():
        observed = _normalized_semantic_value(name, _config_value(config, name))
        if observed != expected_value:
            invalid[name] = observed
    if invalid:
        raise RuntimeError(f"Unsupported Nemotron-H checkpoint semantics: {invalid}")
    return profile_name, expected


def _padding_sizes_from_hf_config(config: Any | None) -> tuple[int, int]:
    _, expected = _profile_for_hf_config(config)
    logical = expected["moe_ffn_hidden_size"]
    return logical, round_up_to_multiple(logical, _MOE_FFN_ALIGNMENT)


@lru_cache(maxsize=4)
def _config_dict(base_model: str) -> dict[str, Any]:
    path = Path(base_model) / "config.json"
    if not path.exists():
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(base_model, "config.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def _padding_sizes_from_adapter_config(
    adapter_config: dict[str, Any],
) -> tuple[int, int, int, int]:
    base_model = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model:
        raise RuntimeError(
            "Nemotron-H LoRA conversion requires base_model_name_or_path"
        )
    _, expected = _profile_for_hf_config(_config_dict(base_model))
    logical = expected["moe_ffn_hidden_size"]
    return (
        logical,
        round_up_to_multiple(logical, _MOE_FFN_ALIGNMENT),
        expected["num_moe_experts"],
        expected["hidden_size"],
    )


def _normalize_target_modules(raw_targets: Any, *, field: str) -> list[str]:
    targets: list[str]
    if isinstance(raw_targets, str):
        targets = [raw_targets]
    elif raw_targets is None:
        targets = []
    elif isinstance(raw_targets, (list, tuple, set)) and all(
        isinstance(target, str) for target in raw_targets
    ):
        targets = (
            sorted(raw_targets) if isinstance(raw_targets, set) else list(raw_targets)
        )
    else:
        raise RuntimeError(f"Nemotron-H {field} must be a string or string sequence")
    if len(targets) != len(set(targets)):
        raise RuntimeError(f"Nemotron-H {field} must not contain duplicates")
    return targets


def _serving_target_modules(logical_targets: list[str]) -> list[str]:
    targets = list(logical_targets)
    if {"experts", "up_proj", "down_proj"}.intersection(targets):
        if "experts" not in targets:
            targets.append("experts")
    return targets


def _adapter_target_modules(adapter_config: dict[str, Any]) -> list[str]:
    targets = _normalize_target_modules(
        adapter_config.get("target_modules"), field="target_modules"
    )
    target_set = set(targets)
    if "experts" in target_set and (
        ("up_proj" in target_set) != ("down_proj" in target_set)
    ):
        raise RuntimeError(
            "Nemotron-H persisted target_modules cannot mix experts with only one "
            "routed projection; keep the logical subset in adapter_config and add "
            "experts only to the ephemeral vLLM config"
        )
    return targets


def _persisted_lora_config(adapter_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(adapter_config)
    targets = _adapter_target_modules(config)
    if config.get("target_modules") is not None or targets:
        config["target_modules"] = targets
    return config


def _vllm_lora_config(
    adapter_config: dict[str, Any], *, has_routed_tensors: bool = False
) -> dict[str, Any]:
    config = _persisted_lora_config(adapter_config)
    logical_targets = _adapter_target_modules(config)
    has_routed_targets = bool(
        {"experts", "up_proj", "down_proj"}.intersection(logical_targets)
    )
    targets = (
        _serving_target_modules(logical_targets)
        if has_routed_tensors or has_routed_targets
        else list(logical_targets)
    )
    if config.get("target_modules") is not None or targets:
        config["target_modules"] = targets
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

    class _PaddedExpertMapping(AutoMapping):
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
                    f"{self.hf_param}: expected logical expert dim {logical} on "
                    f"axis {self.axis}, got {tuple(hf_weights.shape)}"
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
    found: set[str] = set()
    for mapping in upstream.mappings:
        name = str(getattr(mapping, "megatron_param", ""))
        axis = _EXPERT_MAPPING_AXES.get(name)
        if axis is None:
            mappings.append(mapping)
            continue
        hf_param = getattr(mapping, "hf_param", None)
        if not isinstance(hf_param, str):
            raise TypeError(f"Nemotron-H expert mapping {name} has non-string HF key")
        mappings.append(
            _PaddedExpertMapping(
                name,
                hf_param,
                axis,
                getattr(mapping, "permute_dims", None),
            )
        )
        found.add(name)
    if found != set(_EXPERT_MAPPING_AXES):
        raise RuntimeError(
            "Nemotron-H Bridge expert mappings changed: "
            f"found={sorted(found)} expected={sorted(_EXPERT_MAPPING_AXES)}"
        )
    return MegatronMappingRegistry(*mappings)


def _patch_mapping_registry(model_bridge: Any) -> None:
    bridge_type = type(model_bridge)
    original = getattr(bridge_type, "mapping_registry", None)
    if original is None:
        raise TypeError(f"{bridge_type.__name__} has no mapping_registry")
    if getattr(original, "_art_nemotron_h_padding_patch", False):
        return

    def mapping_registry(self: Any) -> Any:
        config = _model_bridge_hf_config(self)
        logical, internal = _padding_sizes_from_hf_config(config)
        return _padded_mapping_registry(
            original(self), logical=logical, internal=internal
        )

    setattr(mapping_registry, "_art_nemotron_h_padding_patch", True)
    bridge_type.mapping_registry = mapping_registry


def _patch_config_export(model_bridge: Any) -> None:
    bridge_type = type(model_bridge)
    current = bridge_type.__dict__.get("megatron_to_hf_config")
    function = current.__func__ if isinstance(current, classmethod) else current
    if getattr(function, "_art_nemotron_h_padding_patch", False):
        return
    original = getattr(bridge_type, "megatron_to_hf_config", None)
    if not callable(original):
        raise TypeError(f"{bridge_type.__name__} has no megatron_to_hf_config")

    def megatron_to_hf_config(cls: type[Any], provider: Any) -> dict[str, Any]:
        del cls
        _validate_provider_semantics(provider)
        logical, internal = _padding_sizes_from_provider(provider)
        if int(provider.moe_ffn_hidden_size) != internal:
            raise RuntimeError(
                "Nemotron-H Bridge export requires padded provider width"
            )
        config = dict(original(provider))
        config.update(_HF_SEMANTICS)
        config["moe_intermediate_size"] = logical
        config["n_shared_experts"] = 1
        return config

    setattr(megatron_to_hf_config, "_art_nemotron_h_padding_patch", True)
    bridge_type.megatron_to_hf_config = classmethod(megatron_to_hf_config)


def _model_bridge_hf_config(model_bridge: Any) -> Any | None:
    config = getattr(model_bridge, "hf_config", None)
    if config is None:
        config = getattr(getattr(model_bridge, "hf_pretrained", None), "config", None)
    return config


def _routed_lora_tensor(
    key: str,
    tensor: torch.Tensor,
    *,
    logical: int,
    internal: int,
    to_vllm: bool,
) -> torch.Tensor:
    match = _ROUTED_LORA_RE.match(key)
    if match is None:
        return tensor.contiguous()
    projection, lora = match.group("projection", "lora")
    axis = 0 if (projection, lora) == ("up_proj", "lora_B") else -1
    if (projection, lora) not in {
        ("up_proj", "lora_B"),
        ("down_proj", "lora_A"),
    }:
        return tensor.contiguous()
    observed = int(tensor.shape[axis])
    expected = {logical, internal} if to_vllm else {logical}
    if observed not in expected:
        raise RuntimeError(
            f"{key}: expected routed LoRA dim {sorted(expected)} on axis {axis}, "
            f"got {tuple(tensor.shape)}"
        )
    if to_vllm:
        return (
            trim_dim_right(tensor, dim=axis, size=logical)
            if observed == internal
            else tensor.contiguous()
        )
    return pad_dim_right(tensor, dim=axis, size=internal)


def _canonicalize_layer_lora_namespace(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    namespaces: set[str] = set()
    matches: dict[str, re.Match[str]] = {}
    malformed: list[str] = []
    for key in tensors:
        match = _LAYER_LORA_NAMESPACE_RE.match(key)
        if match is not None:
            namespaces.add(match.group("namespace"))
            matches[key] = match
        elif ".lora_" in key and (".layers." in key or ".mixer." in key):
            malformed.append(key)
    if malformed:
        raise RuntimeError(
            f"Malformed Nemotron-H layer LoRA namespaces: {sorted(malformed)}"
        )
    if len(namespaces) > 1:
        raise RuntimeError(
            "Nemotron-H LoRA cannot mix backbone.layers and model.layers namespaces"
        )

    canonical: dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        match = matches.get(key)
        output_key = (
            "base_model.model.model.layers."
            f"{match.group('layer')}.{match.group('suffix')}"
            if match is not None
            else key
        )
        if output_key in canonical:
            raise RuntimeError(
                f"Duplicate Nemotron-H LoRA key after namespace mapping: {output_key}"
            )
        canonical[output_key] = value
    return canonical


def _routed_packed_lora_tensor(
    key: str,
    tensor: torch.Tensor,
    *,
    logical: int,
    internal: int,
    num_experts: int,
    to_vllm: bool,
) -> torch.Tensor:
    match = _ROUTED_PACKED_LORA_RE.match(key)
    if match is None:
        return tensor.contiguous()
    projection, lora = match.group("projection", "lora")
    if tensor.ndim != 2 or tensor.shape[0] % num_experts:
        raise RuntimeError(
            f"{key}: expected 2D expert-row tensor divisible by {num_experts}, "
            f"got {tuple(tensor.shape)}"
        )
    expected = internal if to_vllm else logical
    if (projection, lora) == ("up_proj", "lora_B"):
        if int(tensor.shape[0]) != num_experts * expected:
            raise RuntimeError(
                f"{key}: expected {num_experts} expert blocks of width {expected}, "
                f"got {tuple(tensor.shape)}"
            )
        blocks = tensor.reshape(num_experts, expected, tensor.shape[1])
        converted = (
            trim_dim_right(blocks, dim=1, size=logical)
            if to_vllm
            else pad_dim_right(blocks, dim=1, size=internal)
        )
        output_size = logical if to_vllm else internal
        return converted.reshape(
            num_experts * output_size, tensor.shape[1]
        ).contiguous()
    if (projection, lora) == ("down_proj", "lora_A"):
        if int(tensor.shape[-1]) != expected:
            raise RuntimeError(
                f"{key}: expected routed LoRA dim {expected} on axis -1, "
                f"got {tuple(tensor.shape)}"
            )
        return (
            trim_dim_right(tensor, dim=-1, size=logical)
            if to_vllm
            else pad_dim_right(tensor, dim=-1, size=internal)
        )
    return tensor.contiguous()


def _convert_lora_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    adapter_config: dict[str, Any],
    to_vllm: bool,
) -> dict[str, torch.Tensor]:
    tensors = _canonicalize_layer_lora_namespace(tensors)
    logical, internal, num_experts, hidden_size = _padding_sizes_from_adapter_config(
        adapter_config
    )
    numeric = {key for key in tensors if _ROUTED_LORA_RE.match(key)}
    packed = {key for key in tensors if _ROUTED_PACKED_LORA_RE.match(key)}
    malformed = {
        key
        for key in tensors
        if ".mixer.experts" in key
        and ".lora_" in key
        and key not in numeric
        and key not in packed
    }
    if malformed:
        raise RuntimeError(f"Invalid Nemotron-H routed LoRA keys: {sorted(malformed)}")
    if numeric and packed:
        raise RuntimeError(
            "Nemotron-H LoRA contains both packed and per-expert routed blocks"
        )
    if packed and not to_vllm:
        raise RuntimeError(
            "vLLM Nemotron-H LoRA must use numeric per-expert routed keys"
        )
    rank_value = adapter_config.get("r")
    if numeric or packed:
        if (
            isinstance(rank_value, bool)
            or not isinstance(rank_value, int)
            or rank_value <= 0
        ):
            raise RuntimeError(
                "Nemotron-H routed LoRA conversion requires positive integer r"
            )
        rank = rank_value
    else:
        rank = 0

    targets = set(_adapter_target_modules(adapter_config))
    logical_projections = (
        {"up_proj", "down_proj"}
        if "experts" in targets
        else {"up_proj", "down_proj"}.intersection(targets)
    )
    if (numeric or packed) and not logical_projections:
        raise RuntimeError(
            "Nemotron-H routed LoRA tensors require experts, up_proj, or "
            "down_proj in target_modules"
        )
    serving_projections = {"up_proj", "down_proj"}

    numeric_blocks: dict[str, dict[int, dict[str, set[str]]]] = {}
    for key in numeric:
        if match := _ROUTED_LORA_RE.match(key):
            expert = int(match.group("expert"))
            if expert not in range(num_experts):
                raise RuntimeError(
                    f"{key}: expert {expert} is outside [0, {num_experts})"
                )
            numeric_blocks.setdefault(match.group("prefix"), {}).setdefault(
                expert, {}
            ).setdefault(match.group("projection"), set()).add(match.group("lora"))
    for prefix, experts in numeric_blocks.items():
        if set(experts) != set(range(num_experts)):
            raise RuntimeError(
                f"Incomplete Nemotron-H routed expert coverage for {prefix}: "
                f"found={sorted(experts)} expected=0..{num_experts - 1}"
            )
        projection_sets = {frozenset(projections) for projections in experts.values()}
        allowed_projection_sets = {frozenset(logical_projections)}
        if not to_vllm:
            allowed_projection_sets.add(frozenset(serving_projections))
        if len(projection_sets) != 1 or not projection_sets.issubset(
            allowed_projection_sets
        ):
            raise RuntimeError(
                "Nemotron-H routed LoRA projections do not match target_modules "
                f"for {prefix}: present={sorted(map(sorted, projection_sets))} "
                f"expected={sorted(map(sorted, allowed_projection_sets))}"
            )
        for expert, projections in experts.items():
            incomplete = {
                projection
                for projection, names in projections.items()
                if names != {"lora_A", "lora_B"}
            }
            if incomplete:
                raise RuntimeError(
                    "Incomplete Nemotron-H routed LoRA projection pairs for "
                    f"{prefix}.{expert}: {sorted(incomplete)}"
                )
    routed_widths = {logical, internal} if to_vllm else {logical}
    for key in numeric:
        match = _ROUTED_LORA_RE.match(key)
        assert match is not None
        projection, lora = match.group("projection", "lora")
        value = tensors[key]
        valid = value.ndim == 2 and (
            tuple(value.shape) == (rank, hidden_size)
            if (projection, lora) == ("up_proj", "lora_A")
            else tuple(value.shape) == (hidden_size, rank)
            if (projection, lora) == ("down_proj", "lora_B")
            else value.shape[0] in routed_widths and value.shape[1] == rank
            if (projection, lora) == ("up_proj", "lora_B")
            else value.shape[0] == rank and value.shape[1] in routed_widths
        )
        if not valid:
            raise RuntimeError(
                f"Invalid Nemotron-H routed LoRA shape for {key}: {tuple(value.shape)}"
            )

    converted: dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        converted[key] = _routed_lora_tensor(
            key,
            value,
            logical=logical,
            internal=internal,
            to_vllm=to_vllm,
        )
    if numeric and to_vllm:
        for prefix, experts in numeric_blocks.items():
            for expert, projections in experts.items():
                for projection in {"up_proj", "down_proj"} - set(projections):
                    input_size, output_size = (
                        (hidden_size, logical)
                        if projection == "up_proj"
                        else (logical, hidden_size)
                    )
                    for lora, shape in (
                        ("lora_A", (rank, input_size)),
                        ("lora_B", (output_size, rank)),
                    ):
                        source = next(
                            converted[
                                f"{prefix}.{expert}.{present_projection}.{lora}.weight"
                            ]
                            for present_projection in projections
                        )
                        converted[f"{prefix}.{expert}.{projection}.{lora}.weight"] = (
                            source.new_zeros(shape)
                        )
    if numeric and not to_vllm:
        for key in tuple(converted):
            match = _ROUTED_LORA_RE.match(key)
            if match is None or match.group("projection") in logical_projections:
                continue
            if torch.count_nonzero(converted[key]).item():
                raise RuntimeError(
                    f"Nemotron-H serving-only routed LoRA tensor is nonzero: {key}"
                )
            del converted[key]
    if not packed:
        return converted

    grouped: dict[str, dict[tuple[str, str], torch.Tensor]] = {}
    for key in packed:
        match = _ROUTED_PACKED_LORA_RE.match(key)
        assert match is not None
        grouped.setdefault(match.group("prefix"), {})[
            (match.group("projection"), match.group("lora"))
        ] = _routed_packed_lora_tensor(
            key,
            tensors[key],
            logical=logical,
            internal=internal,
            num_experts=num_experts,
            to_vllm=True,
        )

    expanded: dict[str, torch.Tensor] = {}
    for prefix, slots in grouped.items():
        present_projections = {projection for projection, _ in slots}
        incomplete_projections = {
            projection
            for projection in present_projections
            if {
                lora for slot_projection, lora in slots if slot_projection == projection
            }
            != {"lora_A", "lora_B"}
        }
        if incomplete_projections:
            raise RuntimeError(
                "Incomplete packed Nemotron-H routed LoRA projection pairs for "
                f"{prefix}: {sorted(incomplete_projections)}"
            )
        if present_projections != logical_projections:
            raise RuntimeError(
                "Packed Nemotron-H routed LoRA projections do not match "
                f"target_modules for {prefix}: present={sorted(present_projections)} "
                f"expected={sorted(logical_projections)}"
            )
        expected_shapes = {
            ("up_proj", "lora_A"): (num_experts * rank, hidden_size),
            ("up_proj", "lora_B"): (num_experts * logical, rank),
            ("down_proj", "lora_A"): (num_experts * rank, logical),
            ("down_proj", "lora_B"): (num_experts * hidden_size, rank),
        }
        invalid_shapes = {
            slot: tuple(slots[slot].shape)
            for slot, shape in expected_shapes.items()
            if slot in slots
            if tuple(slots[slot].shape) != shape
        }
        if invalid_shapes:
            raise RuntimeError(
                f"Invalid packed Nemotron-H routed LoRA shapes for {prefix}: "
                f"{invalid_shapes}"
            )
        shaped = {
            (projection, lora): value.reshape(
                num_experts,
                rank if lora == "lora_A" else value.shape[0] // num_experts,
                value.shape[1] if lora == "lora_A" else rank,
            )
            for (projection, lora), value in slots.items()
        }
        for projection in {"up_proj", "down_proj"} - present_projections:
            source_a = next(
                value for (_, lora), value in shaped.items() if lora == "lora_A"
            )
            source_b = next(
                value for (_, lora), value in shaped.items() if lora == "lora_B"
            )
            input_size, output_size = (
                (hidden_size, logical)
                if projection == "up_proj"
                else (logical, hidden_size)
            )
            shaped[(projection, "lora_A")] = source_a.new_zeros(
                num_experts, rank, input_size
            )
            shaped[(projection, "lora_B")] = source_b.new_zeros(
                num_experts, output_size, rank
            )
        for (projection, lora), value in shaped.items():
            for expert in range(num_experts):
                key = f"{prefix}.{expert}.{projection}.{lora}.weight"
                if key in tensors or key in expanded:
                    raise RuntimeError(
                        f"Duplicate expanded Nemotron-H routed LoRA key: {key}"
                    )
                expanded[key] = value[expert].clone().contiguous()
    return {
        **{key: value for key, value in converted.items() if key not in packed},
        **expanded,
    }


def _zero_routed_lora_padding(
    model_chunks: Sequence[Any], *, grads: bool, params: bool
) -> None:
    sizes = _padding_sizes_from_model_chunks(model_chunks)
    if sizes is None:
        return
    logical, internal = sizes
    with torch.no_grad():
        for chunk in model_chunks:
            for module in chunk.modules():
                prefix = getattr(module, "adapter_model_prefix", None)
                match = (
                    _ROUTED_LORA_PREFIX_RE.match(prefix)
                    if isinstance(prefix, str)
                    else None
                )
                if match is None:
                    continue
                projection = match.group("projection")
                param_name, dim = (
                    ("B_T", -1) if projection == "up_proj" else ("A_T", -2)
                )
                param = getattr(module, param_name, None)
                if isinstance(param, torch.nn.Parameter):
                    zero_lora_padding(
                        param,
                        dim=dim,
                        logical=logical,
                        internal=internal,
                        components=(internal,),
                        grads=grads,
                        params=params,
                    )


def _local_expert_padding(
    *, logical: int, internal: int, world_size: int, rank: int
) -> tuple[int, int, int]:
    if world_size <= 0 or rank not in range(world_size) or internal % world_size:
        raise RuntimeError(
            "Invalid Nemotron-H expert-TP padding topology: "
            f"width={internal}, world_size={world_size}, rank={rank}"
        )
    local_size = internal // world_size
    shard_start = rank * local_size
    global_start = max(logical, shard_start)
    global_end = min(internal, shard_start + local_size)
    if global_end <= global_start:
        return local_size, 0, 0
    return local_size, global_start - shard_start, global_end - shard_start


def _zero_base_parameter_padding(
    param: torch.nn.Parameter,
    *,
    dim: int,
    local_size: int,
    start: int,
    end: int,
    grads: bool,
    params: bool,
) -> None:
    dim = dim if dim >= 0 else param.ndim + dim
    if param.ndim != 2 or int(param.shape[dim]) != local_size:
        raise RuntimeError(
            "Nemotron-H grouped expert weight has an unexpected local shape: "
            f"shape={tuple(param.shape)}, dim={dim}, expected={local_size}"
        )
    tensors: list[torch.Tensor] = [param.data] if params else []
    if grads:
        for value in (param.grad, getattr(param, "main_grad", None)):
            local = getattr(value, "_local_tensor", None)
            tensor = value if torch.is_tensor(value) else local
            if torch.is_tensor(tensor):
                tensors.append(tensor)
    for tensor in tensors:
        if tuple(tensor.shape) != tuple(param.shape):
            raise RuntimeError(
                "Nemotron-H grouped expert gradient shape changed: "
                f"{tuple(tensor.shape)} != {tuple(param.shape)}"
            )
        if end > start:
            tensor.narrow(dim, start, end - start).zero_()


def _zero_routed_base_padding(
    model_chunks: Sequence[Any], *, grads: bool, params: bool
) -> None:
    config = _model_config(model_chunks)
    if config is None:
        return
    logical, internal = _padding_sizes_from_provider(config)
    if logical == internal:
        return

    from megatron.core.transformer.moe.experts import TEGroupedMLP

    from art.megatron import lora

    configured_world_size = int(getattr(config, "expert_tensor_parallel_size", 1) or 1)
    world_size = lora._get_shard_world_size("expert_tp")
    rank = lora._get_shard_rank("expert_tp")
    if world_size != configured_world_size:
        raise RuntimeError(
            "Nemotron-H expert-TP runtime/config mismatch: "
            f"runtime={world_size}, configured={configured_world_size}"
        )
    local_size, start, end = _local_expert_padding(
        logical=logical,
        internal=internal,
        world_size=world_size,
        rank=rank,
    )
    for chunk in model_chunks:
        for module in chunk.modules():
            if not isinstance(module, TEGroupedMLP):
                continue
            linear_fc1 = module.linear_fc1
            if isinstance(linear_fc1, lora.MLPExpertsLinearFC1LoRA):
                if (
                    not linear_fc1.non_gated
                    or _ROUTED_LORA_PREFIX_RE.match(
                        linear_fc1.up_lora.adapter_model_prefix
                    )
                    is None
                ):
                    raise RuntimeError("Invalid Nemotron-H routed FC1 LoRA wrapper")
                linear_fc1 = linear_fc1.linear_fc1
            linear_fc2 = module.linear_fc2
            if isinstance(linear_fc2, lora.MLPExpertsLinearFC2LoRA):
                if (
                    _ROUTED_LORA_PREFIX_RE.match(linear_fc2.lora.adapter_model_prefix)
                    is None
                ):
                    raise RuntimeError("Invalid Nemotron-H routed FC2 LoRA wrapper")
                linear_fc2 = linear_fc2.linear_fc2
            for linear_name, linear, dim in (
                ("linear_fc1", linear_fc1, 0),
                ("linear_fc2", linear_fc2, 1),
            ):
                for expert in range(module.num_local_experts):
                    param = getattr(linear, f"weight{expert}", None)
                    if not isinstance(param, torch.nn.Parameter):
                        raise RuntimeError(
                            "Nemotron-H grouped expert base weight is missing: "
                            f"{linear_name}.weight{expert}"
                        )
                    _zero_base_parameter_padding(
                        param,
                        dim=dim,
                        local_size=local_size,
                        start=start,
                        end=end,
                        grads=grads,
                        params=params,
                    )


def _zero_loaded_padding(
    state: dict[str, Any], model_chunks: Sequence[Any]
) -> dict[str, Any]:
    sizes = _padding_sizes_from_model_chunks(model_chunks)
    if sizes is None:
        return state
    logical, internal = sizes
    canonical: dict[str, Any] = {}
    for key, value in state.items():
        if not torch.is_tensor(value) or (match := _ROUTED_LORA_RE.match(key)) is None:
            canonical[key] = value
            continue
        projection, lora = match.group("projection", "lora")
        result = value.clone().contiguous()
        if (projection, lora) == ("up_proj", "lora_B"):
            if int(result.shape[0]) != internal:
                raise RuntimeError(f"{key}: expected padded up LoRA-B width {internal}")
            zero_ranges(result, dim=0, ranges=((logical, internal),))
        elif (projection, lora) == ("down_proj", "lora_A"):
            if int(result.shape[-1]) != internal:
                raise RuntimeError(
                    f"{key}: expected padded down LoRA-A width {internal}"
                )
            zero_ranges(result, dim=-1, ranges=((logical, internal),))
        canonical[key] = result
    return canonical


class NemotronHMoeHandler(DefaultMoeHandler):
    key = "nemotron_h_moe"
    is_moe = True
    cp_supported = True
    native_vllm_lora_status = "wip"

    def identity_lora_model_config(self, base_config: Any) -> Any:
        return base_config

    def identity_lora_target_parameters(
        self,
        model: Any,
        *,
        target_modules: list[str],
    ) -> list[str]:
        names = super().identity_lora_target_parameters(
            model, target_modules=target_modules
        )
        if "experts" in target_modules:
            names.extend(
                name
                for name, _ in model.named_parameters()
                if ".mixer.experts." in name
                and name.endswith(("up_proj.weight", "down_proj.weight"))
            )
        return list(dict.fromkeys(names))

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
        if model_bridge is None or type(model_bridge).__name__ != "NemotronHBridge":
            raise TypeError(
                "Nemotron-H support requires the native Megatron Bridge "
                f"NemotronHBridge, got {type(model_bridge).__name__}"
            )
        _padding_sizes_from_hf_config(_model_bridge_hf_config(model_bridge))
        _patch_mapping_registry(model_bridge)
        _patch_config_export(model_bridge)

    def configure_provider_for_runtime(self, provider: Any) -> None:
        if type(provider).__name__ != "MambaModelProvider":
            raise TypeError(
                "Nemotron-H support requires MambaModelProvider, got "
                f"{type(provider).__name__}"
            )
        _validate_no_virtual_pipeline(provider)
        _, expected = _require_nano_geometry(provider)
        _configure_internal_padding(provider, expected)
        _configure_provider_semantics(provider)

    def validate_provider_for_runtime(self, provider: Any) -> None:
        _validate_no_virtual_pipeline(provider)
        _validate_provider_semantics(provider)
        _padding_sizes_from_provider(provider)

    def compile_workaround_config(
        self,
        provider: Any,
    ) -> CompileWorkaroundConfig:
        return CompileWorkaroundConfig(
            flags=_compile_workaround_flags_for_provider(
                provider,
                _NEMOTRON_H_MOE_COMPILE_WORKAROUND_FLAGS,
            ),
            shared_expert_state=self._shared_expert_compile_state(provider),
        )

    def linear_recurrent_contract(
        self, provider: Any
    ) -> LinearRecurrentContract | None:
        self.validate_provider_for_runtime(provider)
        tp = int(getattr(provider, "tensor_model_parallel_size", 1) or 1)
        cp = int(getattr(provider, "context_parallel_size", 1) or 1)
        if tp not in {1, 2, 4, 8}:
            raise RuntimeError(f"Nemotron-H requires TP in {{1,2,4,8}}, got {tp}")
        partitions = tp * cp
        heads = int(provider.mamba_num_heads)
        groups = int(provider.mamba_num_groups)
        if heads % partitions or groups % partitions:
            raise RuntimeError(
                "Nemotron-H head-sharded CP requires mamba heads and groups "
                f"divisible by TP*CP={partitions}; heads={heads}, groups={groups}"
            )
        head_dim = int(provider.mamba_head_dim)
        state_dim = int(provider.mamba_state_dim)
        replication_factor = heads // groups
        heads_local_tp = heads // tp
        groups_local_tp = groups // tp
        local_heads = heads // partitions
        local_groups = groups // partitions
        dtype = str(getattr(provider, "params_dtype", torch.bfloat16)).removeprefix(
            "torch."
        )
        return LinearRecurrentContract(
            family_key="mamba_2",
            contract_version="1",
            partition_kind="head_sharded_full_tree",
            projected_streams=(
                ProjectedStreamSpec(
                    name="z",
                    width=heads_local_tp * head_dim,
                    shard_axis="head",
                    shard_count=heads_local_tp,
                ),
                ProjectedStreamSpec(
                    name="x",
                    width=heads_local_tp * head_dim,
                    shard_axis="head",
                    shard_count=heads_local_tp,
                ),
                ProjectedStreamSpec(
                    name="B",
                    width=groups_local_tp * state_dim,
                    shard_axis="group",
                    shard_count=groups_local_tp,
                    replication="group_to_heads",
                    replication_factor=replication_factor,
                ),
                ProjectedStreamSpec(
                    name="C",
                    width=groups_local_tp * state_dim,
                    shard_axis="group",
                    shard_count=groups_local_tp,
                    replication="group_to_heads",
                    replication_factor=replication_factor,
                ),
                ProjectedStreamSpec(
                    name="dt",
                    width=heads_local_tp,
                    shard_axis="head",
                    shard_count=heads_local_tp,
                ),
            ),
            states=(
                RecurrentStateSpec(
                    name="conv",
                    shape=(
                        local_heads * head_dim + 2 * local_groups * state_dim,
                        _CONVOLUTION_WIDTH - 1,
                    ),
                    dtype=dtype,
                ),
                RecurrentStateSpec(
                    name="ssm",
                    shape=(local_heads, head_dim, state_dim),
                    dtype="float32",
                ),
            ),
            convolution_width=_CONVOLUTION_WIDTH,
            local_chunk_size=_LOCAL_CHUNK_SIZE,
            activation="silu",
            local_kernel_implementation_id=_MAMBA_KERNEL_ID,
            layout_compatibility_key=_MAMBA_LAYOUT_KEY,
        )

    def linear_recurrent_planner_config(self, provider: Any) -> object:
        del provider
        return HeadShardedFullTreePlannerConfig(max_padded_tokens=262_144)

    def install_linear_recurrent_hooks(self, model_chunks: Sequence[Any]) -> None:
        from art.megatron.mamba.operator import install_prefix_tree_mamba_hooks

        install_prefix_tree_mamba_hooks(model_chunks)

    def get_forward_kwargs(self, model: Any, **kwargs: Any) -> dict[str, Any]:
        del model
        return {"extra_block_kwargs": kwargs}

    def collect_layer_families(self, provider: Any) -> list[LayerFamilyInstance]:
        pattern = _main_pattern(provider)
        families: list[LayerFamilyInstance] = []
        for index, symbol in enumerate(pattern):
            module_path = f"decoder.layers.{index}"
            if symbol == "M":
                families.append(
                    LayerFamilyInstance(
                        key="mamba2_recurrent",
                        layer_index=index,
                        module_path=module_path,
                        module_type="MambaLayer",
                    )
                )
            elif symbol == "E":
                families.append(
                    LayerFamilyInstance(
                        key="grouped_moe_mlp",
                        layer_index=index,
                        module_path=module_path,
                        module_type="MoETransformerLayer",
                    )
                )
                if int(
                    getattr(provider, "moe_shared_expert_intermediate_size", 0) or 0
                ):
                    families.append(
                        LayerFamilyInstance(
                            key="shared_experts_mlp",
                            layer_index=index,
                            module_path=module_path,
                            module_type="MoETransformerLayer",
                        )
                    )
            elif symbol == "*":
                families.append(
                    LayerFamilyInstance(
                        key="standard_attention",
                        layer_index=index,
                        module_path=module_path,
                        module_type="TransformerLayer",
                    )
                )
            else:
                raise RuntimeError(f"Unsupported Nemotron-H layer symbol {symbol!r}")
        families.append(
            LayerFamilyInstance(
                key="nemotron_h_repeated_hybrid_groups",
                layer_index=12,
                module_path="decoder.layers.12",
                module_type="TransformerLayer",
            )
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
        from megatron.core.ssm.mamba_layer import MambaLayer
        from megatron.core.transformer.attention import SelfAttention
        from megatron.core.transformer.transformer_layer import (
            MoETransformerLayer,
            TransformerLayer,
        )

        from art.megatron.lora import (
            _adapter_model_prefix,
            _is_language_transformer_layer_name,
            wrap_grouped_moe_experts,
            wrap_mamba_mixer,
            wrap_shared_experts_mlp,
            wrap_standard_self_attention,
        )

        self.validate_provider_for_runtime(provider)
        heads = int(provider.mamba_num_heads)
        head_dim = int(provider.mamba_head_dim)
        groups = int(provider.mamba_num_groups)
        state_dim = int(provider.mamba_state_dim)
        in_proj_components = (
            heads * head_dim,
            heads * head_dim,
            groups * state_dim,
            groups * state_dim,
            heads,
        )
        targets = set(target_modules)
        for chunk in model_chunks:
            for module_name, module in chunk.named_modules():
                if not isinstance(module, (MambaLayer, TransformerLayer)):
                    continue
                if not _is_language_transformer_layer_name(module_name):
                    continue
                prefix = _adapter_model_prefix(module)
                if isinstance(module, MambaLayer):
                    wrap_mamba_mixer(
                        module.mixer,
                        adapter_model_prefix=f"{prefix}.mixer",
                        provider=provider,
                        target_modules=targets,
                        component_sizes=in_proj_components,
                        rank=rank,
                        alpha=alpha,
                    )
                elif isinstance(module, MoETransformerLayer):
                    wrap_grouped_moe_experts(
                        module.mlp.experts,
                        adapter_model_prefix=prefix,
                        target_modules=targets,
                        rank=rank,
                        alpha=alpha,
                        non_gated=True,
                        module_namespace="mixer.experts",
                    )
                    shared = getattr(module.mlp, "shared_experts", None)
                    if shared is not None:
                        wrap_shared_experts_mlp(
                            shared,
                            adapter_model_prefix=prefix,
                            provider=provider,
                            target_modules=targets,
                            rank=rank,
                            alpha=alpha,
                            non_gated=True,
                            module_namespace="mixer.shared_experts",
                        )
                elif isinstance(module.self_attention, SelfAttention):
                    wrap_standard_self_attention(
                        module.self_attention,
                        adapter_model_prefix=prefix,
                        provider=provider,
                        target_modules=targets,
                        rank=rank,
                        alpha=alpha,
                        projection_namespace="mixer",
                    )
                else:
                    raise TypeError(
                        "Unsupported Nemotron-H TransformerLayer payload: "
                        f"{type(module.self_attention).__name__}"
                    )

    def build_adapter_weights_by_base(
        self, model_chunks: Sequence[Any]
    ) -> dict[str, list[Any]]:
        from art.megatron.weights.adapter_export import (
            build_mamba_stack_adapter_weights,
        )

        return build_mamba_stack_adapter_weights(model_chunks)

    def expert_packed_lora_groups(self) -> tuple[ExpertPackedLoraGroup, ...]:
        return (
            ExpertPackedLoraGroup(
                art_group_suffix=".mixer.experts",
                slots=tuple(
                    ExpertPackedLoraSlot(
                        source_projection=projection,
                        source_lora=lora,
                        output_suffix=f"{projection}.{lora}.weight",
                        pack_layout="expert_rows",
                    )
                    for projection in ("up_proj", "down_proj")
                    for lora in ("lora_A", "lora_B")
                ),
            ),
        )

    def to_vllm_lora_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        adapter_config: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        converted = _convert_lora_tensors(
            tensors, adapter_config=adapter_config, to_vllm=True
        )
        return converted, _persisted_lora_config(adapter_config)

    def to_vllm_lora_config(self, adapter_config: dict[str, Any]) -> dict[str, Any]:
        return _vllm_lora_config(adapter_config)

    def from_vllm_lora_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        adapter_config: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        return _convert_lora_tensors(
            tensors, adapter_config=adapter_config, to_vllm=False
        )

    def zero_internal_padding_grads(self, model_chunks: Sequence[Any]) -> None:
        _zero_routed_base_padding(model_chunks, grads=True, params=False)
        _zero_routed_lora_padding(model_chunks, grads=True, params=False)

    def zero_internal_padding_params(self, model_chunks: Sequence[Any]) -> None:
        _zero_routed_base_padding(model_chunks, grads=False, params=True)
        _zero_routed_lora_padding(model_chunks, grads=False, params=True)

    def canonicalize_loaded_lora_state(
        self, state: dict[str, Any], model_chunks: Sequence[Any]
    ) -> dict[str, Any]:
        return _zero_loaded_padding(state, model_chunks)

    def correctness_precision(self) -> Literal["bf16", "fp32"]:
        return "bf16"

    def correctness_use_fp32_lora_reference(self) -> bool:
        return False

    def correctness_phase_pass_fns(self, oracle_harness: Any) -> dict[str, Any]:
        nonzero = {"typical_abs_scale": 1e-30, "candidate_abs_scale": 1e-30}
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
                limits={
                    "topk_mismatch_fraction": 0.0,
                    "top1_mismatch_fraction": 0.0,
                }
            ),
        }


NEMOTRON_H_MOE_HANDLER = NemotronHMoeHandler()
