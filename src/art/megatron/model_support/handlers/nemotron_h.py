from __future__ import annotations

import sys
from typing import Any, Literal, Sequence, cast

import torch

from art.megatron.mamba.plan import build_mamba_execution_plan
from art.megatron.mamba.runtime import MAMBA_STATE_KEY, install_mamba_prefix_tree_hooks
from art.megatron.model_support.handlers.default_dense import (
    DefaultMoeHandler,
    _require_moe_experts,
)
from art.megatron.model_support.internal_padding import (
    pad_dim_right,
    round_up_to_multiple,
    trim_dim_right,
    zero_lora_padding,
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


def _hf_reference_mamba_forward(
    mixer: Any,
    input_states: torch.Tensor,
    cache_params: Any = None,
    cache_position: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use the pinned training SSD scan in the otherwise-native HF mixer."""

    if cache_params is not None:
        raise RuntimeError("Nemotron-H HF parity does not support Mamba cache state")
    batch, sequence, _ = input_states.shape
    if sequence <= 0 or (
        cache_position is not None
        and (
            cache_position.dtype != torch.int64
            or tuple(cache_position.shape) != (sequence,)
            or not torch.equal(
                cache_position,
                torch.arange(sequence, device=cache_position.device),
            )
        )
    ):
        raise RuntimeError("Nemotron-H HF parity received invalid cache positions")
    implementation = sys.modules.get(type(mixer).__module__)
    mask_padding = getattr(implementation, "apply_mask_to_padding_states", None)
    if not callable(mask_padding):
        raise RuntimeError("Nemotron-H HF padding-mask helper is unavailable")
    input_dtype = input_states.dtype
    projected = mixer.in_proj(mask_padding(input_states, attention_mask))
    expected_width = (
        2 * mixer.intermediate_size
        + 2 * mixer.n_groups * mixer.ssm_state_size
        + mixer.num_heads
    )
    if int(projected.shape[-1]) != expected_width:
        raise RuntimeError("Nemotron-H HF Mamba projection geometry changed")
    gate, convolved, dt = projected.split(
        [mixer.intermediate_size, mixer.conv_dim, mixer.num_heads], dim=-1
    )
    convolved = mixer.act(
        mixer.conv1d(convolved.transpose(1, 2))[..., :sequence].transpose(1, 2)
    )
    convolved = mask_padding(convolved, attention_mask)
    group_width = mixer.n_groups * mixer.ssm_state_size
    x, b, c = convolved.split(
        [mixer.intermediate_size, group_width, group_width], dim=-1
    )
    scan_length = round_up_to_multiple(sequence, int(mixer.chunk_size))

    def pad(value: torch.Tensor, fill: float) -> torch.Tensor:
        tail = value.new_full((batch, scan_length - sequence, *value.shape[2:]), fill)
        return torch.cat((value, tail), dim=1)

    from art.megatron.mamba.operator import _mamba_chunk_scan_combined

    output = _mamba_chunk_scan_combined()(
        pad(x.view(batch, sequence, mixer.num_heads, mixer.head_dim), 0.0),
        pad(dt, -torch.inf),
        -torch.exp(mixer.A_log.float()),
        pad(b.view(batch, sequence, mixer.n_groups, mixer.ssm_state_size), 0.0),
        pad(c.view(batch, sequence, mixer.n_groups, mixer.ssm_state_size), 0.0),
        chunk_size=mixer.chunk_size,
        D=mixer.D.float(),
        z=None,
        dt_bias=mixer.dt_bias.float(),
        dt_softplus=True,
        return_final_states=False,
        state_dtype=torch.float32,
    )[:, :sequence]
    output = mixer.norm(output.reshape(batch, sequence, -1), gate)
    return mixer.out_proj(output.to(input_dtype))


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


def _padding_sizes_from_provider(provider: Any) -> tuple[int, int]:
    logical = int(getattr(provider, _LOGICAL_MOE_FFN_ATTR, 0) or 0)
    internal = int(getattr(provider, "moe_ffn_hidden_size", 0) or 0)
    if logical <= 0 or internal != round_up_to_multiple(logical, _MOE_FFN_ALIGNMENT):
        raise RuntimeError(f"Invalid Nemotron-H expert padding: {logical}->{internal}")
    return logical, internal


def _model_config(model_chunks: Sequence[Any]) -> Any | None:
    for chunk in model_chunks:
        config = getattr(chunk, "config", None)
        if config is None:
            config = getattr(getattr(chunk, "module", None), "config", None)
        if config is not None:
            return config
    return None


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


def _zero_padded_tensor(
    parameter: torch.nn.Parameter,
    *,
    dim: int,
    start: int,
    end: int,
    grads: bool,
    params: bool,
) -> None:
    tensors = [parameter.data] if params else []
    if grads:
        for value in (parameter.grad, getattr(parameter, "main_grad", None)):
            local = getattr(value, "_local_tensor", None)
            tensor = value if torch.is_tensor(value) else local
            if torch.is_tensor(tensor):
                tensors.append(tensor)
    dim %= parameter.ndim
    for tensor in tensors:
        tensor.narrow(dim, start, end - start).zero_()


def _zero_expert_padding(
    model_chunks: Sequence[Any],
    *,
    grads: bool,
    params: bool,
) -> None:
    config = _model_config(model_chunks)
    if config is None:
        return
    logical, internal = _padding_sizes_from_provider(config)
    if logical == internal:
        return

    from megatron.core.transformer.moe.experts import TEGroupedMLP

    from art.megatron import lora

    etp_size = lora._get_shard_world_size("expert_tp")
    etp_rank = lora._get_shard_rank("expert_tp")
    if internal % etp_size:
        raise RuntimeError(
            f"Padded expert width {internal} does not divide ETP{etp_size}"
        )
    local_size = internal // etp_size
    shard_start = etp_rank * local_size
    start = max(logical, shard_start) - shard_start
    end = min(internal, shard_start + local_size) - shard_start

    with torch.no_grad():
        for chunk in model_chunks:
            for module in chunk.modules():
                if isinstance(module, TEGroupedMLP):
                    fc1 = getattr(module.linear_fc1, "linear_fc1", module.linear_fc1)
                    fc2 = getattr(module.linear_fc2, "linear_fc2", module.linear_fc2)
                    for linear, dim in ((fc1, 0), (fc2, 1)):
                        for expert in range(module.num_local_experts):
                            parameter = getattr(linear, f"weight{expert}")
                            if end > start:
                                _zero_padded_tensor(
                                    parameter,
                                    dim=dim,
                                    start=start,
                                    end=end,
                                    grads=grads,
                                    params=params,
                                )
                if not isinstance(module, lora.LoRA):
                    continue
                prefix = module.adapter_model_prefix
                if ".mixer.experts." not in prefix:
                    continue
                if prefix.endswith(".up_proj"):
                    parameter, dim = module.B_T, -1
                elif prefix.endswith(".down_proj"):
                    parameter, dim = module.A_T, -2
                else:
                    continue
                zero_lora_padding(
                    parameter,
                    dim=dim,
                    logical=logical,
                    internal=internal,
                    components=(internal,),
                    grads=grads,
                    params=params,
                )


class NemotronHHandler(DefaultMoeHandler):
    key = "nemotron_h_moe"
    has_recurrent_layers = True
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
        from megatron.core.transformer.transformer_layer import (
            MoETransformerLayer,
            TransformerLayer,
        )

        from art.megatron.lora import (
            wrap_grouped_moe_experts,
            wrap_mamba_mixer,
            wrap_shared_experts_mlp,
            wrap_standard_self_attention,
        )

        heads = int(provider.mamba_num_heads)
        inner = heads * int(provider.mamba_head_dim)
        state = int(provider.mamba_num_groups) * int(provider.mamba_state_dim)
        components = (inner, inner, state, state, heads)
        targets = set(target_modules)
        for chunk in model_chunks:
            for module in chunk.modules():
                if not isinstance(module, (MambaLayer, TransformerLayer)):
                    continue
                prefix = f"base_model.model.backbone.layers.{module.layer_number - 1}"
                if isinstance(module, MambaLayer):
                    wrap_mamba_mixer(
                        module.mixer,
                        adapter_model_prefix=f"{prefix}.mixer",
                        provider=provider,
                        target_modules=targets,
                        component_sizes=components,
                        rank=rank,
                        alpha=alpha,
                    )
                elif isinstance(module, MoETransformerLayer):
                    wrap_grouped_moe_experts(
                        _require_moe_experts(module),
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
                else:
                    wrap_standard_self_attention(
                        module.self_attention,
                        adapter_model_prefix=prefix,
                        provider=provider,
                        target_modules=targets,
                        rank=rank,
                        alpha=alpha,
                        projection_namespace="mixer",
                    )

    def build_adapter_weights_by_base(
        self,
        model_chunks: Sequence[Any],
    ) -> dict[str, list[Any]]:
        from art.megatron.weights.adapter_export import (
            build_mamba_stack_adapter_weights,
        )

        return build_mamba_stack_adapter_weights(model_chunks)

    def zero_internal_padding_grads(self, model_chunks: Sequence[Any]) -> None:
        _zero_expert_padding(model_chunks, grads=True, params=False)

    def zero_internal_padding_params(self, model_chunks: Sequence[Any]) -> None:
        _zero_expert_padding(model_chunks, grads=False, params=True)

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

    def prepare_hf_reference_model_class(self, model_class: type[Any]) -> type[Any]:
        if model_class.__name__ != "NemotronHForCausalLM":
            raise TypeError("Nemotron-H HF reference model class changed")
        strict_fp32 = tuple(
            getattr(model_class, "_keep_in_fp32_modules_strict", ()) or ()
        ) + (
            "*.mixer.A_log",
            "*.mixer.D",
            "*.mixer.gate.e_score_correction_bias",
        )

        def dtype_plan(model: Any, dtype: torch.dtype) -> dict[str, torch.dtype]:
            return {**model_class._get_dtype_plan(model, dtype), "*": dtype}

        return type(
            f"ArtStrictFp32{model_class.__name__}",
            (model_class,),
            {
                "__module__": model_class.__module__,
                "_get_dtype_plan": dtype_plan,
                "_keep_in_fp32_modules_strict": strict_fp32,
            },
        )

    def prepare_hf_reference_model(self, model: Any) -> Any:
        pattern = str(model.config.hybrid_override_pattern)
        fp32_names = {
            name
            for index, symbol in enumerate(pattern)
            for name in (
                (
                    f"backbone.layers.{index}.mixer.A_log",
                    f"backbone.layers.{index}.mixer.D",
                )
                if symbol == "M"
                else (f"backbone.layers.{index}.mixer.gate.e_score_correction_bias",)
                if symbol == "E"
                else ()
            )
        }
        state = model.state_dict()
        params_dtype = model.backbone.embeddings.weight.dtype
        invalid = {
            name: tensor.dtype
            for name, tensor in state.items()
            if tensor.is_floating_point()
            and tensor.dtype != (torch.float32 if name in fp32_names else params_dtype)
        }
        if fp32_names - state.keys() or invalid:
            raise RuntimeError("Nemotron-H HF reference precision changed")
        expected_names = [
            f"backbone.layers.{index}.mixer"
            for index, symbol in enumerate(pattern)
            if symbol == "M"
        ]
        mixers = [
            (name, module)
            for name, module in model.named_modules()
            if type(module).__name__ == "NemotronHMamba2Mixer"
        ]
        if [name for name, _ in mixers] != expected_names:
            raise RuntimeError("Nemotron-H HF Mamba topology changed")
        mixer_types = {type(module) for _, module in mixers}
        if len(mixer_types) != 1:
            raise RuntimeError("Nemotron-H HF Mamba implementation changed")
        mixer_type = mixer_types.pop()
        reference_type = type(
            "ArtReferenceNemotronHMamba2Mixer",
            (mixer_type,),
            {
                "__module__": mixer_type.__module__,
                "torch_forward": _hf_reference_mamba_forward,
            },
        )
        for _, mixer in mixers:
            mixer.__class__ = reference_type
        return model

    def correctness_precision(self) -> Literal["bf16", "fp32"]:
        return "bf16"

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
        }

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
