from __future__ import annotations

from types import MethodType
from typing import Any, Callable, Iterable, Literal, NamedTuple, Sequence, cast

import torch
from torch import Tensor
import torch.distributed as dist
import torch.nn.functional as F

from .conv_gelu import packed_varlen_causal_conv
from .fla_cp import chunk_gated_delta_rule_native_cp
from .gdn_shared_prefix import (
    GdnPackedExecutionSpec,
    GdnRankExecutionPlan,
    GdnSegmentBucketPlan,
    GdnStateExchangePlan,
    build_gdn_rank_execution_plan,
    parse_gdn_shared_prefix_segments,
)
from .segment_layout import (
    gather_bucket_streams_compact as _gather_bucket_streams_compact_fused,
)
from .segment_layout import (
    prepare_packed_recurrent_inputs as _prepare_packed_recurrent_inputs_fused,
)
from .segment_layout import (
    scatter_bucket_output_compact as _scatter_bucket_output_fused,
)

_GDN_ATTENTION_ORIGINAL_SHAPE_ATTR = "_art_gdn_attention_original_shape"
_GDN_TRACE_TOKEN_UID_HOOKS: Any | None = None


class _GdnIslandBoundary(NamedTuple):
    is_gdn: bool
    island_id: int | None
    input_layout: Literal["attention", "gdn"]
    output_layout: Literal["attention", "gdn"]


def set_gdn_trace_token_uid_hooks(hooks: Any | None) -> Any | None:
    global _GDN_TRACE_TOKEN_UID_HOOKS
    previous = _GDN_TRACE_TOKEN_UID_HOOKS
    _GDN_TRACE_TOKEN_UID_HOOKS = hooks
    return previous


def install_shared_prefix_gdn_hooks(model_chunks: Sequence[Any]) -> None:
    """Patch Megatron GatedDeltaNet modules to honor ART shared-prefix packing."""

    gated_delta_net_type = _optional_gated_delta_net_type()
    if gated_delta_net_type is None:
        return
    for chunk in model_chunks:
        for module in chunk.modules():
            if not isinstance(module, gated_delta_net_type):
                continue
            if getattr(module, "_art_shared_prefix_gdn_hooked", False):
                continue
            original_forward = module.forward
            module._art_physical_forward = original_forward
            module.forward = MethodType(_shared_prefix_forward, module)
            module._art_shared_prefix_gdn_hooked = True


def install_gdn_island_hooks(model_chunks: Sequence[Any]) -> None:
    """Hoist CP layout conversion across consecutive Transformer GDN layers."""

    gated_delta_net_type = _optional_gated_delta_net_type()
    transformer_layer_type = _optional_transformer_layer_type()
    if gated_delta_net_type is None or transformer_layer_type is None:
        return

    next_island_id = 0
    for chunk in model_chunks:
        _install_empty_safe_norm_hooks(chunk)
        layers = [
            module
            for module in chunk.modules()
            if isinstance(module, transformer_layer_type)
            and hasattr(module, "self_attention")
        ]
        boundaries, next_island_id = _build_gdn_island_boundaries(
            layers,
            gated_delta_net_type,
            next_island_id=next_island_id,
        )
        for layer, boundary in zip(layers, boundaries, strict=True):
            layer._art_gdn_island_boundary = boundary
            if getattr(layer, "_art_gdn_island_hooked", False):
                continue
            layer._art_gdn_island_physical_forward = layer.forward
            layer.forward = MethodType(_gdn_island_layer_forward, layer)
            layer._art_gdn_island_hooked = True


def _build_gdn_island_boundaries(
    layers: Sequence[Any],
    gated_delta_net_type: type[Any],
    *,
    next_island_id: int,
) -> tuple[list[_GdnIslandBoundary], int]:
    layer_is_gdn = [
        isinstance(layer.self_attention, gated_delta_net_type) for layer in layers
    ]
    boundaries: list[_GdnIslandBoundary] = []
    active_island_id: int | None = None
    for index, is_gdn in enumerate(layer_is_gdn):
        prev_is_gdn = index > 0 and layer_is_gdn[index - 1]
        next_is_gdn = index + 1 < len(layer_is_gdn) and layer_is_gdn[index + 1]
        if is_gdn:
            if not prev_is_gdn:
                active_island_id = next_island_id
                next_island_id += 1
            boundaries.append(
                _GdnIslandBoundary(
                    True,
                    active_island_id,
                    "gdn" if prev_is_gdn else "attention",
                    "gdn" if next_is_gdn else "attention",
                )
            )
        else:
            active_island_id = None
            boundaries.append(_GdnIslandBoundary(False, None, "attention", "attention"))
    return boundaries, next_island_id


def _optional_gated_delta_net_type() -> type[Any] | None:
    try:
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    except ImportError:
        return None
    return GatedDeltaNet


def _optional_transformer_layer_type() -> type[Any] | None:
    try:
        from megatron.core.transformer.transformer_layer import TransformerLayer
    except ImportError:
        return None
    return TransformerLayer


def _gdn_island_layer_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
    attention_bias = kwargs.get("attention_bias")
    plan = getattr(attention_bias, "gdn_execution_plan", None)
    original_forward = cast(Callable[..., Any], self._art_gdn_island_physical_forward)
    if plan is None or int(getattr(plan, "cp_size", 1)) <= 1:
        return original_forward(*args, **kwargs)

    hidden_states = _layer_forward_hidden_states(args, kwargs)
    if hidden_states is None:
        return original_forward(*args, **kwargs)

    boundary = cast(_GdnIslandBoundary, self._art_gdn_island_boundary)
    if not boundary.is_gdn:
        if getattr(attention_bias, "gdn_hidden_layout", "attention") != "attention":
            _mark_attention_layout_active(attention_bias, hidden_states)
        return original_forward(*args, **kwargs)

    if boundary.input_layout == "gdn":
        original_shape = _gdn_attention_original_shape_from_tensor(
            hidden_states
        ) or _gdn_attention_original_shape_from_state(
            attention_bias,
            gdn=self.self_attention,
            island_id=boundary.island_id,
        )
        if original_shape is not None:
            _store_gdn_attention_original_shape(
                attention_bias,
                original_shape,
                gdn=self.self_attention,
                island_id=boundary.island_id,
            )
        _mark_gdn_layout_active(
            attention_bias,
            hidden_states,
            gdn=self.self_attention,
            island_id=boundary.island_id,
        )
    else:
        hidden_states = _enter_gdn_island_layout(
            hidden_states,
            attention_bias,
            gdn=self.self_attention,
            island_id=boundary.island_id,
            force=True,
        )
        args, kwargs = _replace_layer_hidden_states(args, kwargs, hidden_states)
    previous_input_layout = getattr(attention_bias, "gdn_input_layout", None)
    previous_output_layout = getattr(attention_bias, "gdn_output_layout", None)
    setattr(attention_bias, "gdn_input_layout", "gdn")
    setattr(attention_bias, "gdn_output_layout", "gdn")

    try:
        output = original_forward(*args, **kwargs)
    finally:
        setattr(attention_bias, "gdn_input_layout", previous_input_layout)
        setattr(attention_bias, "gdn_output_layout", previous_output_layout)
    if boundary.output_layout == "gdn":
        original_shape = _gdn_attention_original_shape_from_state(
            attention_bias, gdn=self.self_attention, island_id=boundary.island_id
        )
        hidden_out = _attach_gdn_attention_original_shape(
            _layer_output_hidden_states(output),
            original_shape,
        )
        _mark_gdn_layout_active(
            attention_bias,
            hidden_out,
            gdn=self.self_attention,
            island_id=boundary.island_id,
        )
        return _replace_layer_output_hidden_states(output, hidden_out)
    hidden_out = _leave_gdn_island_layout(
        _layer_output_hidden_states(output),
        attention_bias,
        gdn=self.self_attention,
        island_id=boundary.island_id,
    )
    return _replace_layer_output_hidden_states(output, hidden_out)


def _layer_forward_hidden_states(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Tensor | None:
    hidden_states = kwargs.get("hidden_states")
    if isinstance(hidden_states, Tensor):
        return hidden_states
    if args and isinstance(args[0], Tensor):
        return args[0]
    return None


def _replace_layer_hidden_states(
    args: tuple[Any, ...], kwargs: dict[str, Any], hidden_states: Tensor
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if "hidden_states" in kwargs:
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = hidden_states
        return args, kwargs
    if args:
        return (hidden_states, *args[1:]), kwargs
    kwargs = dict(kwargs)
    kwargs["hidden_states"] = hidden_states
    return args, kwargs


def _layer_output_hidden_states(output: Any) -> Tensor:
    if isinstance(output, tuple):
        return cast(Tensor, output[0])
    return cast(Tensor, output)


def _replace_layer_output_hidden_states(output: Any, hidden_states: Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden_states, *output[1:])
    return hidden_states


def _install_empty_safe_norm_hooks(root: Any) -> None:
    if not isinstance(root, torch.nn.Module):
        return
    for module in root.modules():
        if getattr(module, "_art_empty_safe_norm_hooked", False):
            continue
        if not _is_empty_safe_norm_target(module):
            continue
        module._art_empty_safe_norm_physical_forward = module.forward
        module.forward = MethodType(_empty_safe_norm_forward, module)
        module._art_empty_safe_norm_hooked = True


def _is_empty_safe_norm_target(module: Any) -> bool:
    if not isinstance(getattr(module, "weight", None), Tensor):
        return False
    module_name = type(module).__name__
    module_path = type(module).__module__
    return module_name in {"RMSNorm", "LayerNorm"} and module_path.startswith(
        "transformer_engine."
    )


def _empty_safe_norm_forward(
    self: Any, input_: Tensor, *args: Any, **kwargs: Any
) -> Any:
    if isinstance(input_, Tensor) and int(input_.numel()) == 0:
        return _apply_explicit_norm(
            self,
            input_,
            config=None,
            weight_name="weight",
            bias_name="bias",
        )
    original_forward = cast(
        Callable[..., Any], self._art_empty_safe_norm_physical_forward
    )
    return original_forward(input_, *args, **kwargs)


@torch.compiler.disable
def _shared_prefix_forward(
    self: Any,
    hidden_states: Tensor,
    attention_mask: Tensor,
    key_value_states: Tensor | None = None,
    inference_context: Any | None = None,
    attention_bias: Any | None = None,
    packed_seq_params: Any | None = None,
    sequence_len_offset: int | None = None,
    *,
    inference_params: Any | None = None,
    **kwargs: Any,
) -> tuple[Tensor, Tensor | None]:
    group_ids = getattr(attention_bias, "group_ids", None)
    parent_ids = getattr(attention_bias, "parent_ids", None)
    execution_spec = getattr(attention_bias, "gdn_execution_spec", None)
    execution_plan = getattr(attention_bias, "gdn_execution_plan", None)
    if group_ids is None or parent_ids is None:
        original_forward = cast(
            Callable[..., tuple[Tensor, Tensor | None]], self._art_physical_forward
        )
        return original_forward(
            hidden_states,
            attention_mask,
            key_value_states=key_value_states,
            inference_context=inference_context,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            inference_params=inference_params,
            **kwargs,
        )

    del attention_mask, key_value_states, sequence_len_offset, kwargs
    if inference_context is not None or inference_params is not None:
        raise NotImplementedError("ART shared-prefix GDN does not support inference.")
    if packed_seq_params is not None:
        raise NotImplementedError(
            "PackedSeqParams is not used in ART shared-prefix GDN."
        )
    current_layout = _normalize_cp_layout(
        getattr(attention_bias, "gdn_hidden_layout", "attention")
    )
    input_layout = _normalize_cp_layout(
        getattr(attention_bias, "gdn_input_layout", None) or current_layout
    )
    output_layout = _normalize_cp_layout(
        getattr(attention_bias, "gdn_output_layout", None) or current_layout
    )
    mark_layout = execution_plan is not None and int(execution_plan.cp_size) > 1
    if mark_layout:
        _mark_cp_layout_active(
            attention_bias, hidden_states, gdn=self, layout=input_layout
        )
    output = gdn_shared_prefix_forward(
        self,
        hidden_states,
        group_ids=cast(Tensor, group_ids),
        parent_ids=cast(Tensor, parent_ids),
        execution_spec=cast(GdnPackedExecutionSpec | None, execution_spec),
        execution_plan=cast(GdnRankExecutionPlan | None, execution_plan),
        input_layout=input_layout,
        output_layout=output_layout,
        require_prebuilt_plan=True,
    )
    if mark_layout:
        _mark_cp_layout_active(
            attention_bias,
            _layer_output_hidden_states(output),
            gdn=self,
            layout=output_layout,
        )
    return output


@torch.compiler.disable
def gdn_shared_prefix_forward(
    gdn: Any,
    hidden_states: Tensor,
    *,
    group_ids: Tensor,
    parent_ids: Tensor,
    execution_spec: GdnPackedExecutionSpec | None = None,
    execution_plan: GdnRankExecutionPlan | None = None,
    cp_group: Any | None = None,
    require_prebuilt_plan: bool = False,
    input_layout: Literal["attention", "gdn"] = "attention",
    output_layout: Literal["attention", "gdn"] = "attention",
) -> tuple[Tensor, Tensor | None]:
    """Run one GDN layer over ART shared-prefix packed rows."""

    return run_gdn_layer(
        gdn,
        hidden_states,
        group_ids=group_ids,
        parent_ids=parent_ids,
        execution_spec=execution_spec,
        execution_plan=execution_plan,
        cp_group=cp_group,
        require_prebuilt_plan=require_prebuilt_plan,
        input_layout=input_layout,
        output_layout=output_layout,
    )


@torch.compiler.disable
def run_gdn_layer(
    gdn: Any,
    hidden_states: Tensor,
    *,
    group_ids: Tensor,
    parent_ids: Tensor,
    execution_spec: GdnPackedExecutionSpec | None = None,
    execution_plan: GdnRankExecutionPlan | None = None,
    cp_group: Any | None = None,
    require_prebuilt_plan: bool = False,
    input_layout: Literal["attention", "gdn"] = "attention",
    output_layout: Literal["attention", "gdn"] = "attention",
) -> tuple[Tensor, Tensor | None]:
    """Run one production shared-prefix GDN layer."""

    _disable_reentrant_te_linear_transpose_cache(gdn)
    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must be [S, B, D], got {tuple(hidden_states.shape)}"
        )
    seq_len, batch_size, _ = hidden_states.shape
    requested_cp_size = (
        execution_plan.cp_size
        if execution_plan is not None
        else int(getattr(gdn, "sp_size", 1))
    )
    cp_rank = (
        execution_plan.cp_rank
        if execution_plan is not None
        else _default_cp_rank(requested_cp_size)
    )
    full_shape_required = requested_cp_size == 1
    expected_group_seq_len = seq_len
    if full_shape_required and _gdn_uses_sequence_parallel(gdn):
        expected_group_seq_len *= int(getattr(gdn, "sp_size", 1))
    if full_shape_required and (
        int(group_ids.shape[0]) != batch_size
        or int(group_ids.shape[1]) != expected_group_seq_len
    ):
        raise ValueError(
            "shared-prefix GDN group_ids shape must match the logical sequence "
            "processed by Megatron GDN after sequence-parallel input gather, got "
            f"hidden={tuple(hidden_states.shape)} "
            f"group_ids={tuple(group_ids.shape)} "
            f"expected_group_shape={(batch_size, expected_group_seq_len)}"
        )

    if require_prebuilt_plan and execution_plan is None:
        raise ValueError(
            "ART shared-prefix GDN production path requires a prebuilt "
            "GDN execution plan on SharedPrefixAttentionState. Build it once "
            "per packed sequence via create_shared_prefix_state(..., "
            "build_gdn_execution_spec=True)."
        )

    if execution_spec is None and execution_plan is None:
        execution_spec = parse_gdn_shared_prefix_segments(group_ids, parent_ids)
    if (
        execution_spec is not None
        and requested_cp_size == 1
        and (
            execution_spec.batch_size != batch_size
            or execution_spec.sequence_length != expected_group_seq_len
        )
    ):
        raise ValueError(
            "GDN execution spec shape must match hidden_states, got "
            f"spec={(execution_spec.batch_size, execution_spec.sequence_length)} "
            f"expected={(batch_size, expected_group_seq_len)} "
            f"hidden={(batch_size, seq_len)}"
        )
    if execution_plan is None:
        if execution_spec is None:
            raise ValueError("GDN execution spec is required to build a missing plan")
        execution_plan = build_gdn_rank_execution_plan(
            execution_spec,
            device=hidden_states.device,
            cp_rank=cp_rank,
            cp_size=requested_cp_size,
        )
    elif execution_plan.cp_size == 1 and (
        execution_plan.batch_size != batch_size
        or execution_plan.sequence_length != expected_group_seq_len
    ):
        raise ValueError(
            "GDN execution plan shape must match hidden_states, got "
            f"plan={(execution_plan.batch_size, execution_plan.sequence_length)} "
            f"expected={(batch_size, expected_group_seq_len)} "
            f"hidden={(batch_size, seq_len)}"
        )
    if execution_plan.cp_size != 1:
        return _run_cp_planned_prefixes_and_completions(
            gdn,
            hidden_states,
            execution_plan,
            group=cp_group or _default_cp_group(execution_plan.cp_size),
            input_layout=input_layout,
            output_layout=output_layout,
        )
    if input_layout != "attention" or output_layout != "attention":
        raise ValueError("GDN layout controls require a CP execution plan")
    return _run_tree_prefixes(gdn, hidden_states, execution_plan)


def _run_tree_prefixes(
    gdn: Any,
    hidden_states: Tensor,
    plan: GdnRankExecutionPlan,
) -> tuple[Tensor, Tensor | None]:
    qkv, gate, beta, recurrent_g = _project_gdn_inputs(gdn, hidden_states)
    gate = gate.clone()
    recurrent_output = torch.zeros_like(gate)
    recurrent_output, _cp_dependency = _run_tree_depth_buckets(
        gdn,
        qkv,
        beta,
        recurrent_g,
        recurrent_output,
        plan,
        state_reference=hidden_states,
    )
    return _project_gdn_output(gdn, recurrent_output, gate, plan)


def _run_tree_depth_buckets(
    gdn: Any,
    qkv: Tensor,
    beta: Tensor,
    recurrent_g: Tensor,
    recurrent_output: Tensor,
    plan: GdnRankExecutionPlan,
    *,
    state_reference: Tensor,
    group: Any | None = None,
    cp_dependency: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    state_cache = _TreeStateChunkCache(
        device=state_reference.device,
    )

    for depth, buckets in enumerate(plan.tree_segment_buckets_by_depth):
        if depth < len(plan.tree_state_exchanges_by_depth):
            cp_dependency = state_cache.exchange_remote_parent_states(
                gdn,
                plan.tree_state_exchanges_by_depth[depth],
                state_reference=state_reference,
                rank=plan.cp_rank,
                group=group,
                cp_dependency=cp_dependency,
            )
        if depth < len(plan.tree_chain_buckets_by_depth):
            for bucket in plan.tree_chain_buckets_by_depth[depth]:
                recurrent_output, cp_dependency = _run_tree_bucket(
                    gdn,
                    qkv,
                    beta,
                    recurrent_g,
                    recurrent_output,
                    state_cache,
                    bucket,
                    state_reference=state_reference,
                    group=group,
                    cp_dependency=cp_dependency,
                    recurrent_cp=True,
                    scale_parent_state_gradient=1.0 / plan.cp_size,
                )

        for bucket in buckets:
            recurrent_output, cp_dependency = _run_tree_bucket(
                gdn,
                qkv,
                beta,
                recurrent_g,
                recurrent_output,
                state_cache,
                bucket,
                state_reference=state_reference,
                cp_dependency=cp_dependency,
            )

    return recurrent_output, cp_dependency


def _run_tree_bucket(
    gdn: Any,
    qkv: Tensor,
    beta: Tensor,
    recurrent_g: Tensor,
    recurrent_output: Tensor,
    state_cache: "_TreeStateChunkCache",
    bucket: GdnSegmentBucketPlan,
    *,
    state_reference: Tensor,
    group: Any | None = None,
    cp_dependency: Tensor | None = None,
    recurrent_cp: bool = False,
    scale_parent_state_gradient: float | None = None,
) -> tuple[Tensor, Tensor | None]:
    parent_conv, parent_rec = state_cache.parent_states(
        gdn,
        bucket,
        state_reference=state_reference,
    )
    if _bucket_has_parent_state(bucket):
        parent_conv, parent_rec = _couple_parent_states(parent_conv, parent_rec)
        if scale_parent_state_gradient is not None:
            parent_conv = _scale_state_gradient(
                parent_conv,
                scale_parent_state_gradient,
            )
            parent_rec = _scale_state_gradient(parent_rec, scale_parent_state_gradient)
    segment_qkv, segment_beta, segment_g = _gather_bucket_streams(
        qkv,
        beta,
        recurrent_g,
        bucket,
    )
    if cp_dependency is not None:
        segment_qkv = _add_autograd_dependency(segment_qkv, cp_dependency)
        segment_beta = _add_autograd_dependency(segment_beta, cp_dependency)
        segment_g = _add_autograd_dependency(segment_g, cp_dependency)
        parent_conv = _add_autograd_dependency(parent_conv, cp_dependency)
        parent_rec = _add_autograd_dependency(parent_rec, cp_dependency)
    segment_out, segment_conv, segment_rec = run_gdn_bucket(
        bucket,
        (segment_qkv, segment_beta, segment_g),
        (parent_conv, parent_rec),
        gdn=gdn,
        group=group,
        recurrent_cp=recurrent_cp,
        output_final_state=bucket.needs_final_state or recurrent_cp,
    )
    if bucket.needs_final_state and (segment_conv is None or segment_rec is None):
        raise RuntimeError("tree GDN execution must return final states")
    if (
        bucket.needs_final_state
        and segment_conv is not None
        and segment_rec is not None
    ):
        cp_dependency = _make_autograd_dependency(
            segment_out, segment_conv, segment_rec
        )
    else:
        cp_dependency = _make_autograd_dependency(segment_out)
    recurrent_output = _scatter_bucket_recurrent_output(
        recurrent_output,
        bucket,
        segment_out,
    )
    if bucket.needs_final_state:
        state_cache.append(
            bucket,
            cast(Tensor, segment_conv),
            cast(Tensor, segment_rec),
        )
    return recurrent_output, cp_dependency


class _TreeStateChunkCache:
    def __init__(self, *, device: torch.device) -> None:
        self._device = device
        self._conv_chunks: list[Tensor] = []
        self._rec_chunks: list[Tensor] = []
        self._source_by_family: list[tuple[int, int] | None] = []

    def append(self, bucket: GdnSegmentBucketPlan, conv: Tensor, rec: Tensor) -> None:
        self.append_families(_bucket_family_indices_cpu(bucket), conv, rec)

    def append_families(
        self, family_indices: Sequence[int], conv: Tensor, rec: Tensor
    ) -> None:
        if len(family_indices) == 0:
            return
        if int(conv.shape[0]) != len(family_indices):
            raise ValueError(
                "tree GDN state cache conv batch must match family count, got "
                f"{tuple(conv.shape)} and {len(family_indices)} families"
            )
        if int(rec.shape[0]) != len(family_indices):
            raise ValueError(
                "tree GDN state cache recurrent batch must match family count, got "
                f"{tuple(rec.shape)} and {len(family_indices)} families"
            )
        chunk_index = len(self._conv_chunks)
        self._conv_chunks.append(conv)
        self._rec_chunks.append(rec)
        max_family = max(int(index) for index in family_indices)
        if max_family >= len(self._source_by_family):
            self._source_by_family.extend(
                None for _ in range(max_family + 1 - len(self._source_by_family))
            )
        for source_row, family_index in enumerate(family_indices):
            self._source_by_family[int(family_index)] = (chunk_index, source_row)

    def exchange_remote_parent_states(
        self,
        gdn: Any,
        exchange: GdnStateExchangePlan | None,
        *,
        state_reference: Tensor,
        rank: int,
        group: Any | None,
        cp_dependency: Tensor | None,
    ) -> Tensor | None:
        if exchange is None:
            return cp_dependency
        from .layout import exchange_rank_tensor_all_to_all

        source_conv, source_rec = self.states_for_families(
            gdn,
            exchange.source_family_indices,
            state_reference=state_reference,
        )
        if cp_dependency is not None:
            source_conv = _add_autograd_dependency(source_conv, cp_dependency)
            source_rec = _add_autograd_dependency(source_rec, cp_dependency)
        remote_conv = exchange_rank_tensor_all_to_all(
            source_conv,
            exchange.exchange,
            rank=rank,
            group=group,
            backward_plan=exchange.reverse_exchange,
        )
        remote_rec = exchange_rank_tensor_all_to_all(
            source_rec,
            exchange.exchange,
            rank=rank,
            group=group,
            backward_plan=exchange.reverse_exchange,
        )
        self.append_families(exchange.dest_family_indices, remote_conv, remote_rec)
        dependency = _make_zero_autograd_dependency(
            source_conv, source_rec, remote_conv, remote_rec
        )
        return dependency if cp_dependency is None else dependency + cp_dependency

    def states_for_families(
        self,
        gdn: Any,
        family_indices: Sequence[int],
        *,
        state_reference: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if len(family_indices) == 0:
            conv = _zero_conv_state(gdn, state_reference, batch_size=0)
            rec = _zero_recurrent_state(gdn, state_reference, batch_size=0)
            return conv.requires_grad_(True), rec.requires_grad_(True)
        return self._mixed_parent_states(
            gdn,
            tuple(int(index) for index in family_indices),
            state_reference=state_reference,
            batch_size=len(family_indices),
            roots_allowed=False,
        )

    def parent_states(
        self,
        gdn: Any,
        bucket: GdnSegmentBucketPlan,
        *,
        state_reference: Tensor,
    ) -> tuple[Tensor, Tensor]:
        parent_indices = bucket.parent_indices
        if parent_indices is None:
            raise RuntimeError("tree GDN bucket is missing parent indices")
        parent_indices_cpu = _bucket_parent_indices_cpu(bucket)
        batch_size = bucket.segment_count
        if all(parent_index < 0 for parent_index in parent_indices_cpu):
            return (
                _zero_conv_state(gdn, state_reference, batch_size=batch_size),
                _zero_recurrent_state(gdn, state_reference, batch_size=batch_size),
            )

        return self._mixed_parent_states(
            gdn,
            parent_indices_cpu,
            state_reference=state_reference,
            batch_size=batch_size,
        )

    def _mixed_parent_states(
        self,
        gdn: Any,
        parent_indices_cpu: tuple[int, ...],
        *,
        state_reference: Tensor,
        batch_size: int,
        roots_allowed: bool = True,
    ) -> tuple[Tensor, Tensor]:
        sources_by_chunk: dict[int, list[tuple[int, int]]] = {}
        missing_parents: list[int] = []
        for dest_row, parent_index in enumerate(parent_indices_cpu):
            if parent_index < 0:
                if roots_allowed:
                    continue
                missing_parents.append(parent_index)
                continue
            source = (
                self._source_by_family[parent_index]
                if parent_index < len(self._source_by_family)
                else None
            )
            if source is None:
                missing_parents.append(parent_index)
                continue
            chunk_index, source_row = source
            sources_by_chunk.setdefault(chunk_index, []).append((dest_row, source_row))
        if missing_parents:
            raise RuntimeError(
                "tree GDN append-only execution is missing parent state for "
                f"families {tuple(missing_parents)}"
            )

        single_source_chunk = next(iter(sources_by_chunk.values()))
        if len(sources_by_chunk) == 1 and len(single_source_chunk) == batch_size:
            chunk_index, pairs = next(iter(sources_by_chunk.items()))
            return (
                _select_state_rows(self._conv_chunks[chunk_index], pairs),
                _select_state_rows(self._rec_chunks[chunk_index], pairs),
            )

        conv = _zero_conv_state(gdn, state_reference, batch_size=batch_size)
        rec = _zero_recurrent_state(gdn, state_reference, batch_size=batch_size)
        for chunk_index, pairs in sources_by_chunk.items():
            dest_rows = _long_tensor(
                (dest_row for dest_row, _ in pairs),
                device=self._device,
            )
            source_rows = _long_tensor(
                (source_row for _, source_row in pairs),
                device=self._device,
            )
            conv = conv.index_copy(
                0,
                dest_rows,
                self._conv_chunks[chunk_index].index_select(0, source_rows),
            )
            rec = rec.index_copy(
                0,
                dest_rows,
                self._rec_chunks[chunk_index].index_select(0, source_rows),
            )
        return conv, rec


def _select_state_rows(chunk: Tensor, pairs: Sequence[tuple[int, int]]) -> Tensor:
    source_rows = tuple(source_row for _, source_row in pairs)
    if len(set(source_rows)) == 1:
        return chunk.narrow(0, source_rows[0], 1).expand(
            len(source_rows),
            *tuple(chunk.shape[1:]),
        )
    first_row = source_rows[0]
    if source_rows == tuple(range(first_row, first_row + len(source_rows))):
        return chunk.narrow(0, first_row, len(source_rows))
    return chunk.index_select(
        0,
        _long_tensor(source_rows, device=chunk.device),
    )


def _bucket_family_indices_cpu(bucket: GdnSegmentBucketPlan) -> tuple[int, ...]:
    family_indices = bucket.family_indices_cpu
    if family_indices is None:
        family_indices = bucket.family_indices.detach().cpu()
    return tuple(int(index) for index in family_indices.tolist())


def _bucket_parent_indices_cpu(bucket: GdnSegmentBucketPlan) -> tuple[int, ...]:
    parent_indices = bucket.parent_indices
    if parent_indices is None:
        raise RuntimeError("tree GDN bucket is missing parent indices")
    parent_indices_cpu = bucket.parent_indices_cpu
    if parent_indices_cpu is None:
        parent_indices_cpu = parent_indices.detach().cpu()
    return tuple(int(index) for index in parent_indices_cpu.tolist())


def _long_tensor(values: Iterable[int], *, device: torch.device) -> Tensor:
    return torch.tensor(tuple(values), dtype=torch.long, device=device)


def _bucket_has_parent_state(bucket: GdnSegmentBucketPlan) -> bool:
    return any(parent_index >= 0 for parent_index in _bucket_parent_indices_cpu(bucket))


def _bucket_has_uniform_lengths(bucket: GdnSegmentBucketPlan) -> bool:
    lengths_cpu = bucket.lengths_cpu
    if lengths_cpu is None:
        lengths_cpu = bucket.lengths.detach().cpu()
    return all(int(length) == int(bucket.length) for length in lengths_cpu.tolist())


def _run_cp_planned_prefixes_and_completions(
    gdn: Any,
    hidden_states: Tensor,
    plan: GdnRankExecutionPlan,
    *,
    group: Any,
    input_layout: Literal["attention", "gdn"],
    output_layout: Literal["attention", "gdn"],
) -> tuple[Tensor, Tensor | None]:
    if plan.attention_to_gdn is None or plan.gdn_to_attention is None:
        raise ValueError("CP GDN execution requires prebuilt exchange plans")
    if input_layout not in ("attention", "gdn") or output_layout not in (
        "attention",
        "gdn",
    ):
        raise ValueError(
            f"unsupported GDN CP layouts: {input_layout=} {output_layout=}"
        )
    if input_layout == "attention":
        gdn_hidden, _original_shape = gdn_cp_attention_to_gdn_layout(
            hidden_states,
            plan,
            group,
            gdn=gdn,
        )
    else:
        gdn_hidden = _validate_gdn_hidden_for_cp_plan(hidden_states, plan, gdn=gdn)
    empty_gdn_rank = plan.gdn_token_count == 0
    if empty_gdn_rank:
        qkv, gate, beta, recurrent_g = _project_empty_gdn_inputs(gdn, gdn_hidden)
    else:
        qkv, gate, beta, recurrent_g = _project_gdn_inputs(gdn, gdn_hidden)
    cp_dependency = (
        _make_zero_autograd_dependency(gdn_hidden)
        if empty_gdn_rank
        else _empty_autograd_dependency(qkv)
    )
    if not plan.tree_segment_buckets_by_depth:
        raise ValueError("CP shared-prefix GDN requires a tree execution plan")
    gate = gate.clone()
    recurrent_output = torch.zeros_like(gate)
    recurrent_output, cp_dependency = _run_tree_depth_buckets(
        gdn,
        qkv,
        beta,
        recurrent_g,
        recurrent_output,
        plan,
        state_reference=qkv,
        group=group,
        cp_dependency=cp_dependency,
    )
    projected, out_bias = _project_cp_gdn_output(
        gdn,
        recurrent_output,
        gate,
        plan,
        group=group,
        output_layout=output_layout,
        dependency=cp_dependency,
    )
    return projected, out_bias


@torch.compiler.disable
def gdn_cp_attention_to_gdn_layout(
    hidden_states: Tensor,
    plan: GdnRankExecutionPlan,
    group: Any,
    gdn: Any | None = None,
) -> tuple[Tensor, tuple[int, int, int]]:
    from .layout import exchange_rank_tensor_all_to_all

    if plan.attention_to_gdn is None or plan.gdn_to_attention is None:
        raise ValueError("CP GDN layout conversion requires prebuilt exchange plans")
    exchange_plan, backward_plan, rank, group = _hidden_layout_exchange_context(
        plan,
        gdn=gdn,
        group=group,
        forward_plan=plan.attention_to_gdn,
        backward_plan=plan.gdn_to_attention,
    )
    attention_flat, original_shape = _flatten_hidden_for_exchange_plan(
        hidden_states, exchange_plan, rank=rank
    )
    gdn_flat = exchange_rank_tensor_all_to_all(
        attention_flat,
        exchange_plan,
        rank=rank,
        group=group,
        backward_plan=backward_plan,
    )
    return gdn_flat.unsqueeze(1).contiguous(), original_shape


@torch.compiler.disable
def gdn_cp_gdn_to_attention_layout(
    gdn_hidden: Tensor,
    plan: GdnRankExecutionPlan,
    original_shape: tuple[int, int, int] | None,
    group: Any,
    gdn: Any | None = None,
) -> Tensor:
    if original_shape is None:
        raise RuntimeError("GDN CP output layout conversion requires original_shape")
    return _cp_output_to_attention(gdn_hidden, plan, original_shape, group, gdn=gdn)


def _normalize_cp_layout(value: Any) -> Literal["attention", "gdn"]:
    if value in ("attention", "gdn"):
        return cast(Literal["attention", "gdn"], value)
    raise ValueError(f"unsupported GDN CP layout {value!r}")


def _enter_gdn_island_layout(
    hidden_states: Tensor,
    attention_bias: Any,
    *,
    gdn: Any | None = None,
    island_id: int | None = None,
    force: bool = False,
) -> Tensor:
    plan = _require_gdn_cp_plan(attention_bias)
    if not force and getattr(attention_bias, "gdn_hidden_layout", "attention") == "gdn":
        return _validate_gdn_hidden_for_cp_plan(hidden_states, plan, gdn=gdn)
    gdn_hidden, original_shape = gdn_cp_attention_to_gdn_layout(
        hidden_states,
        plan,
        _default_cp_group(plan.cp_size),
        gdn=gdn,
    )
    attention_bias.gdn_hidden_layout = "gdn"
    _store_gdn_attention_original_shape(
        attention_bias, original_shape, gdn=gdn, island_id=island_id
    )
    if gdn is not None:
        attention_bias.gdn_active_module = gdn
    token_uids = (
        _local_layout_token_uids(plan, "gdn", hidden_states=gdn_hidden, gdn=gdn)
        if _layout_token_uids_enabled()
        else None
    )
    _set_active_routing_replay_layout("gdn")
    return _attach_gdn_attention_original_shape(
        _attach_trace_token_uids(gdn_hidden, token_uids),
        original_shape,
    )


def _mark_cp_layout_active(
    attention_bias: Any,
    hidden_states: Tensor | None,
    *,
    gdn: Any | None,
    island_id: int | None = None,
    layout: Literal["attention", "gdn"],
) -> None:
    if layout == "gdn":
        _mark_gdn_layout_active(
            attention_bias, hidden_states, gdn=gdn, island_id=island_id
        )
    else:
        _mark_attention_layout_active(attention_bias, hidden_states, gdn=gdn)


def _mark_attention_layout_active(
    attention_bias: Any,
    hidden_states: Tensor | None = None,
    *,
    gdn: Any | None = None,
) -> None:
    attention_bias.gdn_hidden_layout = "attention"
    attention_bias.gdn_attention_original_shape = None
    attention_bias.gdn_attention_token_uids = None
    attention_bias.gdn_active_module = None
    if hidden_states is None:
        return
    plan = _require_gdn_cp_plan(attention_bias)
    token_uids = (
        _local_layout_token_uids(
            plan, "attention", hidden_states=hidden_states, gdn=gdn
        )
        if _layout_token_uids_enabled()
        else None
    )
    _set_active_routing_replay_layout("attention")
    _attach_trace_token_uids(hidden_states, token_uids)


def _mark_gdn_layout_active(
    attention_bias: Any,
    hidden_states: Tensor | None,
    *,
    gdn: Any | None = None,
    island_id: int | None = None,
) -> None:
    plan = _require_gdn_cp_plan(attention_bias)
    attention_bias.gdn_hidden_layout = "gdn"
    if gdn is not None:
        attention_bias.gdn_active_module = gdn
    if hidden_states is None:
        return
    original_shape = _gdn_attention_original_shape_from_tensor(hidden_states)
    if original_shape is not None:
        _store_gdn_attention_original_shape(
            attention_bias, original_shape, gdn=gdn, island_id=island_id
        )
    gdn_token_uids = (
        _local_layout_token_uids(plan, "gdn", hidden_states=hidden_states, gdn=gdn)
        if _layout_token_uids_enabled()
        else None
    )
    _set_active_routing_replay_layout("gdn")
    _attach_trace_token_uids(hidden_states, gdn_token_uids)


def _leave_gdn_island_layout(
    hidden_states: Tensor,
    attention_bias: Any,
    *,
    gdn: Any | None = None,
    island_id: int | None = None,
) -> Tensor:
    plan = _require_gdn_cp_plan(attention_bias)
    gdn_hidden = _validate_gdn_hidden_for_cp_plan(hidden_states, plan, gdn=gdn)
    original_shape = _gdn_attention_original_shape_from_state(
        attention_bias, gdn=gdn, island_id=island_id
    )
    if original_shape is None:
        original_shape = _gdn_attention_original_shape_from_tensor(hidden_states)
        if original_shape is not None:
            _store_gdn_attention_original_shape(
                attention_bias, original_shape, gdn=gdn, island_id=island_id
            )
    attention_hidden = gdn_cp_gdn_to_attention_layout(
        gdn_hidden,
        plan,
        original_shape,
        _default_cp_group(plan.cp_size),
        gdn=gdn,
    )
    _mark_attention_layout_active(attention_bias)
    token_uids = (
        _local_layout_token_uids(
            plan, "attention", hidden_states=attention_hidden, gdn=gdn
        )
        if _layout_token_uids_enabled()
        else None
    )
    _set_active_routing_replay_layout("attention")
    return _attach_trace_token_uids(attention_hidden, token_uids)


def _require_gdn_cp_plan(attention_bias: Any) -> GdnRankExecutionPlan:
    plan = getattr(attention_bias, "gdn_execution_plan", None)
    if plan is None or int(getattr(plan, "cp_size", 1)) <= 1:
        raise ValueError("GDN island layout conversion requires a CP execution plan")
    return cast(GdnRankExecutionPlan, plan)


def _cp_output_to_attention(
    gdn_output: Tensor,
    plan: GdnRankExecutionPlan,
    original_shape: tuple[int, int, int],
    group: Any,
    *,
    gdn: Any | None = None,
) -> Tensor:
    from .layout import exchange_rank_tensor_all_to_all

    if plan.gdn_to_attention is None:
        raise ValueError("CP GDN execution requires a GDN-to-attention exchange plan")
    if plan.attention_to_gdn is None:
        raise ValueError("CP GDN execution requires an attention-to-GDN backward plan")
    exchange_plan, backward_plan, rank, group = _hidden_layout_exchange_context(
        plan,
        gdn=gdn,
        group=group,
        forward_plan=plan.gdn_to_attention,
        backward_plan=plan.attention_to_gdn,
    )
    gdn_flat, _ = _flatten_hidden_for_exchange_plan(
        gdn_output, exchange_plan, rank=rank
    )
    attention_flat = exchange_rank_tensor_all_to_all(
        gdn_flat,
        exchange_plan,
        rank=rank,
        group=group,
        backward_plan=backward_plan,
    )
    return _restore_hidden_from_cp_flat(attention_flat, original_shape)


def _hidden_layout_exchange_context(
    plan: GdnRankExecutionPlan,
    *,
    gdn: Any | None,
    group: Any,
    forward_plan: Any,
    backward_plan: Any,
) -> tuple[Any, Any, int, Any]:
    projection = _gdn_output_projection(gdn) or _gdn_input_projection(gdn)
    if projection is None or not _uses_sequence_parallel(projection):
        return forward_plan, backward_plan, int(plan.cp_rank), group
    from .layout import shard_cp_exchange_plan_for_sequence_parallel

    tp_size = _tp_world_size(projection)
    tp_rank = _tp_rank(projection)
    tp_cp_group = _default_tp_cp_group(plan.cp_size, tp_size)
    tp_cp_rank = _group_rank(tp_cp_group)
    sharded_forward = shard_cp_exchange_plan_for_sequence_parallel(
        forward_plan,
        cp_rank=int(plan.cp_rank),
        tp_rank=tp_rank,
        tp_size=tp_size,
        tp_cp_rank=tp_cp_rank,
        device=_exchange_plan_device(forward_plan),
    )
    sharded_backward = shard_cp_exchange_plan_for_sequence_parallel(
        backward_plan,
        cp_rank=int(plan.cp_rank),
        tp_rank=tp_rank,
        tp_size=tp_size,
        tp_cp_rank=tp_cp_rank,
        device=_exchange_plan_device(backward_plan),
    )
    return (
        sharded_forward.plan,
        sharded_backward.plan,
        sharded_forward.rank,
        tp_cp_group,
    )


def _flatten_hidden_for_exchange_plan(
    hidden_states: Tensor, plan: Any, *, rank: int
) -> tuple[Tensor, tuple[int, int, int]]:
    seq_len, batch_size, hidden_size = hidden_states.shape
    flat = hidden_states.transpose(0, 1).reshape(seq_len * batch_size, hidden_size)
    expected = int(plan.source_token_counts_by_rank[rank])
    if int(flat.shape[0]) < expected:
        raise ValueError(
            "CP GDN hidden token count must match the exchange source layout, "
            f"got {int(flat.shape[0])} tokens and expected {expected}"
        )
    return flat[:expected].contiguous(), (seq_len, batch_size, hidden_size)


def _exchange_plan_device(plan: Any) -> torch.device | str | None:
    for transfer in getattr(plan, "transfers", ()):
        for tensor in (
            getattr(transfer, "source_positions_tensor", None),
            getattr(transfer, "dest_positions_tensor", None),
        ):
            if isinstance(tensor, Tensor):
                return tensor.device
    return None


def _hidden_token_count(hidden_states: Tensor) -> int:
    if hidden_states.ndim < 2:
        return 0
    return int(hidden_states.shape[0]) * int(hidden_states.shape[1])


def _layout_token_uids(
    plan: GdnRankExecutionPlan, layout: Literal["attention", "gdn"]
) -> Tensor:
    indices = (
        plan.gdn_token_indices if layout == "gdn" else plan.attention_token_indices
    )
    return torch.tensor(indices, dtype=torch.int64)


def _trace_token_uids_enabled() -> bool:
    return _GDN_TRACE_TOKEN_UID_HOOKS is not None


def _local_layout_token_uids(
    plan: GdnRankExecutionPlan,
    layout: Literal["attention", "gdn"],
    *,
    hidden_states: Tensor,
    gdn: Any | None,
) -> Tensor:
    token_uids = _layout_token_uids(plan, layout)
    token_count = _hidden_token_count(hidden_states)
    if token_count == int(token_uids.numel()):
        return token_uids
    if token_count <= 0:
        return token_uids.new_empty((0,))
    projection = _gdn_output_projection(gdn)
    tp_rank = _tp_rank(projection) if projection is not None else 0
    start = tp_rank * token_count
    end = min(start + token_count, int(token_uids.numel()))
    local_uids = token_uids.new_full((token_count,), -1)
    if start >= int(token_uids.numel()):
        return local_uids
    real_uids = token_uids[start:end]
    local_uids[: int(real_uids.numel())] = real_uids
    return local_uids


def _replicated_layout_token_uids(
    plan: GdnRankExecutionPlan,
    layout: Literal["attention", "gdn"],
    *,
    hidden_states: Tensor,
) -> Tensor:
    token_uids = _layout_token_uids(plan, layout)
    token_count = _hidden_token_count(hidden_states)
    if token_count == int(token_uids.numel()):
        return token_uids
    if token_count <= 0:
        return token_uids.new_empty((0,))
    local_uids = token_uids.new_full((token_count,), -1)
    real_uids = token_uids[: min(token_count, int(token_uids.numel()))]
    local_uids[: int(real_uids.numel())] = real_uids
    return local_uids


def _attach_trace_token_uids(tensor: Tensor, token_uids: Tensor | None) -> Tensor:
    hooks = _GDN_TRACE_TOKEN_UID_HOOKS
    if hooks is None or token_uids is None:
        return tensor
    attach = getattr(hooks, "attach_token_uids", None)
    return tensor if attach is None else cast(Tensor, attach(tensor, token_uids))


def _prepare_in_proj_trace_token_uids(gdn: Any, hidden_states: Tensor) -> None:
    hooks = _GDN_TRACE_TOKEN_UID_HOOKS
    if hooks is None:
        return
    prepare = getattr(hooks, "prepare_in_proj_token_uids", None)
    if prepare is not None:
        prepare(gdn, hidden_states)


def _set_out_proj_lora_trace_token_uids(gdn: Any, hidden_states: Tensor) -> None:
    hooks = _GDN_TRACE_TOKEN_UID_HOOKS
    if hooks is None:
        return
    setter = getattr(hooks, "set_out_proj_lora_token_uids", None)
    if setter is not None:
        setter(gdn, hidden_states)


def _set_out_norm_trace_token_uids(gdn: Any, token_uids: Tensor | None) -> None:
    hooks = _GDN_TRACE_TOKEN_UID_HOOKS
    if hooks is None or token_uids is None:
        return
    setter = getattr(hooks, "set_out_norm_token_uids", None)
    if setter is not None:
        setter(gdn, token_uids)


def _set_out_proj_trace_token_uids(
    gdn: Any,
    hidden_states: Tensor,
    *,
    sequence_parallel_output: bool,
) -> None:
    hooks = _GDN_TRACE_TOKEN_UID_HOOKS
    if hooks is None:
        return
    setter = getattr(hooks, "set_out_proj_token_uids", None)
    if setter is not None:
        setter(
            gdn,
            hidden_states,
            sequence_parallel_output=sequence_parallel_output,
        )


def _pad_trace_token_uids_for_stream(token_uids: Tensor, stream: Tensor) -> Tensor:
    hooks = _GDN_TRACE_TOKEN_UID_HOOKS
    pad = None if hooks is None else getattr(hooks, "pad_token_uids_for_stream", None)
    if pad is not None:
        return cast(Tensor, pad(token_uids, stream))
    return token_uids


def _attach_gdn_attention_original_shape(
    tensor: Tensor, original_shape: tuple[int, int, int] | None
) -> Tensor:
    if original_shape is not None:
        setattr(
            tensor,
            _GDN_ATTENTION_ORIGINAL_SHAPE_ATTR,
            tuple(int(dim) for dim in original_shape),
        )
    return tensor


def _store_gdn_attention_original_shape(
    attention_bias: Any,
    original_shape: tuple[int, int, int],
    *,
    gdn: Any | None,
    island_id: int | None = None,
) -> tuple[int, int, int]:
    normalized = (
        int(original_shape[0]),
        int(original_shape[1]),
        int(original_shape[2]),
    )
    attention_bias.gdn_attention_original_shape = normalized
    cache = _gdn_attention_original_shape_cache(attention_bias)
    cache[_gdn_attention_original_shape_cache_key(gdn)] = normalized
    if island_id is not None:
        cache[_gdn_attention_original_shape_cache_key(None, island_id)] = normalized
    return normalized


def _gdn_attention_original_shape_from_state(
    attention_bias: Any,
    *,
    gdn: Any | None,
    island_id: int | None = None,
) -> tuple[int, int, int] | None:
    cache = getattr(attention_bias, "gdn_attention_original_shapes", None)
    if isinstance(cache, dict):
        if island_id is not None:
            original_shape = _normalize_gdn_attention_original_shape(
                cache.get(_gdn_attention_original_shape_cache_key(None, island_id))
            )
            if original_shape is not None:
                return original_shape
        if gdn is not None:
            original_shape = _normalize_gdn_attention_original_shape(
                cache.get(_gdn_attention_original_shape_cache_key(gdn))
            )
            if original_shape is not None:
                return original_shape
        active_gdn = getattr(attention_bias, "gdn_active_module", None)
        if active_gdn is not None:
            original_shape = _normalize_gdn_attention_original_shape(
                cache.get(_gdn_attention_original_shape_cache_key(active_gdn))
            )
            if original_shape is not None:
                return original_shape
        if gdn is None:
            original_shape = _normalize_gdn_attention_original_shape(
                cache.get(_gdn_attention_original_shape_cache_key(None))
            )
            if original_shape is not None:
                return original_shape
    original_shape = _normalize_gdn_attention_original_shape(
        getattr(attention_bias, "gdn_attention_original_shape", None)
    )
    active_gdn = getattr(attention_bias, "gdn_active_module", None)
    if original_shape is None or (
        gdn is not None and active_gdn is not None and active_gdn is not gdn
    ):
        return None
    return original_shape


def _gdn_attention_original_shape_cache(
    attention_bias: Any,
) -> dict[int, tuple[int, int, int]]:
    cache = getattr(attention_bias, "gdn_attention_original_shapes", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(attention_bias, "gdn_attention_original_shapes", cache)
    return cast(dict[int, tuple[int, int, int]], cache)


def _gdn_attention_original_shape_cache_key(
    gdn: Any | None, island_id: int | None = None
) -> int:
    if island_id is not None:
        return -int(island_id) - 1
    return 0 if gdn is None else id(gdn)


def _normalize_gdn_attention_original_shape(
    original_shape: Any,
) -> tuple[int, int, int] | None:
    if not isinstance(original_shape, tuple) or len(original_shape) != 3:
        return None
    return (int(original_shape[0]), int(original_shape[1]), int(original_shape[2]))


def _gdn_attention_original_shape_from_tensor(
    tensor: Tensor,
) -> tuple[int, int, int] | None:
    original_shape = getattr(tensor, _GDN_ATTENTION_ORIGINAL_SHAPE_ATTR, None)
    return _normalize_gdn_attention_original_shape(original_shape)


def _active_routing_replay_controller() -> Any | None:
    try:
        from art.megatron.routing_replay import _active_routing_replay_controller
    except ImportError:
        return None
    return _active_routing_replay_controller()


def _layout_token_uids_enabled() -> bool:
    return (
        _trace_token_uids_enabled() or _active_routing_replay_controller() is not None
    )


def _set_active_routing_replay_layout(
    layout: Literal["attention", "gdn"],
) -> None:
    controller = _active_routing_replay_controller()
    if controller is None:
        return
    controller.set_active_token_uid_key(layout)


def _validate_gdn_hidden_for_cp_plan(
    hidden_states: Tensor, plan: GdnRankExecutionPlan, *, gdn: Any | None = None
) -> Tensor:
    expected = _local_layout_token_count_for_hidden(
        plan, "gdn", hidden_states=hidden_states, gdn=gdn
    )
    if hidden_states.ndim != 3 or int(hidden_states.shape[0]) != expected:
        raise ValueError(
            "CP GDN-layout hidden_states must be [rank_gdn_tokens, 1, D], "
            f"got {tuple(hidden_states.shape)} for {expected} planned tokens"
        )
    if int(hidden_states.shape[1]) != 1:
        raise ValueError(
            "CP GDN-layout hidden_states must use a flattened local batch, "
            f"got batch dimension {int(hidden_states.shape[1])}"
        )
    return hidden_states.contiguous()


def _local_layout_token_count_for_hidden(
    plan: GdnRankExecutionPlan,
    layout: Literal["attention", "gdn"],
    *,
    hidden_states: Tensor,
    gdn: Any | None,
) -> int:
    del hidden_states
    real_count = (
        int(plan.gdn_token_count)
        if layout == "gdn"
        else int(plan.attention_token_count)
    )
    projection = _gdn_output_projection(gdn) or _gdn_input_projection(gdn)
    if projection is None or not _uses_sequence_parallel(projection):
        return real_count
    return (real_count + _tp_world_size(projection) - 1) // _tp_world_size(projection)


def _restore_hidden_from_cp_flat(
    flat: Tensor, original_shape: tuple[int, int, int]
) -> Tensor:
    seq_len, batch_size, hidden_size = original_shape
    token_count = seq_len * batch_size
    if int(flat.shape[0]) > token_count:
        raise ValueError(
            "CP GDN output token count changed across layout exchange, got "
            f"{int(flat.shape[0])} for original shape {original_shape}"
        )
    if int(flat.shape[0]) < token_count:
        padded = flat.new_zeros((token_count, hidden_size))
        if int(flat.shape[0]) > 0:
            padded[: int(flat.shape[0])] = flat
        flat = padded
    return flat.reshape(batch_size, seq_len, hidden_size).transpose(0, 1).contiguous()


def _empty_autograd_dependency(reference: Tensor) -> Tensor:
    return reference.new_zeros(())


def _make_autograd_dependency(*tensors: Tensor | None) -> Tensor:
    dependency: Tensor | None = None
    for tensor in tensors:
        if tensor is None or int(tensor.numel()) == 0:
            continue
        piece = tensor.reshape(-1)[:1].sum() * 0
        dependency = piece if dependency is None else dependency + piece
    if dependency is None:
        raise ValueError("at least one non-empty tensor is required")
    return dependency


def _make_zero_autograd_dependency(*tensors: Tensor) -> Tensor:
    if not tensors:
        raise ValueError("at least one tensor is required")
    dependency = tensors[0].sum() * 0
    for tensor in tensors[1:]:
        dependency = dependency + tensor.sum() * 0
    return dependency


def _add_autograd_dependency(tensor: Tensor, dependency: Tensor) -> Tensor:
    return tensor + dependency.to(dtype=tensor.dtype)


def _couple_parent_states(
    conv_state: Tensor, recurrent_state: Tensor
) -> tuple[Tensor, Tensor]:
    return _CoupledParentStates.apply(conv_state, recurrent_state)


class _CoupledParentStates(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, conv_state: Tensor, recurrent_state: Tensor
    ) -> tuple[Tensor, Tensor]:
        del ctx
        return conv_state, recurrent_state

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Tensor | None
    ) -> tuple[Tensor | None, Tensor | None]:
        del ctx
        grad_conv, grad_recurrent = grad_outputs
        return grad_conv, grad_recurrent


def _scale_state_gradient(tensor: Tensor, scale: float) -> Tensor:
    return _ScaleStateGradient.apply(tensor, scale)


class _ScaleStateGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tensor: Tensor, scale: float) -> Tensor:
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor | None) -> tuple[Tensor | None, None]:
        (grad_output,) = grad_outputs
        if grad_output is None:
            return None, None
        return grad_output * ctx.scale, None


def _project_gdn_inputs(
    gdn: Any,
    hidden_states: Tensor,
    *,
    sequence_parallel_input: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    seq_len, batch_size, _ = hidden_states.shape
    if sequence_parallel_input:
        seq_len *= int(getattr(gdn, "sp_size", 1))
    qkvzba, _ = _in_proj(
        gdn,
        hidden_states,
        sequence_parallel_input=sequence_parallel_input,
    )
    qkvzba = qkvzba.transpose(0, 1)
    if int(qkvzba.shape[0]) != batch_size:
        raise ValueError(
            "GDN input projection changed the packed batch dimension, "
            f"got {int(qkvzba.shape[0])} and expected {batch_size}"
        )
    qkv, gate, beta, alpha = torch.split(
        qkvzba,
        [
            (gdn.qk_dim * 2 + gdn.v_dim) // gdn.tp_size,
            gdn.v_dim // gdn.tp_size,
            gdn.num_value_heads // gdn.tp_size,
            gdn.num_value_heads // gdn.tp_size,
        ],
        dim=-1,
    )
    value_heads = _local_value_heads(gdn)
    gate = gate.reshape(
        batch_size, seq_len, value_heads, gdn.value_head_dim
    ).contiguous()
    beta = beta.reshape(batch_size, seq_len, value_heads).sigmoid().contiguous()
    alpha = alpha.reshape(batch_size, seq_len, value_heads)
    recurrent_g = (
        -gdn.A_log.exp() * F.softplus(alpha.float() + gdn.dt_bias)
    ).contiguous()
    return qkv.contiguous(), gate, beta, recurrent_g


def _project_empty_gdn_inputs(
    gdn: Any,
    hidden_states: Tensor,
    *,
    sequence_parallel_input: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    seq_len, batch_size, _ = hidden_states.shape
    if sequence_parallel_input:
        seq_len *= int(getattr(gdn, "sp_size", 1))
    value_heads = _local_value_heads(gdn)
    qkv_width = (gdn.qk_dim * 2 + gdn.v_dim) // gdn.tp_size
    dependency = hidden_states.sum() * 0
    qkv = hidden_states.new_zeros((batch_size, seq_len, qkv_width)) + dependency
    gate = (
        hidden_states.new_zeros((batch_size, seq_len, value_heads, gdn.value_head_dim))
        + dependency
    )
    beta = hidden_states.new_zeros((batch_size, seq_len, value_heads)) + dependency
    recurrent_g = (
        hidden_states.new_zeros((batch_size, seq_len, value_heads)) + dependency
    )
    return (
        qkv.contiguous(),
        gate.contiguous(),
        beta.contiguous(),
        recurrent_g.contiguous(),
    )


def _in_proj(
    gdn: Any,
    hidden_states: Tensor,
    *,
    sequence_parallel_input: bool = True,
) -> tuple[Tensor, Tensor | None]:
    del sequence_parallel_input
    _prepare_in_proj_trace_token_uids(gdn, hidden_states)
    return gdn.in_proj(hidden_states)


def _gather_bucket_streams(
    qkv: Tensor,
    beta: Tensor,
    recurrent_g: Tensor,
    bucket: GdnSegmentBucketPlan,
) -> tuple[Tensor, Tensor, Tensor]:
    return _gather_bucket_streams_compact_fused(
        qkv.reshape(-1, int(qkv.shape[-1])),
        beta.reshape(-1, int(beta.shape[-1])),
        recurrent_g.reshape(-1, int(recurrent_g.shape[-1])),
        bucket.row_indices,
        bucket.position_indices,
        bucket.cu_seqlens,
        token_count=int(bucket.real_token_count),
        segment_count=int(bucket.segment_count),
        sequence_length=int(qkv.shape[1]),
    )


def _project_gdn_output(
    gdn: Any,
    recurrent_output: Tensor,
    gate: Tensor,
    plan: GdnRankExecutionPlan,
    *,
    sequence_parallel_output: bool = True,
    reduce_tensor_parallel_output: bool = True,
) -> tuple[Tensor, Tensor | None]:
    batch_size, seq_len, _, _ = recurrent_output.shape
    token_uids = (
        _replicated_layout_token_uids(plan, "gdn", hidden_states=recurrent_output)
        if _trace_token_uids_enabled()
        else None
    )
    _set_out_norm_trace_token_uids(gdn, token_uids)
    norm_out = _apply_gated_rms_norm(gdn, recurrent_output, gate)
    norm_out = norm_out.reshape(batch_size, seq_len, _local_value_dim(gdn))
    norm_out = norm_out.transpose(0, 1).contiguous()
    if token_uids is not None:
        token_uids = _replicated_layout_token_uids(plan, "gdn", hidden_states=norm_out)
    _attach_trace_token_uids(norm_out, token_uids)
    out, out_bias = _out_proj(
        gdn,
        norm_out,
        sequence_parallel_output=sequence_parallel_output,
        reduce_tensor_parallel_output=reduce_tensor_parallel_output,
    )
    return _mask_gdn_output(gdn, out, plan), out_bias


def _mask_gdn_output(gdn: Any, out: Tensor, plan: GdnRankExecutionPlan) -> Tensor:
    real_mask = plan.real_token_mask.transpose(0, 1).unsqueeze(-1)
    if tuple(real_mask.shape[:2]) == tuple(out.shape[:2]):
        return out.masked_fill(~real_mask, 0)
    full_batch = int(plan.packed_batch_size or plan.batch_size)
    full_seq = int(plan.packed_sequence_length or plan.sequence_length)
    full_count = full_batch * full_seq
    local_indices = torch.tensor(
        plan.gdn_token_indices, device=out.device, dtype=torch.long
    )
    full_flat = torch.zeros(full_count, device=out.device, dtype=torch.bool)
    if int(local_indices.numel()):
        full_flat = full_flat.index_fill(0, local_indices, True)
    full_mask = full_flat.reshape(full_batch, full_seq).transpose(0, 1).unsqueeze(-1)
    if tuple(full_mask.shape[:2]) == tuple(out.shape[:2]):
        return out.masked_fill(~full_mask, 0)
    projection = _gdn_output_projection(gdn)
    rank = _tp_rank(projection) if projection is not None else 0
    start = rank * int(out.shape[0])
    end = start + int(out.shape[0])
    if end <= int(full_mask.shape[0]) and int(full_mask.shape[1]) == int(out.shape[1]):
        return out.masked_fill(~full_mask[start:end], 0)
    raise ValueError(
        "GDN output mask shape must match projected output, got "
        f"mask={tuple(real_mask.shape)} full_mask={tuple(full_mask.shape)} "
        f"out={tuple(out.shape)}"
    )


def _project_cp_gdn_output(
    gdn: Any,
    recurrent_output: Tensor,
    gate: Tensor,
    plan: GdnRankExecutionPlan,
    *,
    group: Any,
    output_layout: Literal["attention", "gdn"],
    dependency: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    batch_size, seq_len, _, _ = recurrent_output.shape
    token_uids = (
        _replicated_layout_token_uids(plan, "gdn", hidden_states=recurrent_output)
        if _trace_token_uids_enabled()
        else None
    )
    _set_out_norm_trace_token_uids(gdn, token_uids)
    norm_out = _apply_gated_rms_norm(gdn, recurrent_output, gate)
    norm_out = norm_out.reshape(batch_size, seq_len, _local_value_dim(gdn))
    norm_out = norm_out.transpose(0, 1).contiguous()
    if dependency is not None:
        norm_out = _add_autograd_dependency(norm_out, dependency)
    if token_uids is not None:
        token_uids = _replicated_layout_token_uids(plan, "gdn", hidden_states=norm_out)
    _attach_trace_token_uids(norm_out, token_uids)
    if output_layout == "attention":
        norm_out = _exchange_cp_sequence_stream(
            norm_out,
            plan=plan,
            group=group,
            source_layout="gdn",
            dest_layout="attention",
        )
        if token_uids is not None:
            token_uids = _replicated_layout_token_uids(
                plan, "attention", hidden_states=norm_out
            )
        _attach_trace_token_uids(norm_out, token_uids)
    norm_out = _pad_sequence_parallel_output_stream(gdn, norm_out)
    if token_uids is not None:
        token_uids = _pad_trace_token_uids_for_stream(token_uids, norm_out)
    _attach_trace_token_uids(norm_out, token_uids)
    return _out_proj(gdn, norm_out)


def _pad_sequence_parallel_output_stream(gdn: Any, stream: Tensor) -> Tensor:
    projection = _gdn_output_projection(gdn)
    if projection is None or not _uses_sequence_parallel(projection):
        return stream
    tp_size = _tp_world_size(projection)
    remainder = int(stream.shape[0]) % tp_size
    if remainder == 0:
        return stream
    padding = stream.new_zeros((tp_size - remainder, *stream.shape[1:]))
    return torch.cat((stream, padding), dim=0).contiguous()


def _exchange_cp_sequence_stream(
    stream: Tensor,
    *,
    plan: GdnRankExecutionPlan,
    group: Any,
    source_layout: Literal["attention", "gdn"],
    dest_layout: Literal["attention", "gdn"],
) -> Tensor:
    return (
        _exchange_cp_batch_stream(
            stream.transpose(0, 1).contiguous(),
            plan=plan,
            group=group,
            source_layout=source_layout,
            dest_layout=dest_layout,
        )
        .transpose(0, 1)
        .contiguous()
    )


def _exchange_cp_batch_stream(
    stream: Tensor,
    *,
    plan: GdnRankExecutionPlan,
    group: Any,
    source_layout: Literal["attention", "gdn"],
    dest_layout: Literal["attention", "gdn"],
) -> Tensor:
    from .layout import exchange_rank_tensor_all_to_all

    if source_layout == dest_layout:
        return stream
    exchange_plan = (
        plan.attention_to_gdn if source_layout == "attention" else plan.gdn_to_attention
    )
    backward_plan = (
        plan.gdn_to_attention if source_layout == "attention" else plan.attention_to_gdn
    )
    if exchange_plan is None or backward_plan is None:
        raise ValueError("CP GDN stream exchange requires prebuilt exchange plans")
    source_tokens = (
        int(plan.attention_token_count)
        if source_layout == "attention"
        else int(plan.gdn_token_count)
    )
    dest_tokens = (
        int(plan.attention_token_count)
        if dest_layout == "attention"
        else int(plan.gdn_token_count)
    )
    feature_shape = tuple(stream.shape[2:])
    flat = stream.reshape(-1, *feature_shape)
    if int(flat.shape[0]) < source_tokens:
        raise ValueError(
            "CP GDN stream token count is smaller than the exchange source layout, "
            f"got {int(flat.shape[0])} and expected at least {source_tokens}"
        )
    exchanged = exchange_rank_tensor_all_to_all(
        flat[:source_tokens].contiguous(),
        exchange_plan,
        rank=plan.cp_rank,
        group=group,
        backward_plan=backward_plan,
    )
    return exchanged.reshape(1, dest_tokens, *feature_shape).contiguous()


def _apply_gated_rms_norm(gdn: Any, x: Tensor, gate: Tensor) -> Tensor:
    if x.dtype != torch.float32 and int(x.numel()) != 0:
        return gdn._apply_gated_norm(x, gate)
    x_dtype = x.dtype
    hidden = _apply_explicit_norm(
        gdn.out_norm,
        x.reshape(-1, int(x.shape[-1])),
        config=getattr(gdn, "config", None),
        weight_name="weight",
        bias_name="bias",
    )
    gate = gate.reshape(-1, int(gate.shape[-1]))
    return (hidden * gdn.act_fn(gate.float())).to(x_dtype)


def _out_proj(
    gdn: Any,
    hidden_states: Tensor,
    *,
    force_explicit: bool = False,
    sequence_parallel_output: bool = True,
    reduce_tensor_parallel_output: bool = True,
) -> tuple[Tensor, Tensor | None]:
    projection = gdn.out_proj
    _set_out_proj_trace_token_uids(
        gdn,
        hidden_states,
        sequence_parallel_output=sequence_parallel_output,
    )
    if (
        int(hidden_states.numel()) != 0
        and not force_explicit
        and reduce_tensor_parallel_output
        and hidden_states.dtype != torch.float32
    ):
        return projection(hidden_states)
    return _explicit_out_proj(
        gdn,
        hidden_states,
        sequence_parallel_output=sequence_parallel_output,
        reduce_tensor_parallel_output=reduce_tensor_parallel_output,
    )


def _explicit_out_proj(
    gdn: Any,
    hidden_states: Tensor,
    *,
    sequence_parallel_output: bool = True,
    reduce_tensor_parallel_output: bool = True,
) -> tuple[Tensor, Tensor | None]:
    projection = gdn.out_proj
    base_projection = getattr(projection, "linear_proj", projection)
    bias = _linear_bias(base_projection)
    out = _stable_fp32_linear(hidden_states, base_projection.weight, None)
    if reduce_tensor_parallel_output:
        out = _row_parallel_output(
            out, base_projection, sequence_parallel_output=sequence_parallel_output
        )
    if bias is not None and not _returns_bias(base_projection):
        out = out + bias
    if hasattr(projection, "lora"):
        _set_out_proj_lora_trace_token_uids(gdn, hidden_states)
        lora_output = projection.lora(hidden_states)
        if reduce_tensor_parallel_output and bool(
            getattr(projection, "reduce_output", True)
        ):
            lora_output = _row_parallel_output(
                lora_output,
                base_projection,
                sequence_parallel_output=sequence_parallel_output,
            )
        out = out + lora_output
    return out, bias if _returns_bias(base_projection) else None


def _stable_fp32_linear(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    if x.dtype != torch.float32:
        return F.linear(x, weight, bias)
    out = F.linear(
        x.to(dtype=torch.float64),
        weight.to(dtype=torch.float64),
        None if bias is None else bias.to(dtype=torch.float64),
    )
    return out.to(dtype=torch.float32)


def _apply_explicit_norm(
    module: Any,
    x: Tensor,
    *,
    config: Any,
    weight_name: str,
    bias_name: str,
) -> Tensor:
    weight = getattr(module, weight_name, None)
    if not isinstance(weight, Tensor):
        return x
    x_dtype = x.dtype
    x_float = x.float()
    eps = float(getattr(module, "eps", getattr(config, "layernorm_epsilon", 1e-5)))
    normalization = getattr(module, "normalization", None)
    if normalization is None and config is not None:
        normalization = getattr(config, "normalization", None)
    if normalization is None:
        module_name = type(module).__name__
        normalization = "LayerNorm" if module_name == "LayerNorm" else "RMSNorm"
    normalization = str(normalization)
    if normalization == "RMSNorm":
        normed = x_float * torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + eps
        )
    elif normalization == "LayerNorm":
        centered = x_float - x_float.mean(dim=-1, keepdim=True)
        normed = centered * torch.rsqrt(
            centered.square().mean(dim=-1, keepdim=True) + eps
        )
    else:
        raise ValueError(f"unsupported GDN normalization '{normalization}'")
    scale = weight.float()
    if bool(getattr(module, "zero_centered_gamma", False)):
        scale = scale + 1.0
    normed = normed * scale
    bias = getattr(module, bias_name, None)
    if isinstance(bias, Tensor):
        normed = normed + bias.float()
    return normed.to(dtype=x_dtype)


def _gdn_uses_sequence_parallel(gdn: Any | None) -> bool:
    return any(
        projection is not None and _uses_sequence_parallel(projection)
        for projection in (_gdn_input_projection(gdn), _gdn_output_projection(gdn))
    )


def _gdn_input_projection(gdn: Any | None) -> Any | None:
    if gdn is None:
        return None
    projection = getattr(gdn, "in_proj", None)
    if projection is None:
        return None
    return getattr(projection, "in_proj", projection)


def _gdn_output_projection(gdn: Any | None) -> Any | None:
    if gdn is None:
        return None
    projection = getattr(gdn, "out_proj", None)
    if projection is None:
        return None
    return getattr(projection, "linear_proj", projection)


def _column_parallel_input(x: Tensor, projection: Any) -> Tensor:
    if not _uses_sequence_parallel(projection):
        return x
    from megatron.core.tensor_parallel.mappings import (
        gather_from_sequence_parallel_region,
    )

    return gather_from_sequence_parallel_region(x, group=_tp_group(projection))


def _row_parallel_output(
    x: Tensor, projection: Any, *, sequence_parallel_output: bool = True
) -> Tensor:
    if _tp_world_size(projection) <= 1:
        return x
    if _uses_sequence_parallel(projection) and sequence_parallel_output:
        from megatron.core.tensor_parallel.mappings import (
            reduce_scatter_to_sequence_parallel_region,
        )

        return reduce_scatter_to_sequence_parallel_region(
            x, group=_tp_group(projection)
        )
    from megatron.core.tensor_parallel.mappings import (
        reduce_from_tensor_model_parallel_region,
    )

    return reduce_from_tensor_model_parallel_region(x, group=_tp_group(projection))


def _uses_sequence_parallel(projection: Any) -> bool:
    return bool(getattr(projection, "sequence_parallel", False)) and (
        _tp_world_size(projection) > 1
    )


def _tp_world_size(projection: Any) -> int:
    group = _tp_group(projection)
    if group is not None and dist.is_initialized():  # ty: ignore[possibly-missing-attribute]
        return int(dist.get_world_size(group))  # ty: ignore[possibly-missing-attribute]
    return int(getattr(projection, "tp_size", 1))


def _tp_rank(projection: Any) -> int:
    group = _tp_group(projection)
    if group is not None and dist.is_initialized():  # ty: ignore[possibly-missing-attribute]
        return int(dist.get_rank(group))  # ty: ignore[possibly-missing-attribute]
    for name in ("tp_rank", "tensor_model_parallel_rank"):
        value = getattr(projection, name, None)
        if isinstance(value, int):
            return value
    return 0


def _tp_group(projection: Any) -> Any | None:
    return getattr(projection, "_tp_group", getattr(projection, "tp_group", None))


def _linear_bias(projection: Any) -> Tensor | None:
    bias = getattr(projection, "bias", None)
    if not isinstance(bias, Tensor) or int(bias.numel()) == 0:
        return None
    return bias


def _returns_bias(projection: Any) -> bool:
    return bool(getattr(projection, "te_return_bias", False))


def _local_key_heads(gdn: Any) -> int:
    return int(gdn.num_key_heads // gdn.tp_size)


def _local_value_heads(gdn: Any) -> int:
    return int(gdn.num_value_heads // gdn.tp_size)


def _local_value_dim(gdn: Any) -> int:
    return _local_value_heads(gdn) * int(gdn.value_head_dim)


def _prepare_dense_recurrent_inputs(
    qkv: Tensor,
    beta: Tensor,
    recurrent_g: Tensor,
    *,
    key_heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    key_channels = int(key_heads) * int(key_dim)
    value_channels = int(value_heads) * int(value_dim)
    query = qkv[..., :key_channels].reshape(*qkv.shape[:2], key_heads, key_dim)
    key = qkv[..., key_channels : 2 * key_channels].reshape(
        *qkv.shape[:2],
        key_heads,
        key_dim,
    )
    value = qkv[..., 2 * key_channels : 2 * key_channels + value_channels].reshape(
        *qkv.shape[:2],
        value_heads,
        value_dim,
    )
    repeat = int(value_heads) // int(key_heads)
    if repeat != 1:
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
    return query, key, value, beta, recurrent_g


def _scatter_bucket_recurrent_output(
    output: Tensor, bucket: GdnSegmentBucketPlan, bucket_output: Tensor
) -> Tensor:
    return _scatter_bucket_output_fused(
        output,
        bucket_output,
        bucket.row_indices,
        bucket.position_indices,
        _bucket_output_mask(bucket),
        bucket.cu_seqlens,
    )


def _bucket_output_mask(bucket: GdnSegmentBucketPlan) -> Tensor:
    output_mask = bucket.output_mask
    return bucket.real_mask if output_mask is None else output_mask


def run_gdn_bucket(
    bucket: GdnSegmentBucketPlan,
    projected_streams: tuple[Tensor, Tensor, Tensor],
    parent_states: tuple[Tensor, Tensor],
    *,
    gdn: Any,
    group: Any | None = None,
    recurrent_cp: bool = False,
    output_final_state: bool = True,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    _disable_reentrant_te_linear_transpose_cache(gdn)
    qkv, beta, recurrent_g = projected_streams
    conv_initial, recurrent_initial = parent_states
    token_count = int(qkv.shape[0]) if qkv.ndim == 2 else -1
    batch_size = int(bucket.segment_count)
    if qkv.ndim != 2:
        raise ValueError(
            "GDN bucket execution requires compact projected streams; "
            f"got qkv shape {tuple(qkv.shape)}"
        )
    if token_count != int(bucket.real_token_count):
        raise ValueError(
            "GDN packed varlen token count mismatch, got "
            f"qkv={tuple(qkv.shape)} and bucket tokens={bucket.real_token_count}"
        )
    if tuple(beta.shape[:1]) != (token_count,) or tuple(recurrent_g.shape) != tuple(
        beta.shape
    ):
        raise ValueError(
            "packed beta/recurrent_g must be [tokens, heads], got "
            f"{tuple(beta.shape)} and {tuple(recurrent_g.shape)}"
        )
    if int(conv_initial.shape[0]) != batch_size:
        raise ValueError(
            "conv_initial batch must match bucket segment count, got "
            f"{tuple(conv_initial.shape)} for {batch_size} segments"
        )
    if int(recurrent_initial.shape[0]) != batch_size:
        raise ValueError(
            "recurrent_initial batch must match bucket segment count, got "
            f"{tuple(recurrent_initial.shape)} for {batch_size} segments"
        )

    conv_output_final_state = output_final_state
    chain_conv_final: Tensor | None = None
    chain_gradient_dependency: Tensor | None = None
    if recurrent_cp:
        conv_initial, chain_conv_final, chain_gradient_dependency = (
            _chain_conv_initial_and_final(
                qkv,
                bucket.cu_seqlens_cpu,
                bucket.lengths_by_rank_cpu,
                conv_initial,
                group=group,
                output_final_state=output_final_state,
            )
        )
        conv_output_final_state = False

    qkv, conv_final = _causal_conv1d_packed_varlen_with_state(
        gdn,
        qkv,
        conv_initial,
        bucket.cu_seqlens,
        output_final_state=conv_output_final_state,
    )
    if recurrent_cp:
        conv_final = chain_conv_final

    dense_local_bucket = not recurrent_cp and _bucket_has_uniform_lengths(bucket)
    if dense_local_bucket:
        query, key, value, beta, recurrent_g = _prepare_dense_recurrent_inputs(
            qkv.reshape(batch_size, int(bucket.length), int(qkv.shape[-1])),
            beta.reshape(batch_size, int(bucket.length), int(beta.shape[-1])),
            recurrent_g.reshape(
                batch_size,
                int(bucket.length),
                int(recurrent_g.shape[-1]),
            ),
            key_heads=_local_key_heads(gdn),
            value_heads=_local_value_heads(gdn),
            key_dim=int(gdn.key_head_dim),
            value_dim=int(gdn.value_head_dim),
        )
    else:
        query, key, value, beta, recurrent_g = _prepare_packed_recurrent_inputs_fused(
            qkv,
            beta,
            recurrent_g,
            key_heads=_local_key_heads(gdn),
            value_heads=_local_value_heads(gdn),
            key_dim=int(gdn.key_head_dim),
            value_dim=int(gdn.value_head_dim),
        )
    if gdn.use_qk_l2norm:
        query = _l2norm(query.contiguous())
        key = _l2norm(key.contiguous())

    if recurrent_cp:
        if group is None:
            raise ValueError("CP recurrent GDN bucket requires a process group")
        recurrent_out, recurrent_final = chunk_gated_delta_rule_native_cp(
            query,
            key,
            value,
            g=recurrent_g,
            beta=beta,
            initial_state=recurrent_initial,
            group=group,
            output_final_state=output_final_state,
            cu_seqlens=bucket.cu_seqlens,
            cu_seqlens_cpu=bucket.cu_seqlens_cpu,
            lengths_by_rank_cpu=bucket.lengths_by_rank_cpu,
        )
    else:
        recurrent_out, recurrent_final = _chunk_gated_delta_rule(
            query,
            key,
            value,
            g=recurrent_g,
            beta=beta,
            initial_state=recurrent_initial,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=None if dense_local_bucket else bucket.cu_seqlens,
        )
        if dense_local_bucket:
            recurrent_out = recurrent_out.reshape(
                1,
                token_count,
                int(recurrent_out.shape[-2]),
                int(recurrent_out.shape[-1]),
            )
    if chain_gradient_dependency is not None:
        recurrent_out = _add_autograd_dependency(
            recurrent_out,
            chain_gradient_dependency,
        )
        if conv_final is not None:
            conv_final = _add_autograd_dependency(conv_final, chain_gradient_dependency)
        if recurrent_final is not None:
            recurrent_final = _add_autograd_dependency(
                recurrent_final,
                chain_gradient_dependency,
            )
    return recurrent_out, conv_final, recurrent_final


def _chain_conv_initial_and_final(
    qkv: Tensor,
    cu_seqlens_cpu: Tensor,
    lengths_by_rank_cpu: Tensor | None,
    parent_initial: Tensor,
    *,
    group: Any,
    output_final_state: bool,
) -> tuple[Tensor, Tensor | None, Tensor]:
    if group is None:
        raise ValueError("CP chain conv state requires a process group")
    if not dist.is_available() or not dist.is_initialized():  # ty: ignore[possibly-missing-attribute]
        raise RuntimeError("torch.distributed must be initialized for CP chain conv")
    parent_initial, gradient_dependency = _AllReduceGradient.apply(
        parent_initial,
        group,
    )
    tail_width = int(parent_initial.shape[-1])
    if tail_width <= 0:
        return (
            parent_initial,
            parent_initial if output_final_state else None,
            gradient_dependency,
        )
    if lengths_by_rank_cpu is None:
        raise ValueError("CP chain conv requires static all-rank bucket lengths")
    if cu_seqlens_cpu.device.type != "cpu" or lengths_by_rank_cpu.device.type != "cpu":
        raise ValueError("CP chain conv metadata must stay on CPU")
    local_tail = _local_packed_conv_tail(qkv, cu_seqlens_cpu, tail_width)
    gathered_tails = _AllGatherReplicated.apply(local_tail, group)
    rank = dist.get_rank(group)  # ty: ignore[possibly-missing-attribute]
    conv_initial = _scan_conv_tail_batch(
        parent_initial,
        gathered_tails,
        lengths_by_rank_cpu.clamp(max=tail_width),
        stop_rank=rank,
    )
    conv_initial = _add_autograd_dependency(
        conv_initial, gathered_tails.reshape(-1)[:1].sum() * 0
    )
    conv_final = (
        _scan_conv_tail_batch(
            parent_initial,
            gathered_tails,
            lengths_by_rank_cpu.clamp(max=tail_width),
            stop_rank=dist.get_world_size(group),  # ty: ignore[possibly-missing-attribute]
        )
        if output_final_state
        else None
    )
    return conv_initial, conv_final, gradient_dependency


def _local_packed_conv_tail(
    qkv: Tensor, cu_seqlens_cpu: Tensor, tail_width: int
) -> Tensor:
    segment_count = int(cu_seqlens_cpu.numel()) - 1
    channels = int(qkv.shape[1])
    tails = qkv.new_zeros(segment_count, channels, tail_width)
    lengths = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
    valid_lengths = torch.clamp(lengths, max=tail_width).tolist()
    ends = cu_seqlens_cpu[1:].tolist()
    for segment, valid in enumerate(valid_lengths):
        valid = int(valid)
        if valid <= 0:
            continue
        end = int(ends[segment])
        tails[segment, :, :valid] = qkv[end - valid : end].transpose(0, 1)
    return tails


def _scan_conv_tail_batch(
    parent_initial: Tensor,
    tails_by_rank: Tensor,
    lengths_by_rank_cpu: Tensor,
    *,
    stop_rank: int,
) -> Tensor:
    states = []
    tail_width = int(parent_initial.shape[-1])
    host_lengths = lengths_by_rank_cpu.tolist()
    for segment in range(int(parent_initial.shape[0])):
        state = parent_initial[segment]
        for peer in range(int(stop_rank)):
            valid = int(host_lengths[peer][segment])
            if valid <= 0:
                continue
            state = torch.cat([state, tails_by_rank[peer, segment, :, :valid]], dim=-1)[
                :, -tail_width:
            ]
        states.append(state)
    return torch.stack(states, dim=0)


class _AllGatherReplicated(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, local_tensor: Tensor, group: Any) -> Tensor:
        ctx.group = group
        ctx.rank = dist.get_rank(group)  # ty: ignore[possibly-missing-attribute]
        gathered = torch.empty(
            dist.get_world_size(group),  # ty: ignore[possibly-missing-attribute]
            *local_tensor.shape,
            device=local_tensor.device,
            dtype=local_tensor.dtype,
        )
        dist.all_gather_into_tensor(  # ty: ignore[possibly-missing-attribute]
            gathered,
            local_tensor.contiguous(),
            group=group,
        )
        return gathered

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor, None]:
        (grad_output,) = grad_outputs
        grad_input = torch.empty_like(grad_output[ctx.rank])
        dist.reduce_scatter_tensor(  # ty: ignore[possibly-missing-attribute]
            grad_input,
            grad_output.contiguous(),
            op=dist.ReduceOp.SUM,  # ty: ignore[possibly-missing-attribute]
            group=ctx.group,
        )
        return grad_input, None


class _AllReduceGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tensor: Tensor, group: Any) -> tuple[Tensor, Tensor]:
        ctx.group = group
        ctx.save_for_backward(tensor)
        return tensor, tensor.new_zeros(())

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor | None) -> tuple[Tensor, None]:
        grad_output, _grad_dependency = grad_outputs
        (reference,) = ctx.saved_tensors
        grad_input = (
            reference.new_zeros(reference.shape)
            if grad_output is None
            else grad_output.contiguous()
        )
        dist.all_reduce(  # ty: ignore[possibly-missing-attribute]
            grad_input,
            op=dist.ReduceOp.SUM,  # ty: ignore[possibly-missing-attribute]
            group=ctx.group,
        )
        return grad_input, None


def _causal_conv1d_packed_varlen_with_state(
    gdn: Any,
    qkv: Tensor,
    conv_initial: Tensor,
    cu_seqlens: Tensor,
    *,
    output_final_state: bool,
) -> tuple[Tensor, Tensor | None]:
    weight = gdn.conv1d.weight.squeeze(1)
    bias = gdn.conv1d.bias
    return packed_varlen_causal_conv(
        qkv,
        cu_seqlens,
        conv_initial,
        weight,
        bias,
        activation=str(getattr(gdn, "activation", "gelu")),
        output_final_state=output_final_state,
    )


def _disable_reentrant_te_linear_transpose_cache(gdn: Any) -> None:
    if getattr(gdn, "_art_reentrant_te_linear_transpose_cache_disabled", False):
        return
    for root in (getattr(gdn, "in_proj", None), getattr(gdn, "out_proj", None)):
        if isinstance(root, torch.nn.Module):
            linears = root.modules()
        else:
            linears = (root,)
        for linear in linears:
            if hasattr(linear, "disable_parameter_transpose_cache"):
                linear.disable_parameter_transpose_cache = True
    gdn._art_reentrant_te_linear_transpose_cache_disabled = True


def _zero_conv_state(
    gdn: Any,
    hidden_states: Tensor,
    row: int | None = None,
    *,
    batch_size: int = 1,
) -> Tensor:
    del row
    return hidden_states.new_zeros(
        batch_size,
        gdn.conv_dim_local_tp,
        gdn.conv_kernel_dim - 1,
    )


def _zero_recurrent_state(
    gdn: Any,
    hidden_states: Tensor,
    row: int | None = None,
    *,
    batch_size: int = 1,
) -> Tensor:
    del row
    return hidden_states.new_zeros(
        batch_size,
        gdn.num_v_heads_local_tp,
        gdn.key_head_dim,
        gdn.value_head_dim,
        dtype=torch.float32,
    )


def _default_cp_rank(cp_size: int) -> int:
    if cp_size == 1:
        return 0
    try:
        from megatron.core import parallel_state as ps

        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            return int(ps.get_context_parallel_rank())
    except Exception:
        pass
    if torch.distributed.is_available() and torch.distributed.is_initialized():  # ty: ignore[possibly-missing-attribute]
        return int(torch.distributed.get_rank())  # ty: ignore[possibly-missing-attribute]
    return 0


def _default_cp_group(cp_size: int) -> Any:
    if cp_size == 1:
        return None
    try:
        from megatron.core import parallel_state as ps

        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            return ps.get_context_parallel_group()
    except Exception:
        pass
    if torch.distributed.is_available() and torch.distributed.is_initialized():  # ty: ignore[possibly-missing-attribute]
        return torch.distributed.group.WORLD  # ty: ignore[possibly-missing-attribute]
    raise RuntimeError("CP GDN execution requires torch.distributed initialization")


def _default_tp_cp_group(cp_size: int, tp_size: int) -> Any:
    if cp_size == 1 and tp_size == 1:
        return None
    try:
        from megatron.core import parallel_state as ps

        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            return ps.get_tensor_and_context_parallel_group()
    except Exception:
        pass
    if torch.distributed.is_available() and torch.distributed.is_initialized():  # ty: ignore[possibly-missing-attribute]
        return torch.distributed.group.WORLD  # ty: ignore[possibly-missing-attribute]
    raise RuntimeError(
        "CP GDN layout exchange requires torch.distributed initialization"
    )


def _group_rank(group: Any | None) -> int:
    if group is None:
        return 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():  # ty: ignore[possibly-missing-attribute]
        return int(torch.distributed.get_rank(group))  # ty: ignore[possibly-missing-attribute]
    return 0


def _l2norm(x: Tensor) -> Tensor:
    try:
        from fla.modules.l2norm import l2norm
    except ImportError:
        return F.normalize(x, p=2, dim=-1)
    return l2norm(x)


def _chunk_gated_delta_rule(*args: Any, **kwargs: Any) -> tuple[Tensor, Tensor | None]:
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError as exc:
        raise ImportError(
            "FLA is required for ART shared-prefix GDN execution."
        ) from exc
    return chunk_gated_delta_rule(*args, **kwargs)
