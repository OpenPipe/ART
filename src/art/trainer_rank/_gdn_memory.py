"""Partial CP1 checkpoint pending-save accounting, not a backward upper bound.

The bucket schedule mirrors the CP1 chunk-aligned GDN planner. It deliberately
uses CPU segment metadata, not input values or tensor allocation observations.
FLA source-version and generated-save limitations are documented with this
partial floor; other shared-expert saves, AOT saves and workspace remain unpriced.
"""

from dataclasses import dataclass
from types import MethodType
from typing import Any, Sequence


@dataclass(frozen=True)
class Bucket:
    # family, parent (-1 for a root), number of executed rows
    columns: tuple[tuple[int, int, int], ...]
    final: bool


def cp1_buckets(segments: Sequence[Any]) -> tuple[Bucket, ...]:
    """Original 64-row root boundary / replayed-tail / child bucket schedule."""
    count = len(segments)
    if not count:
        return ()
    children: list[list[int]] = [[] for _ in segments]
    depths: list[int] = []
    parents: list[int] = []
    lengths: list[int] = []
    ids = {s.group_id: i for i, s in enumerate(segments)}
    if len(ids) != count:
        raise ValueError("Duplicate GDN segment identity")
    cursor = 0
    for i, s in enumerate(segments):
        if any(
            type(v) is not int
            for v in (s.start, s.end, s.packed_start, s.group_id, s.parent_id)
        ):
            raise ValueError("Noninteger GDN segment metadata")
        length = s.end - s.start
        parent = -1 if s.parent_id == s.group_id else ids.get(s.parent_id, count)
        if (
            length <= 0
            or s.start < 0
            or s.packed_start != cursor
            or not -1 <= parent < i
        ):
            raise ValueError("Invalid ordered GDN segment geometry")
        cursor += length
        parents.append(parent)
        lengths.append(length)
        depths.append(0 if parent < 0 else depths[parent] + 1)
        if parent >= 0:
            children[parent].append(i)
    boundary: list[tuple[int, int, int]] = []
    regular: list[tuple[int, int, int]] = []
    explicit: dict[int, list[tuple[int, int, int]]] = {}
    for i, length in enumerate(lengths):
        if parents[i] < 0:
            if not children[i]:
                regular.append((i, -1, length))
            elif length // 64:
                boundary.append((i, -1, length // 64 * 64))
        for child in children[i]:
            tail = length % 64 if parents[i] < 0 else 0
            parent = i if parents[i] >= 0 or length >= 64 else -1
            explicit.setdefault(depths[child], []).append(
                (child, parent, tail + lengths[child])
            )
    buckets = []
    for columns in (boundary, regular):
        if columns:
            buckets.append(
                Bucket(
                    tuple(sorted(columns, key=lambda c: (c[2], c[0]))),
                    any(children[c[0]] for c in columns),
                )
            )
    for depth in sorted(explicit):
        columns = tuple(explicit[depth])
        buckets.append(Bucket(columns, any(children[c[0]] for c in columns)))
    return tuple(buckets)


@dataclass(frozen=True)
class Shape:
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    conv_width: int
    output_lora_rank: int
    moe_bytes_per_row: int

    def pending(self, packed_rows: int, buckets: tuple[Bucket, ...]) -> int:
        """Known save-set envelope; final-state backing may alias initial saves.

        Charge each bucket's initial-state extent and each produced final-state
        backing once. This intentionally overcounts aliases: a child view can
        keep an entire parent batch live; request/root count cannot bound it.
        No claim that every charged backing remains live at the MoE peak.
        """
        hk, hv, dk, dv = self.key_heads, self.value_heads, self.key_dim, self.value_dim
        conv = 2 * hk * dk + hv * dv
        # FLA q/k/v, FP32 cumulative g, BF16 beta and A; convolution input;
        # external q/k L2 reciprocal norms. All counts follow executed buckets.
        per_bucket_row = (2 * hv * dk + hv * dv + hv + hv * 64 + conv) * 2 + hv * 4 * 3
        # Recurrent norm input/rstd and the ordinary trainable output LoRA's
        # input plus rank temporary; these use final packed rows, not replay.
        per_output_row = hv * dv * 2 + hv * 4
        if self.output_lora_rank:
            per_output_row += hv * dv * 2 + self.output_lora_rank * 2
        state = hv * dk * dv * 4 + conv * (self.conv_width - 1) * 2
        executed = sum(c[2] for b in buckets for c in b.columns)
        state_rows = sum(len(b.columns) * (1 + int(b.final)) for b in buckets)
        return (
            executed * per_bucket_row
            + packed_rows * per_output_row
            + state_rows * state
        )


def model_shapes(
    rank: Any, slot_ref: Any = None
) -> tuple[int, tuple[Shape, ...]] | None:
    """Conditional original-owner metadata; no model execution or CUDA read."""
    if not getattr(rank, "_gdn_layers", 0):
        return None
    import torch

    from art.trainer_rank._impl import (
        _expert_parallel_shape,
        _language_model,
        _slot_lora_tensors,
    )

    if (
        len(rank.runtime.model) != 1
        or rank._topology_key()[1:] != (1, 1, 1)
        or _expert_parallel_shape(rank.runtime.provider) != (1, 1)
    ):
        return None
    try:
        decoder = _language_model(rank.runtime.model[0]).decoder
    except (AttributeError, RuntimeError):
        return None
    if type(decoder).__name__ != "TransformerBlock":
        return None
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    from megatron.core.transformer.transformer_block import TransformerBlock
    from transformer_engine.pytorch import RMSNorm

    from art.megatron.gdn.operator import _empty_safe_norm_forward, _prefix_tree_forward
    from art.megatron.lora import LoRA, SelfAttentionLinearProjLoRA

    config = decoder.config
    expected = dict(
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        distribute_saved_activations=False,
        sequence_parallel=False,
        fp32_residual_connection=False,
        cpu_offloading=False,
        cuda_graph_impl="none",
    )
    if (
        type(decoder) is not TransformerBlock
        or decoder.training is not True
        or not decoder.layers
        or len(decoder.layers) != decoder.num_layers_per_pipeline_rank
        or len(decoder.layers) != config.num_layers
        or config.hidden_size != rank._hidden_size
        or config.params_dtype is not torch.bfloat16
        or rank._param_dtype_size != 2
        or next(rank.runtime.model[0].parameters()).dtype is not torch.bfloat16
        or any(
            type(getattr(config, k, None)) is not type(v) or getattr(config, k) != v
            for k, v in expected.items()
        )
        or getattr(config, "fp8", None)
        or getattr(config, "fp4", None)
        or any(
            n in vars(decoder)
            for n in ("forward", "_checkpointed_forward", "_get_layer")
        )
        or decoder._forward_hooks
        or decoder._forward_pre_hooks
    ):
        return None
    # The constructor priced the original owners before installing dispatcher
    # caches. Repricing their owned partials now would silently return zero.
    # Like the existing static floor, this cache requires unchanged model,
    # dtype and topology since construction; rebuilding the rank invalidates it.
    moe = rank._moe_output_bytes_per_token
    if type(moe) is not int or moe < 0 or (rank._moe_layers and not moe):
        raise ValueError("Invalid constructor MoE coefficient for GDN pending floor")
    rank._checkpoint_moe_bytes_per_token()
    shapes = []
    for layer in decoder.layers:
        gdn = getattr(layer, "self_attention", None)
        if gdn is None or type(gdn) is not GatedDeltaNet:
            continue
        if (
            getattr(gdn.forward, "__func__", None) is not _prefix_tree_forward
            or gdn.use_qk_l2norm is not True
            or gdn.tp_size != 1
            or gdn.sp_size != 1
            or gdn._forward_hooks
            or gdn._forward_pre_hooks
        ):
            return None
        dimensions = tuple(
            getattr(gdn, n)
            for n in (
                "num_key_heads",
                "num_value_heads",
                "key_head_dim",
                "value_head_dim",
                "conv_kernel_dim",
            )
        )
        if (
            any(type(n) is not int or n <= 0 for n in dimensions)
            or dimensions[1] % dimensions[0]
        ):
            return None
        hk, hv, dk, dv, kernel = dimensions
        conv = 2 * hk * dk + hv * dv
        norm = gdn.out_norm
        if type(norm) is not RMSNorm:
            return None
        forward = getattr(norm, "forward", None)
        physical = getattr(norm, "_art_empty_safe_norm_physical_forward", None)
        if (
            gdn.conv1d.weight.dtype is not torch.bfloat16
            or tuple(gdn.conv1d.weight.shape) != (conv, 1, kernel)
            or gdn.out_norm.weight.numel() != dv
            or gdn.out_norm.weight.dtype is not torch.bfloat16
            # Original GDN setup installs this wrapper even for nonempty CP1.
            # Its nonempty path delegates unchanged to the saved bound method.
            or (
                "forward" in vars(norm)
                and not (
                    type(forward) is MethodType
                    and forward.__self__ is norm
                    and forward.__func__ is _empty_safe_norm_forward
                    and getattr(norm, "_art_empty_safe_norm_hooked", None) is True
                    and physical is not None
                    and type(physical) is MethodType
                    and physical.__self__ is norm
                    and physical.__func__ is RMSNorm.forward
                )
            )
            or gdn.out_norm._forward_hooks
            or gdn.out_norm._forward_pre_hooks
        ):
            return None
        out = gdn.out_proj
        lora_rank = 0
        if type(out) is SelfAttentionLinearProjLoRA and type(out.lora) is LoRA:
            lora = out.lora
            if (
                "forward" in vars(out)
                or "forward" in vars(lora)
                or "active_lora_tensors" in vars(lora)
                or "_slot" in vars(lora)
                or out._forward_hooks
                or lora._forward_hooks
                or out._forward_pre_hooks
                or lora._forward_pre_hooks
            ):
                return None
            # Keep the previous conservative inactive-adapter enclosure.
            tensors = _slot_lora_tensors(
                lora,
                None if slot_ref is not None and slot_ref.name is None else slot_ref,
            )
            if tensors is None:
                shapes.append(Shape(hk, hv, dk, dv, kernel, 0, moe))
                continue
            a, b = tensors
            if (
                a.ndim != 2
                or b.ndim != 2
                or a.dtype is not torch.bfloat16
                or b.dtype is not torch.bfloat16
                or a.shape[0] != dimensions[1] * dimensions[3]
                or a.shape[1] != b.shape[0]
            ):
                return None
            # Use the admitted slot rank without changing its active context.
            lora_rank = int(a.shape[1])
        shapes.append(Shape(hk, hv, dk, dv, kernel, lora_rank, moe))
    return (len(decoder.layers), tuple(shapes)) if shapes else None


def plan_floor(rank: Any, plan: Any) -> tuple[int, int]:
    """Boundary retention plus pending attention and the cached MoE maximum.

    Combining maxima from different layers may conservatively overcharge.
    """
    gradients = [g for g in plan.groups if g.grad_enabled]
    if not gradients:
        return 0, 0
    retained = 0
    workspace = 0
    for group in plan.groups:
        rows = int(group.packed.tokens.numel())
        model = model_shapes(rank, group.slot_ref)
        if model is None:
            return 0, 0
        layers, shapes = model
        if not group.grad_enabled:
            # Earlier gradient groups remain live during a later reference
            # group. Only its existing MoE component enters this stage.
            workspace = max(
                workspace, rank._moe_workspace_bytes(rows, slot_ref=group.slot_ref)
            )
            continue
        buckets = cp1_buckets(group.packed.segments)
        if sum(s.length for s in group.packed.segments) != rows:
            raise ValueError("GDN packed rows disagree with segment geometry")
        retained += rows * layers * rank._hidden_size * 2
        workspace = max(
            workspace,
            *(
                rank._moe_workspace_bytes(
                    rows, checkpoint_grad=True, slot_ref=group.slot_ref
                )
                + s.pending(rows, buckets)
                for s in shapes
            ),
        )
    return retained, workspace
