from typing import Any, NamedTuple, cast

import einops
from megatron.core.transformer.transformer_config import TransformerConfig
from pydantic import BaseModel, ConfigDict, Field
import torch
import torch.nn as nn
from torch.nn import Linear

from art.megatron.dsv4.kernel.precision_aligned_ops import linear_bf16_fp32
from art.megatron.dsv4.rope import (
    apply_rotary_emb,
    configure_rope_cache,
    get_rope_cache,
    get_rope_cache_at_positions,
)
from art.megatron.dsv4.utils import rotate_activation
from art.megatron.prefix_tree import (
    PrefixTreeRow,
    PrefixTreeSegment,
    parse_prefix_tree,
)


class Dsv4CompressionLayout(NamedTuple):
    current_indices: torch.Tensor
    previous_indices: torch.Tensor
    entry_group_pre_order: torch.Tensor
    entry_group_post_order: torch.Tensor
    entry_start_positions: torch.Tensor
    entry_end_positions: torch.Tensor
    query_group_pre_order: torch.Tensor


class _Dsv4CompressionPlan(NamedTuple):
    row: PrefixTreeRow
    position_to_index_by_group: dict[int, dict[int, int]]
    positions_by_group: dict[int, list[int]]
    group_pre_order: dict[int, int]
    group_post_order: dict[int, int]
    query_group_pre_order: list[int]


class Dsv4PrefixTreeState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    compression_layouts: dict[int, Dsv4CompressionLayout]
    topk_idx_cache: dict[Any, Any] = Field(default_factory=dict)


def move_compression_layout_to_device(
    layout: Dsv4CompressionLayout,
    device: torch.device,
) -> Dsv4CompressionLayout:
    return Dsv4CompressionLayout(
        *(tensor.to(device=device, non_blocking=True) for tensor in layout)
    )


def build_prefix_tree_compression_layouts(
    *,
    position_ids: torch.Tensor,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    device: torch.device,
) -> dict[int, Dsv4CompressionLayout]:
    plan = _build_prefix_tree_compression_plan(
        position_ids=position_ids,
        group_ids=group_ids,
        parent_ids=parent_ids,
    )
    return {
        ratio: move_compression_layout_to_device(
            _emit_prefix_tree_compression_layout(plan=plan, ratio=ratio),
            device,
        )
        for ratio in (4, 128)
    }


def _segment_positions(
    position_row: torch.Tensor,
    segment: PrefixTreeSegment,
) -> list[int]:
    positions = [int(value) for value in position_row[segment.start : segment.end]]
    if positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError(
            "DSV4 prefix-tree compression requires contiguous positions within "
            f"group={segment.group_id}, got {positions[:4]}...{positions[-4:]}."
        )
    return positions


def _segment_path(
    row: PrefixTreeRow,
    segment: PrefixTreeSegment,
) -> tuple[PrefixTreeSegment, ...]:
    by_group = {candidate.group_id: candidate for candidate in row.segments}
    return tuple(
        by_group[group_id] for group_id in (*segment.ancestors, segment.group_id)
    )


def _branch_position_map(
    *,
    row: PrefixTreeRow,
    segment: PrefixTreeSegment,
    positions_by_group: dict[int, list[int]],
) -> dict[int, int]:
    by_group = {candidate.group_id: candidate for candidate in row.segments}
    mapping: dict[int, int] = {}
    expected = 0
    for path_segment in _segment_path(row, segment):
        positions = positions_by_group[path_segment.group_id]
        if positions[0] != expected:
            raise ValueError(
                "DSV4 prefix-tree compression requires contiguous branch "
                f"positions; expected {expected}, got {positions[0]} for "
                f"group={path_segment.group_id}."
            )
        for offset, logical_pos in enumerate(positions):
            mapping[logical_pos] = by_group[path_segment.group_id].start + offset
        expected = positions[-1] + 1
    return mapping


def _group_preorder_intervals(
    row: PrefixTreeRow,
) -> tuple[dict[int, int], dict[int, int]]:
    children_by_group: dict[int, list[int]] = {
        segment.group_id: [] for segment in row.segments
    }
    roots: list[int] = []
    for segment in row.segments:
        if segment.depth == 0:
            roots.append(segment.group_id)
        else:
            children_by_group[segment.parent_id].append(segment.group_id)

    pre_order: dict[int, int] = {}
    post_order: dict[int, int] = {}
    next_order = 0

    def visit(group_id: int) -> None:
        nonlocal next_order
        pre_order[group_id] = next_order
        next_order += 1
        for child_group_id in children_by_group[group_id]:
            visit(child_group_id)
        post_order[group_id] = next_order

    for root_group_id in roots:
        visit(root_group_id)
    return pre_order, post_order


def _build_prefix_tree_compression_plan(
    *,
    position_ids: torch.Tensor,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
) -> _Dsv4CompressionPlan:
    if position_ids.ndim != 2 or group_ids.ndim != 2 or parent_ids.ndim != 2:
        raise ValueError("DSV4 prefix-tree metadata must be rank-2.")
    if int(position_ids.shape[0]) != 1:
        raise ValueError(
            "DSV4 prefix-tree compression expects one packed sequence per "
            f"Megatron microbatch, got batch={int(position_ids.shape[0])}."
        )
    if tuple(position_ids.shape) != tuple(group_ids.shape) or tuple(
        group_ids.shape
    ) != tuple(parent_ids.shape):
        raise ValueError(
            "DSV4 prefix-tree position/group/parent metadata must share shape, "
            f"got {tuple(position_ids.shape)}, {tuple(group_ids.shape)}, "
            f"{tuple(parent_ids.shape)}."
        )

    position_cpu = position_ids.detach().cpu()
    group_cpu = group_ids.detach().cpu()
    parent_cpu = parent_ids.detach().cpu()
    (row,) = parse_prefix_tree(group_ids=group_cpu, parent_ids=parent_cpu)
    positions_by_group = {
        segment.group_id: _segment_positions(position_cpu[0], segment)
        for segment in row.segments
    }
    group_pre_order, group_post_order = _group_preorder_intervals(row)
    query_group_pre_order = [-1] * int(position_ids.shape[1])
    for segment in row.segments:
        pre_order = group_pre_order[segment.group_id]
        for token_index in range(segment.start, segment.end):
            query_group_pre_order[token_index] = pre_order
    position_to_index_by_group = {
        segment.group_id: _branch_position_map(
            row=row,
            segment=segment,
            positions_by_group=positions_by_group,
        )
        for segment in row.segments
    }
    return _Dsv4CompressionPlan(
        row=row,
        position_to_index_by_group=position_to_index_by_group,
        positions_by_group=positions_by_group,
        group_pre_order=group_pre_order,
        group_post_order=group_post_order,
        query_group_pre_order=query_group_pre_order,
    )


def _logical_window_indices(
    position_to_index: dict[int, int],
    *,
    logical_start: int,
    ratio: int,
    allow_negative_padding: bool,
) -> list[int]:
    indices: list[int] = []
    for logical_pos in range(logical_start, logical_start + ratio):
        index = position_to_index.get(logical_pos)
        if index is None:
            if logical_pos < 0 and allow_negative_padding:
                indices.append(-1)
                continue
            raise ValueError(
                "DSV4 prefix-tree compression window references missing "
                f"logical position {logical_pos}."
            )
        indices.append(index)
    return indices


def build_prefix_tree_compression_layout(
    *,
    position_ids: torch.Tensor,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    ratio: int,
) -> Dsv4CompressionLayout:
    return _emit_prefix_tree_compression_layout(
        plan=_build_prefix_tree_compression_plan(
            position_ids=position_ids,
            group_ids=group_ids,
            parent_ids=parent_ids,
        ),
        ratio=ratio,
    )


def _emit_prefix_tree_compression_layout(
    *,
    plan: _Dsv4CompressionPlan,
    ratio: int,
) -> Dsv4CompressionLayout:
    current_rows: list[list[int]] = []
    previous_rows: list[list[int]] = []
    entry_pre_orders: list[int] = []
    entry_post_orders: list[int] = []
    start_rows: list[int] = []
    end_rows: list[int] = []

    for segment in plan.row.segments:
        position_to_index = plan.position_to_index_by_group[segment.group_id]
        segment_positions = plan.positions_by_group[segment.group_id]
        segment_end_pos = segment_positions[-1]
        parent_end_pos = (
            -1
            if not segment.ancestors
            else plan.positions_by_group[segment.ancestors[-1]][-1]
        )
        usable = ((segment_end_pos + 1) // ratio) * ratio
        for logical_start in range(0, usable, ratio):
            if logical_start + ratio - 1 <= parent_end_pos:
                continue
            current_rows.append(
                _logical_window_indices(
                    position_to_index,
                    logical_start=logical_start,
                    ratio=ratio,
                    allow_negative_padding=False,
                )
            )
            previous_rows.append(
                _logical_window_indices(
                    position_to_index,
                    logical_start=logical_start - ratio,
                    ratio=ratio,
                    allow_negative_padding=True,
                )
            )
            entry_pre_orders.append(plan.group_pre_order[segment.group_id])
            entry_post_orders.append(plan.group_post_order[segment.group_id])
            start_rows.append(logical_start)
            end_rows.append(logical_start + ratio - 1)

    entry_count = len(current_rows)
    current = torch.empty((entry_count, ratio), dtype=torch.long)
    previous = torch.empty_like(current)
    if entry_count:
        current[:] = torch.tensor(current_rows, dtype=torch.long)
        previous[:] = torch.tensor(previous_rows, dtype=torch.long)
    entry_pre_order = torch.tensor(entry_pre_orders, dtype=torch.int32)
    entry_post_order = torch.tensor(entry_post_orders, dtype=torch.int32)
    starts = torch.tensor(start_rows, dtype=torch.long)
    ends = torch.tensor(end_rows, dtype=torch.long)
    query_pre_order = torch.tensor(plan.query_group_pre_order, dtype=torch.int32)
    return Dsv4CompressionLayout(
        current,
        previous,
        entry_pre_order,
        entry_post_order,
        starts,
        ends,
        query_pre_order,
    )


def compressed_layout_visibility(
    layout: Dsv4CompressionLayout,
    *,
    position_ids: torch.Tensor,
    q_start: int = 0,
    q_end: int | None = None,
) -> torch.Tensor:
    if int(position_ids.shape[0]) != 1:
        raise ValueError(
            "DSV4 prefix-tree compressed visibility expects one packed sequence."
        )
    if q_end is None:
        q_end = position_ids.shape[1]
    q_pos = position_ids[0, q_start:q_end].to(dtype=torch.long).unsqueeze(-1)
    q_group_pre = layout.query_group_pre_order[q_start:q_end].unsqueeze(-1)
    visible = (
        (layout.entry_end_positions.unsqueeze(0) <= q_pos)
        & (layout.entry_group_pre_order.unsqueeze(0) <= q_group_pre)
        & (q_group_pre < layout.entry_group_post_order.unsqueeze(0))
    )
    return visible.unsqueeze(0)


def compressed_layout_topk_idxs(
    layout: Dsv4CompressionLayout,
    *,
    position_ids: torch.Tensor,
    offset: int,
) -> torch.Tensor:
    visibility = compressed_layout_visibility(
        layout,
        position_ids=position_ids,
    )
    entry_ids = torch.arange(
        layout.entry_group_pre_order.shape[0],
        device=layout.entry_group_pre_order.device,
        dtype=torch.long,
    ).view(1, 1, -1)
    return torch.where(visibility, entry_ids + offset, torch.full_like(entry_ids, -1))


class RMSNorm(nn.Module):
    """
    Kept in pure PyTorch with FP32 weights to match SGLang's compressor norm.

    Args:
        dim: Dimension of the input tensor.
        eps: Epsilon for numerical stability. Defaults to ``1e-6``.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


def _overlap_transform(
    tensor: torch.Tensor, *, compress_ratio: int, head_dim: int, value=0
) -> torch.Tensor:
    """Overlap-transform for compress_ratio=4: for each token group of size ``ratio``,
    split into (first_half, second_half) halves along ``head_dim`` and re-arrange
    them across a doubled ratio axis (`2 * ratio`), shifting the first half by one
    group so that adjacent groups overlap by ``ratio`` positions.
    """
    b, s, _, _ = tensor.size()
    new_tensor = tensor.new_full((b, s, 2 * compress_ratio, head_dim), value)
    new_tensor[:, :, compress_ratio:] = tensor[:, :, :, head_dim:]
    new_tensor[:, 1:, :compress_ratio] = tensor[:, :-1, :, :head_dim]
    return new_tensor


def _add_lora_if_present(
    owner: nn.Module,
    attr_name: str,
    base: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    lora = getattr(owner, attr_name, None)
    if lora is None:
        return base
    return base + lora(x)


def _gather_projected_tokens(
    tensor: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    bsz, seqlen, channels = tensor.shape
    if indices.numel() == 0:
        if indices.ndim == 2:
            return tensor.new_empty((bsz, *indices.shape, channels))
        return tensor.new_empty((*indices.shape, channels))
    if indices.ndim == 2:
        if bsz != 1:
            raise ValueError(
                "DSV4 prefix-tree compression expects one packed sequence per "
                f"Megatron microbatch, got batch={bsz}."
            )
        safe_indices = indices.clamp(0, max(seqlen - 1, 0))
        gathered = tensor[0].index_select(0, safe_indices.reshape(-1))
        return gathered.view(*indices.shape, channels).unsqueeze(0)
    batch_offsets = (
        torch.arange(bsz, device=tensor.device, dtype=torch.long).view(bsz, 1, 1)
        * seqlen
    )
    safe_indices = indices.clamp(0, max(seqlen - 1, 0))
    flat_indices = (safe_indices + batch_offsets).reshape(-1)
    gathered = tensor.reshape(bsz * seqlen, channels).index_select(0, flat_indices)
    return gathered.view(*indices.shape, channels)


class DeepSeekV4Compressor(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        head_dim: int,
        compress_ratio: int,
        rotate: bool,
        cp_group: Any | None = None,
    ):
        super().__init__()

        cfg = cast(Any, config)
        dim = config.hidden_size
        rope_head_dim = int(cfg.qk_pos_emb_head_dim)
        norm_eps = config.layernorm_epsilon

        assert head_dim in {128, 512}
        assert rope_head_dim == 64
        assert compress_ratio in {4, 128}
        assert norm_eps == 1e-6

        self.config = config
        self.dim = dim
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.nope_head_dim = head_dim - rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap

        self.cp_group = cp_group
        self.cp_size = cp_group.size() if cp_group is not None else 1
        self.cp_rank = cp_group.rank() if cp_group is not None else 0

        self.ape = nn.Parameter(
            torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32)
        )
        weight_dtype = config.params_dtype
        if weight_dtype not in {torch.bfloat16, torch.float32}:
            raise TypeError(
                f"DeepSeek-V4 compressor requires bf16/fp32 params, got {weight_dtype}"
            )
        self.wkv = Linear(
            self.dim, coff * self.head_dim, bias=False, dtype=weight_dtype
        )
        self.wgate = Linear(
            self.dim, coff * self.head_dim, bias=False, dtype=weight_dtype
        )
        self.norm = RMSNorm(self.head_dim, norm_eps)

        self._keep_fp32_parameters = ("ape",)
        setattr(self.ape, "_keep_fp32", True)

        base = cfg.dsv4_compress_rope_theta
        assert rope_head_dim == 64
        assert base == 160000
        configure_rope_cache(self, config, rope_head_dim=rope_head_dim, base=base)

    @property
    def weight(self) -> torch.Tensor:
        return self.ape

    def overlap_transform_raw(self, tensor: torch.Tensor, value=0):
        """Raw overlap transform without CP handling."""
        return _overlap_transform(
            tensor,
            compress_ratio=self.compress_ratio,
            head_dim=self.head_dim,
            value=value,
        )

    def overlap_transform_with_cp(self, tensor: torch.Tensor, value=0) -> torch.Tensor:
        """
        Overlap transform with CP support.

        Args:
            tensor: [bsz, G_local, ratio, coff*d]
            value: Fill value for overlap transform (0 for kv, -inf for score)

        Returns:
            [bsz, G_local, ratio, coff*d]
        """
        if self.cp_size != 1:
            raise RuntimeError(
                "DeepSeek-V4 non-CP compressor received context_parallel_size > 1."
            )
        return self.overlap_transform_raw(tensor, value)

    def _project_raw(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ape.dtype != torch.float32:
            raise TypeError(
                f"DeepSeek-V4 compressor APE must stay fp32, got {self.ape.dtype}."
            )
        if self.wkv.weight.dtype not in {torch.bfloat16, torch.float32}:
            raise TypeError(
                "DeepSeek-V4 compressor KV projection requires bf16/fp32 weights, got "
                f"{self.wkv.weight.dtype}."
            )
        if self.wgate.weight.dtype not in {torch.bfloat16, torch.float32}:
            raise TypeError(
                "DeepSeek-V4 compressor gate projection requires bf16/fp32 weights, "
                f"got {self.wgate.weight.dtype}."
            )

        kv = _add_lora_if_present(
            self, "kv_proj_lora", linear_bf16_fp32(x, self.wkv.weight), x
        )
        score = _add_lora_if_present(
            self, "gate_proj_lora", linear_bf16_fp32(x, self.wgate.weight), x
        )
        return kv, score

    def _compress_projected(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seqlen_local, _ = kv.size()
        ratio, overlap, _ = self.compress_ratio, self.overlap, self.head_dim
        dtype = kv.dtype

        usable = (seqlen_local // ratio) * ratio
        if usable == 0:
            return kv.new_zeros((bsz, 0, self.head_dim))
        kv = kv[:, :usable]
        score = score[:, :usable]
        if self.cp_size > 1:
            assert usable % (ratio * 2) == 0

        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + self.ape

        if overlap:
            kv = self.overlap_transform_with_cp(kv, 0)
            score = self.overlap_transform_with_cp(score, float("-inf"))

        score_softmax = score.softmax(dim=2, dtype=torch.float32).to(kv.dtype)
        kv = (kv * score_softmax).sum(dim=2)

        kv = self.norm(kv.to(dtype))

        apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)

        if self.rotate:
            kv = rotate_activation(kv)

        return kv

    def _compress_prefix_tree_projected(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        *,
        layout: Dsv4CompressionLayout,
    ) -> torch.Tensor:
        dtype = kv.dtype
        ratio = self.compress_ratio
        current_valid = layout.current_indices >= 0
        current_kv = _gather_projected_tokens(kv, layout.current_indices)
        current_score = _gather_projected_tokens(score, layout.current_indices)
        current_kv = torch.where(
            current_valid.unsqueeze(-1), current_kv, torch.zeros_like(current_kv)
        )
        current_score = torch.where(
            current_valid.unsqueeze(-1),
            current_score,
            torch.full_like(current_score, float("-inf")),
        )
        if self.overlap:
            previous_valid = layout.previous_indices >= 0
            previous_kv = _gather_projected_tokens(kv, layout.previous_indices)
            previous_score = _gather_projected_tokens(score, layout.previous_indices)
            previous_kv = torch.where(
                previous_valid.unsqueeze(-1),
                previous_kv,
                torch.zeros_like(previous_kv),
            )
            previous_score = torch.where(
                previous_valid.unsqueeze(-1),
                previous_score,
                torch.full_like(previous_score, float("-inf")),
            )
            current_score = current_score + self.ape.view(1, 1, ratio, -1)
            previous_score = previous_score + self.ape.view(1, 1, ratio, -1)
            slots_kv = torch.cat(
                [previous_kv[..., : self.head_dim], current_kv[..., self.head_dim :]],
                dim=2,
            )
            slots_score = torch.cat(
                [
                    previous_score[..., : self.head_dim],
                    current_score[..., self.head_dim :],
                ],
                dim=2,
            )
        else:
            slots_kv = current_kv
            slots_score = current_score + self.ape.view(1, 1, ratio, -1)

        score_softmax = slots_score.softmax(dim=2, dtype=torch.float32).to(
            slots_kv.dtype
        )
        compressed = (slots_kv * score_softmax).sum(dim=2)
        compressed = self.norm(compressed.to(dtype))
        freqs_cis = get_rope_cache_at_positions(
            self, position_ids=layout.entry_start_positions, device=kv.device
        )
        apply_rotary_emb(compressed[..., -self.rope_head_dim :], freqs_cis)

        if self.rotate:
            compressed = rotate_activation(compressed)
        return compressed

    def forward_raw(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        shared_layout: Dsv4CompressionLayout | None = None,
    ) -> torch.Tensor:
        kv, score = self._project_raw(x)
        if shared_layout is not None:
            return self._compress_prefix_tree_projected(kv, score, layout=shared_layout)
        usable = (x.shape[1] // self.compress_ratio) * self.compress_ratio
        freqs_cis = get_rope_cache(self, seqlen=usable, device=x.device)[
            : usable : self.compress_ratio
        ]
        if position_ids is not None:
            freqs_cis = get_rope_cache_at_positions(
                self,
                position_ids=position_ids[:, : usable : self.compress_ratio],
                device=x.device,
            )
        return self._compress_projected(kv, score, freqs_cis=freqs_cis)

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        shared_layout: Dsv4CompressionLayout | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [seqlen, batch, dim] SBHD layout (Megatron standard)
        Returns:
            k: [floor(seqlen / compress_ratio), batch, head_dim] SBHD layout
        """
        x_bshd = einops.rearrange(x, "s b d -> b s d")
        k_bshd = self.forward_raw(
            x_bshd,
            position_ids=position_ids,
            shared_layout=shared_layout,
        )
        k = einops.rearrange(k_bshd, "b sc d -> sc b d")
        return k
