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


class Dsv4CompressionLayout(NamedTuple):
    current_indices: torch.Tensor
    previous_indices: torch.Tensor
    entry_group_ids: torch.Tensor
    entry_parent_visible: torch.Tensor
    entry_start_positions: torch.Tensor
    entry_end_positions: torch.Tensor
    entry_valid: torch.Tensor


class Dsv4SharedPrefixState(BaseModel):
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


def build_shared_prefix_compression_layouts(
    *,
    position_ids: torch.Tensor,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    device: torch.device,
) -> dict[int, Dsv4CompressionLayout]:
    return {
        ratio: move_compression_layout_to_device(
            build_shared_prefix_compression_layout(
                position_ids=position_ids,
                group_ids=group_ids,
                parent_ids=parent_ids,
                ratio=ratio,
                duplicate_prompt_entries=ratio == 4,
            ),
            device,
        )
        for ratio in (4, 128)
    }


def _logical_window_indices(
    *,
    prompt_start: int,
    prompt_len: int,
    completion_start: int | None,
    logical_start: int,
    ratio: int,
) -> list[int]:
    indices: list[int] = []
    for logical_pos in range(logical_start, logical_start + ratio):
        if logical_pos < 0:
            indices.append(-1)
        elif logical_pos < prompt_len:
            indices.append(prompt_start + logical_pos)
        elif completion_start is not None:
            indices.append(completion_start + logical_pos - prompt_len)
        else:
            indices.append(-1)
    return indices


def build_shared_prefix_compression_layout(
    *,
    position_ids: torch.Tensor,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    ratio: int,
    duplicate_prompt_entries: bool = False,
) -> Dsv4CompressionLayout:
    device = position_ids.device
    bsz, seqlen = position_ids.shape
    position_cpu = position_ids.detach().cpu()
    group_cpu = group_ids.detach().cpu()
    parent_cpu = parent_ids.detach().cpu()
    current_rows: list[list[list[int]]] = []
    previous_rows: list[list[list[int]]] = []
    group_rows: list[list[int]] = []
    parent_visible_rows: list[list[bool]] = []
    start_rows: list[list[int]] = []
    end_rows: list[list[int]] = []

    for b in range(bsz):
        valid_mask = (group_cpu[b] != -1) & (parent_cpu[b] != -1)
        padding = torch.nonzero(~valid_mask, as_tuple=False)
        valid_len = int(padding[0].item()) if padding.numel() else seqlen
        if bool(valid_mask[valid_len:].any().item()):
            raise ValueError("DSV4 shared-prefix metadata must pad only at row end.")
        segments: list[tuple[int, int, int, int, int, int]] = []
        cursor = 0
        while cursor < valid_len:
            group = int(group_cpu[b, cursor].item())
            parent = int(parent_cpu[b, cursor].item())
            start = cursor
            cursor += 1
            while cursor < valid_len and int(group_cpu[b, cursor].item()) == group:
                cursor += 1
            end = cursor
            start_pos = int(position_cpu[b, start].item())
            end_pos = int(position_cpu[b, end - 1].item())
            segments.append((group, parent, start, end, start_pos, end_pos))

        row_current: list[list[int]] = []
        row_previous: list[list[int]] = []
        row_groups: list[int] = []
        row_parent_visible: list[bool] = []
        row_starts: list[int] = []
        row_ends: list[int] = []

        def append_entry(
            *,
            entry_group: int,
            parent_visible: bool,
            current_indices: list[int],
            previous_indices: list[int],
            logical_start: int,
        ) -> None:
            row_current.append(current_indices)
            row_previous.append(previous_indices)
            row_groups.append(entry_group)
            row_parent_visible.append(parent_visible)
            row_starts.append(logical_start)
            row_ends.append(logical_start + ratio - 1)

        seg_idx = 0
        while seg_idx < len(segments):
            group, parent, prompt_start, prompt_end, _, _ = segments[seg_idx]
            if group != parent:
                seg_idx += 1
                continue
            prompt_len = prompt_end - prompt_start
            shared_usable = (prompt_len // ratio) * ratio
            shared_windows: list[tuple[list[int], list[int], int]] = []
            for logical_start in range(0, shared_usable, ratio):
                current_indices = _logical_window_indices(
                    prompt_start=prompt_start,
                    prompt_len=prompt_len,
                    completion_start=None,
                    logical_start=logical_start,
                    ratio=ratio,
                )
                previous_indices = _logical_window_indices(
                    prompt_start=prompt_start,
                    prompt_len=prompt_len,
                    completion_start=None,
                    logical_start=logical_start - ratio,
                    ratio=ratio,
                )
                shared_windows.append(
                    (current_indices, previous_indices, logical_start)
                )
                append_entry(
                    entry_group=group,
                    parent_visible=not duplicate_prompt_entries,
                    current_indices=current_indices,
                    previous_indices=previous_indices,
                    logical_start=logical_start,
                )

            child_idx = seg_idx + 1
            while child_idx < len(segments):
                child_group, child_parent, child_start, _, _, child_end_pos = segments[
                    child_idx
                ]
                if child_group == child_parent or child_parent != group:
                    break
                if duplicate_prompt_entries:
                    for (
                        current_indices,
                        previous_indices,
                        logical_start,
                    ) in shared_windows:
                        append_entry(
                            entry_group=child_group,
                            parent_visible=False,
                            current_indices=current_indices,
                            previous_indices=previous_indices,
                            logical_start=logical_start,
                        )
                branch_usable = ((child_end_pos + 1) // ratio) * ratio
                for logical_start in range(0, branch_usable, ratio):
                    if logical_start + ratio <= prompt_len:
                        continue
                    append_entry(
                        entry_group=child_group,
                        parent_visible=False,
                        current_indices=_logical_window_indices(
                            prompt_start=prompt_start,
                            prompt_len=prompt_len,
                            completion_start=child_start,
                            logical_start=logical_start,
                            ratio=ratio,
                        ),
                        previous_indices=_logical_window_indices(
                            prompt_start=prompt_start,
                            prompt_len=prompt_len,
                            completion_start=child_start,
                            logical_start=logical_start - ratio,
                            ratio=ratio,
                        ),
                        logical_start=logical_start,
                    )
                child_idx += 1
            seg_idx = child_idx

        current_rows.append(row_current)
        previous_rows.append(row_previous)
        group_rows.append(row_groups)
        parent_visible_rows.append(row_parent_visible)
        start_rows.append(row_starts)
        end_rows.append(row_ends)

    max_entries = max((len(row) for row in current_rows), default=0)
    current = torch.full((bsz, max_entries, ratio), -1, dtype=torch.long, device=device)
    previous = torch.full_like(current, -1)
    entry_groups = torch.full((bsz, max_entries), -1, dtype=torch.long, device=device)
    parent_visible = torch.zeros((bsz, max_entries), dtype=torch.bool, device=device)
    starts = torch.full_like(entry_groups, -1)
    ends = torch.full_like(entry_groups, -1)
    valid = torch.zeros((bsz, max_entries), dtype=torch.bool, device=device)
    for b, row in enumerate(current_rows):
        if not row:
            continue
        count = len(row)
        current[b, :count] = torch.tensor(row, dtype=torch.long, device=device)
        previous[b, :count] = torch.tensor(
            previous_rows[b], dtype=torch.long, device=device
        )
        entry_groups[b, :count] = torch.tensor(
            group_rows[b], dtype=torch.long, device=device
        )
        parent_visible[b, :count] = torch.tensor(
            parent_visible_rows[b], dtype=torch.bool, device=device
        )
        starts[b, :count] = torch.tensor(start_rows[b], dtype=torch.long, device=device)
        ends[b, :count] = torch.tensor(end_rows[b], dtype=torch.long, device=device)
        valid[b, :count] = True
    return Dsv4CompressionLayout(
        current,
        previous,
        entry_groups,
        parent_visible,
        starts,
        ends,
        valid,
    )


def compressed_layout_visibility(
    layout: Dsv4CompressionLayout,
    *,
    position_ids: torch.Tensor,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    q_start: int = 0,
    q_end: int | None = None,
) -> torch.Tensor:
    if q_end is None:
        q_end = position_ids.shape[1]
    q_pos = position_ids[:, q_start:q_end].to(dtype=torch.long).unsqueeze(-1)
    q_group = group_ids[:, q_start:q_end].to(dtype=torch.long).unsqueeze(-1)
    q_parent = parent_ids[:, q_start:q_end].to(dtype=torch.long).unsqueeze(-1)
    entry_group = layout.entry_group_ids.unsqueeze(1)
    parent_visible = layout.entry_parent_visible.unsqueeze(1)
    return (
        layout.entry_valid.unsqueeze(1)
        & (layout.entry_end_positions.unsqueeze(1) <= q_pos)
        & ((entry_group == q_group) | (parent_visible & (entry_group == q_parent)))
    )


def compressed_layout_topk_idxs(
    layout: Dsv4CompressionLayout,
    *,
    position_ids: torch.Tensor,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    offset: int,
) -> torch.Tensor:
    visibility = compressed_layout_visibility(
        layout,
        position_ids=position_ids,
        group_ids=group_ids,
        parent_ids=parent_ids,
    )
    entry_ids = torch.arange(
        layout.entry_group_ids.shape[1],
        device=layout.entry_group_ids.device,
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
        return tensor.new_empty((*indices.shape, channels))
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

    def _compress_shared_prefix_projected(
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

        slots_score = torch.where(
            layout.entry_valid[:, :, None, None],
            slots_score,
            torch.zeros_like(slots_score),
        )
        score_softmax = slots_score.softmax(dim=2, dtype=torch.float32).to(
            slots_kv.dtype
        )
        compressed = (slots_kv * score_softmax).sum(dim=2)
        compressed = self.norm(compressed.to(dtype))
        freqs_cis = get_rope_cache_at_positions(
            self, position_ids=layout.entry_start_positions, device=kv.device
        )
        apply_rotary_emb(compressed[..., -self.rope_head_dim :], freqs_cis)
        compressed = torch.where(
            layout.entry_valid.unsqueeze(-1),
            compressed,
            torch.zeros_like(compressed),
        )

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
            return self._compress_shared_prefix_projected(
                kv, score, layout=shared_layout
            )
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
