"""Adapter-sorted, rank-local dense LoRA, with ordinary slot parameters.

The planner owns row order. This module never permutes model activations: attention
and GDN plans must describe the same order, including padding and SP gathers.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import contextvars
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .lora import LoRASlotRef


@dataclass(frozen=True)
class LoRASegments:
    """One contiguous span per adapter, in flattened (sequence, batch) order.

    ``None`` selects the module's default parameters; a base slot or a missing
    module contributes zero. Counts live on CPU and are validated once, before
    entering the hot path. A segment with zero rows is allowed. The caller
    must pin referenced slot versions until backward/recompute completes.
    """

    refs: tuple["LoRASlotRef | None", ...]
    counts: tuple[int, ...]
    grouped: bool = False

    def __post_init__(self) -> None:
        if not self.refs or len(self.refs) != len(self.counts):
            raise ValueError("segments require equally sized, nonempty refs/counts")
        if any(type(n) is not int or n < 0 for n in self.counts):
            raise ValueError("segment counts must be nonnegative Python integers")
        if len(set(self.refs)) != len(self.refs):
            raise ValueError("an adapter must occupy exactly one sorted segment")

    @property
    def rows(self) -> int:
        return sum(self.counts)

    def adapter_indices(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Materialize the per-token adapter table for a planner/debugger."""
        return torch.repeat_interleave(
            torch.arange(len(self.refs), device=device),
            torch.tensor(self.counts, device=device),
        )

    @classmethod
    def from_adapter_indices(
        cls,
        refs: Sequence["LoRASlotRef | None"],
        indices: torch.Tensor,
    ) -> "LoRASegments":
        # This is a plan-construction API; a CUDA input intentionally synchronizes.
        if indices.ndim != 1 or indices.dtype not in {torch.int32, torch.int64}:
            raise ValueError("adapter indices must be a 1D int32/int64 table")
        ids = indices.detach().cpu().to(torch.int64)
        if ids.numel() and (
            int(ids.min()) < 0
            or int(ids.max()) >= len(refs)
            or bool((ids[1:] < ids[:-1]).any())
        ):
            raise ValueError("rows must be sorted by an in-range adapter index")
        return cls(
            tuple(refs), tuple(torch.bincount(ids, minlength=len(refs)).tolist())
        )


_CURRENT_LORA_SEGMENTS: contextvars.ContextVar[LoRASegments | None] = (
    contextvars.ContextVar("art_megatron_current_lora_segments", default=None)
)


@contextmanager
def use_lora_segments(segments: LoRASegments | None) -> Iterator[None]:
    token = _CURRENT_LORA_SEGMENTS.set(segments)
    try:
        yield
    finally:
        _CURRENT_LORA_SEGMENTS.reset(token)


def stack_lora_factors(
    factors: Sequence[tuple[torch.Tensor, torch.Tensor, float] | None],
    *,
    template_a: torch.Tensor,
    template_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad and stack without detaching: gradients return to each owning slot.

    Scaling B makes QuACK's scalar scale=1 correct for heterogeneous alphas.
    These transient stacks must not be cached across backward/optimizer steps.
    """
    rank = max(
        (int(pair[0].shape[-1]) for pair in factors if pair is not None), default=1
    )
    rank = max(8, (rank + 7) // 8 * 8)
    a_stack, b_stack = [], []
    for pair in factors:
        if pair is None:
            a_stack.append(template_a.new_zeros(template_a.shape[-2], rank))
            b_stack.append(template_b.new_zeros(rank, template_b.shape[-1]))
        else:
            a, b, scale = pair
            a_stack.append(F.pad(a, (0, rank - a.shape[-1])))
            b_stack.append(F.pad(b * scale, (0, 0, 0, rank - b.shape[-2])))
    return torch.stack(a_stack), torch.stack(b_stack)


def segmented_lora_factors(
    x: torch.Tensor,
    factors: Sequence[tuple[torch.Tensor, torch.Tensor, float] | None],
    segments: LoRASegments,
    *,
    template_a: torch.Tensor,
    template_b: torch.Tensor,
) -> torch.Tensor:
    """Keep sequential rank/scale geometry unless grouped math is requested.

    Both modes use the original slot Parameters and build a fresh graph. The
    enclosing executor must isolate attention/GDN roots and pin slot versions.
    """
    if x.numel() // x.shape[-1] != segments.rows or len(factors) != len(segments.refs):
        raise ValueError("projection rows/factors disagree with the segment table")
    if segments.grouped:
        a, b = stack_lora_factors(factors, template_a=template_a, template_b=template_b)
        return segmented_lora(x, a, b, segments)
    outputs = []
    for chunk, pair in zip(
        x.reshape(-1, x.shape[-1]).split(segments.counts), factors, strict=True
    ):
        if pair is None:
            outputs.append(chunk.new_zeros((chunk.shape[0], template_b.shape[-1])))
        else:
            a, b, scale = pair
            output = (chunk @ a) @ b
            outputs.append(output if scale == 1.0 else output * scale)
    return torch.cat(outputs).reshape(*x.shape[:-1], template_b.shape[-1])


def segmented_lora(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    segments: LoRASegments,
) -> torch.Tensor:
    """Use QuACK for CUDA bf16/fp16 and Torch otherwise; rounding can differ."""
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    if flat.shape[0] != segments.rows:
        raise ValueError(
            f"segment table has {segments.rows} rows, projection has {flat.shape[0]}; "
            "supply metadata for this rank's actual layout (including SP/padding)"
        )
    if a.shape[0] != len(segments.refs) or b.shape[0] != len(segments.refs):
        raise ValueError("stacked factors must have one group per segment")
    if flat.shape[0] == 0:
        # Keep zero gradients attached to every participating parameter.
        out = flat.new_zeros(0, b.shape[-1]) + (a.sum() + b.sum() + flat.sum()) * 0
    elif x.is_cuda and x.dtype in {torch.bfloat16, torch.float16}:
        from .kernels.cute_grouped_lora_quack import quack_grouped_lora

        out = quack_grouped_lora(flat, a, b, list(segments.counts))
    else:
        out = torch.cat(
            [
                (chunk @ a[i]) @ b[i]
                for i, chunk in enumerate(torch.split(flat, list(segments.counts)))
            ],
            dim=0,
        )
    return out.reshape(*shape[:-1], b.shape[-1])
