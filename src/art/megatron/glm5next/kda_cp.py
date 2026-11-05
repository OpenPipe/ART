"""KDA CP boundaries and ART prefix-tree token exchanges.

FLA 0.5 CP cannot accept a parent initial state. For branching trees we expand
root-to-leaf paths, run the native CP kernel/conv halo, and retain one copy of
each token's output. Autograd sums all descendant contributions to the shared
prefix. This is an exact compatibility path with extra prefix computation.
The scalar-gate affine kernels in gdn/fla_cp.py cannot implement KDA's vector
decay, so they must not be called with channel-wise gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
import torch.distributed as dist

from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.gdn.gdn_prefix_tree import (
    GdnPackedExecutionSpec,
    parse_gdn_prefix_tree_segments,
)
from art.megatron.gdn.layout import (
    GdnCpExchangePlan,
    GdnCpPeerTransfer,
    exchange_rank_tensor_all_to_all,
)


def validate_boundaries(cu: Tensor, total: int) -> None:
    if cu.ndim != 1 or cu.dtype not in (torch.int32, torch.int64):
        raise ValueError("KDA boundaries must be a 1D integer tensor")
    values = cu.detach().cpu().tolist()
    if len(values) < 2 or values[0] != 0 or values[-1] != total:
        raise ValueError(f"KDA boundaries must cover [0,{total}]")
    if any(end <= start for start, end in zip(values[:-1], values[1:], strict=True)):
        raise ValueError("KDA boundaries must be strictly increasing")


@torch.compiler.disable
def build_context(cu: Tensor | None, group: Any, kernel_size: int) -> Any:
    from fla.ops.cp import build_cp_context

    from .kda_cp_kernels import install_kda_cp_precision

    if cu is None:
        raise ValueError("KDA CP requires global sequence boundaries")
    world = dist.get_world_size(group)
    total = int(cu[-1].item())
    if total % world or total // world < kernel_size:
        raise ValueError(
            "FLA CP requires equal contiguous shards at least as long as the convolution kernel"
        )
    install_kda_cp_precision()
    return build_cp_context(
        cu.to(dtype=torch.int32), group=group, conv1d_kernel_size=kernel_size
    )


def reference_gather(x: Tensor, group: Any) -> Tensor:
    """CPU rehearsal only: autograd collectives, without claiming FLA parity."""
    from torch.distributed.nn.functional import all_gather

    return torch.cat(all_gather(x.contiguous(), group=group), dim=1)


@dataclass(frozen=True)
class KdaPrefixPlan:
    indices: Tensor
    output_positions: Tensor
    original_positions: Tensor
    cu_seqlens: Tensor
    expanded_tokens: int
    original_tokens: int
    source_layout: TokenLayoutIndex
    to_packed: GdnCpExchangePlan | None = None
    from_packed: GdnCpExchangePlan | None = None


def _reverse(plan: GdnCpExchangePlan) -> GdnCpExchangePlan:
    return GdnCpExchangePlan(
        cp_size=plan.cp_size,
        source_token_counts_by_rank=plan.dest_token_counts_by_rank,
        dest_token_counts_by_rank=plan.source_token_counts_by_rank,
        transfers=tuple(
            GdnCpPeerTransfer(
                source_rank=p.dest_rank,
                dest_rank=p.source_rank,
                token_count=p.token_count,
                source_positions_cpu=p.dest_positions_cpu,
                dest_positions_cpu=p.source_positions_cpu,
                source_positions_tensor=p.dest_positions_tensor,
                dest_positions_tensor=p.source_positions_tensor,
            )
            for p in plan.transfers
        ),
        cross_rank_token_count_override=plan.cross_rank_token_count_override,
    )


def _exchanges(
    layout: TokenLayoutIndex,
    expanded: list[int],
    chosen: dict[int, int],
    device: torch.device,
) -> tuple[GdnCpExchangePlan, GdnCpExchangePlan]:
    world = len(layout.token_counts_by_rank)
    shard = len(expanded) // world
    owners: dict[int, tuple[int, int]] = {}
    for rank, ranges in enumerate(layout.ownership_ranges_by_rank):
        for start, end, offset in ranges:
            for position in range(start, end):
                if position in owners:
                    raise ValueError("KDA attention token ownership overlaps")
                owners[position] = (rank, offset + position - start)
    inputs: dict[tuple[int, int], tuple[list[int], list[int]]] = {}
    outputs: dict[tuple[int, int], tuple[list[int], list[int]]] = {}
    for packed_position, original in enumerate(expanded):
        if original < 0:
            continue
        if original not in owners:
            raise ValueError(f"KDA prefix token {original} has no attention owner")
        source, source_position = owners[original]
        dest, dest_position = divmod(packed_position, shard)
        indices = inputs.setdefault((source, dest), ([], []))
        indices[0].append(source_position)
        indices[1].append(dest_position)
        if chosen[original] == packed_position:
            indices = outputs.setdefault((dest, source), ([], []))
            indices[0].append(dest_position)
            indices[1].append(source_position)

    def plan(
        pairs: dict, source_counts: tuple[int, ...], dest_counts: tuple[int, ...]
    ) -> GdnCpExchangePlan:
        return GdnCpExchangePlan(
            cp_size=world,
            source_token_counts_by_rank=source_counts,
            dest_token_counts_by_rank=dest_counts,
            transfers=tuple(
                GdnCpPeerTransfer(
                    source_rank=source,
                    dest_rank=dest,
                    token_count=len(src),
                    source_positions_cpu=tuple(src),
                    dest_positions_cpu=tuple(dst),
                    source_positions_tensor=torch.tensor(
                        src, device=device, dtype=torch.long
                    ),
                    dest_positions_tensor=torch.tensor(
                        dst, device=device, dtype=torch.long
                    ),
                )
                for (source, dest), (src, dst) in sorted(pairs.items())
            ),
            # Always use the autograd collective boundary at CP>1, including
            # all-local maps: its zero-fill also covers dummy/padded tokens.
            cross_rank_token_count_override=max(
                1, sum(len(src) for (a, b), (src, _) in pairs.items() if a != b)
            ),
        )

    return plan(inputs, layout.token_counts_by_rank, (shard,) * world), plan(
        outputs, (shard,) * world, layout.token_counts_by_rank
    )


def build_prefix_plan(
    spec: GdnPackedExecutionSpec,
    *,
    device: torch.device,
    source_layout: TokenLayoutIndex | None = None,
    cp_size: int = 1,
) -> KdaPrefixPlan:
    """Compile ART's parsed GDN tree into independent varlen leaf paths."""
    if source_layout is None:
        if cp_size != 1:
            raise ValueError("KDA prefix CP requires the ART attention token layout")
        count = spec.batch_size * spec.sequence_length
        source_layout = TokenLayoutIndex((((0, count, 0),),), (count,))
    if len(source_layout.token_counts_by_rank) != cp_size:
        raise ValueError("KDA token layout CP size does not match process group")
    parents = spec.tree_parent_indices
    nonleaves = {parent for parent in parents if parent >= 0}
    expanded: list[int] = []
    chosen: dict[int, int] = {}
    boundaries = [0]
    for leaf in range(len(spec.tree_segments)):
        if leaf in nonleaves:
            continue
        path = []
        current = leaf
        while current >= 0:
            path.append(current)
            current = parents[current]
        for index in reversed(path):
            node = spec.tree_segments[index]
            for position in range(node.start, node.end):
                original = node.row_index * spec.sequence_length + position
                chosen.setdefault(original, len(expanded))
                expanded.append(original)
        boundaries.append(len(expanded))
    if cp_size > 1 and expanded:
        # FLA partitions evenly. A separate dummy sequence cannot change any
        # real sequence's history, and aligns local shards to KDA's 64 tokens.
        padding = -len(expanded) % (cp_size * 64)
        if padding:
            expanded.extend([-1] * padding)
            boundaries.append(len(expanded))
    to_packed = from_packed = None
    if cp_size > 1 and expanded:
        to_packed, from_packed = _exchanges(source_layout, expanded, chosen, device)
    return KdaPrefixPlan(
        indices=torch.tensor(expanded, device=device, dtype=torch.long),
        output_positions=torch.tensor(
            list(chosen.values()), device=device, dtype=torch.long
        ),
        original_positions=torch.tensor(list(chosen), device=device, dtype=torch.long),
        cu_seqlens=torch.tensor(boundaries, device=device, dtype=torch.int32),
        expanded_tokens=len(expanded),
        original_tokens=spec.real_token_count,
        source_layout=source_layout,
        to_packed=to_packed,
        from_packed=from_packed,
    )


@torch.compiler.disable
def prefix_tree_forward(module: Any, hidden: Tensor, state: Any, group: Any) -> Tensor:
    cp_size = dist.get_world_size(group) if group is not None else 1
    rank = dist.get_rank(group) if group is not None else 0
    if getattr(state, "gdn_hidden_layout", "attention") != "attention":
        raise ValueError(
            "KDA expects ART attention layout; GDN island hooks must exclude KDA"
        )
    spec = getattr(state, "gdn_execution_spec", None)
    if spec is None:
        if cp_size > 1:
            raise ValueError(
                "Prefix KDA CP requires a prebuilt global gdn_execution_spec"
            )
        spec = parse_gdn_prefix_tree_segments(state.group_ids, state.parent_ids)
    layout = getattr(getattr(state, "rank_plan", None), "token_layout_index", None)
    layout = layout or getattr(state, "kda_token_layout", None)
    # State/model_state is per batch, safe across layers and checkpoint replay.
    cache = getattr(state, "model_state", None)
    key = f"_art_kda_prefix_plan_cp{cp_size}_{hidden.device}"
    plan = cache.get(key) if cache is not None else None
    if plan is None:
        plan = build_prefix_plan(
            spec, device=hidden.device, source_layout=layout, cp_size=cp_size
        )
        if cache is not None:
            cache[key] = plan
    flat = hidden.transpose(0, 1).reshape(-1, hidden.shape[-1])
    if cp_size == 1:
        if tuple(hidden.shape[:2]) != (spec.sequence_length, spec.batch_size):
            raise ValueError("KDA prefix spec shape does not match hidden states")
        if not plan.expanded_tokens:
            return hidden * 0
        packed = flat.index_select(0, plan.indices)
        output = module._forward_packed(packed.unsqueeze(1), plan.cu_seqlens).squeeze(1)
        result = torch.zeros_like(flat).index_copy(
            0, plan.original_positions, output.index_select(0, plan.output_positions)
        )
    else:
        if hidden.shape[1] != 1:
            raise ValueError(
                "KDA prefix CP requires compact [local_tokens,1,D] attention layout"
            )
        count = plan.source_layout.token_counts_by_rank[rank]
        if flat.shape[0] < count:
            raise ValueError(
                "KDA attention layout has fewer tokens than its ownership plan"
            )
        if not plan.expanded_tokens:
            return hidden * 0
        if plan.to_packed is None or plan.from_packed is None:
            raise RuntimeError("KDA CP prefix exchanges are missing")
        packed = exchange_rank_tensor_all_to_all(
            flat[:count],
            plan.to_packed,
            rank=rank,
            group=group,
            backward_plan=_reverse(plan.to_packed),
        )
        output = module._forward_packed(
            packed.unsqueeze(1), plan.cu_seqlens, group
        ).squeeze(1)
        result = exchange_rank_tensor_all_to_all(
            output,
            plan.from_packed,
            rank=rank,
            group=group,
            backward_plan=_reverse(plan.from_packed),
        )
        if flat.shape[0] > count:
            result = torch.cat((result, flat[count:] * 0), dim=0)
    return (
        result.reshape(hidden.shape[1], hidden.shape[0], -1)
        .transpose(0, 1)
        .contiguous()
    )
