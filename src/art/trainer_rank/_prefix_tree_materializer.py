"""Tensor materialization for TrainerRank-owned prefix-tree layouts."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from art.megatron.prefix_tree_packing import PrefixTreePack, PrefixTreePackSegment

from ._prefix_tree_planner import CanonicalPrefixTree, PrefixTreeLayout


def materialize_prefix_tree_layout(
    sequences: Sequence[torch.Tensor],
    tree: CanonicalPrefixTree,
    layout: PrefixTreeLayout,
    *,
    verify_shared_tokens: bool = True,
) -> PrefixTreePack:
    """Materialize one already-selected layout without making policy choices.

    ``verify_shared_tokens`` re-checks that planned shared segments contain
    byte-identical tokens across member rows. Callers that already verified
    content identity (e.g. via a content-hash cache key over these exact
    tensors) may disable it; on CUDA inputs each comparison is a device sync.
    """

    tensors = tuple(
        sequence.detach().reshape(-1).to(dtype=torch.long) for sequence in sequences
    )
    if not tensors:
        raise ValueError("a nonempty layout requires at least one sequence")
    if tuple(int(tensor.numel()) for tensor in tensors) != tree.sequence_lengths:
        raise ValueError("sequence lengths do not match the canonical tree")
    if layout.tree_fingerprint != tree.fingerprint:
        raise ValueError("layout belongs to a different canonical tree")
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError("all sequences must be on the same device")
    device = tensors[0].device

    token_chunks: list[torch.Tensor] = []
    group_chunks: list[torch.Tensor] = []
    parent_chunks: list[torch.Tensor] = []
    position_chunks: list[torch.Tensor] = []
    positions_by_sequence: list[list[torch.Tensor]] = [[] for _ in tensors]
    packed_segments: list[PrefixTreePackSegment] = []
    cursor = 0
    for segment_index, segment in enumerate(layout.segments):
        source = tensors[segment.sequence_indices[0]][segment.start : segment.end]
        if verify_shared_tokens:
            for sequence_index in segment.sequence_indices[1:]:
                candidate = tensors[sequence_index][segment.start : segment.end]
                if not torch.equal(source, candidate):
                    raise ValueError("planned shared segment contains unequal tokens")
        group_id = segment_index + 1
        parent_id = (
            group_id
            if segment.parent_segment_index is None
            else segment.parent_segment_index + 1
        )
        packed_positions = torch.arange(
            cursor,
            cursor + segment.length,
            dtype=torch.long,
            device=device,
        )
        token_chunks.append(source)
        group_chunks.append(
            torch.full((segment.length,), group_id, dtype=torch.long, device=device)
        )
        parent_chunks.append(
            torch.full((segment.length,), parent_id, dtype=torch.long, device=device)
        )
        position_chunks.append(
            torch.arange(segment.start, segment.end, dtype=torch.long, device=device)
        )
        for sequence_index in segment.sequence_indices:
            positions_by_sequence[sequence_index].append(packed_positions)
        packed_segments.append(
            PrefixTreePackSegment(
                sequence_indices=segment.sequence_indices,
                start=segment.start,
                end=segment.end,
                packed_start=cursor,
                group_id=group_id,
                parent_id=parent_id,
            )
        )
        cursor += segment.length

    return PrefixTreePack(
        tokens=torch.cat(token_chunks).unsqueeze(0),
        group_ids=torch.cat(group_chunks).unsqueeze(0),
        parent_ids=torch.cat(parent_chunks).unsqueeze(0),
        position_ids=torch.cat(position_chunks).unsqueeze(0),
        positions_by_sequence=tuple(
            torch.cat(chunks) for chunks in positions_by_sequence
        ),
        segments=tuple(packed_segments),
    )
