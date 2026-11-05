"""MVP packing: independent causal roots with aligned, unselected tails."""

from dataclasses import replace

import torch

from art.megatron.prefix_tree_packing import PrefixTreePack


def validate_single_roots(group_ids: torch.Tensor, parent_ids: torch.Tensor) -> None:
    valid = group_ids >= 0
    if bool(((group_ids[valid] != parent_ids[valid]) & (parent_ids[valid] >= 0)).any()):
        raise ValueError(
            "GLM-5.3 Flash MVP requires single-root packing; trees are unsupported."
        )


def pad_single_roots(batch: PrefixTreePack, *, cp_size: int) -> PrefixTreePack:
    # CP splits both halves of each root; every split must preserve 4-token pools.
    if cp_size < 1:
        raise ValueError("CP size must be positive.")
    alignment = 4 if cp_size == 1 else 8 * cp_size
    validate_single_roots(batch.group_ids, batch.parent_ids)
    tokens, groups, positions = [], [], []
    mapping = list(batch.positions_by_sequence)
    segments = []
    cursor = 0
    for segment in batch.segments:
        if segment.start != 0 or len(segment.sequence_indices) != 1:
            raise ValueError("GLM-5.3 Flash requires one complete root per datum.")
        length = segment.length
        padded = length + (-length % alignment)
        source = batch.tokens[0, segment.packed_start : segment.packed_start + length]
        tokens.append(torch.cat((source, source.new_zeros(padded - length))))
        groups.append(source.new_full((padded,), segment.group_id))
        # Keep the same complete-pool candidate width at every CP size. Extra
        # CP alignment is physical storage, not part of the indexer's document:
        # torch.topk resolves tied visible pool scores differently when even
        # masked future candidates change width.
        pool_aligned = length + (-length % 4)
        positions.append(
            torch.cat(
                (
                    torch.arange(pool_aligned, device=source.device),
                    source.new_full((padded - pool_aligned,), -1),
                )
            )
        )
        mapping[segment.sequence_indices[0]] = torch.arange(
            cursor, cursor + length, device=source.device
        )
        segments.append(replace(segment, packed_start=cursor))
        cursor += padded
    if not tokens:
        return batch
    group_ids = torch.cat(groups).unsqueeze(0)
    return PrefixTreePack(
        tokens=torch.cat(tokens).unsqueeze(0),
        group_ids=group_ids,
        parent_ids=group_ids.clone(),
        position_ids=torch.cat(positions).unsqueeze(0),
        positions_by_sequence=tuple(mapping),
        segments=tuple(segments),
    )
