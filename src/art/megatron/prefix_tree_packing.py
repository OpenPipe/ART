from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrefixTreePack:
    tokens: torch.Tensor
    group_ids: torch.Tensor
    parent_ids: torch.Tensor
    position_ids: torch.Tensor
    positions_by_sequence: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _PrefixSegment:
    sequence_indices: tuple[int, ...]
    start: int
    end: int
    group_id: int
    parent_id: int


def prefix_tree_pack(
    sequences: Iterable[torch.Tensor],
    *,
    max_depth: int,
    shareable_lengths: Iterable[int] | None = None,
) -> PrefixTreePack:
    """Pack token sequences by storing prefix trees once.

    This is the small packing step that lets `TrainerRank.dp_rank_forward()` run one
    model pass over a compact prefix tree instead of replaying the same prompt
    tokens for every request. Think of each input sequence as a path through a
    tree: when several paths start with the same tokens, this function writes
    that shared segment once, then writes each branch after it.

    Args:
        sequences: 1-D token tensors to pack.
        max_depth: How many nested prefix-tree levels to emit. `0` disables
            tree sharing and writes each sequence as its own root segment. `1`
            shares the first common segment in each branch; larger values allow
            branches to contain shared sub-branches.

    Returns:
        `tokens` is the compact model input, shaped `[1, packed_length]`.
        `group_ids` and `parent_ids` describe the prefix tree to prefix-tree
        attention. Positions in the same emitted segment share a group, and each
        group points at the parent segment it continues from. Root groups point
        to themselves.
        `position_ids` keeps each token's original sequence position for
        positional embeddings/rotary attention.
        `positions_by_sequence` is the reverse index used after the model call
        to unpack logits, logprobs, or hidden states back into one tensor per
        original request.

    The implementation is a tiny radix-tree walk. It finds the longest prefix
    shared by the active sequences, emits that segment once, then partitions the
    remaining sequences by their next token while preserving first-seen order.
    Single sequences, empty branches, and branches past `max_depth` are emitted
    as ordinary unshared tails.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")

    tensors = tuple(_sequence_tensor(sequence) for sequence in sequences)
    if not tensors:
        return _empty_pack()
    share_limits = _shareable_lengths(tensors, shareable_lengths)

    device = tensors[0].device
    rows = tuple(tensor.detach().cpu().tolist() for tensor in tensors)
    segments = _prefix_segments(
        rows, max_depth=max_depth, shareable_lengths=share_limits
    )
    if not segments:
        return _empty_pack(len(tensors), device=device)

    token_chunks: list[torch.Tensor] = []
    group_chunks: list[torch.Tensor] = []
    parent_chunks: list[torch.Tensor] = []
    position_chunks: list[torch.Tensor] = []
    positions_by_sequence: list[list[torch.Tensor]] = [[] for _ in tensors]
    cursor = 0

    for planned in segments:
        segment = tensors[planned.sequence_indices[0]][planned.start : planned.end]
        packed_positions = torch.arange(cursor, cursor + len(segment), device=device)
        token_chunks.append(segment)
        group_chunks.append(torch.full_like(segment, planned.group_id))
        parent_chunks.append(torch.full_like(segment, planned.parent_id))
        position_chunks.append(torch.arange(planned.start, planned.end, device=device))
        for sequence_index in planned.sequence_indices:
            positions_by_sequence[sequence_index].append(packed_positions)
        cursor += len(segment)

    return PrefixTreePack(
        tokens=torch.cat(token_chunks).unsqueeze(0),
        group_ids=torch.cat(group_chunks).unsqueeze(0),
        parent_ids=torch.cat(parent_chunks).unsqueeze(0),
        position_ids=torch.cat(position_chunks).unsqueeze(0),
        positions_by_sequence=tuple(
            torch.cat(chunks)
            if chunks
            else torch.empty(0, dtype=torch.long, device=device)
            for chunks in positions_by_sequence
        ),
    )


def estimate_prefix_tree_packed_tokens(
    sequences: Iterable[torch.Tensor],
    *,
    max_depth: int,
    shareable_lengths: Iterable[int] | None = None,
) -> int | None:
    """Return the exact packed token count without building a packed batch.

    The estimator intentionally only handles CPU tensors. For CUDA tensors, many
    tiny prefix probes would launch many tiny kernels, so callers should fall
    back to full packing instead.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")

    tensors: list[torch.Tensor] = []
    rows: list[list[int]] = []
    for sequence in sequences:
        tensor = _sequence_tensor(sequence)
        if tensor.device.type != "cpu":
            return None
        tensors.append(tensor)
        rows.append(tensor.tolist())
    share_limits = _shareable_lengths(tuple(tensors), shareable_lengths)

    return sum(
        segment.end - segment.start
        for segment in _prefix_segments(
            tuple(rows),
            max_depth=max_depth,
            shareable_lengths=share_limits,
        )
    )


def _local_position_pairs(
    local_global_positions: torch.Tensor,
    item_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = local_global_positions.reshape(-1).to(device=item_positions.device)
    local_positions = torch.nonzero(flat >= 0, as_tuple=False).reshape(-1)
    global_positions = flat.index_select(0, local_positions)
    source_offsets, local_offsets = _matching_offsets(item_positions, global_positions)
    return (
        local_positions.index_select(0, local_offsets).to("cpu"),
        source_offsets.to("cpu"),
    )


def _prefix_segments(
    rows: tuple[list[int], ...],
    *,
    max_depth: int,
    shareable_lengths: tuple[int, ...],
) -> tuple[_PrefixSegment, ...]:
    lengths = tuple(len(row) for row in rows)
    segments: list[_PrefixSegment] = []
    next_group_id = 1

    def emit(
        indices: tuple[int, ...],
        start: int,
        end: int,
        parent_group_id: int | None,
    ) -> int:
        nonlocal next_group_id
        group_id = next_group_id
        next_group_id += 1
        segments.append(
            _PrefixSegment(
                sequence_indices=indices,
                start=start,
                end=end,
                group_id=group_id,
                parent_id=group_id if parent_group_id is None else parent_group_id,
            )
        )
        return group_id

    def shared_end(indices: tuple[int, ...], start: int) -> int:
        end = min(min(lengths[index], shareable_lengths[index]) for index in indices)
        low = high = rows[indices[0]]
        for index in indices[1:]:
            row = rows[index]
            if row < low:
                low = row
            elif row > high:
                high = row
        while start < end:
            if low[start] != high[start]:
                break
            start += 1
        return start

    def branch_groups(indices: tuple[int, ...], start: int) -> list[tuple[int, ...]]:
        groups: dict[int, list[int]] = {}
        order: list[int] = []
        for index in indices:
            token = rows[index][start]
            if token not in groups:
                groups[token] = []
                order.append(token)
            groups[token].append(index)
        return [tuple(groups[token]) for token in order]

    def walk(
        indices: tuple[int, ...],
        start: int,
        parent_group_id: int | None,
        depth: int,
    ) -> None:
        active = tuple(index for index in indices if lengths[index] > start)
        if not active:
            return
        shareable = tuple(index for index in active if shareable_lengths[index] > start)
        for index in active:
            if index not in shareable:
                emit((index,), start, lengths[index], parent_group_id)
        if not shareable:
            return
        if (
            max_depth == 0
            or len(shareable) == 1
            or (parent_group_id is not None and depth >= max_depth)
        ):
            for index in shareable:
                emit((index,), start, lengths[index], parent_group_id)
            return

        end = shared_end(shareable, start)
        if end > start:
            walk(
                shareable,
                end,
                emit(shareable, start, end, parent_group_id),
                depth + 1,
            )
            return
        for group in branch_groups(shareable, start):
            walk(group, start, parent_group_id, depth)

    walk(tuple(range(len(rows))), 0, None, 0)
    return tuple(segments)


def _empty_pack(
    sequence_count: int = 0,
    *,
    device: torch.device | None = None,
) -> PrefixTreePack:
    flat = torch.empty(0, dtype=torch.long, device=device)
    row = flat.unsqueeze(0)
    return PrefixTreePack(
        tokens=row,
        group_ids=row,
        parent_ids=row,
        position_ids=row,
        positions_by_sequence=tuple(flat for _ in range(sequence_count)),
    )


def _sequence_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 1:
        raise ValueError(
            f"prefix_tree_pack expects 1-D tensors, got {tuple(tensor.shape)}"
        )
    return tensor.detach().to(dtype=torch.long).contiguous()


def _shareable_lengths(
    tensors: tuple[torch.Tensor, ...],
    lengths: Iterable[int] | None,
) -> tuple[int, ...]:
    if lengths is None:
        return tuple(int(tensor.numel()) for tensor in tensors)
    values = tuple(int(length) for length in lengths)
    if len(values) != len(tensors):
        raise ValueError(
            "shareable_lengths must have one entry per sequence, "
            f"got {len(values)} for {len(tensors)} sequences"
        )
    return tuple(
        max(0, min(value, int(tensor.numel())))
        for value, tensor in zip(values, tensors, strict=True)
    )


def _matching_offsets(
    positions: torch.Tensor,
    chunk_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(positions.numel()) == 0 or int(chunk_rows.numel()) == 0:
        empty = torch.empty(0, dtype=torch.long, device=positions.device)
        return empty, empty
    sorted_rows, order = chunk_rows.sort()
    indices = torch.searchsorted(sorted_rows, positions)
    in_bounds = indices < int(sorted_rows.numel())
    source_offsets = torch.arange(
        int(positions.numel()),
        device=positions.device,
        dtype=torch.long,
    )[in_bounds]
    found = indices[in_bounds]
    keep = sorted_rows.index_select(0, found) == positions.index_select(
        0,
        source_offsets,
    )
    return source_offsets[keep], order.index_select(0, found[keep])
