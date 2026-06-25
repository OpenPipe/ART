from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SharedPrefixSegment:
    group_id: int
    parent_id: int
    start: int
    end: int
    family_index: int
    ancestors: tuple[int, ...]

    @property
    def depth(self) -> int:
        return len(self.ancestors)


@dataclass(frozen=True, slots=True)
class SharedPrefixRowTree:
    row_index: int
    valid_tokens: int
    segments: tuple[SharedPrefixSegment, ...]


def parse_shared_prefix_tree(
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    ignore_padding_group_id: int = -1,
) -> tuple[SharedPrefixRowTree, ...]:
    if group_ids.shape != parent_ids.shape:
        raise RuntimeError(
            "group_ids and parent_ids must share shape, got "
            f"{tuple(group_ids.shape)} vs {tuple(parent_ids.shape)}"
        )
    if group_ids.ndim != 2:
        raise RuntimeError(
            "group_ids and parent_ids must be rank-2 packed tensors, got "
            f"{group_ids.ndim}"
        )
    return tuple(
        parse_shared_prefix_row(
            group_ids=group_ids[row_index],
            parent_ids=parent_ids[row_index],
            row_index=row_index,
            ignore_padding_group_id=ignore_padding_group_id,
        )
        for row_index in range(int(group_ids.shape[0]))
    )


def parse_shared_prefix_row(
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    row_index: int = 0,
    ignore_padding_group_id: int = -1,
) -> SharedPrefixRowTree:
    if group_ids.shape != parent_ids.shape:
        raise RuntimeError(
            "group_ids and parent_ids must share shape, got "
            f"{tuple(group_ids.shape)} vs {tuple(parent_ids.shape)}"
        )
    if group_ids.ndim != 1:
        raise RuntimeError(
            f"group_ids and parent_ids must be rank-1 row tensors, got {group_ids.ndim}"
        )

    valid_tokens = _valid_length(
        group_ids,
        parent_ids,
        ignore_padding_group_id=ignore_padding_group_id,
    )
    if valid_tokens == 0:
        return SharedPrefixRowTree(row_index=row_index, valid_tokens=0, segments=())

    runs = _scan_runs(group_ids[:valid_tokens], parent_ids[:valid_tokens])
    first_segment_by_group: dict[int, SharedPrefixSegment] = {}
    family_by_group: dict[int, int] = {}
    ancestors_by_group: dict[int, tuple[int, ...]] = {}
    segments: list[SharedPrefixSegment] = []
    next_family_index = 0

    seen_groups: set[int] = set()
    repeated_groups: dict[int, int] = {}
    for _start, _end, group_id, _parent_id in runs:
        if group_id in seen_groups and group_id != ignore_padding_group_id:
            repeated_groups[group_id] = repeated_groups.get(group_id, 1) + 1
        seen_groups.add(group_id)
    if repeated_groups:
        raise RuntimeError(
            "Shared-prefix metadata requires contiguous group runs per row, "
            f"found repeats in row {row_index}: {repeated_groups}"
        )

    for start, end, group_id, parent_id in runs:
        is_root = group_id == parent_id or (
            start == 0 and parent_id == ignore_padding_group_id
        )
        if is_root:
            family_index = next_family_index
            next_family_index += 1
            ancestors: tuple[int, ...] = ()
        else:
            parent_segment = first_segment_by_group.get(parent_id)
            if parent_segment is None:
                raise RuntimeError(
                    "Shared-prefix run points to a missing parent run: "
                    f"row={row_index}, group_id={group_id}, parent_id={parent_id}"
                )
            if int(parent_segment.end) > int(start):
                raise RuntimeError(
                    "Shared-prefix parent run must end before its child starts: "
                    f"row={row_index}, group_id={group_id}, parent_id={parent_id}"
                )
            family_index = family_by_group[parent_id]
            ancestors = (*ancestors_by_group[parent_id], parent_id)

        segment = SharedPrefixSegment(
            group_id=group_id,
            parent_id=parent_id,
            start=start,
            end=end,
            family_index=family_index,
            ancestors=ancestors,
        )
        first_segment_by_group[group_id] = segment
        family_by_group[group_id] = family_index
        ancestors_by_group[group_id] = ancestors
        segments.append(segment)

    return SharedPrefixRowTree(
        row_index=row_index,
        valid_tokens=valid_tokens,
        segments=tuple(segments),
    )


def _valid_length(
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    *,
    ignore_padding_group_id: int,
) -> int:
    valid_mask = group_ids != ignore_padding_group_id
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return 0
    if not bool(valid_mask[:valid_count].all().item()):
        raise RuntimeError("Padding tokens must be a contiguous tail")
    return _infer_terminal_padding_length(
        group_ids[:valid_count],
        parent_ids[:valid_count],
    )


def _infer_terminal_padding_length(
    group_row: torch.Tensor,
    parent_row: torch.Tensor,
) -> int:
    if group_row.numel() == 0:
        return 0
    runs = _scan_runs(group_row, parent_row)
    if len(runs) < 2:
        return int(group_row.numel())
    last_start, _last_end, last_group_id, last_parent_id = runs[-1]
    if last_parent_id >= 0:
        return int(group_row.numel())
    terminal_pair = (last_group_id, last_parent_id)
    if any(
        (group_id, parent_id) == terminal_pair
        for _start, _end, group_id, parent_id in runs[:-1]
    ):
        return last_start
    return int(group_row.numel())


def _scan_runs(
    group_row: torch.Tensor,
    parent_row: torch.Tensor,
) -> list[tuple[int, int, int, int]]:
    length = int(group_row.numel())
    if length == 0:
        return []

    group_changes = group_row[1:] != group_row[:-1]
    parent_changes = parent_row[1:] != parent_row[:-1]
    inconsistent_parent = torch.nonzero(
        torch.logical_not(group_changes) & parent_changes,
        as_tuple=False,
    ).flatten()
    if int(inconsistent_parent.numel()) > 0:
        mismatch_index = int(inconsistent_parent[0].item()) + 1
        prior_boundaries = torch.nonzero(
            group_changes[: mismatch_index - 1],
            as_tuple=False,
        ).flatten()
        start = (
            0
            if int(prior_boundaries.numel()) == 0
            else int(prior_boundaries[-1].item()) + 1
        )
        group_id = int(group_row[start].item())
        raise RuntimeError(
            "Found one group run with inconsistent parent ids: "
            f"group_id={group_id}, start={start}, end={mismatch_index}"
        )

    run_starts = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=group_row.device),
            torch.nonzero(group_changes, as_tuple=False).flatten() + 1,
        )
    )
    run_ends = torch.cat(
        (
            run_starts[1:],
            torch.tensor([length], dtype=torch.int64, device=group_row.device),
        )
    )
    starts = run_starts.to(device="cpu").tolist()
    ends = run_ends.to(device="cpu").tolist()
    group_ids = group_row.index_select(0, run_starts).to(device="cpu").tolist()
    parent_ids = parent_row.index_select(0, run_starts).to(device="cpu").tolist()
    return [
        (int(start), int(end), int(group_id), int(parent_id))
        for start, end, group_id, parent_id in zip(
            starts, ends, group_ids, parent_ids, strict=True
        )
    ]
