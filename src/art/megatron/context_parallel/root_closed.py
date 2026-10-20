"""Opt-in root ownership. Placement leases survive wave changes.

The scheduler may bin-pack a root when first admitted, then must retain its
first_rank/rank_count until its backward completes. Wave offsets may change.
Each root is padded at its tail (loss weight zero) to ``alignment``. Padding
inside trees must not change parent positions: pad only the final leaf.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from math import lcm

RootPlacements = tuple[tuple[int, int, int, int] | tuple[int, int, int, int, bool], ...]
_PLACEMENTS: ContextVar[RootPlacements] = ContextVar("art_root_placements", default=())


def current_root_placements() -> RootPlacements:
    return _PLACEMENTS.get()


@contextmanager
def execution(placements: RootPlacements) -> Iterator[None]:
    """Scope exact padded-wave ownership through forward/recompute/backward.

    All participating ranks enter the same synchronous scope. The caller
    supplies already padded complete roots, matching the packing exactly;
    retain these placements until every backward graph has completed.
    An empty tuple temporarily selects ART's global planning policy.
    """
    token = _PLACEMENTS.set(tuple(placements))
    try:
        yield
    finally:
        _PLACEMENTS.reset(token)


@lru_cache(maxsize=1024)
def root_owner_spans(length: int, rank_count: int, alignment: int = 128):
    """Cache root-relative contiguous spans, with no batch dependencies."""
    if length <= 0 or alignment <= 0 or alignment % 64 or length % alignment:
        raise ValueError("root length must be a positive multiple of alignment (64+)")
    if rank_count not in (1, 2, 4, 8) or length // alignment < rank_count:
        raise ValueError("root span requires one aligned block per participating rank")
    blocks, remainder = divmod(length // alignment, rank_count)
    start = 0
    result = []
    for rank in range(rank_count):
        end = start + (blocks + (rank < remainder)) * alignment
        result.append((start, end, rank))
        start = end
    return tuple(result)


def root_closed_assignment(*, row_spec, placements, cp_size: int, block_size: int):
    """Union root templates. One remote wave; no global ownership search."""
    from .types import TokenRange

    alignment = lcm(64, block_size)
    cursor = 0
    ranges, owners = [], []
    for placement in placements:
        start, end, first_rank, count = placement[:4]
        reverse = bool(placement[4]) if len(placement) == 5 else False
        if start != cursor or start % alignment or end % alignment:
            raise ValueError(
                "root placements must cover the row with block-aligned offsets"
            )
        if first_rank < 0 or first_rank + count > cp_size:
            raise ValueError("root rank span is outside the replica")
        for a, b, rank in root_owner_spans(end - start, count, alignment):
            ranges.append(TokenRange(start=start + a, end=start + b))
            owners.append(first_rank + (count - 1 - rank if reverse else rank))
        cursor = end
    if cursor != row_spec.valid_tokens:
        raise ValueError("root placements must cover every packed token")
    # Reject a lease boundary that cuts a tree or allows cross-root attention.
    for piece in row_spec.slices:
        qroot = next(
            (
                i
                for i, (a, b, *_) in enumerate(placements)
                if a <= piece.q_range.start and piece.q_range.end <= b
            ),
            None,
        )
        kroot = next(
            (
                i
                for i, (a, b, *_) in enumerate(placements)
                if a <= piece.k_range.start and piece.k_range.end <= b
            ),
            None,
        )
        if qroot is None or qroot != kroot:
            raise ValueError("root placement cuts an attention dependency")
    return tuple(ranges), tuple(owners), (0,) * len(ranges)
