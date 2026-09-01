"""Vectorized canonical-tree construction must equal the scalar algorithm.

``build_canonical_prefix_tree`` replaces the original per-position Python
comparison loop with one tensor comparison per shared-segment scan and hashes
row content from tensor bytes. Both changes must be behavior-preserving: the
segments, decision indices, and every fingerprint must match the scalar
reference implementation (reproduced verbatim below from the sealed research
source) on random trees and on the edge cases that stress sibling ordering.
"""

from __future__ import annotations

import hashlib
import random
import struct
from typing import Any

import pytest
import torch

from art.trainer_rank._prefix_tree_planner import (
    build_canonical_prefix_tree,
    canonical_token_row_fingerprint,
)

_U64 = struct.Struct("<Q")
_I64 = struct.Struct("<q")
_ROW_DOMAIN = b"art.trainer_rank.canonical_token_row.v2\0"
_ROWS_DOMAIN = b"art.trainer_rank.canonical_token_rows.v2\0"


# --- Scalar reference (sealed research algorithm, verbatim) -----------------


def _reference_row_fingerprint(tokens: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(_ROW_DOMAIN)
    for token in tokens:
        digest.update(_I64.pack(int(token)))
    digest.update(_U64.pack(len(tokens)))
    return digest.hexdigest()


def _reference_rows_fingerprint(rows: tuple[tuple[int, ...], ...]) -> str:
    digest = hashlib.sha256()
    digest.update(_ROWS_DOMAIN)
    digest.update(_U64.pack(len(rows)))
    for row in rows:
        digest.update(_U64.pack(len(row)))
        digest.update(bytes.fromhex(_reference_row_fingerprint(row)))
    return digest.hexdigest()


def _reference_segments(
    rows: tuple[tuple[int, ...], ...],
) -> list[tuple[int, int | None, tuple[int, ...], int, int, int]]:
    lengths = tuple(len(row) for row in rows)
    segments: list[tuple[int, int | None, tuple[int, ...], int, int, int]] = []
    tasks: list[tuple[tuple[int, ...], int, int | None]] = [
        (tuple(range(len(rows))), 0, None)
    ]
    while tasks:
        indices, start, parent_index = tasks.pop()
        active = tuple(index for index in indices if lengths[index] > start)
        if not active:
            continue
        depth = 1 if parent_index is None else segments[parent_index][5] + 1
        if len(active) == 1:
            segments.append(
                (len(segments), parent_index, active, start, lengths[active[0]], depth)
            )
            continue
        shared_end = min(lengths[index] for index in active)
        reference = rows[active[0]]
        cursor = start
        while cursor < shared_end and all(
            rows[index][cursor] == reference[cursor] for index in active[1:]
        ):
            cursor += 1
        if cursor > start:
            segment_index = len(segments)
            segments.append((segment_index, parent_index, active, start, cursor, depth))
            tasks.append((active, cursor, segment_index))
            continue
        groups: dict[int, list[int]] = {}
        sibling_tokens: list[int] = []
        for sequence_index in active:
            token = rows[sequence_index][start]
            if token not in groups:
                groups[token] = []
                sibling_tokens.append(token)
            groups[token].append(sequence_index)
        for token in reversed(sibling_tokens):
            tasks.append((tuple(groups[token]), start, parent_index))
    return segments


def _assert_equivalent(rows: tuple[tuple[int, ...], ...], inputs: Any) -> None:
    tree = build_canonical_prefix_tree(inputs)
    expected = _reference_segments(rows)
    actual = [
        (s.index, s.parent_index, s.sequence_indices, s.start, s.end, s.depth)
        for s in tree.segments
    ]
    assert actual == expected
    assert tree.sequence_lengths == tuple(len(row) for row in rows)
    assert tree.content_fingerprint == _reference_rows_fingerprint(rows)


def _random_rows(rng: random.Random) -> tuple[tuple[int, ...], ...]:
    """Trees with shared roots, branching at random depths, duplicates."""

    alphabet = rng.choice((2, 3, 50))
    row_count = rng.randint(1, 12)
    root = tuple(rng.randrange(alphabet) for _ in range(rng.randint(0, 6)))
    rows: list[tuple[int, ...]] = []
    for _ in range(row_count):
        if rows and rng.random() < 0.15:
            rows.append(rows[rng.randrange(len(rows))])  # exact duplicate
            continue
        if rows and rng.random() < 0.15:
            source = rows[rng.randrange(len(rows))]
            cut = rng.randint(1, len(source))
            rows.append(source[:cut])  # strict prefix of another row
            continue
        rows.append(
            root + tuple(rng.randrange(alphabet) for _ in range(rng.randint(1, 9)))
        )
    return tuple(rows)


@pytest.mark.parametrize("seed", range(300))
def test_vectorized_build_matches_scalar_reference(seed: int) -> None:
    rng = random.Random(20260901 + seed)
    rows = _random_rows(rng)
    _assert_equivalent(rows, rows)
    _assert_equivalent(
        rows, tuple(torch.tensor(row, dtype=torch.long) for row in rows)
    )


def test_int32_and_cuda_layout_inputs_normalize_to_the_same_tree() -> None:
    rows = ((1, 2, 3, 4), (1, 2, 9), (1, 2, 3, 7, 8))
    int32 = tuple(torch.tensor(row, dtype=torch.int32) for row in rows)
    tree_a = build_canonical_prefix_tree(rows)
    tree_b = build_canonical_prefix_tree(int32)
    assert tree_a.fingerprint == tree_b.fingerprint


def test_tensor_row_fingerprint_matches_scalar_encoding() -> None:
    rows = ((0,), (1, -1, 2**62), tuple(range(5_000)))
    for row in rows:
        assert canonical_token_row_fingerprint(
            torch.tensor(row, dtype=torch.long)
        ) == _reference_row_fingerprint(row)
