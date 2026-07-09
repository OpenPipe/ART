from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenLayoutIndex:
    ownership_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...]
    token_counts_by_rank: tuple[int, ...]
