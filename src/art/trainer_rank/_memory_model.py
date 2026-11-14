"""Pure CPU sufficient statistics for conservative fixed + slope peak fits."""

from __future__ import annotations

from collections.abc import Iterable
import math

_MEMORY_PROFILE_VERSION = 2
_MEMORY_SLOPE_MIN_TOKENS = 8192
_MEMORY_COLD_EXTRAPOLATION_FACTOR = 1.75
MemoryPoint = tuple[int, int]


def merge_points(*observations: Iterable[MemoryPoint]) -> tuple[MemoryPoint, ...]:
    """Upper concave hull: preserve max(peak - slope*tokens) for every slope.

    Keep both endpoints, including the longest root used as the slope anchor.
    Hull(union(hulls)) == hull(union(points)), so incremental and restored
    profiles have exactly the same fit regardless of order or partitioning.
    """
    peaks: dict[int, int] = {}
    for points in observations:
        for tokens, peak in points:
            if (
                type(tokens) is not int
                or tokens <= 0
                or type(peak) is not int
                or peak < 0
            ):
                raise ValueError("invalid memory observation")
            peaks[tokens] = max(peak, peaks.get(tokens, 0))
    hull: list[MemoryPoint] = []
    for point in sorted(peaks.items()):
        while len(hull) >= 2:
            (a, x), (b, y) = hull[-2:]
            c, z = point
            if (y - x) * (c - b) > (z - y) * (b - a):
                break
            hull.pop()
        hull.append(point)
    return tuple(hull)


def fit_points(
    points: tuple[MemoryPoint, ...],
    *,
    min_tokens: int = _MEMORY_SLOPE_MIN_TOKENS,
    rate_points: tuple[MemoryPoint, ...] | None = None,
    absolute_floor: int = 0,
) -> tuple[int, MemoryPoint]:
    """Return an upward-rounded fixed term and exact rational slope (bytes, tokens)."""
    if not points:
        return 0, (0, 1)
    rates = points if rate_points is None else rate_points
    length, peak = rates[-1] if rates else (0, 0)
    numerator, denominator = (peak, length) if length >= min_tokens else (0, 1)
    fixed = max(
        absolute_floor,
        max(
            (compute * denominator - numerator * tokens + denominator - 1)
            // denominator
            for tokens, compute in points
        ),
    )
    return fixed, (numerator, denominator)


def predict(fixed: int, slope: MemoryPoint, tokens: int | float) -> int:
    numerator, denominator = slope
    if isinstance(tokens, int):
        return fixed + (numerator * tokens + denominator - 1) // denominator
    # Logical/packed normalization may produce a float. Keep fractional rows
    # and round the resulting byte demand upward, never round rows downward.
    return fixed + math.ceil(numerator * tokens / denominator)
