"""Optional physical-row ceiling; admission still prices the selected layout."""

from collections.abc import Sequence


def capped_owners(
    owners: tuple[int, ...],
    balanced: tuple[int, ...],
    lengths: Sequence[int],
    cp_size: int,
    limit: int,
) -> tuple[int, ...]:
    if type(limit) is not int or limit <= 0:
        raise ValueError("CP owned-token ceiling must be a positive integer")

    def fits(candidate):
        if len(candidate) != len(lengths):
            raise ValueError("CP owner/chunk inventory mismatch")
        loads = [0] * cp_size
        for owner, length in zip(candidate, lengths, strict=True):
            if not 0 <= owner < cp_size or length <= 0:
                raise ValueError("Invalid CP owner/chunk geometry")
            loads[owner] += length
        return max(loads, default=0) <= limit

    if fits(owners):
        return owners
    if fits(balanced):
        return balanced
    raise ValueError("Neither CP ownership candidate fits the owned-token ceiling")
