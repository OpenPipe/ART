from __future__ import annotations

from collections.abc import Callable
import traceback
from typing import TypeVar

import torch.distributed as dist

T = TypeVar("T")


def rank0_checked(label: str, check: Callable[[], T]) -> T | None:
    """Run a check on rank 0, then make every rank fail together."""

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    result: T | None = None
    error: str | None = None
    cause: Exception | None = None
    if rank == 0:
        try:
            result = check()
        except Exception as exc:
            cause = exc
            error = traceback.format_exc()
    payload = [error]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    _raise_together(label, payload[0], cause)
    return result


def all_ranks_checked(label: str, check: Callable[[], None]) -> None:
    """Run a local check everywhere, then report every rank's failures together."""

    error: str | None = None
    cause: Exception | None = None
    try:
        check()
    except Exception as exc:
        cause = exc
        error = traceback.format_exc()
    if not (dist.is_available() and dist.is_initialized()):
        _raise_together(label, error, cause)
        return
    errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    combined = "\n".join(
        f"rank {rank}:\n{message}"
        for rank, message in enumerate(errors)
        if message is not None
    )
    _raise_together(label, combined or None, cause)


def _raise_together(
    label: str,
    error: str | None,
    cause: Exception | None,
) -> None:
    if error is None:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    raise AssertionError(f"{label} failed:\n{error}") from cause
