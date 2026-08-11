"""Small collective helpers shared by Megatron checkpoint exporters."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import torch
import torch.distributed as dist


def distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def gather_objects[T](
    value: T,
    *,
    group: dist.ProcessGroup | None = None,
) -> tuple[T, ...]:
    if not distributed():
        return (value,)
    values: list[T | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return tuple(cast(T, item) for item in values)


def raise_distributed(
    error: BaseException | None,
    phase: str,
    *,
    group: dist.ProcessGroup | None = None,
) -> None:
    errors = gather_objects(None if error is None else repr(error), group=group)
    if not any(errors):
        return
    if error is not None:
        raise error
    raise RuntimeError(
        f"Another rank failed to {phase}: {next(item for item in errors if item)}"
    )


@contextmanager
def collective_errors(
    phase: str, *, group: dist.ProcessGroup | None = None
) -> Iterator[None]:
    error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        error = exc
    raise_distributed(error, phase, group=group)


def rank() -> int:
    return dist.get_rank() if distributed() else 0


def device() -> torch.device:
    return (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def dtype_from_name(name: str) -> torch.dtype:
    if isinstance(dtype := getattr(torch, name, None), torch.dtype):
        return dtype
    raise RuntimeError(f"Unsupported tensor dtype: {name!r}")


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")
