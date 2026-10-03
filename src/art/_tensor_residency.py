"""Observe tensors held outside autograd saved-variable hooks without owning them."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_observer: ContextVar[Callable[[Sequence[torch.Tensor]], None] | None] = ContextVar(
    "art_tensor_residency_observer", default=None
)


@torch.compiler.disable
def record_resident_tensors(tensors: Sequence[torch.Tensor]) -> None:
    if observer := _observer.get():
        observer(tensors)


@contextmanager
def observe_resident_tensors(
    observer: Callable[[Sequence[torch.Tensor]], None],
) -> Iterator[None]:
    token = _observer.set(observer)
    try:
        yield
    finally:
        _observer.reset(token)
