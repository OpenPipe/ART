from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import functools
import threading
from typing import Any, Literal, TypeVar, cast

import torch
import torch.nn.functional as F
import torch.utils.deterministic

POLICY_VERSION = "dense-root-v1"
ROOT_ALIGNMENT = 128
NumericalPolicy = Literal["native", "canonical"]
_F = TypeVar("_F", bound=Callable[..., Any])


@dataclass(frozen=True)
class RootLayout:
    """Logical lengths in physical packing order; every root starts at row zero."""

    lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "lengths", tuple(self.lengths))
        if not self.lengths or any(type(n) is not int or n <= 0 for n in self.lengths):
            raise ValueError("canonical roots require positive integer lengths")

    @property
    def padded_lengths(self) -> tuple[int, ...]:
        return tuple(n + (-n % ROOT_ALIGNMENT) for n in self.lengths)

    @property
    def rows(self) -> int:
        return sum(self.padded_lengths)

    def slices(self) -> Iterator[slice]:
        start = 0
        for length in self.padded_lengths:
            yield slice(start, start + length)
            start += length

    def pad(self, roots: list[torch.Tensor], *, value: int = 0) -> torch.Tensor:
        """Pad each root along its first axis, then pack; labels use value=-100."""
        if tuple(len(root) for root in roots) != self.lengths:
            raise ValueError("root tensors do not match the logical layout")
        pieces = []
        for root, padded in zip(roots, self.padded_lengths, strict=True):
            padding = [0, 0] * (root.ndim - 1) + [0, padded - len(root)]
            pieces.append(F.pad(root, padding, value=value))
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    def unpad(self, tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if tensor.shape[0] != self.rows:
            raise ValueError("tensor does not match canonical padded layout")
        return tuple(
            tensor[part][:length]
            for part, length in zip(self.slices(), self.lengths, strict=True)
        )


_LAYOUT: ContextVar[RootLayout | None] = ContextVar(
    "art_canonical_layout", default=None
)
# Deterministic-algorithm and allocation settings are process-wide. ART trainer
# kernels execute synchronously; serialize policy scopes across worker threads.
_EXECUTION_LOCK = threading.RLock()


def current_layout() -> RootLayout | None:
    return _LAYOUT.get()


def capture(function: _F) -> _F:
    """Carry immutable root metadata onto autograd's recomputation thread.

    The execution scope stays open through backward, retaining process settings.
    Do not acquire its thread lock on an autograd worker: the caller holds it.
    """
    layout = current_layout()

    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        token = _LAYOUT.set(layout)
        try:
            return function(*args, **kwargs)
        finally:
            _LAYOUT.reset(token)

    return cast(_F, wrapped)


@contextmanager
def execution(
    layout: RootLayout | None, *, numerical_policy: NumericalPolicy = "canonical"
) -> Iterator[None]:
    """One synchronous forward/recompute/backward scope; no async suspension.

    Native calls are no-ops outside a scope and explicitly mask an outer layout.
    Allocate outputs normally, with deterministic algorithms still enabled but
    debug fills disabled, matching the qualified elastic-canonical fast policy.
    """
    if numerical_policy not in ("native", "canonical"):
        raise ValueError(f"unknown numerical policy: {numerical_policy}")
    if numerical_policy == "canonical" and layout is None:
        raise ValueError("canonical execution requires a root layout")
    with _EXECUTION_LOCK:
        token = _LAYOUT.set(layout if numerical_policy == "canonical" else None)
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        fill = getattr(torch.utils.deterministic, "fill_uninitialized_memory")
        try:
            if numerical_policy == "canonical":
                torch.use_deterministic_algorithms(True, warn_only=False)
                setattr(torch.utils.deterministic, "fill_uninitialized_memory", False)
            yield
        finally:
            setattr(torch.utils.deterministic, "fill_uninitialized_memory", fill)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            _LAYOUT.reset(token)
