"""Shard saved root activations while keeping CP1 operator geometry.

Every group member executes the same full-root forward and backward. Only
checkpoint storage is divided; no partial parameter gradients are produced.
Use an outer group of CP1 models, disable automatic DP gradient reduction,
and export one complete parameter-gradient copy without reduction.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import torch
import torch.distributed as dist

CheckpointPartition = tuple[int, tuple[int, int, int]]
_PARTITION: ContextVar[CheckpointPartition | None] = ContextVar(
    "art_checkpoint_partition", default=None
)


def current_checkpoint_partition() -> CheckpointPartition | None:
    return _PARTITION.get()


@dataclass
class CheckpointShardStats:
    tensors: int = 0
    original_bytes: int = 0
    stored_bytes: int = 0
    restored_bytes: int = 0


@dataclass(frozen=True)
class _Shard:
    tensor: torch.Tensor
    ordinal: int
    shape: tuple[int, ...]
    numel: int


@contextmanager
def shard_saved_activations(
    group: Any, *, shape: tuple[int, int, int], key: str
) -> Iterator[CheckpointShardStats]:
    """Forward-only scope for uniform one-layer recomputation checkpoints.

    ``shape`` is the complete [root_tokens, 1, hidden] activation shape. All
    ranks must have identical values and checkpoint save/unpack order. A
    collective ordinal fence rejects mismatched backward order. Uneven shards
    are supported without changing root padding or any arithmetic geometry.
    ``key`` identifies run/epoch/request/datum and numerical-policy version.

    Exit this scope before backward: unpack hooks remain bound to saved
    tensors, while recomputation's temporary saves should remain local. Keep
    the numerical-policy scope open across both forward and backward.
    """
    if len(shape) != 3 or shape[1] != 1 or any(n <= 0 for n in shape):
        raise ValueError("checkpoint activation shape must be [tokens, 1, hidden]")
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    identity = int.from_bytes(sha256(key.encode()).digest()[:8], "little") % (2**63)
    stats = CheckpointShardStats()
    if world == 1:
        yield stats
        return
    if world not in (2, 4, 8):
        raise ValueError("checkpoint sharding requires one-node groups of 2/4/8")

    def pack(tensor: torch.Tensor) -> torch.Tensor | _Shard:
        if tuple(tensor.shape) != shape or not tensor.is_floating_point():
            return tensor
        numel = tensor.numel()
        width = (numel + world - 1) // world
        start, end = rank * width, min((rank + 1) * width, numel)
        local = tensor.new_zeros(width)
        local[: max(0, end - start)].copy_(tensor.detach().reshape(-1)[start:end])
        ordinal = stats.tensors
        stats.tensors += 1
        stats.original_bytes += numel * tensor.element_size()
        stats.stored_bytes += local.numel() * local.element_size()
        return _Shard(local, ordinal, tuple(tensor.shape), numel)

    def unpack(saved: torch.Tensor | _Shard) -> torch.Tensor:
        if isinstance(saved, torch.Tensor):
            return saved
        header = torch.tensor(
            (identity, saved.ordinal, saved.numel),
            device=saved.tensor.device,
            dtype=torch.int64,
        )
        headers = header.new_empty(world * header.numel())
        dist.all_gather_into_tensor(headers, header, group=group)
        if not bool((headers.reshape(world, -1) == header).all()):
            raise RuntimeError(
                "checkpoint shards have mismatched identity or backward order"
            )
        full = saved.tensor.new_empty(world * saved.tensor.numel())
        dist.all_gather_into_tensor(full, saved.tensor, group=group)
        stats.restored_bytes += saved.numel * saved.tensor.element_size()
        return full[: saved.numel].view(saved.shape)

    if _PARTITION.get() is not None:
        raise RuntimeError("checkpoint sharding scopes cannot overlap")
    token = _PARTITION.set((world, shape))
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            yield stats
    finally:
        _PARTITION.reset(token)
