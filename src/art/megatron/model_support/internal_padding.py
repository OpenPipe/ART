from __future__ import annotations

from collections.abc import Sequence
from re import Pattern
from typing import Any

import torch


def round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def pad_dim_right(tensor: torch.Tensor, *, dim: int, size: int) -> torch.Tensor:
    dim = dim if dim >= 0 else tensor.ndim + dim
    current = int(tensor.shape[dim])
    if current == size:
        return tensor.contiguous()
    if current > size:
        raise RuntimeError(f"Cannot pad tensor dim {dim} from {current} down to {size}")
    shape = list(tensor.shape)
    shape[dim] = size - current
    return torch.cat((tensor, tensor.new_zeros(shape)), dim=dim).contiguous()


def trim_dim_right(tensor: torch.Tensor, *, dim: int, size: int) -> torch.Tensor:
    dim = dim if dim >= 0 else tensor.ndim + dim
    current = int(tensor.shape[dim])
    if current == size:
        return tensor.contiguous()
    if current < size:
        raise RuntimeError(f"Cannot trim tensor dim {dim} from {current} up to {size}")
    return tensor.narrow(dim, 0, size).contiguous()


def pack_vllm_3d_lora_b(blocks: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(blocks, dim=0)
    return stacked.permute(1, 2, 0).reshape(stacked.shape[1], -1).contiguous()


def unpack_vllm_3d_lora_b(
    tensor: torch.Tensor, *, num_experts: int, rank: int
) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], rank, num_experts).permute(2, 0, 1)


def group_expert_lora_tensors(
    tensors: dict[str, torch.Tensor], pattern: Pattern[str]
) -> dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]]:
    grouped: dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]] = {}
    for key, tensor in tensors.items():
        match = pattern.match(key)
        if match is not None:
            grouped.setdefault(match.group("prefix"), {}).setdefault(
                int(match.group("expert")), {}
            ).setdefault(match.group("module"), {})[match.group("lora")] = tensor
    return grouped


def _local_padding_ranges(
    param: torch.nn.Parameter,
    tensor: torch.Tensor,
    logical: int,
    internal: int,
    components: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    if logical == internal:
        return ()
    if not bool(getattr(param, "lora_tp_sharded", False)):
        if any(size != internal for size in components):
            raise RuntimeError(
                f"Padded component sizes {components} must all equal {internal}"
            )
        return tuple(
            (offset + logical, offset + internal)
            for offset in range(0, sum(components), internal)
        )

    from art.megatron import lora

    domain = getattr(param, "lora_shard_domain")
    world_size = lora._get_shard_world_size(domain)  # type: ignore[attr-defined]
    shard_rank = lora._get_shard_rank(domain)  # type: ignore[attr-defined]
    strategy = getattr(param, "lora_tp_shard_strategy", "uniform")
    if strategy == "uniform":
        if components != (internal,):
            raise RuntimeError("Uniform padding masks require one component")
        if internal % world_size:
            raise RuntimeError(
                f"Internal size {internal} is not divisible by world size {world_size}"
            )
        shard_size = internal // world_size
        shard_start = shard_rank * shard_size
        start = max(logical, shard_start) - shard_start
        end = min(internal, shard_start + shard_size) - shard_start
        return ((start, end),) if end > start else ()
    if strategy != "componentwise":
        raise RuntimeError(f"Unsupported padding shard strategy={strategy!r}")

    component_sizes = tuple(
        int(size) for size in getattr(param, "lora_tp_component_sizes", ())
    )
    if component_sizes != components:
        raise RuntimeError(
            f"Unexpected component sizes {component_sizes}; expected {components}"
        )
    ranges: list[tuple[int, int]] = []
    local_offset = 0
    for component_size in component_sizes:
        if component_size % world_size:
            raise RuntimeError(
                f"Component size {component_size} is not divisible by world size {world_size}"
            )
        shard_size = component_size // world_size
        shard_start = shard_rank * shard_size
        start = max(logical, shard_start) - shard_start
        end = min(internal, shard_start + shard_size) - shard_start
        if end > start:
            ranges.append((local_offset + start, local_offset + end))
        local_offset += shard_size
    if local_offset != tensor.shape[-1]:
        raise RuntimeError(
            f"Componentwise padding expected local extent {local_offset}, got {tensor.shape[-1]}"
        )
    return tuple(ranges)


def zero_lora_padding(
    param: torch.nn.Parameter,
    *,
    dim: int,
    logical: int,
    internal: int,
    components: tuple[int, ...],
    grads: bool,
    params: bool,
) -> None:
    tensors: list[torch.Tensor] = [param.data] if params else []
    if grads:
        for value in (param.grad, getattr(param, "main_grad", None)):
            tensor = _tensor_or_local_tensor(value)
            if tensor is not None:
                tensors.append(tensor)
    if not tensors:
        return
    ranges = _local_padding_ranges(param, tensors[0], logical, internal, components)
    dim = dim if dim >= 0 else tensors[0].ndim + dim
    for tensor in tensors:
        for start, end in ranges:
            tensor.narrow(dim, start, end - start).zero_()


def zero_ranges(
    tensor: torch.Tensor,
    *,
    dim: int,
    ranges: Sequence[tuple[int, int]],
) -> None:
    dim = dim if dim >= 0 else tensor.ndim + dim
    for start, end in ranges:
        if end > start:
            tensor.narrow(dim, start, end - start).zero_()


def _tensor_or_local_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    local = getattr(value, "_local_tensor", None)
    return local if torch.is_tensor(local) else None
