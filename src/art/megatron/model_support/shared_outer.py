from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any

import torch

from .spec import ExpertPackedLoraGroup


def canonicalize_identity_shared_outer(
    tensors: dict[str, torch.Tensor],
    *,
    adapter_config: dict[str, Any],
    groups: Iterable[ExpertPackedLoraGroup],
) -> dict[str, torch.Tensor]:
    """Initialize each shared factor once while preserving a zero LoRA delta."""
    if adapter_config.get("moe_parameterization") != "shared_outer":
        return tensors
    result = dict(tensors)
    for group in groups:
        for experts in _expanded_groups(result, group).values():
            for slot in group.slots:
                if not slot.shared_outer_factor:
                    continue
                entries = _slot_entries(
                    experts, slot.source_projection, slot.source_lora
                )
                if entries:
                    shared = entries[0][1]
                    for key, _ in entries[1:]:
                        result[key] = shared
        for slot in group.slots:
            if not slot.shared_outer_factor:
                continue
            suffix = f"{group.art_group_suffix}.{slot.output_suffix}"
            for key, tensor in tuple(result.items()):
                if key.endswith(suffix):
                    result[key] = _share_packed_factor(
                        tensor,
                        rank=int(adapter_config["r"]),
                        layout=slot.pack_layout,
                        key=key,
                    )
    return result


def expand_shared_outer(
    tensors: dict[str, torch.Tensor],
    *,
    adapter_config: dict[str, Any],
    groups: Iterable[ExpertPackedLoraGroup],
) -> dict[str, torch.Tensor]:
    result = dict(tensors)
    for group in groups:
        expanded = _expanded_groups(result, group)
        compact = _compact_groups(result, group)
        if compact and adapter_config.get("moe_parameterization") != "shared_outer":
            raise RuntimeError(
                "compact shared-outer LoRA requires matching adapter config"
            )
        for prefix, slots in compact.items():
            experts = expanded.get(prefix, {})
            if not experts:
                raise RuntimeError(
                    f"shared-outer LoRA cannot infer experts for {prefix}"
                )
            for slot_key, (key, tensor) in slots.items():
                for expert in experts:
                    output = _expert_key(prefix, expert, *slot_key)
                    if output in result:
                        raise RuntimeError(
                            f"mixed compact and expanded shared LoRA key: {output}"
                        )
                    result[output] = tensor
                result.pop(key)
    return result


def compact_shared_outer(
    tensors: dict[str, torch.Tensor],
    *,
    adapter_config: dict[str, Any],
    groups: Iterable[ExpertPackedLoraGroup],
) -> dict[str, torch.Tensor]:
    if adapter_config.get("moe_parameterization") != "shared_outer":
        return tensors
    result = dict(tensors)
    for group in groups:
        for prefix, experts in _expanded_groups(result, group).items():
            for slot in group.slots:
                if not slot.shared_outer_factor:
                    continue
                slot_key = slot.source_projection, slot.source_lora
                entries = _slot_entries(experts, *slot_key)
                if not entries:
                    continue
                if len(entries) != len(experts):
                    raise RuntimeError(
                        f"incomplete shared-outer LoRA factor for {prefix}"
                    )
                shared = entries[0][1]
                if any(not torch.equal(shared, tensor) for _, tensor in entries[1:]):
                    raise RuntimeError(
                        f"shared-outer LoRA factor differs across experts for "
                        f"{prefix}.{slot.source_projection}.{slot.source_lora}"
                    )
                compact_key = _compact_key(prefix, *slot_key)
                existing = result.get(compact_key)
                if existing is not None and not torch.equal(existing, shared):
                    raise RuntimeError(
                        f"mixed compact and expanded shared LoRA key: {compact_key}"
                    )
                for key, _ in entries:
                    result.pop(key)
                result[compact_key] = shared.contiguous()
    return result


def _expanded_groups(
    tensors: dict[str, torch.Tensor], group: ExpertPackedLoraGroup
) -> dict[str, dict[int, dict[tuple[str, str], tuple[str, torch.Tensor]]]]:
    declared = {
        (slot.source_projection, slot.source_lora) for slot in group.slots
    }
    projections = "|".join(re.escape(slot.source_projection) for slot in group.slots)
    loras = "|".join(
        re.escape(lora)
        for lora in dict.fromkeys(slot.source_lora for slot in group.slots)
    )
    pattern = re.compile(
        rf"^(?P<prefix>.*{re.escape(group.art_group_suffix)})\."
        rf"(?P<expert>\d+)\.(?P<projection>{projections})\."
        rf"(?P<lora>{loras})\.weight$"
    )
    result: dict[str, dict[int, dict[tuple[str, str], tuple[str, torch.Tensor]]]] = {}
    for key, tensor in tensors.items():
        match = pattern.match(key)
        if match is None:
            continue
        prefix = match.group("prefix")
        experts = result.setdefault(prefix, {})
        slot = match.group("projection"), match.group("lora")
        if slot not in declared:
            continue
        values = experts.setdefault(int(match.group("expert")), {})
        if slot in values:
            raise RuntimeError(f"duplicate expert LoRA factor: {key}")
        values[slot] = key, tensor
    return result


def _compact_groups(
    tensors: dict[str, torch.Tensor], group: ExpertPackedLoraGroup
) -> dict[str, dict[tuple[str, str], tuple[str, torch.Tensor]]]:
    shared = tuple(slot for slot in group.slots if slot.shared_outer_factor)
    if not shared:
        return {}
    declared = {(slot.source_projection, slot.source_lora) for slot in shared}
    projections = "|".join(re.escape(slot.source_projection) for slot in shared)
    loras = "|".join(
        re.escape(lora)
        for lora in dict.fromkeys(slot.source_lora for slot in shared)
    )
    pattern = re.compile(
        rf"^(?P<prefix>.*{re.escape(group.art_group_suffix)})\.shared\."
        rf"(?P<projection>{projections})\.(?P<lora>{loras})\.weight$"
    )
    result: dict[str, dict[tuple[str, str], tuple[str, torch.Tensor]]] = {}
    for key, tensor in tensors.items():
        match = pattern.match(key)
        if match is None:
            continue
        slot = match.group("projection"), match.group("lora")
        if slot not in declared:
            continue
        values = result.setdefault(match.group("prefix"), {})
        if slot in values:
            raise RuntimeError(f"duplicate shared LoRA factor: {key}")
        values[slot] = key, tensor
    return result


def _slot_entries(
    experts: dict[int, dict[tuple[str, str], tuple[str, torch.Tensor]]],
    projection: str,
    lora: str,
) -> list[tuple[str, torch.Tensor]]:
    slot = projection, lora
    return [
        experts[expert][slot] for expert in sorted(experts) if slot in experts[expert]
    ]


def _expert_key(prefix: str, expert: int, projection: str, lora: str) -> str:
    return f"{prefix}.{expert}.{projection}.{lora}.weight"


def _compact_key(prefix: str, projection: str, lora: str) -> str:
    return f"{prefix}.shared.{projection}.{lora}.weight"


def _share_packed_factor(
    tensor: torch.Tensor,
    *,
    rank: int,
    layout: str,
    key: str,
) -> torch.Tensor:
    axis = 0 if layout == "expert_rows" else tensor.ndim - 1
    if tensor.shape[axis] % rank:
        raise RuntimeError(
            f"identity shared-outer factor {key} shape {tuple(tensor.shape)} "
            f"is not divisible by rank {rank}"
        )
    experts = tensor.shape[axis] // rank
    if experts <= 1:
        return tensor
    if layout == "expert_rows":
        return tensor.narrow(0, 0, rank).repeat(
            experts, *(1 for _ in tensor.shape[1:])
        )
    shape = (*tensor.shape[:-1], rank, experts)
    return (
        tensor.reshape(shape)
        .narrow(-1, 0, 1)
        .expand(*shape)
        .reshape(tensor.shape)
        .contiguous()
    )
