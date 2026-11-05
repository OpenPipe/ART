"""Validated, CPU-capable GLM shared-outer vLLM/PEFT codecs and EP merge.

Inputs are ART's logical per-expert 2D disk tensors, after shard transport.
The vLLM codec preserves singleton hidden factors; PEFT expands them for merge.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import math
import re
from typing import Any, Literal

import torch
import torch.nn.functional as F

from .lora import (
    ART_PREFIX,
    HEAD_PREFIX,
    LoraDimensions,
    LoraFlags,
    LoraTarget,
    lora_targets,
    validate_rank,
)

META = "art_glm5next_lora"
VLLM_PREFIX = "base_model.model.language_model.model.layers."
PEFT_PREFIX = "base_model.model.model.language_model.layers."
_SUFFIXES = (".lora_A.weight", ".lora_B.weight")
_W = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}


def adapter_config(
    rank: int,
    *,
    alpha: float = 32,
    dimensions: LoraDimensions = LoraDimensions(),
    flags: LoraFlags = LoraFlags(),
) -> dict[str, Any]:
    validate_rank(rank)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("GLM LoRA alpha must be finite and positive")
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "bias": "none",
        "base_model_name_or_path": "zai-org/GLM-5.3-Flash-BF16",
        "revision": "f9fed3bbc82291073373d67452e9f25317ef7577",
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "use_rslora": False,
        "inference_mode": True,
        "target_modules": sorted(
            {t.name.rsplit(".", 1)[-1] for t in lora_targets(dimensions, flags)}
        ),
        "art_lora_layout": {"expert": "shared_outer"},
        META: {
            "version": "shared_outer_v1",
            "format": "art",
            "rank": rank,
            "alpha": alpha,
            "dimensions": asdict(dimensions),
            "flags": asdict(flags),
        },
    }


def _contract(
    config: Mapping[str, Any], fmt: str
) -> tuple[int, LoraDimensions, LoraFlags]:
    meta = config.get(META, {})
    if meta.get("version") != "shared_outer_v1" or meta.get("format") != fmt:
        raise ValueError(f"Expected GLM shared_outer_v1 {fmt} metadata")
    rank = meta["rank"]
    validate_rank(rank)
    alpha = meta["alpha"]
    factor = 2 if fmt == "peft" else 1
    if config.get("r") != factor * rank or config.get("lora_alpha") != factor * alpha:
        raise ValueError(
            "GLM rank/alpha mismatch; expert top-k normalization is unsupported"
        )
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("GLM alpha must be finite and positive")
    if any(
        config.get(k)
        for k in (
            "rank_pattern",
            "alpha_pattern",
            "use_rslora",
            "use_dora",
            "fan_in_fan_out",
            "lora_bias",
            "lora_dropout",
        )
    ):
        raise ValueError(
            "GLM requires uniform alpha/r without rank/alpha patterns or LoRA variants"
        )
    values = dict(meta["dimensions"])
    values["mla_layers"] = tuple(values["mla_layers"])
    return rank, LoraDimensions(**values), LoraFlags(**meta["flags"])


def _name(name: str, fmt: str) -> str:
    if fmt == "vllm":
        return (
            name.replace(ART_PREFIX, VLLM_PREFIX, 1)
            if name.startswith(ART_PREFIX)
            else "base_model.model.language_model.lm_head"
        )
    return (
        name.replace(ART_PREFIX, PEFT_PREFIX, 1)
        if name.startswith(ART_PREFIX)
        else HEAD_PREFIX
    )


def _same(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if not torch.equal(actual, expected):
        raise ValueError(
            f"Incompatible independent factors or nonzero padding: {label}"
        )


def _checked(tensor: torch.Tensor, shape: tuple[int, ...], key: str) -> torch.Tensor:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"Shape mismatch {key}: {tuple(tensor.shape)} != {shape}")
    if not tensor.is_floating_point():
        raise ValueError(f"Non-floating GLM LoRA tensor {key}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Non-finite GLM LoRA tensor {key}")
    return tensor


def _native_pairs(
    tensors: Mapping[str, torch.Tensor], targets: tuple[LoraTarget, ...], rank: int
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    used: set[str] = set()
    result = {}
    for t in targets:
        pairs = []
        for e in range(t.experts or 1):
            name = t.name.format(expert=e)
            keys = tuple(name + s for s in _SUFFIXES)
            if any(k not in tensors for k in keys):
                raise ValueError(f"Missing GLM LoRA factors: {name}")
            a = _checked(tensors[keys[0]], (rank, t.in_features), keys[0])
            b = _checked(tensors[keys[1]], (t.out_features, rank), keys[1])
            used.update(keys)
            pairs.append((a, b))
        if t.experts:
            shared = 0 if t.shared == "A" else 1
            for pair in pairs[1:]:
                _same(pair[shared], pairs[0][shared], t.name)
            a = torch.stack([p[0] for p in pairs[: 1 if shared == 0 else t.experts]])
            b = torch.stack([p[1] for p in pairs[: 1 if shared == 1 else t.experts]])
            result[t.name] = (a, b)
        else:
            result[t.name] = pairs[0]
    if set(tensors) != used:
        raise ValueError(f"Unexpected GLM LoRA keys: {sorted(set(tensors) - used)[:8]}")
    return result


def validate_native_lora(
    tensors: Mapping[str, torch.Tensor], *, adapter_config: Mapping[str, Any]
) -> None:
    rank, dims, flags = _contract(adapter_config, "art")
    _native_pairs(tensors, lora_targets(dims, flags), rank)


def to_vllm_lora_tensors(
    tensors: dict[str, torch.Tensor], *, adapter_config: dict[str, Any]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    rank, dims, flags = _contract(adapter_config, "art")
    targets = lora_targets(dims, flags)
    pairs = _native_pairs(tensors, targets, rank)
    result: dict[str, torch.Tensor] = {}
    for t in targets:
        name = _name(t.name, "vllm")
        if t.experts:
            name = name.replace(
                "{expert}." + t.name.rsplit(".", 1)[-1], _W[t.name.rsplit(".", 1)[-1]]
            )
        for suffix, value in zip(_SUFFIXES, pairs[t.name], strict=True):
            # vLLM optimize() multiplies B in place. Ordinary factors may be
            # live snapshot views; never let a consumer mutate that source.
            result[name + suffix] = (
                value.contiguous()
                if t.experts
                else value.detach().clone(memory_format=torch.contiguous_format)
            )
    config = {
        **adapter_config,
        META: {**adapter_config[META], "format": "vllm"},
        "art_lora_format": "vllm",
    }
    config["target_modules"] = sorted(
        ({t.name.rsplit(".", 1)[-1] for t in targets if not t.experts})
        | ({"experts"} if any(t.experts for t in targets) else set())
    )
    # Serving must enable this; it is engine configuration, not a PEFT field.
    config["art_glm5next_vllm"] = {
        "enable_moe_shared_loras": flags.train_mlp,
        "language_model_only": False,
    }
    return result, config


def from_vllm_lora_tensors(
    tensors: dict[str, torch.Tensor], *, adapter_config: dict[str, Any]
) -> dict[str, torch.Tensor]:
    rank, dims, flags = _contract(adapter_config, "vllm")
    result: dict[str, torch.Tensor] = {}
    used: set[str] = set()
    for t in lora_targets(dims, flags):
        name = _name(t.name, "vllm")
        if t.experts:
            name = name.replace(
                "{expert}." + t.name.rsplit(".", 1)[-1], _W[t.name.rsplit(".", 1)[-1]]
            )
        keys = tuple(name + s for s in _SUFFIXES)
        if any(k not in tensors for k in keys):
            raise ValueError(f"Missing GLM vLLM factors: {name}")
        pair = tuple(
            _checked(tensors[k], shape, k)
            for k, shape in zip(keys, t.shapes(rank), strict=True)
        )
        used.update(keys)
        for e in range(t.experts or 1):
            for i, suffix in enumerate(_SUFFIXES):
                value = pair[i]
                if t.experts:
                    value = value[0 if t.shared == ("A" if i == 0 else "B") else e]
                result[t.name.format(expert=e) + suffix] = value.contiguous()
    if used != set(tensors):
        raise ValueError("Unexpected GLM vLLM keys")
    return result


def _expanded_pair(
    pair: tuple[torch.Tensor, torch.Tensor], t: LoraTarget
) -> tuple[torch.Tensor, torch.Tensor]:
    a, b = pair
    return a.expand(t.experts, *a.shape[1:]), b.expand(t.experts, *b.shape[1:])


def _peft_pack(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return a.flatten(0, 1).contiguous(), b.permute(1, 2, 0).flatten(1).contiguous()


def to_peft_lora_tensors(
    tensors: dict[str, torch.Tensor], *, adapter_config: dict[str, Any]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    rank, dims, flags = _contract(adapter_config, "art")
    targets = lora_targets(dims, flags)
    pairs = _native_pairs(tensors, targets, rank)
    result: dict[str, torch.Tensor] = {}
    ordinary: list[str] = []
    parameters: list[str] = []
    for t in targets:
        if not t.experts:
            name = _name(t.name, "peft")
            a, b = pairs[t.name]
            result[name + _SUFFIXES[0]] = F.pad(a, (0, 0, 0, rank))
            result[name + _SUFFIXES[1]] = F.pad(b, (0, rank))
            ordinary.append(name.removeprefix("base_model.model."))
            continue
        projection = t.name.rsplit(".", 1)[-1]
        if projection == "up_proj":
            continue
        prefix = t.name.split(".{expert}.")[0]
        name = _name(prefix, "peft")
        a, b = _expanded_pair(pairs[t.name], t)
        if projection == "gate_proj":
            ua, ub = _expanded_pair(pairs[prefix + ".{expert}.up_proj"], t)
            a = torch.cat((a, ua), dim=1)
            zero = torch.zeros_like(b)
            b = torch.cat(
                (torch.cat((b, zero), dim=2), torch.cat((zero, ub), dim=2)), dim=1
            )
            name += ".base_layer"
            parameters.append(
                _name(prefix, "peft").removeprefix("base_model.model.")
                + ".gate_up_proj"
            )
        else:
            a, b = F.pad(a, (0, 0, 0, rank)), F.pad(b, (0, rank))
            parameters.append(name.removeprefix("base_model.model.") + ".down_proj")
        for suffix, v in zip(_SUFFIXES, _peft_pack(a, b), strict=True):
            result[name + suffix] = v
    config = {
        **adapter_config,
        "r": 2 * rank,
        "lora_alpha": 2 * adapter_config["lora_alpha"],
        META: {**adapter_config[META], "format": "peft"},
        "target_modules": "^(?:" + "|".join(re.escape(n) for n in ordinary) + ")$"
        if ordinary
        else [],
        "target_parameters": parameters,
    }
    if not targets:
        # PEFT's explicit no-op adapter sentinel; [] alone raises at load time.
        config["target_modules"] = "dummy-target-modules"
    return result, config


def from_peft_lora_tensors(
    tensors: dict[str, torch.Tensor], *, adapter_config: dict[str, Any]
) -> dict[str, torch.Tensor]:
    rank, dims, flags = _contract(adapter_config, "peft")
    result: dict[str, torch.Tensor] = {}
    used: set[str] = set()
    for t in lora_targets(dims, flags):
        name = _name(t.name, "peft")
        if t.experts:
            projection = t.name.rsplit(".", 1)[-1]
            name = _name(t.name.split(".{expert}.")[0], "peft")
            is_fc1 = projection != "down_proj"
            name += ".base_layer" if is_fc1 else ""
            out = 2 * dims.expert_width if is_fc1 else dims.hidden
            inp = dims.hidden if is_fc1 else dims.expert_width
            shapes = ((t.experts * 2 * rank, inp), (out, 2 * rank * t.experts))
        else:
            shapes = ((2 * rank, t.in_features), (t.out_features, 2 * rank))
        keys = tuple(name + s for s in _SUFFIXES)
        if any(k not in tensors for k in keys):
            raise ValueError(f"Missing GLM PEFT factors: {name}")
        a, b = (_checked(tensors[k], s, k) for k, s in zip(keys, shapes, strict=True))
        used.update(keys)
        if t.experts:
            a = a.reshape(t.experts, 2 * rank, -1)
            b = b.reshape(b.shape[0], 2 * rank, t.experts).permute(2, 0, 1)
            if projection != "down_proj":
                _same(
                    b[:, : dims.expert_width, rank:],
                    torch.zeros_like(b[:, : dims.expert_width, rank:]),
                    name,
                )
                _same(
                    b[:, dims.expert_width :, :rank],
                    torch.zeros_like(b[:, dims.expert_width :, :rank]),
                    name,
                )
                offset = 0 if projection == "gate_proj" else rank
                row = 0 if projection == "gate_proj" else dims.expert_width
                a, b = (
                    a[:, offset : offset + rank],
                    b[:, row : row + dims.expert_width, offset : offset + rank],
                )
            else:
                _same(a[:, rank:], torch.zeros_like(a[:, rank:]), name)
                _same(b[:, :, rank:], torch.zeros_like(b[:, :, rank:]), name)
                a, b = a[:, :rank], b[:, :, :rank]
            shared = a if t.shared == "A" else b
            for v in shared[1:]:
                _same(v, shared[0], name)
        else:
            _same(a[rank:], torch.zeros_like(a[rank:]), name)
            _same(b[:, rank:], torch.zeros_like(b[:, rank:]), name)
            a, b = a[:rank], b[:, :rank]
        for e in range(t.experts or 1):
            for suffix, v in zip(_SUFFIXES, (a, b), strict=True):
                result[t.name.format(expert=e) + suffix] = (
                    v[e] if t.experts else v
                ).contiguous()
    if used != set(tensors):
        raise ValueError("Unexpected GLM PEFT keys")
    return result


def gather_and_export_lora(
    entries_by_key: dict[str, list[tuple[dict[str, Any], torch.Tensor]]],
    *,
    adapter_config: dict[str, Any],
    format: Literal["vllm", "peft"] = "vllm",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """M6: convert collected ART manifests after rank transport completes.

    Uses ART's TP/componentwise shard merge. EP experts already have global IDs
    in keys; codec validation enforces complete coverage and shared-factor ties.
    CP/DP replica selection belongs to ART's existing transport, before here.
    """
    _contract(adapter_config, "art")
    if format not in ("vllm", "peft"):
        raise ValueError(f"Unknown GLM LoRA export format {format}")
    from art.megatron.weights.lora_publish import merge_sharded_adapter_entries

    tensors = merge_sharded_adapter_entries(entries_by_key)
    codec = to_vllm_lora_tensors if format == "vllm" else to_peft_lora_tensors
    return codec(tensors, adapter_config=adapter_config)
