"""Lossless Stark Qwen adapters for stock Transformers and PEFT.

Tinker's independent projection ranks remain intact in the compatibility
artifact. This separate codec fuses them only in the HF adapter, increasing
rank and alpha together so every effective weight update stays the same.
"""

from __future__ import annotations

import re
from typing import Any

import torch

_EXPERT = re.compile(
    r"^(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj|gate_up_proj)"
    r"\.lora_([AB])\.weight$"
)


def _fuse(pairs: list[tuple[torch.Tensor, torch.Tensor]]):
    return torch.cat([a for a, _ in pairs]), torch.block_diag(*[b for _, b in pairs])


def convert(
    tensors: dict[str, torch.Tensor], *, adapter_config: dict[str, Any]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from art.megatron.model_support.handlers.qwen3_5 import (
        _to_vllm_key,
        _to_vllm_lora_tensor,
    )

    # The full HF conditional model uses the same language_model namespace and
    # gated-query row order as vLLM. Only packed expert factors differ.
    result, experts = {}, {}
    for name, tensor in tensors.items():
        if match := _EXPERT.match(name):
            prefix, expert, projection, suffix = match.groups()
            experts.setdefault(prefix, {}).setdefault(int(expert), {}).setdefault(
                projection, {}
            )[suffix] = tensor
        else:
            public, value = _to_vllm_lora_tensor(
                name, tensor, adapter_config=adapter_config
            )
            result[public] = value

    rank = int(adapter_config["r"])
    alpha = int(adapter_config["lora_alpha"])
    rank_pattern, alpha_pattern = {}, {}
    qkv_prefixes = {
        name.removesuffix(".in_proj_q.lora_A.weight")
        for name in result
        if name.endswith(".linear_attn.in_proj_q.lora_A.weight")
    }
    for prefix in sorted(qkv_prefixes):
        pairs = [
            tuple(
                result.pop(prefix + f".in_proj_{part}.lora_{suffix}.weight")
                for suffix in ("A", "B")
            )
            for part in ("q", "k", "v")
        ]
        a, b = _fuse(pairs)
        for suffix, value in (("A", a), ("B", b)):
            result[prefix + f".in_proj_qkv.lora_{suffix}.weight"] = value
        rank_pattern["in_proj_qkv"] = 3 * rank
        alpha_pattern["in_proj_qkv"] = 3 * alpha

    for prefix, group in sorted(experts.items()):
        if sorted(group) != list(range(len(group))):
            raise ValueError("HF PEFT export requires every expert")
        gate_up, down = [], []
        for _, projections in sorted(group.items()):
            if "gate_up_proj" in projections:
                pair = projections["gate_up_proj"]
                gate_up.append((pair["A"], pair["B"]))
            else:
                gate_up.append(
                    _fuse(
                        [
                            (projections[name]["A"], projections[name]["B"])
                            for name in ("gate_proj", "up_proj")
                        ]
                    )
                )
            pair = projections["down_proj"]
            down.append((pair["A"], pair["B"]))
        gate_rank = gate_up[0][0].shape[0]
        rank_pattern["gate_up_proj"] = gate_rank
        alpha_pattern["gate_up_proj"] = alpha * gate_rank // rank
        # PEFT nests ParamWrappers in target_parameters order. Gate/up is the
        # inner wrapper; down is outer. Its B layout is [out, rank, experts].
        public = _to_vllm_key(prefix)
        for nested, pairs in ((".base_layer", gate_up), ("", down)):
            a = torch.stack([a for a, _ in pairs]).flatten(0, 1)
            b = torch.stack([b for _, b in pairs]).permute(1, 2, 0).flatten(1)
            result[public + nested + ".lora_A.weight"] = a
            result[public + nested + ".lora_B.weight"] = b

    target_modules = sorted(
        {
            name.removeprefix("base_model.model.").rsplit(".lora_", 1)[0]
            for name in result
            if ".mlp.experts." not in name
        }
    )
    config = {
        "base_model_name_or_path": adapter_config["base_model_name_or_path"],
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "bias": "none",
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "target_modules": target_modules,
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
    }
    if experts:
        config["target_parameters"] = [
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
        ]
    return result, config
