import re
from typing import Any, Sequence

import torch

from art.megatron.model_support.handlers.default_dense import DefaultMoeHandler
from art.megatron.model_support.handlers.qwen3_common import (
    install_qwen3_text_preprocess_patch,
)
from art.megatron.model_support.spec import CompileWorkaroundConfig

_QWEN3_MOE_COMPILE_WORKAROUND_FLAGS = (
    "alltoall_dtoh",
    "alltoall_dispatch_preprocess",
    "deepep_permute_restore",
)
_QWEN3_FUSED_MOE_KEY_RE = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts)\."
    r"(?:(?P<base_layer>base_layer)\.)?(?P<lora>lora_[AB])\.weight$"
)
_QWEN3_EXPERT_MOE_KEY_RE = re.compile(
    r"^.*\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)\.lora_[AB]\.weight$"
)


class Qwen3MoeHandler(DefaultMoeHandler):
    key = "qwen3_moe"
    native_vllm_lora_status = "validated"

    def to_vllm_lora_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        adapter_config: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return _to_vllm_lora_tensors(tensors, adapter_config=adapter_config)

    def install_preprocess_patch(self, model_chunks: Sequence[Any]) -> None:
        install_qwen3_text_preprocess_patch(model_chunks)

    def compile_workaround_config(
        self,
        provider: Any,
    ) -> CompileWorkaroundConfig:
        del provider
        return CompileWorkaroundConfig(flags=_QWEN3_MOE_COMPILE_WORKAROUND_FLAGS)


QWEN3_MOE_HANDLER = Qwen3MoeHandler()


def _qwen3_moe_config(adapter_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(adapter_config)
    target_modules = list(config.get("target_modules") or [])
    if "experts" not in target_modules:
        target_modules.append("experts")
    config["target_modules"] = target_modules
    return config


def _packed_lora_b_by_expert(
    tensor: torch.Tensor,
    *,
    num_experts: int,
    rank: int,
) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], rank, num_experts).permute(2, 0, 1)


def _clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clone().contiguous()


def _expand_fused_moe_lora(
    prefix: str,
    slots: dict[str, torch.Tensor],
    *,
    rank: int,
) -> dict[str, torch.Tensor]:
    try:
        gate_up_a = slots["base_layer.lora_A"]
        gate_up_b = slots["base_layer.lora_B"]
        down_a = slots["lora_A"]
        down_b = slots["lora_B"]
    except KeyError as exc:
        raise RuntimeError(f"Incomplete Qwen3 MoE LoRA block for {prefix}") from exc

    if (
        gate_up_a.ndim != 2
        or gate_up_b.ndim != 2
        or down_a.ndim != 2
        or down_b.ndim != 2
    ):
        raise RuntimeError(f"Qwen3 MoE LoRA tensors for {prefix} must be 2D")
    if gate_up_a.shape[0] % rank != 0:
        raise RuntimeError(
            f"{prefix}: gate/up lora_A shape {tuple(gate_up_a.shape)} "
            f"is not divisible by rank {rank}"
        )
    if gate_up_b.shape[0] % 2 != 0:
        raise RuntimeError(
            f"{prefix}: gate/up lora_B rows {gate_up_b.shape[0]} are not even"
        )
    num_experts = gate_up_a.shape[0] // rank
    expected_rank_cols = num_experts * rank
    intermediate = gate_up_b.shape[0] // 2
    if gate_up_b.shape[1] != expected_rank_cols:
        raise RuntimeError(
            f"{prefix}: gate/up lora_B shape {tuple(gate_up_b.shape)} does not "
            f"match {num_experts} experts at rank {rank}"
        )
    if down_a.shape != (expected_rank_cols, intermediate):
        raise RuntimeError(
            f"{prefix}: down lora_A shape {tuple(down_a.shape)} does not match "
            f"expected {(expected_rank_cols, intermediate)}"
        )
    if down_b.shape[1] != expected_rank_cols:
        raise RuntimeError(
            f"{prefix}: down lora_B shape {tuple(down_b.shape)} does not match "
            f"{num_experts} experts at rank {rank}"
        )

    gate_up_b_by_expert = _packed_lora_b_by_expert(
        gate_up_b,
        num_experts=num_experts,
        rank=rank,
    )
    down_b_by_expert = _packed_lora_b_by_expert(
        down_b,
        num_experts=num_experts,
        rank=rank,
    )
    expanded: dict[str, torch.Tensor] = {}
    for expert in range(num_experts):
        rows = slice(expert * rank, (expert + 1) * rank)
        gate_b, up_b = gate_up_b_by_expert[expert].split(intermediate, dim=0)
        expert_prefix = f"{prefix}.{expert}"
        expanded[f"{expert_prefix}.gate_proj.lora_A.weight"] = _clone(gate_up_a[rows])
        expanded[f"{expert_prefix}.gate_proj.lora_B.weight"] = _clone(gate_b)
        expanded[f"{expert_prefix}.up_proj.lora_A.weight"] = _clone(gate_up_a[rows])
        expanded[f"{expert_prefix}.up_proj.lora_B.weight"] = _clone(up_b)
        expanded[f"{expert_prefix}.down_proj.lora_A.weight"] = _clone(down_a[rows])
        expanded[f"{expert_prefix}.down_proj.lora_B.weight"] = _clone(
            down_b_by_expert[expert]
        )
    return expanded


def _to_vllm_lora_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    adapter_config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in tensors.items():
        match = _QWEN3_FUSED_MOE_KEY_RE.match(key)
        if match is None:
            continue
        slot = (
            f"{'base_layer.' if match.group('base_layer') else ''}{match.group('lora')}"
        )
        grouped.setdefault(match.group("prefix"), {})[slot] = tensor

    if not grouped:
        if any(_QWEN3_EXPERT_MOE_KEY_RE.match(key) for key in tensors):
            return tensors, _qwen3_moe_config(adapter_config)
        return tensors, adapter_config

    rank = int(adapter_config["r"])
    transformed: dict[str, torch.Tensor] = {}
    used_keys: set[str] = set()
    for prefix, slots in grouped.items():
        transformed.update(_expand_fused_moe_lora(prefix, slots, rank=rank))
        used_keys.update(
            {
                f"{prefix}.base_layer.lora_A.weight",
                f"{prefix}.base_layer.lora_B.weight",
                f"{prefix}.lora_A.weight",
                f"{prefix}.lora_B.weight",
            }
        )
    for key, tensor in tensors.items():
        if key in used_keys:
            continue
        if key in transformed:
            raise RuntimeError(f"Duplicate Qwen3 LoRA tensor after conversion: {key}")
        transformed[key] = tensor
    return transformed, _qwen3_moe_config(adapter_config)
