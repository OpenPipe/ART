"""Convert fused MoE LoRA adapters to per-expert format for vLLM compatibility.

Unsloth with transformers v5 saves MoE expert LoRA as fused 2D tensors:
  mlp.experts.base_layer.lora_A  [num_experts*rank, intermediate*2]  (gate_up_proj)
  mlp.experts.base_layer.lora_B  [hidden, num_experts*rank]          (gate_up_proj)
  mlp.experts.lora_A             [num_experts*rank, hidden]          (down_proj)
  mlp.experts.lora_B             [intermediate, num_experts*rank]    (down_proj)

vLLM expects per-expert keys:
  mlp.experts.0.gate_proj.lora_A [rank, hidden]
  mlp.experts.0.gate_proj.lora_B [intermediate, rank]
  ...
"""

import json
import os
import re

import safetensors.torch
import torch


def _has_fused_moe_lora(tensors: dict[str, torch.Tensor]) -> bool:
    """Check if the adapter contains fused MoE LoRA tensors."""
    return any(
        re.search(r"mlp\.experts\.(base_layer\.)?lora_[AB]\.weight$", key)
        for key in tensors
    )


def _infer_moe_params(
    tensors: dict[str, torch.Tensor],
    adapter_config: dict,
) -> tuple[int, int, int, int]:
    """Infer num_experts, rank, intermediate_size, hidden_size from tensor shapes."""
    rank = adapter_config.get("r", adapter_config.get("lora_rank", 8))

    for key, tensor in tensors.items():
        # gate_up_proj lora_A: [num_experts*rank, intermediate*2]
        if re.search(r"mlp\.experts\.base_layer\.lora_A\.weight$", key):
            num_experts_times_rank = tensor.shape[0]
            intermediate_times_2 = tensor.shape[1]
            num_experts = num_experts_times_rank // rank
            intermediate_size = intermediate_times_2 // 2
            break
        # down_proj lora_B: [intermediate, num_experts*rank]
        if re.search(r"mlp\.experts\.lora_B\.weight$", key):
            intermediate_size = tensor.shape[0]
            num_experts = tensor.shape[1] // rank
            break
    else:
        raise ValueError("Could not find fused MoE tensors to infer parameters")

    # Get hidden_size from gate_up_proj lora_B: [hidden, num_experts*rank]
    # or from down_proj lora_A: [num_experts*rank, hidden]
    for key, tensor in tensors.items():
        if re.search(r"mlp\.experts\.base_layer\.lora_B\.weight$", key):
            hidden_size = tensor.shape[0]
            break
        if re.search(r"mlp\.experts\.lora_A\.weight$", key):
            hidden_size = tensor.shape[1]
            break
    else:
        raise ValueError("Could not infer hidden_size from fused MoE tensors")

    return num_experts, rank, intermediate_size, hidden_size


def convert_fused_moe_lora(
    tensors: dict[str, torch.Tensor],
    num_experts: int,
    rank: int,
    intermediate_size: int,
    hidden_size: int,
) -> dict[str, torch.Tensor]:
    """Convert fused MoE LoRA tensors to per-expert format.

    Non-expert tensors (e.g. self_attn) are passed through unchanged.
    """
    new_tensors: dict[str, torch.Tensor] = {}

    for key, tensor in tensors.items():
        # Non-expert tensors: keep as-is
        m = re.match(
            r"(.*\.mlp\.experts)\.(base_layer\.lora_(A|B)|lora_(A|B))\.weight$",
            key,
        )
        if not m:
            new_tensors[key] = tensor
            continue

        prefix = m.group(1)
        is_base_layer = "base_layer" in key
        is_A = "lora_A" in key

        if is_base_layer:
            # gate_up_proj (fused gate + up)
            if is_A:
                # [num_experts*rank, intermediate*2] → per expert
                per_expert = tensor.reshape(num_experts, rank, intermediate_size * 2)
                for e in range(num_experts):
                    expert_a = per_expert[e]  # [rank, intermediate*2]
                    gate_a = expert_a[:, :intermediate_size]
                    up_a = expert_a[:, intermediate_size:]
                    new_tensors[f"{prefix}.{e}.gate_proj.lora_B.weight"] = (
                        gate_a.T.contiguous()
                    )
                    new_tensors[f"{prefix}.{e}.up_proj.lora_B.weight"] = (
                        up_a.T.contiguous()
                    )
            else:
                # [hidden, num_experts*rank] → per expert
                per_expert = tensor.reshape(hidden_size, num_experts, rank)
                for e in range(num_experts):
                    expert_b = per_expert[:, e, :]  # [hidden, rank]
                    new_tensors[f"{prefix}.{e}.gate_proj.lora_A.weight"] = (
                        expert_b.T.contiguous()
                    )
                    new_tensors[f"{prefix}.{e}.up_proj.lora_A.weight"] = (
                        expert_b.T.contiguous()
                    )
        else:
            # down_proj
            if is_A:
                # [num_experts*rank, hidden] → per expert
                per_expert = tensor.reshape(num_experts, rank, hidden_size)
                for e in range(num_experts):
                    expert_a = per_expert[e]  # [rank, hidden]
                    new_tensors[f"{prefix}.{e}.down_proj.lora_B.weight"] = (
                        expert_a.T.contiguous()
                    )
            else:
                # [intermediate, num_experts*rank] → per expert
                per_expert = tensor.reshape(intermediate_size, num_experts, rank)
                for e in range(num_experts):
                    expert_b = per_expert[:, e, :]  # [intermediate, rank]
                    new_tensors[f"{prefix}.{e}.down_proj.lora_A.weight"] = (
                        expert_b.T.contiguous()
                    )

    return new_tensors


def convert_checkpoint_if_needed(checkpoint_dir: str) -> None:
    """Convert a checkpoint's MoE LoRA adapter to per-expert format if needed.

    This is a no-op for non-MoE adapters.
    """
    adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    config_path = os.path.join(checkpoint_dir, "adapter_config.json")

    if not os.path.exists(adapter_path) or not os.path.exists(config_path):
        return

    tensors = safetensors.torch.load_file(adapter_path)
    if not _has_fused_moe_lora(tensors):
        return

    with open(config_path) as f:
        adapter_config = json.load(f)

    num_experts, rank, intermediate_size, hidden_size = _infer_moe_params(
        tensors, adapter_config
    )

    new_tensors = convert_fused_moe_lora(
        tensors, num_experts, rank, intermediate_size, hidden_size
    )

    # Overwrite the adapter with the converted tensors
    safetensors.torch.save_file(new_tensors, adapter_path)

    # Update adapter_config.json target_modules
    adapter_config["target_modules"] = [
        m for m in adapter_config.get("target_modules", []) if "experts" not in m
    ] + ["gate_proj", "up_proj", "down_proj"]
    # Remove target_parameters if present (not needed for per-expert format)
    adapter_config.pop("target_parameters", None)

    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)
