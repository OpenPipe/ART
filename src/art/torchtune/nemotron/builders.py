# nemotron_tt/builders.py
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional
import re
import torch
from torch import nn

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

try:
    import peft
    from peft import LoraConfig, get_peft_model
except Exception as e:
    raise ImportError("Please `pip install peft` for LoRA/QLoRA support") from e


# ----------------------------
# Configuration payload
# ----------------------------
@dataclass(frozen=True)
class NemotronBuildArgs:
    model_id: str = "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"
    # choose bf16 whenever possible on Hopper/Ampere
    dtype: str = "bf16"               # "bf16" | "fp16" | "fp32"
    load_in_4bit: bool = True         # QLoRA path
    double_quant: bool = True
    quant_type: str = "nf4"           # "nf4" | "fp4"
    trust_remote_code: bool = True
    attn_implementation: str = "sdpa" # "sdpa" | "flash_attention_2"
    gradient_checkpointing: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None  # auto-discover if None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


# ----------------------------
# Helper: dtype parsing
# ----------------------------
@lru_cache(maxsize=None)
def _to_torch_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "bf16": return torch.bfloat16
    if name == "fp16": return torch.float16
    if name == "fp32": return torch.float32
    raise ValueError(f"Unknown dtype '{name}'")


# ----------------------------
# Helper: robust target module discovery
# - Nemotron NAS introduces non-uniform blocks, some layers skip attention.
# - We choose common Llama-linear names and ONLY keep those that exist.
# ----------------------------
@lru_cache(maxsize=1)
def _candidate_suffixes() -> List[str]:
    # attention projections + MLPs; keep names generic & suffix-based
    return ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj",
            "output_proj"]  # some remote impls use output_proj

def discover_lora_targets(model: nn.Module) -> List[str]:
    present = set()
    for name, module in model.named_modules():
        # only consider linear-like modules
        if isinstance(module, nn.Linear):
            for suf in _candidate_suffixes():
                if name.endswith(suf):
                    present.add(suf)
    # stable order for reproducibility
    return sorted(present)


# ----------------------------
# Core builder used by Torchtune
# Returns a PEFT-wrapped model ready for training (LoRA/QLoRA).
# ----------------------------
def build_nemotron49b_v1_5_for_lora(
    model_id: str = NemotronBuildArgs.model_id,
    dtype: str = NemotronBuildArgs.dtype,
    load_in_4bit: bool = NemotronBuildArgs.load_in_4bit,
    double_quant: bool = NemotronBuildArgs.double_quant,
    quant_type: str = NemotronBuildArgs.quant_type,
    trust_remote_code: bool = NemotronBuildArgs.trust_remote_code,
    attn_implementation: str = NemotronBuildArgs.attn_implementation,
    gradient_checkpointing: bool = NemotronBuildArgs.gradient_checkpointing,
    lora_r: int = NemotronBuildArgs.lora_r,
    lora_alpha: int = NemotronBuildArgs.lora_alpha,
    lora_dropout: float = NemotronBuildArgs.lora_dropout,
    lora_target_modules: Optional[List[str]] = None,
    bias: str = NemotronBuildArgs.bias,
    task_type: str = NemotronBuildArgs.task_type,
):
    """
    Build a Nemotron v1.5 (49B) model for LoRA/QLoRA training.

    IMPORTANT:
    - We instantiate from HF with trust_remote_code so Transformer's remote
      Nemotron/DeciLM implementation is used (NAS/skip-attn aware).
    - We DO NOT rely on Torchtune's weight-conversion checkpointers.
    """
    torch_dtype = _to_torch_dtype(dtype)

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_quant_type=quant_type,
        )

    # Prefer from_pretrained (loads weights + respects remote code).
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        quantization_config=quant_config,
        device_map=None,  # Let torchtune/recipe place modules
    )

    # Training niceties
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Auto-discover LoRA targets if not passed
    if lora_target_modules is None:
        lora_target_modules = discover_lora_targets(model)
        if not lora_target_modules:
            raise RuntimeError(
                "Could not discover LoRA targets. Pass lora_target_modules manually."
            )

    lconf = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias=bias,
        task_type=task_type,
    )
    model = get_peft_model(model, lconf)

    # Make sure only adapters train
    for p in model.base_model.model.parameters():
        p.requires_grad_(False)
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)

    # Torchtune expects an nn.Module. Tokenizer is configured separately in YAML.
    return model
