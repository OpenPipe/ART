from __future__ import annotations

import tinker

from art.megatron.model_support import default_target_modules_for_model
from art.training import AdapterSpec, TrainingRunSpec

from .errors import UnsupportedCapabilityError

_ATTENTION_TARGETS = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_a_proj",
        "q_b_proj",
        "kv_proj",
        "kv_a_proj_with_mqa",
        "o_a_proj",
        "o_b_proj",
        "in_proj_qkv",
        "in_proj_z",
        "out_proj",
        "compressor.kv_proj",
        "compressor.gate_proj",
    }
)
_MLP_TARGETS = frozenset({"gate_proj", "up_proj", "down_proj", "experts"})


def resolve_tinker_target_modules(
    base_model: str,
    *,
    train_attn: bool,
    train_mlp: bool,
    train_unembed: bool,
) -> tuple[str, ...]:
    if train_unembed:
        raise UnsupportedCapabilityError(
            "current Megatron handlers expose no unembedding LoRA target"
        )
    if not train_mlp and not train_attn:
        raise ValueError("at least one LoRA target group must be enabled")

    defaults = tuple(default_target_modules_for_model(base_model))
    unknown = sorted(set(defaults) - _ATTENTION_TARGETS - _MLP_TARGETS)
    if unknown:
        raise UnsupportedCapabilityError(
            f"cannot map Tinker target switches for handler targets {unknown}"
        )
    enabled = (_ATTENTION_TARGETS if train_attn else frozenset()) | (
        _MLP_TARGETS if train_mlp else frozenset()
    )
    selected = tuple(target for target in defaults if target in enabled)
    if not selected:
        raise UnsupportedCapabilityError(
            "the selected Tinker target group has no targets in this model handler"
        )
    return selected


def translate_tinker_lora_config(
    base_model: str,
    lora_config: tinker.LoraConfig,
) -> TrainingRunSpec:
    """Map a pinned Tinker LoRA config to ART's canonical run contract."""

    return TrainingRunSpec(
        base_model=base_model,
        adapter=AdapterSpec(
            rank=lora_config.rank,
            target_modules=resolve_tinker_target_modules(
                base_model,
                train_attn=lora_config.train_attn,
                train_mlp=lora_config.train_mlp,
                train_unembed=lora_config.train_unembed,
            ),
        ),
        seed=lora_config.seed,
    )
