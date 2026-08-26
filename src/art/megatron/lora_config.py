from typing import Any

MOE_LORA_RANK = 1
DENSE_LORA_RANK = 8
LORA_ALPHA = 32
MEGATRON_LORA_RANK_ENV = "ART_MEGATRON_LORA_RANK"
MEGATRON_LORA_TARGET_MODULES_ENV = "ART_MEGATRON_LORA_TARGET_MODULES"


def default_lora_rank_for_handler(handler: Any) -> int:
    return MOE_LORA_RANK if bool(getattr(handler, "is_moe", False)) else DENSE_LORA_RANK
