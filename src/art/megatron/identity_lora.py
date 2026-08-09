import os
from typing import Any
import warnings

from peft.tuners.lora.config import LoraConfig
import torch

from art.dev.get_model_config import default_target_modules

from .lora_config import LORA_ALPHA, default_lora_rank_for_handler
from .model_support.lora_disk import normalize_lora_checkpoint_to_vllm
from .model_support.spec import ModelSupportHandler


def create_identity_lora(
    base_model: str,
    lora_path: str,
    rank: int | None = None,
    target_modules: list[str] | None = None,
    lora_alpha: int = LORA_ALPHA,
    random_state: int | None = None,
    allow_unvalidated_arch: bool = False,
    handler: ModelSupportHandler | None = None,
) -> None:
    """Create an identity LoRA adapter for a Megatron model."""
    from unittest.mock import patch

    from accelerate import init_empty_weights
    from peft import get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    from .model_support import get_model_support_handler

    if random_state is not None:
        torch.manual_seed(random_state)
    target_modules = target_modules or default_target_modules(base_model)
    handler = handler or get_model_support_handler(
        base_model, allow_unvalidated_arch=allow_unvalidated_arch
    )
    if rank is None:
        rank = default_lora_rank_for_handler(handler)
    base_config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    model_config = handler.identity_lora_model_config(base_config)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            model_config, dtype=torch.bfloat16, trust_remote_code=True
        )
    model.name_or_path = base_model

    lora_config = LoraConfig(
        base_model_name_or_path=base_model,
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=[],
        target_parameters=handler.identity_lora_target_parameters(
            model,
            target_modules=target_modules,
        ),
        bias="none",
    )
    meta = torch.device("meta")
    orig_to = torch.nn.Module.to

    def _skip_meta_to(
        module: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        device = kwargs.get("device") or (args[0] if args else None)
        if device == meta or str(device) == "meta":
            dtype = kwargs.get("dtype")
            return module if dtype is None else orig_to(module, dtype=dtype)
        return orig_to(module, *args, **kwargs)

    with warnings.catch_warnings():
        if bool(getattr(handler, "is_moe", False)):
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"Unsupported layer type '.*MoeExperts.*' encountered, "
                    r"proceed at your own risk\."
                ),
                category=UserWarning,
                module=r"peft\.tuners\.tuners_utils",
            )
        with patch.object(torch.nn.Module, "to", _skip_meta_to):
            peft_model = get_peft_model(
                model,
                lora_config,
                autocast_adapter_dtype=False,
            )

    os.makedirs(lora_path, exist_ok=True)
    peft_model.save_pretrained(lora_path)
    final_config = LoraConfig(
        base_model_name_or_path=base_model,
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        bias="none",
    ).to_dict()
    normalize_lora_checkpoint_to_vllm(
        lora_path,
        handler=handler,
        adapter_config=final_config,
    )
    del peft_model, model
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
