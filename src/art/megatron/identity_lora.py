import math
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4
import warnings

from peft.tuners.lora.config import LoraConfig
import torch

from art.dev.get_model_config import default_target_modules

from .lora_config import LORA_ALPHA, default_lora_rank_for_handler
from .model_support.lora_disk import (
    normalize_lora_checkpoint_to_vllm,
    save_vllm_lora_tensors,
)
from .model_support.spec import ModelSupportHandler


def _direct_2d_linear_identity_tensors(
    model: torch.nn.Module,
    *,
    target_parameters: list[str],
    rank: int,
) -> dict[str, torch.Tensor]:
    """Build PEFT-compatible source tensors for an exact 2D Linear target set."""
    if rank <= 0:
        raise ValueError(f"identity LoRA rank must be positive, got {rank}")
    if not target_parameters:
        raise RuntimeError("Direct identity LoRA target set is empty")
    if len(target_parameters) != len(set(target_parameters)):
        raise RuntimeError("Direct identity LoRA target parameters contain duplicates")

    targets = set(target_parameters)
    resolved = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name in targets
    ]
    missing = targets - {name for name, _ in resolved}
    if missing:
        raise RuntimeError(
            f"Direct identity LoRA target parameters were not found: {sorted(missing)}"
        )

    specifications: list[tuple[str, torch.dtype, int, int]] = []
    invalid: dict[str, str] = {}
    owners: set[str] = set()
    for name, parameter in resolved:
        module_name, separator, parameter_name = name.rpartition(".")
        module = model.get_submodule(module_name) if separator else None
        if parameter_name != "weight":
            invalid[name] = f"parameter leaf is {parameter_name!r}, expected 'weight'"
        elif not isinstance(module, torch.nn.Linear):
            invalid[name] = f"owner is {type(module).__name__}, expected Linear"
        elif module.weight is not parameter:
            invalid[name] = "owner weight does not resolve to the selected parameter"
        elif parameter.ndim != 2:
            invalid[name] = f"shape is {tuple(parameter.shape)}, expected 2D"
        elif parameter.device.type != "meta":
            invalid[name] = f"device is {parameter.device}, expected meta"
        elif parameter.dtype != torch.bfloat16:
            invalid[name] = f"dtype is {parameter.dtype}, expected torch.bfloat16"
        elif module_name in owners:
            invalid[name] = f"owner {module_name!r} has multiple selected parameters"
        else:
            owners.add(module_name)
            specifications.append(
                (module_name, parameter.dtype, parameter.shape[1], parameter.shape[0])
            )
    if invalid:
        details = "; ".join(f"{name}: {reason}" for name, reason in invalid.items())
        raise RuntimeError(f"Invalid direct identity LoRA targets: {details}")

    tensors: dict[str, torch.Tensor] = {}
    expected: dict[str, tuple[tuple[int, int], torch.dtype, bool]] = {}
    for module_name, dtype, in_features, out_features in specifications:
        lora_a = torch.nn.Linear(
            in_features, rank, bias=False, device="cpu", dtype=torch.float32
        )
        lora_b = torch.nn.Linear(
            rank, out_features, bias=False, device="cpu", dtype=torch.float32
        )
        torch.nn.init.kaiming_uniform_(lora_a.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(lora_b.weight)
        prefix = f"base_model.model.{module_name}"
        key_a = f"{prefix}.lora_A.weight"
        key_b = f"{prefix}.lora_B.weight"
        tensors[key_a] = lora_a.weight.detach().to(dtype=dtype).contiguous()
        tensors[key_b] = lora_b.weight.detach().to(dtype=dtype).contiguous()
        expected[key_a] = ((rank, in_features), dtype, True)
        expected[key_b] = ((out_features, rank), dtype, False)

    if set(tensors) != set(expected) or len(tensors) != 2 * len(specifications):
        raise RuntimeError(
            "Direct identity LoRA tensor keys are incomplete or duplicated"
        )
    invalid_tensors = {
        key: (tuple(tensor.shape), tensor.dtype, tensor.device, tensor.is_contiguous())
        for key, tensor in tensors.items()
        if tuple(tensor.shape) != expected[key][0]
        or tensor.dtype != expected[key][1]
        or tensor.device.type != "cpu"
        or not tensor.is_contiguous()
        or bool(torch.count_nonzero(tensor).item()) != expected[key][2]
    }
    if invalid_tensors:
        raise RuntimeError(f"Invalid direct identity LoRA tensors: {invalid_tensors}")
    return tensors


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
    """Create and atomically publish an identity LoRA adapter."""
    from unittest.mock import patch

    from accelerate import init_empty_weights
    from peft import get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    from .model_support import get_model_support_handler

    destination = Path(lora_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"identity LoRA destination already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = (
        destination.parent.parent / "megatron_runtime/staging"
        if destination.name == "0000" and destination.parent.name == "checkpoints"
        else destination.parent
    )
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = staging_root / f"identity-{uuid4().hex}"
    staging.mkdir()
    model: Any | None = None
    peft_model: Any | None = None
    source_tensors: dict[str, torch.Tensor] | None = None
    converted_tensors: dict[str, torch.Tensor] | None = None
    try:
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

        target_parameters = handler.identity_lora_target_parameters(
            model,
            target_modules=target_modules,
        )
        final_config = LoraConfig(
            base_model_name_or_path=base_model,
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            bias="none",
        ).to_dict()
        if handler.identity_lora_factory == "direct_2d_linear":
            source_tensors = _direct_2d_linear_identity_tensors(
                model,
                target_parameters=target_parameters,
                rank=rank,
            )
            converted_tensors, published_config = handler.to_vllm_lora_tensors(
                source_tensors,
                adapter_config=final_config,
            )
            save_vllm_lora_tensors(staging, converted_tensors, published_config)
        elif handler.identity_lora_factory == "peft_target_parameters":
            lora_config = LoraConfig(
                base_model_name_or_path=base_model,
                r=rank,
                lora_alpha=lora_alpha,
                target_modules=[],
                target_parameters=target_parameters,
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
            peft_model.save_pretrained(str(staging))
            normalize_lora_checkpoint_to_vllm(
                staging,
                handler=handler,
                adapter_config=final_config,
            )
        else:
            raise RuntimeError(
                f"Unsupported identity LoRA factory {handler.identity_lora_factory!r} "
                f"for handler {handler.key!r}"
            )
        os.replace(staging, destination)
    finally:
        if source_tensors is not None:
            source_tensors.clear()
        if converted_tensors is not None and converted_tensors is not source_tensors:
            converted_tensors.clear()
        del peft_model, model
        if staging.exists():
            shutil.rmtree(staging)
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
