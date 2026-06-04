import importlib
import json
from pathlib import Path
from typing import Any

import torch

from art.megatron.model_support.spec import ModelSupportHandler

safetensors = importlib.import_module("safetensors")
safetensors_torch = importlib.import_module("safetensors.torch")
safe_open = safetensors.safe_open
save_file = safetensors_torch.save_file


def _jsonable_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable_config(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_jsonable_config(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_jsonable_config(item) for item in value]
    return value


def load_adapter_config(lora_path: str | Path) -> dict[str, Any]:
    config_path = Path(lora_path) / "adapter_config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    return config if isinstance(config, dict) else {}


def save_adapter_config(lora_path: str | Path, adapter_config: dict[str, Any]) -> None:
    config_path = Path(lora_path) / "adapter_config.json"
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(
            _jsonable_config(adapter_config),
            config_file,
            indent=2,
            sort_keys=True,
        )
        config_file.write("\n")


def resolve_lora_handler(
    lora_path: str | Path,
    handler: ModelSupportHandler | None = None,
    *,
    allow_unvalidated_arch: bool = False,
) -> ModelSupportHandler:
    if handler is not None:
        return handler
    base_model = load_adapter_config(lora_path).get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model:
        raise RuntimeError(f"Missing base_model_name_or_path in {lora_path}")
    from art.megatron.model_support import get_model_support_handler

    return get_model_support_handler(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )


def load_vllm_lora_tensors(
    lora_path: str | Path,
) -> dict[str, torch.Tensor]:
    adapter_model_path = Path(lora_path) / "adapter_model.safetensors"
    with safe_open(adapter_model_path, framework="pt") as adapter_file:
        return {key: adapter_file.get_tensor(key) for key in adapter_file.keys()}


def save_vllm_lora_tensors(
    lora_path: str | Path,
    tensors: dict[str, torch.Tensor],
    adapter_config: dict[str, Any],
) -> None:
    base_dir = Path(lora_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, base_dir / "adapter_model.safetensors")
    save_adapter_config(base_dir, adapter_config)


def normalize_lora_checkpoint_to_vllm(
    lora_path: str | Path,
    *,
    handler: ModelSupportHandler | None = None,
    adapter_config: dict[str, Any] | None = None,
    allow_unvalidated_arch: bool = False,
) -> None:
    adapter_model_path = Path(lora_path) / "adapter_model.safetensors"
    if not adapter_model_path.exists():
        return
    resolved_handler = resolve_lora_handler(
        lora_path,
        handler,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    if adapter_config is None:
        adapter_config = load_adapter_config(lora_path)
    tensors = load_vllm_lora_tensors(lora_path)
    tensors, adapter_config = resolved_handler.to_vllm_lora_tensors(
        tensors,
        adapter_config=adapter_config,
    )
    save_vllm_lora_tensors(lora_path, tensors, adapter_config)


def load_lora_tensors_for_megatron(
    lora_path: str | Path,
    *,
    handler: ModelSupportHandler | None = None,
    allow_unvalidated_arch: bool = False,
) -> dict[str, torch.Tensor]:
    resolved_handler = resolve_lora_handler(
        lora_path,
        handler,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    return resolved_handler.from_vllm_lora_tensors(
        load_vllm_lora_tensors(lora_path),
        adapter_config=load_adapter_config(lora_path),
    )
