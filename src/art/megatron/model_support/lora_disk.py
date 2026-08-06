import importlib
import json
from pathlib import Path
import struct
import sys
from typing import Any

import torch

from art.megatron.model_support.spec import ModelSupportHandler

ART_LORA_FORMAT_CONFIG_KEY = "art_lora_format"
ART_LORA_FORMAT_VLLM = "vllm"

safetensors = importlib.import_module("safetensors")
safe_open = safetensors.safe_open

_SAFETENSORS_DTYPES = {
    dtype: name
    for name, dtype in {
        "BOOL": torch.bool,
        "U8": torch.uint8,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F64": torch.float64,
        "C64": torch.complex64,
        "U16": getattr(torch, "uint16", None),
        "U32": getattr(torch, "uint32", None),
        "U64": getattr(torch, "uint64", None),
        "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
        "F8_E5M2": getattr(torch, "float8_e5m2", None),
    }.items()
    if dtype is not None
}


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


def _save_safetensors(tensors: dict[str, torch.Tensor], path: Path) -> None:
    """Stream tensor buffers without save_file's copy into GIL-holding bytes."""
    if sys.byteorder != "little":
        raise RuntimeError("ART's zero-copy safetensors writer requires little endian")
    header: dict[str, Any] = {}
    buffers: list[memoryview] = []
    offset = 0
    for name, tensor in sorted(tensors.items()):
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise RuntimeError(f"LoRA tensor {name!r} must be contiguous CPU storage")
        dtype = _SAFETENSORS_DTYPES.get(tensor.dtype)
        if dtype is None:
            raise RuntimeError(f"Unsupported LoRA tensor dtype: {tensor.dtype}")
        data = memoryview(tensor.view(torch.uint8).reshape(-1).numpy())
        header[name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + data.nbytes],
        }
        offset += data.nbytes
        buffers.append(data)

    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb", buffering=0) as output:
        for data in (memoryview(struct.pack("<Q", len(encoded))), encoded, *buffers):
            while data:
                written = output.write(data)
                if not written:
                    raise OSError(f"Short write while saving {path}")
                data = data[written:]


def save_vllm_lora_tensors(
    lora_path: str | Path,
    tensors: dict[str, torch.Tensor],
    adapter_config: dict[str, Any],
) -> None:
    base_dir = Path(lora_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    _save_safetensors(tensors, base_dir / "adapter_model.safetensors")
    save_adapter_config(
        base_dir,
        {**adapter_config, ART_LORA_FORMAT_CONFIG_KEY: ART_LORA_FORMAT_VLLM},
    )


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
    if adapter_config is None:
        adapter_config = load_adapter_config(lora_path)
    if adapter_config.get(ART_LORA_FORMAT_CONFIG_KEY) == ART_LORA_FORMAT_VLLM:
        return
    resolved_handler = resolve_lora_handler(
        lora_path,
        handler,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
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
