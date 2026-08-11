import importlib
import json
from pathlib import Path
import struct
from typing import Any

import torch

from art.megatron.model_support.spec import ModelSupportHandler

ART_LORA_FORMAT_CONFIG_KEY = "art_lora_format"
ART_LORA_FORMAT_MEGATRON = "megatron"
ART_LORA_FORMAT_VLLM = "vllm"

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
    save_adapter_config(
        base_dir,
        {**adapter_config, ART_LORA_FORMAT_CONFIG_KEY: ART_LORA_FORMAT_VLLM},
    )


def _consolidate_safetensors(
    shards: list[Path], output: Path, *, chunk_size: int = 8 * 1024 * 1024
) -> None:
    """Join safetensors shards without materializing their tensor payloads."""
    sources: dict[str, tuple[Path, int, int, int, dict[str, object]]] = {}
    for shard in shards:
        with shard.open("rb") as handle:
            encoded_length = handle.read(8)
            if len(encoded_length) != 8:
                raise RuntimeError(f"Invalid safetensors header: {shard}")
            header_length = struct.unpack("<Q", encoded_length)[0]
            header = json.loads(handle.read(header_length))
        if not isinstance(header, dict):
            raise RuntimeError(f"Invalid safetensors index: {shard}")
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if key in sources or not isinstance(value, dict):
                raise RuntimeError(f"Invalid safetensors tensor index: {key!r}")
            offsets = value.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(offset, int) for offset in offsets)
                or offsets[0] < 0
                or offsets[1] < offsets[0]
            ):
                raise RuntimeError(f"Invalid safetensors offsets for {key!r}")
            sources[key] = (
                shard,
                8 + header_length,
                offsets[0],
                offsets[1],
                {name: item for name, item in value.items() if name != "data_offsets"},
            )

    offset = 0
    header: dict[str, dict[str, object]] = {}
    for key in sorted(sources):
        _path, _data_start, start, end, metadata = sources[key]
        header[key] = {**metadata, "data_offsets": [offset, offset + end - start]}
        offset += end - start
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as target:
        target.write(struct.pack("<Q", len(encoded)))
        target.write(encoded)
        for key in sorted(sources):
            source_path, data_start, start, end, _metadata = sources[key]
            remaining = end - start
            with source_path.open("rb") as source:
                source.seek(data_start + start)
                while remaining:
                    chunk = source.read(min(chunk_size, remaining))
                    if not chunk:
                        raise RuntimeError(f"Truncated safetensors payload for {key!r}")
                    target.write(chunk)
                    remaining -= len(chunk)


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
    adapter_config = load_adapter_config(lora_path)
    tensors = load_vllm_lora_tensors(lora_path)
    if adapter_config.get(ART_LORA_FORMAT_CONFIG_KEY) == ART_LORA_FORMAT_MEGATRON:
        return tensors
    resolved_handler = resolve_lora_handler(
        lora_path,
        handler,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    return resolved_handler.from_vllm_lora_tensors(
        tensors,
        adapter_config=adapter_config,
    )
