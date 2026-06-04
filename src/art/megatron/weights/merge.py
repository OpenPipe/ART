import importlib
import json
from pathlib import Path
from typing import Any

import torch

from art.megatron.model_support.lora_disk import (
    load_adapter_config,
    load_lora_tensors_for_megatron,
    normalize_lora_checkpoint_to_vllm,
    resolve_lora_handler,
    save_vllm_lora_tensors,
)
from art.megatron.model_support.spec import ModelSupportHandler

safetensors = importlib.import_module("safetensors")
safetensors_torch = importlib.import_module("safetensors.torch")
safe_open = safetensors.safe_open
save_file = safetensors_torch.save_file


def _merge_sharded_tensor(
    key: str,
    *,
    ordered_shards: list[torch.Tensor],
    manifest: dict[str, Any],
) -> torch.Tensor:
    strategy = manifest.get("export_shard_strategy")
    assert strategy is not None
    axis = int(manifest.get("export_shard_dim", 1 if "lora_A" in key else 0))
    if strategy == "componentwise":
        component_sizes = [int(size) for size in manifest.get("component_sizes", [])]
        world_size = int(manifest["shard_world_size"])
        if not component_sizes:
            raise RuntimeError(
                f"Missing component_sizes for key={key} shard strategy={strategy}"
            )
        local_sizes = []
        for size in component_sizes:
            if size % world_size != 0:
                raise RuntimeError(
                    f"Component size {size} is not divisible by shard_world_size={world_size} for key={key}"
                )
            local_sizes.append(size // world_size)
        split_shards = [
            torch.split(shard, local_sizes, dim=axis) for shard in ordered_shards
        ]
        merged_components = [
            torch.cat([parts[index] for parts in split_shards], dim=axis)
            for index in range(len(local_sizes))
        ]
        return torch.cat(merged_components, dim=axis).contiguous()
    if strategy != "uniform":
        raise RuntimeError(f"Unsupported shard strategy={strategy} for key={key}")
    return torch.cat(ordered_shards, dim=axis).contiguous()


def merge_sharded_adapter_entries(
    entries_by_key: dict[str, list[tuple[dict[str, Any], torch.Tensor]]],
) -> dict[str, torch.Tensor]:
    adapter_model: dict[str, torch.Tensor] = {}
    for key, key_entries in entries_by_key.items():
        first_manifest = key_entries[0][0]
        sharded = bool(first_manifest["sharded"])
        shard_world_size = int(first_manifest["shard_world_size"])
        for manifest_entry, _tensor in key_entries:
            if bool(manifest_entry["sharded"]) != sharded:
                raise RuntimeError(f"Inconsistent sharded flag for key={key}")
            if int(manifest_entry["shard_world_size"]) != shard_world_size:
                raise RuntimeError(f"Inconsistent shard world size for key={key}")

        if not sharded:
            if len(key_entries) != 1:
                raise RuntimeError(
                    f"Replicated key={key} expected 1 shard, got {len(key_entries)}"
                )
            adapter_model[key] = key_entries[0][1]
            continue

        shard_rank_to_tensor: dict[int, torch.Tensor] = {}
        for manifest_entry, shard_tensor in key_entries:
            shard_rank = int(manifest_entry["shard_rank"])
            if shard_rank in shard_rank_to_tensor:
                raise RuntimeError(f"Duplicate shard_rank={shard_rank} for key={key}")
            shard_rank_to_tensor[shard_rank] = shard_tensor

        expected_shard_ranks = set(range(shard_world_size))
        if set(shard_rank_to_tensor) != expected_shard_ranks:
            raise RuntimeError(
                f"Shard rank coverage mismatch for key={key}: "
                f"expected {sorted(expected_shard_ranks)}, got {sorted(shard_rank_to_tensor)}"
            )

        ordered_shards = [
            shard_rank_to_tensor[shard_rank] for shard_rank in range(shard_world_size)
        ]
        adapter_model[key] = _merge_sharded_tensor(
            key,
            ordered_shards=ordered_shards,
            manifest=first_manifest,
        )
    return adapter_model


def _load_adapter_shards(
    base_dir: Path,
) -> tuple[
    dict[str, torch.Tensor],
    list[Path],
    list[Path],
]:
    shard_filenames = sorted(base_dir.glob("adapter_model-*-of-*.safetensors"))
    if not shard_filenames:
        raise FileNotFoundError(f"No adapter shards found in {base_dir}")

    shard_files_by_suffix = {
        path.name.removeprefix("adapter_model-").removesuffix(".safetensors"): path
        for path in shard_filenames
    }
    manifest_filenames = sorted(base_dir.glob("adapter_manifest-*-of-*.json"))
    manifest_files_by_suffix = {
        path.name.removeprefix("adapter_manifest-").removesuffix(".json"): path
        for path in manifest_filenames
    }

    if set(shard_files_by_suffix) != set(manifest_files_by_suffix):
        raise RuntimeError(
            "Shard/manifest coverage mismatch: "
            f"shards={sorted(shard_files_by_suffix)}, "
            f"manifests={sorted(manifest_files_by_suffix)}"
        )

    entries_by_key: dict[str, list[tuple[dict[str, Any], torch.Tensor]]] = {}
    for suffix in sorted(shard_files_by_suffix):
        shard_path = shard_files_by_suffix[suffix]
        manifest_path = manifest_files_by_suffix[suffix]
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            shard_manifest: dict[str, dict[str, Any]] = json.load(manifest_file)
        with safe_open(shard_path, framework="pt") as file:
            shard_tensors = {key: file.get_tensor(key) for key in file.keys()}

        if set(shard_tensors) != set(shard_manifest):
            raise RuntimeError(
                f"Tensor/manifest key mismatch for shard suffix={suffix}: "
                f"tensor_keys={sorted(shard_tensors)}, "
                f"manifest_keys={sorted(shard_manifest)}"
            )
        for key, tensor in shard_tensors.items():
            entries_by_key.setdefault(key, []).append((shard_manifest[key], tensor))

    adapter_model = merge_sharded_adapter_entries(entries_by_key)
    return adapter_model, shard_filenames, manifest_filenames


def load_lora_adapter_state_dict(
    lora_path: str,
    *,
    handler: ModelSupportHandler | None = None,
    allow_unvalidated_arch: bool = False,
) -> dict[str, torch.Tensor]:
    base_dir = Path(lora_path)
    adapter_model_path = base_dir / "adapter_model.safetensors"
    if adapter_model_path.exists():
        return load_lora_tensors_for_megatron(
            lora_path,
            handler=handler,
            allow_unvalidated_arch=allow_unvalidated_arch,
        )

    adapter_model, _shard_filenames, _manifest_filenames = _load_adapter_shards(
        base_dir
    )
    return adapter_model


def merge_lora_adapter(
    lora_path: str,
    *,
    output_dir: str | Path | None = None,
    allow_unvalidated_arch: bool = False,
) -> None:
    base_dir = Path(lora_path)
    adapter_model, shard_filenames, manifest_filenames = _load_adapter_shards(base_dir)
    target_dir = Path(output_dir) if output_dir is not None else base_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    if target_dir == base_dir:
        save_file(adapter_model, base_dir / "adapter_model.safetensors")
        normalize_lora_checkpoint_to_vllm(
            base_dir,
            allow_unvalidated_arch=allow_unvalidated_arch,
        )
    else:
        handler = resolve_lora_handler(
            base_dir,
            allow_unvalidated_arch=allow_unvalidated_arch,
        )
        adapter_config = load_adapter_config(base_dir)
        tensors, adapter_config = handler.to_vllm_lora_tensors(
            adapter_model,
            adapter_config=adapter_config,
        )
        save_vllm_lora_tensors(target_dir, tensors, adapter_config)
    for filename in shard_filenames:
        filename.unlink()
    for filename in manifest_filenames:
        filename.unlink()
