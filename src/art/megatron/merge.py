import importlib
import json
from pathlib import Path
from typing import Any

import torch

safetensors = importlib.import_module("safetensors")
safetensors_torch = importlib.import_module("safetensors.torch")
safe_open = safetensors.safe_open
save_file = safetensors_torch.save_file


def merge_lora_adapter(lora_path: str) -> None:
    base_dir = Path(lora_path)
    shard_filenames = sorted(base_dir.glob("adapter_model-*-of-*.safetensors"))
    if not shard_filenames:
        return

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
            tensor = key_entries[0][1]
        else:
            shard_rank_to_tensor: dict[int, torch.Tensor] = {}
            for manifest_entry, shard_tensor in key_entries:
                shard_rank = int(manifest_entry["shard_rank"])
                if shard_rank in shard_rank_to_tensor:
                    raise RuntimeError(
                        f"Duplicate shard_rank={shard_rank} for key={key}"
                    )
                shard_rank_to_tensor[shard_rank] = shard_tensor

            expected_shard_ranks = set(range(shard_world_size))
            if set(shard_rank_to_tensor) != expected_shard_ranks:
                raise RuntimeError(
                    f"Shard rank coverage mismatch for key={key}: "
                    f"expected {sorted(expected_shard_ranks)}, got {sorted(shard_rank_to_tensor)}"
                )

            ordered_shards = [
                shard_rank_to_tensor[shard_rank]
                for shard_rank in range(shard_world_size)
            ]
            concat_dim = 1 if "lora_A" in key else 0
            tensor = torch.cat(ordered_shards, dim=concat_dim)
        adapter_model[key] = tensor

    adapter_model_path = base_dir / "adapter_model.safetensors"
    save_file(adapter_model, adapter_model_path)
    for filename in shard_filenames:
        filename.unlink()
    for filename in manifest_filenames:
        filename.unlink()
