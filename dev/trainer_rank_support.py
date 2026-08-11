from typing import Any

import torch
import torch.distributed as dist

from art.trainer_rank import TrainerRank


def load_random_checkpoint_slots(
    runtime: Any,
    rank: TrainerRank,
    count: int,
    *,
    lora_rank: int = 8,
    site_limit: int | None = None,
) -> tuple[str, ...]:
    assert count >= 0, "slots must be >= 0"
    if count == 0:
        return ()
    from art.megatron.weights.lora_publish import collect_local_lora_entries

    _tensors, local_metadata = collect_local_lora_entries(
        runtime.model, {}, owner_rank=dist.get_rank()
    )
    gathered: list[list[Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_metadata)
    metadata = {meta.key: meta for values in gathered if values for meta in values}
    selected = sorted(metadata.values(), key=lambda item: item.key)
    if site_limit is not None:
        pairs = []
        for meta in selected:
            if ".lora_A." not in meta.key or ".experts." in meta.key:
                continue
            b_key = meta.key.replace(".lora_A.", ".lora_B.")
            if b_meta := metadata.get(b_key):
                pairs.append((meta, b_meta))
        selected = [meta for pair in pairs[:site_limit] for meta in pair]
        if not selected:
            raise RuntimeError("No replicated LoRA sites are available for the check")
    dtype = next(runtime.model[0].parameters()).dtype
    names = tuple(f"S{index}" for index in range(count))
    for index, name in enumerate(names):
        generator = torch.Generator(device=rank.device).manual_seed(index + 1)
        adapter: dict[str, torch.Tensor] = {}
        for meta in selected:
            shape = list(meta.shape)
            if meta.manifest["sharded"]:
                axis = int(meta.manifest["export_shard_dim"])
                shape[axis] = sum(
                    map(
                        int,
                        meta.manifest.get("component_sizes")
                        or [shape[axis] * int(meta.manifest["shard_world_size"])],
                    )
                )
            is_a = ".lora_A." in meta.key
            shape[0 if is_a else -1] = lora_rank
            tensor = torch.randn(
                shape, device=rank.device, dtype=dtype, generator=generator
            )
            adapter[meta.key] = tensor if is_a else tensor.mul_(1e-3)
        loaded = rank._load_checkpoint_slot(name, adapter, alpha=lora_rank)
        assert loaded > 0, "TrainerRank check requires installed LoRA adapter sites"
        ref = rank._slot_ref(name)
        rank._checkpoint_slot_params_by_name[name] = tuple(
            rank._iter_slot_parameters(ref)
        )
        rank._checkpoint_revisions[name] = 0
    return names
