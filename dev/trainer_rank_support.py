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
) -> tuple[str, ...]:
    assert count >= 0, "slots must be >= 0"
    if count == 0:
        return ()
    from art.megatron.lora import LoRAPublishPlanner

    gathered: list[list[Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(
        gathered, LoRAPublishPlanner(runtime.model).global_metadata({})
    )
    metadata = {meta.key: meta for values in gathered if values for meta in values}
    dtype = next(runtime.model[0].parameters()).dtype
    names = tuple(f"S{index}" for index in range(count))
    for index, name in enumerate(names):
        generator = torch.Generator(device=rank.device).manual_seed(index + 1)
        adapter: dict[str, torch.Tensor] = {}
        for meta in sorted(metadata.values(), key=lambda item: item.key):
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
        assert rank.load_checkpoint_slot(name, adapter) > 0, (
            "TrainerRank check requires installed LoRA adapter sites"
        )
    return names
