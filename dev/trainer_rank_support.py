from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
import torch.distributed as dist

from art.trainer_rank import MaterializedCheckpoint, TrainerRank


def load_random_checkpoints(
    runtime: Any,
    rank: TrainerRank,
    count: int,
    *,
    base_model: str,
    lora_rank: int = 8,
    site_limit: int | None = None,
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
    config: dict[str, object] = {
        "base_model_name_or_path": base_model,
        "r": lora_rank,
        "lora_alpha": 32,
        "target_modules": list(runtime.model_support_spec.default_target_modules),
    }
    for name, value in {
        "num_attention_heads": getattr(runtime.provider, "num_attention_heads", None),
        "num_key_value_heads": getattr(runtime.provider, "num_query_groups", None),
        "head_dim": getattr(runtime.provider, "kv_channels", None),
        "hidden_size": getattr(runtime.provider, "hidden_size", None),
    }.items():
        if value is not None:
            config[name] = int(value)
    names = tuple(f"S{index}" for index in range(count))
    from art.megatron.model_support.lora_disk import save_vllm_lora_tensors

    with TemporaryDirectory(prefix="trainer-rank-checkpoints-") as root:
        for index, name in enumerate(names):
            generator = torch.Generator().manual_seed(index + 1)
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
                tensor = torch.randn(shape, generator=generator).to(dtype)
                adapter[meta.key] = tensor if is_a else tensor.mul_(1e-3)
            tensors, published_config = (
                runtime.model_support_handler.to_vllm_lora_tensors(
                    adapter,
                    adapter_config=dict(config),
                )
            )
            path = Path(root, name)
            save_vllm_lora_tensors(path, tensors, published_config)
            with rank.push_checkpoint(MaterializedCheckpoint(name, str(path))):
                pass
    return names
