from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple

import torch

from art.megatron.lora import (
    LoRA,
    LoRAPublishPlanner,
    LoraShardMeta,
    _block_for_key,
    _dtype_name,
)
from art.megatron.lora import (
    _distributed_initialized as _distributed_ready,
)
from art.megatron.model_support.lora_disk import save_vllm_lora_tensors
from art.megatron.model_support.spec import ExpertPackedLoraGroup, ExpertPackedLoraSlot
from art.megatron.training.model_chunks import ModelChunks


class PackedExpertShardMeta(NamedTuple):
    key: str
    owner_rank: int
    shape: tuple[int, ...]
    dtype_name: str
    manifest: dict[str, Any]
    expert_start: int
    expert_count: int
    pack_layout: str

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total


class _PinnedCpuStager:
    def __init__(self) -> None:
        self._events: list[torch.cuda.Event] = []
        self._stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def stage(self, tensor: torch.Tensor) -> torch.Tensor:
        source = tensor.detach()
        if self._stream is None or not source.is_cuda:
            return source.cpu()

        source = source.contiguous()
        target = torch.empty_like(source, device="cpu", pin_memory=True)
        source_stream = torch.cuda.current_stream(source.device)
        self._stream.wait_stream(source_stream)
        with torch.cuda.stream(self._stream):
            target.copy_(source, non_blocking=True)
            source.record_stream(self._stream)
            event = torch.cuda.Event()
            event.record(self._stream)
        self._events.append(event)
        return target

    def finish(self) -> None:
        for event in self._events:
            event.synchronize()
        self._events.clear()


def iter_lora_modules(model_chunks: ModelChunks) -> Iterable[LoRA]:
    for chunk in model_chunks:
        for module in chunk.modules():
            if isinstance(module, LoRA):
                yield module


def _dtype_from_name(name: str) -> torch.dtype:
    if isinstance(dtype := getattr(torch, name, None), torch.dtype):
        return dtype
    raise RuntimeError(f"Unsupported LoRA tensor dtype={name!r}")


def _packed_expert_slot(
    adapter_model_prefix: str,
    suffix: str,
    groups: Sequence[ExpertPackedLoraGroup],
) -> tuple[str, ExpertPackedLoraSlot] | None:
    group_prefix, separator, projection = adapter_model_prefix.partition(".{expert}.")
    if not separator:
        return None
    lora_name = suffix.removesuffix(".weight")
    for group in groups:
        if not group_prefix.endswith(group.art_group_suffix):
            continue
        for slot in group.slots:
            if slot.source_projection == projection and slot.source_lora == lora_name:
                return group_prefix, slot
    return None


def _uses_packed_expert_publish(
    module: LoRA,
    groups: Sequence[ExpertPackedLoraGroup],
) -> bool:
    if module.num_local_experts <= 1:
        return False
    params = tuple(module._lora_params())
    return bool(params) and all(
        _packed_expert_slot(module.adapter_model_prefix, suffix, groups) is not None
        for suffix, _param in params
    )


def collect_local_lora_entries(
    model_chunks: ModelChunks,
    adapter_model: dict[str, torch.Tensor],
    *,
    owner_rank: int,
    packed_expert_groups: Sequence[ExpertPackedLoraGroup] = (),
) -> tuple[dict[str, torch.Tensor], list[LoraShardMeta]]:
    local_tensors: dict[str, torch.Tensor] = {}
    local_manifest: dict[str, dict[str, Any]] = {}
    for module in iter_lora_modules(model_chunks):
        if _uses_packed_expert_publish(module, packed_expert_groups):
            continue
        for key, value in module.sharded_lora_state_dict().items():
            target_dtype = (
                adapter_model[key].dtype if key in adapter_model else value.dtype
            )
            local_tensors[key] = value.to(target_dtype).contiguous()
        local_manifest.update(module.sharded_lora_manifest())

    if set(local_tensors) != set(local_manifest):
        raise RuntimeError(
            "LoRA tensor/manifest mismatch: "
            f"tensors={sorted(local_tensors)}, manifest={sorted(local_manifest)}"
        )

    metadata = [
        LoraShardMeta(
            key=key,
            owner_rank=owner_rank,
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype_name=_dtype_name(tensor.dtype),
            manifest=local_manifest[key],
            block=_block_for_key(key),
        )
        for key, tensor in local_tensors.items()
    ]
    return local_tensors, metadata


def collect_local_packed_expert_entries(
    model_chunks: ModelChunks,
    adapter_model: dict[str, torch.Tensor],
    *,
    owner_rank: int,
    packed_expert_groups: Sequence[ExpertPackedLoraGroup],
) -> tuple[dict[str, torch.Tensor], list[PackedExpertShardMeta]]:
    local_tensors: dict[str, torch.Tensor] = {}
    metadata: list[PackedExpertShardMeta] = []
    for module in iter_lora_modules(model_chunks):
        if not _uses_packed_expert_publish(module, packed_expert_groups):
            continue
        expert_start = int(module._expert_offset)
        expert_count = int(module.num_local_experts)
        for suffix, param in module._lora_params():
            slot_match = _packed_expert_slot(
                module.adapter_model_prefix,
                suffix,
                packed_expert_groups,
            )
            if slot_match is None or not module._should_export_parameter(param):
                continue
            group_prefix, slot = slot_match
            key = f"{group_prefix}.{slot.output_suffix}"
            tensor = param.data.transpose(1, 2).contiguous()
            source_keys = module._expected_weight_keys(suffix.removesuffix(".weight"))
            target_dtype = (
                adapter_model[source_keys[0]].dtype
                if source_keys and source_keys[0] in adapter_model
                else tensor.dtype
            )
            tensor = tensor.to(target_dtype).contiguous()
            if key in local_tensors:
                raise RuntimeError(f"Duplicate packed expert LoRA tensor: {key}")
            local_tensors[key] = tensor
            metadata.append(
                PackedExpertShardMeta(
                    key=key,
                    owner_rank=owner_rank,
                    shape=tuple(int(dim) for dim in tensor.shape),
                    dtype_name=_dtype_name(tensor.dtype),
                    manifest=module._manifest_for_param(param),
                    expert_start=expert_start,
                    expert_count=expert_count,
                    pack_layout=slot.pack_layout,
                )
            )
    return local_tensors, metadata


def _global_packed_expert_metadata(
    planner: LoRAPublishPlanner,
    adapter_model: dict[str, torch.Tensor],
    packed_expert_groups: Sequence[ExpertPackedLoraGroup],
) -> list[PackedExpertShardMeta]:
    metadata: list[PackedExpertShardMeta] = []
    for template in planner.templates:
        if int(template.num_local_experts) <= 1:
            continue
        slot_match = _packed_expert_slot(
            template.adapter_model_prefix,
            template.suffix,
            packed_expert_groups,
        )
        if slot_match is None:
            continue
        group_prefix, slot = slot_match
        shard_ranks = range(template.shard_world_size) if template.sharded else (0,)
        ep_world_size = 1
        if _distributed_ready():
            from megatron.core import parallel_state as ps

            ep_world_size = ps.get_expert_model_parallel_world_size()
        for ep_rank in range(ep_world_size):
            expert_start = ep_rank * template.num_local_experts
            expert_key = (
                f"{template.adapter_model_prefix.format(expert=expert_start)}."
                f"{template.suffix}"
            )
            for shard_rank in shard_ranks:
                owner_rank = planner._expert_owner_rank(ep_rank, shard_rank)
                per_expert_meta = planner._make_metadata(
                    template,
                    key=expert_key,
                    owner_rank=owner_rank,
                    shard_rank=shard_rank,
                    adapter_model=adapter_model,
                )
                metadata.append(
                    PackedExpertShardMeta(
                        key=f"{group_prefix}.{slot.output_suffix}",
                        owner_rank=owner_rank,
                        shape=(template.num_local_experts, *per_expert_meta.shape),
                        dtype_name=per_expert_meta.dtype_name,
                        manifest=per_expert_meta.manifest,
                        expert_start=expert_start,
                        expert_count=template.num_local_experts,
                        pack_layout=slot.pack_layout,
                    )
                )
    return metadata


def _global_regular_metadata(
    planner: LoRAPublishPlanner,
    adapter_model: dict[str, torch.Tensor],
    packed_expert_groups: Sequence[ExpertPackedLoraGroup],
) -> list[LoraShardMeta]:
    if not packed_expert_groups:
        return planner.global_metadata(adapter_model)
    if _distributed_ready():
        from megatron.core import parallel_state as ps

        pp_world_size = ps.get_pipeline_model_parallel_world_size()
        if pp_world_size != 1:
            raise RuntimeError(
                "LoRA publish planner requires pipeline_model_parallel_size=1; "
                f"got {pp_world_size}. Rank-local modules cannot describe remote "
                "pipeline stages without exchanging templates."
            )
    metadata: list[LoraShardMeta] = []
    for template in planner.templates:
        if (
            _packed_expert_slot(
                template.adapter_model_prefix,
                template.suffix,
                packed_expert_groups,
            )
            is not None
        ):
            continue
        metadata.extend(planner._metadata_for_template(template, adapter_model))
    return metadata


def _merge_sharded_tensor(
    key: str,
    *,
    ordered_shards: Sequence[torch.Tensor],
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
    return torch.cat(tuple(ordered_shards), dim=axis).contiguous()


def _merge_manifest_entries(
    key: str,
    key_entries: Sequence[tuple[dict[str, Any], torch.Tensor]],
    *,
    manifest: dict[str, Any] | None = None,
) -> torch.Tensor:
    first_manifest = key_entries[0][0]
    sharded = bool(first_manifest["sharded"])
    shard_world_size = int(first_manifest["shard_world_size"])
    for entry_manifest, _tensor in key_entries:
        if bool(entry_manifest["sharded"]) != sharded:
            raise RuntimeError(f"Inconsistent sharded flag for key={key}")
        if int(entry_manifest["shard_world_size"]) != shard_world_size:
            raise RuntimeError(f"Inconsistent shard world size for key={key}")

    if not sharded:
        if len(key_entries) != 1:
            raise RuntimeError(
                f"Replicated key={key} expected 1 shard, got {len(key_entries)}"
            )
        return key_entries[0][1]

    shard_rank_to_tensor: dict[int, torch.Tensor] = {}
    for entry_manifest, shard_tensor in key_entries:
        shard_rank = int(entry_manifest["shard_rank"])
        if shard_rank in shard_rank_to_tensor:
            raise RuntimeError(f"Duplicate shard_rank={shard_rank} for key={key}")
        shard_rank_to_tensor[shard_rank] = shard_tensor

    expected_shard_ranks = set(range(shard_world_size))
    if set(shard_rank_to_tensor) != expected_shard_ranks:
        raise RuntimeError(
            f"Shard rank coverage mismatch for key={key}: "
            f"expected {sorted(expected_shard_ranks)}, got {sorted(shard_rank_to_tensor)}"
        )
    return _merge_sharded_tensor(
        key,
        ordered_shards=[
            shard_rank_to_tensor[shard_rank] for shard_rank in range(shard_world_size)
        ],
        manifest=first_manifest if manifest is None else manifest,
    )


def merge_sharded_adapter_entries(
    entries_by_key: dict[str, list[tuple[dict[str, Any], torch.Tensor]]],
) -> dict[str, torch.Tensor]:
    return {
        key: _merge_manifest_entries(key, key_entries)
        for key, key_entries in entries_by_key.items()
    }


def _rank_and_device() -> tuple[int, torch.device]:
    return (
        torch.distributed.get_rank() if _distributed_ready() else 0,  # type: ignore[possibly-missing-attribute]
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu"),
    )


def _metadata_by_owner_dtype(
    metadata: Sequence[Any],
) -> dict[tuple[int, str], list[Any]]:
    grouped: dict[tuple[int, str], list[Any]] = {}
    for meta in metadata:
        grouped.setdefault((meta.owner_rank, meta.dtype_name), []).append(meta)
    return {
        key: sorted(group, key=lambda meta: meta.key)
        for key, group in sorted(grouped.items())
    }


def _pack_metadata_tensors(
    metadata: Sequence[Any],
    tensors: dict[str, torch.Tensor],
) -> torch.Tensor:
    return torch.cat(
        [tensors[meta.key].detach().contiguous().view(-1) for meta in metadata]
    )


def _views_from_flat(
    *,
    owner_rank: int,
    metadata: Sequence[Any],
    flat: torch.Tensor,
) -> dict[tuple[int, str], torch.Tensor]:
    views: dict[tuple[int, str], torch.Tensor] = {}
    offset = 0
    for meta in metadata:
        views[(owner_rank, meta.key)] = flat.narrow(0, offset, meta.numel).view(
            meta.shape
        )
        offset += meta.numel
    return views


def _exchange_batched_tensors(
    metadata: Sequence[Any],
    *,
    local_tensors: dict[str, torch.Tensor],
    rank: int,
    device: torch.device,
) -> dict[tuple[int, str], torch.Tensor]:
    if not _distributed_ready():
        return {
            (rank, meta.key): local_tensors[meta.key].contiguous() for meta in metadata
        }

    received: dict[tuple[int, str], torch.Tensor] = {}
    for (owner_rank, dtype_name), group_metadata in _metadata_by_owner_dtype(
        metadata
    ).items():
        if rank == owner_rank:
            flat = _pack_metadata_tensors(group_metadata, local_tensors)
            if rank == 0:
                received.update(
                    _views_from_flat(
                        owner_rank=owner_rank,
                        metadata=group_metadata,
                        flat=flat,
                    )
                )
            else:
                torch.distributed.send(flat, dst=0)  # type: ignore[possibly-missing-attribute]
        elif rank == 0:
            flat = torch.empty(
                sum(meta.numel for meta in group_metadata),
                dtype=_dtype_from_name(dtype_name),
                device=device,
            )
            torch.distributed.recv(flat, src=owner_rank)  # type: ignore[possibly-missing-attribute]
            received.update(
                _views_from_flat(
                    owner_rank=owner_rank,
                    metadata=group_metadata,
                    flat=flat,
                )
            )
    return received


def _entries_by_key(
    metadata: list[LoraShardMeta],
    tensors_by_owner_key: dict[tuple[int, str], torch.Tensor],
) -> dict[str, list[tuple[dict[str, Any], torch.Tensor]]]:
    entries: dict[str, list[tuple[dict[str, Any], torch.Tensor]]] = {}
    for meta in metadata:
        entries.setdefault(meta.key, []).append(
            (meta.manifest, tensors_by_owner_key[(meta.owner_rank, meta.key)])
        )
    return entries


def _merge_packed_expert_block(
    key: str,
    key_entries: list[tuple[dict[str, Any], torch.Tensor]],
) -> torch.Tensor:
    manifest = dict(key_entries[0][0])
    if bool(manifest["sharded"]):
        manifest["export_shard_dim"] = int(manifest["export_shard_dim"]) + 1
    return _merge_manifest_entries(key, key_entries, manifest=manifest)


def _pack_merged_expert_blocks(
    key: str,
    blocks: list[tuple[PackedExpertShardMeta, torch.Tensor]],
) -> torch.Tensor:
    first_layout = blocks[0][0].pack_layout
    next_expert = 0
    ordered_blocks: list[torch.Tensor] = []
    for meta, block in sorted(blocks, key=lambda item: item[0].expert_start):
        if meta.pack_layout != first_layout:
            raise RuntimeError(f"Inconsistent packed layout for key={key}")
        if meta.expert_start != next_expert:
            raise RuntimeError(
                f"Packed expert coverage mismatch for key={key}: "
                f"expected expert_start={next_expert}, got {meta.expert_start}"
            )
        if int(block.shape[0]) != meta.expert_count:
            raise RuntimeError(
                f"Packed expert block shape mismatch for key={key}: "
                f"shape={tuple(block.shape)} expert_count={meta.expert_count}"
            )
        ordered_blocks.append(block)
        next_expert += meta.expert_count

    joined = torch.cat(ordered_blocks, dim=0)
    if first_layout == "expert_rows":
        if joined.ndim != 3:
            raise RuntimeError(f"{key}: expert_rows layout requires 3D blocks")
        return joined.flatten(0, 1).contiguous()
    if first_layout == "rank_major_expert_cols":
        if joined.ndim != 3:
            raise RuntimeError(
                f"{key}: rank_major_expert_cols layout requires 3D blocks"
            )
        return (
            joined.permute(1, 2, 0)
            .reshape(
                joined.shape[1],
                joined.shape[2] * joined.shape[0],
            )
            .contiguous()
        )
    raise RuntimeError(f"Unsupported packed expert LoRA layout={first_layout!r}")


def merge_packed_expert_adapter_entries(
    metadata: list[PackedExpertShardMeta],
    tensors_by_owner_key: dict[tuple[int, str], torch.Tensor],
) -> dict[str, torch.Tensor]:
    entries_by_key_start: dict[
        tuple[str, int],
        list[tuple[PackedExpertShardMeta, dict[str, Any], torch.Tensor]],
    ] = {}
    for meta in metadata:
        entries_by_key_start.setdefault((meta.key, meta.expert_start), []).append(
            (
                meta,
                meta.manifest,
                tensors_by_owner_key[(meta.owner_rank, meta.key)],
            )
        )

    blocks_by_key: dict[str, list[tuple[PackedExpertShardMeta, torch.Tensor]]] = {}
    for (key, _expert_start), entries in entries_by_key_start.items():
        representative = entries[0][0]
        block = _merge_packed_expert_block(
            key,
            [(manifest, tensor) for _meta, manifest, tensor in entries],
        )
        blocks_by_key.setdefault(key, []).append((representative, block))

    return {
        key: _pack_merged_expert_blocks(key, blocks)
        for key, blocks in blocks_by_key.items()
    }


def _stage_published_tensors(
    tensors: dict[str, torch.Tensor],
    stager: _PinnedCpuStager,
) -> dict[str, torch.Tensor]:
    grouped: dict[tuple[str, int | None, str], list[tuple[str, torch.Tensor]]] = {}
    for key, tensor in tensors.items():
        dtype_name = _dtype_name(tensor.dtype)
        group_key = (tensor.device.type, tensor.device.index, dtype_name)
        grouped.setdefault(group_key, []).append((key, tensor))

    staged: dict[str, torch.Tensor] = {}
    for _group_key, group in sorted(grouped.items()):
        flat = torch.cat(
            [tensor.detach().contiguous().view(-1) for _key, tensor in sorted(group)]
        )
        staged_flat = stager.stage(flat)
        offset = 0
        for key, tensor in sorted(group):
            numel = tensor.numel()
            if key in staged:
                raise RuntimeError(
                    f"Duplicate vLLM LoRA tensor after conversion: {key}"
                )
            staged[key] = staged_flat.narrow(0, offset, numel).view(tensor.shape)
            offset += numel
    return staged


def _save_rank0_vllm_lora(
    *,
    metadata: list[LoraShardMeta],
    tensors_by_owner_key: dict[tuple[int, str], torch.Tensor],
    packed_expert_metadata: list[PackedExpertShardMeta] | None = None,
    packed_expert_tensors_by_owner_key: (
        dict[tuple[int, str], torch.Tensor] | None
    ) = None,
    handler: Any,
    adapter_config: dict[str, Any],
    output_dir: str,
) -> None:
    merged_tensors = merge_sharded_adapter_entries(
        _entries_by_key(metadata, tensors_by_owner_key)
    )
    if packed_expert_metadata:
        if packed_expert_tensors_by_owner_key is None:
            raise RuntimeError("Missing packed expert tensors for LoRA publish")
        packed_tensors = merge_packed_expert_adapter_entries(
            packed_expert_metadata,
            packed_expert_tensors_by_owner_key,
        )
        for key, tensor in packed_tensors.items():
            if key in merged_tensors:
                raise RuntimeError(f"Duplicate LoRA tensor after packed publish: {key}")
            merged_tensors[key] = tensor
    vllm_tensors, published_config = handler.to_vllm_lora_tensors(
        merged_tensors,
        adapter_config=dict(adapter_config),
    )
    stager = _PinnedCpuStager()
    published_tensors = _stage_published_tensors(vllm_tensors, stager)
    stager.finish()
    save_vllm_lora_tensors(output_dir, published_tensors, published_config)


def save_vllm_lora_from_model(
    *,
    model: ModelChunks,
    adapter_model: dict[str, torch.Tensor],
    handler: Any,
    adapter_config: dict[str, Any],
    output_dir: str,
    rank: int,
    world_size: int,
) -> None:
    actual_rank, device = _rank_and_device()
    if _distributed_ready():
        actual_world_size = torch.distributed.get_world_size()  # type: ignore[possibly-missing-attribute]
        if actual_rank != rank or actual_world_size != world_size:
            raise RuntimeError(
                "LoRA publisher rank/world-size mismatch: "
                f"runtime=({rank}, {world_size}) distributed=({actual_rank}, {actual_world_size})"
            )
    else:
        if rank != 0 or world_size != 1:
            raise RuntimeError(
                "Non-distributed LoRA publish requires rank=0 and world_size=1, "
                f"got rank={rank} world_size={world_size}"
            )
        rank = 0
    packed_expert_groups = tuple(handler.expert_packed_lora_groups())
    planner = LoRAPublishPlanner(model)
    local_tensors, local_metadata = collect_local_lora_entries(
        model,
        adapter_model,
        owner_rank=rank,
        packed_expert_groups=packed_expert_groups,
    )
    local_packed_tensors, local_packed_metadata = collect_local_packed_expert_entries(
        model,
        adapter_model,
        owner_rank=rank,
        packed_expert_groups=packed_expert_groups,
    )
    all_packed_metadata = (
        _global_packed_expert_metadata(planner, adapter_model, packed_expert_groups)
        if rank == 0
        else local_packed_metadata
    )
    if rank == 0:
        all_metadata = _global_regular_metadata(
            planner,
            adapter_model,
            packed_expert_groups if all_packed_metadata else (),
        )
    else:
        all_metadata = local_metadata
    exchanged_tensors = _exchange_batched_tensors(
        all_metadata,
        local_tensors=local_tensors,
        rank=rank,
        device=device,
    )
    exchanged_packed_tensors = _exchange_batched_tensors(
        all_packed_metadata,
        local_tensors=local_packed_tensors,
        rank=rank,
        device=device,
    )

    if rank != 0:
        return

    _save_rank0_vllm_lora(
        metadata=all_metadata,
        tensors_by_owner_key=exchanged_tensors,
        packed_expert_metadata=all_packed_metadata,
        packed_expert_tensors_by_owner_key=exchanged_packed_tensors,
        handler=handler,
        adapter_config=adapter_config,
        output_dir=output_dir,
    )
