from collections.abc import Iterable, Sequence
import re
from typing import Any, NamedTuple

import torch

from art.megatron.lora import LoRAPublishPlanner, LoraShardMeta
from art.megatron.model_support.lora_disk import save_vllm_lora_tensors
from art.megatron.model_support.spec import ExpertPackedLoraGroup, ExpertPackedLoraSlot
from art.megatron.training.model_chunks import ModelChunks

_LAYER_BLOCK_RE = re.compile(r"^(?P<block>.*\.layers\.\d+)\.")


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


def iter_lora_modules(model_chunks: ModelChunks) -> Iterable[Any]:
    for chunk in model_chunks:
        for module in chunk.modules():
            yield module


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"Unsupported LoRA tensor dtype={name!r}")
    return dtype


def _block_for_key(key: str) -> str:
    match = _LAYER_BLOCK_RE.match(key)
    if match is not None:
        return match.group("block")
    return "__global__"


def _expert_prefix_projection(adapter_model_prefix: str) -> tuple[str, str] | None:
    group_prefix, separator, projection = adapter_model_prefix.partition(".{expert}.")
    if not separator:
        return None
    return group_prefix, projection


def _packed_expert_slot(
    adapter_model_prefix: str,
    suffix: str,
    groups: Sequence[ExpertPackedLoraGroup],
) -> tuple[str, ExpertPackedLoraSlot] | None:
    parts = _expert_prefix_projection(adapter_model_prefix)
    if parts is None:
        return None
    group_prefix, projection = parts
    lora_name = suffix.removesuffix(".weight")
    for group in groups:
        if not group_prefix.endswith(group.art_group_suffix):
            continue
        for slot in group.slots:
            if slot.source_projection == projection and slot.source_lora == lora_name:
                return group_prefix, slot
    return None


def _uses_packed_expert_publish(
    module: Any,
    groups: Sequence[ExpertPackedLoraGroup],
) -> bool:
    if int(getattr(module, "num_local_experts", 1)) <= 1:
        return False
    if not hasattr(module, "_lora_params"):
        return False
    adapter_model_prefix = getattr(module, "adapter_model_prefix", "")
    if not isinstance(adapter_model_prefix, str):
        return False
    lora_suffixes = [
        suffix
        for suffix, _param in module._lora_params()  # type: ignore[attr-defined]
    ]
    return bool(lora_suffixes) and all(
        _packed_expert_slot(adapter_model_prefix, suffix, groups) is not None
        for suffix in lora_suffixes
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
        if hasattr(module, "sharded_lora_state_dict"):
            module_state: dict[str, torch.Tensor] = module.sharded_lora_state_dict()  # type: ignore[attr-defined]
            for key, value in module_state.items():
                target_dtype = (
                    adapter_model[key].dtype if key in adapter_model else value.dtype
                )
                local_tensors[key] = value.to(target_dtype).contiguous()
        if hasattr(module, "sharded_lora_manifest"):
            local_manifest.update(module.sharded_lora_manifest())  # type: ignore[attr-defined]

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


def _target_dtype_for_lora_param(
    module: Any,
    adapter_model: dict[str, torch.Tensor],
    suffix: str,
    fallback: torch.dtype,
) -> torch.dtype:
    keys = module._expected_weight_keys(suffix.removesuffix(".weight"))  # type: ignore[attr-defined]
    return (
        adapter_model[keys[0]].dtype if keys and keys[0] in adapter_model else fallback
    )


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
        adapter_model_prefix = module.adapter_model_prefix  # type: ignore[attr-defined]
        expert_start = int(module._expert_offset)  # type: ignore[attr-defined]
        expert_count = int(module.num_local_experts)  # type: ignore[attr-defined]
        for suffix, param in module._lora_params():  # type: ignore[attr-defined]
            slot_match = _packed_expert_slot(
                adapter_model_prefix,
                suffix,
                packed_expert_groups,
            )
            if slot_match is None or not module._should_export_parameter(param):  # type: ignore[attr-defined]
                continue
            group_prefix, slot = slot_match
            key = f"{group_prefix}.{slot.output_suffix}"
            tensor = param.data.transpose(1, 2).contiguous()
            target_dtype = _target_dtype_for_lora_param(
                module,
                adapter_model,
                suffix,
                tensor.dtype,
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
                    manifest=module._manifest_for_param(param),  # type: ignore[attr-defined]
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
        for ep_rank in range(planner._expert_model_world_size()):
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


def _distributed_ready() -> bool:
    is_initialized = getattr(torch.distributed, "is_initialized", None)
    return (
        torch.distributed.is_available()
        and callable(is_initialized)
        and bool(is_initialized())
    )


def _rank_and_device() -> tuple[int, torch.device]:
    if _distributed_ready():
        rank = torch.distributed.get_rank()  # type: ignore[possibly-missing-attribute]
    else:
        rank = 0
    if torch.cuda.is_available():
        return rank, torch.device("cuda", torch.cuda.current_device())
    return rank, torch.device("cpu")


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
    first_manifest = key_entries[0][0]
    sharded = bool(first_manifest["sharded"])
    shard_world_size = int(first_manifest["shard_world_size"])
    if not sharded:
        if len(key_entries) != 1:
            raise RuntimeError(
                f"Replicated packed key={key} expected 1 shard, got {len(key_entries)}"
            )
        return key_entries[0][1]

    shard_rank_to_tensor: dict[int, torch.Tensor] = {}
    for manifest_entry, shard_tensor in key_entries:
        if bool(manifest_entry["sharded"]) != sharded:
            raise RuntimeError(f"Inconsistent sharded flag for packed key={key}")
        if int(manifest_entry["shard_world_size"]) != shard_world_size:
            raise RuntimeError(f"Inconsistent shard world size for packed key={key}")
        shard_rank = int(manifest_entry["shard_rank"])
        if shard_rank in shard_rank_to_tensor:
            raise RuntimeError(
                f"Duplicate shard_rank={shard_rank} for packed key={key}"
            )
        shard_rank_to_tensor[shard_rank] = shard_tensor

    expected_shard_ranks = set(range(shard_world_size))
    if set(shard_rank_to_tensor) != expected_shard_ranks:
        raise RuntimeError(
            f"Shard rank coverage mismatch for packed key={key}: "
            f"expected {sorted(expected_shard_ranks)}, got {sorted(shard_rank_to_tensor)}"
        )

    manifest = dict(first_manifest)
    manifest["export_shard_dim"] = int(manifest["export_shard_dim"]) + 1
    return _merge_sharded_tensor(
        key,
        ordered_shards=[
            shard_rank_to_tensor[shard_rank] for shard_rank in range(shard_world_size)
        ],
        manifest=manifest,
    )


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
    vllm_tensors, published_config = _rank0_vllm_lora_tensors(
        metadata=metadata,
        tensors_by_owner_key=tensors_by_owner_key,
        packed_expert_metadata=packed_expert_metadata,
        packed_expert_tensors_by_owner_key=packed_expert_tensors_by_owner_key,
        handler=handler,
        adapter_config=adapter_config,
    )
    stager = _PinnedCpuStager()
    published_tensors = _stage_published_tensors(vllm_tensors, stager)
    stager.finish()
    save_vllm_lora_tensors(output_dir, published_tensors, published_config)


def _rank0_vllm_lora_tensors(
    *,
    metadata: list[LoraShardMeta],
    tensors_by_owner_key: dict[tuple[int, str], torch.Tensor],
    packed_expert_metadata: list[PackedExpertShardMeta] | None = None,
    packed_expert_tensors_by_owner_key: (
        dict[tuple[int, str], torch.Tensor] | None
    ) = None,
    handler: Any,
    adapter_config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
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
    return handler.to_vllm_lora_tensors(
        merged_tensors,
        adapter_config=dict(adapter_config),
    )


def build_vllm_lora_tensors_from_model(
    *,
    model: ModelChunks,
    adapter_model: dict[str, torch.Tensor],
    handler: Any,
    adapter_config: dict[str, Any],
    rank: int,
    world_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]] | None:
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
        return None

    return _rank0_vllm_lora_tensors(
        metadata=all_metadata,
        tensors_by_owner_key=exchanged_tensors,
        packed_expert_metadata=all_packed_metadata,
        packed_expert_tensors_by_owner_key=exchanged_packed_tensors,
        handler=handler,
        adapter_config=adapter_config,
    )


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
    result = build_vllm_lora_tensors_from_model(
        model=model,
        adapter_model=adapter_model,
        handler=handler,
        adapter_config=adapter_config,
        rank=rank,
        world_size=world_size,
    )
    if result is None:
        return
    vllm_tensors, published_config = result
    stager = _PinnedCpuStager()
    published_tensors = _stage_published_tensors(vllm_tensors, stager)
    stager.finish()
    save_vllm_lora_tensors(output_dir, published_tensors, published_config)
