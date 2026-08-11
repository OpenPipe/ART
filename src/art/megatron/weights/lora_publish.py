from collections.abc import Iterable, Sequence
import hashlib
from pathlib import Path
from typing import Any, NamedTuple

from safetensors.torch import save_file
import torch

from art.megatron._collective import (
    collective_errors as _collective_errors,
)
from art.megatron._collective import (
    device as _device,
)
from art.megatron._collective import (
    distributed as _distributed_ready,
)
from art.megatron._collective import (
    dtype_from_name as _dtype_from_name,
)
from art.megatron._collective import (
    gather_objects as _gather_objects,
)
from art.megatron._collective import (
    raise_distributed as _raise_rank_errors,
)
from art.megatron._collective import (
    rank as _rank,
)
from art.megatron.lora import (
    LoRA,
    LoraShardManifest,
    LoraShardMeta,
    LoRASlotRef,
    _block_for_key,
    _dtype_name,
)
from art.megatron.model_support.lora_disk import (
    ART_LORA_FORMAT_CONFIG_KEY,
    ART_LORA_FORMAT_MEGATRON,
    ART_LORA_FORMAT_VLLM,
    _consolidate_safetensors,
    save_adapter_config,
)
from art.megatron.model_support.spec import (
    ExpertPackedLoraGroup,
    ExpertPackedLoraSlot,
    ModelSupportHandler,
)
from art.megatron.training.model_chunks import ModelChunks


class PackedExpertShardMeta(NamedTuple):
    key: str
    owner_rank: int
    shape: tuple[int, ...]
    dtype_name: str
    manifest: LoraShardManifest
    expert_start: int
    expert_count: int
    pack_layout: str

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total


def iter_lora_modules(model_chunks: ModelChunks) -> Iterable[LoRA]:
    for chunk in model_chunks:
        for module in chunk.modules():
            if isinstance(module, LoRA):
                yield module


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
    slot_ref: LoRASlotRef | None = None,
) -> bool:
    if module.num_local_experts <= 1:
        return False
    params = tuple(module._lora_params(slot_ref))
    return bool(params) and all(
        _packed_expert_slot(module.adapter_model_prefix, suffix, groups) is not None
        for suffix, _param in params
    )


def collect_local_lora_entries(
    model_chunks: ModelChunks,
    adapter_dtypes: dict[str, torch.dtype],
    *,
    owner_rank: int,
    packed_expert_groups: Sequence[ExpertPackedLoraGroup] = (),
    slot_ref: LoRASlotRef | None = None,
    include_replicas: bool = False,
) -> tuple[dict[str, torch.Tensor], list[LoraShardMeta]]:
    local_tensors: dict[str, torch.Tensor] = {}
    local_manifest: dict[str, LoraShardManifest] = {}
    for module in iter_lora_modules(model_chunks):
        if _uses_packed_expert_publish(module, packed_expert_groups, slot_ref):
            continue
        for key, value in module.sharded_lora_state_dict(
            slot_ref, include_replicas=include_replicas
        ).items():
            target_dtype = adapter_dtypes[key] if key in adapter_dtypes else value.dtype
            local_tensors[key] = value.to(target_dtype)
        local_manifest.update(
            module.sharded_lora_manifest(slot_ref, include_replicas=include_replicas)
        )

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
    adapter_dtypes: dict[str, torch.dtype],
    *,
    owner_rank: int,
    packed_expert_groups: Sequence[ExpertPackedLoraGroup],
    slot_ref: LoRASlotRef | None = None,
) -> tuple[dict[str, torch.Tensor], list[PackedExpertShardMeta]]:
    local_tensors: dict[str, torch.Tensor] = {}
    metadata: list[PackedExpertShardMeta] = []
    for module in iter_lora_modules(model_chunks):
        if not _uses_packed_expert_publish(module, packed_expert_groups, slot_ref):
            continue
        expert_start = int(module._expert_offset)
        expert_count = int(module.num_local_experts)
        for suffix, param in module._lora_params(slot_ref):
            slot_match = _packed_expert_slot(
                module.adapter_model_prefix,
                suffix,
                packed_expert_groups,
            )
            if slot_match is None or not module._should_export_parameter(param):
                continue
            group_prefix, slot = slot_match
            key = f"{group_prefix}.{slot.output_suffix}"
            tensor = param.data.transpose(1, 2)
            source_keys = module._expected_weight_keys(suffix.removesuffix(".weight"))
            target_dtype = (
                adapter_dtypes[source_keys[0]]
                if source_keys and source_keys[0] in adapter_dtypes
                else tensor.dtype
            )
            tensor = tensor.to(target_dtype)
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


def _merge_sharded_tensor(
    key: str,
    *,
    ordered_shards: Sequence[torch.Tensor],
    manifest: LoraShardManifest,
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
    key_entries: Sequence[tuple[LoraShardManifest, torch.Tensor]],
    *,
    manifest: LoraShardManifest | None = None,
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
    entries_by_key: dict[str, list[tuple[LoraShardManifest, torch.Tensor]]],
) -> dict[str, torch.Tensor]:
    return {
        key: _merge_manifest_entries(key, key_entries)
        for key, key_entries in entries_by_key.items()
    }


def _gather_metadata[T](
    local: list[T], *, group: torch.distributed.ProcessGroup | None = None
) -> list[T]:
    return [item for values in _gather_objects(local, group=group) for item in values]


def _tensor_digest(tensor: torch.Tensor) -> str:
    """Hash a tensor with bounded host staging and a bounded collective payload."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(_dtype_name(tensor.dtype).encode())
    digest.update(repr(tuple(tensor.shape)).encode())
    data = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    chunk_bytes = 1024 * 1024
    for offset in range(0, data.numel(), chunk_bytes):
        chunk = data.narrow(0, offset, min(chunk_bytes, data.numel() - offset))
        digest.update(chunk.cpu().numpy().tobytes())
    return digest.hexdigest()


def _contributor_identity(
    metadata: LoraShardMeta | PackedExpertShardMeta,
) -> tuple[object, ...]:
    shard_rank = int(metadata.manifest.get("shard_rank", 0))
    if isinstance(metadata, PackedExpertShardMeta):
        return ("packed", metadata.key, metadata.expert_start, shard_rank)
    return ("lora", metadata.key, shard_rank)


def _validate_replica_digests(
    records: Iterable[tuple[LoraShardMeta | PackedExpertShardMeta, str]],
) -> None:
    digests: dict[tuple[object, ...], str] = {}
    for metadata, digest in records:
        identity = _contributor_identity(metadata)
        previous = digests.setdefault(identity, digest)
        if previous != digest:
            raise RuntimeError(
                f"Inconsistent replicated tensor contents for {metadata.key!r}"
            )


def _elect_lora_contributors(
    metadata: list[LoraShardMeta],
) -> list[LoraShardMeta]:
    elected: dict[tuple[str, int], LoraShardMeta] = {}
    for candidate in metadata:
        identity = (candidate.key, int(candidate.manifest.get("shard_rank", 0)))
        current = elected.get(identity)
        if current is not None and (
            current.shape != candidate.shape
            or current.dtype_name != candidate.dtype_name
            or current.manifest != candidate.manifest
            or current.block != candidate.block
        ):
            raise RuntimeError(
                f"Inconsistent replicated LoRA metadata for {candidate.key!r}"
            )
        if current is None or candidate.owner_rank < current.owner_rank:
            elected[identity] = candidate
    return list(elected.values())


def _elect_packed_expert_contributors(
    metadata: list[PackedExpertShardMeta],
) -> list[PackedExpertShardMeta]:
    elected: dict[tuple[str, int, int], PackedExpertShardMeta] = {}
    for candidate in metadata:
        identity = (
            candidate.key,
            candidate.expert_start,
            int(candidate.manifest.get("shard_rank", 0)),
        )
        current = elected.get(identity)
        if (
            current is not None
            and current._replace(owner_rank=candidate.owner_rank) != candidate
        ):
            raise RuntimeError(
                f"Inconsistent replicated packed LoRA metadata for {candidate.key!r}"
            )
        if current is None or candidate.owner_rank < current.owner_rank:
            elected[identity] = candidate
    return list(elected.values())


def _rank_and_device() -> tuple[int, torch.device]:
    return _rank(), _device()


def _prepare_exchange_buffers(
    metadata: Sequence[LoraShardMeta | PackedExpertShardMeta],
    *,
    local_tensors: dict[str, torch.Tensor],
    rank: int,
    device: torch.device,
) -> tuple[
    dict[tuple[int, str], torch.Tensor],
    dict[tuple[int, str], torch.Tensor],
    dict[tuple[int, str], torch.Tensor],
]:
    sends: dict[tuple[int, str], torch.Tensor] = {}
    receives: dict[tuple[int, str], torch.Tensor] = {}
    received: dict[tuple[int, str], torch.Tensor] = {}
    for meta in metadata:
        identity = (meta.owner_rank, meta.key)
        if rank == meta.owner_rank:
            tensor = local_tensors[meta.key].detach().contiguous()
            if tuple(tensor.shape) != meta.shape:
                raise RuntimeError(
                    f"Tensor {meta.key!r} shape {tuple(tensor.shape)} does not match "
                    f"exchange metadata {meta.shape}"
                )
            dtype_name = _dtype_name(tensor.dtype)
            if dtype_name != meta.dtype_name:
                raise RuntimeError(
                    f"Tensor {meta.key!r} dtype {dtype_name!r} does not match "
                    f"exchange metadata {meta.dtype_name!r}"
                )
            if rank == 0:
                received[identity] = tensor.cpu().contiguous()
            else:
                sends[identity] = tensor
        elif rank == 0:
            receives[identity] = torch.empty(
                meta.shape,
                dtype=_dtype_from_name(meta.dtype_name),
                device=device,
            )
    return sends, receives, received


def _exchange_tensors(
    metadata: Sequence[LoraShardMeta | PackedExpertShardMeta],
    *,
    local_tensors: dict[str, torch.Tensor],
    rank: int,
    device: torch.device,
    group: torch.distributed.ProcessGroup | None = None,
) -> dict[tuple[int, str], torch.Tensor]:
    ordered = sorted(
        metadata,
        key=lambda meta: (
            meta.owner_rank,
            meta.key,
            int(meta.manifest.get("shard_rank", 0)),
        ),
    )
    if not _distributed_ready():
        return {
            (rank, meta.key): local_tensors[meta.key].detach().cpu().contiguous()
            for meta in ordered
        }

    sends: dict[tuple[int, str], torch.Tensor] = {}
    receives: dict[tuple[int, str], torch.Tensor] = {}
    received: dict[tuple[int, str], torch.Tensor] = {}
    error: BaseException | None = None
    try:
        sends, receives, received = _prepare_exchange_buffers(
            ordered,
            local_tensors=local_tensors,
            rank=rank,
            device=device,
        )
    except BaseException as exc:
        error = exc
    _raise_rank_errors(error, "prepare tensor exchange", group=group)

    for meta in ordered:
        identity = (meta.owner_rank, meta.key)
        if rank == meta.owner_rank and rank != 0:
            torch.distributed.send(sends[identity], dst=0, group=group)  # type: ignore[possibly-missing-attribute]
        elif rank == 0 and meta.owner_rank != 0:
            torch.distributed.recv(  # type: ignore[possibly-missing-attribute]
                receives[identity], src=meta.owner_rank, group=group
            )

    error = None
    try:
        if rank == 0:
            received.update(
                (identity, tensor.cpu().contiguous())
                for identity, tensor in receives.items()
            )
    except BaseException as exc:
        error = exc
    _raise_rank_errors(error, "finalize tensor exchange", group=group)
    return received


def _entries_by_key(
    metadata: list[LoraShardMeta],
    tensors_by_owner_key: dict[tuple[int, str], torch.Tensor],
) -> dict[str, list[tuple[LoraShardManifest, torch.Tensor]]]:
    entries: dict[str, list[tuple[LoraShardManifest, torch.Tensor]]] = {}
    for meta in metadata:
        entries.setdefault(meta.key, []).append(
            (meta.manifest, tensors_by_owner_key[(meta.owner_rank, meta.key)])
        )
    return entries


def _gather_merged_adapter_tensors(
    metadata: Sequence[LoraShardMeta],
    *,
    local_tensors: dict[str, torch.Tensor],
    rank: int,
    device: torch.device,
    group: torch.distributed.ProcessGroup | None = None,
) -> dict[str, torch.Tensor]:
    exchanged = _exchange_tensors(
        metadata,
        local_tensors=local_tensors,
        rank=rank,
        device=device,
        group=group,
    )
    merged: dict[str, torch.Tensor] = {}
    error: BaseException | None = None
    try:
        if rank == 0:
            merged = merge_sharded_adapter_entries(
                _entries_by_key(list(metadata), exchanged)
            )
    except BaseException as exc:
        error = exc
    _raise_rank_errors(error, "merge adapter tensors", group=group)
    return merged


def _merge_packed_expert_block(
    key: str,
    key_entries: list[tuple[LoraShardManifest, torch.Tensor]],
) -> torch.Tensor:
    manifest: LoraShardManifest = {**key_entries[0][0]}
    if manifest["sharded"]:
        manifest["export_shard_dim"] = manifest.get("export_shard_dim", 0) + 1
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
    if first_layout == "interleaved_gate_up_rank_major_expert_cols":
        if joined.ndim != 3:
            raise RuntimeError(
                f"{key}: interleaved_gate_up_rank_major_expert_cols layout "
                "requires 3D blocks"
            )
        if joined.shape[1] % 2 != 0:
            raise RuntimeError(
                f"{key}: interleaved gate/up layout requires an even output dim"
            )
        gate, up = joined.split(joined.shape[1] // 2, dim=1)
        interleaved = torch.stack((gate, up), dim=2).flatten(1, 2)
        return (
            interleaved.permute(1, 2, 0)
            .reshape(
                interleaved.shape[1],
                interleaved.shape[2] * interleaved.shape[0],
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
        list[tuple[PackedExpertShardMeta, LoraShardManifest, torch.Tensor]],
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


def _rank0_vllm_lora_tensors(
    *,
    metadata: list[LoraShardMeta],
    tensors_by_owner_key: dict[tuple[int, str], torch.Tensor],
    packed_expert_metadata: list[PackedExpertShardMeta] | None = None,
    packed_expert_tensors_by_owner_key: (
        dict[tuple[int, str], torch.Tensor] | None
    ) = None,
    handler: ModelSupportHandler,
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


class _LocalLoraExport(NamedTuple):
    rank: int
    device: torch.device
    tensors: dict[str, torch.Tensor]
    metadata: list[LoraShardMeta]
    packed_tensors: dict[str, torch.Tensor]
    packed_metadata: list[PackedExpertShardMeta]
    blocks: tuple[str, ...]


def _prepare_local_lora_export(
    *,
    model: ModelChunks,
    adapter_dtypes: dict[str, torch.dtype],
    handler: ModelSupportHandler,
    rank: int,
    world_size: int,
    slot_ref: LoRASlotRef | None = None,
    native_format: bool = False,
) -> _LocalLoraExport:
    actual_rank, device = _rank_and_device()
    local_tensors: dict[str, torch.Tensor] = {}
    local_metadata: list[LoraShardMeta] = []
    local_digests: dict[str, str] = {}
    local_packed_tensors: dict[str, torch.Tensor] = {}
    local_packed_metadata: list[PackedExpertShardMeta] = []
    local_packed_digests: dict[str, str] = {}
    with _collective_errors("prepare LoRA export"):
        if _distributed_ready():
            actual_world_size = torch.distributed.get_world_size()  # type: ignore[possibly-missing-attribute]
            if actual_rank != rank or actual_world_size != world_size:
                raise RuntimeError(
                    "LoRA publisher rank/world-size mismatch: "
                    f"runtime=({rank}, {world_size}) "
                    f"distributed=({actual_rank}, {actual_world_size})"
                )
        elif rank != 0 or world_size != 1:
            raise RuntimeError(
                "Non-distributed LoRA publish requires rank=0 and world_size=1, "
                f"got rank={rank} world_size={world_size}"
            )
        packed_expert_groups = (
            () if native_format else tuple(handler.expert_packed_lora_groups())
        )
        local_tensors, local_metadata = collect_local_lora_entries(
            model,
            adapter_dtypes,
            owner_rank=rank,
            packed_expert_groups=packed_expert_groups,
            slot_ref=slot_ref,
        )
        (local_packed_tensors, local_packed_metadata) = (
            collect_local_packed_expert_entries(
                model,
                adapter_dtypes,
                owner_rank=rank,
                packed_expert_groups=packed_expert_groups,
                slot_ref=slot_ref,
            )
        )
        local_digests = {
            key: _tensor_digest(tensor) for key, tensor in local_tensors.items()
        }
        local_packed_digests = {
            key: _tensor_digest(tensor) for key, tensor in local_packed_tensors.items()
        }
    gathered_metadata = _gather_metadata(
        [(metadata, local_digests[metadata.key]) for metadata in local_metadata]
    )
    gathered_packed_metadata = _gather_metadata(
        [
            (metadata, local_packed_digests[metadata.key])
            for metadata in local_packed_metadata
        ]
    )
    _validate_replica_digests(gathered_metadata)
    _validate_replica_digests(gathered_packed_metadata)
    metadata = _elect_lora_contributors(
        [metadata for metadata, _digest in gathered_metadata]
    )
    packed_metadata = _elect_packed_expert_contributors(
        [metadata for metadata, _digest in gathered_packed_metadata]
    )
    blocks = tuple(
        sorted(
            {item.block for item in metadata}
            | {_block_for_key(item.key) for item in packed_metadata}
        )
    )
    return _LocalLoraExport(
        rank,
        device,
        local_tensors,
        metadata,
        local_packed_tensors,
        packed_metadata,
        blocks,
    )


def build_vllm_lora_tensors_from_model(
    *,
    model: ModelChunks,
    adapter_dtypes: dict[str, torch.dtype],
    handler: ModelSupportHandler,
    adapter_config: dict[str, Any],
    rank: int,
    world_size: int,
    slot_ref: LoRASlotRef | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]] | None:
    local = _prepare_local_lora_export(
        model=model,
        adapter_dtypes=adapter_dtypes,
        handler=handler,
        rank=rank,
        world_size=world_size,
        slot_ref=slot_ref,
    )
    exchanged_tensors = _exchange_tensors(
        local.metadata,
        local_tensors=local.tensors,
        rank=rank,
        device=local.device,
    )
    exchanged_packed_tensors = _exchange_tensors(
        local.packed_metadata,
        local_tensors=local.packed_tensors,
        rank=rank,
        device=local.device,
    )

    if rank != 0:
        return None

    return _rank0_vllm_lora_tensors(
        metadata=local.metadata,
        tensors_by_owner_key=exchanged_tensors,
        packed_expert_metadata=local.packed_metadata,
        packed_expert_tensors_by_owner_key=exchanged_packed_tensors,
        handler=handler,
        adapter_config=adapter_config,
    )


def save_vllm_lora_from_model(
    *,
    model: ModelChunks,
    adapter_dtypes: dict[str, torch.dtype],
    handler: ModelSupportHandler,
    adapter_config: dict[str, Any],
    output_dir: str,
    rank: int,
    world_size: int,
    slot_ref: LoRASlotRef | None = None,
    native_format: bool = False,
) -> _LocalLoraExport:
    local = _prepare_local_lora_export(
        model=model,
        adapter_dtypes=adapter_dtypes,
        handler=handler,
        rank=rank,
        world_size=world_size,
        slot_ref=slot_ref,
        native_format=native_format,
    )
    blocks = local.blocks
    root = Path(output_dir)
    shards: list[Path] = []
    published_config = dict(adapter_config)
    written: set[str] = set()
    with _collective_errors("prepare LoRA output"):
        if rank == 0:
            root.mkdir(parents=True, exist_ok=True)

    try:
        for index, block in enumerate(blocks):
            metadata = [meta for meta in local.metadata if meta.block == block]
            packed_metadata = [
                meta
                for meta in local.packed_metadata
                if _block_for_key(meta.key) == block
            ]
            exchanged = _exchange_tensors(
                metadata,
                local_tensors=local.tensors,
                rank=rank,
                device=local.device,
            )
            exchanged_packed = _exchange_tensors(
                packed_metadata,
                local_tensors=local.packed_tensors,
                rank=rank,
                device=local.device,
            )
            with _collective_errors(f"serialize LoRA block {block}"):
                if rank == 0:
                    if native_format:
                        tensors = merge_sharded_adapter_entries(
                            _entries_by_key(metadata, exchanged)
                        )
                        block_config = adapter_config
                    else:
                        tensors, block_config = _rank0_vllm_lora_tensors(
                            metadata=metadata,
                            tensors_by_owner_key=exchanged,
                            packed_expert_metadata=packed_metadata,
                            packed_expert_tensors_by_owner_key=exchanged_packed,
                            handler=handler,
                            adapter_config=adapter_config,
                        )
                    if duplicates := written & tensors.keys():
                        raise RuntimeError(
                            "Duplicate LoRA tensors across model blocks: "
                            f"{sorted(duplicates)}"
                        )
                    written.update(tensors)
                    if block_config != adapter_config:
                        if (
                            published_config != adapter_config
                            and published_config != block_config
                        ):
                            raise RuntimeError(
                                "Model blocks produced inconsistent LoRA configs"
                            )
                        published_config = block_config
                    shard = root / f".adapter_model-{index:06d}.safetensors"
                    shards.append(shard)
                    save_file(
                        {
                            key: tensor.detach().cpu().contiguous()
                            for key, tensor in tensors.items()
                        },
                        shard,
                    )
        with _collective_errors("finalize LoRA export"):
            if rank == 0:
                if not shards:
                    raise RuntimeError("No LoRA tensors were available to export")
                _consolidate_safetensors(shards, root / "adapter_model.safetensors")
                save_adapter_config(
                    root,
                    {
                        **published_config,
                        ART_LORA_FORMAT_CONFIG_KEY: (
                            ART_LORA_FORMAT_MEGATRON
                            if native_format
                            else ART_LORA_FORMAT_VLLM
                        ),
                    },
                )
    finally:
        if rank == 0:
            for shard in shards:
                shard.unlink(missing_ok=True)
    return local
