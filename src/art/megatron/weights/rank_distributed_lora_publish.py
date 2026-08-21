from collections import defaultdict
from collections.abc import Sequence
import json
import math
import struct
import sys
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from art.distributed.object_store import (
    BinaryObjectFile,
    OrderedBinaryObjectPlan,
    OrderedBinaryObjectRef,
    OrderedBinaryObjectShard,
    OrderedBinaryObjectTarget,
    S3BinaryObjectStore,
    StoredOrderedBinaryObjectShard,
    ordered_binary_object_ref_from_plan,
)
from art.megatron.lora import LoraShardMeta, _block_for_key
from art.megatron.model_support.lora_disk import (
    ART_LORA_FORMAT_CONFIG_KEY,
    ART_LORA_FORMAT_VLLM,
    encode_adapter_config,
)
from art.megatron.tensor_snapshot import PendingCpuSnapshot, PinnedCpuSnapshotStager
from art.megatron.weights.lora_publish import (
    PackedExpertShardMeta,
    _stage_published_tensors,
    merge_packed_expert_adapter_entries,
    merge_sharded_adapter_entries,
)

_SAFETENSORS_DTYPES = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.complex64: "C64",
    **{
        dtype: name
        for name, dtype in (
            ("U16", getattr(torch, "uint16", None)),
            ("U32", getattr(torch, "uint32", None)),
            ("U64", getattr(torch, "uint64", None)),
            ("F8_E4M3", getattr(torch, "float8_e4m3fn", None)),
            ("F8_E5M2", getattr(torch, "float8_e5m2", None)),
        )
        if dtype is not None
    },
}


class RankOwnedTensor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    owner_rank: int = Field(ge=0)
    shape: tuple[int, ...]
    dtype_name: str = Field(min_length=1)
    byte_count: int = Field(ge=1)


class RankOwnedSourceSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["adapter_config", "safetensors_header", "tensor"]
    tensor_name: str | None = None
    source_offset: int = Field(ge=0)
    byte_count: int = Field(ge=1)

    @model_validator(mode="after")
    def _tensor_identity(self) -> "RankOwnedSourceSegment":
        if (self.source == "tensor") != (self.tensor_name is not None):
            raise ValueError("only tensor source segments identify a tensor")
        return self


class RankOwnedOrderedShard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    owner_rank: int = Field(ge=0)
    segments: tuple[RankOwnedSourceSegment, ...] = Field(min_length=1)


class RankDistributedLoraLayout(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    coordinator_rank: int = Field(ge=0)
    world_size: int = Field(ge=1)
    target: OrderedBinaryObjectTarget
    tensors: tuple[RankOwnedTensor, ...] = Field(min_length=1)
    plan: OrderedBinaryObjectPlan
    ref: OrderedBinaryObjectRef
    shards: tuple[RankOwnedOrderedShard, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _complete_ownership(self) -> "RankDistributedLoraLayout":
        if self.coordinator_rank >= self.world_size:
            raise ValueError("LoRA publication coordinator leaves its world")
        if any(tensor.owner_rank >= self.world_size for tensor in self.tensors):
            raise ValueError("LoRA tensor owner leaves its world")
        if tuple(tensor.name for tensor in self.tensors) != tuple(
            sorted(tensor.name for tensor in self.tensors)
        ):
            raise ValueError("LoRA tensors must use canonical name order")
        if len({tensor.name for tensor in self.tensors}) != len(self.tensors):
            raise ValueError("LoRA tensor ownership must be unique")
        if tuple(shard.index for shard in self.shards) != tuple(
            range(len(self.shards))
        ):
            raise ValueError("LoRA shard ownership must cover every shard once")
        if tuple(shard.index for shard in self.plan.shards) != tuple(
            shard.index for shard in self.shards
        ):
            raise ValueError("LoRA shard ownership differs from the ordered plan")
        for planned, owned in zip(self.plan.shards, self.shards, strict=True):
            if owned.owner_rank >= self.world_size:
                raise ValueError("LoRA shard owner leaves its world")
            if (
                sum(segment.byte_count for segment in owned.segments)
                != planned.byte_count
            ):
                raise ValueError("LoRA shard source does not cover its planned range")
        if self.ref.files != self.plan.files or self.ref.shard_count != len(
            self.shards
        ):
            raise ValueError("LoRA ordered reference differs from its plan")
        if self.ref != ordered_binary_object_ref_from_plan(self.target, self.plan):
            raise ValueError("LoRA ordered reference differs from its target")
        return self


class RankDistributedLoraStats(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=0)
    world_size: int = Field(ge=1)
    source_bytes: int = Field(ge=0)
    sent_bytes: int = Field(ge=0)
    received_bytes: int = Field(ge=0)
    owned_tensor_bytes: int = Field(ge=0)
    peak_accounted_owner_bytes: int = Field(ge=0)
    owned_upload_bytes: int = Field(ge=0)
    owned_tensor_count: int = Field(ge=0)
    owned_block_count: int = Field(ge=0)


class PreparedRankDistributedLora(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    layout: RankDistributedLoraLayout
    rank: int = Field(ge=0)
    tensors: dict[str, torch.Tensor]
    adapter_config: bytes
    safetensors_header: bytes
    stats: RankDistributedLoraStats

    def shard_payloads(self) -> dict[int, tuple[memoryview, ...]]:
        sources = {
            "adapter_config": memoryview(self.adapter_config),
            "safetensors_header": memoryview(self.safetensors_header),
        }
        tensor_sources = {
            name: memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast("B")
            for name, tensor in self.tensors.items()
        }
        payloads: dict[int, tuple[memoryview, ...]] = {}
        for shard in self.layout.shards:
            if shard.owner_rank != self.rank:
                continue
            chunks: list[memoryview] = []
            for segment in shard.segments:
                source = (
                    tensor_sources[segment.tensor_name]
                    if segment.tensor_name is not None
                    else sources[segment.source]
                )
                chunks.append(
                    source[
                        segment.source_offset : segment.source_offset
                        + segment.byte_count
                    ]
                )
            payloads[shard.index] = tuple(chunks)
        return payloads


def _rank_world(group: Any | None) -> tuple[int, int]:
    if not torch.distributed.is_initialized():  # type: ignore[possibly-missing-attribute]
        return 0, 1
    return (
        int(torch.distributed.get_rank(group)),  # type: ignore[possibly-missing-attribute]
        int(torch.distributed.get_world_size(group)),  # type: ignore[possibly-missing-attribute]
    )


def _normalize_local_owners(
    metadata: Sequence[LoraShardMeta | PackedExpertShardMeta],
    *,
    rank: int,
    group: Any | None,
) -> list[LoraShardMeta | PackedExpertShardMeta]:
    if not torch.distributed.is_initialized():  # type: ignore[possibly-missing-attribute]
        global_rank = 0
    else:
        global_rank = int(torch.distributed.get_rank())  # type: ignore[possibly-missing-attribute]
        if group is not None:
            group_ranks = tuple(
                int(value)
                for value in torch.distributed.get_process_group_ranks(group)  # type: ignore[possibly-missing-attribute]
            )
            if group_ranks[rank] != global_rank:
                raise RuntimeError("LoRA process-group rank mapping is inconsistent")
    if any(meta.owner_rank != global_rank for meta in metadata):
        raise ValueError("local LoRA metadata must use its existing global owner rank")
    return [meta._replace(owner_rank=rank) for meta in metadata]


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"unsupported LoRA dtype {name!r}")
    return dtype


def _metadata_block(meta: LoraShardMeta | PackedExpertShardMeta) -> str:
    return meta.block if isinstance(meta, LoraShardMeta) else _block_for_key(meta.key)


def _metadata_identity(
    meta: LoraShardMeta | PackedExpertShardMeta,
) -> tuple[str, str, int, int]:
    return (
        "packed" if isinstance(meta, PackedExpertShardMeta) else "regular",
        meta.key,
        int(meta.manifest.get("shard_rank", 0)),
        int(getattr(meta, "expert_start", -1)),
    )


def _metadata_bytes(meta: LoraShardMeta | PackedExpertShardMeta) -> int:
    return (
        int(meta.numel)
        * torch.empty((), dtype=_dtype_from_name(meta.dtype_name)).element_size()
    )


def _metadata_without_owner(
    meta: LoraShardMeta | PackedExpertShardMeta,
) -> LoraShardMeta | PackedExpertShardMeta:
    return meta._replace(owner_rank=0)


def _gather_candidates(
    local: Sequence[LoraShardMeta | PackedExpertShardMeta],
    *,
    rank: int,
    world_size: int,
    group: Any | None,
) -> dict[
    tuple[str, str, int, int],
    tuple[LoraShardMeta | PackedExpertShardMeta, ...],
]:
    for meta in local:
        if meta.owner_rank != rank:
            raise ValueError("local LoRA metadata identifies another source rank")
    gathered: list[list[LoraShardMeta | PackedExpertShardMeta] | None] = [
        None
    ] * world_size
    if world_size == 1:
        gathered[0] = list(local)
    else:
        torch.distributed.all_gather_object(gathered, list(local), group=group)  # type: ignore[possibly-missing-attribute]
    candidates: dict[
        tuple[str, str, int, int], list[LoraShardMeta | PackedExpertShardMeta]
    ] = defaultdict(list)
    for rank_entries in gathered:
        if rank_entries is None:
            raise RuntimeError("LoRA metadata gather returned a missing rank")
        for meta in rank_entries:
            candidates[_metadata_identity(meta)].append(meta)
    for identity, entries in candidates.items():
        if len({entry.owner_rank for entry in entries}) != len(entries):
            raise RuntimeError(f"duplicate LoRA source metadata for {identity}")
        expected = _metadata_without_owner(entries[0])
        if any(_metadata_without_owner(entry) != expected for entry in entries[1:]):
            raise RuntimeError(f"inconsistent replicated LoRA metadata for {identity}")
    return {identity: tuple(entries) for identity, entries in candidates.items()}


def _assign_blocks(
    candidates: dict[
        tuple[str, str, int, int],
        tuple[LoraShardMeta | PackedExpertShardMeta, ...],
    ],
    world_size: int,
) -> dict[str, int]:
    costs: dict[str, int] = defaultdict(int)
    local_bytes: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for entries in candidates.values():
        meta = entries[0]
        block = _metadata_block(meta)
        byte_count = _metadata_bytes(meta)
        costs[block] += byte_count
        for candidate in entries:
            local_bytes[block][candidate.owner_rank] += byte_count
    loads = [0] * world_size
    owners: dict[str, int] = {}
    for block, cost in sorted(costs.items(), key=lambda item: (-item[1], item[0])):
        minimum = min(loads)
        eligible = [rank for rank, load in enumerate(loads) if load == minimum]
        owner = min(eligible, key=lambda rank: (-local_bytes[block][rank], rank))
        owners[block] = owner
        loads[owner] += cost
    return owners


def _canonical_sources(
    candidates: dict[
        tuple[str, str, int, int],
        tuple[LoraShardMeta | PackedExpertShardMeta, ...],
    ],
    block_owners: dict[str, int],
) -> tuple[list[LoraShardMeta], list[PackedExpertShardMeta]]:
    regular: list[LoraShardMeta] = []
    packed: list[PackedExpertShardMeta] = []
    for identity in sorted(candidates):
        entries = candidates[identity]
        destination = block_owners[_metadata_block(entries[0])]
        selected = next(
            (entry for entry in entries if entry.owner_rank == destination),
            min(entries, key=lambda entry: entry.owner_rank),
        )
        (packed if isinstance(selected, PackedExpertShardMeta) else regular).append(
            selected
        )
    return regular, packed


def _validate_local_tensors(
    tensors: dict[str, torch.Tensor],
    metadata: Sequence[LoraShardMeta | PackedExpertShardMeta],
    device: torch.device,
) -> None:
    if set(tensors) != {meta.key for meta in metadata}:
        raise ValueError("local LoRA tensors and metadata differ")
    for meta in metadata:
        if isinstance(meta, LoraShardMeta) and meta.block != _block_for_key(meta.key):
            raise ValueError(f"LoRA conversion block differs from its key: {meta.key}")
        tensor = tensors[meta.key]
        if (
            tensor.device != device
            or not tensor.is_contiguous()
            or tuple(tensor.shape) != meta.shape
            or tensor.dtype != _dtype_from_name(meta.dtype_name)
        ):
            raise ValueError(f"local LoRA tensor differs from metadata: {meta.key}")


def _exchange_to_block_owners(
    metadata: Sequence[LoraShardMeta | PackedExpertShardMeta],
    local_tensors: dict[str, torch.Tensor],
    block_owners: dict[str, int],
    *,
    rank: int,
    world_size: int,
    group: Any | None,
    device: torch.device,
) -> tuple[dict[tuple[int, str], torch.Tensor], int, int]:
    received: dict[tuple[int, str], torch.Tensor] = {}
    sent_bytes = 0
    received_bytes = 0
    sort_key = lambda meta: (
        _metadata_block(meta),
        meta.key,
        int(getattr(meta, "expert_start", -1)),
        int(meta.manifest.get("shard_rank", 0)),
    )
    for dtype_name in sorted({meta.dtype_name for meta in metadata}):
        dtype = _dtype_from_name(dtype_name)
        typed = [meta for meta in metadata if meta.dtype_name == dtype_name]
        remote = any(
            meta.owner_rank != block_owners[_metadata_block(meta)] for meta in typed
        )
        for meta in typed:
            destination = block_owners[_metadata_block(meta)]
            if meta.owner_rank == rank == destination:
                received[(rank, meta.key)] = local_tensors[meta.key]
        if not remote:
            continue
        sends = [
            sorted(
                [
                    meta
                    for meta in typed
                    if meta.owner_rank == rank
                    and block_owners[_metadata_block(meta)] == destination
                    and destination != rank
                ],
                key=sort_key,
            )
            for destination in range(world_size)
        ]
        receives = [
            sorted(
                [
                    meta
                    for meta in typed
                    if meta.owner_rank == source
                    and block_owners[_metadata_block(meta)] == rank
                    and source != rank
                ],
                key=sort_key,
            )
            for source in range(world_size)
        ]
        input_splits = [sum(meta.numel for meta in entries) for entries in sends]
        output_splits = [sum(meta.numel for meta in entries) for entries in receives]
        send_parts = [
            local_tensors[meta.key].reshape(-1) for entries in sends for meta in entries
        ]
        send = (
            torch.cat(send_parts)
            if len(send_parts) > 1
            else send_parts[0]
            if send_parts
            else torch.empty(0, dtype=dtype, device=device)
        )
        output = torch.empty(sum(output_splits), dtype=dtype, device=device)
        torch.distributed.all_to_all_single(  # type: ignore[possibly-missing-attribute]
            output,
            send,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
        )
        sent_bytes += send.nbytes
        received_bytes += output.nbytes
        offset = 0
        for entries in receives:
            for meta in entries:
                key = (meta.owner_rank, meta.key)
                if key in received:
                    raise RuntimeError(f"duplicate exchanged LoRA tensor {key}")
                received[key] = output.narrow(0, offset, meta.numel).view(meta.shape)
                offset += meta.numel
        if offset != output.numel():
            raise RuntimeError("LoRA exchange did not consume its receive buffer")
    return received, sent_bytes, received_bytes


def _merge_owned_blocks(
    regular: Sequence[LoraShardMeta],
    packed: Sequence[PackedExpertShardMeta],
    regular_tensors: dict[tuple[int, str], torch.Tensor],
    packed_tensors: dict[tuple[int, str], torch.Tensor],
    block_owners: dict[str, int],
    *,
    rank: int,
    handler: Any,
    adapter_config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int, int]:
    output: dict[str, torch.Tensor] = {}
    configs: list[dict[str, Any]] = []
    owned_blocks = sorted(
        block for block, owner in block_owners.items() if owner == rank
    )
    retained_input_bytes = _unique_storage_bytes(
        (*regular_tensors.values(), *packed_tensors.values())
    )
    peak_accounted_bytes = retained_input_bytes
    for block in owned_blocks:
        block_regular = [meta for meta in regular if _metadata_block(meta) == block]
        entries: dict[str, list[tuple[dict[str, Any], torch.Tensor]]] = defaultdict(
            list
        )
        for meta in block_regular:
            entries[meta.key].append(
                (meta.manifest, regular_tensors[(meta.owner_rank, meta.key)])
            )
        merged = merge_sharded_adapter_entries(dict(entries))
        block_packed = [meta for meta in packed if _metadata_block(meta) == block]
        if block_packed:
            merged_packed = merge_packed_expert_adapter_entries(
                block_packed,
                packed_tensors,
            )
            overlap = set(merged).intersection(merged_packed)
            if overlap:
                raise RuntimeError(f"duplicate packed LoRA tensors: {sorted(overlap)}")
            merged.update(merged_packed)
        converted, published_config = handler.to_vllm_lora_tensors(
            merged,
            adapter_config=dict(adapter_config),
        )
        overlap = set(output).intersection(converted)
        if overlap:
            raise RuntimeError(f"duplicate converted LoRA tensors: {sorted(overlap)}")
        for name, tensor in converted.items():
            if not tensor.is_contiguous():
                raise RuntimeError(f"converted LoRA tensor is not contiguous: {name}")
        output.update(converted)
        peak_accounted_bytes = max(
            peak_accounted_bytes,
            retained_input_bytes
            + _unique_storage_bytes((*merged.values(), *output.values())),
        )
        configs.append(published_config)
    return output, configs, len(owned_blocks), peak_accounted_bytes


def _unique_storage_bytes(tensors: Sequence[torch.Tensor]) -> int:
    storages: dict[tuple[str, int | None, int], int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[(tensor.device.type, tensor.device.index, storage.data_ptr())] = (
            storage.nbytes()
        )
    return sum(storages.values())


def _published_config(
    local_configs: list[dict[str, Any]],
    *,
    handler: Any,
    adapter_config: dict[str, Any],
    world_size: int,
    group: Any | None,
) -> dict[str, Any]:
    gathered: list[list[dict[str, Any]] | None] = [None] * world_size
    if world_size == 1:
        gathered[0] = local_configs
    else:
        torch.distributed.all_gather_object(gathered, local_configs, group=group)  # type: ignore[possibly-missing-attribute]
    unique: dict[bytes, dict[str, Any]] = {}
    for values in gathered:
        if values is None:
            raise RuntimeError("LoRA config gather returned a missing rank")
        for value in values:
            unique.setdefault(encode_adapter_config(value), value)
    baseline = encode_adapter_config(adapter_config)
    declared = handler.to_vllm_lora_config(dict(adapter_config))
    declared_bytes = encode_adapter_config(declared)
    changed = {key: value for key, value in unique.items() if key != baseline}
    if declared_bytes != baseline:
        unexpected = set(changed).difference({declared_bytes})
        if unexpected:
            raise RuntimeError("handler blocks produced inconsistent LoRA configs")
        selected = declared
    elif len(changed) == 1:
        selected = next(iter(changed.values()))
    elif changed:
        raise RuntimeError("handler blocks produced multiple LoRA configs")
    else:
        selected = adapter_config
    return {**selected, ART_LORA_FORMAT_CONFIG_KEY: ART_LORA_FORMAT_VLLM}


def _gather_output_metadata(
    tensors: dict[str, torch.Tensor],
    *,
    rank: int,
    world_size: int,
    group: Any | None,
) -> tuple[RankOwnedTensor, ...]:
    local = [
        RankOwnedTensor(
            name=name,
            owner_rank=rank,
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype_name=str(tensor.dtype).removeprefix("torch."),
            byte_count=tensor.nbytes,
        )
        for name, tensor in sorted(tensors.items())
    ]
    gathered: list[list[RankOwnedTensor] | None] = [None] * world_size
    if world_size == 1:
        gathered[0] = local
    else:
        torch.distributed.all_gather_object(gathered, local, group=group)  # type: ignore[possibly-missing-attribute]
    if any(values is None for values in gathered):
        raise RuntimeError("LoRA output metadata gather returned a missing rank")
    result = sorted(
        (tensor for values in gathered for tensor in values or ()),
        key=lambda tensor: tensor.name,
    )
    if len({tensor.name for tensor in result}) != len(result):
        raise RuntimeError("LoRA output tensor names are not globally unique")
    return tuple(result)


def _safetensors_header(tensors: Sequence[RankOwnedTensor]) -> bytes:
    if sys.byteorder != "little":
        raise RuntimeError("ART's ordered safetensors publisher requires little endian")
    offset = 0
    header: dict[str, dict[str, Any]] = {}
    for tensor in tensors:
        dtype = _dtype_from_name(tensor.dtype_name)
        safetensors_dtype = _SAFETENSORS_DTYPES.get(dtype)
        if safetensors_dtype is None:
            raise RuntimeError(f"unsupported safetensors dtype: {dtype}")
        if (
            tensor.byte_count
            != math.prod(tensor.shape) * torch.empty((), dtype=dtype).element_size()
        ):
            raise RuntimeError(f"LoRA output tensor byte count changed: {tensor.name}")
        header[tensor.name] = {
            "dtype": safetensors_dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.byte_count],
        }
        offset += tensor.byte_count
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded


def _build_layout(
    target: OrderedBinaryObjectTarget,
    tensors: tuple[RankOwnedTensor, ...],
    adapter_config: bytes,
    safetensors_header: bytes,
    *,
    coordinator_rank: int,
    world_size: int,
) -> RankDistributedLoraLayout:
    source_regions = {
        "adapter_config.json": [
            (
                coordinator_rank,
                RankOwnedSourceSegment(
                    source="adapter_config",
                    source_offset=0,
                    byte_count=len(adapter_config),
                ),
            )
        ],
        "adapter_model.safetensors": [
            (
                coordinator_rank,
                RankOwnedSourceSegment(
                    source="safetensors_header",
                    source_offset=0,
                    byte_count=len(safetensors_header),
                ),
            ),
            *(
                (
                    tensor.owner_rank,
                    RankOwnedSourceSegment(
                        source="tensor",
                        tensor_name=tensor.name,
                        source_offset=0,
                        byte_count=tensor.byte_count,
                    ),
                )
                for tensor in tensors
            ),
        ],
    }
    files = tuple(
        BinaryObjectFile(
            relative_path=path,
            byte_count=sum(segment.byte_count for _owner, segment in regions),
            sha256=None,
        )
        for path, regions in sorted(source_regions.items())
    )
    planned_shards: list[OrderedBinaryObjectShard] = []
    owned_shards: list[RankOwnedOrderedShard] = []
    for path, regions in sorted(source_regions.items()):
        file_offset = 0
        shard_owner: int | None = None
        shard_offset = 0
        shard_segments: list[RankOwnedSourceSegment] = []
        shard_bytes = 0

        def flush() -> None:
            nonlocal shard_owner, shard_offset, shard_segments, shard_bytes
            if shard_owner is None:
                return
            index = len(planned_shards)
            planned_shards.append(
                OrderedBinaryObjectShard(
                    index=index,
                    relative_path=path,
                    file_offset=shard_offset,
                    byte_count=shard_bytes,
                )
            )
            owned_shards.append(
                RankOwnedOrderedShard(
                    index=index,
                    owner_rank=shard_owner,
                    segments=tuple(shard_segments),
                )
            )
            shard_owner = None
            shard_segments = []
            shard_bytes = 0

        for owner, region in regions:
            consumed = 0
            while consumed < region.byte_count:
                if shard_owner is not None and (
                    shard_owner != owner or shard_bytes == target.shard_bytes
                ):
                    flush()
                if shard_owner is None:
                    shard_owner = owner
                    shard_offset = file_offset
                count = min(
                    region.byte_count - consumed,
                    target.shard_bytes - shard_bytes,
                )
                shard_segments.append(
                    region.model_copy(
                        update={
                            "source_offset": region.source_offset + consumed,
                            "byte_count": count,
                        }
                    )
                )
                shard_bytes += count
                consumed += count
                file_offset += count
        flush()
    plan = OrderedBinaryObjectPlan(
        object_id=target.object_id,
        format=target.format,
        files=files,
        shards=tuple(planned_shards),
        metadata=target.metadata,
    )
    ref = ordered_binary_object_ref_from_plan(target, plan)
    return RankDistributedLoraLayout(
        coordinator_rank=coordinator_rank,
        world_size=world_size,
        target=target,
        tensors=tensors,
        plan=plan,
        ref=ref,
        shards=tuple(owned_shards),
    )


def prepare_rank_distributed_vllm_lora(
    *,
    target: OrderedBinaryObjectTarget,
    local_tensors: dict[str, torch.Tensor],
    local_metadata: Sequence[LoraShardMeta],
    local_packed_tensors: dict[str, torch.Tensor],
    local_packed_metadata: Sequence[PackedExpertShardMeta],
    handler: Any,
    adapter_config: dict[str, Any],
    group: Any | None = None,
    coordinator_rank: int = 0,
    exchange_device: torch.device,
    stager: PinnedCpuSnapshotStager,
) -> PendingCpuSnapshot[PreparedRankDistributedLora]:
    """Gather complete layer conversion groups and prepare rank-owned L2 ranges.

    The supplied tensors retain current global-rank metadata and must remain immutable
    through owner exchange and conversion. A NCCL caller may invoke this on a side
    stream after its snapshot-ready event; it must register the returned pending
    snapshot before allowing the optimizer to mutate the source weights.
    """
    rank, world_size = _rank_world(group)
    exchange_device = torch.device(exchange_device)
    if exchange_device.type == "cuda" and exchange_device.index is None:
        exchange_device = torch.device("cuda", torch.cuda.current_device())
    if coordinator_rank >= world_size:
        raise ValueError("LoRA publication coordinator leaves its world")
    if world_size > 1:
        backend = str(torch.distributed.get_backend(group))  # type: ignore[possibly-missing-attribute]
        expected_backend = "gloo" if exchange_device.type == "cpu" else "nccl"
        if exchange_device.type not in {"cpu", "cuda"} or backend != expected_backend:
            raise RuntimeError(
                "LoRA exchange device must match its distributed backend: "
                f"device={exchange_device.type} backend={backend}"
            )
    local_metadata = _normalize_local_owners(
        local_metadata,
        rank=rank,
        group=group,
    )
    local_packed_metadata = _normalize_local_owners(
        local_packed_metadata,
        rank=rank,
        group=group,
    )
    _validate_local_tensors(local_tensors, local_metadata, exchange_device)
    _validate_local_tensors(
        local_packed_tensors,
        local_packed_metadata,
        exchange_device,
    )
    regular_candidates = _gather_candidates(
        local_metadata,
        rank=rank,
        world_size=world_size,
        group=group,
    )
    packed_candidates = _gather_candidates(
        local_packed_metadata,
        rank=rank,
        world_size=world_size,
        group=group,
    )
    candidates = {**regular_candidates, **packed_candidates}
    if len(candidates) != len(regular_candidates) + len(packed_candidates):
        raise RuntimeError("regular and packed LoRA identities overlap")
    if not candidates:
        raise RuntimeError("rank-distributed LoRA publication has no tensors")
    block_owners = _assign_blocks(candidates, world_size)
    regular, packed = _canonical_sources(candidates, block_owners)
    exchanged_regular, sent_regular, received_regular = _exchange_to_block_owners(
        regular,
        local_tensors,
        block_owners,
        rank=rank,
        world_size=world_size,
        group=group,
        device=exchange_device,
    )
    exchanged_packed, sent_packed, received_packed = _exchange_to_block_owners(
        packed,
        local_packed_tensors,
        block_owners,
        rank=rank,
        world_size=world_size,
        group=group,
        device=exchange_device,
    )
    output, local_configs, owned_block_count, peak_accounted_owner_bytes = (
        _merge_owned_blocks(
            regular,
            packed,
            exchanged_regular,
            exchanged_packed,
            block_owners,
            rank=rank,
            handler=handler,
            adapter_config=adapter_config,
        )
    )
    published_config = _published_config(
        local_configs,
        handler=handler,
        adapter_config=adapter_config,
        world_size=world_size,
        group=group,
    )
    output_metadata = _gather_output_metadata(
        output,
        rank=rank,
        world_size=world_size,
        group=group,
    )
    config_bytes = encode_adapter_config(published_config)
    header = _safetensors_header(output_metadata)
    layout = _build_layout(
        target,
        output_metadata,
        config_bytes,
        header,
        coordinator_rank=coordinator_rank,
        world_size=world_size,
    )
    owned_upload_bytes = sum(
        shard.byte_count
        for shard, ownership in zip(layout.plan.shards, layout.shards, strict=True)
        if ownership.owner_rank == rank
    )
    builder = stager.begin()
    staged_output = (
        _stage_published_tensors(output, builder)
        if exchange_device.type == "cuda"
        else output
    )
    prepared = PreparedRankDistributedLora(
        layout=layout,
        rank=rank,
        tensors=staged_output,
        adapter_config=config_bytes,
        safetensors_header=header,
        stats=RankDistributedLoraStats(
            rank=rank,
            world_size=world_size,
            source_bytes=sum(tensor.nbytes for tensor in local_tensors.values())
            + sum(tensor.nbytes for tensor in local_packed_tensors.values()),
            sent_bytes=sent_regular + sent_packed,
            received_bytes=received_regular + received_packed,
            owned_tensor_bytes=sum(tensor.nbytes for tensor in output.values()),
            peak_accounted_owner_bytes=peak_accounted_owner_bytes,
            owned_upload_bytes=owned_upload_bytes,
            owned_tensor_count=len(output),
            owned_block_count=owned_block_count,
        ),
    )
    return builder.finish(prepared)


def _broadcast_result(
    value: Any,
    error: BaseException | None,
    *,
    coordinator_rank: int,
    rank: int,
    world_size: int,
    group: Any | None,
) -> Any:
    payload = [
        (
            value,
            None if error is None else f"{type(error).__name__}: {error}",
        )
        if rank == coordinator_rank
        else None
    ]
    if world_size > 1:
        torch.distributed.broadcast_object_list(  # type: ignore[possibly-missing-attribute]
            payload,
            src=coordinator_rank,
            group=group,
        )
    result = payload[0]
    if result is None:
        raise RuntimeError("LoRA publication broadcast returned no result")
    value, message = result
    if message is not None:
        raise RuntimeError(f"rank-distributed LoRA publication failed: {message}")
    return value


def publish_rank_distributed_vllm_lora(
    prepared: PreparedRankDistributedLora,
    store: S3BinaryObjectStore,
    *,
    group: Any | None = None,
) -> OrderedBinaryObjectRef:
    """Publish plan first, rank-owned shards concurrently, and commit last."""
    rank, world_size = _rank_world(group)
    layout = prepared.layout
    if (rank, world_size) != (prepared.rank, layout.world_size):
        raise RuntimeError("rank-distributed LoRA publication world changed")
    coordinator = layout.coordinator_rank
    plan_ref = None
    plan_error = None
    if rank == coordinator:
        try:
            plan_ref = store.publish_ordered_plan(
                target=layout.target,
                plan=layout.plan,
            )
        except BaseException as error:
            plan_error = error
    plan_ref = _broadcast_result(
        plan_ref,
        plan_error,
        coordinator_rank=coordinator,
        rank=rank,
        world_size=world_size,
        group=group,
    )
    if plan_ref != layout.ref:
        raise RuntimeError("published LoRA plan differs from prepared reference")
    stored: tuple[StoredOrderedBinaryObjectShard, ...] = ()
    upload_error = None
    try:
        stored = store.upload_ordered_shards(
            layout.target,
            layout.plan,
            prepared.shard_payloads(),
        )
    except BaseException as error:
        upload_error = f"{type(error).__name__}: {error}"
    uploads: list[
        tuple[tuple[StoredOrderedBinaryObjectShard, ...], str | None] | None
    ] = [None] * world_size
    local_upload = (stored, upload_error)
    if world_size == 1:
        uploads[0] = local_upload
    else:
        torch.distributed.all_gather_object(uploads, local_upload, group=group)  # type: ignore[possibly-missing-attribute]
    failures = [
        error for value in uploads if value is not None for error in [value[1]] if error
    ]
    if any(value is None for value in uploads):
        failures.append("a rank returned no shard-upload result")
    if failures:
        raise RuntimeError(
            "rank-distributed LoRA shard upload failed: " + "; ".join(failures)
        )
    committed = None
    commit_error = None
    if rank == coordinator:
        try:
            all_shards = tuple(
                shard for value in uploads if value is not None for shard in value[0]
            )
            committed = store.commit_ordered(
                layout.target,
                layout.plan,
                all_shards,
            )
        except BaseException as error:
            commit_error = error
    committed = _broadcast_result(
        committed,
        commit_error,
        coordinator_rank=coordinator,
        rank=rank,
        world_size=world_size,
        group=group,
    )
    if committed != layout.ref:
        raise RuntimeError("committed LoRA object differs from its prepared reference")
    return committed
