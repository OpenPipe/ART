from collections import defaultdict
from collections.abc import Callable, Sequence
import json
import math
import struct
import sys
from typing import Any, Literal, cast

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
from art.megatron.lora import LoraShardMeta
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


class RankDistributedLoraCallContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target: OrderedBinaryObjectTarget | None = None
    handler_key: str = Field(min_length=1)
    adapter_config: bytes = Field(min_length=1)
    coordinator_rank: int = Field(ge=0)
    exchange_device_type: Literal["cpu", "cuda"]


class RankDistributedLoraPrepareRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    contract: RankDistributedLoraCallContract | None = None
    regular: tuple[LoraShardMeta, ...] = ()
    packed: tuple[PackedExpertShardMeta, ...] = ()
    conversion_groups: tuple[tuple[str, str], ...] = ()
    error: str | None = None

    @model_validator(mode="after")
    def _valid_or_failed(self) -> "RankDistributedLoraPrepareRecord":
        if (self.error is None) != (self.contract is not None):
            raise ValueError("LoRA prepare record must be valid or failed")
        if self.error is not None and (
            self.regular or self.packed or self.conversion_groups
        ):
            raise ValueError("failed LoRA prepare record cannot carry metadata")
        return self


class RankDistributedLoraOutputRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tensors: tuple[RankOwnedTensor, ...] = ()
    configs: tuple[dict[str, Any], ...] = ()
    error: str | None = None

    @model_validator(mode="after")
    def _valid_or_failed(self) -> "RankDistributedLoraOutputRecord":
        if self.error is not None and (self.tensors or self.configs):
            raise ValueError("failed LoRA output record cannot carry metadata")
        return self


class RankDistributedLoraPublicationReadiness(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    error: str | None = None


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


class PreparedRankDistributedLoraSource(BaseModel):
    """Target-independent, converted rank-owned sampler weights."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    coordinator_rank: int = Field(ge=0)
    world_size: int = Field(ge=1)
    rank: int = Field(ge=0)
    metadata: tuple[RankOwnedTensor, ...] = Field(min_length=1)
    tensors: dict[str, torch.Tensor]
    adapter_config: bytes
    safetensors_header: bytes
    stats: RankDistributedLoraStats

    def bind_target(
        self, target: OrderedBinaryObjectTarget
    ) -> PreparedRankDistributedLora:
        layout = _build_layout(
            target,
            self.metadata,
            self.adapter_config,
            self.safetensors_header,
            coordinator_rank=self.coordinator_rank,
            world_size=self.world_size,
        )
        owned_upload_bytes = sum(
            shard.byte_count
            for shard, ownership in zip(
                layout.plan.shards, layout.shards, strict=True
            )
            if ownership.owner_rank == self.rank
        )
        return PreparedRankDistributedLora(
            layout=layout,
            rank=self.rank,
            tensors=self.tensors,
            adapter_config=self.adapter_config,
            safetensors_header=self.safetensors_header,
            stats=self.stats.model_copy(
                update={"owned_upload_bytes": owned_upload_bytes}
            ),
        )


class ConsolidatedRankDistributedLora(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    tensors: dict[str, torch.Tensor]
    adapter_config: dict[str, Any]


def consolidate_rank_distributed_vllm_lora_source(
    source: PreparedRankDistributedLoraSource,
    *,
    group: Any | None = None,
) -> ConsolidatedRankDistributedLora | None:
    """Gather immutable CPU sampler tensors to the coordinator for persistence."""
    rank, world_size = _rank_world(group)
    if (rank, world_size) != (source.rank, source.world_size):
        raise RuntimeError("rank-distributed LoRA source world changed")
    if world_size > 1 and str(torch.distributed.get_backend(group)) != "gloo":  # type: ignore[possibly-missing-attribute]
        raise RuntimeError("LoRA persistence consolidation requires Gloo")
    local_names = tuple(
        tensor.name for tensor in source.metadata if tensor.owner_rank == rank
    )
    if set(source.tensors) != set(local_names):
        raise RuntimeError("rank-distributed LoRA source tensors changed")
    coordinator = source.coordinator_rank
    consolidated: dict[str, torch.Tensor] = {}
    for tag, meta in enumerate(source.metadata):
        dtype = _dtype_from_name(meta.dtype_name)
        if rank == meta.owner_rank:
            tensor = source.tensors[meta.name]
            if (
                tensor.device.type != "cpu"
                or not tensor.is_contiguous()
                or tuple(tensor.shape) != meta.shape
                or tensor.dtype != dtype
            ):
                raise RuntimeError(
                    f"rank-distributed LoRA tensor changed: {meta.name}"
                )
            if rank == coordinator:
                consolidated[meta.name] = tensor
            elif world_size > 1:
                torch.distributed.send(tensor, dst=coordinator, group=group, tag=tag)  # type: ignore[possibly-missing-attribute]
        elif rank == coordinator:
            tensor = torch.empty(meta.shape, dtype=dtype, device="cpu")
            torch.distributed.recv(  # type: ignore[possibly-missing-attribute]
                tensor, src=meta.owner_rank, group=group, tag=tag
            )
            consolidated[meta.name] = tensor
    if world_size > 1:
        torch.distributed.barrier(group=group)  # type: ignore[possibly-missing-attribute]
    if rank != coordinator:
        return None
    return ConsolidatedRankDistributedLora(
        tensors=consolidated,
        adapter_config=json.loads(source.adapter_config),
    )


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


def _all_gather_records(value: Any, world_size: int, group: Any | None) -> list[Any]:
    gathered = [None] * world_size
    if world_size == 1:
        gathered[0] = value
    else:
        torch.distributed.all_gather_object(gathered, value, group=group)  # type: ignore[possibly-missing-attribute]
    if any(record is None for record in gathered):
        raise RuntimeError("rank-distributed LoRA metadata gather omitted a rank")
    return gathered


def _collect_prepare_records(
    *,
    target_contract: OrderedBinaryObjectTarget | None,
    handler: Any,
    adapter_config: dict[str, Any],
    coordinator_rank: int,
    exchange_device: torch.device,
    local_tensors: dict[str, torch.Tensor],
    local_metadata: Sequence[LoraShardMeta],
    local_packed_tensors: dict[str, torch.Tensor],
    local_packed_metadata: Sequence[PackedExpertShardMeta],
    conversion_group_for_key: Callable[[str], str],
    rank: int,
    world_size: int,
    exchange_group: Any | None,
    metadata_group: Any | None,
    local_error: BaseException | None = None,
) -> tuple[
    list[RankDistributedLoraPrepareRecord],
    list[LoraShardMeta],
    list[PackedExpertShardMeta],
]:
    try:
        if local_error is not None:
            raise local_error
        contract = RankDistributedLoraCallContract(
            target=target_contract,
            handler_key=str(getattr(handler, "key", "")),
            adapter_config=encode_adapter_config(adapter_config),
            coordinator_rank=coordinator_rank,
            exchange_device_type=cast(Literal["cpu", "cuda"], exchange_device.type),
        )
        regular = cast(
            list[LoraShardMeta],
            _normalize_local_owners(local_metadata, rank=rank, group=exchange_group),
        )
        packed = cast(
            list[PackedExpertShardMeta],
            _normalize_local_owners(
                local_packed_metadata, rank=rank, group=exchange_group
            ),
        )
        _validate_local_tensors(local_tensors, regular, exchange_device)
        _validate_local_tensors(local_packed_tensors, packed, exchange_device)
        conversion_groups = tuple(
            sorted(
                {
                    meta.key: _metadata_group(meta, conversion_group_for_key)
                    for meta in (*regular, *packed)
                }.items()
            )
        )
        record = RankDistributedLoraPrepareRecord(
            contract=contract,
            regular=tuple(regular),
            packed=tuple(packed),
            conversion_groups=conversion_groups,
        )
    except BaseException as error:
        regular, packed = [], []
        record = RankDistributedLoraPrepareRecord(
            error=f"{type(error).__name__}: {error}"
        )
    gathered = [
        RankDistributedLoraPrepareRecord.model_validate(value)
        for value in _all_gather_records(record, world_size, metadata_group)
    ]
    failures = [
        f"rank {owner}: {value.error}"
        for owner, value in enumerate(gathered)
        if value.error is not None
    ]
    if failures:
        raise RuntimeError("LoRA preparation validation failed: " + "; ".join(failures))
    contract = gathered[0].contract
    if contract is None or any(value.contract != contract for value in gathered[1:]):
        raise RuntimeError("rank-distributed LoRA publication calls are inconsistent")
    return gathered, regular, packed


def _metadata_group(
    meta: LoraShardMeta | PackedExpertShardMeta,
    conversion_group_for_key: Callable[[str], str],
) -> str:
    group = conversion_group_for_key(meta.key)
    if not group:
        raise ValueError(f"LoRA conversion dependency group is empty: {meta.key}")
    return group


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
    gathered: Sequence[Sequence[LoraShardMeta | PackedExpertShardMeta]],
) -> dict[
    tuple[str, str, int, int],
    tuple[LoraShardMeta | PackedExpertShardMeta, ...],
]:
    candidates: dict[
        tuple[str, str, int, int], list[LoraShardMeta | PackedExpertShardMeta]
    ] = defaultdict(list)
    for rank, rank_entries in enumerate(gathered):
        for meta in rank_entries:
            if meta.owner_rank != rank:
                raise RuntimeError("LoRA metadata identifies another source rank")
            candidates[_metadata_identity(meta)].append(meta)
    for identity, entries in candidates.items():
        if len({entry.owner_rank for entry in entries}) != len(entries):
            raise RuntimeError(f"duplicate LoRA source metadata for {identity}")
        expected = _metadata_without_owner(entries[0])
        if any(_metadata_without_owner(entry) != expected for entry in entries[1:]):
            raise RuntimeError(f"inconsistent replicated LoRA metadata for {identity}")
    return {identity: tuple(entries) for identity, entries in candidates.items()}


def _validated_conversion_groups(
    candidates: dict[
        tuple[str, str, int, int],
        tuple[LoraShardMeta | PackedExpertShardMeta, ...],
    ],
    records: Sequence[RankDistributedLoraPrepareRecord],
) -> dict[str, str]:
    groups: dict[str, str] = {}
    for record in records:
        for key, value in record.conversion_groups:
            previous = groups.setdefault(key, value)
            if previous != value:
                raise RuntimeError(
                    f"LoRA conversion dependency group changed across ranks: {key}"
                )
    expected = {entries[0].key for entries in candidates.values()}
    if set(groups) != expected:
        raise RuntimeError("LoRA conversion dependency groups do not cover metadata")
    return groups


def _assign_blocks(
    candidates: dict[
        tuple[str, str, int, int],
        tuple[LoraShardMeta | PackedExpertShardMeta, ...],
    ],
    world_size: int,
    conversion_group_for_key: Callable[[str], str],
) -> dict[str, int]:
    costs: dict[str, int] = defaultdict(int)
    local_bytes: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for entries in candidates.values():
        meta = entries[0]
        block = _metadata_group(meta, conversion_group_for_key)
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
    conversion_group_for_key: Callable[[str], str],
) -> tuple[list[LoraShardMeta], list[PackedExpertShardMeta]]:
    regular: list[LoraShardMeta] = []
    packed: list[PackedExpertShardMeta] = []
    for identity in sorted(candidates):
        entries = candidates[identity]
        destination = block_owners[
            _metadata_group(entries[0], conversion_group_for_key)
        ]
        selected = next(
            (entry for entry in entries if entry.owner_rank == destination),
            min(entries, key=lambda entry: entry.owner_rank),
        )
        if isinstance(selected, PackedExpertShardMeta):
            packed.append(selected)
        else:
            regular.append(selected)
    return regular, packed


def _validate_local_tensors(
    tensors: dict[str, torch.Tensor],
    metadata: Sequence[LoraShardMeta | PackedExpertShardMeta],
    device: torch.device,
) -> None:
    if set(tensors) != {meta.key for meta in metadata}:
        raise ValueError("local LoRA tensors and metadata differ")
    for meta in metadata:
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
    conversion_group_for_key: Callable[[str], str],
) -> tuple[dict[tuple[int, str], torch.Tensor], int, int]:
    received: dict[tuple[int, str], torch.Tensor] = {}
    sent_bytes = 0
    received_bytes = 0
    sort_key = lambda meta: (
        _metadata_group(meta, conversion_group_for_key),
        meta.key,
        int(getattr(meta, "expert_start", -1)),
        int(meta.manifest.get("shard_rank", 0)),
    )
    for dtype_name in sorted({meta.dtype_name for meta in metadata}):
        dtype = _dtype_from_name(dtype_name)
        typed = [meta for meta in metadata if meta.dtype_name == dtype_name]
        remote = any(
            meta.owner_rank
            != block_owners[_metadata_group(meta, conversion_group_for_key)]
            for meta in typed
        )
        for meta in typed:
            destination = block_owners[_metadata_group(meta, conversion_group_for_key)]
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
                    and block_owners[
                        _metadata_group(meta, conversion_group_for_key)
                    ]
                    == destination
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
                    and block_owners[
                        _metadata_group(meta, conversion_group_for_key)
                    ]
                    == rank
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
    conversion_group_for_key: Callable[[str], str],
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
        block_regular = [
            meta
            for meta in regular
            if _metadata_group(meta, conversion_group_for_key) == block
        ]
        entries: dict[str, list[tuple[dict[str, Any], torch.Tensor]]] = defaultdict(
            list
        )
        for meta in block_regular:
            entries[meta.key].append(
                (meta.manifest, regular_tensors[(meta.owner_rank, meta.key)])
            )
        merged = merge_sharded_adapter_entries(dict(entries))
        block_packed = [
            meta
            for meta in packed
            if _metadata_group(meta, conversion_group_for_key) == block
        ]
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
    records: Sequence[RankDistributedLoraOutputRecord],
    *,
    handler: Any,
    adapter_config: dict[str, Any],
) -> dict[str, Any]:
    unique: dict[bytes, dict[str, Any]] = {}
    for record in records:
        for value in record.configs:
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


def _local_output_metadata(
    tensors: dict[str, torch.Tensor],
    *,
    rank: int,
) -> tuple[RankOwnedTensor, ...]:
    return tuple(
        RankOwnedTensor(
            name=name,
            owner_rank=rank,
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype_name=str(tensor.dtype).removeprefix("torch."),
            byte_count=tensor.nbytes,
        )
        for name, tensor in sorted(tensors.items())
    )


def _collect_output_records(
    output: dict[str, torch.Tensor],
    local_configs: list[dict[str, Any]],
    *,
    rank: int,
    world_size: int,
    metadata_group: Any | None,
    error: BaseException | None = None,
) -> tuple[RankDistributedLoraOutputRecord, ...]:
    record = (
        RankDistributedLoraOutputRecord(
            tensors=_local_output_metadata(output, rank=rank),
            configs=tuple(local_configs),
        )
        if error is None
        else RankDistributedLoraOutputRecord(error=f"{type(error).__name__}: {error}")
    )
    gathered = tuple(
        RankDistributedLoraOutputRecord.model_validate(value)
        for value in _all_gather_records(record, world_size, metadata_group)
    )
    failures = [
        f"rank {owner}: {value.error}"
        for owner, value in enumerate(gathered)
        if value.error is not None
    ]
    if failures:
        raise RuntimeError("LoRA conversion failed: " + "; ".join(failures))
    return gathered


def _global_output_metadata(
    records: Sequence[RankDistributedLoraOutputRecord],
) -> tuple[RankOwnedTensor, ...]:
    result = sorted(
        (tensor for record in records for tensor in record.tensors),
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


def prepare_rank_distributed_vllm_lora_source(
    *,
    local_tensors: dict[str, torch.Tensor],
    local_metadata: Sequence[LoraShardMeta],
    local_packed_tensors: dict[str, torch.Tensor],
    local_packed_metadata: Sequence[PackedExpertShardMeta],
    handler: Any | None,
    adapter_config: dict[str, Any],
    conversion_group_for_key: Callable[[str], str],
    group: Any | None = None,
    metadata_group: Any | None = None,
    coordinator_rank: int = 0,
    exchange_device: torch.device | None,
    stager: PinnedCpuSnapshotStager | None,
    local_error: BaseException | None = None,
    target_contract: OrderedBinaryObjectTarget | None = None,
) -> PendingCpuSnapshot[PreparedRankDistributedLoraSource]:
    """Gather declared conversion dependency groups and prepare rank-owned L2 ranges.

    The supplied tensors retain current global-rank metadata and must remain immutable
    through owner exchange and conversion. A NCCL caller may invoke this on a side
    stream after its snapshot-ready event; it must register the returned pending
    snapshot before allowing the optimizer to mutate the source weights. The required
    grouping callable is part of the handler contract: every key jointly needed for
    converted tensors or a returned config decision must map to the same group.
    """
    rank, world_size = _rank_world(group)
    if world_size > 1 and metadata_group is None:
        metadata_group = group
    try:
        if local_error is not None:
            raise local_error
        if handler is None or stager is None or exchange_device is None:
            raise RuntimeError("rank-distributed LoRA preparation is incomplete")
        exchange_device = torch.device(exchange_device)
        if exchange_device.type == "cuda" and exchange_device.index is None:
            exchange_device = torch.device("cuda", torch.cuda.current_device())
        if exchange_device.type not in {"cpu", "cuda"}:
            raise ValueError(
                f"unsupported LoRA exchange device: {exchange_device.type}"
            )
        if coordinator_rank >= world_size:
            raise ValueError("LoRA publication coordinator leaves its world")
        if world_size > 1:
            backend = str(torch.distributed.get_backend(group))  # type: ignore[possibly-missing-attribute]
            expected_backend = "gloo" if exchange_device.type == "cpu" else "nccl"
            if backend != expected_backend:
                raise RuntimeError(
                    "LoRA exchange device must match its distributed backend: "
                    f"device={exchange_device.type} backend={backend}"
                )
            metadata_backend = str(torch.distributed.get_backend(metadata_group))  # type: ignore[possibly-missing-attribute]
            if metadata_backend != "gloo":
                raise RuntimeError(
                    "LoRA metadata exchange requires an all-rank Gloo group"
                )
            if (
                torch.distributed.get_world_size(metadata_group) != world_size  # type: ignore[possibly-missing-attribute]
                or tuple(torch.distributed.get_process_group_ranks(metadata_group))  # type: ignore[possibly-missing-attribute]
                != tuple(torch.distributed.get_process_group_ranks(group))  # type: ignore[possibly-missing-attribute]
            ):
                raise RuntimeError(
                    "LoRA metadata and tensor groups cover different ranks"
                )
    except BaseException as error:
        local_error = error
        exchange_device = torch.device("cpu")
    records, local_metadata, local_packed_metadata = _collect_prepare_records(
        target_contract=target_contract,
        handler=handler,
        adapter_config=adapter_config,
        coordinator_rank=coordinator_rank,
        exchange_device=exchange_device,
        local_tensors=local_tensors,
        local_metadata=local_metadata,
        local_packed_tensors=local_packed_tensors,
        local_packed_metadata=local_packed_metadata,
        conversion_group_for_key=conversion_group_for_key,
        rank=rank,
        world_size=world_size,
        exchange_group=group,
        metadata_group=metadata_group,
        local_error=local_error,
    )
    if handler is None or stager is None:
        raise RuntimeError("LoRA readiness collective accepted incomplete preparation")
    regular_candidates = _gather_candidates(
        tuple(record.regular for record in records),
    )
    packed_candidates = _gather_candidates(
        tuple(record.packed for record in records),
    )
    candidates = {**regular_candidates, **packed_candidates}
    if len(candidates) != len(regular_candidates) + len(packed_candidates):
        raise RuntimeError("regular and packed LoRA identities overlap")
    if not candidates:
        raise RuntimeError("rank-distributed LoRA publication has no tensors")
    conversion_groups = _validated_conversion_groups(
        candidates,
        records,
    )
    conversion_group_for_key = conversion_groups.__getitem__
    block_owners = _assign_blocks(
        candidates,
        world_size,
        conversion_group_for_key,
    )
    regular, packed = _canonical_sources(
        candidates,
        block_owners,
        conversion_group_for_key,
    )
    exchanged_regular, sent_regular, received_regular = _exchange_to_block_owners(
        regular,
        local_tensors,
        block_owners,
        rank=rank,
        world_size=world_size,
        group=group,
        device=exchange_device,
        conversion_group_for_key=conversion_group_for_key,
    )
    exchanged_packed, sent_packed, received_packed = _exchange_to_block_owners(
        packed,
        local_packed_tensors,
        block_owners,
        rank=rank,
        world_size=world_size,
        group=group,
        device=exchange_device,
        conversion_group_for_key=conversion_group_for_key,
    )
    output: dict[str, torch.Tensor] = {}
    local_configs: list[dict[str, Any]] = []
    owned_block_count = 0
    peak_accounted_owner_bytes = 0
    conversion_error: BaseException | None = None
    try:
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
                conversion_group_for_key=conversion_group_for_key,
            )
        )
    except BaseException as error:
        conversion_error = error
    output_records = _collect_output_records(
        output,
        local_configs,
        rank=rank,
        world_size=world_size,
        metadata_group=metadata_group,
        error=conversion_error,
    )
    published_config = _published_config(
        output_records,
        handler=handler,
        adapter_config=adapter_config,
    )
    output_metadata = _global_output_metadata(output_records)
    config_bytes = encode_adapter_config(published_config)
    header = _safetensors_header(output_metadata)
    builder = stager.begin()
    staged_output = (
        _stage_published_tensors(output, builder)
        if exchange_device.type == "cuda"
        else output
    )
    if exchange_device.type == "cuda":
        if exchange_device.index is None:
            raise RuntimeError("CUDA LoRA exchange device has no index")
        builder.fence_current_stream(exchange_device.index)
    prepared = PreparedRankDistributedLoraSource(
        coordinator_rank=coordinator_rank,
        world_size=world_size,
        rank=rank,
        metadata=output_metadata,
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
            owned_upload_bytes=0,
            owned_tensor_count=len(output),
            owned_block_count=owned_block_count,
        ),
    )
    return builder.finish(prepared)


def prepare_rank_distributed_vllm_lora(
    *,
    target: OrderedBinaryObjectTarget,
    local_tensors: dict[str, torch.Tensor],
    local_metadata: Sequence[LoraShardMeta],
    local_packed_tensors: dict[str, torch.Tensor],
    local_packed_metadata: Sequence[PackedExpertShardMeta],
    handler: Any | None,
    adapter_config: dict[str, Any],
    conversion_group_for_key: Callable[[str], str],
    group: Any | None = None,
    metadata_group: Any | None = None,
    coordinator_rank: int = 0,
    exchange_device: torch.device | None,
    stager: PinnedCpuSnapshotStager | None,
    local_error: BaseException | None = None,
) -> PendingCpuSnapshot[PreparedRankDistributedLora]:
    """Compatibility entrypoint that binds a target after source preparation."""
    return prepare_rank_distributed_vllm_lora_source(
        local_tensors=local_tensors,
        local_metadata=local_metadata,
        local_packed_tensors=local_packed_tensors,
        local_packed_metadata=local_packed_metadata,
        handler=handler,
        adapter_config=adapter_config,
        conversion_group_for_key=conversion_group_for_key,
        group=group,
        metadata_group=metadata_group,
        coordinator_rank=coordinator_rank,
        exchange_device=exchange_device,
        stager=stager,
        local_error=local_error,
        target_contract=target,
    ).map(lambda source: source.bind_target(target))


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
    store: S3BinaryObjectStore | None,
    *,
    group: Any | None = None,
    local_error: BaseException | None = None,
) -> OrderedBinaryObjectRef:
    """Publish plan first, rank-owned shards concurrently, and commit last."""
    rank, world_size = _rank_world(group)
    layout = prepared.layout
    if (rank, world_size) != (prepared.rank, layout.world_size):
        raise RuntimeError("rank-distributed LoRA publication world changed")
    readiness = RankDistributedLoraPublicationReadiness(
        error=(
            None
            if local_error is None
            else f"{type(local_error).__name__}: {local_error}"
        )
    )
    ready = tuple(
        RankDistributedLoraPublicationReadiness.model_validate(value)
        for value in _all_gather_records(readiness, world_size, group)
    )
    failures = [
        f"rank {owner}: {value.error}"
        for owner, value in enumerate(ready)
        if value.error is not None
    ]
    if failures:
        raise RuntimeError(
            "rank-distributed LoRA publication is not ready: " + "; ".join(failures)
        )
    if store is None:
        raise RuntimeError("rank-distributed LoRA publication has no object store")
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
