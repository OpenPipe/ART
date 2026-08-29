from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
import hashlib
import importlib
import json
import math
import struct
import sys
from typing import Any, Literal, Protocol, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from art.megatron.lora import LoraShardMeta, LoRASlotRef, _block_for_key
from art.megatron.model_support.lora_disk import (
    ART_LORA_FORMAT_CONFIG_KEY,
    ART_LORA_FORMAT_VLLM,
)
from art.megatron.tensor_snapshot import PendingCpuSnapshot, PinnedCpuSnapshotStager
from art.megatron.training.model_chunks import ModelChunks
from art.megatron.weights.lora_publish import (
    PackedExpertShardMeta,
    _stage_published_tensors,
    collect_local_lora_entries,
    collect_local_packed_expert_entries,
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
ExternalLoraPath = Literal["adapter_config.json", "adapter_model.safetensors"]
_T = TypeVar("_T")
_MISSING = object()


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExternalLoraTarget(_Record):
    """Immutable publication identity and bounded artifact geometry."""

    tenant_id: str = Field(min_length=1, max_length=255)
    run_id: str = Field(min_length=1, max_length=255)
    operation_id: str = Field(min_length=1, max_length=255)
    training_session_id: str = Field(min_length=1, max_length=255)
    publication_id: str = Field(min_length=1, max_length=255)
    policy_step: int = Field(ge=0)
    generation_id: str = Field(pattern=r"^step-\d{8,}-[0-9a-f]{32}$")
    model_identity: str = Field(min_length=1, max_length=4096)
    active_alias: str = Field(min_length=1, max_length=255)
    runtime_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    shard_bytes: int = Field(default=64 << 20, ge=1, le=5 << 30)
    max_shards: int = Field(default=1024, ge=1, le=10_000)
    max_bytes: int = Field(default=64 << 30, ge=1)

    @model_validator(mode="after")
    def _validate_generation_step(self) -> ExternalLoraTarget:
        if int(self.generation_id.split("-", 2)[1]) != self.policy_step:
            raise ValueError("external LoRA generation and policy step differ")
        return self


class ExternalLoraTargetGrant(_Record):
    """Service authorization for exactly one immutable publication plan."""

    authorization_id: str = Field(min_length=1, max_length=255)
    target_revision: str = Field(min_length=1, max_length=255)
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExternalLoraSinkSpec(_Record):
    """Trusted installed service factory for one rank-local publication sink."""

    module: str = Field(min_length=1, max_length=255)
    qualname: str = Field(min_length=1, max_length=255)
    config: dict[str, Any]

    @model_validator(mode="after")
    def _validate_config(self) -> ExternalLoraSinkSpec:
        if len(_canonical_bytes(self.config)) > 64 << 10:
            raise ValueError("external LoRA sink config exceeds 64 KiB")
        return self

    def create(self) -> ExternalLoraPublicationSink:
        value: Any = importlib.import_module(self.module)
        for component in self.qualname.split("."):
            value = getattr(value, component)
        sink = value(self.config)
        for method in ("authorize", "put_shard", "complete", "abort"):
            if not callable(getattr(sink, method, None)):
                raise TypeError(
                    f"external LoRA sink factory returned no {method} method"
                )
        return cast(ExternalLoraPublicationSink, sink)


class ExternalLoraObjectRef(_Record):
    locator: str = Field(min_length=1, max_length=4096)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExternalLoraFile(_Record):
    relative_path: ExternalLoraPath
    size_bytes: int = Field(gt=0)


class ExternalLoraTensor(_Record):
    name: str = Field(min_length=1, max_length=8192)
    owner_rank: int = Field(ge=0)
    shape: tuple[int, ...] = Field(min_length=1)
    dtype_name: str = Field(min_length=1, max_length=32)
    size_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_shape(self) -> ExternalLoraTensor:
        if any(dimension < 1 for dimension in self.shape):
            raise ValueError("external LoRA tensor dimensions must be positive")
        return self


class ExternalLoraSourceSegment(_Record):
    source: Literal["adapter_config", "safetensors_header", "tensor"]
    tensor_name: str | None = Field(default=None, min_length=1, max_length=8192)
    source_offset: int = Field(ge=0)
    size_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_tensor_source(self) -> ExternalLoraSourceSegment:
        if (self.source == "tensor") != (self.tensor_name is not None):
            raise ValueError("only tensor segments identify a tensor")
        return self


class ExternalLoraShardPlan(_Record):
    index: int = Field(ge=0)
    owner_rank: int = Field(ge=0)
    relative_path: ExternalLoraPath
    file_offset: int = Field(ge=0)
    size_bytes: int = Field(gt=0)
    segments: tuple[ExternalLoraSourceSegment, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_size(self) -> ExternalLoraShardPlan:
        if sum(segment.size_bytes for segment in self.segments) != self.size_bytes:
            raise ValueError("external LoRA shard sources do not match its size")
        return self


class ExternalLoraPlan(_Record):
    format: Literal["art_external_lora_plan_v1"] = "art_external_lora_plan_v1"
    target: ExternalLoraTarget
    handler_key: str = Field(min_length=1, max_length=255)
    source_topology: str = Field(min_length=1, max_length=1024)
    coordinator_rank: int = Field(ge=0)
    world_size: int = Field(ge=1)
    files: tuple[ExternalLoraFile, ...] = Field(min_length=2, max_length=2)
    tensors: tuple[ExternalLoraTensor, ...] = Field(min_length=1)
    shards: tuple[ExternalLoraShardPlan, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_layout(self) -> ExternalLoraPlan:
        if self.coordinator_rank >= self.world_size:
            raise ValueError("external LoRA coordinator leaves its trainer world")
        if tuple(file.relative_path for file in self.files) != (
            "adapter_config.json",
            "adapter_model.safetensors",
        ):
            raise ValueError("external LoRA files must use the standard adapter layout")
        names = tuple(tensor.name for tensor in self.tensors)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("external LoRA tensor names must be unique and sorted")
        if any(tensor.owner_rank >= self.world_size for tensor in self.tensors):
            raise ValueError("external LoRA tensor owner leaves its trainer world")
        if tuple(shard.index for shard in self.shards) != tuple(
            range(len(self.shards))
        ):
            raise ValueError("external LoRA shard indexes must be contiguous")
        if len(self.shards) > self.target.max_shards:
            raise ValueError("external LoRA plan exceeds its shard limit")
        tensor_owners = {tensor.name: tensor.owner_rank for tensor in self.tensors}
        cursors = dict.fromkeys((file.relative_path for file in self.files), 0)
        for shard in self.shards:
            if shard.owner_rank >= self.world_size:
                raise ValueError("external LoRA shard owner leaves its trainer world")
            if shard.file_offset != cursors[shard.relative_path]:
                raise ValueError("external LoRA shards leave a file gap")
            cursors[shard.relative_path] += shard.size_bytes
            for segment in shard.segments:
                if (
                    segment.tensor_name is not None
                    and tensor_owners.get(segment.tensor_name) != shard.owner_rank
                ):
                    raise ValueError("external LoRA shard reads another rank's tensor")
                if segment.tensor_name is None and (
                    shard.owner_rank != self.coordinator_rank
                ):
                    raise ValueError("only the coordinator owns adapter metadata")
        if any(cursors[file.relative_path] != file.size_bytes for file in self.files):
            raise ValueError("external LoRA shards do not cover every file")
        if sum(file.size_bytes for file in self.files) > self.target.max_bytes:
            raise ValueError("external LoRA plan exceeds its byte limit")
        return self

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.model_dump(mode="json"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class ExternalLoraShardReceipt(_Record):
    index: int = Field(ge=0)
    ref: ExternalLoraObjectRef


class ExternalLoraRankCompletion(_Record):
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rank: int = Field(ge=0)
    shards: tuple[ExternalLoraShardReceipt, ...] = ()

    @model_validator(mode="after")
    def _validate_receipt_order(self) -> ExternalLoraRankCompletion:
        indexes = tuple(receipt.index for receipt in self.shards)
        if indexes != tuple(sorted(set(indexes))):
            raise ValueError("external LoRA rank receipts must be unique and sorted")
        return self


class ExternalLoraManifest(_Record):
    format: Literal["art_external_lora_v1"] = "art_external_lora_v1"
    plan: ExternalLoraPlan
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shards: tuple[ExternalLoraShardReceipt, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_receipts(self) -> ExternalLoraManifest:
        if self.plan_sha256 != self.plan.sha256:
            raise ValueError("external LoRA manifest changed its plan")
        if tuple(receipt.index for receipt in self.shards) != tuple(
            range(len(self.plan.shards))
        ):
            raise ValueError("external LoRA manifest does not cover every shard")
        for shard, receipt in zip(self.plan.shards, self.shards, strict=True):
            if receipt.ref.size_bytes != shard.size_bytes:
                raise ValueError("external LoRA receipt changed its shard size")
        return self

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.model_dump(mode="json"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class ExternalLoraPublication(_Record):
    manifest: ExternalLoraManifest
    manifest_ref: ExternalLoraObjectRef

    @model_validator(mode="after")
    def _validate_manifest_ref(self) -> ExternalLoraPublication:
        if (
            self.manifest_ref.sha256 != self.manifest.sha256
            or self.manifest_ref.size_bytes != len(self.manifest.canonical_bytes())
        ):
            raise ValueError("external LoRA manifest reference changed its identity")
        return self


class ExternalLoraPublicationSink(Protocol):
    """Service-owned authorization, object write, and manifest settlement."""

    def authorize(self, plan: ExternalLoraPlan) -> ExternalLoraTargetGrant: ...

    def put_shard(
        self,
        grant: ExternalLoraTargetGrant,
        shard: ExternalLoraShardPlan,
        chunks: Sequence[memoryview],
    ) -> ExternalLoraObjectRef: ...

    def complete(
        self,
        grant: ExternalLoraTargetGrant,
        plan: ExternalLoraPlan,
        completions: Sequence[ExternalLoraRankCompletion],
    ) -> ExternalLoraPublication: ...

    def abort(
        self,
        grant: ExternalLoraTargetGrant,
        completion: ExternalLoraRankCompletion,
        error: str,
    ) -> None: ...


class PreparedExternalLora(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    plan: ExternalLoraPlan
    rank: int = Field(ge=0)
    tensors: dict[str, torch.Tensor]
    adapter_config: bytes
    safetensors_header: bytes

    def shard_payloads(self) -> dict[int, tuple[memoryview, ...]]:
        fixed = {
            "adapter_config": memoryview(self.adapter_config),
            "safetensors_header": memoryview(self.safetensors_header),
        }
        tensors = {
            name: memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast("B")
            for name, tensor in self.tensors.items()
        }
        payloads: dict[int, tuple[memoryview, ...]] = {}
        for shard in self.plan.shards:
            if shard.owner_rank != self.rank:
                continue
            chunks = []
            for segment in shard.segments:
                source = (
                    tensors[segment.tensor_name]
                    if segment.tensor_name is not None
                    else fixed[segment.source]
                )
                chunks.append(
                    source[
                        segment.source_offset : segment.source_offset
                        + segment.size_bytes
                    ]
                )
            payloads[shard.index] = tuple(chunks)
        return payloads


def stage_external_lora_from_model(
    *,
    model: ModelChunks,
    adapter_dtypes: dict[str, torch.dtype],
    handler: Any,
    adapter_config: dict[str, Any],
    target: ExternalLoraTarget,
    source_topology: str,
    stager: PinnedCpuSnapshotStager,
    group: Any | None = None,
    coordinator_rank: int = 0,
    slot_ref: LoRASlotRef | None = None,
) -> PendingCpuSnapshot[PreparedExternalLora]:
    rank, _world_size = _rank_world(group)
    packed_groups = tuple(handler.expert_packed_lora_groups())
    regular, regular_metadata = collect_local_lora_entries(
        model,
        adapter_dtypes,
        owner_rank=rank,
        packed_expert_groups=packed_groups,
        slot_ref=slot_ref,
    )
    packed, packed_metadata = collect_local_packed_expert_entries(
        model,
        adapter_dtypes,
        owner_rank=rank,
        packed_expert_groups=packed_groups,
        slot_ref=slot_ref,
    )
    devices = {tensor.device for tensor in (*regular.values(), *packed.values())}
    if len(devices) > 1:
        raise RuntimeError("external LoRA source tensors must share one device")
    if devices:
        exchange_device = next(iter(devices))
    elif torch.distributed.is_initialized():  # type: ignore[possibly-missing-attribute]
        exchange_device = (
            torch.device("cuda", torch.cuda.current_device())
            if str(torch.distributed.get_backend(group)) == "nccl"  # type: ignore[possibly-missing-attribute]
            else torch.device("cpu")
        )
    else:
        exchange_device = torch.device("cpu")
    return prepare_external_lora(
        target=target,
        source_topology=source_topology,
        local_tensors=regular,
        local_metadata=regular_metadata,
        local_packed_tensors=packed,
        local_packed_metadata=packed_metadata,
        handler=handler,
        adapter_config=adapter_config,
        exchange_device=exchange_device,
        stager=stager,
        group=group,
        coordinator_rank=coordinator_rank,
    )


def prepare_external_lora(
    *,
    target: ExternalLoraTarget,
    source_topology: str,
    local_tensors: dict[str, torch.Tensor],
    local_metadata: Sequence[LoraShardMeta],
    local_packed_tensors: dict[str, torch.Tensor],
    local_packed_metadata: Sequence[PackedExpertShardMeta],
    handler: Any,
    adapter_config: dict[str, Any],
    exchange_device: torch.device,
    stager: PinnedCpuSnapshotStager,
    group: Any | None = None,
    coordinator_rank: int = 0,
) -> PendingCpuSnapshot[PreparedExternalLora]:
    """Prepare rank-owned ranges of one standard, externally loadable LoRA."""

    rank, world_size = _rank_world(group)
    device = torch.device(exchange_device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    _synchronize_local(
        "source preflight",
        lambda: _validate_preparation(
            local_tensors,
            local_metadata,
            local_packed_tensors,
            local_packed_metadata,
            rank=rank,
            world_size=world_size,
            coordinator_rank=coordinator_rank,
            device=device,
            group=group,
        ),
        world_size=world_size,
        group=group,
    )
    regular_candidates = _gather_candidates(
        local_metadata, rank=rank, world_size=world_size, group=group
    )
    packed_candidates = _gather_candidates(
        local_packed_metadata, rank=rank, world_size=world_size, group=group
    )
    candidates = {**regular_candidates, **packed_candidates}
    if len(candidates) != len(regular_candidates) + len(packed_candidates):
        raise RuntimeError("regular and packed external LoRA identities overlap")
    if not candidates:
        raise RuntimeError("external LoRA publication has no tensors")
    block_owners = _assign_blocks(candidates, world_size)
    regular_sources, packed_sources = _canonical_sources(candidates, block_owners)
    exchanged_regular = _exchange_to_block_owners(
        regular_sources,
        local_tensors,
        block_owners,
        rank=rank,
        world_size=world_size,
        group=group,
        device=device,
    )
    exchanged_packed = _exchange_to_block_owners(
        packed_sources,
        local_packed_tensors,
        block_owners,
        rank=rank,
        world_size=world_size,
        group=group,
        device=device,
    )
    output, local_configs = _synchronize_local(
        "conversion",
        lambda: _convert_owned_blocks(
            regular_sources,
            packed_sources,
            exchanged_regular,
            exchanged_packed,
            block_owners,
            rank=rank,
            handler=handler,
            adapter_config=adapter_config,
        ),
        world_size=world_size,
        group=group,
    )
    published_config = _published_config(
        local_configs,
        handler=handler,
        adapter_config=adapter_config,
        world_size=world_size,
        group=group,
    )
    output_metadata = _gather_output_metadata(
        output, rank=rank, world_size=world_size, group=group
    )
    config_bytes = _encode_adapter_config(published_config)
    header = _safetensors_header(output_metadata)
    plan = _synchronize_local(
        "layout",
        lambda: _build_plan(
            target,
            output_metadata,
            config_bytes,
            header,
            handler_key=str(handler.key),
            source_topology=source_topology,
            coordinator_rank=coordinator_rank,
            world_size=world_size,
        ),
        world_size=world_size,
        group=group,
    )
    if len(set(_gather_values(plan.sha256, world_size, group))) != 1:
        raise RuntimeError("external LoRA ranks produced different plans")
    builder = stager.begin()
    staged = (
        _stage_published_tensors(output, builder) if device.type == "cuda" else output
    )
    return builder.finish(
        PreparedExternalLora(
            plan=plan,
            rank=rank,
            tensors=staged,
            adapter_config=config_bytes,
            safetensors_header=header,
        )
    )


def publish_external_lora_rank(
    prepared: PreparedExternalLora,
    sink: ExternalLoraPublicationSink,
    *,
    group: Any | None = None,
) -> ExternalLoraPublication | None:
    """Write this rank's ranges; the service settles the complete manifest."""

    rank, world_size = _rank_world(group)
    plan = prepared.plan
    if (rank, world_size) != (prepared.rank, plan.world_size):
        raise RuntimeError("external LoRA publication world changed")
    payloads = prepared.shard_payloads()
    expected = tuple(
        shard.index for shard in plan.shards if shard.owner_rank == prepared.rank
    )
    if tuple(payloads) != expected:
        raise RuntimeError("external LoRA rank payloads changed their plan")
    for index, chunks in payloads.items():
        if sum(chunk.nbytes for chunk in chunks) != plan.shards[index].size_bytes:
            raise RuntimeError("external LoRA rank payload changed its shard size")

    grant: ExternalLoraTargetGrant | None = None
    grant_error = None
    try:
        grant = sink.authorize(plan)
        if grant.plan_sha256 != plan.sha256:
            raise RuntimeError("external LoRA authorization changed its plan")
    except BaseException as error:
        grant_error = _error_text(error)
    grants = _gather_values(
        (None if grant is None else grant.model_dump(mode="json"), grant_error),
        world_size,
        group,
    )
    grant_errors = [error for _value, error in grants if error]
    if grant_errors:
        raise RuntimeError(
            "external LoRA authorization failed: " + "; ".join(grant_errors)
        )
    grant_values = [value for value, _error in grants]
    if (
        None in grant_values
        or len({_canonical_bytes(value) for value in grant_values}) != 1
    ):
        raise RuntimeError("external LoRA ranks received different authorizations")
    assert grant is not None

    receipts: list[ExternalLoraShardReceipt] = []
    upload_error = None
    try:
        for index, chunks in payloads.items():
            shard = plan.shards[index]
            ref = sink.put_shard(grant, shard, chunks)
            if ref.size_bytes != shard.size_bytes:
                raise RuntimeError("external LoRA sink changed a shard size")
            receipts.append(ExternalLoraShardReceipt(index=index, ref=ref))
    except BaseException as error:
        upload_error = _error_text(error)

    completion = ExternalLoraRankCompletion(
        plan_sha256=plan.sha256,
        rank=rank,
        shards=tuple(receipts),
    )
    upload_errors = [
        error for error in _gather_values(upload_error, world_size, group) if error
    ]
    if upload_errors:
        cleanup_error = None
        try:
            sink.abort(grant, completion, "; ".join(upload_errors))
        except BaseException as error:
            cleanup_error = _error_text(error)
        cleanup_errors = [
            error for error in _gather_values(cleanup_error, world_size, group) if error
        ]
        detail = "; ".join((*upload_errors, *cleanup_errors))
        raise RuntimeError(f"external LoRA upload failed: {detail}")

    completions = tuple(
        ExternalLoraRankCompletion.model_validate(value)
        for value in _gather_values(
            completion.model_dump(mode="json"), world_size, group
        )
    )
    publication = None
    settlement_error = None
    if rank == plan.coordinator_rank:
        try:
            publication = sink.complete(grant, plan, completions)
            if publication.manifest.plan_sha256 != plan.sha256:
                raise RuntimeError("external LoRA settlement changed its plan")
        except BaseException as error:
            settlement_error = _error_text(error)
    settlement = _gather_values(
        (
            None if publication is None else publication.model_dump(mode="json"),
            settlement_error,
        ),
        world_size,
        group,
    )[plan.coordinator_rank]
    publication_value, settlement_error = settlement
    if settlement_error:
        cleanup_error = None
        try:
            sink.abort(grant, completion, settlement_error)
        except BaseException as error:
            cleanup_error = _error_text(error)
        cleanup_errors = [
            error for error in _gather_values(cleanup_error, world_size, group) if error
        ]
        detail = "; ".join((settlement_error, *cleanup_errors))
        raise RuntimeError(f"external LoRA settlement failed: {detail}")
    if publication_value is None:
        raise RuntimeError("external LoRA coordinator returned no publication")
    resolved = ExternalLoraPublication.model_validate(publication_value)
    return resolved if rank == plan.coordinator_rank else None


def _rank_world(group: Any | None) -> tuple[int, int]:
    if not torch.distributed.is_initialized():  # type: ignore[possibly-missing-attribute]
        return 0, 1
    return (
        int(torch.distributed.get_rank(group)),  # type: ignore[possibly-missing-attribute]
        int(torch.distributed.get_world_size(group)),  # type: ignore[possibly-missing-attribute]
    )


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"unsupported external LoRA dtype {name!r}")
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


def _metadata_without_owner(
    meta: LoraShardMeta | PackedExpertShardMeta,
) -> LoraShardMeta | PackedExpertShardMeta:
    return meta._replace(owner_rank=0)


def _validate_preparation(
    tensors: Mapping[str, torch.Tensor],
    metadata: Sequence[LoraShardMeta],
    packed_tensors: Mapping[str, torch.Tensor],
    packed_metadata: Sequence[PackedExpertShardMeta],
    *,
    rank: int,
    world_size: int,
    coordinator_rank: int,
    device: torch.device,
    group: Any | None,
) -> None:
    if coordinator_rank >= world_size:
        raise ValueError("external LoRA coordinator leaves its trainer world")
    if world_size > 1:
        backend = str(torch.distributed.get_backend(group))  # type: ignore[possibly-missing-attribute]
        expected = "gloo" if device.type == "cpu" else "nccl"
        if device.type not in {"cpu", "cuda"} or backend != expected:
            raise RuntimeError(
                "external LoRA exchange device does not match its backend: "
                f"device={device.type} backend={backend}"
            )
    _validate_local_tensors(tensors, metadata, rank=rank, device=device)
    _validate_local_tensors(packed_tensors, packed_metadata, rank=rank, device=device)


def _validate_local_tensors(
    tensors: Mapping[str, torch.Tensor],
    metadata: Sequence[LoraShardMeta | PackedExpertShardMeta],
    *,
    rank: int,
    device: torch.device,
) -> None:
    if set(tensors) != {meta.key for meta in metadata}:
        raise ValueError("external LoRA tensors and metadata differ")
    for meta in metadata:
        if meta.owner_rank != rank:
            raise ValueError("external LoRA metadata identifies another rank")
        tensor = tensors[meta.key]
        if (
            tensor.device != device
            or not tensor.is_contiguous()
            or tuple(tensor.shape) != meta.shape
            or tensor.dtype != _dtype_from_name(meta.dtype_name)
        ):
            raise ValueError(f"external LoRA tensor differs from metadata: {meta.key}")


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
    gathered = _gather_values(list(local), world_size, group)
    candidates: dict[
        tuple[str, str, int, int], list[LoraShardMeta | PackedExpertShardMeta]
    ] = defaultdict(list)
    for entries in gathered:
        for meta in entries:
            candidates[_metadata_identity(meta)].append(meta)
    for identity, entries in candidates.items():
        if len({entry.owner_rank for entry in entries}) != len(entries):
            raise RuntimeError(f"duplicate external LoRA source metadata: {identity}")
        expected = _metadata_without_owner(entries[0])
        if any(_metadata_without_owner(entry) != expected for entry in entries[1:]):
            raise RuntimeError(f"inconsistent external LoRA replicas: {identity}")
    return {identity: tuple(entries) for identity, entries in candidates.items()}


def _assign_blocks(
    candidates: Mapping[
        tuple[str, str, int, int],
        Sequence[LoraShardMeta | PackedExpertShardMeta],
    ],
    world_size: int,
) -> dict[str, int]:
    costs: dict[str, int] = defaultdict(int)
    local_bytes: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for entries in candidates.values():
        meta = entries[0]
        block = _metadata_block(meta)
        size = (
            meta.numel
            * torch.empty((), dtype=_dtype_from_name(meta.dtype_name)).element_size()
        )
        costs[block] += size
        for candidate in entries:
            local_bytes[block][candidate.owner_rank] += size
    loads = [0] * world_size
    owners = {}
    for block, cost in sorted(costs.items(), key=lambda item: (-item[1], item[0])):
        minimum = min(loads)
        eligible = [rank for rank, load in enumerate(loads) if load == minimum]
        owner = min(eligible, key=lambda rank: (-local_bytes[block][rank], rank))
        owners[block] = owner
        loads[owner] += cost
    return owners


def _canonical_sources(
    candidates: Mapping[
        tuple[str, str, int, int],
        Sequence[LoraShardMeta | PackedExpertShardMeta],
    ],
    block_owners: Mapping[str, int],
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
        if isinstance(selected, PackedExpertShardMeta):
            packed.append(selected)
        else:
            regular.append(selected)
    return regular, packed


def _exchange_to_block_owners(
    metadata: Sequence[LoraShardMeta | PackedExpertShardMeta],
    local_tensors: Mapping[str, torch.Tensor],
    block_owners: Mapping[str, int],
    *,
    rank: int,
    world_size: int,
    group: Any | None,
    device: torch.device,
) -> dict[tuple[int, str], torch.Tensor]:
    received: dict[tuple[int, str], torch.Tensor] = {}
    sort_key = lambda meta: (
        _metadata_block(meta),
        meta.key,
        int(getattr(meta, "expert_start", -1)),
        int(meta.manifest.get("shard_rank", 0)),
    )
    for dtype_name in sorted({meta.dtype_name for meta in metadata}):
        dtype = _dtype_from_name(dtype_name)
        typed = [meta for meta in metadata if meta.dtype_name == dtype_name]
        for meta in typed:
            destination = block_owners[_metadata_block(meta)]
            if meta.owner_rank == rank == destination:
                received[(rank, meta.key)] = local_tensors[meta.key]
        if not any(
            meta.owner_rank != block_owners[_metadata_block(meta)] for meta in typed
        ):
            continue
        sends = [
            sorted(
                (
                    meta
                    for meta in typed
                    if meta.owner_rank == rank
                    and block_owners[_metadata_block(meta)] == destination
                    and destination != rank
                ),
                key=sort_key,
            )
            for destination in range(world_size)
        ]
        receives = [
            sorted(
                (
                    meta
                    for meta in typed
                    if meta.owner_rank == source
                    and block_owners[_metadata_block(meta)] == rank
                    and source != rank
                ),
                key=sort_key,
            )
            for source in range(world_size)
        ]
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
        output_splits = [sum(meta.numel for meta in entries) for entries in receives]
        output = torch.empty(sum(output_splits), dtype=dtype, device=device)
        torch.distributed.all_to_all_single(  # type: ignore[possibly-missing-attribute]
            output,
            send,
            output_split_sizes=output_splits,
            input_split_sizes=[
                sum(meta.numel for meta in entries) for entries in sends
            ],
            group=group,
        )
        offset = 0
        for entries in receives:
            for meta in entries:
                key = (meta.owner_rank, meta.key)
                if key in received:
                    raise RuntimeError(
                        f"duplicate exchanged external LoRA tensor: {key}"
                    )
                received[key] = output.narrow(0, offset, meta.numel).view(meta.shape)
                offset += meta.numel
        if offset != output.numel():
            raise RuntimeError("external LoRA exchange did not consume its output")
    return received


def _convert_owned_blocks(
    regular: Sequence[LoraShardMeta],
    packed: Sequence[PackedExpertShardMeta],
    regular_tensors: dict[tuple[int, str], torch.Tensor],
    packed_tensors: dict[tuple[int, str], torch.Tensor],
    block_owners: Mapping[str, int],
    *,
    rank: int,
    handler: Any,
    adapter_config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    output: dict[str, torch.Tensor] = {}
    configs = []
    for block in sorted(key for key, owner in block_owners.items() if owner == rank):
        entries: dict[str, list[tuple[dict[str, Any], torch.Tensor]]] = defaultdict(
            list
        )
        for meta in regular:
            if _metadata_block(meta) == block:
                entries[meta.key].append(
                    (meta.manifest, regular_tensors[(meta.owner_rank, meta.key)])
                )
        merged = merge_sharded_adapter_entries(dict(entries))
        block_packed = [meta for meta in packed if _metadata_block(meta) == block]
        if block_packed:
            packed_output = merge_packed_expert_adapter_entries(
                block_packed, packed_tensors
            )
            if set(merged).intersection(packed_output):
                raise RuntimeError(
                    "packed external LoRA tensors overlap regular tensors"
                )
            merged.update(packed_output)
        converted, config = handler.to_vllm_lora_tensors(
            merged, adapter_config=dict(adapter_config)
        )
        if set(output).intersection(converted):
            raise RuntimeError("external LoRA conversion produced duplicate tensors")
        if any(not tensor.is_contiguous() for tensor in converted.values()):
            raise RuntimeError("external LoRA conversion produced a strided tensor")
        output.update(converted)
        configs.append(config)
    return output, configs


def _published_config(
    local_configs: list[dict[str, Any]],
    *,
    handler: Any,
    adapter_config: dict[str, Any],
    world_size: int,
    group: Any | None,
) -> dict[str, Any]:
    gathered = _gather_values(local_configs, world_size, group)
    unique = {
        _encode_adapter_config(config): config
        for rank_configs in gathered
        for config in rank_configs
    }
    baseline = _encode_adapter_config(adapter_config)
    declared = handler.to_vllm_lora_config(dict(adapter_config))
    declared_bytes = _encode_adapter_config(declared)
    changed = {key: value for key, value in unique.items() if key != baseline}
    if declared_bytes != baseline:
        if set(changed).difference({declared_bytes}):
            raise RuntimeError("external LoRA blocks produced inconsistent configs")
        selected = declared
    elif len(changed) == 1:
        selected = next(iter(changed.values()))
    elif changed:
        raise RuntimeError("external LoRA blocks produced multiple configs")
    else:
        selected = adapter_config
    return {**selected, ART_LORA_FORMAT_CONFIG_KEY: ART_LORA_FORMAT_VLLM}


def _gather_output_metadata(
    tensors: Mapping[str, torch.Tensor],
    *,
    rank: int,
    world_size: int,
    group: Any | None,
) -> tuple[ExternalLoraTensor, ...]:
    local = [
        ExternalLoraTensor(
            name=name,
            owner_rank=rank,
            shape=tuple(int(dimension) for dimension in tensor.shape),
            dtype_name=str(tensor.dtype).removeprefix("torch."),
            size_bytes=tensor.nbytes,
        )
        for name, tensor in sorted(tensors.items())
    ]
    gathered = _gather_values(local, world_size, group)
    result = tuple(
        sorted(
            (tensor for values in gathered for tensor in values),
            key=lambda tensor: tensor.name,
        )
    )
    if len({tensor.name for tensor in result}) != len(result):
        raise RuntimeError("external LoRA output tensor names are not unique")
    return result


def _safetensors_header(tensors: Sequence[ExternalLoraTensor]) -> bytes:
    if sys.byteorder != "little":
        raise RuntimeError("external LoRA safetensors require little endian")
    offset = 0
    header = {}
    for tensor in tensors:
        dtype = _dtype_from_name(tensor.dtype_name)
        safetensors_dtype = _SAFETENSORS_DTYPES.get(dtype)
        if safetensors_dtype is None:
            raise RuntimeError(f"unsupported external safetensors dtype: {dtype}")
        if (
            tensor.size_bytes
            != math.prod(tensor.shape) * torch.empty((), dtype=dtype).element_size()
        ):
            raise RuntimeError(
                f"external LoRA tensor byte count changed: {tensor.name}"
            )
        header[tensor.name] = {
            "dtype": safetensors_dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.size_bytes],
        }
        offset += tensor.size_bytes
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded


def _build_plan(
    target: ExternalLoraTarget,
    tensors: tuple[ExternalLoraTensor, ...],
    adapter_config: bytes,
    safetensors_header: bytes,
    *,
    handler_key: str,
    source_topology: str,
    coordinator_rank: int,
    world_size: int,
) -> ExternalLoraPlan:
    regions: dict[ExternalLoraPath, list[tuple[int, ExternalLoraSourceSegment]]] = {
        "adapter_config.json": [
            (
                coordinator_rank,
                ExternalLoraSourceSegment(
                    source="adapter_config",
                    source_offset=0,
                    size_bytes=len(adapter_config),
                ),
            )
        ],
        "adapter_model.safetensors": [
            (
                coordinator_rank,
                ExternalLoraSourceSegment(
                    source="safetensors_header",
                    source_offset=0,
                    size_bytes=len(safetensors_header),
                ),
            ),
            *(
                (
                    tensor.owner_rank,
                    ExternalLoraSourceSegment(
                        source="tensor",
                        tensor_name=tensor.name,
                        source_offset=0,
                        size_bytes=tensor.size_bytes,
                    ),
                )
                for tensor in tensors
            ),
        ],
    }
    files = tuple(
        ExternalLoraFile(
            relative_path=path,
            size_bytes=sum(segment.size_bytes for _owner, segment in values),
        )
        for path, values in sorted(regions.items())
    )
    shards: list[ExternalLoraShardPlan] = []
    for path, values in sorted(regions.items()):
        file_offset = 0
        owner: int | None = None
        shard_offset = 0
        size = 0
        segments: list[ExternalLoraSourceSegment] = []

        def flush() -> None:
            nonlocal owner, shard_offset, size, segments
            if owner is None:
                return
            shards.append(
                ExternalLoraShardPlan(
                    index=len(shards),
                    owner_rank=owner,
                    relative_path=path,
                    file_offset=shard_offset,
                    size_bytes=size,
                    segments=tuple(segments),
                )
            )
            owner = None
            size = 0
            segments = []

        for region_owner, region in values:
            consumed = 0
            while consumed < region.size_bytes:
                if owner is not None and (
                    owner != region_owner or size == target.shard_bytes
                ):
                    flush()
                if owner is None:
                    owner = region_owner
                    shard_offset = file_offset
                count = min(region.size_bytes - consumed, target.shard_bytes - size)
                segments.append(
                    region.model_copy(
                        update={
                            "source_offset": region.source_offset + consumed,
                            "size_bytes": count,
                        }
                    )
                )
                size += count
                consumed += count
                file_offset += count
        flush()
    return ExternalLoraPlan(
        target=target,
        handler_key=handler_key,
        source_topology=source_topology,
        coordinator_rank=coordinator_rank,
        world_size=world_size,
        files=files,
        tensors=tensors,
        shards=tuple(shards),
    )


def _gather_values(value: Any, world_size: int, group: Any | None) -> list[Any]:
    if world_size == 1:
        return [value]
    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, value, group=group)  # type: ignore[possibly-missing-attribute]
    return gathered


def _synchronize_local(
    operation: str,
    call: Callable[[], _T],
    *,
    world_size: int,
    group: Any | None,
) -> _T:
    result: _T | object = _MISSING
    error = None
    try:
        result = call()
    except BaseException as caught:
        error = _error_text(caught)
    errors = _gather_values(error, world_size, group)
    if any(errors):
        raise RuntimeError(
            f"external LoRA {operation} failed: "
            + "; ".join(item for item in errors if item)
        )
    if result is _MISSING:
        raise AssertionError(f"external LoRA {operation} returned no result")
    return cast(_T, result)


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _encode_adapter_config(value: Any) -> bytes:
    return _canonical_bytes(_jsonable(value))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
