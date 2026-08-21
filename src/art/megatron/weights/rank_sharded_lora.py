from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

if TYPE_CHECKING:
    from art.megatron.weights.rank_distributed_lora_publish import (
        PreparedRankDistributedLoraSource,
    )

RANK_SHARDED_LORA_MANIFEST = "adapter_manifest.json"
RANK_SHARDED_LORA_MAX_SHARD_BYTES = 64 << 20
RANK_SHARDED_LORA_FORMAT = "art_rank_sharded_lora_v1"
_LOGICAL_ADAPTER_PATH = "adapter_model.safetensors"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_GENERATION_PATTERN = r"^step-(\d{8,})-[0-9a-f]{32}$"
_SHARD_PATTERN = re.compile(r"^shards/(\d{8})\.bin$")
_PUBLICATION_ACK = ".optimizer-published.json"


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CheckpointFile(_Record):
    name: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _safe_relative_path(self) -> "CheckpointFile":
        path = PurePosixPath(self.name)
        if (
            path.is_absolute()
            or str(path) != self.name
            or self.name in {"", "."}
            or ".." in path.parts
            or "\\" in self.name
            or "\0" in self.name
        ):
            raise ValueError("checkpoint file name must be a normalized relative path")
        return self


class RankShardedLoraFileIdentity(_Record):
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=_SHA256_PATTERN)


class RankShardedLoraLogicalFile(_Record):
    path: Literal["adapter_model.safetensors"] = _LOGICAL_ADAPTER_PATH
    size_bytes: int = Field(gt=0)


class RankShardedLoraShard(_Record):
    path: str
    logical_offset: int = Field(ge=0)
    size_bytes: int = Field(gt=0, le=RANK_SHARDED_LORA_MAX_SHARD_BYTES)
    owner_rank: int = Field(ge=0)
    sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _safe_shard_path(self) -> "RankShardedLoraShard":
        if _SHARD_PATTERN.fullmatch(self.path) is None:
            raise ValueError("rank-sharded LoRA shard path is not canonical")
        return self


class RankShardedLoraManifest(_Record):
    format: Literal["art_rank_sharded_lora_v1"] = RANK_SHARDED_LORA_FORMAT
    training_session_id: str = Field(min_length=1)
    generation_id: str = Field(pattern=_GENERATION_PATTERN)
    step: int = Field(ge=0)
    world_size: int = Field(gt=0)
    adapter_config: RankShardedLoraFileIdentity
    logical_file: RankShardedLoraLogicalFile
    shards: tuple[RankShardedLoraShard, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_layout(self) -> "RankShardedLoraManifest":
        match = re.fullmatch(_GENERATION_PATTERN, self.generation_id)
        if match is None or int(match.group(1)) != self.step:
            raise ValueError("adapter generation ID and policy step must match")
        cursor = 0
        previous: RankShardedLoraShard | None = None
        for index, shard in enumerate(self.shards):
            if shard.path != _shard_path(index):
                raise ValueError("rank-sharded LoRA shard names are not deterministic")
            if shard.owner_rank >= self.world_size:
                raise ValueError("rank-sharded LoRA shard owner leaves its world")
            if shard.logical_offset != cursor:
                raise ValueError("rank-sharded LoRA shard coverage has a gap or overlap")
            if (
                previous is not None
                and previous.owner_rank == shard.owner_rank
                and previous.size_bytes < RANK_SHARDED_LORA_MAX_SHARD_BYTES
            ):
                raise ValueError("rank-sharded LoRA split is not canonical")
            cursor += shard.size_bytes
            previous = shard
        if cursor != self.logical_file.size_bytes:
            raise ValueError("rank-sharded LoRA shard coverage is incomplete")
        return self


class RankShardedLoraTensor(_Record):
    name: str = Field(min_length=1)
    owner_rank: int = Field(ge=0)
    shape: tuple[int, ...]
    dtype_name: str = Field(min_length=1)
    byte_count: int = Field(gt=0)


class RankShardedLoraSourceSegment(_Record):
    source: Literal["safetensors_header", "tensor"]
    tensor_name: str | None = None
    source_offset: int = Field(ge=0)
    size_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_source(self) -> "RankShardedLoraSourceSegment":
        if (self.source == "tensor") != (self.tensor_name is not None):
            raise ValueError("only tensor shard segments identify a tensor")
        return self


class RankShardedLoraPlannedShard(_Record):
    path: str
    logical_offset: int = Field(ge=0)
    size_bytes: int = Field(gt=0, le=RANK_SHARDED_LORA_MAX_SHARD_BYTES)
    owner_rank: int = Field(ge=0)
    segments: tuple[RankShardedLoraSourceSegment, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_segments(self) -> "RankShardedLoraPlannedShard":
        if _SHARD_PATTERN.fullmatch(self.path) is None:
            raise ValueError("planned LoRA shard path is not canonical")
        if sum(segment.size_bytes for segment in self.segments) != self.size_bytes:
            raise ValueError("planned LoRA shard segments have incomplete coverage")
        return self


class PreparedRankShardedLora(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    training_session_id: str = Field(min_length=1)
    generation_id: str = Field(pattern=_GENERATION_PATTERN)
    step: int = Field(ge=0)
    coordinator_rank: int = Field(ge=0)
    world_size: int = Field(gt=0)
    rank: int = Field(ge=0)
    tensors: tuple[RankShardedLoraTensor, ...] = Field(min_length=1)
    local_tensors: dict[str, torch.Tensor]
    adapter_config: bytes = Field(min_length=1)
    safetensors_header: bytes = Field(min_length=1)
    shards: tuple[RankShardedLoraPlannedShard, ...] = Field(min_length=1)
    plan_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_plan(self) -> "PreparedRankShardedLora":
        if self.rank >= self.world_size or self.coordinator_rank >= self.world_size:
            raise ValueError("rank-sharded LoRA rank leaves its world")
        if tuple(tensor.name for tensor in self.tensors) != tuple(
            sorted(tensor.name for tensor in self.tensors)
        ) or len({tensor.name for tensor in self.tensors}) != len(self.tensors):
            raise ValueError("rank-sharded LoRA tensors are not canonically ordered")
        if any(tensor.owner_rank >= self.world_size for tensor in self.tensors):
            raise ValueError("rank-sharded LoRA tensor owner leaves its world")
        expected_local = {
            tensor.name for tensor in self.tensors if tensor.owner_rank == self.rank
        }
        if set(self.local_tensors) != expected_local:
            raise ValueError("rank-sharded LoRA local tensor ownership changed")
        _validate_planned_coverage(self)
        if self.plan_sha256 != _prepared_plan_sha256(self):
            raise ValueError("rank-sharded LoRA plan identity changed")
        return self


class RankShardedLoraRankWrite(_Record):
    rank: int = Field(ge=0)
    plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    shards: tuple[RankShardedLoraShard, ...]


class RankShardedLoraCheckpoint(_Record):
    manifest: RankShardedLoraManifest
    files: tuple[CheckpointFile, ...]

    @model_validator(mode="after")
    def _validate_files(self) -> "RankShardedLoraCheckpoint":
        if self.files != rank_sharded_lora_checkpoint_files(self.manifest):
            raise ValueError("rank-sharded LoRA physical file coverage changed")
        return self


def prepare_rank_sharded_lora(
    source: PreparedRankDistributedLoraSource,
    *,
    training_session_id: str,
    generation_id: str,
    step: int,
) -> PreparedRankShardedLora:
    tensors = tuple(
        RankShardedLoraTensor(
            name=tensor.name,
            owner_rank=tensor.owner_rank,
            shape=tensor.shape,
            dtype_name=tensor.dtype_name,
            byte_count=tensor.byte_count,
        )
        for tensor in source.metadata
    )
    shards = _plan_shards(
        tensors,
        source.safetensors_header,
        coordinator_rank=source.coordinator_rank,
    )
    values: dict[str, Any] = {
        "training_session_id": training_session_id,
        "generation_id": generation_id,
        "step": step,
        "coordinator_rank": source.coordinator_rank,
        "world_size": source.world_size,
        "rank": source.rank,
        "tensors": tensors,
        "local_tensors": source.tensors,
        "adapter_config": source.adapter_config,
        "safetensors_header": source.safetensors_header,
        "shards": shards,
    }
    provisional = PreparedRankShardedLora.model_construct(
        **values, plan_sha256="0" * 64
    )
    return PreparedRankShardedLora(
        **values, plan_sha256=_prepared_plan_sha256(provisional)
    )


def write_rank_sharded_lora_owned(
    prepared: PreparedRankShardedLora,
    staging_path: str | Path,
) -> RankShardedLoraRankWrite:
    root = Path(staging_path).absolute()
    shard_root = root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    metadata = {tensor.name: tensor for tensor in prepared.tensors}
    buffers = {
        name: _validated_tensor_buffer(tensor, metadata[name])
        for name, tensor in prepared.local_tensors.items()
    }
    written: list[RankShardedLoraShard] = []
    for shard in prepared.shards:
        if shard.owner_rank != prepared.rank:
            continue
        path = root / shard.path
        digest = hashlib.sha256()
        size = 0
        with path.open("xb", buffering=0) as output:
            for segment in shard.segments:
                source = (
                    memoryview(prepared.safetensors_header)
                    if segment.source == "safetensors_header"
                    else buffers[segment.tensor_name or ""]
                )
                chunk = source[
                    segment.source_offset : segment.source_offset + segment.size_bytes
                ]
                _write_all(output.fileno(), chunk)
                digest.update(chunk)
                size += chunk.nbytes
            os.fsync(output.fileno())
        if size != shard.size_bytes:
            raise RuntimeError(f"rank-sharded LoRA shard write was short: {path}")
        written.append(
            RankShardedLoraShard(
                path=shard.path,
                logical_offset=shard.logical_offset,
                size_bytes=size,
                owner_rank=prepared.rank,
                sha256=digest.hexdigest(),
            )
        )
    _fsync_directory(shard_root)
    return RankShardedLoraRankWrite(
        rank=prepared.rank,
        plan_sha256=prepared.plan_sha256,
        shards=tuple(written),
    )


def gather_rank_sharded_lora_checkpoint(
    prepared: PreparedRankShardedLora,
    local_write: RankShardedLoraRankWrite,
    *,
    group: Any | None = None,
) -> RankShardedLoraCheckpoint:
    gathered = _all_gather(
        prepared,
        {"write": local_write.model_dump(mode="json"), "error": None},
        group,
    )
    return _checkpoint_from_gathered(prepared, gathered)


def write_rank_sharded_lora_metadata(
    prepared: PreparedRankShardedLora,
    checkpoint: RankShardedLoraCheckpoint,
    staging_path: str | Path,
) -> None:
    if prepared.rank != prepared.coordinator_rank:
        raise RuntimeError("only the LoRA coordinator may write checkpoint metadata")
    expected = _checkpoint_from_shards(
        prepared,
        checkpoint.manifest.shards,
    )
    if checkpoint != expected:
        raise RuntimeError("rank-sharded LoRA checkpoint differs from its plan")
    root = Path(staging_path).absolute()
    _write_file(root / "adapter_config.json", prepared.adapter_config)
    _write_file(
        root / RANK_SHARDED_LORA_MANIFEST,
        encode_rank_sharded_lora_manifest(checkpoint.manifest),
    )
    current = read_rank_sharded_lora_checkpoint(root, verify_files=False)
    if current != checkpoint:
        raise RuntimeError("rank-sharded LoRA metadata changed after write")
    _fsync_directory(root)


def write_rank_sharded_lora_checkpoint(
    prepared: PreparedRankShardedLora,
    staging_path: str | Path,
    *,
    group: Any | None = None,
) -> RankShardedLoraCheckpoint:
    local_write: RankShardedLoraRankWrite | None = None
    local_error: BaseException | None = None
    try:
        local_write = write_rank_sharded_lora_owned(prepared, staging_path)
    except BaseException as error:
        local_error = error
    gathered = _all_gather(
        prepared,
        {
            "write": (
                None if local_write is None else local_write.model_dump(mode="json")
            ),
            "error": _error_text(local_error),
        },
        group,
    )
    checkpoint = _checkpoint_from_gathered(prepared, gathered)
    metadata_error: BaseException | None = None
    if prepared.rank == prepared.coordinator_rank:
        try:
            write_rank_sharded_lora_metadata(prepared, checkpoint, staging_path)
        except BaseException as error:
            metadata_error = error
    completions = _all_gather(
        prepared,
        {"rank": prepared.rank, "error": _error_text(metadata_error)},
        group,
    )
    _raise_gathered_errors(completions, "rank-sharded LoRA metadata write failed")
    return checkpoint


def encode_rank_sharded_lora_manifest(manifest: RankShardedLoraManifest) -> bytes:
    return (
        json.dumps(manifest.model_dump(mode="json"), separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def rank_sharded_lora_checkpoint_files(
    manifest: RankShardedLoraManifest,
) -> tuple[CheckpointFile, ...]:
    manifest_bytes = encode_rank_sharded_lora_manifest(manifest)
    return (
        CheckpointFile(
            name="adapter_config.json",
            size_bytes=manifest.adapter_config.size_bytes,
            sha256=manifest.adapter_config.sha256,
        ),
        CheckpointFile(
            name=RANK_SHARDED_LORA_MANIFEST,
            size_bytes=len(manifest_bytes),
            sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        ),
        *(
            CheckpointFile(
                name=shard.path,
                size_bytes=shard.size_bytes,
                sha256=shard.sha256,
            )
            for shard in manifest.shards
        ),
    )


def read_rank_sharded_lora_manifest(
    path: str | Path,
    *,
    verify_files: bool = False,
) -> RankShardedLoraManifest:
    return read_rank_sharded_lora_checkpoint(
        path, verify_files=verify_files
    ).manifest


def read_rank_sharded_lora_checkpoint(
    path: str | Path,
    *,
    verify_files: bool = False,
) -> RankShardedLoraCheckpoint:
    root = Path(path).absolute()
    manifest_path = root / RANK_SHARDED_LORA_MANIFEST
    if manifest_path.is_symlink():
        raise RuntimeError(f"rank-sharded LoRA manifest is a symlink: {manifest_path}")
    try:
        payload = manifest_path.read_bytes()
        manifest = RankShardedLoraManifest.model_validate_json(payload)
    except Exception as error:
        raise RuntimeError(f"invalid rank-sharded LoRA manifest: {manifest_path}") from error
    if payload != encode_rank_sharded_lora_manifest(manifest):
        raise RuntimeError("rank-sharded LoRA manifest encoding is not canonical")
    files = rank_sharded_lora_checkpoint_files(manifest)
    _validate_physical_coverage(root, files)
    for record in files:
        physical = root / record.name
        if not physical.is_file() or physical.is_symlink():
            raise RuntimeError(f"rank-sharded LoRA file is missing: {physical}")
        if physical.stat().st_size != record.size_bytes:
            raise RuntimeError(f"rank-sharded LoRA file size differs: {physical}")
        if verify_files and _file_sha256(physical) != record.sha256:
            raise RuntimeError(f"rank-sharded LoRA file digest differs: {physical}")
    return RankShardedLoraCheckpoint(manifest=manifest, files=files)


def detect_adapter_checkpoint_format(
    path: str | Path,
) -> Literal["canonical", "rank_sharded"]:
    root = Path(path)
    config = (root / "adapter_config.json").is_file()
    canonical = (root / _LOGICAL_ADAPTER_PATH).is_file()
    manifest = (root / RANK_SHARDED_LORA_MANIFEST).is_file()
    shards = (root / "shards").is_dir()
    if config and canonical and not manifest and not shards:
        return "canonical"
    if config and manifest and shards and not canonical:
        return "rank_sharded"
    raise RuntimeError(f"adapter checkpoint physical format is invalid: {root}")


class RankShardedLoraReader:
    def __init__(
        self,
        path: str | Path,
        manifest: RankShardedLoraManifest | None = None,
    ) -> None:
        self.root = Path(path).absolute()
        self.manifest = manifest or read_rank_sharded_lora_manifest(self.root)
        self._offsets = tuple(shard.logical_offset for shard in self.manifest.shards)
        self._cached_index: int | None = None
        self._cached_payload: bytes | None = None

    def read(self, offset: int, size_bytes: int) -> bytes:
        end = offset + size_bytes
        if offset < 0 or size_bytes < 0 or end > self.manifest.logical_file.size_bytes:
            raise ValueError("rank-sharded LoRA logical read is out of bounds")
        output = bytearray(size_bytes)
        cursor = offset
        output_offset = 0
        while cursor < end:
            index = bisect_right(self._offsets, cursor) - 1
            if index < 0:
                raise RuntimeError("rank-sharded LoRA logical coverage has a gap")
            shard = self.manifest.shards[index]
            shard_end = shard.logical_offset + shard.size_bytes
            count = min(end, shard_end) - cursor
            if count <= 0:
                raise RuntimeError("rank-sharded LoRA logical coverage has an overlap")
            payload = self._load_shard(index)
            start = cursor - shard.logical_offset
            output[output_offset : output_offset + count] = payload[start : start + count]
            cursor += count
            output_offset += count
        return bytes(output)

    def _load_shard(self, index: int) -> bytes:
        if self._cached_index == index:
            assert self._cached_payload is not None
            return self._cached_payload
        shard = self.manifest.shards[index]
        path = self.root / shard.path
        with path.open("rb", buffering=0) as source:
            payload = source.read(shard.size_bytes + 1)
        if len(payload) != shard.size_bytes:
            raise RuntimeError(f"rank-sharded LoRA shard size differs: {path}")
        if hashlib.sha256(payload).hexdigest() != shard.sha256:
            raise RuntimeError(f"rank-sharded LoRA shard digest differs: {path}")
        self._cached_index = index
        self._cached_payload = payload
        return payload


def load_rank_sharded_lora_tensors(
    path: str | Path,
) -> dict[str, torch.Tensor]:
    checkpoint = read_rank_sharded_lora_checkpoint(path, verify_files=False)
    reader = RankShardedLoraReader(path, checkpoint.manifest)
    header_size = int.from_bytes(reader.read(0, 8), "little")
    data_start = 8 + header_size
    if header_size == 0 or data_start > checkpoint.manifest.logical_file.size_bytes:
        raise RuntimeError("rank-sharded LoRA safetensors header is invalid")
    header = _load_unique_json(reader.read(8, header_size))
    if not isinstance(header, dict):
        raise RuntimeError("rank-sharded LoRA safetensors header is not an object")
    entries: list[tuple[int, str, torch.dtype, tuple[int, ...], int]] = []
    for name, value in header.items():
        if name == "__metadata__":
            if not isinstance(value, dict):
                raise RuntimeError("rank-sharded LoRA safetensors metadata is invalid")
            continue
        if not isinstance(name, str) or not isinstance(value, dict):
            raise RuntimeError("rank-sharded LoRA safetensors tensor entry is invalid")
        try:
            if set(value) != {"dtype", "shape", "data_offsets"}:
                raise ValueError("unexpected tensor metadata fields")
            dtype = _safetensors_dtype(value["dtype"])
            shape = _integer_tuple(value["shape"])
            offsets = _integer_tuple(value["data_offsets"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"rank-sharded LoRA safetensors metadata is invalid: {name}"
            ) from error
        if (
            len(offsets) != 2
            or offsets[0] < 0
            or offsets[1] <= offsets[0]
            or any(dim < 0 for dim in shape)
        ):
            raise RuntimeError(
                f"rank-sharded LoRA safetensors offsets are invalid: {name}"
            )
        size = offsets[1] - offsets[0]
        if size != math.prod(shape) * torch.empty((), dtype=dtype).element_size():
            raise RuntimeError(
                f"rank-sharded LoRA safetensors tensor size is invalid: {name}"
            )
        entries.append((offsets[0], name, dtype, shape, size))
    entries.sort()
    cursor = 0
    tensors: dict[str, torch.Tensor] = {}
    for offset, name, dtype, shape, size in entries:
        if offset != cursor:
            raise RuntimeError("rank-sharded LoRA safetensors data has a gap or overlap")
        payload = bytearray(reader.read(data_start + offset, size))
        tensors[name] = torch.frombuffer(payload, dtype=dtype).reshape(shape)
        cursor += size
    if data_start + cursor != checkpoint.manifest.logical_file.size_bytes:
        raise RuntimeError("rank-sharded LoRA safetensors data coverage is incomplete")
    return tensors


def _plan_shards(
    tensors: tuple[RankShardedLoraTensor, ...],
    header: bytes,
    *,
    coordinator_rank: int,
) -> tuple[RankShardedLoraPlannedShard, ...]:
    regions = [
        (
            coordinator_rank,
            RankShardedLoraSourceSegment(
                source="safetensors_header",
                source_offset=0,
                size_bytes=len(header),
            ),
        ),
        *(
            (
                tensor.owner_rank,
                RankShardedLoraSourceSegment(
                    source="tensor",
                    tensor_name=tensor.name,
                    source_offset=0,
                    size_bytes=tensor.byte_count,
                ),
            )
            for tensor in tensors
        ),
    ]
    shards: list[RankShardedLoraPlannedShard] = []
    logical_offset = 0
    shard_owner: int | None = None
    shard_offset = 0
    shard_size = 0
    segments: list[RankShardedLoraSourceSegment] = []

    def flush() -> None:
        nonlocal shard_owner, shard_offset, shard_size, segments
        if shard_owner is None:
            return
        shards.append(
            RankShardedLoraPlannedShard(
                path=_shard_path(len(shards)),
                logical_offset=shard_offset,
                size_bytes=shard_size,
                owner_rank=shard_owner,
                segments=tuple(segments),
            )
        )
        shard_owner = None
        shard_size = 0
        segments = []

    for owner, region in regions:
        consumed = 0
        while consumed < region.size_bytes:
            if shard_owner != owner or shard_size == RANK_SHARDED_LORA_MAX_SHARD_BYTES:
                flush()
                shard_owner = owner
                shard_offset = logical_offset
            count = min(
                region.size_bytes - consumed,
                RANK_SHARDED_LORA_MAX_SHARD_BYTES - shard_size,
            )
            segments.append(
                region.model_copy(
                    update={
                        "source_offset": region.source_offset + consumed,
                        "size_bytes": count,
                    }
                )
            )
            consumed += count
            shard_size += count
            logical_offset += count
    flush()
    return tuple(shards)


def _validate_planned_coverage(prepared: PreparedRankShardedLora) -> None:
    tensors = {tensor.name: tensor for tensor in prepared.tensors}
    cursor = 0
    consumed: dict[tuple[str, str | None], int] = {}
    previous: RankShardedLoraPlannedShard | None = None
    for index, shard in enumerate(prepared.shards):
        if shard.path != _shard_path(index) or shard.logical_offset != cursor:
            raise ValueError("planned rank-sharded LoRA coverage is not canonical")
        if shard.owner_rank >= prepared.world_size:
            raise ValueError("planned rank-sharded LoRA owner leaves its world")
        if (
            previous is not None
            and previous.owner_rank == shard.owner_rank
            and previous.size_bytes < RANK_SHARDED_LORA_MAX_SHARD_BYTES
        ):
            raise ValueError("planned rank-sharded LoRA split is not canonical")
        for segment in shard.segments:
            key = (segment.source, segment.tensor_name)
            if segment.source_offset != consumed.get(key, 0):
                raise ValueError("planned rank-sharded LoRA source coverage changed")
            if segment.source == "tensor":
                tensor = tensors.get(segment.tensor_name or "")
                if tensor is None:
                    raise ValueError("planned LoRA shard names an unknown tensor")
                if tensor.owner_rank != shard.owner_rank:
                    raise ValueError("planned LoRA tensor crosses an owner boundary")
            elif shard.owner_rank != prepared.coordinator_rank:
                raise ValueError("planned LoRA header is not coordinator-owned")
            consumed[key] = segment.source_offset + segment.size_bytes
        cursor += shard.size_bytes
        previous = shard
    expected = {("safetensors_header", None): len(prepared.safetensors_header)}
    expected.update(
        {("tensor", name): tensor.byte_count for name, tensor in tensors.items()}
    )
    if consumed != expected:
        raise ValueError("planned rank-sharded LoRA source coverage is incomplete")


def _prepared_plan_sha256(prepared: PreparedRankShardedLora) -> str:
    payload = {
        "training_session_id": prepared.training_session_id,
        "generation_id": prepared.generation_id,
        "step": prepared.step,
        "coordinator_rank": prepared.coordinator_rank,
        "world_size": prepared.world_size,
        "tensors": [tensor.model_dump(mode="json") for tensor in prepared.tensors],
        "adapter_config": hashlib.sha256(prepared.adapter_config).hexdigest(),
        "safetensors_header": hashlib.sha256(prepared.safetensors_header).hexdigest(),
        "shards": [shard.model_dump(mode="json") for shard in prepared.shards],
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _checkpoint_from_gathered(
    prepared: PreparedRankShardedLora,
    gathered: list[Any],
) -> RankShardedLoraCheckpoint:
    _raise_gathered_errors(gathered, "rank-sharded LoRA shard write failed")
    writes = tuple(
        RankShardedLoraRankWrite.model_validate(item["write"]) for item in gathered
    )
    if tuple(write.rank for write in writes) != tuple(range(prepared.world_size)):
        raise RuntimeError("rank-sharded LoRA metadata gather changed rank order")
    if any(write.plan_sha256 != prepared.plan_sha256 for write in writes):
        raise RuntimeError("rank-sharded LoRA ranks used different plans")
    for write in writes:
        expected = {
            shard.path for shard in prepared.shards if shard.owner_rank == write.rank
        }
        if {shard.path for shard in write.shards} != expected or any(
            shard.owner_rank != write.rank for shard in write.shards
        ):
            raise RuntimeError("rank-sharded LoRA rank wrote another owner's shards")
    shards = tuple(shard for write in writes for shard in write.shards)
    return _checkpoint_from_shards(prepared, shards)


def _checkpoint_from_shards(
    prepared: PreparedRankShardedLora,
    shards: tuple[RankShardedLoraShard, ...],
) -> RankShardedLoraCheckpoint:
    by_path = {shard.path: shard for shard in shards}
    if len(by_path) != len(shards) or set(by_path) != {
        shard.path for shard in prepared.shards
    }:
        raise RuntimeError("rank-sharded LoRA writes have incomplete shard coverage")
    ordered = tuple(by_path[planned.path] for planned in prepared.shards)
    if any(
        (
            shard.logical_offset,
            shard.size_bytes,
            shard.owner_rank,
        )
        != (
            planned.logical_offset,
            planned.size_bytes,
            planned.owner_rank,
        )
        for shard, planned in zip(ordered, prepared.shards, strict=True)
    ):
        raise RuntimeError("rank-sharded LoRA write differs from its plan")
    manifest = RankShardedLoraManifest(
        training_session_id=prepared.training_session_id,
        generation_id=prepared.generation_id,
        step=prepared.step,
        world_size=prepared.world_size,
        adapter_config=RankShardedLoraFileIdentity(
            size_bytes=len(prepared.adapter_config),
            sha256=hashlib.sha256(prepared.adapter_config).hexdigest(),
        ),
        logical_file=RankShardedLoraLogicalFile(
            size_bytes=len(prepared.safetensors_header)
            + sum(tensor.byte_count for tensor in prepared.tensors)
        ),
        shards=ordered,
    )
    return RankShardedLoraCheckpoint(
        manifest=manifest,
        files=rank_sharded_lora_checkpoint_files(manifest),
    )


def _all_gather(
    prepared: PreparedRankShardedLora,
    value: Any,
    group: Any | None,
) -> list[Any]:
    if prepared.world_size == 1:
        if prepared.rank != 0:
            raise RuntimeError("single-rank LoRA checkpoint has a nonzero rank")
        return [value]
    if not torch.distributed.is_initialized():  # type: ignore[possibly-missing-attribute]
        raise RuntimeError("multi-rank LoRA checkpoint requires distributed metadata")
    rank = int(torch.distributed.get_rank(group))  # type: ignore[possibly-missing-attribute]
    world_size = int(torch.distributed.get_world_size(group))  # type: ignore[possibly-missing-attribute]
    backend = str(torch.distributed.get_backend(group))  # type: ignore[possibly-missing-attribute]
    if (rank, world_size) != (prepared.rank, prepared.world_size):
        raise RuntimeError("rank-sharded LoRA metadata group changed rank identity")
    if backend != "gloo":
        raise RuntimeError("rank-sharded LoRA metadata all-gather requires Gloo")
    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, value, group=group)  # type: ignore[possibly-missing-attribute]
    if any(item is None for item in gathered):
        raise RuntimeError("rank-sharded LoRA metadata gather omitted a rank")
    return gathered


def _raise_gathered_errors(gathered: list[Any], message: str) -> None:
    errors = [
        f"rank {rank}: {item.get('error')}"
        for rank, item in enumerate(gathered)
        if isinstance(item, dict) and item.get("error") is not None
    ]
    if errors:
        raise RuntimeError(f"{message}: {'; '.join(errors)}")


def _error_text(error: BaseException | None) -> str | None:
    return None if error is None else f"{type(error).__name__}: {error}"


def _validated_tensor_buffer(
    tensor: torch.Tensor,
    metadata: RankShardedLoraTensor,
) -> memoryview:
    dtype = getattr(torch, metadata.dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"unsupported rank-sharded LoRA dtype: {metadata.dtype_name}")
    if (
        tensor.device.type != "cpu"
        or not tensor.is_contiguous()
        or tensor.dtype != dtype
        or tuple(tensor.shape) != metadata.shape
        or tensor.nbytes != metadata.byte_count
    ):
        raise RuntimeError(f"rank-sharded LoRA tensor changed: {metadata.name}")
    return memoryview(tensor.reshape(-1).view(torch.uint8).numpy())


def _write_all(descriptor: int, payload: memoryview) -> None:
    while payload:
        written = os.write(descriptor, payload)
        if written <= 0:
            raise OSError("short rank-sharded LoRA write")
        payload = payload[written:]


def _write_file(path: Path, payload: bytes) -> None:
    with path.open("xb", buffering=0) as output:
        _write_all(output.fileno(), memoryview(payload))
        os.fsync(output.fileno())


def _validate_physical_coverage(
    root: Path,
    files: tuple[CheckpointFile, ...],
) -> None:
    expected_top = {"adapter_config.json", RANK_SHARDED_LORA_MANIFEST, "shards"}
    actual_top = {
        entry.name for entry in root.iterdir() if entry.name != _PUBLICATION_ACK
    }
    if actual_top != expected_top:
        raise RuntimeError("rank-sharded LoRA top-level coverage is incomplete")
    shard_root = root / "shards"
    expected_shards = {PurePosixPath(file.name).name for file in files[2:]}
    actual_shards = {entry.name for entry in shard_root.iterdir()}
    if actual_shards != expected_shards:
        raise RuntimeError("rank-sharded LoRA physical shard coverage is incomplete")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as value:
        while chunk := value.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _shard_path(index: int) -> str:
    return f"shards/{index:08d}.bin"


def _load_unique_json(payload: bytes) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError("rank-sharded LoRA safetensors header is invalid") from error


def _safetensors_dtype(name: Any) -> torch.dtype:
    dtypes = {
        key: dtype
        for key, dtype in {
            "BOOL": torch.bool,
            "U8": torch.uint8,
            "I8": torch.int8,
            "I16": torch.int16,
            "I32": torch.int32,
            "I64": torch.int64,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "F32": torch.float32,
            "F64": torch.float64,
            "C64": torch.complex64,
            "U16": getattr(torch, "uint16", None),
            "U32": getattr(torch, "uint32", None),
            "U64": getattr(torch, "uint64", None),
            "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
            "F8_E5M2": getattr(torch, "float8_e5m2", None),
        }.items()
        if dtype is not None
    }
    dtype = dtypes.get(name)
    if dtype is None:
        raise RuntimeError(f"unsupported rank-sharded LoRA safetensors dtype: {name!r}")
    return dtype


def _integer_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ValueError("safetensors integer array is invalid")
    return tuple(value)
