from __future__ import annotations

from array import array
from collections.abc import AsyncIterable, AsyncIterator, Iterator, Mapping
import hashlib
import struct
import sys
from typing import Any, Literal, NamedTuple, TypeVar, cast
import uuid

from msgspec import msgpack
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from art.distributed.moe_route_store import (
    MoeRouteGroupPayload,
    MoeRouteObjectPayload,
)
from art.distributed.packing import TrajectoryGroupPayload
from art.distributed.trajectory_store import TrajectoryGroupBundle
from art.megatron.prefix_tree_packing import prefix_tree_pack_segments
from art.preprocessing.moe_routing import (
    COMPLETION_TOKEN_IDS_KEY,
    PROMPT_TOKEN_IDS_KEY,
)
from art.training.contracts import (
    OperationResult,
    RlTrajectoryBatch,
    SupervisedTrajectoryBatch,
    TokenizedTrainingBatch,
    TrainingBatch,
)
from art.training.tokenized import MAX_TOKENIZED_LOGPROB_VALUES, TokenizedMoeRoutes

from .contracts import (
    MAX_RL_GROUPS_PER_BATCH,
    RL_GROUP_DATA_FORMAT,
    SFT_DATA_FORMAT,
    TOKENIZED_DATA_FORMAT,
    OperationResultRef,
    RemoteForwardRequest,
    RemoteRlBatchRef,
    RemoteRlGroupRef,
    RemoteRouteObject,
    RemoteRouteObjectRef,
    RemoteRouteSlice,
    RemoteSftBatchRef,
    RemoteTokenizedBatchRef,
    RemoteTrainingBatchRef,
    TrainingDataRef,
    command_route_object_refs,
    training_data_refs,
)

_SFT_BATCH_ADAPTER = TypeAdapter(SupervisedTrajectoryBatch)
_TOKENIZED_BATCH_ADAPTER = TypeAdapter(TokenizedTrainingBatch)
_UINT32_ARRAY_EXT = 1
ResultT = TypeVar("ResultT", bound=OperationResult)
RouteWireEncoding = Literal["inline", "prefix_tree"]
FORWARD_SUBMISSION_MEDIA_TYPE = "application/vnd.art.forward-framed+msgpack"
FORWARD_SUBMISSION_PREFIX_BYTES = 16
MAX_FORWARD_SUBMISSION_MANIFEST_BYTES = 16 << 20
MAX_FORWARD_SUBMISSION_BYTES = 4 << 30
MAX_FORWARD_SUBMISSION_CHUNKS = 1 << 16
FORWARD_SUBMISSION_CHUNK_BYTES = 1 << 20
_FORWARD_SUBMISSION_MAGIC = b"ARTFWD02"
_FORWARD_SUBMISSION_PREFIX = struct.Struct("!8sQ")


class _RouteEntry(NamedTuple):
    trajectory_index: int
    scope: Literal["messages", "additional_history", "exchange"]
    scope_index: int
    choice_index: int
    route: Any


class _PackedRouteSequences(NamedTuple):
    chunks: tuple[memoryview, ...]
    byte_count: int
    slices: tuple[tuple[tuple[int, int], ...], ...]
    sha256: str | None


class _PackedRouteObject(NamedTuple):
    chunks: tuple[memoryview, ...]
    byte_count: int
    sha256: str
    slices: tuple[RemoteRouteSlice, ...]


class _RouteByteSlice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    segment_index: int
    offset: int
    byte_count: int


class _TokenizedRouteBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    datum_index: int
    num_experts: int
    dtype: Literal["uint8", "uint16"]
    shape: tuple[int, int, int]
    slices: tuple[_RouteByteSlice, ...]


class _TokenizedRoutePool(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data: bytes
    bindings: tuple[_TokenizedRouteBinding, ...]

    @model_validator(mode="after")
    def _validate_layout(self) -> "_TokenizedRoutePool":
        if not self.data or not self.bindings:
            raise ValueError("tokenized route pool cannot be empty")
        if len({value.datum_index for value in self.bindings}) != len(self.bindings):
            raise ValueError("tokenized route pool repeats a datum")
        for binding in self.bindings:
            if tuple(value.segment_index for value in binding.slices) != tuple(
                range(len(binding.slices))
            ):
                raise ValueError("tokenized route slices must be contiguous")
            if any(
                value.byte_count <= 0
                or value.offset < 0
                or value.offset + value.byte_count > len(self.data)
                for value in binding.slices
            ):
                raise ValueError("tokenized route slice leaves its pool bounds")
        return self


class _TokenizedWireBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batch: TokenizedTrainingBatch
    route_pool: _TokenizedRoutePool | None = None

    @model_validator(mode="after")
    def _validate_route_source(self) -> "_TokenizedWireBatch":
        inline = [datum.moe_routes is not None for datum in self.batch.datums]
        if self.route_pool is None:
            return self
        if any(inline):
            raise ValueError("tokenized routes cannot be inline and pooled")
        indices = tuple(value.datum_index for value in self.route_pool.bindings)
        if indices != tuple(range(len(self.batch.datums))):
            raise ValueError("tokenized route pool must cover every datum in order")
        return self


class EncodedTrainingObject(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: TrainingDataRef
    payload: bytes

    @model_validator(mode="after")
    def _validate_payload(self) -> "EncodedTrainingObject":
        _validate_training_object(self.ref, self.payload, self.ref.format)
        return self


class EncodedRouteObject(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    ref: RemoteRouteObjectRef
    chunks: tuple[memoryview, ...]

    @model_validator(mode="after")
    def _validate_payload(self) -> "EncodedRouteObject":
        if self.ref.transport != "command" or self.ref.sha256 is None:
            raise ValueError("encoded route bytes require command transport")
        if _validate_wire_chunks(self.chunks) != self.ref.byte_count:
            raise ValueError("route byte count differs from its reference")
        if self.ref.byte_count > MAX_FORWARD_SUBMISSION_BYTES:
            raise ValueError("route object exceeds the configured limit")
        return self


class EncodedRlGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data: EncodedTrainingObject
    routes: tuple[EncodedRouteObject, ...]
    remote: RemoteRlGroupRef


class DecodedRlGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    bundle: TrajectoryGroupBundle
    routes: MoeRouteGroupPayload


class EncodedTrainingBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    batch: TrainingBatch
    remote: RemoteTrainingBatchRef
    objects: tuple[EncodedTrainingObject, ...]
    route_objects: tuple[EncodedRouteObject, ...] = ()


class ForwardSubmissionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["art_forward_submission_v2"] = "art_forward_submission_v2"
    request: RemoteForwardRequest
    objects: tuple[TrainingDataRef, ...] = Field(max_length=MAX_RL_GROUPS_PER_BATCH)
    route_objects: tuple[RemoteRouteObjectRef, ...] = Field(
        default=(), max_length=MAX_FORWARD_SUBMISSION_CHUNKS
    )

    @model_validator(mode="after")
    def _validate_objects(self) -> "ForwardSubmissionManifest":
        route_refs = {
            ref.object_id: ref for ref in command_route_object_refs(self.request.batch)
        }
        route_data_ids = set(route_refs)
        expected_data = {
            ref.object_id: ref
            for ref in training_data_refs(self.request.batch)
            if ref.object_id not in route_data_ids
        }
        if (
            len(self.objects) != len(expected_data)
            or {value.object_id: value for value in self.objects} != expected_data
        ):
            raise ValueError("forward submission data objects do not match request")
        if {
            value.object_id: value for value in self.route_objects
        } != route_refs or len(self.route_objects) != len(route_refs):
            raise ValueError("forward submission route objects do not match request")
        return self


class EncodedForwardSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    preamble: bytes
    chunks: tuple[memoryview, ...]
    byte_count: int

    @model_validator(mode="after")
    def _validate_size(self) -> "EncodedForwardSubmission":
        if self.byte_count != len(self.preamble) + _validate_wire_chunks(self.chunks):
            raise ValueError("forward submission byte count is inconsistent")
        if self.byte_count > MAX_FORWARD_SUBMISSION_BYTES:
            raise ValueError("forward submission exceeds the configured limit")
        return self

    def stream(self) -> AsyncIterable[bytes]:
        return _ForwardSubmissionStream(self.preamble, self.chunks)


class _ForwardSubmissionStream:
    def __init__(self, preamble: bytes, chunks: tuple[memoryview, ...]) -> None:
        self._preamble = preamble
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[bytes]:
        for chunk in _iter_readonly_chunks(self._preamble):
            yield cast(bytes, chunk)
        for chunk in self._chunks:
            yield cast(bytes, chunk)


def _iter_readonly_chunks(data: bytes | memoryview) -> Iterator[memoryview]:
    view = memoryview(data)
    if not view.c_contiguous:
        raise ValueError("wire chunks must be contiguous")
    view = view.cast("B").toreadonly()
    for offset in range(0, len(view), FORWARD_SUBMISSION_CHUNK_BYTES):
        yield view[offset : offset + FORWARD_SUBMISSION_CHUNK_BYTES]


def _validate_wire_chunks(chunks: tuple[memoryview, ...]) -> int:
    if not chunks or len(chunks) > MAX_FORWARD_SUBMISSION_CHUNKS:
        raise ValueError("wire chunk count is outside the configured bounds")
    byte_count = 0
    for chunk in chunks:
        if (
            not chunk.readonly
            or not chunk.c_contiguous
            or chunk.ndim != 1
            or chunk.format != "B"
            or not 0 < len(chunk) <= FORWARD_SUBMISSION_CHUNK_BYTES
        ):
            raise ValueError("wire chunks must be bounded readonly byte views")
        byte_count += len(chunk)
    return byte_count


def encode_trajectory_group(
    bundle: TrajectoryGroupBundle,
    *,
    object_id: str | None = None,
    route_encoding: RouteWireEncoding = "prefix_tree",
) -> EncodedRlGroup:
    data_id = object_id or _object_id(uuid.uuid4().hex)
    source = bundle.payload()
    if route_encoding == "prefix_tree":
        stripped, route_payload = _extract_routes(source)
    elif route_encoding == "inline":
        stripped, route_payload = source, None
    else:
        raise ValueError(f"unsupported route wire encoding: {route_encoding!r}")
    data = _encode_training_object(
        msgpack.encode(
            TrajectoryGroupBundle.from_payload(stripped).model_dump(mode="python")
        ),
        data_format=RL_GROUP_DATA_FORMAT,
        object_id=data_id,
    )
    routes: tuple[EncodedRouteObject, ...] = ()
    remote_routes: tuple[RemoteRouteObject, ...] = ()
    if route_payload is not None and route_payload.byte_count:
        route_ref = RemoteRouteObjectRef(
            object_id=_object_id(f"{data_id}:{route_payload.sha256}"),
            byte_count=route_payload.byte_count,
            transport="command",
            sha256=route_payload.sha256,
        )
        routes = (EncodedRouteObject(ref=route_ref, chunks=route_payload.chunks),)
        remote_routes = (RemoteRouteObject(ref=route_ref, slices=route_payload.slices),)
    return EncodedRlGroup(
        data=data,
        routes=routes,
        remote=RemoteRlGroupRef(data=data.ref, routes=remote_routes),
    )


def prepare_training_batch(
    batch: TrainingBatch,
    *,
    identity: str | None = None,
    route_encoding: RouteWireEncoding = "prefix_tree",
) -> EncodedTrainingBatch:
    if isinstance(batch, RlTrajectoryBatch):
        annotations = batch.local_group_annotations()
        groups = tuple(
            encode_trajectory_group(
                group,
                object_id=_object_id(f"{identity}:{index}") if identity else None,
                route_encoding=route_encoding,
            )
            for index, group in enumerate(batch.groups)
        )
        objects = tuple(group.data for group in groups)
        route_objects = tuple(route for group in groups for route in group.routes)
        remote: RemoteTrainingBatchRef = RemoteRlBatchRef(
            groups=tuple(
                group.remote.model_copy(
                    update={
                        "annotations": (annotations[index] if annotations else None)
                    }
                )
                for index, group in enumerate(groups)
            ),
            min_source_version=batch.min_source_version,
            max_source_version=batch.max_source_version,
        )
    elif isinstance(batch, SupervisedTrajectoryBatch):
        value = _encode_training_object(
            msgpack.encode(batch.model_dump(mode="python")),
            data_format=SFT_DATA_FORMAT,
            object_id=_object_id(identity) if identity else None,
        )
        objects = (value,)
        route_objects = ()
        remote = RemoteSftBatchRef(data=value.ref)
    else:
        payload = _encode_tokenized_wire_batch(batch, route_encoding=route_encoding)
        value = _encode_training_object(
            payload,
            data_format=TOKENIZED_DATA_FORMAT,
            object_id=_object_id(identity) if identity else None,
        )
        objects = (value,)
        route_objects = ()
        remote = RemoteTokenizedBatchRef(data=value.ref)
    return EncodedTrainingBatch(
        batch=batch,
        remote=remote,
        objects=objects,
        route_objects=route_objects,
    )


def decode_trajectory_group(
    ref: RemoteRlGroupRef,
    payload: bytes,
    *,
    route_payloads: Mapping[str, bytes],
) -> DecodedRlGroup:
    _validate_training_object(ref.data, payload, RL_GROUP_DATA_FORMAT)
    bundle = TrajectoryGroupBundle.model_validate(msgpack.decode(payload))
    return DecodedRlGroup(
        bundle=bundle,
        routes=MoeRouteGroupPayload(
            objects=tuple(
                MoeRouteObjectPayload(
                    data=_route_payload(value.ref.object_id, route_payloads),
                    slices=value.slices,
                )
                for value in ref.routes
            )
        ),
    )


def _extract_routes(
    payload: TrajectoryGroupPayload,
) -> tuple[TrajectoryGroupPayload, _PackedRouteObject]:
    entries: list[_RouteEntry] = []
    trajectories = []
    for trajectory_index, trajectory in enumerate(payload.trajectories):
        histories = tuple(
            _strip_route_map(
                values,
                trajectory_index=trajectory_index,
                scope="additional_history",
                scope_index=index,
                entries=entries,
            )
            for index, values in enumerate(
                trajectory.additional_history_choice_routing_metadata
            )
        )
        exchanges = tuple(
            _strip_route_map(
                values,
                trajectory_index=trajectory_index,
                scope="exchange",
                scope_index=index,
                entries=entries,
            )
            for index, values in enumerate(trajectory.exchange_choice_routing_metadata)
        )
        trajectories.append(
            trajectory.model_copy(
                update={
                    "choice_routing_metadata": _strip_route_map(
                        trajectory.choice_routing_metadata,
                        trajectory_index=trajectory_index,
                        scope="messages",
                        scope_index=0,
                        entries=entries,
                    ),
                    "additional_history_choice_routing_metadata": histories,
                    "exchange_choice_routing_metadata": exchanges,
                }
            )
        )
    return payload.model_copy(
        update={"trajectories": tuple(trajectories)}
    ), _pack_route_entries(entries)


def _strip_route_map(
    values: dict[int, Any],
    *,
    trajectory_index: int,
    scope: Literal["messages", "additional_history", "exchange"],
    scope_index: int,
    entries: list[_RouteEntry],
) -> dict[int, Any]:
    stripped = {}
    for choice_index, route in values.items():
        if not route.data:
            raise RuntimeError("inline routed experts are empty")
        entries.append(
            _RouteEntry(
                trajectory_index,
                scope,
                scope_index,
                choice_index,
                route,
            )
        )
        stripped[choice_index] = route.model_copy(update={"data": ()})
    return stripped


def _pack_route_entries(
    entries: list[_RouteEntry],
) -> _PackedRouteObject:
    packed = _pack_route_sequences(
        tuple((entry.route, _route_token_ids(entry.route)) for entry in entries),
        compute_sha256=True,
    )
    if packed.sha256 is None:
        raise RuntimeError("route object identity was not computed")
    slices = tuple(
        RemoteRouteSlice(
            trajectory_index=entry.trajectory_index,
            scope=entry.scope,
            scope_index=entry.scope_index,
            choice_index=entry.choice_index,
            segment_index=segment_index,
            offset=offset,
            byte_count=byte_count,
        )
        for entry, route_slices in zip(entries, packed.slices, strict=True)
        for segment_index, (offset, byte_count) in enumerate(route_slices)
    )
    return _PackedRouteObject(
        chunks=packed.chunks,
        byte_count=packed.byte_count,
        sha256=packed.sha256,
        slices=slices,
    )


def _pack_route_sequences(
    entries: tuple[tuple[Any, tuple[int, ...]], ...],
    *,
    compute_sha256: bool = False,
) -> _PackedRouteSequences:
    if not entries:
        return _PackedRouteSequences(
            chunks=(),
            byte_count=0,
            slices=(),
            sha256=hashlib.sha256().hexdigest() if compute_sha256 else None,
        )
    token_keys: dict[tuple[int, str, int, int, memoryview], int] = {}
    rows = []
    for route, token_ids in entries:
        tokens = _route_token_views(route)
        rows.append(
            tuple(
                token_keys.setdefault(
                    (token_id, route.dtype, route.shape[1], route.shape[2], token),
                    len(token_keys),
                )
                for token_id, token in zip(token_ids, tokens, strict=True)
            )
        )
    planned = prefix_tree_pack_segments(
        rows,
        max_depth=max(map(len, rows)),
        shareable_lengths=tuple(map(len, rows)),
        min_shared_segment_length=1,
    )
    chunks: list[memoryview] = []
    byte_count = 0
    digest = hashlib.sha256() if compute_sha256 else None
    slices: list[list[tuple[int, int]]] = [[] for _ in entries]
    for segment in planned:
        route = entries[segment.sequence_indices[0]][0]
        bytes_per_token = _route_bytes_per_token(route)
        source_chunks = _route_data_range(
            route.data,
            segment.start * bytes_per_token,
            segment.end * bytes_per_token,
        )
        offset = byte_count
        for source_chunk in source_chunks:
            for chunk in _iter_readonly_chunks(source_chunk):
                chunks.append(chunk)
                byte_count += len(chunk)
                if byte_count > MAX_FORWARD_SUBMISSION_BYTES:
                    raise ValueError("route object exceeds the configured limit")
                if len(chunks) > MAX_FORWARD_SUBMISSION_CHUNKS:
                    raise ValueError("route object has too many wire chunks")
                if digest is not None:
                    digest.update(chunk)
        segment_byte_count = byte_count - offset
        for sequence_index in segment.sequence_indices:
            slices[sequence_index].append((offset, segment_byte_count))
    return _PackedRouteSequences(
        chunks=tuple(chunks),
        byte_count=byte_count,
        slices=tuple(tuple(value) for value in slices),
        sha256=digest.hexdigest() if digest is not None else None,
    )


def _route_token_ids(route: Any) -> tuple[int, ...]:
    prompt = route.metadata.get(PROMPT_TOKEN_IDS_KEY)
    completion = route.metadata.get(COMPLETION_TOKEN_IDS_KEY)
    if not isinstance(prompt, list) or not isinstance(completion, list):
        raise RuntimeError("routed experts are missing exact token ids")
    route_count = route.shape[0]
    completion_count = route_count - len(prompt)
    if completion_count not in {len(completion), max(len(completion) - 1, 0)}:
        raise RuntimeError("routed-expert token ids do not match their route shape")
    token_ids = tuple(prompt) + tuple(completion[:completion_count])
    if any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in token_ids
    ):
        raise RuntimeError("routed-expert token ids do not match their route shape")
    return token_ids


def _route_bytes_per_token(route: Any) -> int:
    return (1 if route.dtype == "uint8" else 2) * route.shape[1] * route.shape[2]


def _route_token_views(route: Any) -> tuple[memoryview, ...]:
    bytes_per_token = _route_bytes_per_token(route)
    tokens = tuple(
        view[offset : offset + bytes_per_token]
        for raw in route.data
        for view in (memoryview(raw).cast("B").toreadonly(),)
        for offset in range(0, len(view), bytes_per_token)
    )
    if len(tokens) != route.shape[0]:
        raise RuntimeError("routed experts do not cover their declared shape")
    return tokens


def _route_data_range(
    data: tuple[bytes | memoryview, ...], start: int, end: int
) -> tuple[memoryview, ...]:
    result = []
    cursor = 0
    for raw in data:
        segment = memoryview(raw).cast("B").toreadonly()
        segment_end = cursor + len(segment)
        overlap_start = max(start, cursor)
        overlap_end = min(end, segment_end)
        if overlap_start < overlap_end:
            result.append(segment[overlap_start - cursor : overlap_end - cursor])
        cursor = segment_end
        if cursor >= end:
            break
    if cursor < end:
        raise RuntimeError("routed experts do not cover their declared shape")
    return tuple(result)


def _route_payload(object_id: str, payloads: Mapping[str, bytes]) -> bytes:
    try:
        return payloads[object_id]
    except KeyError:
        raise RuntimeError(f"route object is unavailable: {object_id}") from None


def decode_sft_batch(ref: TrainingDataRef, payload: bytes) -> SupervisedTrajectoryBatch:
    _validate_training_object(ref, payload, SFT_DATA_FORMAT)
    return _SFT_BATCH_ADAPTER.validate_python(msgpack.decode(payload))


def _encode_tokenized_wire_batch(
    batch: TokenizedTrainingBatch,
    *,
    route_encoding: RouteWireEncoding,
) -> bytes:
    if route_encoding not in {"inline", "prefix_tree"}:
        raise ValueError(f"unsupported route wire encoding: {route_encoding!r}")
    if route_encoding == "inline" or batch.datums[0].moe_routes is None:
        wire = _TokenizedWireBatch(batch=batch)
    elif route_encoding == "prefix_tree":
        entries = tuple(
            (datum.moe_routes, datum.input_tokens) for datum in batch.datums
        )
        if any(route is None for route, _ in entries):
            raise ValueError("tokenized routes must be present for every datum")
        packed = _pack_route_sequences(
            tuple((route, tokens) for route, tokens in entries if route is not None)
        )
        stripped = batch.model_copy(
            update={
                "datums": tuple(
                    datum.model_copy(update={"moe_routes": None})
                    for datum in batch.datums
                )
            }
        )
        wire = _TokenizedWireBatch(
            batch=stripped,
            route_pool=_TokenizedRoutePool(
                data=b"".join(packed.chunks),
                bindings=tuple(
                    _TokenizedRouteBinding(
                        datum_index=index,
                        num_experts=route.num_experts,
                        dtype=route.dtype,
                        shape=route.shape,
                        slices=tuple(
                            _RouteByteSlice(
                                segment_index=segment_index,
                                offset=offset,
                                byte_count=byte_count,
                            )
                            for segment_index, (offset, byte_count) in enumerate(
                                datum_slices
                            )
                        ),
                    )
                    for index, ((route, _), datum_slices) in enumerate(
                        zip(entries, packed.slices, strict=True)
                    )
                    if route is not None
                ),
            ),
        )
    return msgpack.encode(wire.model_dump(mode="python"))


def decode_tokenized_batch(
    ref: TrainingDataRef, payload: bytes
) -> TokenizedTrainingBatch:
    _validate_training_object(ref, payload, TOKENIZED_DATA_FORMAT)
    wire = _TokenizedWireBatch.model_validate(msgpack.decode(payload))
    if wire.route_pool is None:
        return wire.batch
    datums = list(wire.batch.datums)
    route_pool = memoryview(wire.route_pool.data).toreadonly()
    for binding in wire.route_pool.bindings:
        datums[binding.datum_index] = datums[binding.datum_index].model_copy(
            update={
                "moe_routes": TokenizedMoeRoutes(
                    num_experts=binding.num_experts,
                    dtype=binding.dtype,
                    shape=binding.shape,
                    data=tuple(
                        route_pool[value.offset : value.offset + value.byte_count]
                        for value in binding.slices
                    ),
                )
            }
        )
    return _TOKENIZED_BATCH_ADAPTER.validate_python(
        wire.batch.model_copy(update={"datums": tuple(datums)}).model_dump(
            mode="python"
        )
    )


def encode_forward_submission(
    request: RemoteForwardRequest,
    batch: EncodedTrainingBatch,
) -> EncodedForwardSubmission:
    if request.batch != batch.remote:
        raise ValueError("forward submission request and encoded batch differ")
    manifest = msgpack.encode(
        ForwardSubmissionManifest(
            request=request,
            objects=tuple(value.ref for value in batch.objects),
            route_objects=tuple(value.ref for value in batch.route_objects),
        ).model_dump(mode="python")
    )
    if len(manifest) > MAX_FORWARD_SUBMISSION_MANIFEST_BYTES:
        raise ValueError("forward submission manifest exceeds the configured limit")
    preamble = (
        _FORWARD_SUBMISSION_PREFIX.pack(_FORWARD_SUBMISSION_MAGIC, len(manifest))
        + manifest
    )
    chunks = tuple(
        chunk
        for value in batch.objects
        for chunk in _iter_readonly_chunks(value.payload)
    ) + tuple(chunk for value in batch.route_objects for chunk in value.chunks)
    return EncodedForwardSubmission(
        preamble=preamble,
        chunks=chunks,
        byte_count=len(preamble) + sum(map(len, chunks)),
    )


def decode_forward_submission_prefix(payload: bytes) -> int:
    if len(payload) != FORWARD_SUBMISSION_PREFIX_BYTES:
        raise ValueError("forward submission prefix has the wrong size")
    magic, manifest_bytes = _FORWARD_SUBMISSION_PREFIX.unpack(payload)
    if magic != _FORWARD_SUBMISSION_MAGIC:
        raise ValueError("forward submission has the wrong wire format")
    if not 0 < manifest_bytes <= MAX_FORWARD_SUBMISSION_MANIFEST_BYTES:
        raise ValueError("forward submission manifest size is invalid")
    return manifest_bytes


def decode_forward_submission_manifest(payload: bytes) -> ForwardSubmissionManifest:
    if not 0 < len(payload) <= MAX_FORWARD_SUBMISSION_MANIFEST_BYTES:
        raise ValueError("forward submission manifest size is invalid")
    return ForwardSubmissionManifest.model_validate(msgpack.decode(payload))


def forward_submission_byte_count(
    manifest: ForwardSubmissionManifest, manifest_bytes: int
) -> int:
    return (
        FORWARD_SUBMISSION_PREFIX_BYTES
        + manifest_bytes
        + sum(value.byte_count for value in manifest.objects)
        + sum(value.byte_count for value in manifest.route_objects)
    )


def _encode_training_object(
    payload: bytes,
    *,
    data_format: Literal[
        "art_trajectory_group_msgpack_v3",
        "art_sft_batch_msgpack_v1",
        "art_tokenized_batch_msgpack_v2",
    ],
    object_id: str | None,
) -> EncodedTrainingObject:
    digest = hashlib.sha256(payload).hexdigest()
    return EncodedTrainingObject(
        ref=TrainingDataRef(
            object_id=object_id or _object_id(uuid.uuid4().hex),
            sha256=digest,
            byte_count=len(payload),
            format=data_format,
        ),
        payload=payload,
    )


def _object_id(identity: str) -> str:
    return hashlib.sha256(identity.encode()).hexdigest()


def _validate_training_object(
    ref: TrainingDataRef, payload: bytes, expected_format: str
) -> None:
    if ref.format != expected_format:
        raise ValueError("training data has the wrong wire format")
    if len(payload) != ref.byte_count:
        raise ValueError("training data byte count differs from its reference")


def encode_operation_result(
    result: OperationResult,
) -> tuple[OperationResultRef, bytes]:
    value = result.model_dump(mode="python")
    result_values = sum(
        len(output["token_logprobs"]["data"]) // 4
        for output in value.get("loss_fn_outputs", ())
    )
    if result_values > MAX_TOKENIZED_LOGPROB_VALUES:
        raise ValueError(
            "operation result exceeds the configured logprob value limit: "
            f"{result_values} > {MAX_TOKENIZED_LOGPROB_VALUES}"
        )
    payload = msgpack.encode(value, enc_hook=_encode_ext)
    return (
        OperationResultRef(
            object_id=hashlib.sha256(payload).hexdigest(),
            byte_count=len(payload),
        ),
        payload,
    )


def decode_operation_result(
    ref: OperationResultRef, payload: bytes | bytearray, result_type: type[ResultT]
) -> ResultT:
    if len(payload) != ref.byte_count:
        raise ValueError("operation result byte count differs from its reference")
    if hashlib.sha256(payload).hexdigest() != ref.object_id:
        raise ValueError("operation result hash differs from its reference")
    return result_type.model_validate(msgpack.decode(payload, ext_hook=_decode_ext))


def _encode_ext(value: object):
    if not isinstance(value, array) or value.typecode != "I" or value.itemsize != 4:
        raise TypeError(f"unsupported operation result value: {type(value).__name__}")
    data = array("I", value)
    if sys.byteorder != "little":
        data.byteswap()
    return msgpack.Ext(_UINT32_ARRAY_EXT, data.tobytes())


def _decode_ext(code: int, data: memoryview) -> array:
    if code != _UINT32_ARRAY_EXT or len(data) % 4:
        raise ValueError("operation result contains an invalid MessagePack extension")
    value = array("I")
    value.frombytes(data)
    if sys.byteorder != "little":
        value.byteswap()
    return value
