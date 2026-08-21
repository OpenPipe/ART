from __future__ import annotations

from array import array
from collections.abc import AsyncIterable, AsyncIterator, Iterator, Mapping
import hashlib
import struct
import sys
from typing import Any, Literal, NamedTuple, TypeVar, cast

from msgspec import msgpack
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from art.distributed.moe_route_store import (
    MoeRouteGroupPayload,
    MoeRouteObjectBatchTransfer,
    MoeRouteObjectPayload,
    MoeRouteStoredObject,
)
from art.distributed.trajectory_store import (
    TrajectoryGroupBundle,
    TrajectoryGroupDataIdentity,
    TrajectoryGroupDataLayout,
    TrajectoryRouteSequence,
)
from art.megatron.prefix_tree_packing import prefix_tree_pack_segments
from art.training.contracts import (
    ForwardResult,
    OperationResult,
    RlTrajectoryBatch,
    SupervisedTrajectoryBatch,
    TokenizedTrainingBatch,
    TrainingBatch,
)
from art.training.tokenized import (
    MAX_TOKENIZED_LOGPROB_VALUES,
    TokenizedDatum,
    TokenizedMoeRoutes,
    TokenizedPolicySpan,
)

from .contracts import (
    MAX_INLINE_OPERATION_RESULT_BYTES,
    MAX_RL_GROUPS_PER_BATCH,
    RL_GROUP_DATA_FORMAT,
    SFT_DATA_FORMAT,
    TOKENIZED_DATA_FORMAT,
    InlineOperationResult,
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
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    ref: TrainingDataRef
    payload: bytes | None = None
    chunks: tuple[memoryview, ...] = ()

    @model_validator(mode="after")
    def _validate_payload(self) -> "EncodedTrainingObject":
        if (self.payload is None) == (not self.chunks):
            raise ValueError("training data requires exactly one byte source")
        if self.payload is not None:
            _validate_training_object(self.ref, self.payload, self.ref.format)
        elif _validate_wire_chunks(self.chunks) != self.ref.byte_count:
            raise ValueError("training data byte count differs from its reference")
        return self

    def wire_chunks(self) -> tuple[memoryview, ...]:
        if self.payload is not None:
            return tuple(_iter_readonly_chunks(self.payload))
        return self.chunks


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


class VerifiedOperationResultPayload(BaseModel):
    """Operation-result bytes whose digest was verified during receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    ref: OperationResultRef
    payload: memoryview

    @field_validator("payload", mode="before")
    @classmethod
    def _readonly_bytes(cls, value: bytes | bytearray | memoryview) -> memoryview:
        view = memoryview(value)
        if not view.c_contiguous:
            raise ValueError("verified operation result must be contiguous")
        return view.cast("B").toreadonly()

    @model_validator(mode="after")
    def _validate_size(self) -> "VerifiedOperationResultPayload":
        if len(self.payload) != self.ref.byte_count:
            raise ValueError("verified operation result has the wrong byte count")
        return self


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
    route_group: MoeRouteGroupPayload | None = None,
    stored_routes: tuple[MoeRouteStoredObject, ...] | None = None,
) -> EncodedRlGroup:
    sources = (route_group is not None, stored_routes is not None)
    if sum(sources) > 1:
        raise ValueError("RL trajectory group has multiple explicit route sources")
    if route_encoding not in {"inline", "prefix_tree"}:
        raise ValueError(f"unsupported route wire encoding: {route_encoding!r}")
    layout = TrajectoryGroupDataLayout(
        header_byte_count=len(bundle.header),
        record_byte_counts=tuple(map(len, bundle.records)),
    )
    data = _encode_training_object_chunks(
        (bundle.header, *bundle.records),
        identity=bundle.route_free_identity,
        data_format=RL_GROUP_DATA_FORMAT,
        object_id=object_id,
    )
    data_id = data.ref.object_id
    if stored_routes is not None:
        _validate_route_bindings(bundle, stored_routes)
        routes, remote_routes = (), tuple(
            RemoteRouteObject(
                ref=RemoteRouteObjectRef(
                    object_id=value.object_id,
                    byte_count=value.byte_count,
                    transport="object_store",
                ),
                slices=value.slices,
            )
            for value in stored_routes
        )
    elif route_group is not None:
        _validate_route_bindings(bundle, route_group.objects)
        routes, remote_routes = _encode_route_group(route_group, data_id=data_id)
    else:
        packed = _pack_trajectory_routes(bundle.route_sequences)
        if packed is None:
            routes, remote_routes = (), ()
        else:
            route_ref = RemoteRouteObjectRef(
                object_id=_object_id(f"{data_id}:{packed.sha256}"),
                byte_count=packed.byte_count,
                transport="command",
                sha256=packed.sha256,
            )
            routes = (EncodedRouteObject(ref=route_ref, chunks=packed.chunks),)
            remote_routes = (
                RemoteRouteObject(ref=route_ref, slices=packed.slices),
            )
    return EncodedRlGroup(
        data=data,
        routes=tuple(routes),
        remote=RemoteRlGroupRef(
            data=data.ref,
            layout=layout,
            routes=tuple(remote_routes),
        ),
    )


def _validate_route_bindings(
    bundle: TrajectoryGroupBundle,
    objects: tuple[MoeRouteObjectPayload | MoeRouteStoredObject, ...],
) -> None:
    if not bundle.route_sequences:
        return
    expected = {
        (
            sequence.trajectory_index,
            sequence.scope,
            sequence.scope_index,
            sequence.choice_index,
        )
        for sequence in bundle.route_sequences
    }
    received = {
        (
            value.trajectory_index,
            value.scope,
            value.scope_index,
            value.choice_index,
        )
        for obj in objects
        for value in obj.slices
    }
    if received != expected:
        raise ValueError("explicit routes do not match trajectory route bindings")


def _encode_route_group(
    group: MoeRouteGroupPayload, *, data_id: str
) -> tuple[tuple[EncodedRouteObject, ...], tuple[RemoteRouteObject, ...]]:
    encoded = []
    remote = []
    for index, value in enumerate(group.objects):
        chunks = tuple(_iter_readonly_chunks(value.data))
        digest = hashlib.sha256()
        for chunk in chunks:
            digest.update(chunk)
        sha256 = digest.hexdigest()
        ref = RemoteRouteObjectRef(
            object_id=_object_id(f"{data_id}:{index}:{sha256}"),
            byte_count=sum(map(len, chunks)),
            transport="command",
            sha256=sha256,
        )
        encoded.append(EncodedRouteObject(ref=ref, chunks=chunks))
        remote.append(RemoteRouteObject(ref=ref, slices=value.slices))
    return tuple(encoded), tuple(remote)


def _pack_trajectory_routes(
    sequences: tuple[TrajectoryRouteSequence, ...],
) -> _PackedRouteObject | None:
    if not sequences:
        return None
    packed = _pack_route_sequences(
        tuple((sequence, sequence.token_ids) for sequence in sequences),
        compute_sha256=True,
    )
    if packed.sha256 is None:
        raise RuntimeError("route object identity was not computed")
    slices = tuple(
        RemoteRouteSlice(
            trajectory_index=sequence.trajectory_index,
            scope=sequence.scope,
            scope_index=sequence.scope_index,
            choice_index=sequence.choice_index,
            segment_index=segment_index,
            offset=offset,
            byte_count=byte_count,
        )
        for sequence, sequence_slices in zip(sequences, packed.slices, strict=True)
        for segment_index, (offset, byte_count) in enumerate(sequence_slices)
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
        offset = byte_count
        for source_chunk in _route_data_range(
            route.data,
            segment.start * bytes_per_token,
            segment.end * bytes_per_token,
        ):
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


def prepare_training_batch(
    batch: TrainingBatch,
    *,
    route_encoding: RouteWireEncoding = "prefix_tree",
) -> EncodedTrainingBatch:
    if isinstance(batch, RlTrajectoryBatch):
        annotations = batch.local_group_annotations()
        route_groups = batch.local_moe_route_groups()
        route_transfer = batch.local_moe_route_object_transfer()
        if route_groups and route_transfer is not None:
            raise ValueError("RL batch has both inline and stored route sources")
        groups = tuple(
            encode_trajectory_group(
                group,
                route_encoding=route_encoding,
                route_group=route_groups[index] if route_groups else None,
                stored_routes=(
                    route_transfer.groups[index]
                    if route_transfer is not None
                    else None
                ),
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
            object_id=None,
        )
        objects = (value,)
        route_objects = ()
        remote = RemoteSftBatchRef(data=value.ref)
    else:
        payload = (
            encode_tokenized_batch_wire(batch)
            if route_encoding == "prefix_tree"
            else _encode_tokenized_wire_batch(batch, route_encoding=route_encoding)
        )
        value = _encode_training_object(
            payload,
            data_format=TOKENIZED_DATA_FORMAT,
            object_id=None,
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
    route_payloads: Mapping[str, bytes | memoryview],
) -> DecodedRlGroup:
    _validate_training_object(ref.data, payload, RL_GROUP_DATA_FORMAT)
    layout = ref.layout
    if len(payload) != layout.byte_count:
        raise ValueError("RL trajectory layout differs from its data object")
    source = memoryview(payload).toreadonly()
    offset = layout.header_byte_count
    header = source[:offset]
    records = []
    for byte_count in layout.record_byte_counts:
        end = offset + byte_count
        records.append(source[offset:end])
        offset = end
    bundle = TrajectoryGroupBundle(
        header=header,
        records=tuple(records),
        route_free_identity=TrajectoryGroupDataIdentity(
            sha256=ref.data.sha256, byte_count=ref.data.byte_count
        ),
    )
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


def _route_payload(
    object_id: str, payloads: Mapping[str, bytes | memoryview]
) -> bytes | memoryview:
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


def encode_tokenized_batch_wire(batch: TokenizedTrainingBatch) -> bytes:
    """Return the immutable compact v2 representation used by remote packing."""
    payload = batch.encoded_payload()
    if payload is None:
        payload = _encode_tokenized_wire_batch(batch, route_encoding="prefix_tree")
    return payload


def _validated_tokenized_wire(payload: bytes) -> _TokenizedWireBatch:
    return _TokenizedWireBatch.model_validate(msgpack.decode(payload))


def _materialize_validated_tokenized_wire(
    wire: _TokenizedWireBatch,
) -> TokenizedTrainingBatch:
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
    return wire.batch.model_copy(update={"datums": tuple(datums)})


def decode_tokenized_batch(
    ref: TrainingDataRef, payload: bytes
) -> TokenizedTrainingBatch:
    _validate_training_object(ref, payload, TOKENIZED_DATA_FORMAT)
    batch = _materialize_validated_tokenized_wire(_validated_tokenized_wire(payload))
    batch.remember_encoded_payload(payload)
    return batch


def _trusted_moe_routes(
    value: Mapping[str, Any],
    *,
    data: tuple[memoryview, ...] | None = None,
) -> TokenizedMoeRoutes:
    return TokenizedMoeRoutes.model_construct(
        num_experts=value["num_experts"],
        dtype=value["dtype"],
        shape=tuple(value["shape"]),
        data=(
            data
            if data is not None
            else tuple(memoryview(segment) for segment in value["data"])
        ),
    )


def _trusted_tokenized_datum(
    value: Mapping[str, Any],
    *,
    moe_routes: TokenizedMoeRoutes | None,
) -> TokenizedDatum:
    def matrix(name: str) -> tuple[tuple[Any, ...], ...] | None:
        rows = value[name]
        return None if rows is None else tuple(tuple(row) for row in rows)

    return TokenizedDatum.model_construct(
        input_tokens=tuple(value["input_tokens"]),
        target_tokens=matrix("target_tokens"),
        weights=matrix("weights"),
        logprobs=matrix("logprobs"),
        advantages=matrix("advantages"),
        packing_group_id=value["packing_group_id"],
        policy_spans=tuple(
            TokenizedPolicySpan.model_construct(**span)
            for span in value["policy_spans"]
        ),
        moe_routes=moe_routes,
    )


def decode_trusted_tokenized_batch_wire(
    payload: bytes | bytearray | memoryview,
) -> TokenizedTrainingBatch:
    """Materialize compact bytes already validated at the service boundary."""
    wire = msgpack.decode(payload)
    batch = wire["batch"]
    raw_datums = batch["datums"]
    route_pool = wire["route_pool"]
    if route_pool is None:
        routes = tuple(
            None
            if datum["moe_routes"] is None
            else _trusted_moe_routes(datum["moe_routes"])
            for datum in raw_datums
        )
    else:
        pooled = memoryview(route_pool["data"])
        routes = tuple(
            _trusted_moe_routes(
                binding,
                data=tuple(
                    pooled[value["offset"] : value["offset"] + value["byte_count"]]
                    for value in binding["slices"]
                ),
            )
            for binding in route_pool["bindings"]
        )
    return TokenizedTrainingBatch.model_construct(
        kind="tokenized",
        datums=tuple(
            _trusted_tokenized_datum(datum, moe_routes=route)
            for datum, route in zip(raw_datums, routes, strict=True)
        ),
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
        chunk for value in batch.objects for chunk in value.wire_chunks()
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
        "art_sft_batch_msgpack_v1",
        "art_tokenized_batch_msgpack_v2",
    ],
    object_id: str | None,
) -> EncodedTrainingObject:
    digest = hashlib.sha256(payload).hexdigest()
    return EncodedTrainingObject(
        ref=TrainingDataRef(
            object_id=object_id or _object_id(f"{data_format}\0{digest}"),
            sha256=digest,
            byte_count=len(payload),
            format=data_format,
        ),
        payload=payload,
    )


def _encode_training_object_chunks(
    payloads: tuple[bytes | memoryview, ...],
    *,
    identity: TrajectoryGroupDataIdentity,
    data_format: Literal["art_trajectory_group_records_v4"],
    object_id: str | None,
) -> EncodedTrainingObject:
    chunks = tuple(
        chunk for payload in payloads for chunk in _iter_readonly_chunks(payload)
    )
    return EncodedTrainingObject(
        ref=TrainingDataRef(
            object_id=object_id or _object_id(f"{data_format}\0{identity.sha256}"),
            sha256=identity.sha256,
            byte_count=identity.byte_count,
            format=data_format,
        ),
        chunks=chunks,
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


def encode_inline_operation_result(
    result: OperationResult,
) -> tuple[InlineOperationResult, bytes] | None:
    if isinstance(result, ForwardResult):
        logprob_bytes = sum(
            len(output.token_logprobs.data) for output in result.loss_fn_outputs
        )
        # Base64 alone exceeds the envelope bound. This is an exact proof about
        # encoded JSON size, not a semantic decision based on requested outputs.
        if 4 * ((logprob_bytes + 2) // 3) > MAX_INLINE_OPERATION_RESULT_BYTES:
            return None
        token_id_json_bytes = sum(
            sum(len(str(token_id)) + 1 for token_id in leaf.token_ids)
            for shape in result.packing.group_shapes
            for leaf in shape.leaves
        )
        if token_id_json_bytes > MAX_INLINE_OPERATION_RESULT_BYTES:
            return None
    inline = InlineOperationResult(result=result.model_dump(mode="json"))
    payload = inline.model_dump_json().encode()
    if len(payload) > MAX_INLINE_OPERATION_RESULT_BYTES:
        return None
    return inline, payload


def decode_operation_result(
    ref: OperationResultRef,
    payload: bytes | bytearray | memoryview,
    result_type: type[ResultT],
) -> ResultT:
    if len(payload) != ref.byte_count:
        raise ValueError("operation result byte count differs from its reference")
    if hashlib.sha256(payload).hexdigest() != ref.object_id:
        raise ValueError("operation result hash differs from its reference")
    return _decode_operation_result(payload, result_type)


def decode_verified_operation_result(
    verified: VerifiedOperationResultPayload, result_type: type[ResultT]
) -> ResultT:
    return _decode_operation_result(verified.payload, result_type)


def _decode_operation_result(
    payload: bytes | bytearray | memoryview, result_type: type[ResultT]
) -> ResultT:
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
