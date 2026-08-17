from __future__ import annotations

from array import array
from collections.abc import Mapping
import hashlib
import sys
from typing import Any, Literal, NamedTuple, TypeVar
import uuid

from msgspec import msgpack
from pydantic import BaseModel, ConfigDict, TypeAdapter, model_validator

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
_FLOAT32_ARRAY_EXT = 2
_LOGPROB_SHAPE_KEY = "__art_shape__"
_LOGPROB_VALUES_KEY = "__art_values__"
ResultT = TypeVar("ResultT", bound=OperationResult)
RouteWireEncoding = Literal["inline", "prefix_tree"]


class _RouteEntry(NamedTuple):
    trajectory_index: int
    scope: Literal["messages", "additional_history", "exchange"]
    scope_index: int
    choice_index: int
    route: Any


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
    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: RemoteRouteObjectRef
    payload: bytes

    @model_validator(mode="after")
    def _validate_payload(self) -> "EncodedRouteObject":
        if self.ref.transport != "command" or self.ref.sha256 is None:
            raise ValueError("encoded route bytes require command transport")
        if len(self.payload) != self.ref.byte_count:
            raise ValueError("route byte count differs from its reference")
        if hashlib.sha256(self.payload).hexdigest() != self.ref.sha256:
            raise ValueError("route payload hash differs from its reference")
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


class EncodedForwardSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request: RemoteForwardRequest
    objects: tuple[EncodedTrainingObject, ...]
    route_objects: tuple[EncodedRouteObject, ...] = ()

    @model_validator(mode="after")
    def _validate_objects(self) -> "EncodedForwardSubmission":
        route_refs = {
            ref.object_id: ref for ref in command_route_object_refs(self.request.batch)
        }
        route_data_ids = set(route_refs)
        expected_data = {
            ref.object_id: ref
            for ref in training_data_refs(self.request.batch)
            if ref.object_id not in route_data_ids
        }
        if {value.ref.object_id: value.ref for value in self.objects} != expected_data:
            raise ValueError("forward submission data objects do not match request")
        if {
            value.ref.object_id: value.ref for value in self.route_objects
        } != route_refs:
            raise ValueError("forward submission route objects do not match request")
        return self


def encode_trajectory_group(
    bundle: TrajectoryGroupBundle,
    *,
    object_id: str | None = None,
    route_encoding: RouteWireEncoding = "prefix_tree",
) -> EncodedRlGroup:
    data_id = object_id or _object_id(uuid.uuid4().hex)
    source = bundle.payload()
    if route_encoding == "prefix_tree":
        stripped, route_payload, slices = _extract_routes(source)
    elif route_encoding == "inline":
        stripped, route_payload, slices = source, b"", ()
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
    if route_payload:
        route_digest = hashlib.sha256(route_payload).hexdigest()
        route_ref = RemoteRouteObjectRef(
            object_id=_object_id(f"{data_id}:{route_digest}"),
            byte_count=len(route_payload),
            transport="command",
            sha256=route_digest,
        )
        routes = (EncodedRouteObject(ref=route_ref, payload=route_payload),)
        remote_routes = (RemoteRouteObject(ref=route_ref, slices=slices),)
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
) -> tuple[TrajectoryGroupPayload, bytes, tuple[RemoteRouteSlice, ...]]:
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
    return (
        payload.model_copy(update={"trajectories": tuple(trajectories)}),
        *_pack_route_entries(entries),
    )


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
) -> tuple[bytes, tuple[RemoteRouteSlice, ...]]:
    if not entries:
        return b"", ()
    data, route_slices = _pack_route_sequences(
        tuple((entry.route, _route_token_ids(entry.route)) for entry in entries)
    )
    return data, tuple(
        RemoteRouteSlice(
            trajectory_index=entry.trajectory_index,
            scope=entry.scope,
            scope_index=entry.scope_index,
            choice_index=entry.choice_index,
            segment_index=segment_index,
            offset=offset,
            byte_count=byte_count,
        )
        for entry, slices in zip(entries, route_slices, strict=True)
        for segment_index, (offset, byte_count) in enumerate(slices)
    )


def _pack_route_sequences(
    entries: tuple[tuple[Any, tuple[int, ...]], ...],
) -> tuple[bytes, tuple[tuple[tuple[int, int], ...], ...]]:
    if not entries:
        return b"", ()
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
    data = bytearray()
    slices: list[list[tuple[int, int]]] = [[] for _ in entries]
    for segment in planned:
        route = entries[segment.sequence_indices[0]][0]
        bytes_per_token = _route_bytes_per_token(route)
        chunks = _route_data_range(
            route.data,
            segment.start * bytes_per_token,
            segment.end * bytes_per_token,
        )
        offset = len(data)
        for chunk in chunks:
            data.extend(chunk)
        byte_count = len(data) - offset
        for sequence_index in segment.sequence_indices:
            slices[sequence_index].append((offset, byte_count))
    return bytes(data), tuple(tuple(value) for value in slices)


def _route_token_ids(route: Any) -> tuple[int, ...]:
    prompt = route.metadata.get(PROMPT_TOKEN_IDS_KEY)
    completion = route.metadata.get(COMPLETION_TOKEN_IDS_KEY)
    if not isinstance(prompt, list) or not isinstance(completion, list):
        raise RuntimeError("routed experts are missing exact token ids")
    token_ids = tuple(prompt) + tuple(completion)
    if len(token_ids) != route.shape[0] or any(
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
        data, slices = _pack_route_sequences(
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
                data=data,
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
                        zip(entries, slices, strict=True)
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
    for binding in wire.route_pool.bindings:
        datums[binding.datum_index] = datums[binding.datum_index].model_copy(
            update={
                "moe_routes": TokenizedMoeRoutes(
                    num_experts=binding.num_experts,
                    dtype=binding.dtype,
                    shape=binding.shape,
                    data=tuple(
                        wire.route_pool.data[
                            value.offset : value.offset + value.byte_count
                        ]
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
) -> bytes:
    if request.batch != batch.remote:
        raise ValueError("forward submission request and encoded batch differ")
    return msgpack.encode(
        EncodedForwardSubmission(
            request=request,
            objects=batch.objects,
            route_objects=batch.route_objects,
        ).model_dump(mode="python")
    )


def decode_forward_submission(payload: bytes) -> EncodedForwardSubmission:
    return EncodedForwardSubmission.model_validate(msgpack.decode(payload))


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
    if hashlib.sha256(payload).hexdigest() != ref.sha256:
        raise ValueError("training data hash differs from its reference")


def encode_operation_result(
    result: OperationResult,
) -> tuple[OperationResultRef, bytes]:
    value = result.model_dump(mode="python")
    result_values = 0
    for output in value.get("loss_fn_outputs", ()):
        raw = output["token_logprobs"]
        nested = bool(raw) and isinstance(raw[0], list | tuple)
        rows = tuple(raw) if nested else (raw,)
        width = len(rows[0]) if rows else 0
        if any(len(row) != width for row in rows):
            raise ValueError("token logprobs must be rectangular")
        result_values += sum(map(len, rows))
        output["token_logprobs"] = {
            _LOGPROB_SHAPE_KEY: ((len(rows), width) if nested else (len(raw),)),
            _LOGPROB_VALUES_KEY: array("f", (item for row in rows for item in row)),
        }
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
    ref: OperationResultRef, payload: bytes, result_type: type[ResultT]
) -> ResultT:
    if len(payload) != ref.byte_count:
        raise ValueError("operation result byte count differs from its reference")
    if hashlib.sha256(payload).hexdigest() != ref.object_id:
        raise ValueError("operation result hash differs from its reference")
    value = msgpack.decode(payload, ext_hook=_decode_ext)
    for output in value.get("loss_fn_outputs", ()):
        packed = output["token_logprobs"]
        shape = tuple(packed[_LOGPROB_SHAPE_KEY])
        values = packed[_LOGPROB_VALUES_KEY]
        if len(shape) == 1:
            output["token_logprobs"] = tuple(values)
        elif len(shape) == 2 and len(values) == shape[0] * shape[1]:
            output["token_logprobs"] = tuple(
                tuple(values[start : start + shape[1]])
                for start in range(0, len(values), shape[1])
            )
        else:
            raise ValueError("operation result contains an invalid logprob shape")
    return result_type.model_validate(value)


def _encode_ext(value: object):
    if not isinstance(value, array) or value.typecode not in {"I", "f"}:
        raise TypeError(f"unsupported operation result value: {type(value).__name__}")
    if value.itemsize != 4:
        raise TypeError("operation result arrays require 32-bit elements")
    data = array(value.typecode, value)
    if sys.byteorder != "little":
        data.byteswap()
    return msgpack.Ext(
        _UINT32_ARRAY_EXT if value.typecode == "I" else _FLOAT32_ARRAY_EXT,
        data.tobytes(),
    )


def _decode_ext(code: int, data: memoryview) -> array:
    if code not in {_UINT32_ARRAY_EXT, _FLOAT32_ARRAY_EXT} or len(data) % 4:
        raise ValueError("operation result contains an invalid MessagePack extension")
    value = array("I" if code == _UINT32_ARRAY_EXT else "f")
    value.frombytes(data)
    if sys.byteorder != "little":
        value.byteswap()
    return value
