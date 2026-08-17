from __future__ import annotations

from array import array
from collections.abc import Mapping
import hashlib
import sys
from typing import Any, Literal, TypeVar
import uuid

from msgspec import msgpack
from pydantic import BaseModel, ConfigDict, TypeAdapter

from art.distributed.moe_route_store import (
    MoeRouteGroupPayload,
    MoeRouteObjectPayload,
)
from art.distributed.packing import TrajectoryGroupPayload
from art.distributed.trajectory_store import TrajectoryGroupBundle
from art.preprocessing.moe_routing import PROMPT_TOKEN_IDS_KEY
from art.training.contracts import (
    OperationResult,
    RlTrajectoryBatch,
    SupervisedTrajectoryBatch,
    TrainingBatch,
)

from .contracts import (
    RL_GROUP_DATA_FORMAT,
    SFT_DATA_FORMAT,
    OperationResultRef,
    RemoteRlBatchRef,
    RemoteRlGroupRef,
    RemoteRouteObject,
    RemoteRouteObjectRef,
    RemoteRouteSlice,
    RemoteSftBatchRef,
    RemoteTrainingBatchRef,
    TrainingDataRef,
)

_SFT_BATCH_ADAPTER = TypeAdapter(SupervisedTrajectoryBatch)
_UINT32_ARRAY_EXT = 1
_FLOAT32_ARRAY_EXT = 2
ResultT = TypeVar("ResultT", bound=OperationResult)


class EncodedTrainingObject(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: TrainingDataRef
    payload: bytes


class EncodedRouteObject(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: RemoteRouteObjectRef
    payload: bytes


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


def encode_trajectory_group(
    bundle: TrajectoryGroupBundle, *, object_id: str | None = None
) -> EncodedRlGroup:
    data_id = object_id or _object_id(uuid.uuid4().hex)
    stripped, route_payload, slices = _extract_routes(bundle.payload())
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
        )
        routes = (EncodedRouteObject(ref=route_ref, payload=route_payload),)
        remote_routes = (RemoteRouteObject(ref=route_ref, slices=slices),)
    return EncodedRlGroup(
        data=data,
        routes=routes,
        remote=RemoteRlGroupRef(data=data.ref, routes=remote_routes),
    )


def prepare_training_batch(
    batch: TrainingBatch, *, identity: str | None = None
) -> EncodedTrainingBatch:
    if isinstance(batch, RlTrajectoryBatch):
        groups = tuple(
            encode_trajectory_group(
                group,
                object_id=_object_id(f"{identity}:{index}") if identity else None,
            )
            for index, group in enumerate(batch.groups)
        )
        objects = tuple(group.data for group in groups)
        route_objects = tuple(route for group in groups for route in group.routes)
        remote: RemoteTrainingBatchRef = RemoteRlBatchRef(
            groups=tuple(group.remote for group in groups),
            min_source_version=batch.min_source_version,
            max_source_version=batch.max_source_version,
        )
    else:
        value = _encode_training_object(
            msgpack.encode(batch.model_dump(mode="python")),
            data_format=SFT_DATA_FORMAT,
            object_id=_object_id(identity) if identity else None,
        )
        objects = (value,)
        route_objects = ()
        remote = RemoteSftBatchRef(data=value.ref)
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
    data = bytearray()
    slices: list[RemoteRouteSlice] = []
    segments: dict[tuple[str, int, int, memoryview], tuple[int, int]] = {}
    trajectories = []
    for trajectory_index, trajectory in enumerate(payload.trajectories):
        histories = tuple(
            _strip_route_map(
                values,
                trajectory_index=trajectory_index,
                scope="additional_history",
                scope_index=index,
                data=data,
                slices=slices,
                segments=segments,
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
                data=data,
                slices=slices,
                segments=segments,
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
                        data=data,
                        slices=slices,
                        segments=segments,
                    ),
                    "additional_history_choice_routing_metadata": histories,
                    "exchange_choice_routing_metadata": exchanges,
                }
            )
        )
    return (
        payload.model_copy(update={"trajectories": tuple(trajectories)}),
        bytes(data),
        tuple(slices),
    )


def _strip_route_map(
    values: dict[int, Any],
    *,
    trajectory_index: int,
    scope: Literal["messages", "additional_history", "exchange"],
    scope_index: int,
    data: bytearray,
    slices: list[RemoteRouteSlice],
    segments: dict[tuple[str, int, int, memoryview], tuple[int, int]],
) -> dict[int, Any]:
    stripped = {}
    for choice_index, route in values.items():
        if not route.data:
            raise RuntimeError("inline routed experts are empty")
        for segment_index, segment in enumerate(_route_segments(route)):
            key = (route.dtype, route.shape[1], route.shape[2], segment)
            physical = segments.get(key)
            if physical is None:
                physical = len(data), len(segment)
                data.extend(segment)
                segments[key] = physical
            slices.append(
                RemoteRouteSlice(
                    trajectory_index=trajectory_index,
                    scope=scope,
                    scope_index=scope_index,
                    choice_index=choice_index,
                    segment_index=segment_index,
                    offset=physical[0],
                    byte_count=physical[1],
                )
            )
        stripped[choice_index] = route.model_copy(update={"data": ()})
    return stripped


def _route_segments(route: Any) -> tuple[memoryview, ...]:
    prompt_ids = route.metadata.get(PROMPT_TOKEN_IDS_KEY)
    if not isinstance(prompt_ids, list):
        raise RuntimeError("routed experts are missing prompt token ids")
    bytes_per_token = (
        (1 if route.dtype == "uint8" else 2) * route.shape[1] * route.shape[2]
    )
    prompt_end = len(prompt_ids) * bytes_per_token
    total = route.shape[0] * bytes_per_token
    boundaries = (0, prompt_end, total) if 0 < prompt_end < total else (0, total)
    return tuple(
        segment
        for start, end in zip(boundaries, boundaries[1:])
        for segment in _route_data_range(route.data, start, end)
    )


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


def _encode_training_object(
    payload: bytes,
    *,
    data_format: Literal["art_trajectory_group_msgpack_v3", "art_sft_batch_msgpack_v1"],
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
    for output in value.get("loss_fn_outputs", ()):
        output["token_logprobs"] = array("f", output["token_logprobs"])
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
    return result_type.model_validate(msgpack.decode(payload, ext_hook=_decode_ext))


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
