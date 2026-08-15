from __future__ import annotations

from array import array
import hashlib
import sys
from typing import Literal, TypeVar
import uuid

from msgspec import msgpack
from pydantic import BaseModel, ConfigDict, TypeAdapter

from art.distributed.trajectory_store import TrajectoryGroupBundle
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


class EncodedTrainingBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    batch: TrainingBatch
    remote: RemoteTrainingBatchRef
    objects: tuple[EncodedTrainingObject, ...]


def encode_trajectory_group(
    bundle: TrajectoryGroupBundle, *, object_id: str | None = None
) -> EncodedTrainingObject:
    return _encode_training_object(
        msgpack.encode(bundle.model_dump(mode="python")),
        data_format=RL_GROUP_DATA_FORMAT,
        object_id=object_id,
    )


def prepare_training_batch(
    batch: TrainingBatch, *, identity: str | None = None
) -> EncodedTrainingBatch:
    if isinstance(batch, RlTrajectoryBatch):
        objects = tuple(
            encode_trajectory_group(
                group,
                object_id=_object_id(f"{identity}:{index}") if identity else None,
            )
            for index, group in enumerate(batch.groups)
        )
        remote: RemoteTrainingBatchRef = RemoteRlBatchRef(
            groups=tuple(RemoteRlGroupRef(data=value.ref) for value in objects),
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
        remote = RemoteSftBatchRef(data=value.ref)
    return EncodedTrainingBatch(batch=batch, remote=remote, objects=objects)


def decode_trajectory_group(
    ref: TrainingDataRef, payload: bytes
) -> TrajectoryGroupBundle:
    _validate_training_object(ref, payload, RL_GROUP_DATA_FORMAT)
    return TrajectoryGroupBundle.model_validate(msgpack.decode(payload))


def decode_sft_batch(ref: TrainingDataRef, payload: bytes) -> SupervisedTrajectoryBatch:
    _validate_training_object(ref, payload, SFT_DATA_FORMAT)
    return _SFT_BATCH_ADAPTER.validate_python(msgpack.decode(payload))


def _encode_training_object(
    payload: bytes,
    *,
    data_format: Literal["art_trajectory_group_msgpack_v1", "art_sft_batch_msgpack_v1"],
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
