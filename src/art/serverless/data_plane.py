from __future__ import annotations

from array import array
import hashlib
import sys
from typing import TypeVar

from msgspec import msgpack
from pydantic import TypeAdapter

from art.training.contracts import OperationResult, TrainingBatch

from .contracts import OperationResultRef, TrainingDataRef

_BATCH_ADAPTER = TypeAdapter(TrainingBatch)
_UINT32_ARRAY_EXT = 1
_FLOAT32_ARRAY_EXT = 2
ResultT = TypeVar("ResultT", bound=OperationResult)


def encode_training_batch(batch: TrainingBatch) -> tuple[TrainingDataRef, bytes]:
    payload = msgpack.encode(batch.model_dump(mode="python"))
    return (
        TrainingDataRef(
            object_id=hashlib.sha256(payload).hexdigest(),
            byte_count=len(payload),
            batch_kind=batch.kind,
        ),
        payload,
    )


def decode_training_batch(ref: TrainingDataRef, payload: bytes) -> TrainingBatch:
    if len(payload) != ref.byte_count:
        raise ValueError("training data byte count differs from its reference")
    if hashlib.sha256(payload).hexdigest() != ref.object_id:
        raise ValueError("training data hash differs from its reference")
    batch = _BATCH_ADAPTER.validate_python(msgpack.decode(payload))
    if batch.kind != ref.batch_kind:
        raise ValueError("training data kind differs from its reference")
    return batch


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
