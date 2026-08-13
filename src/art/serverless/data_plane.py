from __future__ import annotations

import hashlib

from msgspec import msgpack
from pydantic import TypeAdapter

from art.training.contracts import TrainingBatch

from .contracts import TrainingDataRef

_BATCH_ADAPTER = TypeAdapter(TrainingBatch)


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
