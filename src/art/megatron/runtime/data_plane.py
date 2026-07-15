from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from art.preprocessing.pack import PackedTensors

from .specs import PackedBatchRef


@runtime_checkable
class PackedBatch(Protocol):
    """A leased current-format batch; transport and storage remain data-plane owned."""

    @property
    def ref(self) -> PackedBatchRef: ...


class InMemoryPackedBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    ref: PackedBatchRef
    tensors: PackedTensors


class PackedBatchSource(Protocol):
    """Rank-local view of a host inbox populated before the train RPC."""

    def acquire(self, ref: PackedBatchRef) -> InMemoryPackedBatch: ...

    def release(self, ref: PackedBatchRef) -> None: ...


def validate_packed_batch(batch: InMemoryPackedBatch) -> None:
    tokens = batch.tensors["tokens"]
    shape = tuple(int(size) for size in tokens.shape)
    expected = (batch.ref.num_sequences, batch.ref.sequence_length)
    if shape != expected:
        raise ValueError(
            f"packed token shape {shape} does not match batch ref {expected}"
        )
    for key, tensor in batch.tensors.items():
        is_contiguous = getattr(tensor, "is_contiguous", None)
        if callable(is_contiguous) and not is_contiguous():
            raise ValueError(f"packed tensor {key!r} must be contiguous")
