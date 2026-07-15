from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, PrivateAttr

from art.distributed.data_plane import MappedPackedBatch, PackedBatchRef
from art.preprocessing.pack import PackedTensors


@runtime_checkable
class PackedBatch(Protocol):
    """A leased current-format batch; transport and storage remain data-plane owned."""

    @property
    def ref(self) -> PackedBatchRef: ...


class InMemoryPackedBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    ref: PackedBatchRef
    tensors: PackedTensors
    _mapped: MappedPackedBatch | None = PrivateAttr(default=None)

    @classmethod
    def open(
        cls, ref: PackedBatchRef, local_ref: PackedBatchRef
    ) -> "InMemoryPackedBatch":
        mapped = MappedPackedBatch.open(local_ref)
        batch = cls(ref=ref, tensors=mapped.tensors)
        batch._mapped = mapped
        return batch

    def close(self) -> None:
        if self._mapped is not None:
            self._mapped.close()
            self._mapped = None


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
