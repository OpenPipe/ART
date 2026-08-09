from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from art.distributed.data_plane import MappedPackedBatch, PackedBatchRef
from art.preprocessing.pack import PackedTensors


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


class SFTBatchData(BaseModel):
    """Typed in-memory SFT payload sent directly to warm trainer actors."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    trajectory_tensors: tuple[dict[str, Any], ...]
    learning_rate: float
    num_trajectories: int
    num_tokens: int
    num_trainable_tokens: int

    @model_validator(mode="after")
    def _validate_trajectories(self) -> "SFTBatchData":
        if not self.trajectory_tensors:
            raise ValueError("SFT batch must contain at least one trajectory")
        if self.num_trajectories != len(self.trajectory_tensors):
            raise ValueError("SFT trajectory count does not match its tensor payload")
        required = {"input_ids", "attention_mask", "labels"}
        if any(not required <= tensors.keys() for tensors in self.trajectory_tensors):
            raise ValueError("SFT trajectory tensors are incomplete")
        if self.num_tokens < 1 or self.num_trainable_tokens < 1:
            raise ValueError("SFT batch must contain trainable tokens")
        return self


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
