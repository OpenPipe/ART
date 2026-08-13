from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator
import torch

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
    learning_rate: float = 0.0
    num_trajectories: int
    num_tokens: int
    num_trainable_tokens: int
    num_dropped_trajectories: int = 0

    @model_validator(mode="after")
    def _validate_trajectories(self) -> "SFTBatchData":
        if not self.trajectory_tensors:
            raise ValueError("SFT batch must contain at least one trajectory")
        if self.num_trajectories != len(self.trajectory_tensors):
            raise ValueError("SFT trajectory count does not match its tensor payload")
        required = {"input_ids", "attention_mask", "labels"}
        if any(not required <= tensors.keys() for tensors in self.trajectory_tensors):
            raise ValueError("SFT trajectory tensors are incomplete")
        shapes = [
            tuple(tensors[name].shape)
            for tensors in self.trajectory_tensors
            for name in required
        ]
        if any(len(shape) != 2 or shape[0] != 1 for shape in shapes):
            raise ValueError("SFT trajectory tensors must have shape [1, sequence]")
        if any(
            len(
                {
                    tuple(tensors[name].shape)
                    for name in ("input_ids", "attention_mask", "labels")
                }
            )
            != 1
            for tensors in self.trajectory_tensors
        ):
            raise ValueError("SFT trajectory tensor shapes differ")
        num_tokens = sum(
            int(tensors["attention_mask"].sum().item())
            for tensors in self.trajectory_tensors
        )
        num_trainable_tokens = sum(
            int((tensors["labels"].reshape(-1)[1:] != -100).sum().item())
            for tensors in self.trajectory_tensors
        )
        if (self.num_tokens, self.num_trainable_tokens) != (
            num_tokens,
            num_trainable_tokens,
        ):
            raise ValueError("SFT token counts do not match the tensor payload")
        if num_tokens < 1 or num_trainable_tokens < 1:
            raise ValueError("SFT batch must contain trainable tokens")
        if any(
            not torch.all(tensors["attention_mask"] == 1)
            for tensors in self.trajectory_tensors
        ):
            raise ValueError("SFT command tensors must be unpadded")
        if self.num_dropped_trajectories < 0:
            raise ValueError("SFT dropped trajectory count cannot be negative")
        return self

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "learning_rate": self.learning_rate,
                    "num_trajectories": self.num_trajectories,
                    "num_tokens": self.num_tokens,
                    "num_trainable_tokens": self.num_trainable_tokens,
                    "num_dropped_trajectories": self.num_dropped_trajectories,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        for tensors in self.trajectory_tensors:
            for name in ("input_ids", "attention_mask", "labels"):
                tensor = tensors[name].detach().cpu().contiguous()
                digest.update(name.encode())
                digest.update(str(tensor.dtype).encode())
                digest.update(json.dumps(tuple(tensor.shape)).encode())
                digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()


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
