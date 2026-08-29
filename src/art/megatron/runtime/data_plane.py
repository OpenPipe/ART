from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
import torch

from art.distributed.data_plane import MappedPackedBatch, PackedBatchRef
from art.preprocessing.pack import PackedTensors

_SFT_TENSOR_NAMES = ("input_ids", "attention_mask", "labels")


class InMemoryPackedBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    ref: PackedBatchRef
    _mapped: MappedPackedBatch | None = PrivateAttr(default=None)
    _tensors: PackedTensors | None = PrivateAttr(default=None)

    @property
    def tensors(self) -> PackedTensors:
        if self._tensors is None:
            raise RuntimeError("packed batch is closed")
        return self._tensors

    @classmethod
    def open(
        cls, ref: PackedBatchRef, local_ref: PackedBatchRef
    ) -> "InMemoryPackedBatch":
        mapped = MappedPackedBatch.open(local_ref)
        batch = cls(ref=ref)
        batch._mapped = mapped
        batch._tensors = mapped.tensors
        return batch

    def close(self) -> None:
        if self._mapped is not None:
            self._tensors = None
            self._mapped.close()
            self._mapped = None


class SFTBatchData(BaseModel):
    """Typed in-memory SFT payload sent directly to warm trainer actors."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    trajectory_tensors: tuple[dict[str, Any], ...]
    learning_rate: float
    num_trajectories: int
    num_tokens: int
    num_trainable_tokens: int
    num_dropped_trajectories: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_trajectories(self) -> "SFTBatchData":
        if not self.trajectory_tensors:
            raise ValueError("SFT batch must contain at least one trajectory")
        if self.num_trajectories != len(self.trajectory_tensors):
            raise ValueError("SFT trajectory count does not match its tensor payload")
        required = set(_SFT_TENSOR_NAMES)
        if any(set(tensors) != required for tensors in self.trajectory_tensors):
            raise ValueError("SFT trajectory tensors must have the exact schema")
        if any(
            not isinstance(tensors[name], torch.Tensor)
            or tensors[name].dtype != torch.long
            or tensors[name].device.type != "cpu"
            or not tensors[name].is_contiguous()
            for tensors in self.trajectory_tensors
            for name in _SFT_TENSOR_NAMES
        ):
            raise ValueError("SFT trajectory tensors must be contiguous CPU int64")
        if any(
            len(tensors[name].shape) != 2 or tensors[name].shape[0] != 1
            for tensors in self.trajectory_tensors
            for name in _SFT_TENSOR_NAMES
        ):
            raise ValueError("SFT trajectory tensors must have shape [1, sequence]")
        if any(
            len({tuple(tensors[name].shape) for name in _SFT_TENSOR_NAMES}) != 1
            for tensors in self.trajectory_tensors
        ):
            raise ValueError("SFT trajectory tensor shapes differ")
        if any(
            not bool(torch.all(tensors["attention_mask"] == 1).item())
            for tensors in self.trajectory_tensors
        ):
            raise ValueError("SFT command tensors must be unpadded")
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
        return self

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256(b"art-sft-batch-v1\0")
        digest.update(
            json.dumps(
                {
                    "learning_rate": self.learning_rate,
                    "num_dropped_trajectories": self.num_dropped_trajectories,
                    "num_tokens": self.num_tokens,
                    "num_trainable_tokens": self.num_trainable_tokens,
                    "num_trajectories": self.num_trajectories,
                },
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        )
        for tensors in self.trajectory_tensors:
            for name in _SFT_TENSOR_NAMES:
                tensor = tensors[name]
                digest.update(name.encode())
                digest.update(json.dumps(tuple(tensor.shape)).encode())
                digest.update(tensor.numpy().astype("<i8", copy=False).tobytes())
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
