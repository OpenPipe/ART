from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator
import torch

from art.distributed.data_plane import (
    MappedPackedBatch,
    MappedSftBatch,
    PackedBatchRef,
    SftBatchManifest,
    SftBatchRef,
    TensorSpec,
)
from art.preprocessing.pack import PackedTensors

_SFT_TENSOR_NAMES = ("input_ids", "attention_mask", "labels")


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
    """Typed SFT tensors before distribution or while mapped on one trainer host."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    trajectory_tensors: tuple[dict[str, Any], ...]
    learning_rate: float = 0.0
    num_trajectories: int
    num_tokens: int
    num_trainable_tokens: int
    num_dropped_trajectories: int = 0
    _fingerprint_cache: tuple[tuple[tuple[torch.Tensor, int], ...], str] | None = (
        PrivateAttr(default=None)
    )
    _mapped: MappedSftBatch | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_trajectories(self) -> "SFTBatchData":
        if not self.trajectory_tensors:
            raise ValueError("SFT batch must contain at least one trajectory")
        if self.num_trajectories != len(self.trajectory_tensors):
            raise ValueError("SFT trajectory count does not match its tensor payload")
        required = set(_SFT_TENSOR_NAMES)
        if any(not required <= tensors.keys() for tensors in self.trajectory_tensors):
            raise ValueError("SFT trajectory tensors are incomplete")
        if any(
            not isinstance(tensors[name], torch.Tensor)
            or tensors[name].dtype != torch.long
            or tensors[name].device.type != "cpu"
            or not tensors[name].is_contiguous()
            for tensors in self.trajectory_tensors
            for name in required
        ):
            raise ValueError("SFT trajectory tensors must be contiguous CPU int64")
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

    @staticmethod
    def storage_upper_bound(
        *, num_trajectories: int, max_sequence_length: int
    ) -> int:
        if min(num_trajectories, max_sequence_length) < 1:
            raise ValueError("SFT storage dimensions must be positive")
        return (
            num_trajectories
            * max_sequence_length
            * len(_SFT_TENSOR_NAMES)
            * torch.empty((), dtype=torch.long).element_size()
        )

    @property
    def storage_byte_count(self) -> int:
        return sum(
            tensors[name].numel() * tensors[name].element_size()
            for tensors in self.trajectory_tensors
            for name in _SFT_TENSOR_NAMES
        )

    def distribution_payload(
        self, batch_id: str
    ) -> tuple[SftBatchManifest, tuple[memoryview, ...]]:
        flat = tuple(
            (f"{index}/{name}", tensors[name])
            for index, tensors in enumerate(self.trajectory_tensors)
            for name in _SFT_TENSOR_NAMES
        )
        offset = 0
        specs = []
        chunks = []
        for name, tensor in flat:
            chunk = memoryview(tensor.numpy()).cast("B")
            specs.append(
                TensorSpec(
                    name=name,
                    dtype="int64",
                    shape=tuple(tensor.shape),
                    offset=offset,
                    byte_count=len(chunk),
                )
            )
            chunks.append(chunk)
            offset += len(chunk)
        return (
            SftBatchManifest(
                batch_id=batch_id,
                tensors=tuple(specs),
                storage_byte_count=offset,
                learning_rate=self.learning_rate,
                num_trajectories=self.num_trajectories,
                num_tokens=self.num_tokens,
                num_trainable_tokens=self.num_trainable_tokens,
                num_dropped_trajectories=self.num_dropped_trajectories,
                fingerprint=self.fingerprint,
            ),
            tuple(chunks),
        )

    @classmethod
    def open(cls, manifest: SftBatchManifest, local_ref: SftBatchRef) -> "SFTBatchData":
        if local_ref.manifest != manifest:
            raise ValueError("SFT host lease differs from its logical manifest")
        mapped = MappedSftBatch.open(local_ref)
        # The controller validated these immutable tensors before streaming them.
        # Re-running semantic scans here would read the full batch once per rank.
        batch = cls.model_construct(
            trajectory_tensors=tuple(
                {
                    name: mapped.tensors[f"{index}/{name}"]
                    for name in _SFT_TENSOR_NAMES
                }
                for index in range(manifest.num_trajectories)
            ),
            learning_rate=manifest.learning_rate,
            num_trajectories=manifest.num_trajectories,
            num_tokens=manifest.num_tokens,
            num_trainable_tokens=manifest.num_trainable_tokens,
            num_dropped_trajectories=manifest.num_dropped_trajectories,
        )
        batch._mapped = mapped
        tensors = tuple(
            values[name]
            for values in batch.trajectory_tensors
            for name in _SFT_TENSOR_NAMES
        )
        batch._fingerprint_cache = (
            tuple((tensor, int(tensor._version)) for tensor in tensors),
            manifest.fingerprint,
        )
        return batch

    def close(self) -> None:
        if self._mapped is not None:
            self._mapped.close()
            self._mapped = None

    @property
    def fingerprint(self) -> str:
        tensors = tuple(
            values[name]
            for values in self.trajectory_tensors
            for name in ("input_ids", "attention_mask", "labels")
        )
        try:
            versions = tuple((tensor, int(tensor._version)) for tensor in tensors)
        except RuntimeError:
            versions = None
        cached = self._fingerprint_cache
        if (
            cached is not None
            and versions is not None
            and len(cached[0]) == len(versions)
        ):
            if all(
                prior is current and prior_version == current_version
                for (prior, prior_version), (current, current_version) in zip(
                    cached[0], versions, strict=True
                )
            ):
                return cached[1]
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
                digest.update(memoryview(tensor.numpy()))
        result = digest.hexdigest()
        if versions is not None:
            assert self.__pydantic_private__ is not None
            self.__pydantic_private__["_fingerprint_cache"] = (versions, result)
        return result

    def __getstate__(self) -> dict[Any, Any]:
        state = super().__getstate__()
        private = dict(state["__pydantic_private__"] or {})
        private["_fingerprint_cache"] = None
        state["__pydantic_private__"] = private
        return state


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
