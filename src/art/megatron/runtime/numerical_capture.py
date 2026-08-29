from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from art.distributed.data_plane import _flatten_packed_tensors
from art.training import TokenLogprobs
from art.utils.safetensors import save_safetensors

_SHA256 = r"^[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NumericalTensorArtifact(_Contract):
    kind: Literal["packed_input", "forward_result", "lora_gradients"]
    rank: int = Field(ge=0)
    relative_path: str = Field(min_length=1)
    byte_count: int = Field(gt=0)
    sha256: str = Field(pattern=_SHA256)
    source_ref: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_path(self) -> "NumericalTensorArtifact":
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or str(path) != self.relative_path:
            raise ValueError("numerical capture path must be normalized and relative")
        return self


class ForwardBackwardNumericalRankReceipt(_Contract):
    run_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    contribution_ids: tuple[str, ...] = Field(min_length=1)
    rank: int = Field(ge=0)
    packed_input: NumericalTensorArtifact | None = None
    forward_result: NumericalTensorArtifact | None = None
    lora_gradients: NumericalTensorArtifact

    @model_validator(mode="after")
    def _validate_rank(self) -> "ForwardBackwardNumericalRankReceipt":
        if self.contribution_ids[-1] != self.operation_id:
            raise ValueError("capture must identify the latest open F/B contribution")
        if self.lora_gradients.rank != self.rank:
            raise ValueError("gradient capture rank changed")
        coordinator_files = (
            self.packed_input is not None and self.forward_result is not None
        )
        if coordinator_files != (self.rank == 0):
            raise ValueError("only rank zero may own packed input and forward output")
        return self


class ForwardBackwardNumericalCaptureReceipt(_Contract):
    run_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    contribution_ids: tuple[str, ...] = Field(min_length=1)
    packed_input: NumericalTensorArtifact
    forward_result: NumericalTensorArtifact
    lora_gradients: tuple[NumericalTensorArtifact, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_files(self) -> "ForwardBackwardNumericalCaptureReceipt":
        if self.contribution_ids[-1] != self.operation_id:
            raise ValueError("capture must identify the latest open F/B contribution")
        ranks = tuple(file.rank for file in self.lora_gradients)
        if ranks != tuple(range(len(ranks))):
            raise ValueError("gradient captures must cover ordered trainer ranks")
        if self.packed_input.rank != 0 or self.forward_result.rank != 0:
            raise ValueError("rank zero must own packed input and forward output")
        return self


def capture_forward_backward_rank(
    *,
    root: str | Path,
    run_id: str,
    operation_id: str,
    contribution_ids: tuple[str, ...],
    rank: int,
    packed_tensors: Any,
    token_logprobs: tuple[TokenLogprobs, ...],
    gradients: tuple[torch.Tensor, ...],
) -> ForwardBackwardNumericalRankReceipt:
    """Persist exact gate evidence without extending the public result contract."""

    if not contribution_ids or contribution_ids[-1] != operation_id:
        raise RuntimeError("capture requires the latest open F/B contribution")
    if not gradients:
        raise RuntimeError("capture requires accumulated LoRA gradients")
    capture_root = Path(root).resolve()
    capture_root.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(f"{run_id}\0{operation_id}".encode()).hexdigest()
    directory = capture_root / f"forward-backward-{identity}"
    directory.mkdir(exist_ok=True)

    packed = forward = None
    if rank == 0:
        flat, _metadata = _flatten_packed_tensors(packed_tensors)
        packed = _write_artifact(
            capture_root,
            directory / "packed-input.safetensors",
            kind="packed_input",
            rank=rank,
            tensors={name: _cpu_tensor(value) for name, value in flat},
        )
        if not token_logprobs:
            raise RuntimeError("capture requires returned token logprobs")
        forward = _write_artifact(
            capture_root,
            directory / "forward-result.safetensors",
            kind="forward_result",
            rank=rank,
            tensors={
                f"token_logprobs/{index:06d}": torch.tensor(
                    value.to_values(), dtype=torch.float32
                ).reshape(value.shape)
                for index, value in enumerate(token_logprobs)
            },
        )
    gradient_artifact = _write_artifact(
        capture_root,
        directory / f"lora-gradients-rank-{rank:05d}.safetensors",
        kind="lora_gradients",
        rank=rank,
        tensors={
            f"parameter/{index:06d}": _cpu_tensor(value)
            for index, value in enumerate(gradients)
        },
    )
    return ForwardBackwardNumericalRankReceipt(
        run_id=run_id,
        operation_id=operation_id,
        contribution_ids=contribution_ids,
        rank=rank,
        packed_input=packed,
        forward_result=forward,
        lora_gradients=gradient_artifact,
    )


def build_forward_backward_capture(
    receipts: tuple[ForwardBackwardNumericalRankReceipt, ...],
) -> ForwardBackwardNumericalCaptureReceipt:
    ordered = tuple(sorted(receipts, key=lambda value: value.rank))
    if not ordered or tuple(receipt.rank for receipt in ordered) != tuple(
        range(len(ordered))
    ):
        raise RuntimeError("numerical capture omitted a trainer rank")
    identities = {
        (receipt.run_id, receipt.operation_id, receipt.contribution_ids)
        for receipt in ordered
    }
    if len(identities) != 1:
        raise RuntimeError("trainer ranks disagreed on numerical capture identity")
    coordinator = ordered[0]
    assert coordinator.packed_input is not None
    assert coordinator.forward_result is not None
    return ForwardBackwardNumericalCaptureReceipt(
        run_id=coordinator.run_id,
        operation_id=coordinator.operation_id,
        contribution_ids=coordinator.contribution_ids,
        packed_input=coordinator.packed_input,
        forward_result=coordinator.forward_result,
        lora_gradients=tuple(receipt.lora_gradients for receipt in ordered),
    )


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").contiguous()


def _write_artifact(
    root: Path,
    path: Path,
    *,
    kind: Literal["packed_input", "forward_result", "lora_gradients"],
    rank: int,
    tensors: dict[str, torch.Tensor],
) -> NumericalTensorArtifact:
    if not tensors:
        raise RuntimeError(f"{kind} capture is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
        candidate = Path(temporary) / path.name
        save_safetensors(tensors, candidate)
        candidate_sha256 = _file_sha256(candidate)
        candidate_bytes = candidate.stat().st_size
        try:
            os.link(candidate, path)
        except FileExistsError:
            if (
                path.stat().st_size != candidate_bytes
                or _file_sha256(path) != candidate_sha256
            ):
                raise RuntimeError(f"numerical capture changed: {path.name}")
    return NumericalTensorArtifact(
        kind=kind,
        rank=rank,
        relative_path=path.relative_to(root).as_posix(),
        byte_count=candidate_bytes,
        sha256=candidate_sha256,
        source_ref=str(path),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
