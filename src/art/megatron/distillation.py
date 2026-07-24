"""Deterministic disk packing for Megatron distillation jobs.

The dense sidecars are aligned to input-token positions, not logits positions:
``target_mask[row, j]`` means that ``tokens[row, j]`` is supervised by the
causal-model logits produced at ``j - 1``.  Position zero can therefore never
be a distillation target.

The format deliberately stores ``O(N * S * K)`` sparse teacher data and never
materializes an ``N * S * V`` teacher tensor.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Literal, TypedDict, cast

from pydantic import BaseModel, ConfigDict, Field
import torch

from ..distill.artifact import PreparedPayload, PreparedTrainingBatch
from ..distill.types import TrainingObjectives

DISTILLATION_TENSOR_SCHEMA_VERSION = 1


class DistillationObjectiveConfig(BaseModel):
    """Serializable optimizer-side subset of ``distill.Loss``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["forward_kl"] = "forward_kl"
    coefficient: float = Field(gt=0, allow_inf_nan=False)
    compensate_temperature_squared: bool = False


class DiskPackedDistillationTensors(TypedDict):
    schema_version: Literal[1]
    dir: str
    num_sequences: int
    sequence_length: int
    top_k_width: int
    target_count: int
    logical_vocab_size: int
    tensors_sha256: str


class PackedDistillationTensors(TypedDict):
    tokens: torch.Tensor
    token_mask: torch.Tensor
    input_pos: torch.Tensor
    target_mask: torch.Tensor
    distillation_weights: torch.Tensor
    topk_token_ids: torch.Tensor
    teacher_logprobs: torch.Tensor
    tail_logprobs: torch.Tensor
    temperatures: torch.Tensor


_DTYPES: dict[str, torch.dtype] = {
    "tokens": torch.long,
    "token_mask": torch.bool,
    "input_pos": torch.long,
    "target_mask": torch.bool,
    "distillation_weights": torch.float32,
    "topk_token_ids": torch.long,
    "teacher_logprobs": torch.float32,
    "tail_logprobs": torch.float32,
    "temperatures": torch.float32,
}


def validate_standalone_forward_kl(
    *,
    batch: PreparedTrainingBatch,
    objectives: TrainingObjectives,
    expected_source_revision: int,
    packed_sequence_length: int,
    tensor_parallel_size: int,
    context_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
    expert_tensor_parallel_size: int,
) -> PreparedPayload:
    """Validate every M3 capability before disk/service/optimizer mutation."""

    payload = batch.parsed_payload()
    if payload.constraints.learner_revision != expected_source_revision:
        raise ValueError(
            "prepared learner revision does not match the current model revision"
        )
    if objectives.policy is not None:
        raise ValueError(
            "M3 Megatron prepared-batch training supports standalone distillation only"
        )
    if objectives.distillation is None:
        raise ValueError("prepared-batch training requires a distillation objective")
    if objectives.distillation.divergence.kind != "forward_kl":
        raise ValueError("M3 Megatron distillation supports ForwardKL only")
    topology = (
        tensor_parallel_size,
        context_parallel_size,
        pipeline_parallel_size,
        expert_parallel_size,
        expert_tensor_parallel_size,
    )
    if topology != (1, 1, 1, 1, 1):
        raise ValueError(
            f"M3 Megatron distillation requires TP=CP=PP=EP=ETP=1; received {topology}"
        )
    if packed_sequence_length <= 0:
        raise ValueError("packed_sequence_length must be positive")
    if any(
        len(trajectory.token_ids) > packed_sequence_length
        for group in payload.groups
        for trajectory in group.trajectories
    ):
        raise ValueError(
            "prepared trajectory exceeds packed_sequence_length; "
            "distillation never truncates or drops trajectories"
        )
    if not payload.targets:
        raise ValueError("prepared distillation batch has a zero target denominator")

    widths = {len(target.token_ids) for target in payload.targets}
    if len(widths) != 1:
        raise ValueError("M3 Megatron distillation requires a fixed top-k width")
    width = next(iter(widths))
    if width >= payload.constraints.logical_vocab_size:
        raise ValueError(
            "M3 Megatron distillation requires sparse top-k targets with a tail"
        )
    if any(target.tail_logprob is None for target in payload.targets):
        raise ValueError("every sparse top-k target must include tail mass")
    temperatures = {target.temperature for target in payload.targets}
    if len(temperatures) != 1:
        raise ValueError(
            "M3 Megatron distillation requires one temperature per prepared batch"
        )
    return payload


def pack_prepared_batch(
    *,
    batch: PreparedTrainingBatch,
    payload: PreparedPayload,
    sequence_length: int,
    output_dir: str,
) -> DiskPackedDistillationTensors:
    """Join prepared targets exactly and persist deterministic fixed-K sidecars."""

    targeted_generation_ids = {target.generation_id for target in payload.targets}
    trajectories = tuple(
        trajectory
        for group in payload.groups
        for trajectory in group.trajectories
        if any(
            generation.generation_id in targeted_generation_ids
            for generation in trajectory.generations
        )
    )
    if not trajectories:
        raise ValueError("prepared distillation batch contains no trajectories")
    generation_locations: dict[str, tuple[int, int, int]] = {}
    for row, trajectory in enumerate(trajectories):
        for generation in trajectory.generations:
            generation_locations[generation.generation_id] = (
                row,
                generation.trajectory_token_start,
                len(generation.continuation_token_ids),
            )

    num_sequences = len(trajectories)
    top_k_width = len(payload.targets[0].token_ids)
    packed: PackedDistillationTensors = {
        "tokens": torch.zeros((num_sequences, sequence_length), dtype=torch.long),
        "token_mask": torch.zeros((num_sequences, sequence_length), dtype=torch.bool),
        "input_pos": torch.zeros((num_sequences, sequence_length), dtype=torch.long),
        "target_mask": torch.zeros((num_sequences, sequence_length), dtype=torch.bool),
        "distillation_weights": torch.zeros(
            (num_sequences, sequence_length), dtype=torch.float32
        ),
        "topk_token_ids": torch.full(
            (num_sequences, sequence_length, top_k_width),
            fill_value=-1,
            dtype=torch.long,
        ),
        "teacher_logprobs": torch.zeros(
            (num_sequences, sequence_length, top_k_width), dtype=torch.float32
        ),
        "tail_logprobs": torch.zeros(
            (num_sequences, sequence_length), dtype=torch.float32
        ),
        "temperatures": torch.ones(
            (num_sequences, sequence_length), dtype=torch.float32
        ),
    }
    for row, trajectory in enumerate(trajectories):
        length = len(trajectory.token_ids)
        packed["tokens"][row, :length] = torch.tensor(
            trajectory.token_ids, dtype=torch.long
        )
        packed["token_mask"][row, :length] = True
        packed["input_pos"][row, :length] = torch.arange(length, dtype=torch.long)

    joined: set[tuple[str, int]] = set()
    for target in payload.targets:
        location = generation_locations.get(target.generation_id)
        if location is None:
            raise ValueError("distillation target references an unknown generation")
        row, generation_start, generation_length = location
        if target.position >= generation_length:
            raise ValueError("distillation target exceeds generation bounds")
        token_position = generation_start + target.position
        if token_position == 0:
            raise ValueError(
                "the first trajectory token cannot be supervised by causal logits"
            )
        key = (target.generation_id, target.position)
        if key in joined:
            raise ValueError("duplicate distillation target during packing")
        joined.add(key)
        if packed["target_mask"][row, token_position]:
            raise ValueError("multiple distillation targets map to one token position")
        if int(packed["tokens"][row, token_position]) != target.sampled_token_id:
            raise ValueError("distillation target join changed the sampled token")
        packed["target_mask"][row, token_position] = True
        packed["distillation_weights"][row, token_position] = 1.0
        packed["topk_token_ids"][row, token_position] = torch.tensor(
            target.token_ids, dtype=torch.long
        )
        packed["teacher_logprobs"][row, token_position] = torch.tensor(
            target.teacher_logprobs, dtype=torch.float32
        )
        packed["tail_logprobs"][row, token_position] = cast(float, target.tail_logprob)
        packed["temperatures"][row, token_position] = target.temperature
    if len(joined) != len(payload.targets):
        raise ValueError("not every distillation target was packed exactly once")
    if int(packed["target_mask"].sum().item()) != len(payload.targets):
        raise ValueError("packed target denominator differs from prepared targets")

    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        hashes: list[tuple[str, str]] = []
        packed_items = cast(dict[str, torch.Tensor], packed)
        for name in sorted(packed_items):
            tensor = packed_items[name].contiguous()
            path = temporary / f"{name}.pt"
            with path.open("wb") as handle:
                handle.write(tensor.numpy().tobytes(order="C"))
            hashes.append((name, hashlib.sha256(path.read_bytes()).hexdigest()))
        manifest = json.dumps(
            hashes, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
        tensors_sha256 = hashlib.sha256(
            b"art-distill-tensors-v1\0" + manifest
        ).hexdigest()
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "schema_version": DISTILLATION_TENSOR_SCHEMA_VERSION,
        "dir": str(destination),
        "num_sequences": num_sequences,
        "sequence_length": sequence_length,
        "top_k_width": top_k_width,
        "target_count": len(payload.targets),
        "logical_vocab_size": payload.constraints.logical_vocab_size,
        "tensors_sha256": tensors_sha256,
    }


def packed_distillation_tensors_from_dir(
    disk: DiskPackedDistillationTensors,
) -> PackedDistillationTensors:
    """Load and checksum-verify distillation sidecars without pickle."""

    if disk["schema_version"] != DISTILLATION_TENSOR_SCHEMA_VERSION:
        raise ValueError("unsupported distillation tensor schema")
    n, s, k = (
        disk["num_sequences"],
        disk["sequence_length"],
        disk["top_k_width"],
    )
    shapes = {
        "tokens": (n, s),
        "token_mask": (n, s),
        "input_pos": (n, s),
        "target_mask": (n, s),
        "distillation_weights": (n, s),
        "topk_token_ids": (n, s, k),
        "teacher_logprobs": (n, s, k),
        "tail_logprobs": (n, s),
        "temperatures": (n, s),
    }
    hashes: list[tuple[str, str]] = []
    tensors: dict[str, torch.Tensor] = {}
    for name in sorted(shapes):
        path = Path(disk["dir"]) / f"{name}.pt"
        expected_bytes = (
            math.prod(shapes[name])
            * torch.empty((), dtype=_DTYPES[name]).element_size()
        )
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"distillation tensor {name!r} has an invalid byte length")
        raw = path.read_bytes()
        hashes.append((name, hashlib.sha256(raw).hexdigest()))
        element_count = math.prod(shapes[name])
        tensors[name] = torch.from_file(
            str(path), shared=False, size=element_count, dtype=_DTYPES[name]
        ).view(shapes[name])
    manifest = json.dumps(
        hashes, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    actual_sha256 = hashlib.sha256(b"art-distill-tensors-v1\0" + manifest).hexdigest()
    if actual_sha256 != disk["tensors_sha256"]:
        raise ValueError("distillation tensor checksum mismatch")
    target_count = int(tensors["target_mask"].sum().item())
    if target_count != disk["target_count"] or target_count <= 0:
        raise ValueError("distillation target denominator mismatch")
    target_mask = tensors["target_mask"]
    if bool(torch.any(target_mask & ~tensors["token_mask"])):
        raise ValueError("distillation targets must be a subset of real tokens")
    if bool(torch.any(target_mask[:, 0])):
        raise ValueError("the first token position cannot be a distillation target")
    expected_positions = torch.arange(s, dtype=torch.long).expand(n, s)
    if bool(
        torch.any(
            tensors["input_pos"][tensors["token_mask"]]
            != expected_positions[tensors["token_mask"]]
        )
    ):
        raise ValueError("distillation input positions are not canonical")
    weights = tensors["distillation_weights"]
    if not bool(torch.all(weights[target_mask] == 1.0)):
        raise ValueError("M3 active distillation weights must equal one")
    if not bool(torch.all(weights[~target_mask] == 0.0)):
        raise ValueError("off-mask distillation weights must equal zero")
    target_ids = tensors["topk_token_ids"][target_mask]
    if bool(
        torch.any(target_ids < 0) or torch.any(target_ids >= disk["logical_vocab_size"])
    ):
        raise ValueError("distillation target ID exceeds the logical vocabulary")
    if bool(torch.any(tensors["topk_token_ids"][~target_mask] != -1)):
        raise ValueError("off-mask distillation target IDs must be -1")
    for name in ("teacher_logprobs", "tail_logprobs", "temperatures"):
        values = tensors[name][target_mask]
        if not bool(torch.all(torch.isfinite(values))):
            raise ValueError(
                f"distillation tensor {name!r} contains non-finite targets"
            )
    if bool(torch.any(tensors["temperatures"][target_mask] <= 0)):
        raise ValueError("distillation temperatures must be positive")
    teacher_total = (
        tensors["teacher_logprobs"][target_mask].to(torch.float64).exp().sum(dim=-1)
        + tensors["tail_logprobs"][target_mask].to(torch.float64).exp()
    )
    if not bool(
        torch.allclose(
            teacher_total,
            torch.ones_like(teacher_total),
            rtol=1e-6,
            atol=1e-8,
        )
    ):
        raise ValueError("active teacher distributions are not normalized in float64")
    return cast(PackedDistillationTensors, tensors)
