"""Deterministic disk packing for Megatron distillation jobs.

The dense sidecars are aligned to input-token positions, not logits positions:
``target_mask[row, j]`` means that ``tokens[row, j]`` is supervised by the
causal-model logits produced at ``j - 1``.  Position zero can therefore never
be a distillation target.

The format deliberately stores ``O(N * S * K)`` sparse teacher data and never
materializes an ``N * S * V`` teacher tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from typing_extensions import NotRequired

from ..distill.artifact import PreparedPayload, PreparedTrainingBatch
from ..distill.types import (
    CurrentStep,
    StudentOnPolicy,
    TrainingObjectives,
    TrainingTrajectorySnapshot,
)
from ..trajectories import TokenFlag

DISTILLATION_TENSOR_SCHEMA_VERSION = 1
PREPARED_TENSOR_LAYOUT_VERSION = 2


class CispoObjectiveConfig(BaseModel):
    """Complete resolved CISPO contract supported by prepared Megatron jobs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["cispo"] = "cispo"
    epsilon: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    epsilon_high: float = Field(default=4.0, ge=0, allow_inf_nan=False)
    importance_sampling_level: Literal["token"] = "token"
    scale_rewards: bool = True
    advantage_balance: float = Field(default=0.0, ge=-1.0, le=1.0)


class DistillationObjectiveConfig(BaseModel):
    """Serializable complete objective contract for a prepared update."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["forward_kl"] = "forward_kl"
    coefficient: float = Field(gt=0, allow_inf_nan=False)
    compensate_temperature_squared: bool = False
    policy: CispoObjectiveConfig | None = None


class PolicyPackingConfig(BaseModel):
    """Packing-time subset of legacy CISPO preprocessing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scale_rewards: bool = True
    advantage_balance: float = Field(default=0.0, ge=-1.0, le=1.0)


class DiskPackedDistillationTensors(TypedDict):
    schema_version: Literal[1]
    layout_version: NotRequired[Literal[2]]
    dir: str
    num_sequences: int
    sequence_length: int
    top_k_width: int
    target_count: int
    policy_count: NotRequired[int]
    policy_kind: NotRequired[Literal["cispo"]]
    logical_vocab_size: int
    tensors_sha256: str


class PackedDistillationTensors(TypedDict):
    tokens: torch.Tensor
    token_mask: torch.Tensor
    input_pos: torch.Tensor
    source_group_ids: NotRequired[torch.Tensor]
    policy_mask: NotRequired[torch.Tensor]
    old_logprobs: NotRequired[torch.Tensor]
    policy_advantages: NotRequired[torch.Tensor]
    policy_weights: NotRequired[torch.Tensor]
    policy_group_ids: NotRequired[torch.Tensor]
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
    "source_group_ids": torch.long,
    "policy_mask": torch.bool,
    "old_logprobs": torch.float32,
    "policy_advantages": torch.float32,
    "policy_weights": torch.float32,
    "policy_group_ids": torch.long,
    "target_mask": torch.bool,
    "distillation_weights": torch.float32,
    "topk_token_ids": torch.long,
    "teacher_logprobs": torch.float32,
    "tail_logprobs": torch.float32,
    "temperatures": torch.float32,
}


@dataclass(frozen=True, slots=True)
class _PolicyTrajectoryProjection:
    source_group_index: int
    trajectory: TrainingTrajectorySnapshot
    positions: tuple[int, ...]
    advantage: float
    weight: float


def validate_prepared_forward_kl(
    *,
    batch: PreparedTrainingBatch,
    objectives: TrainingObjectives,
    expected_source_revision: int,
    packed_sequence_length: int,
    policy_config: PolicyPackingConfig | None = None,
) -> PreparedPayload:
    """Validate the topology-independent prepared-data contract.

    This boundary supports standalone KD and additive CISPO plus KD. Backend and
    worker topology capability checks remain outside this packing module.
    """

    payload = batch.parsed_payload()
    if payload.constraints.learner_revision != expected_source_revision:
        raise ValueError(
            "prepared learner revision does not match the current model revision"
        )
    if objectives.policy not in (None, "cispo"):
        raise ValueError("prepared-batch policy training currently supports CISPO only")
    if objectives.distillation is None:
        raise ValueError("prepared-batch training requires a distillation objective")
    if objectives.distillation.divergence.kind != "forward_kl":
        raise ValueError("Megatron distillation supports ForwardKL only")
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
    _validate_fixed_sparse_targets(payload)
    if objectives.policy is not None:
        projections = _policy_projections(
            payload,
            config=policy_config or PolicyPackingConfig(),
        )
        if sum(len(projection.positions) for projection in projections) <= 0:
            raise ValueError("prepared policy objective has a zero token denominator")
    return payload


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

    if objectives.policy is not None:
        raise ValueError(
            "M3 Megatron prepared-batch training supports standalone distillation only"
        )
    payload = validate_prepared_forward_kl(
        batch=batch,
        objectives=objectives,
        expected_source_revision=expected_source_revision,
        packed_sequence_length=packed_sequence_length,
    )
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
    return payload


def _validate_fixed_sparse_targets(payload: PreparedPayload) -> None:
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


def _policy_projections(
    payload: PreparedPayload,
    *,
    config: PolicyPackingConfig,
) -> tuple[_PolicyTrajectoryProjection, ...]:
    """Reproduce legacy default CISPO preprocessing from immutable snapshots."""

    provisional: list[_PolicyTrajectoryProjection] = []
    for source_group_index, group in enumerate(payload.groups):
        rewards = tuple(trajectory.reward for trajectory in group.trajectories)
        reward_mean = math.fsum(rewards) / len(rewards)
        reward_std = math.sqrt(
            math.fsum((reward - reward_mean) ** 2 for reward in rewards) / len(rewards)
        )
        for trajectory, reward in zip(group.trajectories, rewards, strict=True):
            advantage = reward - reward_mean
            if config.scale_rewards:
                advantage /= reward_std + 1e-6
            if config.advantage_balance > 0.0 and advantage < 0.0:
                advantage *= 1.0 - config.advantage_balance
            elif config.advantage_balance < 0.0 and advantage > 0.0:
                advantage *= 1.0 + config.advantage_balance

            sampled_positions = tuple(
                position
                for position, flag in enumerate(trajectory.token_flags)
                if TokenFlag(flag) & TokenFlag.SAMPLED
            )
            positions = sampled_positions if advantage != 0.0 else ()
            if positions:
                _validate_policy_positions(
                    trajectory,
                    positions=positions,
                    payload=payload,
                )
            provisional.append(
                _PolicyTrajectoryProjection(
                    source_group_index=source_group_index,
                    trajectory=trajectory,
                    positions=positions,
                    advantage=advantage,
                    weight=(
                        0.0 if not positions else 1.0 / (len(sampled_positions) + 1e-6)
                    ),
                )
            )

    policy_count = sum(len(projection.positions) for projection in provisional)
    if policy_count <= 0:
        return tuple(provisional)
    mean_weight = (
        math.fsum(
            projection.weight * len(projection.positions) for projection in provisional
        )
        / policy_count
    )
    if not math.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError("prepared policy weights have an invalid normalization")
    normalized_weights = tuple(
        projection.weight / mean_weight if projection.positions else 0.0
        for projection in provisional
    )
    advantage_scale = (
        math.fsum(
            abs(projection.advantage) * weight * len(projection.positions)
            for projection, weight in zip(provisional, normalized_weights, strict=True)
        )
        / policy_count
    )
    if not math.isfinite(advantage_scale) or advantage_scale <= 0:
        raise ValueError("prepared policy objective has a zero token denominator")

    return tuple(
        _PolicyTrajectoryProjection(
            source_group_index=projection.source_group_index,
            trajectory=projection.trajectory,
            positions=projection.positions,
            advantage=(
                projection.advantage / advantage_scale if projection.positions else 0.0
            ),
            weight=weight,
        )
        for projection, weight in zip(provisional, normalized_weights, strict=True)
    )


def _validate_policy_positions(
    trajectory: TrainingTrajectorySnapshot,
    *,
    positions: tuple[int, ...],
    payload: PreparedPayload,
) -> None:
    revisions: dict[int, int] = {}
    for generation in trajectory.generations:
        for span in generation.rollout_spans:
            for local_position in range(span.start, span.end):
                token_position = generation.trajectory_token_start + local_position
                if token_position in revisions:
                    raise ValueError(
                        "policy token has ambiguous rollout revision provenance"
                    )
                revisions[token_position] = span.revision

    require_current_revision = isinstance(
        payload.constraints.rollout_requirement, StudentOnPolicy
    ) or isinstance(payload.constraints.consistency, CurrentStep)
    for position in positions:
        if position == 0:
            raise ValueError(
                "the first trajectory token cannot be supervised by causal logits"
            )
        logprob = trajectory.logprobs[position]
        if logprob is None or not math.isfinite(logprob):
            raise ValueError("policy-eligible token is missing its rollout logprob")
        if logprob > 0.0:
            raise ValueError("policy-eligible rollout logprob must be non-positive")
        revision = revisions.get(position)
        if revision is None:
            raise ValueError(
                "policy-eligible token lacks exact rollout revision provenance"
            )
        if (
            require_current_revision
            and revision != payload.constraints.learner_revision
        ):
            raise ValueError(
                "policy-eligible rollout revision does not match learner revision"
            )


def pack_prepared_batch(
    *,
    batch: PreparedTrainingBatch,
    payload: PreparedPayload,
    sequence_length: int,
    output_dir: str,
    objectives: TrainingObjectives | None = None,
    policy_config: PolicyPackingConfig | None = None,
) -> DiskPackedDistillationTensors:
    """Join immutable policy/KD projections into deterministic fixed-K sidecars.

    Omitting ``objectives`` preserves the M3 standalone-KD packing behavior.
    Supplying an additive CISPO objective retains every original trajectory so a
    teacher failure or unselected generation cannot change the policy cohort.
    """

    targeted_generation_ids = {target.generation_id for target in payload.targets}
    policy_enabled = objectives is not None and objectives.policy is not None
    resolved_policy_config = policy_config or PolicyPackingConfig()
    if objectives is not None:
        validated = validate_prepared_forward_kl(
            batch=batch,
            objectives=objectives,
            expected_source_revision=payload.constraints.learner_revision,
            packed_sequence_length=sequence_length,
            policy_config=resolved_policy_config,
        )
        if validated != payload:
            raise ValueError("prepared payload does not match its batch envelope")
    else:
        if batch.parsed_payload() != payload:
            raise ValueError("prepared payload does not match its batch envelope")
        _validate_fixed_sparse_targets(payload)

    all_trajectories = tuple(
        (source_group_index, trajectory)
        for source_group_index, group in enumerate(payload.groups)
        for trajectory in group.trajectories
    )
    trajectories_with_groups = (
        all_trajectories
        if policy_enabled
        else tuple(
            (source_group_index, trajectory)
            for source_group_index, trajectory in all_trajectories
            if any(
                generation.generation_id in targeted_generation_ids
                for generation in trajectory.generations
            )
        )
    )
    trajectories = tuple(trajectory for _, trajectory in trajectories_with_groups)
    if not trajectories:
        raise ValueError("prepared distillation batch contains no trajectories")

    policy_projections = (
        _policy_projections(payload, config=resolved_policy_config)
        if policy_enabled
        else ()
    )
    policy_by_fingerprint = {
        projection.trajectory.trajectory_fingerprint: projection
        for projection in policy_projections
    }
    policy_count = sum(len(projection.positions) for projection in policy_projections)
    if policy_enabled and policy_count <= 0:
        raise ValueError("prepared policy objective has a zero token denominator")

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
        "source_group_ids": torch.zeros((num_sequences,), dtype=torch.long),
        "policy_mask": torch.zeros((num_sequences, sequence_length), dtype=torch.bool),
        "old_logprobs": torch.full(
            (num_sequences, sequence_length),
            fill_value=float("nan"),
            dtype=torch.float32,
        ),
        "policy_advantages": torch.zeros(
            (num_sequences, sequence_length), dtype=torch.float32
        ),
        "policy_weights": torch.zeros(
            (num_sequences, sequence_length), dtype=torch.float32
        ),
        "policy_group_ids": torch.full(
            (num_sequences, sequence_length), fill_value=-1, dtype=torch.long
        ),
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
    for row, (source_group_index, trajectory) in enumerate(trajectories_with_groups):
        length = len(trajectory.token_ids)
        packed["tokens"][row, :length] = torch.tensor(
            trajectory.token_ids, dtype=torch.long
        )
        packed["token_mask"][row, :length] = True
        packed["input_pos"][row, :length] = torch.arange(length, dtype=torch.long)
        packed["source_group_ids"][row] = source_group_index
        projection = policy_by_fingerprint.get(trajectory.trajectory_fingerprint)
        if projection is None:
            continue
        positions = torch.tensor(projection.positions, dtype=torch.long)
        packed["policy_mask"][row, positions] = True
        packed["old_logprobs"][row, positions] = torch.tensor(
            [
                cast(float, trajectory.logprobs[position])
                for position in projection.positions
            ],
            dtype=torch.float32,
        )
        packed["policy_advantages"][row, positions] = projection.advantage
        packed["policy_weights"][row, positions] = projection.weight
        packed["policy_group_ids"][row, positions] = row

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
        "layout_version": PREPARED_TENSOR_LAYOUT_VERSION,
        "dir": str(destination),
        "num_sequences": num_sequences,
        "sequence_length": sequence_length,
        "top_k_width": top_k_width,
        "target_count": len(payload.targets),
        "policy_count": policy_count,
        **({"policy_kind": "cispo"} if policy_enabled else {}),
        "logical_vocab_size": payload.constraints.logical_vocab_size,
        "tensors_sha256": tensors_sha256,
    }


def packed_distillation_tensors_from_dir(
    disk: DiskPackedDistillationTensors,
) -> PackedDistillationTensors:
    """Load and checksum-verify distillation sidecars without pickle."""

    if disk["schema_version"] != DISTILLATION_TENSOR_SCHEMA_VERSION:
        raise ValueError("unsupported distillation tensor schema")
    layout_version = disk.get("layout_version")
    if layout_version not in (None, PREPARED_TENSOR_LAYOUT_VERSION):
        raise ValueError("unsupported prepared tensor layout")
    extended_layout = layout_version == PREPARED_TENSOR_LAYOUT_VERSION
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
    if extended_layout:
        shapes.update(
            {
                "source_group_ids": (n,),
                "policy_mask": (n, s),
                "old_logprobs": (n, s),
                "policy_advantages": (n, s),
                "policy_weights": (n, s),
                "policy_group_ids": (n, s),
            }
        )
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
    if not extended_layout:
        tensors.update(_empty_policy_sidecars(num_sequences=n, sequence_length=s))
    _validate_common_sidecars(tensors, disk=disk)
    _validate_policy_sidecars(tensors, disk=disk, extended_layout=extended_layout)
    _validate_distillation_sidecars(tensors, disk=disk)
    return cast(PackedDistillationTensors, tensors)


def _empty_policy_sidecars(
    *,
    num_sequences: int,
    sequence_length: int,
) -> dict[str, torch.Tensor]:
    """Adapt an original M3 sidecar directory to the extended in-memory shape."""

    return {
        "source_group_ids": torch.arange(num_sequences, dtype=torch.long),
        "policy_mask": torch.zeros((num_sequences, sequence_length), dtype=torch.bool),
        "old_logprobs": torch.full(
            (num_sequences, sequence_length),
            fill_value=float("nan"),
            dtype=torch.float32,
        ),
        "policy_advantages": torch.zeros(
            (num_sequences, sequence_length), dtype=torch.float32
        ),
        "policy_weights": torch.zeros(
            (num_sequences, sequence_length), dtype=torch.float32
        ),
        "policy_group_ids": torch.full(
            (num_sequences, sequence_length), fill_value=-1, dtype=torch.long
        ),
    }


def _validate_common_sidecars(
    tensors: dict[str, torch.Tensor],
    *,
    disk: DiskPackedDistillationTensors,
) -> None:
    n, s = disk["num_sequences"], disk["sequence_length"]
    if n <= 0 or s <= 0:
        raise ValueError("prepared tensor dimensions must be positive")
    token_mask = tensors["token_mask"]
    token_lengths = token_mask.sum(dim=1, keepdim=True)
    expected_token_mask = torch.arange(s).unsqueeze(0) < token_lengths
    if not torch.equal(token_mask, expected_token_mask):
        raise ValueError("prepared token mask must describe contiguous prefixes")
    expected_positions = torch.arange(s, dtype=torch.long).expand(n, s)
    if bool(
        torch.any(tensors["input_pos"][token_mask] != expected_positions[token_mask])
    ):
        raise ValueError("distillation input positions are not canonical")
    if bool(torch.any(tensors["input_pos"][~token_mask] != 0)):
        raise ValueError("off-token input positions must equal zero")
    source_group_ids = tensors["source_group_ids"]
    if bool(torch.any(source_group_ids < 0)):
        raise ValueError("source group IDs must be non-negative")
    observed_group_ids = source_group_ids.tolist()
    if observed_group_ids != sorted(observed_group_ids):
        raise ValueError("source group IDs must preserve prepared group order")
    if observed_group_ids and set(observed_group_ids) != set(
        range(max(observed_group_ids) + 1)
    ):
        raise ValueError("source group IDs must be contiguous")


def _validate_policy_sidecars(
    tensors: dict[str, torch.Tensor],
    *,
    disk: DiskPackedDistillationTensors,
    extended_layout: bool,
) -> None:
    policy_mask = tensors["policy_mask"]
    if bool(torch.any(policy_mask & ~tensors["token_mask"])):
        raise ValueError("policy mask must be a subset of real tokens")
    if bool(torch.any(policy_mask[:, 0])):
        raise ValueError("the first token position cannot be a policy target")
    policy_count = int(policy_mask.sum().item())
    declared_count = disk.get("policy_count", 0)
    if policy_count != declared_count:
        raise ValueError("prepared policy denominator mismatch")
    policy_kind = disk.get("policy_kind")
    if policy_kind is None:
        if policy_count != 0:
            raise ValueError("policy sidecars require a declared policy objective")
    elif policy_kind == "cispo":
        if policy_count <= 0:
            raise ValueError("enabled policy objective has a zero token denominator")
    else:
        raise ValueError("unsupported prepared policy objective")
    if not extended_layout and (declared_count != 0 or policy_kind is not None):
        raise ValueError("original M3 sidecars cannot declare a policy objective")

    old_logprobs = tensors["old_logprobs"]
    active_logprobs = old_logprobs[policy_mask]
    if not bool(
        torch.all(torch.isfinite(active_logprobs)) and torch.all(active_logprobs <= 0)
    ):
        raise ValueError(
            "active policy rollout logprobs must be finite and non-positive"
        )
    if not bool(torch.all(torch.isnan(old_logprobs[~policy_mask]))):
        raise ValueError("off-mask policy rollout logprobs must be NaN")

    advantages = tensors["policy_advantages"]
    weights = tensors["policy_weights"]
    if not bool(
        torch.all(torch.isfinite(advantages[policy_mask]))
        and torch.all(advantages[policy_mask] != 0)
    ):
        raise ValueError("active policy advantages must be finite and nonzero")
    if not bool(
        torch.all(torch.isfinite(weights[policy_mask]))
        and torch.all(weights[policy_mask] > 0)
    ):
        raise ValueError("active policy weights must be finite and positive")
    if not bool(torch.all(advantages[~policy_mask] == 0)):
        raise ValueError("off-mask policy advantages must equal zero")
    if not bool(torch.all(weights[~policy_mask] == 0)):
        raise ValueError("off-mask policy weights must equal zero")

    expected_group_ids = (
        torch.arange(policy_mask.shape[0], dtype=torch.long)
        .unsqueeze(1)
        .expand_as(tensors["policy_group_ids"])
    )
    group_ids = tensors["policy_group_ids"]
    if not torch.equal(group_ids[policy_mask], expected_group_ids[policy_mask]):
        raise ValueError("active policy group IDs must identify their trajectory row")
    if not bool(torch.all(group_ids[~policy_mask] == -1)):
        raise ValueError("off-mask policy group IDs must equal -1")


def _validate_distillation_sidecars(
    tensors: dict[str, torch.Tensor],
    *,
    disk: DiskPackedDistillationTensors,
) -> None:
    target_count = int(tensors["target_mask"].sum().item())
    if target_count != disk["target_count"] or target_count <= 0:
        raise ValueError("distillation target denominator mismatch")
    target_mask = tensors["target_mask"]
    if bool(torch.any(target_mask & ~tensors["token_mask"])):
        raise ValueError("distillation targets must be a subset of real tokens")
    if bool(torch.any(target_mask[:, 0])):
        raise ValueError("the first token position cannot be a distillation target")
    weights = tensors["distillation_weights"]
    if not bool(
        torch.all(torch.isfinite(weights[target_mask]))
        and torch.all(weights[target_mask] > 0)
    ):
        raise ValueError("active distillation weights must be finite and positive")
    if not bool(torch.all(weights[~target_mask] == 0.0)):
        raise ValueError("off-mask distillation weights must equal zero")
    target_ids = tensors["topk_token_ids"][target_mask]
    if bool(
        torch.any(target_ids < 0) or torch.any(target_ids >= disk["logical_vocab_size"])
    ):
        raise ValueError("distillation target ID exceeds the logical vocabulary")
    if bool(torch.any(tensors["topk_token_ids"][~target_mask] != -1)):
        raise ValueError("off-mask distillation target IDs must be -1")
    if target_ids.shape[-1] > 1:
        sorted_ids = target_ids.sort(dim=-1).values
        if bool(torch.any(sorted_ids[:, 1:] == sorted_ids[:, :-1])):
            raise ValueError("active distillation target IDs must be unique")
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
