"""Immutable, allowlisted snapshots of ART trajectory groups for distillation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import math
from typing import Self

from art.trajectories import (
    TokenFlag,
    Trajectory,
    TrajectoryGroup,
    tokenize_trajectory,
)

from .capture import generations
from .types import (
    CapturedGeneration,
    Example,
    ImmutableModel,
    TrainingGroupSnapshot,
    TrainingTrajectorySnapshot,
    canonical_json,
)


class CandidateTrainingBatch(ImmutableModel):
    """An immutable training projection, detached from mutable rollout objects."""

    groups: tuple[TrainingGroupSnapshot, ...]
    examples: tuple[Example, ...] = ()
    batch_id: str

    @classmethod
    def create(
        cls,
        *,
        groups: tuple[TrainingGroupSnapshot, ...],
        examples: tuple[Example, ...] = (),
    ) -> Self:
        return cls(
            groups=groups,
            examples=examples,
            batch_id=_batch_id(groups),
        )

    def model_post_init(self, __context: object) -> None:
        if not self.groups:
            raise ValueError("candidate batch must contain at least one group")
        expected_batch_id = _batch_id(self.groups)
        if self.batch_id != expected_batch_id:
            raise ValueError("candidate batch ID does not match its groups")

        captured_by_id: dict[str, CapturedGeneration] = {}
        trajectory_ids: set[str] = set()
        group_ids: set[str] = set()
        for group in self.groups:
            if group.group_id in group_ids:
                raise ValueError("candidate group identities must be unique")
            group_ids.add(group.group_id)
            if not group.trajectories:
                raise ValueError("candidate groups must contain trajectories")
            for trajectory in group.trajectories:
                if trajectory.trajectory_fingerprint in trajectory_ids:
                    raise ValueError("candidate trajectory identities must be unique")
                trajectory_ids.add(trajectory.trajectory_fingerprint)
                for generation in trajectory.generations:
                    if generation.generation_id in captured_by_id:
                        raise ValueError(
                            "candidate generation identities must be unique"
                        )
                    captured_by_id[generation.generation_id] = generation

        selected: set[str] = set()
        for example in self.examples:
            generation_id = example.generation.generation_id
            owned = captured_by_id.get(generation_id)
            if owned is None or owned != example.generation:
                raise ValueError(
                    "example generation does not belong to the candidate snapshot"
                )
            if generation_id in selected:
                raise ValueError(
                    "candidate examples must select each generation at most once"
                )
            selected.add(generation_id)

    def generation(self, generation_id: str) -> CapturedGeneration:
        """Resolve one owned capture without exposing batch-relative positions."""

        for group in self.groups:
            for trajectory in group.trajectories:
                for generation in trajectory.generations:
                    if generation.generation_id == generation_id:
                        return generation
        raise KeyError(generation_id)


def snapshot_groups(
    groups: Iterable[TrajectoryGroup],
    *,
    examples: Iterable[Example] = (),
    base_model: str | None = None,
    model: str | None = None,
    chat_template: str | None = None,
    chat_template_kwargs: Mapping[str, object] | None = None,
) -> CandidateTrainingBatch:
    """Build an exact, deterministic candidate without retaining rollout objects.

    This V1 boundary fails closed when exact inference token metadata is absent.
    It never includes trajectory/group metadata, metrics, logs, or exceptions.
    """

    source_groups = tuple(groups)
    if not source_groups:
        raise ValueError("candidate batch must contain at least one group")

    snapshots: list[TrainingGroupSnapshot] = []
    for group in source_groups:
        if group.exceptions:
            raise ValueError("candidate groups containing exceptions are unsupported")
        if not group.trajectories:
            raise ValueError("candidate groups must contain trajectories")

        rewards = tuple(_finite_reward(trajectory) for trajectory in group.trajectories)
        mean_reward = math.fsum(rewards) / len(rewards)
        trajectory_snapshots = tuple(
            _snapshot_trajectory(
                trajectory,
                reward=reward,
                advantage=reward - mean_reward,
                base_model=base_model,
                model=model,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
            )
            for trajectory, reward in zip(
                group.trajectories,
                rewards,
                strict=True,
            )
        )
        group_projection = [
            item.model_dump(mode="json", round_trip=True)
            for item in trajectory_snapshots
        ]
        snapshots.append(
            TrainingGroupSnapshot(
                group_id=hashlib.sha256(
                    b"art-distill-group-v1\0"
                    + canonical_json(group_projection).encode()
                ).hexdigest(),
                trajectories=trajectory_snapshots,
            )
        )

    return CandidateTrainingBatch.create(
        groups=tuple(snapshots),
        examples=tuple(examples),
    )


def _snapshot_trajectory(
    trajectory: Trajectory,
    *,
    reward: float,
    advantage: float,
    base_model: str | None,
    model: str | None,
    chat_template: str | None,
    chat_template_kwargs: Mapping[str, object] | None,
) -> TrainingTrajectorySnapshot:
    captured = generations(trajectory)
    if not captured:
        raise ValueError("candidate trajectories must contain a captured generation")

    tokenized = tokenize_trajectory(
        trajectory,
        base_model=base_model,
        model=model,
        chat_template=chat_template,
        chat_template_kwargs=chat_template_kwargs,
    )
    if any(not (flag & TokenFlag.EXACT) for flag in tokenized.flags):
        raise ValueError(
            "candidate construction requires exact token IDs for the full trajectory"
        )

    logprobs = tuple(
        value if math.isfinite(value) else None for value in tokenized.logprobs
    )
    return TrainingTrajectorySnapshot(
        trajectory_fingerprint=captured[0].trajectory_fingerprint,
        token_ids=tuple(tokenized.token_ids),
        logprobs=logprobs,
        token_flags=tuple(int(flag) for flag in tokenized.flags),
        reward=reward,
        advantage=advantage,
        generations=captured,
    )


def _finite_reward(trajectory: Trajectory) -> float:
    reward = float(trajectory.reward)
    if not math.isfinite(reward):
        raise ValueError("candidate trajectory rewards must be finite")
    return reward


def _batch_id(groups: tuple[TrainingGroupSnapshot, ...]) -> str:
    projection = [group.model_dump(mode="json", round_trip=True) for group in groups]
    return hashlib.sha256(
        b"art-distill-batch-v1\0" + canonical_json(projection).encode()
    ).hexdigest()
