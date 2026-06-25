from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .trajectories import TrajectoryGroup

TRAIN_GRADIENT_STEPS_KEY = "data/step_num_gradient_steps"
SFT_METRIC_PREFIX = "sft"
SFT_GRADIENT_STEP_KEY = "gradient_step"
SFT_WANDB_GRADIENT_STEP_KEY = f"{SFT_METRIC_PREFIX}/{SFT_GRADIENT_STEP_KEY}"
_INVARIANT_METRIC_KEYS = frozenset({TRAIN_GRADIENT_STEPS_KEY})


def average_metric_samples(
    metric_samples: Iterable[dict[str, float]],
) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    invariant_values: dict[str, float] = {}

    for sample in metric_samples:
        for key, value in sample.items():
            numeric_value = float(value)
            if key in _INVARIANT_METRIC_KEYS:
                previous_value = invariant_values.get(key)
                if previous_value is None:
                    invariant_values[key] = numeric_value
                elif previous_value != numeric_value:
                    raise ValueError(
                        f"Metric '{key}' must be invariant across samples, "
                        f"got {previous_value} and {numeric_value}."
                    )

            totals[key] = totals.get(key, 0.0) + numeric_value
            counts[key] = counts.get(key, 0) + 1

    return {
        key: (
            invariant_values[key]
            if key in _INVARIANT_METRIC_KEYS
            else totals[key] / counts[key]
        )
        for key in totals
    }


@dataclass(frozen=True)
class TrajectoryBatchSummary:
    num_scenarios: int
    num_trajectories: int
    num_groups_submitted: int
    num_groups_trainable: int
    scenario_ids: list[str]


def summarize_trajectory_groups(
    trajectory_groups: Iterable[TrajectoryGroup],
) -> TrajectoryBatchSummary:
    groups = list(trajectory_groups)
    scenario_ids: list[str] = []
    seen_scenario_ids: set[str] = set()

    for group in groups:
        scenario_id = _extract_scenario_id(group)
        if scenario_id is None or scenario_id in seen_scenario_ids:
            continue
        seen_scenario_ids.add(scenario_id)
        scenario_ids.append(scenario_id)

    return TrajectoryBatchSummary(
        num_scenarios=len(groups),
        num_trajectories=sum(
            len(group.trajectories) + len(group.exceptions) for group in groups
        ),
        num_groups_submitted=len(groups),
        num_groups_trainable=sum(1 for group in groups if _group_is_trainable(group)),
        scenario_ids=scenario_ids,
    )


def build_data_metrics_from_summary(
    summary: TrajectoryBatchSummary,
    *,
    include_trainable_groups: bool,
) -> dict[str, float]:
    metrics = {
        "data/step_num_scenarios": float(summary.num_scenarios),
        "data/step_num_trajectories": float(summary.num_trajectories),
        "data/step_num_groups_submitted": float(summary.num_groups_submitted),
    }
    if include_trainable_groups:
        metrics["data/step_num_groups_trainable"] = float(summary.num_groups_trainable)
    return metrics


def build_training_summary_metrics(
    summary: TrajectoryBatchSummary,
    *,
    include_trainable_groups: bool,
) -> dict[str, float]:
    return build_data_metrics_from_summary(
        summary,
        include_trainable_groups=include_trainable_groups,
    )


def _group_is_trainable(group: TrajectoryGroup) -> bool:
    rewards = [trajectory.reward for trajectory in group.trajectories]
    return len(rewards) > 1 and len(set(rewards)) > 1


def _extract_scenario_id(group: TrajectoryGroup) -> str | None:
    for metadata in [
        group.metadata,
        *(trajectory.metadata for trajectory in group.trajectories),
    ]:
        scenario_id = _extract_scenario_id_from_metadata(metadata)
        if scenario_id is not None:
            return scenario_id
    return None


def _extract_scenario_id_from_metadata(
    metadata: dict[str, Any],
) -> str | None:
    scenario_id = metadata.get("scenario_id")
    if scenario_id is None:
        return None
    return str(scenario_id)
