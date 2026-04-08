from collections.abc import Iterable
import time
from typing import Literal

from . import dev
from .metrics_taxonomy import (
    average_metric_samples,
    build_training_summary_metrics,
    summarize_trajectory_groups,
)
from .trajectories import TrajectoryGroup
from .types import TrainConfig


def build_rl_train_configs(
    *,
    learning_rate: float,
    advantage_balance: float = 0.0,
    scale_rewards: bool = True,
    importance_sampling_level: Literal[
        "token", "sequence", "average", "geometric_average"
    ] = "token",
    mask_prob_ratio: bool = False,
    ppo: bool = False,
    precalculate_logprobs: bool = False,
    epsilon: float | None = None,
    epsilon_high: float | None = None,
    max_negative_advantage_importance_sampling_weight: float | None = None,
    kimi_k2_tau: float | None = None,
    kl_penalty_coef: float = 0.0,
    allow_training_without_logprobs: bool | None = None,
    plot_tensors: bool | None = None,
    truncated_importance_sampling: float | None = None,
    scale_learning_rate_by_reward_std_dev: bool | None = None,
    logprob_calculation_chunk_size: int | None = None,
    num_trajectories_learning_rate_multiplier_power: float | None = None,
    kl_ref_adapter_path: str | None = None,
) -> tuple[TrainConfig, dev.TrainConfig]:
    config = TrainConfig(
        learning_rate=learning_rate,
        kl_penalty_coef=kl_penalty_coef,
    )
    dev_config: dev.TrainConfig = {
        "advantage_balance": advantage_balance,
        "importance_sampling_level": importance_sampling_level,
        "kl_penalty_coef": kl_penalty_coef,
        "mask_prob_ratio": mask_prob_ratio,
        "ppo": ppo,
        "precalculate_logprobs": precalculate_logprobs,
        "scale_rewards": scale_rewards,
    }

    if allow_training_without_logprobs is not None:
        dev_config["allow_training_without_logprobs"] = allow_training_without_logprobs
    if plot_tensors is not None:
        dev_config["plot_tensors"] = plot_tensors
    if truncated_importance_sampling is not None:
        dev_config["truncated_importance_sampling"] = truncated_importance_sampling
    if scale_learning_rate_by_reward_std_dev is not None:
        dev_config["scale_learning_rate_by_reward_std_dev"] = (
            scale_learning_rate_by_reward_std_dev
        )
    if logprob_calculation_chunk_size is not None:
        dev_config["logprob_calculation_chunk_size"] = logprob_calculation_chunk_size
    if num_trajectories_learning_rate_multiplier_power is not None:
        dev_config["num_trajectories_learning_rate_multiplier_power"] = (
            num_trajectories_learning_rate_multiplier_power
        )
    if epsilon is not None:
        dev_config["epsilon"] = epsilon
    if epsilon_high is not None:
        dev_config["epsilon_high"] = epsilon_high
    if max_negative_advantage_importance_sampling_weight is not None:
        dev_config["max_negative_advantage_importance_sampling_weight"] = (
            max_negative_advantage_importance_sampling_weight
        )
    if kimi_k2_tau is not None:
        dev_config["kimi_k2_tau"] = kimi_k2_tau
    if kl_ref_adapter_path is not None:
        dev_config["kl_ref_adapter_path"] = kl_ref_adapter_path

    return config, dev_config


def aggregate_rl_training_metrics(
    *,
    training_metrics: list[dict[str, float]],
    trajectory_groups: Iterable[TrajectoryGroup],
    trainer_started: float,
) -> dict[str, float]:
    groups_list = list(trajectory_groups)
    avg_metrics = average_metric_samples(training_metrics)
    summary = summarize_trajectory_groups(groups_list)
    avg_metrics.setdefault("time/step_trainer_s", time.monotonic() - trainer_started)
    avg_metrics.update(
        {
            key: value
            for key, value in build_training_summary_metrics(
                summary,
                include_trainable_groups=True,
            ).items()
            if key not in avg_metrics
        }
    )
    return avg_metrics
