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

_GRADIENT_WORKLOAD_METRICS = {
    "data/gradient_step_nonpadding_logical_tokens": (
        "data/step_nonpadding_logical_tokens"
    ),
    "data/gradient_step_loss_bearing_tokens": "data/step_loss_bearing_tokens",
    "data/gradient_step_executed_token_equivalents": (
        "data/step_executed_token_equivalents"
    ),
    "data/gradient_step_nominal_schedule_capacity_tokens": (
        "data/step_nominal_schedule_capacity_tokens"
    ),
    "data/gradient_step_dummy_executed_token_equivalents": (
        "data/step_dummy_executed_token_equivalents"
    ),
    "data/gradient_step_dummy_schedule_capacity_tokens": (
        "data/step_dummy_schedule_capacity_tokens"
    ),
    "pipeline/gradient_step_real_microbatches": "pipeline/global_real_microbatches",
    "pipeline/gradient_step_dummy_microbatches": ("pipeline/global_dummy_microbatches"),
}
_GRADIENT_TRAIN_TIME = "time/gradient_step_train_s"


def _validate_loss_reduction(
    loss_type: dev.LossType | None,
    max_completion_length: int | None,
) -> None:
    if loss_type is not None and (
        not isinstance(loss_type, str) or loss_type not in {"grpo", "bnpo", "dr_grpo"}
    ):
        raise ValueError("loss_type must be one of 'grpo', 'bnpo', or 'dr_grpo'.")
    if max_completion_length is not None and (
        isinstance(max_completion_length, bool)
        or not isinstance(max_completion_length, int)
        or max_completion_length < 1
    ):
        raise ValueError("max_completion_length must be a positive integer")


def build_rl_train_configs(
    *,
    learning_rate: float,
    loss_type: dev.LossType | None = None,
    max_completion_length: int | None = None,
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
    kl_penalty_source: Literal["current_learner", "sample"] = "current_learner",
    allow_training_without_logprobs: bool | None = None,
    plot_tensors: bool | None = None,
    truncated_importance_sampling: float | None = None,
    scale_learning_rate_by_reward_std_dev: bool | None = None,
    logprob_calculation_chunk_size: int | None = None,
    packed_sequence_length: int | None = None,
    num_trajectories_learning_rate_multiplier_power: float | None = None,
    kl_ref_adapter_path: str | None = None,
    optimizer_save_interval: int = 5,
    final_training_step: int | None = None,
    grad_accumulation_sequences: int | None = None,
) -> tuple[TrainConfig, dev.TrainConfig]:
    if loss_type == "dr_grpo" and max_completion_length is None:
        max_completion_length = dev.DEFAULT_MAX_COMPLETION_LENGTH
    _validate_loss_reduction(loss_type, max_completion_length)
    config = TrainConfig(
        learning_rate=learning_rate,
        kl_penalty_coef=kl_penalty_coef,
        kl_penalty_source=kl_penalty_source,
        grad_accumulation_sequences=grad_accumulation_sequences,
        optimizer_save_interval=optimizer_save_interval,
        final_training_step=final_training_step,
    )
    dev_config: dev.TrainConfig = {
        "advantage_balance": advantage_balance,
        "importance_sampling_level": importance_sampling_level,
        "kl_penalty_coef": kl_penalty_coef,
        "kl_penalty_source": kl_penalty_source,
        "mask_prob_ratio": mask_prob_ratio,
        "ppo": ppo,
        "precalculate_logprobs": precalculate_logprobs,
        "scale_rewards": scale_rewards,
    }

    # Keep these optional so existing callers retain ART's historical BNPO
    # behavior, while allowing a backend-level setting to override the
    # trainer's native GRPOConfig when explicitly requested.
    if loss_type is not None:
        dev_config["loss_type"] = loss_type
    if max_completion_length is not None:
        dev_config["max_completion_length"] = max_completion_length

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
    if packed_sequence_length is not None:
        dev_config["packed_sequence_length"] = packed_sequence_length
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
    _aggregate_megatron_workload(training_metrics, avg_metrics)
    tokens_per_second = avg_metrics.pop("tokens_per_second", None)
    if (
        tokens_per_second is not None
        and "throughput/train_executed_tok_equiv_per_s" not in avg_metrics
    ):
        avg_metrics["throughput/train_executed_tok_equiv_per_s"] = float(
            tokens_per_second
        )
    summary = summarize_trajectory_groups(groups_list)
    avg_metrics.setdefault(
        "time/step_backend_train_s", time.monotonic() - trainer_started
    )
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


def _aggregate_megatron_workload(
    training_metrics: list[dict[str, float]],
    output: dict[str, float],
) -> None:
    raw_keys = (*_GRADIENT_WORKLOAD_METRICS, _GRADIENT_TRAIN_TIME)
    if not any(any(key in sample for key in raw_keys) for sample in training_metrics):
        return
    for index, sample in enumerate(training_metrics):
        missing = [key for key in raw_keys if key not in sample]
        if missing:
            raise ValueError(
                f"Megatron gradient-step metrics {index} are incomplete: {missing}"
            )

    totals = {
        raw_key: sum(float(sample[raw_key]) for sample in training_metrics)
        for raw_key in raw_keys
    }
    for raw_key in raw_keys:
        output.pop(raw_key, None)
    for raw_key, step_key in _GRADIENT_WORKLOAD_METRICS.items():
        output[step_key] = totals[raw_key]

    train_s = totals[_GRADIENT_TRAIN_TIME]
    output["time/step_train_s"] = train_s
    for raw_key, rate_key in (
        (
            "data/gradient_step_nonpadding_logical_tokens",
            "throughput/train_nonpadding_logical_tok_per_s",
        ),
        (
            "data/gradient_step_loss_bearing_tokens",
            "throughput/train_loss_bearing_tok_per_s",
        ),
        (
            "data/gradient_step_executed_token_equivalents",
            "throughput/train_executed_tok_equiv_per_s",
        ),
        (
            "data/gradient_step_nominal_schedule_capacity_tokens",
            "throughput/train_nominal_capacity_tok_per_s",
        ),
    ):
        output[rate_key] = totals[raw_key] / train_s if train_s > 0 else 0.0
    logical = totals["data/gradient_step_nonpadding_logical_tokens"]
    nominal = totals["data/gradient_step_nominal_schedule_capacity_tokens"]
    dummy = totals["data/gradient_step_dummy_schedule_capacity_tokens"]
    output["data/step_unused_packed_capacity_tokens"] = max(
        0.0, nominal - dummy - logical
    )
    output["data/step_unused_and_dummy_ratio"] = (
        max(0.0, nominal - logical) / nominal if nominal > 0 else 0.0
    )
