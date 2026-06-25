from typing import TYPE_CHECKING, Literal

from typing_extensions import TypedDict

from art.types import SFTMetricLoggingConfig

if TYPE_CHECKING:
    from art.megatron.routing_replay import MoeRoutingReplayBundle


class TrainConfig(TypedDict, total=False):
    advantage_balance: float
    """Balance between negative and positive advantages in the range [-1.0, 1.0]. \
-1.0 means only training on negative advantages, 1.0 means only training on \
positive advantages. Defaults to 0.0 (perfectly balanced)."""
    allow_training_without_logprobs: bool
    epsilon: float  # clip epsilon, using the same name as TRL
    epsilon_high: (
        float | None
    )  # asymmetric clip upper bound. Defaults to epsilon when None
    importance_sampling_level: Literal[
        "token", "sequence", "average", "geometric_average"
    ]
    kimi_k2_tau: float | None
    kl_penalty_coef: float
    kl_penalty_reference_step: int | None
    kl_penalty_source: Literal["current_learner", "sample"]
    kl_penalty_step_lag: int | None
    kl_ref_adapter_path: str | None
    logprob_calculation_chunk_size: int
    mask_prob_ratio: bool
    max_negative_advantage_importance_sampling_weight: float
    moe_routing_replay_bundle: "MoeRoutingReplayBundle | None"
    moe_routing_replay_path: str | None
    moe_routing_replay_strict: bool
    num_trajectories_learning_rate_multiplier_power: float
    packed_sequence_length: int | None
    plot_tensors: bool
    ppo: bool
    precalculate_logprobs: bool
    scale_learning_rate_by_reward_std_dev: bool
    scale_rewards: bool
    truncated_importance_sampling: float | None


class TrainSFTConfig(TypedDict, total=False):
    """Experimental SFT configuration options. Use at your own risk."""

    metric_logging: SFTMetricLoggingConfig
