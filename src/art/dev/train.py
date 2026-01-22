from typing import Literal

from typing_extensions import TypedDict


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
    logprob_calculation_chunk_size: int
    mask_prob_ratio: bool
    max_negative_advantage_importance_sampling_weight: float
    num_trajectories_learning_rate_multiplier_power: float
    plot_tensors: bool
    ppo: bool
    precalculate_logprobs: bool
    scale_learning_rate_by_reward_std_dev: bool
    scale_rewards: bool
    truncated_importance_sampling: float | None


class SFTConfig(TypedDict, total=False):
    """Experimental SFT configuration options. Use at your own risk.

    Undocumented options (may change):
        instruction_part: Override auto-detected instruction marker for tokenization.
            Used to identify where user turns begin in the chat template.
        response_part: Override auto-detected response marker for tokenization.
            Used to identify where assistant turns begin (train on responses only).
    """

    instruction_part: str
    response_part: str
