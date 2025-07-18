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
    logprob_calculation_chunk_size: int
    plot_tensors: bool
    precalculate_logprobs: bool
    scale_rewards: bool
