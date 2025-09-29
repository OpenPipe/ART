from typing import Literal

from pydantic import BaseModel


class ProjectPolicyConfig(BaseModel):
    max_turns: int = 10
    max_tokens: int = 2048
    log_to_openpipe: bool = False
    stupid_simple_reward_fn: bool = False
    include_qwen3_nothink: bool = False

    # Training configuration fields (previously in TrainingConfig)
    trajectories_per_group: int = 6
    groups_per_step: int = 1
    learning_rate: float = 1.2e-5
    eval_steps: int = 30
    val_set_size: int = 100
    training_dataset_size: int = 4000
    num_epochs: int = 4
    # Model name to use for RULER rescoring (LLM-as-a-judge). Defaults to
    ruler_judge_model: str | None = None
    minimum_reward_std_dev: float = 0.0
    # Random seed to control which subset of the training data is sampled. When None, the sampler can
    # choose its own default (e.g., derive from the current time).
    training_dataset_seed: int | None = None
    messages_only: bool = False

    # Fork configuration
    fork_from_model: str | None = None
    fork_from_project: str | None = None
    fork_not_after_step: int | None = None

    # Training configuration
    scale_rewards: bool = True  # Whether to scale rewards during training
    importance_sampling_level: Literal["token", "sequence"] = "token"
    train_on_stepwise_groups: bool = False

    # Validation configuration
    num_validation_runs: int = 1  # Number of times to run each validation entry

    # Hardware configuration
    accelerator: str = "H200:1"  # GPU accelerator type for training
