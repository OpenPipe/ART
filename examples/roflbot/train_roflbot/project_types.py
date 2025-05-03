from pydantic import BaseModel


class TrainingConfig(BaseModel):
    trajectories_per_group: int = 6
    groups_per_step: int = 1
    learning_rate: float = 1.2e-5
    eval_steps: int = 30
    val_set_size: int = 100
    training_dataset_size: int = 4000
    num_epochs: int = 4
    gpus: str = "H100-SXM:1"


class PolicyConfig(BaseModel):
    max_turns: int = 10
    max_tokens: int = 2048
    thinking_char_budget: int = 100

    training_config: TrainingConfig | None = None
