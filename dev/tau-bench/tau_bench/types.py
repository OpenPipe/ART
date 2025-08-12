# Copyright Sierra

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel

RESPOND_ACTION_NAME = "respond"
RESPOND_ACTION_FIELD_NAME = "content"


class Action(BaseModel):
    name: str
    kwargs: Dict[str, Any]


class Task(BaseModel):
    user_id: str
    actions: List[Action]
    instruction: str
    outputs: List[str]


class RewardOutputInfo(BaseModel):
    r_outputs: float
    outputs: Dict[str, bool]


class RewardActionInfo(BaseModel):
    r_actions: float
    gt_data_hash: str


class RewardResult(BaseModel):
    reward: float
    info: Union[RewardOutputInfo, RewardActionInfo]
    actions: List[Action]


class SolveResult(BaseModel):
    reward: float
    messages: List[Dict[str, Any]]
    info: Dict[str, Any]
    total_cost: Optional[float] = None


class EnvInfo(BaseModel):
    task: Task
    source: Optional[str] = None
    user_cost: Optional[float] = None
    reward_info: Optional[RewardResult] = None


class EnvResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: EnvInfo


class EnvResetResponse(BaseModel):
    observation: str
    info: EnvInfo


class EnvRunResult(BaseModel):
    task_id: int
    reward: float
    info: Dict[str, Any]
    traj: List[Dict[str, Any]]
    trial: int


class RunConfig(BaseModel):
    model_provider: str
    user_model_provider: str
    model: str = "gpt-4.1"
    user_model: str = "gpt-4o"
    num_trials: int = 1
    env: str = "retail"
    agent_strategy: str = "tool-calling"
    temperature: float = 0.0
    task_split: str = "test"
    start_index: int = 0
    end_index: int = -1
    task_ids: Optional[List[int]] = None
    log_dir: str = "results"
    max_concurrency: int = 1
    seed: int = 10
    shuffle: int = 0
    user_strategy: str = "llm"
    few_shot_displays_path: Optional[str] = None
    # art related configs
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    reward_type: str = "real"
    judge_model: str = "o3"
    max_num_steps: int = 30
    skip_eval: bool = False
    add_shadow_trajectory: bool = False
    messages_only: bool = False
    base_model: str = "unsloth/Qwen2.5-14B-Instruct"
    is_multi_gpu: bool = False
    add_no_think: bool = False
    plot_tensors: bool = False


class TauBenchTrainingConfig(BaseModel):
    """Training configuration for ART RL on tau-bench tasks"""

    trajectories_per_group: int = 6
    groups_per_step: int = 10
    learning_rate: float = 1.2e-5
    eval_steps: int = 10
    val_set_size: int = 85
    training_dataset_size: int = 30
    num_epochs: int = 50
    train_mode: str = "sync_rl"
    importance_sampling_level: Literal["token", "sequence"] = "token"


class TauBenchPolicyConfig(BaseModel):
    """Policy configuration for tau-bench agent"""

    # Training configuration
    training_config: TauBenchTrainingConfig | None = None

    # tau-bench specific configs
    run_config: RunConfig
