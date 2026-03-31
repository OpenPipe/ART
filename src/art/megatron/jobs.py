from typing import Any, Literal

from pydantic import BaseModel, Field

from .. import types
from ..preprocessing.pack import DiskPackedTensors

DEFAULT_TRAINING_LOG_PATH = "/tmp/megatron_training_log.jsonl"
DEFAULT_JOBS_DIR = "/tmp/megatron_training_jobs"
DEFAULT_VLLM_WAKE_LOCK_PATH = "/tmp/megatron_vllm_waking"


class MegatronTrainingJob(BaseModel):
    lora_path: str
    optimizer_state_path: str
    disk_packed_tensors: DiskPackedTensors
    config: types.TrainConfig
    experimental_config: dict[str, Any]
    moe_routing_replay_path: str | None = None
    moe_routing_replay_strict: bool = True
    log_path: str = DEFAULT_TRAINING_LOG_PATH


class MergedWeightTransferInitInfo(BaseModel):
    master_address: str
    master_port: int
    rank_offset: int
    world_size: int


class MergedWeightTransferSpec(BaseModel):
    init_info: MergedWeightTransferInitInfo
    vllm_base_url: str
    served_model_name: str


class MegatronMergedTrainJob(MegatronTrainingJob):
    job_type: Literal["merged"] = "merged"
    merged_weight_transfer: MergedWeightTransferSpec


class MegatronSyncJob(BaseModel):
    job_type: Literal["sync"] = "sync"
    lora_path: str
    merged_weight_transfer: MergedWeightTransferSpec
    log_path: str = DEFAULT_TRAINING_LOG_PATH


class MegatronSFTTrainingJob(BaseModel):
    job_type: Literal["sft"] = "sft"
    lora_path: str
    optimizer_state_path: str
    sft_data_dir: str
    num_batches: int
    learning_rates: list[float]
    grad_accumulation_sequences: int | None = None
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    internal_checkpoint_interval: int | None = Field(default=None, ge=1)
    log_path: str = DEFAULT_TRAINING_LOG_PATH


MegatronJob = (
    MegatronTrainingJob
    | MegatronMergedTrainJob
    | MegatronSyncJob
    | MegatronSFTTrainingJob
)
