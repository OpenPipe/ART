from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, Field, TypeAdapter

from ... import types
from ...preprocessing.pack import DiskPackedTensors

DEFAULT_TRAINING_LOG_PATH = "/tmp/megatron_training_log.jsonl"
DEFAULT_JOBS_DIR = "/tmp/megatron_training_jobs"
DEFAULT_VLLM_WAKE_LOCK_PATH = "/tmp/megatron_vllm_waking"
LORA_READY_EVENT = "lora_ready"


class MergedWeightTransferInitInfo(BaseModel):
    master_address: str
    master_port: int
    rank_offset: int
    world_size: int


class MergedWeightTransferSpec(BaseModel):
    init_info: MergedWeightTransferInitInfo
    vllm_base_url: str
    served_model_name: str
    api_key: str | None = None
    nccl_so_path: str | None = None


class _MegatronTrainingJobBase(BaseModel):
    step: int = Field(default=0, ge=0)
    lora_path: str
    allow_unvalidated_arch: bool = False
    optimizer_state_path: str
    disk_packed_tensors: DiskPackedTensors
    config: types.TrainConfig
    experimental_config: dict[str, Any]
    moe_routing_replay_path: str | None = None
    moe_routing_replay_strict: bool = True
    log_path: str = DEFAULT_TRAINING_LOG_PATH


class MegatronTrainingJob(_MegatronTrainingJobBase):
    kind: Literal["train_lora"] = "train_lora"


class MegatronMergedTrainingJob(_MegatronTrainingJobBase):
    kind: Literal["train_merged"] = "train_merged"
    merged_weight_transfer: MergedWeightTransferSpec


class MegatronSyncJob(BaseModel):
    kind: Literal["sync"] = "sync"
    lora_path: str
    allow_unvalidated_arch: bool = False
    merged_weight_transfer: MergedWeightTransferSpec
    log_path: str = DEFAULT_TRAINING_LOG_PATH


class MegatronSFTTrainingJob(BaseModel):
    kind: Literal["sft"] = "sft"
    lora_path: str
    allow_unvalidated_arch: bool = False
    optimizer_state_path: str
    sft_data_dir: str
    num_batches: int
    learning_rates: list[float]
    grad_accumulation_sequences: int | None = None
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    internal_checkpoint_interval: int | None = Field(default=None, ge=1)
    log_path: str = DEFAULT_TRAINING_LOG_PATH


MegatronJob: TypeAlias = Annotated[
    MegatronTrainingJob
    | MegatronMergedTrainingJob
    | MegatronSyncJob
    | MegatronSFTTrainingJob,
    Field(discriminator="kind"),
]

_MEGATRON_JOB_ADAPTER = TypeAdapter(MegatronJob)


def dump_megatron_job(job: MegatronJob) -> str:
    return _MEGATRON_JOB_ADAPTER.dump_json(job).decode()


def load_megatron_job(raw: str | bytes) -> MegatronJob:
    return _MEGATRON_JOB_ADAPTER.validate_json(raw)
